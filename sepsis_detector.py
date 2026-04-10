import json
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from vitals_types import (
    VitalsSample, BaselineData, VITALS, BASELINE_WINDOWS, MAX_HISTORY,
    STATUS_CRITICAL_THRESH, STATUS_HIGH_RISK_THRESH, RF_HIGH_RISK_THRESH,
    MILD_STRESS_MILD_PROB, MILD_STRESS_ANOMALY,
    W_RF, W_ANOMALY, W_QSOFA, W_TRAJ, W_CORR
)
from baseline_establishment import BaselineEstablishment
from derivatives import DerivativeTracker
from anomaly_scoring import AnomalyScorer
from feature_engine import (
    hrv_collapse_severity, immobility_score, lactate_proxy,
    multi_system_correlation, temp_trajectory,
    feature_engine_sepsis_accel, phase_detection
)
from correlation_analyzer import SepsisCorrelationAnalyzer

logger = logging.getLogger(__name__)

# PLATEAU_WINDOW: Consecutive MILD_STRESS windows before plateau suppression activates.
PLATEAU_WINDOW: int = 4

class SepsisDetector:
    """
    Unified sepsis detection engine.
    Orchestrates baseline establishment, feature engineering, and scoring.
    """

    def __init__(
        self,
        population_if: IsolationForest,
        rf_model: RandomForestClassifier,
    ) -> None:
        self._pop_if = population_if
        self._rf = rf_model

        self._baseline_est = BaselineEstablishment()
        self._baseline: Optional[BaselineData] = None
        self._scorer: Optional[AnomalyScorer] = None
        self._deriv_tracker = DerivativeTracker()
        self._corr_analyzer = SepsisCorrelationAnalyzer()

        # Monitoring state
        self._score_history: List[Dict] = []
        self._window_count: int = BASELINE_WINDOWS
        self._consecutive_normal: int = 0

        # Drift correction buffers
        self._locked_means: Dict[str, float] = {}
        self._drift_means: Dict[str, float] = {}

        # Plateau suppression buffers
        self._mild_stress_streak: int = 0
        self._mild_stress_score_buffer: List[float] = []

    @property
    def baseline_locked(self) -> bool:
        return self._baseline is not None

    def add_baseline_window(self, sample: VitalsSample) -> Optional[BaselineData]:
        result = self._baseline_est.add_window(sample)
        if result:
            self._baseline = result
            self._scorer = AnomalyScorer(result, self._baseline_est.personal_if, self._pop_if)
            self._locked_means = dict(result.baseline_means)
            self._drift_means = dict(result.baseline_means)
        return result

    def process_monitoring_window(self, sample: VitalsSample) -> Dict:
        if not self.baseline_locked:
            raise RuntimeError("Baseline not locked. Submit 5 windows first.")

        self._window_count += 1
        baseline = self._baseline

        # Lazy-init guard for tests bypassing add_baseline_window
        if not self._locked_means:
            self._locked_means = dict(baseline.baseline_means)
            self._drift_means = dict(baseline.baseline_means)

        # Artifact detection (movement spike)
        art_contaminated = sample.movement > self._locked_means["movement"] * 2.5

        # Baseline drift correction (continuous slow update if not critical)
        # We use the previous window's status to avoid circular dependency
        prev_status = self._score_history[-1]["status"] if self._score_history else "NORMAL"
        if prev_status != "CRITICAL":
            for v in VITALS:
                self._drift_means[v] = 0.995 * self._drift_means[v] + 0.005 * float(getattr(sample, v))
            if self._window_count % 30 == 0:
                logger.info("Adaptive baseline drift applied at window %d", self._window_count)

        # Z-scores (using adaptive drift_means)
        z_scores = {
            v: round((float(getattr(sample, v)) - self._drift_means[v]) / baseline.baseline_stds[v], 3)
            for v in VITALS
        }

        # Special flags
        temp_bidirectional_flag = abs(z_scores.get("temp", 0.0)) > 2.0

        # Derivatives & Trajectory
        d1, d2, deriv_available = self._deriv_tracker.update(sample)
        traj_boost, accel_cnt = feature_engine_sepsis_accel(d2) if deriv_available and not art_contaminated else (0.0, 0)
        
        # Anomaly scoring
        anomaly_score, anomaly_method = self._scorer.score(sample, z_scores)

        # Feature Engine
        score_hist = self._score_history
        hrv_hist = [h["vitals_current"]["hrv"] for h in score_hist] + [sample.hrv]
        mov_hist = [h["vitals_current"]["movement"] for h in score_hist] + [sample.movement]
        tmp_hist = [h["vitals_current"]["temp"] for h in score_hist] + [sample.temp]

        hrv_sev = hrv_collapse_severity(hrv_hist)
        immo = immobility_score(mov_hist)
        t_traj = temp_trajectory(tmp_hist)
        lact = lactate_proxy(sample.spo2, sample.hr, sample.rr, sample.hrv, sample.movement)
        msc = multi_system_correlation(score_hist)
        msc_val = msc if msc is not None else 0.0

        # ML Model
        rf_input = [[sample.hr, sample.rr, sample.spo2, sample.temp, sample.movement, sample.hrv, sample.rrv, immo, t_traj, lact, msc_val]]
        rf_probs = self._rf.predict_proba(rf_input)[0]
        rf_prob_severe = float(rf_probs[2]) if len(rf_probs) > 2 else 0.0
        
        # qSOFA
        qsofa = int(sample.rr >= 22) + int(sample.hr >= 100) + int(sample.spo2 < 92)

        # Sepsis Correlation Module
        corr_results = self._corr_analyzer.analyze(self._score_history)
        corr_score = corr_results["sepsis_correlation_score"] if corr_results else 0.0

        # Final Score (Revised Weights)
        final_raw = (
            W_RF * rf_prob_severe +
            W_ANOMALY * (anomaly_score / 100.0) +
            W_QSOFA * (qsofa / 4.0) +
            W_TRAJ * traj_boost +
            W_CORR * corr_score
        )
        hrv_multiplier = (1 + 0.3 * hrv_sev) if len(score_hist) >= 10 else 1.0
        final_score = round(min(1.0, final_raw * hrv_multiplier), 4)

        # Status & Plateau Suppression
        raw_status = "NORMAL"
        if final_score > STATUS_CRITICAL_THRESH: raw_status = "CRITICAL"
        elif final_score > STATUS_HIGH_RISK_THRESH or rf_prob_severe > RF_HIGH_RISK_THRESH: raw_status = "HIGH_RISK"
        elif rf_prob_severe > MILD_STRESS_MILD_PROB or anomaly_score > MILD_STRESS_ANOMALY: raw_status = "MILD_STRESS"

        if raw_status == "MILD_STRESS":
            self._mild_stress_streak += 1
            self._mild_stress_score_buffer.append(final_score)
            if len(self._mild_stress_score_buffer) > PLATEAU_WINDOW: self._mild_stress_score_buffer.pop(0)
        else:
            self._mild_stress_streak = 0
            self._mild_stress_score_buffer.clear()

        plateau_active = (self._mild_stress_streak >= PLATEAU_WINDOW and (max(self._mild_stress_score_buffer) - min(self._mild_stress_score_buffer)) < 0.05)
        status = raw_status

        # Update normal counters
        if status == "NORMAL" and all(abs(z) < 1.5 for z in z_scores.values()) and sample.movement < 25:
            self._consecutive_normal += 1
        else:
            self._consecutive_normal = 0

        # Verdict
        sepsis_phase = phase_detection(final_score, hr=sample.hr, spo2=sample.spo2, hrv_collapse=hrv_sev)

        # ---- Build output dict ----------------------------------------------
        output = {
            "phase": "MONITORING",
            "window_number": self._window_count,
            "timestamp": sample.timestamp.isoformat(),
            # Baseline context
            "baseline_state": baseline.mode,
            "baseline_confidence": baseline.confidence,
            "baseline_confidence_breakdown": baseline.confidence_breakdown,
            "drift_from_locked": {
                v: round(self._drift_means[v] - self._locked_means[v], 4)
                for v in VITALS
            },
            # Data quality
            "artifact_contaminated": art_contaminated,
            "derivatives_available": deriv_available,
            "temp_bidirectional_flag": temp_bidirectional_flag,
            # Vitals
            "vitals_current": sample.to_dict(),
            # Anomaly signals
            "z_scores": z_scores,
            "first_derivatives": d1,
            "second_derivatives": d2,
            "anomaly_score": anomaly_score,
            "anomaly_method": anomaly_method,
            # Model outputs
            "rf_prob_normal": round(float(rf_probs[0]), 4),
            "rf_prob_mild": round(float(rf_probs[1]), 4) if len(rf_probs) > 1 else 0.0,
            "rf_prob_severe": round(rf_prob_severe, 4),
            "qsofa_score": qsofa,
            # Sepsis-specific features
            "trajectory_boost": round(traj_boost, 3),
            "hrv_collapse_severity": round(hrv_sev, 3),
            "lactate_proxy": round(lact, 3),
            "immobility_score": round(immo, 3),
            "temp_trajectory_slope": round(t_traj, 6),
            "multi_system_correlation": round(msc, 3) if msc is not None else None,
            "sepsis_acceleration_count": accel_cnt,
            # Sepsis Correlation Analysis (PART 4 Schema)
            "correlation_module_active": corr_results["correlation_module_active"] if corr_results else False,
            "windows_used_for_correlation": corr_results["windows_used_for_correlation"] if corr_results else 0,
            "sepsis_correlation_score": corr_score,
            "correlation_fingerprint": corr_results["correlation_fingerprint"] if corr_results else {},
            "abnormal_correlation_pairs": corr_results["abnormal_correlation_pairs"] if corr_results else [],
            "abnormal_pair_count": corr_results["abnormal_pair_count"] if corr_results else 0,
            "disease_probabilities": corr_results["disease_probabilities"] if corr_results else {},
            "dominant_disease_category": corr_results["dominant_disease_category"] if corr_results else "UNKNOWN",
            "correlation_confidence": corr_results["correlation_confidence"] if corr_results else "LOW",
            # Final verdict
            "final_score": final_score,
            "status": status,
            "plateau_suppression_active": plateau_active,
            "sepsis_phase": sepsis_phase,
            # Internal state
            "consecutive_normal_count": self._consecutive_normal,
            "mild_stress_streak": self._mild_stress_streak,
            "score_history_length": len(self._score_history) + 1,
        }

        self._score_history.append(output)
        if len(self._score_history) > MAX_HISTORY: self._score_history.pop(0)
        return output
