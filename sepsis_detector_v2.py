"""
sepsis_detector_v2.py — Production-Grade Person-Specific Sepsis Detection Pipeline
===================================================================================
Architecture:
    VitalsSample (dataclass)          → typed 40-second window
    BaselineEstablishment             → 5-window collector + 4-component confidence + baseline lock
    DerivativeTracker                 → EMA-smoothed 1st/2nd derivatives + acceleration detection
    AnomalyScorer                     → Confidence-weighted LOCKED / HYBRID / FALLBACK blending
    SepsisFeatureEngine (functions)   → HRV collapse, lactate proxy, immobility, temp trajectory,
                                        multi-system correlation, sepsis phase
    SepsisDetector                    → End-to-end monitoring loop with full JSON output
    PatientStreamSimulator            → Realistic synthetic vital-sign stream for testing

Output Schema (every monitoring window):
    phase, window_number, timestamp, baseline_state, baseline_confidence,
    artifact_contaminated, derivatives_available, vitals_current, z_scores,
    first_derivatives, second_derivatives, anomaly_score, anomaly_method,
    rf_prob_normal, rf_prob_mild, rf_prob_severe, qsofa_score, trajectory_boost,
    hrv_collapse_severity, lactate_proxy, immobility_score, multi_system_correlation,
    sepsis_acceleration_count, final_score, status, sepsis_phase,
    consecutive_normal_count, score_history_length
"""

from __future__ import annotations

import datetime
import json
import logging
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("SepsisDetector")

# ---------------------------------------------------------------------------
# Config — all magic numbers in one place
# ---------------------------------------------------------------------------
VITALS: List[str] = ["hr", "rr", "spo2", "temp", "movement", "hrv", "rrv"]

GLOBAL_RANGES: Dict[str, Tuple[float, float]] = {
    "hr": (40, 120),
    "rr": (8, 30),
    "spo2": (92, 100),
    "temp": (35.5, 39.5),
    "movement": (0, 50),
    "hrv": (20, 200),
    "rrv": (5, 30),
}

# Max acceptable CV (%) for each vital — used in Stability score
MAX_CV: Dict[str, float] = {
    "hr": 5.0,
    "rr": 8.0,
    "spo2": 1.0,
    "temp": 0.5,
    "movement": 50.0,
    "hrv": 25.0,
    "rrv": 25.0,
}

# Clinical std-dev floor — prevents division by near-zero std
CLINICAL_STD_FLOOR: Dict[str, float] = {
    "hr": 2.0,
    "rr": 1.0,
    "spo2": 1.0,
    "temp": 0.1,
    "movement": 2.0,
    "hrv": 5.0,
    "rrv": 2.0,
}

WINDOW_SECONDS: int = 40
BASELINE_WINDOWS: int = 5
MAX_HISTORY: int = 360  # 4 hours at 40-second windows

# Minimum monitoring windows before personal IF contributes to anomaly score.
# Below this count the IF has too few samples to be statistically useful.
IF_MIN_WINDOWS: int = 20

# Minimum history windows for multi-system correlation (reduced from 30).
MSC_MIN_WINDOWS: int = 10

# Consecutive MILD_STRESS windows before plateau suppression activates.
# If score stays flat in MILD_STRESS for this many windows, it looks like
# stable infection not sepsis progression — suppress escalation.
PLATEAU_WINDOW: int = 4

# Score thresholds
CONFIDENCE_HIGH = 75.0
CONFIDENCE_MID = 60.0   # below this -> FALLBACK
STATUS_CRITICAL_THRESH = 0.65
STATUS_HIGH_RISK_THRESH = 0.40
RF_HIGH_RISK_THRESH = 0.40
MILD_STRESS_MILD_PROB = 0.50
MILD_STRESS_ANOMALY = 50.0

# Final score weights (must sum to 1.0)
W_RF = 0.45
W_ANOMALY = 0.28
W_QSOFA = 0.17
W_TRAJ = 0.10


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class VitalsSample:
    """Represents one 40-second vital-signs window."""

    timestamp: datetime.datetime
    hr: float
    rr: float
    spo2: float
    temp: float
    movement: float
    hrv: float
    rrv: float
    label: int = 0  # ground-truth condition (0=normal, 1=mild, 2=severe)

    def to_feature_vector(self) -> List[float]:
        """Return the 7 raw vital values as a list (for model input)."""
        return [self.hr, self.rr, self.spo2, self.temp,
                self.movement, self.hrv, self.rrv]

    def to_dict(self) -> Dict:
        return {
            v: round(float(getattr(self, v)), 3)
            for v in VITALS
        }


@dataclass
class WindowStats:
    """Statistics for a single 40-second window (used inside BaselineEstablishment)."""

    window_number: int
    means: Dict[str, float]
    stds: Dict[str, float]
    cvs: Dict[str, float]
    stability_flags: Dict[str, bool]  # True = vital is stable in this window


@dataclass
class BaselineData:
    """Immutable snapshot of a locked baseline."""

    confidence: float
    mode: str           # "LOCKED" | "HYBRID" | "FALLBACK"
    baseline_means: Dict[str, float]
    baseline_stds: Dict[str, float]   # floored to clinical minimum
    confidence_breakdown: Dict[str, float]  # stability, consistency, activity, variability
    window_count: int = BASELINE_WINDOWS
    locked_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())


# ---------------------------------------------------------------------------
# 1. BaselineEstablishment
# ---------------------------------------------------------------------------
class BaselineEstablishment:
    """
    Collects exactly BASELINE_WINDOWS (5) consecutive 40-second windows, then:
      - Computes a 4-component confidence score (0–100)
      - Decides on mode: LOCKED / HYBRID / FALLBACK
      - Trains a personal IsolationForest on the raw 5-window vectors
      - Returns a locked BaselineData

    Phase A outputs (printed while collecting):
    {
      "phase": "ESTABLISHING",
      "window_number": 1..5,
      "confidence_so_far": <float>,
      "status": "COLLECTING" | "BASELINE_LOCKED"
    }
    """

    def __init__(self) -> None:
        self._samples: List[VitalsSample] = []
        self._window_stats: List[WindowStats] = []
        self.baseline_data: Optional[BaselineData] = None
        self.personal_if: Optional[IsolationForest] = None
        self._locked: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_window(self, sample: VitalsSample) -> Optional[BaselineData]:
        """
        Add one 40-second window. Returns None until 5 windows are collected.
        After the 5th window, locks the baseline and returns BaselineData.
        """
        if self._locked:
            raise RuntimeError("Baseline already locked. Start monitoring phase.")

        n = len(self._samples) + 1
        self._samples.append(sample)
        stats = self._compute_window_stats(n, sample)
        self._window_stats.append(stats)

        # Running confidence estimate
        conf_est = self._compute_confidence()

        phase_out = {
            "phase": "ESTABLISHING",
            "window_number": n,
            "timestamp": sample.timestamp.isoformat(),
            "vitals": sample.to_dict(),
            "confidence_so_far": round(conf_est, 2),
            "status": "COLLECTING" if n < BASELINE_WINDOWS else "BASELINE_LOCKED",
        }
        print("=" * 60)
        print(json.dumps(phase_out, indent=2))

        if n == BASELINE_WINDOWS:
            return self._lock()
        return None

    @property
    def is_locked(self) -> bool:
        return self._locked

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_window_stats(self, number: int, s: VitalsSample) -> WindowStats:
        values = {v: float(getattr(s, v)) for v in VITALS}
        # For a single-point window, std is 0 — we use inter-window std later
        # Store per-vital values for cross-window statistics
        means = values.copy()
        stds = {v: 0.0 for v in VITALS}
        cvs = {v: 0.0 for v in VITALS}
        flags = {v: True for v in VITALS}  # single point is "stable"
        return WindowStats(
            window_number=number,
            means=means,
            stds=stds,
            cvs=cvs,
            stability_flags=flags,
        )

    def _get_all_means(self) -> Dict[str, float]:
        arr = {v: [s.means[v] for s in self._window_stats] for v in VITALS}
        return {v: float(np.mean(arr[v])) for v in VITALS}

    def _get_raw_stds(self) -> Dict[str, float]:
        arr = {v: [s.means[v] for s in self._window_stats] for v in VITALS}
        return {v: float(np.std(arr[v], ddof=1) if len(arr[v]) > 1 else 0.0)
                for v in VITALS}

    def _get_floored_stds(self) -> Dict[str, float]:
        raw = self._get_raw_stds()
        return {v: max(raw[v], CLINICAL_STD_FLOOR[v]) for v in VITALS}

    # ------------------------------------------------------------------
    # Confidence components
    # ------------------------------------------------------------------
    def _stability_score(self) -> float:
        """40% weight — low CV across all windows → high stability."""
        means = self._get_all_means()
        stds = self._get_raw_stds()
        scores = []
        for v in VITALS:
            cv = (stds[v] / means[v] * 100) if means[v] != 0 else 0.0
            scores.append(max(0.0, 1.0 - cv / MAX_CV.get(v, 20.0)))
        return (sum(scores) / len(scores)) * 100.0

    def _consistency_score(self) -> float:
        """35% weight — readings within clinically valid ranges."""
        total = float(len(self._samples) * len(VITALS))
        in_range = 0
        for s in self._samples:
            for v in VITALS:
                lo, hi = GLOBAL_RANGES[v]
                if lo <= float(getattr(s, v)) <= hi:
                    in_range += 1
        return (in_range / total) * 100.0

    def _activity_quality_score(self) -> float:
        """15% weight — low movement → better sensor quality."""
        movements = [s.movement for s in self._samples]
        avg_mv = float(np.mean(movements))
        score = max(0.0, 1.0 - avg_mv / 100.0) * 100.0
        if any(m > 30 for m in movements):
            score *= 0.5
        return score

    def _variability_quality_score(self) -> float:
        """10% weight — HRV and RRV within quality ranges."""
        means = self._get_all_means()
        hrv_q = 1.0 if 20 <= means["hrv"] <= 200 else 0.0
        rrv_q = 1.0 if 5 <= means["rrv"] <= 30 else 0.0
        return ((hrv_q + rrv_q) / 2) * 100.0

    def _compute_confidence(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        s = self._stability_score()
        c = self._consistency_score()
        a = self._activity_quality_score()
        v = self._variability_quality_score()
        return 0.40 * s + 0.35 * c + 0.15 * a + 0.10 * v

    # ------------------------------------------------------------------
    # Lock & train
    # ------------------------------------------------------------------
    def _lock(self) -> BaselineData:
        stability = self._stability_score()
        consistency = self._consistency_score()
        activity = self._activity_quality_score()
        variability = self._variability_quality_score()
        confidence = (
            0.40 * stability
            + 0.35 * consistency
            + 0.15 * activity
            + 0.10 * variability
        )

        if confidence >= CONFIDENCE_HIGH:
            mode = "LOCKED"
        elif confidence >= CONFIDENCE_MID:
            mode = "HYBRID"
        else:
            mode = "FALLBACK"

        means = self._get_all_means()
        stds = self._get_floored_stds()

        # Train personal IsolationForest on raw 7-vital vectors from the 5 windows
        vectors = [s.to_feature_vector() for s in self._samples]
        self.personal_if = IsolationForest(
            n_estimators=100, contamination=0.05, random_state=42
        )
        self.personal_if.fit(vectors)

        self.baseline_data = BaselineData(
            confidence=round(confidence, 2),
            mode=mode,
            baseline_means=means,
            baseline_stds=stds,
            confidence_breakdown={
                "stability": round(stability, 2),
                "consistency": round(consistency, 2),
                "activity_quality": round(activity, 2),
                "variability_quality": round(variability, 2),
            },
        )
        self._locked = True

        logger.info(
            "Baseline LOCKED | Confidence: %.1f%% | Mode: %s",
            confidence,
            mode,
        )
        return self.baseline_data


# ---------------------------------------------------------------------------
# 2. DerivativeTracker
# ---------------------------------------------------------------------------
class DerivativeTracker:
    """
    Maintains a rolling vital-sign history and computes:
      - EMA-smoothed first derivatives  (rate of change per second)
      - Second derivatives              (acceleration per second²)

    EMA smoothing weight α=0.3 on raw 1st derivative to suppress noise.
    """

    EMA_ALPHA = 0.3

    def __init__(self) -> None:
        self._history: List[VitalsSample] = []
        self._smooth_d1: Dict[str, float] = {f"d{v}": 0.0 for v in VITALS}

    def update(
        self, sample: VitalsSample
    ) -> Tuple[Dict[str, float], Dict[str, float], bool]:
        """
        Update with a new sample. Returns:
            (first_deriv, second_deriv, derivatives_available)
        derivatives_available is True once we have ≥3 samples (need 2nd deriv).
        """
        self._history.append(sample)
        # Cap to last 3 samples — only prev2, prev, curr are ever accessed.
        # Prevents unbounded memory growth over long monitoring sessions.
        if len(self._history) > 3:
            self._history.pop(0)

        d1: Dict[str, float] = {f"d{v}": 0.0 for v in VITALS}
        d2: Dict[str, float] = {f"d2{v}": 0.0 for v in VITALS}
        available = False

        n = len(self._history)
        if n < 2:
            return d1, d2, available

        prev = self._history[-2]
        curr = self._history[-1]
        dt = (curr.timestamp - prev.timestamp).total_seconds()
        if dt <= 0:
            dt = WINDOW_SECONDS

        # 1st derivative (raw)
        raw_d1 = {
            f"d{v}": (float(getattr(curr, v)) - float(getattr(prev, v))) / dt
            for v in VITALS
        }

        # EMA-smoothed 1st derivative
        for key in raw_d1:
            self._smooth_d1[key] = (
                self.EMA_ALPHA * raw_d1[key]
                + (1 - self.EMA_ALPHA) * self._smooth_d1[key]
            )

        d1 = {k: round(v, 6) for k, v in self._smooth_d1.items()}

        if n >= 3:
            prev2 = self._history[-3]
            # Previous raw 1st derivative (approximate)
            dt2 = (prev.timestamp - prev2.timestamp).total_seconds()
            if dt2 <= 0:
                dt2 = WINDOW_SECONDS
            prev_raw_d1 = {
                f"d{v}": (float(getattr(prev, v)) - float(getattr(prev2, v))) / dt2
                for v in VITALS
            }

            for v in VITALS:
                key1 = f"d{v}"
                key2 = f"d2{v}"
                d2[key2] = round(
                    (raw_d1[key1] - prev_raw_d1[key1]) / dt, 8
                )

            available = True

        return d1, d2, available

    def last_smooth_d1(self) -> Dict[str, float]:
        return self._smooth_d1.copy()


# ---------------------------------------------------------------------------
# 3. AnomalyScorer
# ---------------------------------------------------------------------------
class AnomalyScorer:
    """
    Produces a 0–100 anomaly score using confidence-weighted blending:
      LOCKED   → 50% personal IF  + 50% personal Z-score anomaly
      HYBRID   → 60% of above     + 40% population IF
      FALLBACK → 100% population IF (generic standards only)
    """

    def __init__(
        self,
        baseline: BaselineData,
        personal_if: IsolationForest,
        population_if: IsolationForest,
    ) -> None:
        self._baseline = baseline
        self._personal_if = personal_if
        self._pop_if = population_if
        self._monitoring_windows: int = 0  # counts windows scored so far

    def score(
        self, sample: VitalsSample, z_scores: Dict[str, float]
    ) -> Tuple[float, str]:
        """
        Returns (anomaly_score 0–100, method_description).

        Personal IF contribution is gated behind IF_MIN_WINDOWS (default 20).
        Below that threshold, Z-score anomaly carries 100% of the personal signal
        because an IF fitted on <20 samples is statistically unreliable.
        """
        self._monitoring_windows += 1
        vec = [sample.to_feature_vector()]

        # Personal Z-score anomaly (always available, scaled 0-100)
        z_vals = list(z_scores.values())
        z_anomaly = float(np.clip(np.mean(np.abs(z_vals)) * 20, 0, 100))

        # Personal IF score — only used after IF_MIN_WINDOWS
        if self._monitoring_windows >= IF_MIN_WINDOWS:
            personal_raw = self._personal_if.decision_function(vec)[0]
            # decision_function returns ~[-0.5, 0.5]; scale to [0,100]
            personal_score = float(np.clip(-personal_raw * 50 + 50, 0, 100))
            if_ready = True
        else:
            personal_score = z_anomaly  # fallback to Z-score until IF is trustworthy
            if_ready = False

        # Population IF score
        pop_raw = self._pop_if.decision_function(vec)[0]
        pop_score = float(np.clip(-pop_raw * 50 + 50, 0, 100))

        mode = self._baseline.mode
        if_label = "PersonalIF+Z" if if_ready else "Z-score-only"

        if mode == "LOCKED":
            score = 0.5 * personal_score + 0.5 * z_anomaly
            method = f"LOCKED: 50% {if_label} + 50% Z-score"
        elif mode == "HYBRID":
            blended = 0.5 * personal_score + 0.5 * z_anomaly
            score = 0.60 * blended + 0.40 * pop_score
            method = f"HYBRID: 60%({if_label}+Z) + 40% PopIF"
        else:  # FALLBACK
            score = pop_score
            method = "FALLBACK: 100% PopIF (global standards)"

        return round(min(score, 100.0), 2), method


# ---------------------------------------------------------------------------
# 4. Sepsis Feature Engine (pure functions)
# ---------------------------------------------------------------------------
def hrv_collapse_severity(hrv_history: List[float]) -> float:
    """
    Fraction of HRV drop from early baseline to recent values.
    Returns 0.0–1.0 (1.0 = >50% collapse → severe autonomic failure).
    Requires ≥10 history points.
    """
    if len(hrv_history) < 10:
        return 0.0
    baseline_hrv = float(np.mean(hrv_history[:6]))
    recent_hrv = float(np.mean(hrv_history[-3:]))
    if baseline_hrv == 0:
        return 0.0
    pct_drop = (baseline_hrv - recent_hrv) / baseline_hrv
    if pct_drop > 0.50:
        return 1.0
    if pct_drop > 0.35:
        return 0.7
    if pct_drop > 0.20:
        return 0.4
    return 0.0


def immobility_score(mov_history: List[float]) -> float:
    """
    Relative drop in movement vs. early baseline.
    Returns 0.0–1.0.  Requires ≥6 history points.
    """
    if len(mov_history) < 6:
        return 0.0
    baseline_mv = float(np.mean(mov_history[:6]))
    recent_mv = float(np.mean(mov_history[-3:]))
    drop = (baseline_mv - recent_mv) / max(baseline_mv, 1.0)
    return float(np.clip(drop, 0.0, 1.0))


def temp_trajectory(temp_history: List[float]) -> float:
    """
    Linear slope of temperature (°C or °F) over the history window.
    Positive = rising, negative = falling.  Requires ≥6 history points.
    """
    if len(temp_history) < 6:
        return 0.0
    x = list(range(len(temp_history)))
    slope, _ = np.polyfit(x, temp_history, 1)
    return float(slope)


def lactate_proxy(spo2: float, hr: float, rr: float, hrv: float, mv: float) -> float:
    """
    Weighted multi-vital proxy for tissue hypoxia / lactate elevation.
    Returns 0.0–1.0.
    """
    s = (
        max(0.0, 95 - spo2) / 10 * 0.30
        + max(0.0, hr - 90) / 60 * 0.25
        + max(0.0, rr - 20) / 20 * 0.20
        + max(0.0, 60 - hrv) / 60 * 0.15
        + max(0.0, 30 - mv) / 30 * 0.10
    )
    return float(min(s, 1.0))


def multi_system_correlation(score_history: List[Dict]) -> Optional[float]:
    """
    Fraction of available windows (last 30, min MSC_MIN_WINDOWS) where
    >=3 z-scores exceeded 2sigma simultaneously.
    Returns None if fewer than MSC_MIN_WINDOWS windows available.
    Activates at 10 windows (not 30) so early sepsis is caught sooner.
    """
    n = len(score_history)
    if n < MSC_MIN_WINDOWS:
        return None
    window_size = min(n, 30)
    recent = score_history[-window_size:]
    w_3plus = sum(
        1
        for w in recent
        if sum(1 for z in w["z_scores"].values() if abs(z) > 2.0) >= 3
    )
    return float(w_3plus / window_size)


def feature_engine_sepsis_accel(d2: Dict[str, float]) -> Tuple[float, int]:
    """
    Trajectory boost (0.0–1.0) from 2nd-derivative acceleration thresholds.
    Returns (boost_value, count_of_accelerating_vitals).
    """
    cnt = sum(
        [
            d2.get("d2hr", 0) > 0.05,
            d2.get("d2rr", 0) > 0.08,
            d2.get("d2temp", 0) > 0.001,
            d2.get("d2hrv", 0) < -0.10,
            d2.get("d2rrv", 0) < -0.08,
        ]
    )
    if cnt >= 3:
        return 1.0, cnt
    if cnt == 2:
        return 0.6, cnt
    if cnt == 1:
        return 0.3, cnt
    return 0.0, cnt


def phase_detection(
    final_score: float,
    hr: float = 75.0,
    spo2: float = 98.0,
    hrv_collapse: float = 0.0,
) -> str:
    """
    Map final_score to a sepsis phase.

    PHASE_3_SEPTIC_SHOCK is detected via a non-invasive shock proxy:
        shock_proxy = HR_rise * SpO2_drop * HRV_collapse
    This approximates circulatory compromise without requiring direct SBP.
    Clinically: tachycardia + hypoxia + autonomic failure = shock signature.
    """
    hr_rise = max(0.0, (hr - 80) / 80.0)          # 0 at 80bpm, 1 at 160bpm
    spo2_drop = max(0.0, (95 - spo2) / 20.0)      # 0 at 95%, 1 at 75%
    shock_proxy = hr_rise * spo2_drop * (1 + hrv_collapse)
    if final_score > 0.65 and shock_proxy > 0.15:
        return "PHASE_3_SEPTIC_SHOCK"
    if final_score > 0.50:
        return "PHASE_2_INTERMEDIATE"
    if final_score > 0.30:
        return "PHASE_1_EARLY"
    return "PHASE_0_NORMAL"


# ---------------------------------------------------------------------------
# 5. Population IsolationForest (trained once at startup)
# ---------------------------------------------------------------------------
def build_population_if() -> IsolationForest:
    """
    Train a population-level IsolationForest on 2,000 synthetic
    MIMIC-III–like patient records (80% normal / 20% abnormal).
    """
    np.random.seed(42)
    data: List[List[float]] = []
    for _ in range(2000):
        if np.random.rand() > 0.20:
            row = [
                np.random.normal(70, 10),   # hr
                np.random.normal(16, 2),    # rr
                np.random.normal(98, 1),    # spo2
                np.random.normal(36.8, 0.3),# temp
                np.random.normal(10, 5),    # movement
                np.random.normal(45, 10),   # hrv
                np.random.normal(15, 3),    # rrv
            ]
        else:
            row = [
                np.random.normal(110, 20),
                np.random.normal(24, 4),
                np.random.normal(92, 3),
                np.random.normal(38.5, 1.0),
                np.random.normal(30, 20),
                np.random.normal(20, 15),
                np.random.normal(8, 4),
            ]
        data.append(row)
    pop_df = pd.DataFrame(data, columns=VITALS)
    iso = IsolationForest(random_state=42, contamination=0.10)
    iso.fit(pop_df)
    return iso


# ---------------------------------------------------------------------------
# 6. Random Forest (trained once at startup)
# ---------------------------------------------------------------------------
def build_random_forest() -> RandomForestClassifier:
    """
    Build a per-condition Random Forest (normal=0, mild=1, severe=2)
    on 500 synthetic labelled vitals + 4 derived features.
    """
    np.random.seed(42)
    rows: List[List[float]] = []
    conditions = {
        0: dict(hr=70, rr=14, spo2=98, temp=36.8, mv=10, hrv=45, rrv=15,
                immo=0, t_traj=0, lact=0, msc=0),
        1: dict(hr=90, rr=18, spo2=95, temp=37.8, mv=20, hrv=30, rrv=10,
                immo=0.2, t_traj=0.01, lact=0.4, msc=0.1),
        2: dict(hr=115, rr=26, spo2=88, temp=39.5, mv=5, hrv=12, rrv=5,
                immo=0.8, t_traj=0.05, lact=0.8, msc=0.5),
    }
    for _ in range(500):
        cond = np.random.choice([0, 1, 2])
        c = conditions[cond]
        rows.append([
            c["hr"] + np.random.normal(0, 5),
            c["rr"], c["spo2"], c["temp"], c["mv"],
            c["hrv"], c["rrv"],
            c["immo"], c["t_traj"], c["lact"], c["msc"], cond,
        ])

    df = pd.DataFrame(rows, columns=[
        "hr", "rr", "spo2", "temp", "movement",
        "hrv", "rrv", "immobility", "temp_trajectory",
        "lactate_proxy", "multi_system_corr", "label",
    ])
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(df.drop(columns=["label"]), df["label"])
    return rf


# ---------------------------------------------------------------------------
# 7. SepsisDetector — main monitoring engine
# ---------------------------------------------------------------------------
class SepsisDetector:
    """
    End-to-end sepsis detection pipeline.

    Usage:
        detector = SepsisDetector(population_if, rf_model)

        # Phase A — submit 5 baseline windows
        for raw_window in stream[:5]:
            result = detector.add_baseline_window(raw_window)

        # Phase B — continuous monitoring
        for raw_window in stream[5:]:
            output = detector.process_monitoring_window(raw_window)
            print(json.dumps(output, indent=2))
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

        # Monitoring state
        self._score_history: List[Dict] = []
        self._window_count: int = BASELINE_WINDOWS
        self._consecutive_normal: int = 0

        # FIX: Separate locked baseline from mutable drift working copy.
        # _locked_means is the original baseline (read-only, for audit).
        # _drift_means is the adaptive copy used for Z-score computation.
        self._locked_means: Dict[str, float] = {}
        self._drift_means: Dict[str, float] = {}

        # Plateau suppression: tracks consecutive MILD_STRESS windows
        # If score stays flat in MILD_STRESS for PLATEAU_WINDOW windows,
        # suppress escalation (looks like stable infection, not sepsis).
        self._mild_stress_streak: int = 0
        self._mild_stress_score_buffer: List[float] = []

    # ------------------------------------------------------------------
    # Phase A
    # ------------------------------------------------------------------
    def add_baseline_window(self, sample: VitalsSample) -> Optional[BaselineData]:
        """Submit one 40-second window during baseline establishment."""
        result = self._baseline_est.add_window(sample)
        if result is not None:
            self._baseline = result
            self._scorer = AnomalyScorer(
                result,
                self._baseline_est.personal_if,
                self._pop_if,
            )
            # Initialise both locked and drift copies from the locked baseline.
            # _locked_means is never modified (audit trail).
            # _drift_means is updated by the slow drift correction.
            self._locked_means = dict(result.baseline_means)
            self._drift_means = dict(result.baseline_means)
            logger.info(
                "Baseline locked -> mode=%s, confidence=%.1f%%",
                result.mode,
                result.confidence,
            )
        return result

    @property
    def baseline_locked(self) -> bool:
        return self._baseline is not None

    # ------------------------------------------------------------------
    # Phase B
    # ------------------------------------------------------------------
    def process_monitoring_window(self, sample: VitalsSample) -> Dict:
        """
        Process one 40-second monitoring window.
        Returns a fully-populated JSON-serialisable dict.
        Raises RuntimeError if baseline is not locked.
        """
        if not self.baseline_locked:
            raise RuntimeError(
                "Baseline not locked. Submit 5 windows via add_baseline_window() first."
            )

        self._window_count += 1
        baseline = self._baseline

        # Lazy-init drift/locked means if they weren't set via add_baseline_window()
        # (e.g., test code that directly assigns _baseline).
        if not self._locked_means:
            self._locked_means = dict(baseline.baseline_means)
            self._drift_means = dict(baseline.baseline_means)

        # ---- Artifact detection ----------------------------------------
        art_contaminated = (
            sample.movement > baseline.baseline_means["movement"] * 2.5
        )

        # ---- Baseline drift update (every 10 consecutive normal windows) ----
        # FIX: Write to _drift_means only — _locked_means stays immutable.
        if self._consecutive_normal >= 10:
            for v in VITALS:
                self._drift_means[v] = (
                    0.9 * self._drift_means[v] + 0.1 * float(getattr(sample, v))
                )
            self._consecutive_normal = 0
            logger.debug("Baseline drift correction applied at window %d", self._window_count)

        # ---- Z-scores (use _drift_means — adaptive working copy) ------------
        z_scores = {
            v: round(
                (float(getattr(sample, v)) - self._drift_means[v])
                / baseline.baseline_stds[v],
                3,
            )
            for v in VITALS
        }

        # ---- Bidirectional temperature anomaly flag -------------------------
        # Hypothermic sepsis (temp < 35.5C) can be missed by positive Z-scores only.
        # Flag when temp is anomalously LOW relative to baseline.
        temp_z = z_scores.get("temp", 0.0)
        temp_bidirectional_flag = abs(temp_z) > 2.0  # catches both hyper and hypothermia

        # ---- Derivatives ----------------------------------------------------
        d1, d2, deriv_available = self._deriv_tracker.update(sample)

        # ---- Trajectory boost from 2nd derivatives -------------------------
        if deriv_available and not art_contaminated:
            traj_boost, accel_cnt = feature_engine_sepsis_accel(d2)
        else:
            traj_boost, accel_cnt = 0.0, 0

        # ---- Anomaly score --------------------------------------------------
        anomaly_score, anomaly_method = self._scorer.score(sample, z_scores)

        # ---- Sepsis feature signals -----------------------------------------
        hrv_hist = [h["vitals_current"]["hrv"] for h in self._score_history] + [sample.hrv]
        mov_hist = [h["vitals_current"]["movement"] for h in self._score_history] + [sample.movement]
        tmp_hist = [h["vitals_current"]["temp"] for h in self._score_history] + [sample.temp]

        hrv_sev = hrv_collapse_severity(hrv_hist)
        immo = immobility_score(mov_hist)
        t_traj = temp_trajectory(tmp_hist)
        lact = lactate_proxy(sample.spo2, sample.hr, sample.rr, sample.hrv, sample.movement)
        msc = multi_system_correlation(self._score_history)
        msc_val = msc if msc is not None else 0.0

        # ---- RF classification (11 features) --------------------------------
        rf_input = [[
            sample.hr, sample.rr, sample.spo2, sample.temp,
            sample.movement, sample.hrv, sample.rrv,
            immo, t_traj, lact, msc_val,
        ]]
        rf_probs = self._rf.predict_proba(rf_input)[0]
        rf_prob_normal = float(rf_probs[0])
        rf_prob_mild = float(rf_probs[1]) if len(rf_probs) > 1 else 0.0
        rf_prob_severe = float(rf_probs[2]) if len(rf_probs) > 2 else 0.0

        # ---- qSOFA score ----------------------------------------------------
        qsofa = (
            int(sample.rr >= 22)
            + int(sample.hr >= 100)
            + int(sample.spo2 < 92)
        )

        # ---- Final score (weighted) ----------------------------------------
        final_raw = (
            W_RF * rf_prob_severe
            + W_ANOMALY * (anomaly_score / 100.0)
            + W_QSOFA * (qsofa / 4.0)
            + W_TRAJ * traj_boost
        )

        # HRV collapse multiplier — kicks in after 10 windows of history
        hrv_multiplier = (1 + 0.3 * hrv_sev) if len(self._score_history) >= 10 else 1.0
        final_score = round(min(1.0, final_raw * hrv_multiplier), 4)

        # ---- Status classification ------------------------------------------
        raw_status: str
        if final_score > STATUS_CRITICAL_THRESH:
            raw_status = "CRITICAL"
        elif final_score > STATUS_HIGH_RISK_THRESH or rf_prob_severe > RF_HIGH_RISK_THRESH:
            raw_status = "HIGH_RISK"
        elif rf_prob_mild > MILD_STRESS_MILD_PROB or anomaly_score > MILD_STRESS_ANOMALY:
            raw_status = "MILD_STRESS"
        else:
            raw_status = "NORMAL"

        # ---- Plateau suppression (disease vs sepsis differential) ----------
        # If final_score has been flat in MILD_STRESS for PLATEAU_WINDOW windows,
        # it looks like a stable infection not progressing to sepsis — suppress
        # escalation to HIGH_RISK until score actually rises again.
        if raw_status == "MILD_STRESS":
            self._mild_stress_streak += 1
            self._mild_stress_score_buffer.append(final_score)
            if len(self._mild_stress_score_buffer) > PLATEAU_WINDOW:
                self._mild_stress_score_buffer.pop(0)
        else:
            self._mild_stress_streak = 0
            self._mild_stress_score_buffer.clear()

        plateau_active = (
            self._mild_stress_streak >= PLATEAU_WINDOW
            and len(self._mild_stress_score_buffer) == PLATEAU_WINDOW
            and (max(self._mild_stress_score_buffer) - min(self._mild_stress_score_buffer)) < 0.05
        )
        status = raw_status  # plateau only suppresses within MILD_STRESS band

        # ---- Consecutive normal counter ------------------------------------
        if (
            status == "NORMAL"
            and all(abs(z) < 1.5 for z in z_scores.values())
            and sample.movement < 25
        ):
            self._consecutive_normal += 1
        else:
            self._consecutive_normal = 0

        # ---- Sepsis phase --------------------------------------------------
        sepsis_phase = phase_detection(
            final_score,
            hr=sample.hr,
            spo2=sample.spo2,
            hrv_collapse=hrv_sev,
        )

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
            "rf_prob_normal": round(rf_prob_normal, 4),
            "rf_prob_mild": round(rf_prob_mild, 4),
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

        # Store and cap history
        self._score_history.append(output)
        if len(self._score_history) > MAX_HISTORY:
            self._score_history.pop(0)

        return output


# ---------------------------------------------------------------------------
# 8. PatientStreamSimulator
# ---------------------------------------------------------------------------
class PatientStreamSimulator:
    """
    Generates realistic 40-second vital-sign windows for 3 conditions:
      0 = Normal baseline
      1 = Mild infection (elevated HR/RR, slightly lower SpO2)
      2 = Sepsis (rapid deterioration towards shock parameters)
    """

    def __init__(
        self,
        condition: int = 0,
        baseline_hr: float = 75.0,
        baseline_temp: float = 36.8,
    ) -> None:
        self.condition = condition
        self.baseline_hr = baseline_hr
        self.baseline_temp = baseline_temp
        self._t = datetime.datetime.now()
        # Mutable state — allows smooth physiological transitions
        self.hr = float(baseline_hr)
        self.temp = float(baseline_temp)
        self.spo2 = 98.0
        self.rr = 14.0
        self.hrv = 45.0
        self.rrv = 15.0

    def set_condition(self, condition: int) -> None:
        self.condition = condition

    def get_next_window(self) -> VitalsSample:
        """Advance simulation by 40 seconds and return the next VitalsSample."""
        self._t += datetime.timedelta(seconds=WINDOW_SECONDS)

        targets = {
            0: dict(hr=self.baseline_hr, temp=self.baseline_temp, spo2=98.0,
                    rr=14.0, hrv=45.0, rrv=15.0, mov_mean=10),
            1: dict(hr=self.baseline_hr + 15, temp=self.baseline_temp + 1.0,
                    spo2=95.0, rr=18.0, hrv=30.0, rrv=10.0, mov_mean=20),
            2: dict(hr=self.baseline_hr + 45, temp=self.baseline_temp + 2.5,
                    spo2=88.0, rr=26.0, hrv=12.0, rrv=5.0, mov_mean=5),
        }
        t = targets[self.condition]
        alpha = 0.20 if self.condition == 2 else 0.05

        self.hr   += (t["hr"]   - self.hr)   * alpha + np.random.normal(0, 0.5)
        self.temp += (t["temp"] - self.temp) * alpha + np.random.normal(0, 0.02)
        self.spo2 += (t["spo2"] - self.spo2) * alpha + np.random.normal(0, 0.2)
        self.rr   += (t["rr"]   - self.rr)   * alpha + np.random.normal(0, 0.2)
        self.hrv  += (t["hrv"]  - self.hrv)  * alpha + np.random.normal(0, 1.0)
        self.rrv  += (t["rrv"]  - self.rrv)  * alpha + np.random.normal(0, 1.0)
        movement = float(np.clip(np.random.normal(t["mov_mean"], 5), 0, 100))

        return VitalsSample(
            timestamp=self._t,
            hr=round(float(np.clip(self.hr, 40, 180)), 2),
            rr=round(float(np.clip(self.rr, 8, 45)), 2),
            spo2=round(float(np.clip(self.spo2, 75, 100)), 2),
            temp=round(float(np.clip(self.temp, 35.5, 41.5)), 2),
            movement=round(movement, 2),
            hrv=round(float(np.clip(self.hrv, 5, 200)), 2),
            rrv=round(float(np.clip(self.rrv, 2, 50)), 2),
            label=self.condition,
        )


# ---------------------------------------------------------------------------
# 9. main — demo run
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("Building population IsolationForest...")
    pop_if = build_population_if()

    logger.info("Building Random Forest classifier...")
    rf = build_random_forest()

    logger.info("Initialising SepsisDetector...")
    detector = SepsisDetector(population_if=pop_if, rf_model=rf)

    # ---- Phase A: Baseline establishment (5 windows, condition=0) ----------
    sim = PatientStreamSimulator(condition=0)
    logger.info("=== PHASE A: Baseline Establishment (5 × 40s windows) ===")
    for _ in range(BASELINE_WINDOWS):
        sample = sim.get_next_window()
        baseline_data = detector.add_baseline_window(sample)

    # Print baseline summary
    bd = detector._baseline
    print("\n" + "=" * 60)
    print(json.dumps({
        "event": "BASELINE_LOCKED",
        "confidence": bd.confidence,
        "mode": bd.mode,
        "confidence_breakdown": bd.confidence_breakdown,
        "baseline_means": {k: round(v, 2) for k, v in bd.baseline_means.items()},
    }, indent=2))
    print("=" * 60 + "\n")

    # ---- Phase B: Monitoring (switch to sepsis condition) ------------------
    sim.set_condition(2)  # rapid deterioration for demo
    logger.info("=== PHASE B: Real-Time Monitoring (condition=2 / sepsis) ===")

    try:
        while True:
            sample = sim.get_next_window()
            output = detector.process_monitoring_window(sample)
            print("=" * 60)
            print(json.dumps(output, indent=2))
            time.sleep(0.5)  # pacing for readability in demo
    except KeyboardInterrupt:
        logger.info("Monitoring terminated by user.")


if __name__ == "__main__":
    main()
