import numpy as np
from typing import Dict, Tuple
from sklearn.ensemble import IsolationForest
from vitals_types import VitalsSample, BaselineData

# Minimum monitoring windows before personal IF contributes to anomaly score.
IF_MIN_WINDOWS: int = 20

class AnomalyScorer:
    """
    Implements the 3-mode hybrid confidence fallback strategy.
    
    Personal IF contribution is gated until window count > 20 for statistical reliability.
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
        self._monitoring_windows: int = 0

    def score(
        self, sample: VitalsSample, z_scores: Dict[str, float]
    ) -> Tuple[float, str]:
        self._monitoring_windows += 1
        vec = [sample.to_feature_vector()]

        # Personal Z-score anomaly (always available)
        z_vals = list(z_scores.values())
        z_anomaly = float(np.clip(np.mean(np.abs(z_vals)) * 20, 0, 100))

        # Personal IF score — only used after 20 samples
        if if_ready := (self._monitoring_windows >= IF_MIN_WINDOWS):
            personal_raw = self._personal_if.decision_function(vec)[0]
            personal_score = float(np.clip(-personal_raw * 50 + 50, 0, 100))
        else:
            personal_score = z_anomaly

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
