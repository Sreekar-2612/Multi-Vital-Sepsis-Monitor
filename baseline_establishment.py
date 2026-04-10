import datetime
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import IsolationForest
from vitals_types import VitalsSample, BaselineData, VITALS, BASELINE_WINDOWS, CONFIDENCE_HIGH, CONFIDENCE_MID

logger = logging.getLogger(__name__)

class BaselineEstablishment:
    """
    Handles Phase A: 5-window baseline collection and confidence scoring.
    
    Confidence (0-100) is a weighted sum of:
    - Stability (40%): low CV across 5 windows
    - Consistency (35%): readings within clinical bounds
    - Activity (15%): low movement at rest
    - Variability (10%): healthy HRV/RRV levels
    """

    def __init__(self) -> None:
        self.windows: List[VitalsSample] = []
        self.status = "COLLECTING"  # COLLECTING, LOCKED
        self.confidence_score: float = 0.0
        self.breakdown: Dict[str, float] = {}
        self.personal_if: Optional[IsolationForest] = None

    def add_window(self, sample: VitalsSample) -> Optional[BaselineData]:
        """Submit one 40-second window. Returns BaselineData once 5 windows collected."""
        if self.status == "LOCKED":
            return None
        
        self.windows.append(sample)
        logger.info("Baseline collection: window %d/5 received.", len(self.windows))

        if len(self.windows) >= BASELINE_WINDOWS:
            return self._finalize()
        
        return None

    def _finalize(self) -> BaselineData:
        # Convert to DataFrame for easier component scoring
        df = pd.DataFrame([s.to_dict() for s in self.windows])
        
        # Component 1: Stability Score (40%)
        stability = self._stability_score(df)
        
        # Component 2: Consistency Score (35%)
        consistency = self._consistency_score(df)
        
        # Component 3: Activity Quality Score (15%)
        activity = self._activity_quality_score(df)
        
        # Component 4: Variability Quality Score (10%)
        variability = self._variability_quality_score(df)
        
        self.confidence_score = (
            0.40 * stability +
            0.35 * consistency +
            0.15 * activity +
            0.10 * variability
        )
        
        self.breakdown = {
            "Stability": round(stability, 1),
            "Consistency": round(consistency, 1),
            "Activity": round(activity, 1),
            "Variability": round(variability, 1)
        }

        # Train Personal IsolationForest on these 5 windows
        # Note: IF on 5 samples is weak; monitoring engine will gate its use.
        features = [s.to_feature_vector() for s in self.windows]
        self.personal_if = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        self.personal_if.fit(features)

        # Decide mode
        if self.confidence_score >= CONFIDENCE_HIGH:
            mode = "LOCKED"
        elif self.confidence_score >= CONFIDENCE_MID:
            mode = "HYBRID"
        else:
            mode = "FALLBACK"

        self.status = "LOCKED"
        
        logger.info("Baseline established. Confidence: %.1f%% (%s)", self.confidence_score, mode)
        for k, v in self.breakdown.items():
            logger.info("  - %s: %.1f%%", k, v)

        return BaselineData(
            mode=mode,
            confidence=round(self.confidence_score, 1),
            confidence_breakdown=self.breakdown,
            baseline_means={v: float(df[v].mean()) for v in VITALS},
            baseline_stds={v: float(df[v].std()) if df[v].std() > 0 else 0.1 for v in VITALS},
            locked_at=datetime.datetime.now()
        )

    def _stability_score(self, df: pd.DataFrame) -> float:
        """Low Coeff of Variation = High Stability."""
        cv_caps = {
            'hr': 5.0, 'rr': 8.0, 'spo2': 1.0, 'temp': 0.5,
            'movement': 15.0, 'hrv': 12.0, 'rrv': 15.0
        }
        scores = []
        for v, cap in cv_caps.items():
            mean_val = df[v].mean()
            if mean_val == 0:
                scores.append(0.3)
                continue
            cv = (df[v].std() / mean_val) * 100
            if cv < cap: scores.append(1.0)
            elif cv < cap * 1.5: scores.append(0.7)
            else: scores.append(0.3)
        return float(np.mean(scores) * 100)

    def _consistency_score(self, df: pd.DataFrame) -> float:
        """Clinical plausibility check."""
        ranges = {
            'hr': (40, 120), 'rr': (8, 30), 'spo2': (92, 100),
            'temp': (35.5, 39.5), 'movement': (0, 50),
            'hrv': (20, 200), 'rrv': (5, 30)
        }
        total_cells = len(df) * len(ranges)
        in_range = 0
        for v, (low, high) in ranges.items():
            in_range += ((df[v] >= low) & (df[v] <= high)).sum()
        return (in_range / total_cells) * 100

    def _activity_quality_score(self, df: pd.DataFrame) -> float:
        """Penalise high movement during baseline."""
        avg_mov = df['movement'].mean()
        if avg_mov < 15: return 100.0
        elif avg_mov < 30: return 70.0
        return 30.0

    def _variability_quality_score(self, df: pd.DataFrame) -> float:
        """Healthy HRV/RRV levels indicate nervous system stability."""
        hrv_h = 25 <= df['hrv'].mean() <= 150
        rrv_h = 8 <= df['rrv'].mean() <= 25
        if hrv_h and rrv_h: return 100.0
        if hrv_h or rrv_h: return 50.0
        return 0.0
