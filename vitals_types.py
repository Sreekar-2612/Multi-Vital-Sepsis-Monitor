import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

VITALS = ["hr", "rr", "spo2", "temp", "movement", "hrv", "rrv"]

# Minimum clinical bounds for artifact detection (z-score filtering handles the rest)
VITAL_LIMITS = {
    "hr": (20, 250),
    "rr": (4, 60),
    "spo2": (50, 100),
    "temp": (30.0, 45.0),
}

# Window configuration
WINDOW_SECONDS: int = 40
BASELINE_WINDOWS: int = 5
MAX_HISTORY: int = 360  # 4 hours at 40-second windows

# Revised Score weights (sum = 1.0)
W_RF: float = 0.40
W_ANOMALY: float = 0.25
W_QSOFA: float = 0.15
W_TRAJ: float = 0.10
W_CORR: float = 0.10

# Thresholds
CONFIDENCE_HIGH = 75.0
CONFIDENCE_MID = 60.0
STATUS_CRITICAL_THRESH = 0.65
STATUS_HIGH_RISK_THRESH = 0.40
RF_HIGH_RISK_THRESH = 0.40
MILD_STRESS_MILD_PROB = 0.50
MILD_STRESS_ANOMALY = 50.0

@dataclass(frozen=True)
class VitalsSample:
    """A single 40-second window of physiological data."""
    timestamp: datetime.datetime
    hr: float
    rr: float
    spo2: float
    temp: float
    movement: float
    hrv: float
    rrv: float
    label: int = 0  # 0:Normal, 1:Infection, 2:Sepsis

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "hr": self.hr, "rr": self.rr, "spo2": self.spo2, "temp": self.temp,
            "movement": self.movement, "hrv": self.hrv, "rrv": self.rrv,
            "label": self.label
        }

    def to_feature_vector(self) -> List[float]:
        return [self.hr, self.rr, self.spo2, self.temp, self.movement, self.hrv, self.rrv]

@dataclass
class BaselineData:
    """The locked physiological profile of a specific patient."""
    mode: str  # LOCKED, HYBRID, FALLBACK
    confidence: float
    confidence_breakdown: Dict[str, float]
    baseline_means: Dict[str, float]
    baseline_stds: Dict[str, float]
    locked_at: datetime.datetime
