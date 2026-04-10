import numpy as np
from typing import Dict, List, Optional, Tuple
from vitals_types import VitalsSample, VITALS

def hrv_collapse_severity(hrv_history: List[float]) -> float:
    """Detect sustained collapse in HRV (0.0-1.0)."""
    if len(hrv_history) < 5: return 0.0
    recent = hrv_history[-5:]
    baseline_hrv = np.mean(hrv_history[:-5]) if len(hrv_history) > 5 else 45.0
    collapse = (baseline_hrv - np.mean(recent)) / baseline_hrv
    return float(np.clip(collapse, 0, 1))

def immobility_score(movement_history: List[float]) -> float:
    """Higher score (0-1) if movement is sustainedly low (illness/fatigue)."""
    if len(movement_history) < 10: return 0.0
    recent_avg = np.mean(movement_history[-10:])
    return float(np.clip((25 - recent_avg) / 25.0, 0, 1))

def lactate_proxy(spo2: float, hr: float, rr: float, hrv: float, mov: float) -> float:
    """Non-invasive proxy for metabolic stress (0.0-1.0)."""
    s = (
        (100 - spo2) / 10.0 +
        (hr - 70) / 100.0 +
        (rr - 14) / 40.0 +
        (50 - hrv) / 100.0 +
        (10 - mov) / 50.0
    )
    return float(np.clip(s, 0, 1.0))

def multi_system_correlation(score_history: List[Dict], min_windows: int = 10) -> Optional[float]:
    """Fraction of windows where >=3 z-scores exceeded 2sigma."""
    n = len(score_history)
    if n < min_windows: return None
    window_size = min(n, 30)
    recent = score_history[-window_size:]
    w_3plus = sum(1 for w in recent if sum(1 for z in w["z_scores"].values() if abs(z) > 2.0) >= 3)
    return float(w_3plus / window_size)

def temp_trajectory(temp_history: List[float]) -> float:
    """Slope of temp over last 30 windows."""
    if len(temp_history) < 5: return 0.0
    x = np.arange(len(temp_history))
    slope, _ = np.polyfit(x, temp_history, 1)
    return float(slope)

def feature_engine_sepsis_accel(d2: Dict[str, float]) -> Tuple[float, int]:
    """Trajectory boost from 2nd-derivative acceleration."""
    cnt = sum([
        d2.get("d2hr", 0) > 0.05,
        d2.get("d2rr", 0) > 0.08,
        d2.get("d2temp", 0) > 0.001,
        d2.get("d2hrv", 0) < -0.10,
        d2.get("d2rrv", 0) < -0.08,
    ])
    if cnt >= 3: return 1.0, cnt
    if cnt == 2: return 0.6, cnt
    if cnt == 1: return 0.3, cnt
    return 0.0, cnt

def phase_detection(final_score: float, hr: float, spo2: float, hrv_collapse: float) -> str:
    """Shock Proxy detection: HR_rise * SpO2_drop * HRV_collapse."""
    hr_rise = max(0.0, (hr - 80) / 80.0)
    spo2_drop = max(0.0, (95 - spo2) / 20.0)
    shock_proxy = hr_rise * spo2_drop * (1 + hrv_collapse)
    
    if final_score > 0.65 and shock_proxy > 0.15: return "PHASE_3_SEPTIC_SHOCK"
    if final_score > 0.50: return "PHASE_2_INTERMEDIATE"
    if final_score > 0.30: return "PHASE_1_EARLY"
    return "PHASE_0_NORMAL"
