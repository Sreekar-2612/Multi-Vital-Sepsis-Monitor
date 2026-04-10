import datetime
import numpy as np
from vitals_types import VitalsSample, WINDOW_SECONDS

class PatientStreamSimulator:
    """Generates realistic physiological data for Normal, Infection, and Sepsis states."""
    def __init__(self, condition: int = 0, baseline_hr: float = 75.0, baseline_temp: float = 36.8) -> None:
        self.condition = condition
        self.baseline_hr = baseline_hr
        self.baseline_temp = baseline_temp
        self._t = datetime.datetime.now()
        self.hr, self.temp, self.spo2 = float(baseline_hr), float(baseline_temp), 98.0
        self.rr, self.hrv, self.rrv = 14.0, 45.0, 15.0

    def set_condition(self, condition: int) -> None:
        self.condition = condition

    def get_next_window(self) -> VitalsSample:
        self._t += datetime.timedelta(seconds=WINDOW_SECONDS)
        t_sec = (self._t - datetime.datetime.fromtimestamp(0)).total_seconds()
        
        # 1. Base Targets
        targets = {
            0: dict(hr=self.baseline_hr, temp=self.baseline_temp, spo2=98.5, rr=14.0, hrv=50.0, rrv=18.0, mov_mean=10),
            1: dict(hr=self.baseline_hr + 15, temp=self.baseline_temp + 1.0, spo2=95.0, rr=18.0, hrv=30.0, rrv=10.0, mov_mean=20),
            2: dict(hr=self.baseline_hr + 45, temp=self.baseline_temp + 2.5, spo2=88.0, rr=26.0, hrv=12.0, rrv=5.0, mov_mean=5),
        }
        target = targets[self.condition]
        
        # 2. Add Physiological Oscillations (Biological Rhythms)
        hr_osc = 2.0 * np.sin(t_sec / 240.0)  # 4-min period
        rr_osc = 1.0 * np.sin(t_sec / 60.0)   # 1-min period
        
        # 3. Dynamic Updates (Smoothing + Random Walk)
        alpha = 0.25 if self.condition == 2 else 0.08
        self.hr   += (target["hr"] + hr_osc - self.hr) * alpha + np.random.normal(0, 0.4)
        self.temp += (target["temp"] - self.temp) * alpha + np.random.normal(0, 0.02)
        self.spo2 += (target["spo2"] - self.spo2) * alpha + np.random.normal(0, 0.15)
        self.rr   += (target["rr"] + rr_osc - self.rr)   * alpha + np.random.normal(0, 0.2)
        self.hrv  += (target["hrv"]  - self.hrv)  * alpha + np.random.normal(0, 0.8)
        self.rrv  += (target["rrv"]  - self.rrv)  * alpha + np.random.normal(0, 0.8)
        
        movement = float(np.clip(np.random.normal(target["mov_mean"], 3), 0, 100))
        
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
