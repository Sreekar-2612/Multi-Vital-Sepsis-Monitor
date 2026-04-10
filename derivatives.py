from typing import Dict, List, Optional, Tuple
from vitals_types import VitalsSample, VITALS, WINDOW_SECONDS

class DerivativeTracker:
    """
    Calculates EMA-smoothed first and second derivatives.
    EMA smoothing weight α=0.3 suppresses noise in 1st derivative.
    Capped history (last 3 samples) prevents memory growth.
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
        """
        self._history.append(sample)
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
        if dt <= 0: dt = WINDOW_SECONDS

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
            dt2 = (prev.timestamp - prev2.timestamp).total_seconds()
            if dt2 <= 0: dt2 = WINDOW_SECONDS
            prev_raw_d1 = {
                f"d{v}": (float(getattr(prev, v)) - float(getattr(prev2, v))) / dt2
                for v in VITALS
            }
            for v in VITALS:
                key1, key2 = f"d{v}", f"d2{v}"
                d2[key2] = round((raw_d1[key1] - prev_raw_d1[key1]) / dt, 8)
            available = True

        return d1, d2, available
