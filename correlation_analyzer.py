import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from vitals_types import VITALS

class SepsisCorrelationAnalyzer:
    """
    Analyzes synchronization and decoupling across 21 vital sign pairs.
    Uses a 30-window rolling window (20 mins at 40s/window).
    """
    
    # 21 Unique Pairs
    PAIRS = [
        ('hr', 'rr'), ('hr', 'spo2'), ('hr', 'temp'), ('hr', 'movement'), ('hr', 'hrv'), ('hr', 'rrv'),
        ('rr', 'spo2'), ('rr', 'temp'), ('rr', 'movement'), ('rr', 'hrv'), ('rr', 'rrv'),
        ('spo2', 'temp'), ('spo2', 'movement'), ('spo2', 'hrv'), ('spo2', 'rrv'),
        ('temp', 'movement'), ('temp', 'hrv'), ('temp', 'rrv'),
        ('movement', 'hrv'), ('movement', 'rrv'),
        ('hrv', 'rrv')
    ]

    # Clinical Weights (Importance to Sepsis Detection)
    WEIGHTS = {
        ('hr', 'hrv'): 2.0, ('rr', 'spo2'): 2.0, 
        ('hr', 'rr'): 1.5, ('hr', 'temp'): 1.5, ('hrv', 'rrv'): 1.5,
        ('temp', 'movement'): 1.2, ('rr', 'hrv'): 1.2, ('hr', 'movement'): 1.0,
        ('hr', 'rrv'): 1.0, ('hr', 'spo2'): 1.2, ('rr', 'temp'): 1.0, ('rr', 'movement'): 1.0,
        ('rr', 'rrv'): 1.0, ('spo2', 'temp'): 1.0, ('spo2', 'movement'): 1.0,
        ('spo2', 'hrv'): 1.0, ('spo2', 'rrv'): 1.0, ('temp', 'hrv'): 1.0,
        ('temp', 'rrv'): 1.0, ('movement', 'hrv'): 0.5, ('movement', 'rrv'): 0.5
    }

    # Literature-based Sepsis Abnormality Thresholds (from PART 1)
    # True if r falls in these ranges
    THRESHOLDS = {
        ('hr', 'rr'): lambda r: r < 0.15 or r > 0.75,
        ('hr', 'spo2'): lambda r: r < -0.6,
        ('hr', 'temp'): lambda r: r < 0.2,
        ('hr', 'movement'): lambda r: r < 0.3, # HR high while movement low
        ('hr', 'hrv'): lambda r: r > -0.1,    # Loss of inverse coupling
        ('hr', 'rrv'): lambda r: r > -0.1,
        ('rr', 'spo2'): lambda r: r < -0.7,
        ('rr', 'temp'): lambda r: r < 0.1,
        ('rr', 'movement'): lambda r: r < 0.2,
        ('rr', 'hrv'): lambda r: r < -0.7,
        ('rr', 'rrv'): lambda r: r > -0.2,
        ('spo2', 'temp'): lambda r: r < -0.2,
        ('spo2', 'movement'): lambda r: r < -0.3,
        ('spo2', 'hrv'): lambda r: r < 0.05,
        ('spo2', 'rrv'): lambda r: r < 0.05,
        ('temp', 'movement'): lambda r: r < 0.0, # Sepsis Imprinting
        ('temp', 'hrv'): lambda r: r > -0.1,
        ('temp', 'rrv'): lambda r: r > -0.05,
        ('movement', 'hrv'): lambda r: r < 0.2,
        ('movement', 'rrv'): lambda r: r < 0.15,
        ('hrv', 'rrv'): lambda r: r < 0.3
    }

    def __init__(self, window_size: int = 30, activation_threshold: int = 15):
        self.window_size = window_size
        self.activation_threshold = activation_threshold
        self._history: List[Dict] = []

    def analyze(self, score_history: List[Dict]) -> Optional[Dict]:
        """
        Input: score_history[] from SepsisDetector
        Returns: Correlation Analysis dict or None if under warmup.
        """
        n = len(score_history)
        if n < self.activation_threshold:
            return None

        # Extract vitals into DataFrame for rolling correlation
        df_list = []
        for h in score_history[-self.window_size:]:
            v = h['vitals_current']
            # Reconstruct dict to match PAIRS keys
            df_list.append({
                'hr': v['hr'], 'rr': v['rr'], 'spo2': v['spo2'], 'temp': v['temp'],
                'movement': v['movement'], 'hrv': v['hrv'], 'rrv': v['rrv']
            })
        df = pd.DataFrame(df_list)

        fingerprint = {}
        abnormal_pairs = []
        total_weighted_abnormality = 0.0
        max_possible_weight = sum(self.WEIGHTS.values())

        for p1, p2 in self.PAIRS:
            # Pearson Correlation
            r = df[p1].corr(df[p2], method='pearson')
            if np.isnan(r): r = 0.0
            
            p_name = f"{p1.upper()}_{p2.upper()}"
            fingerprint[p_name] = round(float(r), 4)

            # Check Abnormality
            is_abnormal = self.THRESHOLDS[(p1, p2)](r)
            if is_abnormal:
                abnormal_pairs.append(p_name)
                total_weighted_abnormality += self.WEIGHTS[(p1, p2)]

        # 2c. Composite Score (0.0 - 1.0)
        sepsis_correlation_score = round(total_weighted_abnormality / max_possible_weight, 4)

        # 2d. Disease Discriminator
        probs, dominant = self._disease_discriminator(fingerprint)

        return {
            "correlation_module_active": True,
            "windows_used_for_correlation": n,
            "sepsis_correlation_score": sepsis_correlation_score,
            "correlation_fingerprint": fingerprint,
            "abnormal_correlation_pairs": abnormal_pairs,
            "abnormal_pair_count": len(abnormal_pairs),
            "disease_probabilities": probs,
            "dominant_disease_category": dominant,
            "correlation_confidence": "HIGH" if n >= self.window_size else "MEDIUM"
        }

    def _disease_discriminator(self, fp: Dict[str, float]) -> Tuple[Dict[str, float], str]:
        """
        Compares the 21-pair fingerprint against ideal templates for each disease.
        """
        # (Template definitions based on clinical characteristic fingerprints)
        templates = {
            "sepsis": {
                "HR_RR": 0.85, "HR_TEMP": 0.05, "TEMP_MOVEMENT": -0.4, 
                "HR_HRV": 0.0, "HRV_RRV": 0.1, "RR_SPO2": -0.85
            },
            "simple_infection": {
                "HR_RR": 0.45, "HR_TEMP": 0.55, "TEMP_MOVEMENT": 0.2, 
                "HR_HRV": -0.3, "HRV_RRV": 0.5, "RR_SPO2": -0.2
            },
            "cardiac_event": {
                "HR_RR": 0.1, "HR_TEMP": 0.0, "HR_HRV": -0.8, 
                "HR_MOVEMENT": 0.1, "RR_SPO2": -0.1
            },
            "respiratory_distress": {
                "HR_RR": 0.6, "RR_SPO2": -0.9, "RR_RRV": 0.1, 
                "HR_SPO2": -0.7, "HR_MOVEMENT": 0.2
            },
            "normal_stress": {
                "HR_RR": 0.4, "HR_MOVEMENT": 0.7, "TEMP_MOVEMENT": 0.4, 
                "HR_HRV": -0.4, "RR_SPO2": -0.1
            }
        }

        scores = {}
        for disease, template in templates.items():
            # Calculate similarity (Inverse of Euclidean distance on shared keys)
            dist = 0.0
            for k, target in template.items():
                dist += (fp.get(k, 0.0) - target)**2
            scores[disease] = 1.0 / (1.0 + np.sqrt(dist))

        # Softmax normalize to sum to 1.0
        total = sum(scores.values())
        norm_probs = {k: round(v / total, 4) for k, v in scores.items()}
        
        dominant = max(norm_probs, key=norm_probs.get)
        return norm_probs, dominant
