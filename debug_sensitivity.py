import datetime
import numpy as np
from vitals_types import VitalsSample
from sepsis_detector import SepsisDetector
from models_factory import build_population_if, build_random_forest

def debug_sensitivity():
    print("Building models...")
    pop_if = build_population_if()
    rf = build_random_forest()
    detector = SepsisDetector(pop_if, rf)

    # 1. Establish baseline (Normal: HR=75, Temp=36.8, etc.)
    print("Establishing baseline...")
    base_t = datetime.datetime.now()
    for i in range(5):
        sample = VitalsSample(
            timestamp=base_t + datetime.timedelta(seconds=i*40),
            hr=75.0, rr=14.0, spo2=98.0, temp=36.8, movement=10.0, hrv=45.0, rrv=15.0
        )
        detector.add_baseline_window(sample)
    
    print(f"Baseline Mode: {detector._baseline.mode}, Confidence: {detector._baseline.confidence}%")

    # 2. Simulate 130 HR spike
    print("\nSimulating 130 HR spike (all other vitals normal)...")
    spike_sample = VitalsSample(
        timestamp=base_t + datetime.timedelta(seconds=240),
        hr=130.0, rr=14.0, spo2=98.0, temp=36.8, movement=10.0, hrv=45.0, rrv=15.0
    )
    output = detector.process_monitoring_window(spike_sample)
    
    print(f"Window: {output['window_number']}")
    print(f"HR: {output['vitals_current']['hr']}")
    print(f"Z-Scores: {output['z_scores']}")
    print(f"Anomaly Score: {output['anomaly_score']} ({output['anomaly_method']})")
    print(f"RF Severe Prob: {output['rf_prob_severe']}")
    print(f"qSOFA: {output['qsofa_score']}")
    print(f"Final Score: {output['final_score']}")
    print(f"Status: {output['status']}")

if __name__ == "__main__":
    debug_sensitivity()
