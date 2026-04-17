import unittest
import json
import datetime
import os
from sepsis_detector import SepsisDetector
from simulator import PatientStreamSimulator
from models_factory import build_population_if, build_random_forest
from vitals_types import VitalsSample, BASELINE_WINDOWS

class TestImprovements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pop_if = build_population_if()
        cls.rf = build_random_forest()

    def test_persistence(self):
        """Verify that SepsisDetector can be serialized and resumed."""
        sim = PatientStreamSimulator(condition=0)
        detector = SepsisDetector(self.pop_if, self.rf)
        
        # 1. Establish baseline
        for _ in range(BASELINE_WINDOWS):
            detector.add_baseline_window(sim.get_next_window())
            
        # 2. Process some monitoring windows
        sim.set_condition(1)
        for _ in range(5):
            detector.process_monitoring_window(sim.get_next_window())
            
        # 3. Serialize
        state = detector.to_dict()
        state_json = json.dumps(state)
        
        # 4. Resume
        new_detector = SepsisDetector.from_dict(json.loads(state_json), self.pop_if, self.rf)
        
        # 5. Verify consistency
        sample = sim.get_next_window()
        out1 = detector.process_monitoring_window(sample)
        out2 = new_detector.process_monitoring_window(sample)
        
        self.assertEqual(out1["window_number"], out2["window_number"])
        self.assertEqual(out1["final_score"], out2["final_score"])
        self.assertEqual(out1["status"], out2["status"])
        print("Persistence test: PASS")

    def test_drift_gating(self):
        """Verify that baseline drift is paused during MILD_STRESS."""
        sim = PatientStreamSimulator(condition=0)
        detector = SepsisDetector(self.pop_if, self.rf)
        
        for _ in range(BASELINE_WINDOWS):
            detector.add_baseline_window(sim.get_next_window())
            
        # Initial drift means
        initial_drift = dict(detector._drift_means)
        
        # Simulate MILD_STRESS (condition 1)
        sim.set_condition(1)
        # We need to process a few windows so score_history reflects MILD_STRESS
        # Drift check uses prev_status. First window after baseline will have NORMAL.
        detector.process_monitoring_window(sim.get_next_window()) 
        
        # Now prev_status should be MILD_STRESS/HIGH_RISK
        current_drift = dict(detector._drift_means)
        detector.process_monitoring_window(sim.get_next_window())
        
        # Verify drift didn't change in the second window because status was not NORMAL
        self.assertEqual(detector._drift_means, current_drift)
        print("Drift gating test: PASS")

if __name__ == "__main__":
    unittest.main()
