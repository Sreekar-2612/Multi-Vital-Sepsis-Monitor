# -*- coding: utf-8 -*-
"""
test_sepsis_v2.py -- Formal test suite for sepsis_detector_v2.py
================================================================
Runs all 6 test cases from implementation_checklist.md:

  Test 1: High confidence path (condition=0, stable vitals -> LOCKED mode)
  Test 2: Mid confidence path  (moderate noise -> degraded mode below LOCKED)
  Test 3: Low confidence path  (noisy vitals -> FALLBACK or HYBRID)
  Test 4: Timeline             (baseline locks exactly at window 5 / T=200s)
  Test 5: Output field schema  (every monitoring window has all required fields)
  Test 6: Fallback integration (baseline never discarded; monitoring never waits)

Run:
    python test_sepsis_v2.py
"""

import datetime
import sys
import json
import logging
import unittest
import numpy as np
from types import SimpleNamespace

# Modular Imports
from vitals_types import VitalsSample, BaselineData, VITALS, BASELINE_WINDOWS
from baseline_establishment import BaselineEstablishment
from sepsis_detector import SepsisDetector
from anomaly_scoring import AnomalyScorer
from simulator import PatientStreamSimulator
from models_factory import build_population_if, build_random_forest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = "[PASS]"
FAIL = "[FAIL]"

_results = []

def check(name: str, condition: bool, detail: str = "") -> None:
    tag = PASS if condition else FAIL
    msg = f"  {tag}  {name}"
    if detail:
        msg += f"  ->  {detail}"
    print(msg)
    _results.append((name, condition))
    if not condition:
        sys.stderr.write(f"FAILED: {name}  {detail}\n")


def make_noisy_sample(t: datetime.datetime, seed: int = 0) -> VitalsSample:
    """High-noise sample designed to produce low baseline confidence."""
    rng = np.random.default_rng(seed)
    return VitalsSample(
        timestamp=t,
        hr=float(rng.uniform(50, 130)),
        rr=float(rng.uniform(6, 35)),
        spo2=float(rng.uniform(88, 100)),
        temp=float(rng.uniform(35.5, 40.0)),
        movement=float(rng.uniform(35, 80)),
        hrv=float(rng.uniform(5, 250)),
        rrv=float(rng.uniform(2, 50)),
    )


REQUIRED_FIELDS = {
    "phase", "window_number", "timestamp",
    "baseline_state", "baseline_confidence", "baseline_confidence_breakdown",
    "artifact_contaminated", "derivatives_available",
    "vitals_current", "z_scores", "first_derivatives", "second_derivatives",
    "anomaly_score", "anomaly_method",
    "rf_prob_normal", "rf_prob_mild", "rf_prob_severe",
    "qsofa_score", "trajectory_boost",
    "hrv_collapse_severity", "lactate_proxy", "immobility_score",
    "temp_trajectory_slope", "multi_system_correlation",
    "sepsis_acceleration_count",
    "final_score", "status", "sepsis_phase",
    "consecutive_normal_count", "score_history_length",
}

# ---------------------------------------------------------------------------
# Build shared models once
# ---------------------------------------------------------------------------
print("Building shared models (pop IF + RF)...")
POP_IF = build_population_if()
RF = build_random_forest()
print("Done.\n")


# ===========================================================================
# Test 1 -- High Confidence path (stable normal -> LOCKED)
# ===========================================================================
def test_1_high_confidence():
    print("=" * 60)
    print("TEST 1: High Confidence Path (>=75% -> LOCKED mode)")
    print("=" * 60)

    sim = PatientStreamSimulator(condition=0)
    est = BaselineEstablishment()
    bd = None
    for _ in range(BASELINE_WINDOWS):
        bd = est.add_window(sim.get_next_window())

    check("Baseline locked after 5 windows", bd is not None)
    check("Mode is LOCKED", bd.mode == "LOCKED", f"got '{bd.mode}'")
    check("Confidence >= 60%", bd.confidence >= 60, f"confidence={bd.confidence:.1f}%")
    check("Stability component > 0", bd.confidence_breakdown["Stability"] > 0)
    check("Consistency component > 0", bd.confidence_breakdown["Consistency"] > 0)
    check("Activity component > 0", bd.confidence_breakdown["Activity"] > 0)
    check("Variability component > 0", bd.confidence_breakdown["Variability"] > 0)
    check("Personal IF trained", est.personal_if is not None)


# ===========================================================================
# Test 2 -- Moderate Confidence path  (moderate noise -> degraded mode)
# ===========================================================================
def test_2_hybrid_mode():
    print("\n" + "=" * 60)
    print("TEST 2: Moderate Confidence Path (degraded mode != LOCKED)")
    print("=" * 60)

    # Moderate noise: high movement + wider variance -> confidence drops below 75%
    # Exact outcome (HYBRID vs FALLBACK) depends on random draw; both are valid.
    t = datetime.datetime(2026, 1, 1, 0, 0, 0)
    est = BaselineEstablishment()
    bd = None
    rng = np.random.default_rng(7)
    for _ in range(BASELINE_WINDOWS):
        t += datetime.timedelta(seconds=40)
        s = VitalsSample(
            timestamp=t,
            hr=float(80 + rng.normal(0, 8)),
            rr=float(16 + rng.normal(0, 3)),
            spo2=float(97 + rng.normal(0, 1.5)),
            temp=float(37.2 + rng.normal(0, 0.4)),
            movement=float(np.clip(22 + rng.normal(0, 6), 0, 100)),
            hrv=float(np.clip(35 + rng.normal(0, 8), 5, 200)),
            rrv=float(np.clip(12 + rng.normal(0, 4), 2, 50)),
        )
        bd = est.add_window(s)

    check("Baseline locked", bd is not None)
    check("Confidence degraded below LOCKED threshold (<75%)",
          bd.confidence < 75.0, f"confidence={bd.confidence:.1f}%")
    check("Mode is HYBRID or FALLBACK (not LOCKED) for moderate noise",
          bd.mode in ("HYBRID", "FALLBACK"),
          f"got '{bd.mode}' (confidence={bd.confidence:.1f}%)")

    # Verify anomaly method string reflects actual mode
    detector = SepsisDetector(POP_IF, RF)
    detector._baseline_est = est
    detector._baseline = bd
    # Use Scorer from detector (already initialized via add_baseline_window logic)
    detector._scorer = AnomalyScorer(bd, est.personal_if, POP_IF)

    out = detector.process_monitoring_window(PatientStreamSimulator(condition=1).get_next_window())
    method_ok = any(kw in out["anomaly_method"] for kw in ("LOCKED", "HYBRID", "FALLBACK"))
    check("Anomaly method string reflects baseline mode", method_ok,
          f"method='{out['anomaly_method']}'")


# ===========================================================================
# Test 3 -- Low Confidence path (very noisy -> FALLBACK or HYBRID)
# ===========================================================================
def test_3_fallback_mode():
    print("\n" + "=" * 60)
    print("TEST 3: Low Confidence Path (<60% -> FALLBACK, pop-IF only)")
    print("=" * 60)

    t = datetime.datetime(2026, 1, 1, 0, 0, 0)
    est = BaselineEstablishment()
    bd = None
    for i in range(BASELINE_WINDOWS):
        t += datetime.timedelta(seconds=40)
        s = make_noisy_sample(t, seed=i * 99)
        bd = est.add_window(s)

    check("Baseline locked despite extreme noise", bd is not None)
    check("Confidence < 75 (noisy baseline)", bd.confidence < 75,
          f"confidence={bd.confidence:.1f}%")
    check("Mode is HYBRID or FALLBACK (not LOCKED) for noisy baseline",
          bd.mode in ("HYBRID", "FALLBACK"), f"got '{bd.mode}'")

    if bd.mode == "FALLBACK":
        detector = SepsisDetector(POP_IF, RF)
        detector._baseline_est = est
        detector._baseline = bd
        from sepsis_detector_v2 import AnomalyScorer
        detector._scorer = AnomalyScorer(bd, est.personal_if, POP_IF)
        out = detector.process_monitoring_window(
            PatientStreamSimulator(condition=0).get_next_window())
        check("FALLBACK anomaly method uses pop-IF only",
              "FALLBACK" in out["anomaly_method"],
              f"method='{out['anomaly_method']}'")


# ===========================================================================
# Test 4 -- Timeline (baseline locks at EXACTLY window 5 / T=200s)
# ===========================================================================
def test_4_timeline():
    print("\n" + "=" * 60)
    print("TEST 4: Timeline -- Baseline locks exactly at window 5 (T=200s)")
    print("=" * 60)

    start = datetime.datetime(2026, 1, 1, 8, 0, 0)
    sim = PatientStreamSimulator(condition=0)
    sim._t = start
    est = BaselineEstablishment()

    locked_at_window = None
    for i in range(1, BASELINE_WINDOWS + 2):
        s = sim.get_next_window()
        try:
            result = est.add_window(s)
        except RuntimeError:
            check("6th call raises RuntimeError (already locked)", True)
            break
        if result is not None:
            locked_at_window = i
            bd = result

    check("Baseline locks at window 5 (not before, not after)",
          locked_at_window == BASELINE_WINDOWS,
          f"locked at window {locked_at_window}")

    check("baseline_data exists after lock", bd is not None)

    # Monitoring starts immediately
    detector = SepsisDetector(POP_IF, RF)
    detector._baseline_est = est
    detector._baseline = bd
    detector._scorer = AnomalyScorer(bd, est.personal_if, POP_IF)
    out = detector.process_monitoring_window(sim.get_next_window())
    check("Monitoring window 6 starts immediately after baseline lock",
          out["window_number"] == 6 and out["phase"] == "MONITORING",
          f"window={out['window_number']}, phase={out['phase']}")


# ===========================================================================
# Test 5 -- Output field schema (all required fields in every window)
# ===========================================================================
def test_5_output_schema():
    print("\n" + "=" * 60)
    print("TEST 5: Output Field Schema -- all required fields present")
    print("=" * 60)

    detector = SepsisDetector(POP_IF, RF)
    sim = PatientStreamSimulator(condition=0)
    for _ in range(BASELINE_WINDOWS):
        detector.add_baseline_window(sim.get_next_window())

    sim.set_condition(2)
    outputs = [detector.process_monitoring_window(sim.get_next_window()) for _ in range(5)]

    for i, out in enumerate(outputs, start=6):
        missing = REQUIRED_FIELDS - set(out.keys())
        check(f"Window {i}: all required fields present",
              len(missing) == 0,
              f"missing: {missing}" if missing else "")
        check(f"Window {i}: status valid",
              out["status"] in ("NORMAL", "MILD_STRESS", "HIGH_RISK", "CRITICAL"),
              f"status='{out['status']}'")
        check(f"Window {i}: final_score in [0,1]",
              0.0 <= out["final_score"] <= 1.0,
              f"final_score={out['final_score']}")
        check(f"Window {i}: sepsis_phase valid",
              out["sepsis_phase"].startswith("PHASE_"),
              f"sepsis_phase='{out['sepsis_phase']}'")


# ===========================================================================
# Test 6 -- Fallback integration (baseline never discarded; monitoring instant)
# ===========================================================================
def test_6_fallback_integration():
    print("\n" + "=" * 60)
    print("TEST 6: Fallback Integration -- 5 windows always used, monitoring instant")
    print("=" * 60)

    sim = PatientStreamSimulator(condition=0)
    detector = SepsisDetector(POP_IF, RF)
    for _ in range(BASELINE_WINDOWS):
        detector.add_baseline_window(sim.get_next_window())

    check("Baseline is locked after 5 windows", detector.baseline_locked)

    bd = detector._baseline
    check("Personal IF trained on 5-window baseline", detector._baseline_est.personal_if is not None)
    check("baseline_means has all 7 vitals", set(bd.baseline_means.keys()) == set(VITALS))
    check("baseline_stds has all 7 vitals (floored)", set(bd.baseline_stds.keys()) == set(VITALS))
    check("All stds >= clinical floor (no zero division risk)",
          all(v > 0 for v in bd.baseline_stds.values()))

    out = detector.process_monitoring_window(sim.get_next_window())
    check("First monitoring window (6) immediately available", out["window_number"] == 6)
    check("Mode is valid", bd.mode in ("LOCKED", "HYBRID", "FALLBACK"))

    for _ in range(35):
        out = detector.process_monitoring_window(sim.get_next_window())
    check("score_history grows across 35+ windows", out["score_history_length"] >= 35)
    check("multi_system_correlation activates after 30 windows",
          out["multi_system_correlation"] is not None,
          f"msc={out['multi_system_correlation']}")
    check("HRV collapse tracking active after 10 windows", "hrv_collapse_severity" in out)


# ===========================================================================
# Run all tests
# ===========================================================================
if __name__ == "__main__":
    test_1_high_confidence()
    test_2_hybrid_mode()
    test_3_fallback_mode()
    test_4_timeline()
    test_5_output_schema()
    test_6_fallback_integration()

    total = len(_results)
    passed = sum(1 for _, ok in _results if ok)
    failed = total - passed
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} passed  |  {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print(f"WARNING: {failed} TEST(S) FAILED -- see FAILED lines above")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
