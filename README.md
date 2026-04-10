# Person-Specific Sepsis Detection

A real-time, wearable-based sepsis detection system that builds a **personalized physiological baseline** for each patient and continuously monitors for deviation using confidence-weighted anomaly detection.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Pipeline — How It Works](#pipeline--how-it-works)
   - [Phase A: Baseline Establishment](#phase-a-baseline-establishment-t0--t200s)
   - [Phase B: Real-Time Monitoring](#phase-b-real-time-monitoring-t200s-onwards)
4. [Output JSON Schema](#output-json-schema)
5. [Confidence Modes](#confidence-modes)
6. [Sepsis Scoring Formula](#sepsis-scoring-formula)
7. [File Structure](#file-structure)
8. [How to Run](#how-to-run)
9. [Test Suite](#test-suite)
10. [Current Status](#current-status)
11. [Known Issues & Risk Areas](#known-issues--risk-areas)
12. [What's Left To Do](#whats-left-to-do)

---

## Project Overview

Most sepsis detection systems use **population-level thresholds** (e.g., HR > 100 bpm = abnormal). This system instead learns each patient's **personal normal** during a 5-window (200-second) calibration phase, then detects deviations from *their* baseline — not the population average.

**Key clinical safety principles:**
- **Never wait** — monitoring begins immediately after 5 windows, regardless of confidence
- **Never discard** — all 5 baseline windows are always used, even if quality is low
- **Graceful degradation** — low-confidence baselines fall back to population-level thresholds automatically

**7 vital parameters monitored:**
`HR` · `RR` · `SpO2` · `Temperature` · `Movement` · `HRV` · `RRV`

---

## Architecture

```
PatientStreamSimulator
        │
        ▼ (5 windows × 40s each)
BaselineEstablishment
  ├── 4-Component Confidence Score
  │     ├── Stability    (40%) — CV across 5 windows
  │     ├── Consistency  (35%) — values within clinical ranges
  │     ├── Activity     (15%) — low movement = better signal
  │     └── Variability  (10%) — HRV/RRV quality check
  └── Mode decision: LOCKED / HYBRID / FALLBACK
        │
        ▼ (window 6+, every 40s)
SepsisDetector
  ├── Artifact detection (movement spike filter)
  ├── Z-scores vs. personal baseline
  ├── DerivativeTracker → 1st + 2nd EMA-smoothed derivatives
  ├── AnomalyScorer → confidence-weighted blend
  ├── Sepsis feature functions
  │     ├── HRV collapse severity
  │     ├── Lactate proxy (multi-vital proxy)
  │     ├── Immobility score
  │     ├── Temperature trajectory slope
  │     └── Multi-system correlation (30-window rolling)
  ├── Random Forest (11 features)
  ├── qSOFA score
  └── Final score + status + phase → JSON output
```

---

## Pipeline — How It Works

### Phase A: Baseline Establishment (T=0 → T=200s)

Collects **5 consecutive 40-second windows** of all 7 vitals.

After window 5, computes a **4-component confidence score**:

| Component | Weight | Method |
|---|---|---|
| Stability | 40% | Per-vital coefficient of variation vs. max acceptable CV |
| Consistency | 35% | Fraction of readings within clinical valid ranges |
| Activity Quality | 15% | Low movement = good sensor contact; >30 unit spikes penalised |
| Variability Quality | 10% | HRV in [20, 200ms] and RRV in [5, 30ms] |

**Mode selection:**

| Confidence | Mode | Meaning |
|---|---|---|
| ≥ 75% | `LOCKED` | Full personalization — trust the individual baseline |
| 60–75% | `HYBRID` | Blend personal + population knowledge |
| < 60% | `FALLBACK` | Too noisy — use population standards only |

A personal `IsolationForest` is trained on the 5 raw-vital vectors at lock time.

---

### Phase B: Real-Time Monitoring (T=200s onwards)

Every 40 seconds, one window is processed:

1. **Artifact check** — if `movement > 2.5× baseline_mean`, skip anomaly update
2. **Baseline drift** — after 10 consecutive NORMAL windows, apply slow EWM correction (90/10 blend)
3. **Z-scores** — per-vital deviation from personal baseline (stds floored to clinical minimum)
4. **Derivatives** — EMA-smoothed rate of change (dX/dt) and acceleration (d²X/dt²)
5. **Anomaly score** — confidence-weighted blend (see Confidence Modes)
6. **Sepsis features** — HRV collapse, immobility, lactate proxy, temp slope
7. **RF classification** — 3-class (normal=0, mild=1, severe=2) with 11 features
8. **qSOFA score** — RR≥22, HR≥100, SpO2<92 (0–3)
9. **Trajectory boost** — from 2nd-derivative acceleration count (0.0–1.0)
10. **Final score** → **status** → **sepsis phase**

---

## Output JSON Schema

Every monitoring window (window 6+) emits a complete JSON:

```json
{
  "phase": "MONITORING",
  "window_number": 7,
  "timestamp": "2026-04-10T20:35:22",
  "baseline_state": "LOCKED",
  "baseline_confidence": 87.82,
  "baseline_confidence_breakdown": {
    "stability": 74.23,
    "consistency": 100.0,
    "activity_quality": 87.51,
    "variability_quality": 100.0
  },
  "artifact_contaminated": false,
  "derivatives_available": true,
  "vitals_current": {
    "hr": 95.4, "rr": 18.1, "spo2": 96.2,
    "temp": 37.6, "movement": 8.3, "hrv": 38.1, "rrv": 12.4
  },
  "z_scores": {
    "hr": 3.21, "rr": 2.84, "spo2": -1.72,
    "temp": 3.91, "movement": -0.51, "hrv": -1.24, "rrv": -0.87
  },
  "first_derivatives": { "dhr": 0.004, "drr": -0.001, ... },
  "second_derivatives": { "d2hr": -0.0005, "d2rr": 0.00003, ... },
  "anomaly_score": 42.1,
  "anomaly_method": "LOCKED: 50% PersonalIF + 50% Z-score",
  "rf_prob_normal": 0.42,
  "rf_prob_mild": 0.38,
  "rf_prob_severe": 0.20,
  "qsofa_score": 1,
  "trajectory_boost": 0.3,
  "hrv_collapse_severity": 0.0,
  "lactate_proxy": 0.18,
  "immobility_score": 0.05,
  "temp_trajectory_slope": 0.012,
  "multi_system_correlation": null,
  "sepsis_acceleration_count": 1,
  "final_score": 0.241,
  "status": "MILD_STRESS",
  "sepsis_phase": "PHASE_0_NORMAL",
  "consecutive_normal_count": 0,
  "score_history_length": 2
}
```

**Status levels:** `NORMAL` · `MILD_STRESS` · `HIGH_RISK` · `CRITICAL`

**Sepsis phases:** `PHASE_0_NORMAL` · `PHASE_1_EARLY` · `PHASE_2_INTERMEDIATE` · `PHASE_3_SEPTIC_SHOCK`

---

## Confidence Modes

| Mode | Condition | Anomaly Blending |
|---|---|---|
| `LOCKED` | Confidence ≥ 75% | 50% personal IF + 50% personal Z-score anomaly |
| `HYBRID` | 60% ≤ confidence < 75% | 60% of LOCKED blend + 40% population IF |
| `FALLBACK` | Confidence < 60% | 100% population IF (global population standards) |

Population IF is trained once at startup on 2,000 synthetic MIMIC-III-like patients (80% normal / 20% septic).

---

## Sepsis Scoring Formula

```
final_raw = 0.45 × RF_prob_severe
          + 0.28 × (anomaly_score / 100)
          + 0.17 × (qSOFA / 4)
          + 0.10 × trajectory_boost

# After 10+ monitoring windows:
final_score = min(1.0, final_raw × (1 + 0.3 × hrv_collapse_severity))
```

**Status thresholds:**
- `CRITICAL`     — final_score > 0.65
- `HIGH_RISK`    — final_score > 0.40 OR rf_prob_severe > 0.40
- `MILD_STRESS`  — rf_prob_mild > 0.50 OR anomaly_score > 50
- `NORMAL`       — everything else

---

## File Structure

```
SEPSIS_PERSON_SPECIFIC/
│
├── sepsis_detector_v2.py          # ✅ Production pipeline (1,037 lines)
│                                  #    VitalsSample → BaselineEstablishment →
│                                  #    DerivativeTracker → AnomalyScorer →
│                                  #    SepsisDetector
│
├── test_sepsis_v2.py              # ✅ Formal test suite — 48/48 passing
│                                  #    Tests: confidence paths, timeline,
│                                  #    schema, fallback integration
│
├── build_notebook.py              # ✅ Generates Stage2 notebook from v2 module
│
├── Person_Specific_Sepsis_Stage2.ipynb    # ✅ Generated — imports from v2
├── Patient_Specific_Sepsis_Analysis.ipynb # ⚠️ OLD — Stage 1, 1-min granularity
├── Realtime_Sepsis_Detection_Architecture.ipynb  # ⚠️ OLD — stale reference
│
├── sepsis_stage2.py               # 🟡 Original v1 pipeline — preserved
│
├── models_cohort/                 # ⚠️ 5,901 .pkl — trained on old pipeline
│                                  #    NOT used by sepsis_detector_v2.py
├── models/                        # Stage 1 model artifacts
├── models_stage2/                 # Stage 2 model artifacts
│
├── implementation_checklist.md    # ⚠️ Stale — all items still checked [ ]
├── confidence_fallback_summary.md # ✅ Design reference (valid)
└── prompt_with_fallback_strategy.md  # ✅ Anthropic API prompt spec
```

---

## How to Run

### Prerequisites
```bash
pip install numpy pandas scikit-learn
```

### Run the live demo
```bash
cd SEPSIS_PERSON_SPECIFIC
python sepsis_detector_v2.py
```

- Prints 5 baseline windows (Phase A) with confidence score
- Prints baseline lock summary (mode + confidence breakdown)
- Runs infinite monitoring loop at condition=2 (sepsis deterioration)
- Press `Ctrl+C` to stop

### Run the test suite
```bash
python test_sepsis_v2.py
```

Expected output:
```
RESULTS: 48/48 passed  |  0 failed
ALL TESTS PASSED
```

### Rebuild the Stage 2 notebook
```bash
python build_notebook.py
# Writes → Person_Specific_Sepsis_Stage2.ipynb
```

---

## Test Suite

`test_sepsis_v2.py` covers 6 test scenarios with 48 individual checks:

| Test | What it validates |
|---|---|
| **Test 1** — High confidence | Stable normal vitals → `LOCKED` mode, confidence ≥ 60%, personal IF trained |
| **Test 2** — Moderate confidence | Moderate-noise vitals → degraded mode (HYBRID or FALLBACK), not LOCKED |
| **Test 3** — Low confidence | Very noisy vitals → FALLBACK mode uses population-only anomaly scoring |
| **Test 4** — Timeline | Baseline locks **exactly** at window 5; window 6 starts monitoring immediately |
| **Test 5** — Output schema | All 30 required JSON fields present per window; status/score/phase valid |
| **Test 6** — Fallback integration | 5 windows always used; stds floored; history grows; MSC activates at 30 windows |

---

## Current Status

| Component | Status |
|---|---|
| Core pipeline (`sepsis_detector_v2.py`) | ✅ Complete |
| Formal test suite (48/48) | ✅ Passing |
| Stage 2 notebook | ✅ Regenerated |
| `build_notebook.py` | ✅ Updated |
| Stage 1 notebook (`Patient_Specific_Sepsis_Analysis.ipynb`) | ❌ Still old 1-min design |
| `Realtime_Sepsis_Detection_Architecture.ipynb` | ❌ Stale |
| `models_cohort/` (5,901 PKL files) | ⚠️ Orphaned — old pipeline |
| `implementation_checklist.md` | ⚠️ Stale — needs update |
| MIMIC-III validation | ❌ No real data available |

---

## Known Issues & Risk Areas

### 🔴 Critical

**1. Personal IsolationForest trained on only 5 samples**
The personal IF in `BaselineEstablishment` is fitted on exactly 5 data points. An IsolationForest with 5 samples is statistically unreliable — it will classify almost everything as normal. This is why `anomaly_score` frequently saturates at `50.0` in live output (the personal IF contributes near-zero signal; the Z-score half carries all the weight in LOCKED mode).

*Fix:* Increase baseline to 20–30 windows (T=800–1200s), or drop the personal IF from the LOCKED blend and use pure Z-score anomaly in that mode instead.

---

**2. `BaselineData` is mutated during monitoring (not actually immutable)**
The drift correction in `SepsisDetector.process_monitoring_window()` writes directly into `baseline.baseline_means`:
```python
baseline.baseline_means[v] = 0.9 * baseline.baseline_means[v] + 0.1 * new_value
```
The docstring says "Immutable snapshot" but the dict is modified in-place. Replaying the same stream will give different results.

*Fix:* Hold drift corrections in a separate `_drift_means` dict inside `SepsisDetector`. Leave `baseline_means` read-only.

---

### 🟡 Medium Risk

**3. `PHASE_3_SEPTIC_SHOCK` is unreachable**
`phase_detection()` requires `sbp_proxy <= 90` but `sbp_proxy` defaults to `120.0` everywhere it's called, and there is no SBP data source. This phase can never trigger.

**4. `DerivativeTracker._history` grows without bound**
`score_history` is capped at 360 entries but `_deriv_tracker._history` is never pruned. At 360 windows (4 hours), this list holds 360 `VitalsSample` objects — small, but will grow indefinitely in a multi-day deployment.

**5. Anomaly score scaling — personal IF signal is compressed**
`personal_score = np.clip(-personal_raw * 100, 0, 100)` — the `decision_function` returns values in approximately [-0.5, 0.5], so multiplying by 100 maps most signals to the 0–50 range aggressively. The scaling factor needs calibration against the actual IF output distribution.

**6. `multi_system_correlation` returns `None` for first 30 windows (20 minutes)**
The RF uses `msc_val = 0.0` as substitute, which underestimates sepsis risk during the first 20-minute observation window.

---

### 🟢 Minor

| Issue | Detail |
|---|---|
| Dead import | `asdict` imported from `dataclasses`, never used |
| `pandas` barely used | Only in `build_population_if()` — can be replaced with numpy |
| `WindowStats.stds/cvs` unused | Always set to 0.0, cross-window stats computed separately |
| `time.sleep(0.5)` in demo | Fine for local demo, must be removed/replaced in real deployment |

---

## What's Left To Do

### Priority 1 — Fix Before Production Use
- [ ] Fix personal IF: increase baseline windows to ≥20, or replace with pure Z-score anomaly scoring in LOCKED mode
- [ ] Fix `BaselineData` immutability — separate `_drift_means` from original locked values
- [ ] Prune `DerivativeTracker._history` to cap memory (max 360 entries)

### Priority 2 — Functional Improvements
- [ ] Add SBP/MAP estimation or remove `PHASE_3_SEPTIC_SHOCK` dependency on `sbp_proxy`
- [ ] Fix personal IF anomaly score scaling (calibrate the -personal_raw × scale factor)
- [ ] Improve `multi_system_correlation` — use a shorter warm-up window (e.g., 10 windows) for early monitoring

### Priority 3 — Documentation & Cleanup
- [ ] Update `Patient_Specific_Sepsis_Analysis.ipynb` to 40-second window pipeline
- [ ] Update `Realtime_Sepsis_Detection_Architecture.ipynb` to reference v2
- [ ] Update `implementation_checklist.md` to reflect all completed work
- [ ] Archive or delete `models_cohort/` (5,901 old-pipeline PKL files)

### Priority 4 — Validation (Requires External Data)
- [ ] Test on real MIMIC-III patient data
- [ ] ROC curve / sensitivity-specificity analysis
- [ ] Clinical threshold justification document
