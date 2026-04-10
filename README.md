# Person-Specific Sepsis Detection (Production Pipeline v2)

A modular, high-granularity sepsis detection system designed for real-time monitoring. The system learns a **personalized physiological baseline** for each patient and continuously scores risk using hybrid anomaly detection and clinical trajectory analysis.

---

## 🏗️ Architecture (Modular)

The monolith has been decomposed into focused modules to ensure production scalability and testability:

- **[vitals_types.py](vitals_types.py)**: Shared dataclasses (`VitalsSample`, `BaselineData`) and pipeline constants.
- **[baseline_establishment.py](baseline_establishment.py)**: Phase A logic — 5-window collection + **4-component confidence scoring** (Stability, Consistency, Activity, Variability).
- **[derivatives.py](derivatives.py)**: EMA-smoothed 1st/2nd derivative tracking with history capping.
- **[feature_engine.py](feature_engine.py)**: Clinical feature extraction (HRV collapse, MSC, **Shock Proxy**, lactate proxy, immobility).
- **[anomaly_scoring.py](anomaly_scoring.py)**: Hybrid blending engine (LOCKED/HYBRID/FALLBACK) with 20-window gating for Personal IF.
- **[sepsis_detector.py](sepsis_detector.py)**: Main orchestration engine. Handles **drift correction**, **plateau suppression**, and the final scoring loop.
- **[models_factory.py](models_factory.py)**: Setup for population-level Isolation Forest and Random Forest models.
- **[simulator.py](simulator.py)**: Physiological stream simulator for testing.

---

## 🛠️ Key Production Features

1. **Hybrid Confidence Fallback**: Blends personalized Isolation Forest and z-score signals with population-level models based on baseline quality.
2. **Adaptive Baseline Drift**: Slow EWM update mechanism corrects for physiological shifts while keeping the original locked baseline for audit.
3. **Clinical Shock Proxy**: Non-invasive Phase 3 detection using `HR_rise * SpO2_drop * HRV_collapse` (replaces dead SBP code).
4. **Disease vs Sepsis Plateau Suppression**: Prevents risk escalation if scores plateau in a mild stress band, indicating stable infection rather than sepsis progression.
5. **Bidirectional Temperature Monitoring**: Properly detects both hyperthermia and hypothermic (cold) sepsis.
6. **Standardized JSON Schema**: Every 40-second window emits a 30-field JSON payload for downstream clinical dashboards.

---

## 🚀 How to Run

### Live Monitoring Demo
```bash
# Runs Phase A (baseline) followed by Phase B (sepsis monitoring)
python sepsis_detector.py
```

### Verification Suite (49/49 Tests)
```bash
# Validates confidence modes, schema, timeline, and clinical logic
python test_sepsis_v2.py
```

### Notebook Generation
```bash
# Regenerates the Stage 2 Jupyter notebook with modular imports
python build_notebook.py
```

---

## 📊 Current Status

| Component | Status | Detail |
|---|---|---|
| **Modularity** | ✅ Complete | Split into 8 focused modules |
| **Verification** | ✅ Passing | 49/49 tests pass (0 failures) |
| **Schema** | ✅ Standardized | 30-field JSON (Phase/Status/Phase/Vitals/Derivs) |
| **Logic** | ✅ Production+ | Includes drift, shock proxy, and plateau suppression |
| **Validation** | ❌ P3 | Requires real-world MIMIC-III data |

---

## 📝 Known Logic Gaps
- **Personal IF Training**: Still fitted on 5 samples; scoring is gated behind a 20-window threshold for safety.
- **Scaling Calibration**: RF weights and anomaly scaling are based on research heuristics; Week 3 (MIMIC validation) will provide empirical thresholds.
