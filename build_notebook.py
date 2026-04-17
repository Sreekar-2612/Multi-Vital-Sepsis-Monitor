import nbformat as nbf

nb = nbf.v4.new_notebook()

# ---------------------------------------------------------------------------
# Cell 0 -- Title markdown
# ---------------------------------------------------------------------------
md_title = """\
# Person-Specific Sepsis Detection -- Production Pipeline
**Version:** v3 (Sensitivity Enhanced)

### Scoring Architecture
The system uses a **Hybrid Weighted Fusion** approach to combine multiple signals:
- **ML Probability (20%):** Patterns recognized by a Random Forest.
- **qSOFA Clinical Rules (30%):** Classical medical thresholds (HR, RR, SpO2).
- **Anomaly Detection (30%):** Uses an **L2-Norm Z-score** for high sensitivity to single-vital spikes.
- **Trajectory & Correlation (20%):** Detection of physiological decoupling and acceleration.

### Phase Definitions
| Phase | Duration | What happens |
|---|---|---|
| **Phase A -- Baseline** | T=0 -> T=200s (5 × 40s windows) | Collect vitals, compute 4-component confidence, lock unique baseline |
| **Phase B -- Monitoring** | T=200s -> ∞ (every 40s) | Continuous Z-scoring, anomaly detection, and sepsis phase classification |
"""

# ---------------------------------------------------------------------------
# Cell 1 -- Imports and model construction
# ---------------------------------------------------------------------------
code_imports = """\
import json
import time
import logging
import sys
import os

# --- ENVIRONMENT DIAGNOSTIC ---
print("Kernel Current Working Directory:", os.getcwd())
print("Looking for Project Files...")

# Search for project root
potential_roots = [
    os.getcwd(),
    "/mnt/c/Users/sreek/OneDrive/Desktop/SEPSIS_PERSON_SPECIFIC",
    "C:/Users/sreek/OneDrive/Desktop/SEPSIS_PERSON_SPECIFIC"
]

found_path = None
for p in potential_roots:
    if os.path.exists(os.path.join(p, "vitals_types.py")):
        if p not in sys.path:
            sys.path.append(p)
        found_path = p
        break

if found_path:
    print(f"SUCCESS: Found project modules at {found_path}")
else:
    print("WARNING: Could not find project files! Please ensure .py files are in the same folder as this notebook.")
    print("Search Path attempted:", potential_roots)
    print("Actual Directory Contents:", os.listdir())

# -----------------------------

from IPython.display import clear_output
from vitals_types import VitalsSample, BASELINE_WINDOWS
from simulator import PatientStreamSimulator
from models_factory import build_population_if, build_random_forest
from sepsis_detector import SepsisDetector

print("Building shared models (population IF + Random Forest)...")
pop_if = build_population_if()
rf     = build_random_forest()
print("Ready.")
"""

# ---------------------------------------------------------------------------
# Cell 1.5 -- Display Model Metrics
# ---------------------------------------------------------------------------
code_metrics = """\
# Load and display model performance metrics
metrics_path = "models/metrics.json"
if os.path.exists(metrics_path):
    with open(metrics_path, 'r') as f:
        m = json.load(f)
    print("=== Model Performance Metrics ===")
    print(f"Accuracy:  {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}")
    print(f"Recall:    {m['recall']:.4f}")
    print(f"F1-Score:  {m['f1']:.4f}")
else:
    print("Note: Performance metrics not found. Run 'python train_and_save_models.py' to generate them.")
"""

# ---------------------------------------------------------------------------
# Cell 2 -- Initialise
# ---------------------------------------------------------------------------
code_init = """\
# 0=normal  1=mild infection  2=sepsis (rapid deterioration)
BASELINE_CONDITION = 0

sim      = PatientStreamSimulator(condition=BASELINE_CONDITION)
detector = SepsisDetector(population_if=pop_if, rf_model=rf)
print(f"Initialised with baseline condition={BASELINE_CONDITION}.")
"""

# ---------------------------------------------------------------------------
# Cell 3 -- Phase A markdown
# ---------------------------------------------------------------------------
md_phase_a = """\
## Phase A -- Baseline Establishment
Collect 5 × 40-second windows. After window 5 the baseline **auto-locks**
and prints the confidence score and mode selection.
"""

# ---------------------------------------------------------------------------
# Cell 4 -- Phase A code
# ---------------------------------------------------------------------------
code_phase_a = """\
print("=== PHASE A: Collecting 5-window Baseline ===\\n")
for _ in range(BASELINE_WINDOWS):
    sample = sim.get_next_window()
    result = detector.add_baseline_window(sample)
    if result:
        print("\\n--- BASELINE LOCKED ---")
        print(json.dumps({
            "confidence":           result.confidence,
            "mode":                 result.mode,
            "confidence_breakdown": result.confidence_breakdown,
            "baseline_means":       {k: round(v, 2) for k, v in result.baseline_means.items()},
        }, indent=2))
"""

# ---------------------------------------------------------------------------
# Cell 5 -- Phase B markdown
# ---------------------------------------------------------------------------
md_phase_b = """\
## Phase B -- Real-Time Monitoring
Runs 20 windows of simulated monitoring (condition=2 / rapid sepsis).
Set `DEMO_WINDOWS = None` for an infinite loop.
"""

# ---------------------------------------------------------------------------
# Cell 6 -- Phase B code
# ---------------------------------------------------------------------------
code_phase_b = """\
DEMO_WINDOWS = None   # Run indefinitely until stopped manually (Kernel -> Interrupt)
sim.set_condition(2)

print("=== PHASE B: Real-Time Monitoring (condition=2 / sepsis) ===\\n")
idx = 0
try:
    while DEMO_WINDOWS is None or idx < DEMO_WINDOWS:
        out = detector.process_monitoring_window(sim.get_next_window())
        clear_output(wait=True)
        print(f"Window {out['window_number']} | Status: {out['status']} | Phase: {out['sepsis_phase']}")
        print(f"Score: {out['final_score']:.3f} | Conf: {out['baseline_confidence']:.1f}% [{out['baseline_state']}]")
        print(f"qSOFA: {out['qsofa_score']} | HRV collapse: {out['hrv_collapse_severity']:.2f}")
        print()
        print(json.dumps(out, indent=2))
        time.sleep(0.8)
        idx += 1
except KeyboardInterrupt:
    print("\\nMonitoring stopped.")
"""

# ---------------------------------------------------------------------------
# Cell 7 -- Tuning guide
# ---------------------------------------------------------------------------
md_tuning = """\
## Tuning Confidence Thresholds

Edit the constants at the top of `sepsis_detector_v2.py`:
```python
CONFIDENCE_HIGH = 75.0   # >= this -> LOCKED
CONFIDENCE_MID  = 60.0   # < this -> FALLBACK; between -> HYBRID
```

Run formal test suite to validate all 6 test cases:
```bash
python test_sepsis_v2.py
```
"""

# ---------------------------------------------------------------------------
# Assemble and write
# ---------------------------------------------------------------------------
nb["cells"] = [
    nbf.v4.new_markdown_cell(md_title),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_metrics),
    nbf.v4.new_code_cell(code_init),
    nbf.v4.new_markdown_cell(md_phase_a),
    nbf.v4.new_code_cell(code_phase_a),
    nbf.v4.new_markdown_cell(md_phase_b),
    nbf.v4.new_code_cell(code_phase_b),
    nbf.v4.new_markdown_cell(md_tuning),
]

with open("Person_Specific_Sepsis_Stage2.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook written -> Person_Specific_Sepsis_Stage2.ipynb")
