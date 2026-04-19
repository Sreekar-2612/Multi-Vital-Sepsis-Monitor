"""
poc_200_patients.py
===================
Self-contained proof of concept:
  - 200 patients (67 Normal / 67 Infection / 66 Sepsis)
  - CASE 1: WITH second derivative acceleration (d2 thresholds calibrated for simulator)
  - CASE 2: WITHOUT second derivative acceleration
Generates side-by-side confusion matrices and metric tables.
"""

import os, sys, datetime, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score, f1_score)
warnings.filterwarnings('ignore')

# --- Project imports ---
from vitals_types import VitalsSample, VITALS, BASELINE_WINDOWS
from models_factory import build_population_if, build_random_forest
from simulator import PatientStreamSimulator
from baseline_establishment import BaselineEstablishment
from anomaly_scoring import AnomalyScorer
from derivatives import DerivativeTracker
from feature_engine import (hrv_collapse_severity, immobility_score,
                             multi_system_correlation, temp_trajectory,
                             phase_detection)
from correlation_analyzer import SepsisCorrelationAnalyzer
import vitals_types

ARTIFACT_DIR = r"C:\Users\sreek\.gemini\antigravity\brain\85a336d8-76ec-41eb-82a9-80aefc82ca6e"
VIZ_DIR = r"c:\Users\sreek\OneDrive\Desktop\SEPSIS_PERSON_SPECIFIC\visualizations"

N_PATIENTS = 200
WINDOWS_PER_PATIENT = 30          # Monitoring windows per patient
BASELINE_WIN = BASELINE_WINDOWS   # 5

# ─── CALIBRATION ────────────────────────────────────────────────────────────
#  Simulator produces smooth 40-second steps: d2hr is typically 0.001–0.005.
#  Standard production thresholds (0.05) never fire.
#  Calibrated thresholds below are tuned to the simulator's actual 2nd-derivative range.
D2HR_THRESH   = 0.0003   # bpm/s² — fires on any meaningful HR acceleration
D2RR_THRESH   = 0.0005   # breaths/s²
D2TEMP_THRESH = 0.00002  # °C/s²
D2HRV_THRESH  = -0.001   # ms/s² — collapse (negative)
D2RRV_THRESH  = -0.001   # /s²

def accel_boost(d2, sensitivity='HIGH'):
    """
    HIGH sensitivity (Case 1 – WITH accel): calibrated to simulator d2 range.
    OFF (Case 2 – WITHOUT accel): always returns 0.
    """
    if sensitivity == 'OFF':
        return 0.0, 0

    cnt = sum([
        d2.get("d2hr",  0)  >  D2HR_THRESH,
        d2.get("d2rr",  0)  >  D2RR_THRESH,
        d2.get("d2temp",0)  >  D2TEMP_THRESH,
        d2.get("d2hrv", 0)  <  D2HRV_THRESH,
        d2.get("d2rrv", 0)  <  D2RRV_THRESH,
    ])
    if cnt >= 3: return 1.0, cnt
    if cnt == 2: return 0.6, cnt
    if cnt == 1: return 0.3, cnt
    return 0.0, 0

def run_patient(p_id, condition, pop_if, rf_model, accel_mode):
    """Simulate one patient; return list of (true_label, pred_label)."""
    sim = PatientStreamSimulator(condition=condition)
    sim.baseline_hr   = float(np.random.normal(72, 8))
    sim.baseline_temp = float(np.random.normal(36.8, 0.4))

    be       = BaselineEstablishment()
    baseline = None
    scorer   = None
    d_track  = DerivativeTracker()
    corr     = SepsisCorrelationAnalyzer()
    score_history = []

    # ── Baseline ──
    for _ in range(BASELINE_WIN):
        s = sim.get_next_window()
        b = be.add_window(s)
        if b:
            baseline = b
            scorer   = AnomalyScorer(baseline, be.personal_if, pop_if)

    if baseline is None:
        return []   # shouldn't happen

    locked_means = dict(baseline.baseline_means)
    drift_means  = dict(baseline.baseline_means)

    preds = []
    hrv_hist = []; mov_hist = []; tmp_hist = []

    for _ in range(WINDOWS_PER_PATIENT):
        s    = sim.get_next_window()
        true = condition

        # Z-scores
        z  = {v: (float(getattr(s,v)) - drift_means[v]) / baseline.baseline_stds[v]
              for v in VITALS}

        # Derivatives
        d1, d2, avail = d_track.update(s)
        tb, _          = accel_boost(d2, accel_mode) if avail else (0.0, 0)

        # Anomaly
        ascore, _ = scorer.score(s, z)

        # Feature history
        hrv_hist.append(s.hrv); mov_hist.append(s.movement); tmp_hist.append(s.temp)
        hrv_sev = hrv_collapse_severity(hrv_hist)
        immo    = immobility_score(mov_hist)
        t_traj  = temp_trajectory(tmp_hist)
        msc     = multi_system_correlation(score_history)
        msc_val = msc if msc else 0.0

        # RF
        rf_in   = [[s.hr, s.rr, s.spo2, s.temp, s.movement, s.hrv, s.rrv,
                    immo, t_traj, msc_val]]
        rf_prob = rf_model.predict_proba(rf_in)[0]
        rf_sev  = float(rf_prob[2]) if len(rf_prob) > 2 else 0.0

        # qSOFA
        qsofa = int(s.rr >= 22) + int(s.hr >= 100) + int(s.spo2 < 92)

        # Correlation
        cr      = corr.analyze(score_history) if len(score_history) >= 10 else None
        cscore  = cr["sepsis_correlation_score"] if cr else 0.0

        # Final score
        raw = (vitals_types.W_RF * rf_sev +
               vitals_types.W_ANOMALY * (ascore / 100.0) +
               vitals_types.W_QSOFA  * (qsofa / 3.0) +
               vitals_types.W_TRAJ   * tb +
               vitals_types.W_CORR   * cscore)

        hrv_mult  = (1 + 0.3 * hrv_sev) if len(score_history) >= 10 else 1.0
        final     = round(min(1.0, raw * hrv_mult), 4)

        # Status thresholds
        if final > vitals_types.STATUS_CRITICAL_THRESH: pred = 2
        elif final > vitals_types.STATUS_HIGH_RISK_THRESH: pred = 1
        else: pred = 0

        preds.append((true, pred))
        score_history.append({"z_scores": z, "vitals_current": s.to_dict(),
                               "status": ["NORMAL","MILD_STRESS","HIGH_RISK","CRITICAL"][pred]})

    return preds

def evaluate(pop_if, rf_model, accel_mode, label):
    print(f"\n{'='*55}\n{label}\n{'='*55}")
    np.random.seed(42)

    y_true, y_pred = [], []
    dist = {0: 67, 1: 67, 2: 66}   # class distribution

    p_id = 0
    for cond, cnt in dist.items():
        for _ in range(cnt):
            pairs = run_patient(p_id, cond, pop_if, rf_model, accel_mode)
            for t, p in pairs:
                y_true.append(t); y_pred.append(p)
            p_id += 1
            if p_id % 50 == 0:
                print(f"  Processed {p_id}/{N_PATIENTS} patients…")

    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=["Normal","Infection","Sepsis"],
                                digits=4))
    cm  = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    pr  = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"Accuracy : {acc:.4f}  |  Precision : {pr:.4f}  |  Recall : {rec:.4f}  |  F1 : {f1:.4f}")
    return cm, acc, pr, rec, f1

def plot_results(cm_with, cm_without, metrics_with, metrics_without):
    labels = ["Normal", "Infection", "Sepsis"]

    fig = plt.figure(figsize=(18, 10), facecolor="#0f1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Confusion matrices ──
    for col, (cm, title) in enumerate(
        [(cm_without, "WITHOUT Acceleration\n(Baseline)"),
         (cm_with,    "WITH Acceleration\n(Calibrated d2)")]):

        ax = fig.add_subplot(gs[0, col])
        sns.heatmap(cm, annot=True, fmt='d',
                    cmap='YlOrRd' if col == 1 else 'Blues',
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, linewidths=.5, cbar=False,
                    annot_kws={"size": 13, "weight": "bold"})
        ax.set_title(title, color='white', fontsize=13, pad=10)
        ax.set_xlabel("Predicted", color='#aaa', fontsize=10)
        ax.set_ylabel("Actual",    color='#aaa', fontsize=10)
        ax.tick_params(colors='#ccc')

    # ── Delta heatmap ──
    ax_delta = fig.add_subplot(gs[0, 2])
    delta_cm = cm_with.astype(int) - cm_without.astype(int)
    sns.heatmap(delta_cm, annot=True, fmt='+d',
                cmap='RdYlGn', center=0,
                xticklabels=labels, yticklabels=labels,
                ax=ax_delta, linewidths=.5, cbar=False,
                annot_kws={"size": 13, "weight": "bold"})
    ax_delta.set_title("Delta  (With − Without)", color='white', fontsize=13, pad=10)
    ax_delta.set_xlabel("Predicted", color='#aaa', fontsize=10)
    ax_delta.set_ylabel("Actual",    color='#aaa', fontsize=10)
    ax_delta.tick_params(colors='#ccc')

    # ── Bar comparison ──
    ax_bar = fig.add_subplot(gs[1, :])
    metric_names  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    vals_without  = list(metrics_without)
    vals_with     = list(metrics_with)
    x   = np.arange(len(metric_names))
    w   = 0.32

    bars1 = ax_bar.bar(x - w/2, vals_without, w, label="WITHOUT Acceleration",
                       color="#4e79a7", alpha=0.9)
    bars2 = ax_bar.bar(x + w/2, vals_with,    w, label="WITH Acceleration",
                       color="#f28e2b", alpha=0.9)

    # Annotate bars
    for b in bars1:
        ax_bar.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f"{b.get_height():.3f}", ha='center', va='bottom',
                    color='white', fontsize=9)
    for b in bars2:
        ax_bar.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f"{b.get_height():.3f}", ha='center', va='bottom',
                    color='white', fontsize=9)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_names, color='white', fontsize=12)
    ax_bar.set_ylim(0, 1.12)
    ax_bar.set_facecolor('#0f1117')
    ax_bar.tick_params(colors='#ccc')
    ax_bar.spines[:].set_color('#333')
    ax_bar.legend(fontsize=11, facecolor='#1c1e26', labelcolor='white')
    ax_bar.set_title("Metric Comparison: With vs Without Second Derivative Acceleration",
                     color='white', fontsize=13, pad=10)

    # Deltas as text
    for i, (a, b) in enumerate(zip(vals_with, vals_without)):
        d = a - b
        color = '#00e676' if d > 0 else '#ff5252'
        ax_bar.text(x[i], 1.06, f"Δ {d:+.3f}", ha='center', color=color, fontsize=10, weight='bold')

    plt.suptitle("Second Derivative (Acceleration) Proof of Concept  |  200 Patients",
                 color='white', fontsize=15, y=1.01, weight='bold')

    for ax in fig.get_axes():
        ax.set_facecolor('#1c1e26')

    out_path = os.path.join(VIZ_DIR, "poc_accel_comparison_200pts.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nVisualization saved to:\n  {out_path}")

# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading models…")
    pop_if   = build_population_if()
    rf_model = build_random_forest()

    cm_without, acc_wo, pr_wo, rec_wo, f1_wo = evaluate(pop_if, rf_model, 'OFF',
                                                          "CASE 2: WITHOUT Second Derivative Acceleration")
    cm_with,    acc_wi, pr_wi, rec_wi, f1_wi = evaluate(pop_if, rf_model, 'HIGH',
                                                          "CASE 1: WITH Second Derivative Acceleration")

    print("\n" + "="*55)
    print("SUMMARY OF DIFFERENCES")
    print("="*55)
    print(f"{'Metric':<14} {'WITHOUT':>10} {'WITH':>10} {'Delta':>10}")
    print("-"*45)
    for name, wo, wi in zip(
        ["Accuracy", "Precision", "Recall", "F1-Score"],
        [acc_wo, pr_wo, rec_wo, f1_wo],
        [acc_wi, pr_wi, rec_wi, f1_wi]):
        d = wi - wo
        print(f"{name:<14} {wo:>10.4f} {wi:>10.4f} {d:>+10.4f}")

    plot_results(cm_with, cm_without,
                 (acc_wi, pr_wi, rec_wi, f1_wi),
                 (acc_wo, pr_wo, rec_wo, f1_wo))
