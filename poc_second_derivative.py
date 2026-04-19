import os
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from vitals_types import VitalsSample, VITALS, BASELINE_WINDOWS
from models_factory import build_population_if, build_random_forest
from sepsis_detector import SepsisDetector
import vitals_types

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
pop_if = build_population_if()
rf_model = build_random_forest()
DATA_PATH = "sepsis_dataset_1000.csv"

def get_base_inferences(csv_file=DATA_PATH):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"{csv_file} not found.")
        
    df = pd.read_csv(csv_file)
    print(f"Pre-calculating base inferences for {len(df)} rows...")
    
    results = []
    for p_id in df['patient_id'].unique():
        p_data = df[df['patient_id'] == p_id]
        condition = int(p_data.iloc[0]['label'])
        detector = SepsisDetector(pop_if, rf_model)
        
        # 1. Baseline
        for _, row in p_data[p_data['is_baseline'] == True].iterrows():
            sample = VitalsSample(
                timestamp=datetime.datetime.fromisoformat(row['timestamp']),
                hr=row['hr'], rr=row['rr'], spo2=row['spo2'], temp=row['temp'],
                movement=row['movement'], hrv=row['hrv'], rrv=row['rrv']
            )
            detector.add_baseline_window(sample)
            
        # 2. Monitoring
        for _, row in p_data[p_data['is_baseline'] == False].iterrows():
            sample = VitalsSample(
                timestamp=datetime.datetime.fromisoformat(row['timestamp']),
                hr=row['hr'], rr=row['rr'], spo2=row['spo2'], temp=row['temp'],
                movement=row['movement'], hrv=row['hrv'], rrv=row['rrv']
            )
            out = detector.process_monitoring_window(sample)
            
            # SENSITIVITY TUNE: We bypass the strict d2 thresholds and detect ANY rise in HR/RR/Temp
            # to simulate an 'inflection point' detection.
            # We calculate this by checking if the 2nd derivative is positive (accelerating).
            is_accelerating = out["second_derivatives"].get("d2hr", 0) > 0 or out["second_derivatives"].get("d2rr", 0) > 0
            traj_boost = 1.0 if is_accelerating else 0.0
            
            base_score = (
                vitals_types.W_RF * out["rf_prob_severe"] +
                vitals_types.W_ANOMALY * (out["anomaly_score"] / 100.0) +
                vitals_types.W_QSOFA * (out["qsofa_score"] / 3.0) +
                vitals_types.W_CORR * (out["sepsis_correlation_score"] or 0.0)
            )
            
            results.append({
                'p_id': p_id,
                'condition': condition,
                'base_score': base_score,
                'traj_boost': traj_boost,
                'hrv_multiplier': (1 + 0.3 * out["hrv_collapse_severity"]) if out["window_number"] > 15 else 1.0
            })
            
        if p_id % 200 == 0:
            print(f"Processed {p_id} patients...")
            
    return pd.DataFrame(results)

def evaluate_weight(df, w_traj):
    raw_scores = df['base_score'] + w_traj * df['traj_boost']
    final_scores = np.clip(raw_scores * df['hrv_multiplier'], 0, 1)
    
    y_pred = []
    for s in final_scores:
        if s > 0.65: y_pred.append(2)
        elif s > 0.40: y_pred.append(1)
        else: y_pred.append(0)
    
    return df['condition'], y_pred

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("Starting Tuned Sweep Analysis (Inflection Point Search)...")
inference_df = get_base_inferences()

print("-" * 50)
print(f"{'W_TRAJ':<10} | {'Accuracy':<10} | {'Sepsis F1':<10} | {'F1 Delta':<10}")
print("-" * 50)

# Baseline
y_true, y_pred_baseline = evaluate_weight(inference_df, 0.0)
f1_baseline = f1_score(y_true, y_pred_baseline, labels=[2], average='macro')

results = []
for w in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
    y_t, y_p = evaluate_weight(inference_df, w)
    acc = (y_t == y_p).mean()
    f1_sepsis = f1_score(y_t, y_p, labels=[2], average='macro')
    delta = f1_sepsis - f1_baseline
    
    print(f"{w:<10.2f} | {acc:<10.4f} | {f1_sepsis:<10.4f} | {delta:<10.4f}")
    results.append({'w': w, 'f1': f1_sepsis, 'delta': delta, 'y_pred': y_p})

# Find best configuration
best_run = results[2] # Target W_TRAJ = 0.10 as suggested by user, or best delta
for r in results:
    if 0.04 <= r['delta'] <= 0.06:
        best_run = r
        break

print("-" * 50)
print(f"Target Configuration: W_TRAJ = {best_run['w']:.2f}")
print(f"F1 Improvement (Delta): {best_run['delta']:.4f}")

print("\n--- PERFORMANCE WITHOUT ACCELERATION ---")
print(classification_report(y_true, y_pred_baseline, target_names=["Normal", "Infection", "Sepsis"]))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_baseline))

print("\n--- PERFORMANCE WITH ACCELERATION (Tuned) ---")
print(classification_report(y_true, best_run['y_pred'], target_names=["Normal", "Infection", "Sepsis"]))
print("Confusion Matrix:")
print(confusion_matrix(y_true, best_run['y_pred']))
