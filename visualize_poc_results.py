import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
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
ARTIFACT_DIR = r"C:\Users\sreek\.gemini\antigravity\brain\85a336d8-76ec-41eb-82a9-80aefc82ca6e"

def get_patient_scores(p_id, use_acceleration: bool):
    df = pd.read_csv(DATA_PATH)
    p_data = df[df['patient_id'] == p_id]
    
    detector = SepsisDetector(pop_if, rf_model)
    original_w_traj = vitals_types.W_TRAJ
    vitals_types.W_TRAJ = 0.10 if use_acceleration else 0.0
    
    scores = []
    boosts = []
    
    # Baseline
    for _, row in p_data[p_data['is_baseline'] == True].iterrows():
        sample = VitalsSample(
            timestamp=datetime.datetime.fromisoformat(row['timestamp']),
            hr=row['hr'], rr=row['rr'], spo2=row['spo2'], temp=row['temp'],
            movement=row['movement'], hrv=row['hrv'], rrv=row['rrv']
        )
        detector.add_baseline_window(sample)
        
    # Monitoring
    for _, row in p_data[p_data['is_baseline'] == False].iterrows():
        sample = VitalsSample(
            timestamp=datetime.datetime.fromisoformat(row['timestamp']),
            hr=row['hr'], rr=row['rr'], spo2=row['spo2'], temp=row['temp'],
            movement=row['movement'], hrv=row['hrv'], rrv=row['rrv']
        )
        res = detector.process_monitoring_window(sample)
        scores.append(res['final_score'])
        boosts.append(res['trajectory_boost'])
        
    vitals_types.W_TRAJ = original_w_traj
    return scores, boosts

def plot_comparison():
    # Pick a Sepsis patient (condition 2)
    df = pd.read_csv(DATA_PATH)
    sepsis_patients = df[df['label'] == 2]['patient_id'].unique()
    p_id = sepsis_patients[0]
    
    print(f"Plotting scores for patient {p_id} (Sepsis)...")
    scores_acc, boosts_acc = get_patient_scores(p_id, True)
    scores_no, _ = get_patient_scores(p_id, False)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(scores_acc, label='With Acceleration Tracking', color='coral', linewidth=2)
    plt.plot(scores_no, label='Without Acceleration Tracking', color='steelblue', linestyle='--')
    plt.axhline(0.65, color='red', linestyle=':', label='CRITICAL Threshold')
    plt.title(f"Sepsis Detection: Impact of Acceleration Tracking (Patient {p_id})")
    plt.ylabel("Final Sepsis Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.bar(range(len(boosts_acc)), boosts_acc, color='orange', alpha=0.6, label='Trajectory Boost Level')
    plt.ylabel("Boost Value")
    plt.xlabel("Time (Windows)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, "score_comparison.png"))
    plt.close()

def plot_heatmaps():
    # Identical confusion matrices from PoC
    cm = np.array([[81, 18936, 33], [7, 14876, 167], [0, 2979, 12921]])
    labels = ["Normal", "Infection", "Sepsis"]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix (Common to both cases)")
    plt.ylabel("Actual State")
    plt.xlabel("Predicted Status")
    
    plt.savefig(os.path.join(ARTIFACT_DIR, "confusion_heatmap.png"))
    plt.close()

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        plot_comparison()
        plot_heatmaps()
        print("Visualizations generated in artifacts directory.")
    else:
        print("Dataset not found. Run generate_dataset.py first.")
