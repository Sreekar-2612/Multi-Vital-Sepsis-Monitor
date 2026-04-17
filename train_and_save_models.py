import os
import json
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_save():
    print("Creating 'models' directory...")
    os.makedirs("models", exist_ok=True)

    # 1. Train Population Isolation Forest (for Anomaly Detection)
    # We simulate 1000 healthy patients with normal vitals variance
    print("Training Population Isolation Forest...")
    X_healthy = np.random.normal(loc=[75, 14, 98, 36.8, 10, 45, 15], 
                                 scale=[5, 2, 1, 0.2, 3, 10, 3], 
                                 size=(1000, 7))
    pop_if = IsolationForest(contamination=0.01, random_state=42)
    pop_if.fit(X_healthy)

    with open("models/pop_if.pkl", "wb") as f:
        pickle.dump(pop_if, f)
    print("Saved -> models/pop_if.pkl")

    # 2. Train Random Forest Classifier (for Sepsis Classification)
    # 0: Normal, 1: Infection, 2: Sepsis
    # Features (10): hr, rr, spo2, temp, movement, hrv, rrv, immo, t_traj, msc
    print("Training Random Forest Classifier...")
    
    # Generate Synthetic Dataset with HEAVY noise and overlap for realism
    X = []
    y = []

    class_configs = [
        {
            "label": 0, # Normal
            "loc":   [75, 14, 98, 36.6, 10, 45, 15, 0.05, 0.0, 0.1], 
            "scale": [15, 6, 3, 0.6, 6, 20, 8, 0.15, 0.002, 0.15],
            "n": 600
        },
        {
            "label": 1, # Infection (overlaps with both)
            "loc":   [105, 24, 93, 37.8, 15, 22, 7, 0.35, 0.003, 0.3], 
            "scale": [25, 10, 6, 1.0, 10, 15, 6, 0.25, 0.004, 0.3],
            "n": 600
        },
        {
            "label": 2, # Sepsis (overlaps heavily with infection)
            "loc":   [125, 32, 86, 39.2, 5, 10, 4, 0.8, 0.009, 0.7], 
            "scale": [35, 12, 10, 1.8, 6, 10, 4, 0.2, 0.006, 0.25],
            "n": 600
        }
    ]

    for config in class_configs:
        # Use more variance in sampling
        data = np.random.normal(loc=config["loc"], scale=config["scale"], size=(config["n"], 10))
        # Clip values to physically possible ranges
        data[:, 2] = np.clip(data[:, 2], 50, 100) # SpO2 %
        data[:, 7] = np.clip(data[:, 7], 0, 1)    # Immobility
        X.append(data)
        y.extend([config["label"]] * config["n"])

    X = np.vstack(X)
    y = np.array(y)

    # Shuffle for better training
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Train-Test Split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Calculate Metrics
    y_pred = rf.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average='weighted'), 4),
        "recall": round(recall_score(y_test, y_pred, average='weighted'), 4),
        "f1": round(f1_score(y_test, y_pred, average='weighted'), 4)
    }

    print("\nModel Performance Metrics:")
    for k, v in metrics.items():
        print(f"  - {k.capitalize()}: {v}")

    # Save Metrics
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved -> models/metrics.json")

    with open("models/rf_model.pkl", "wb") as f:
        pickle.dump(rf, f)
    print("Saved -> models/rf_model.pkl")

    print("\nSUCCESS: Both models are trained and saved. You can now run the Notebook.")

if __name__ == "__main__":
    train_and_save()
