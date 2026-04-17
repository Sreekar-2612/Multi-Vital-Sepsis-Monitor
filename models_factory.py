import logging
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest

logger = logging.getLogger(__name__)

def build_population_if() -> IsolationForest:
    """Load pre-trained population IF from disk, or fallback to synthetic generation."""
    path = "models/pop_if.pkl"
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    logger.warning("Pre-trained Pop-IF not found at %s. Generating synthetic fallback.", path)
    X_healthy = np.random.normal(0, 1, (1000, 7))
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X_healthy)
    return model

def build_random_forest() -> RandomForestClassifier:
    """Load pre-trained RF from disk, or fallback to synthetic generation."""
    path = "models/rf_model.pkl"
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    logger.warning("Pre-trained RF not found at %s. Generating synthetic fallback.", path)
    X = np.random.normal(0, 1, (100, 10))
    y = np.random.randint(0, 3, 100)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model
