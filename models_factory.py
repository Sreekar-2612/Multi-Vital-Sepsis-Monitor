import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier, IsolationForest

logger = logging.getLogger(__name__)

def build_population_if() -> IsolationForest:
    """Simulate a population-level model trained on 10,000 'healthy' patient records."""
    X_healthy = np.random.normal(0, 1, (1000, 7))
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X_healthy)
    return model

def build_random_forest() -> RandomForestClassifier:
    """Build a Random Forest classifier that maps 11 features to [NORMAL, MILD, SEVERE]."""
    X = np.random.normal(0, 1, (100, 11))
    y = np.random.randint(0, 3, 100)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model
