"""
Health Index Computation Module
Maps raw/calibrated multi-dimensional anomaly scores into an intuitive 0-100 Pitcher Health Index.
"""
import numpy as np
import pandas as pd

def compute_pitcher_health_index(
    anomaly_scores: np.ndarray, 
    decay_alpha: float = 0.45,
    min_floor: float = 5.0
) -> np.ndarray:
    """
    Computes Pitcher Health Index = 100 * exp(-alpha * anomaly_score).
    - Anomaly Score = 0 (perfect baseline) -> Health Index = 100.
    - Anomaly Score = 2.0 (moderate drift) -> Health Index ≈ 40.
    - Anomaly Score = 4.0 (severe degradation) -> Health Index ≈ 16.
    """
    scores = np.asarray(anomaly_scores, dtype=float)
    health = 100.0 * np.exp(-decay_alpha * np.maximum(0.0, scores))
    health = np.clip(health, min_floor, 100.0)
    return np.round(health, 1)
