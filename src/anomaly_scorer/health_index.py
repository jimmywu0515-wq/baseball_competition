"""
Mechanics Stability Index (MSI) Module (§13 Critique Fixes)
IMPORTANT HYPOTHESIS & NAMING REFRAMING:
This metric is an inverse exponential transform of the Mahalanobis anomaly distance:
    MSI = 100 * exp(-alpha * D_M)
It measures the mechanical fidelity of the delivery relative to the pitcher's own baseline.
It does NOT directly measure biological fatigue.
"""
import numpy as np

def compute_mechanics_stability_index(
    anomaly_scores: np.ndarray, 
    decay_alpha: float = 0.40,
    min_floor: float = 5.0
) -> np.ndarray:
    """
    Computes Mechanics Stability Index (MSI) on a 0-100 scale:
    - MSI = 100: Delivery perfectly on pitcher's personal baseline.
    - MSI = 60: Moderate mechanical drift (Caution threshold).
    - MSI = 35: Severe mechanical drift (Danger threshold).
    """
    scores = np.asarray(anomaly_scores, dtype=float)
    msi = 100.0 * np.exp(-decay_alpha * np.maximum(0.0, scores))
    msi = np.clip(msi, min_floor, 100.0)
    return np.round(msi, 1)

# Backward compatibility alias
compute_pitcher_health_index = compute_mechanics_stability_index
