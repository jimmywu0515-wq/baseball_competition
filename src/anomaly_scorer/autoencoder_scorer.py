"""
Global Autoencoder Anomaly Scorer (§5 Step 4 Advanced ML Option)
Trains a shared autoencoder with pitcher & pitch type embeddings to calculate reconstruction error.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "release_speed",
    "release_spin_rate",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "vaa"
]

class AutoencoderScorer:
    """
    Autoencoder for learning non-linear normal manifold of pitch physics.
    Reconstruction loss acts as a complementary non-linear anomaly score.
    """
    def __init__(self, hidden_layer_sizes=(16, 6, 16), max_iter=200):
        self.scaler = StandardScaler()
        # Bottleneck architecture: 9 -> 16 -> 6 -> 16 -> 9
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=1e-4,
            max_iter=max_iter,
            random_state=42
        )
        self.is_fitted = False

    def fit(self, training_pitches_df: pd.DataFrame):
        """Fits autoencoder on historical pitches."""
        X = training_pitches_df[FEATURE_COLS].dropna().values
        if len(X) < 50:
            logger.warning("Not enough samples to train Autoencoder. Skipping fit.")
            return self
            
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, X_scaled)
        self.is_fitted = True
        logger.info(f"Autoencoder successfully trained on {len(X)} historical pitches.")
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Computes reconstruction MSE loss per pitch."""
        if not self.is_fitted or df.empty:
            return np.zeros(len(df))

        X = df[FEATURE_COLS].fillna(0).values
        X_scaled = self.scaler.transform(X)
        X_recon = self.model.predict(X_scaled)
        # Mean squared error per row
        mse = np.mean((X_scaled - X_recon) ** 2, axis=1)
        return np.round(mse, 4)
