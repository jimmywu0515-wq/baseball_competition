"""
Intra-Game Shrinkage Calibration (§5 Step 3)
Calibrates historical baseline with first 15-20 pitches of the game using Empirical Bayes shrinkage.
"""
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

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

class ShrinkageCalibrator:
    """
    Calibrates baseline vectors using early-game sample statistics.
    Prevents overfitting compared to training a fresh model on 20 pitches.
    """
    def __init__(self, intra_game_calibration_pitches: int = 20, shrinkage_lambda: float = 0.35):
        self.calib_pitches = intra_game_calibration_pitches
        self.shrinkage_lambda = shrinkage_lambda

    def calibrate_game_pitches(self, 
                              game_pitches: pd.DataFrame, 
                              pitcher_id: int, 
                              game_pk: int, 
                              baseline_store: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Computes calibrated feature residuals and z-scores for a single game.
        """
        df = game_pitches.copy().sort_values("pitch_number_in_game").reset_index(drop=True)
        
        # 1. Early-game subset (first N pitches)
        early_subset = df[df["pitch_number_in_game"] <= self.calib_pitches]
        
        calibrated_means: Dict[str, np.ndarray] = {}

        # 2. Compute shrinkage-adjusted mean vector per pitch type
        for pt in df["pitch_type"].unique():
            key = f"{pitcher_id}_{game_pk}_{pt}"
            base_info = baseline_store.get(key)
            if base_info is None:
                # Fallback if specific baseline is missing
                continue

            hist_mu = base_info["mu_vec"]
            
            pt_early = early_subset[early_subset["pitch_type"] == pt]
            if len(pt_early) >= 3:
                early_mean = pt_early[FEATURE_COLS].mean().values
                # Shrinkage formula: mu_calib = (1 - lambda) * hist_mu + lambda * early_mean
                mu_calib = (1.0 - self.shrinkage_lambda) * hist_mu + self.shrinkage_lambda * early_mean
            else:
                mu_calib = hist_mu

            calibrated_means[pt] = mu_calib

        # 3. Add z-scores and calibrated residuals to dataframe
        for col_idx, col_name in enumerate(FEATURE_COLS):
            z_col = f"z_{col_name}"
            calib_col = f"calib_delta_{col_name}"
            
            # Apply per pitch type
            def compute_z(row):
                pt = row["pitch_type"]
                key = f"{pitcher_id}_{game_pk}_{pt}"
                base = baseline_store.get(key)
                if base and not np.isnan(row.get(col_name, np.nan)):
                    mu = base["mu_vec"][col_idx]
                    sig = max(base["sigma_vec"][col_idx], 1e-4)
                    return (row[col_name] - mu) / sig
                return 0.0

            def compute_calib_delta(row):
                pt = row["pitch_type"]
                if pt in calibrated_means and not np.isnan(row.get(col_name, np.nan)):
                    mu_c = calibrated_means[pt][col_idx]
                    return row[col_name] - mu_c
                return 0.0

            df[z_col] = df.apply(compute_z, axis=1).round(4)
            df[calib_col] = df.apply(compute_calib_delta, axis=1).round(4)

        return df, calibrated_means
