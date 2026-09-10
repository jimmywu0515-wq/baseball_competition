"""
Intra-Outing Shrinkage Calibration (§5 Step 3 & Critique Fixes)
Calibrates baseline vectors using early-game sample statistics.
CRITICAL TEMPORAL ISOLATION:
1. First N (default 20) pitches are used ONLY to calibrate the mean for subsequent pitches.
2. Pitches 1..20 are flagged as `is_calibration_phase = True`.
3. Calibrated alert scoring strictly begins at pitch 21. Changing tomorrow's or pitch 20's data
   never alters pitch 1..19 evaluations!
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
    "spin_axis_cos",
    "spin_axis_sin",
    "pfx_x",
    "pfx_z",
    "vaa"
]

class ShrinkageCalibrator:
    """
    Calibrates baseline vectors using early-outing pitches (pitches 1 to 20).
    Enforces that alerts only begin after the calibration period.
    """
    def __init__(self, intra_game_calibration_pitches: int = 20, shrinkage_lambda: float = 0.35):
        self.calib_pitches = intra_game_calibration_pitches
        self.shrinkage_lambda = shrinkage_lambda

    def calibrate_outing_pitches(self, 
                                 outing_df: pd.DataFrame, 
                                 pitcher_id: int, 
                                 game_pk: int, 
                                 baseline_store: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Computes calibrated feature residuals for an outing.
        Pitches 1..20 are marked is_calibration_phase = True.
        """
        df = outing_df.copy().sort_values("pitch_number_in_outing").reset_index(drop=True)
        
        # Mark calibration phase
        df["is_calibration_phase"] = df["pitch_number_in_outing"] <= self.calib_pitches
        early_subset = df[df["is_calibration_phase"]]

        calibrated_means: Dict[str, np.ndarray] = {}

        for pt in df["pitch_type"].unique():
            key = f"{pitcher_id}_{game_pk}_{pt}"
            base_info = baseline_store.get(key)
            if base_info is None or base_info.get("status") != "QUALIFIED":
                continue

            hist_mu = base_info["mu_vec"]
            pt_early = early_subset[early_subset["pitch_type"] == pt]

            if len(pt_early) >= 3:
                early_mean = pt_early[FEATURE_COLS].mean().values
                mu_calib = (1.0 - self.shrinkage_lambda) * hist_mu + self.shrinkage_lambda * early_mean
            else:
                mu_calib = hist_mu

            calibrated_means[pt] = mu_calib

        # Add z-scores and calibrated deltas
        for col_idx, col_name in enumerate(FEATURE_COLS):
            z_col = f"z_{col_name}"
            calib_col = f"calib_delta_{col_name}"

            def compute_z(row):
                pt = row["pitch_type"]
                base = baseline_store.get(f"{pitcher_id}_{game_pk}_{pt}")
                if base and base.get("status") == "QUALIFIED" and not np.isnan(row.get(col_name, np.nan)):
                    mu = base["mu_vec"][col_idx]
                    sig = max(base["sigma_vec"][col_idx], 1e-4)
                    return (row[col_name] - mu) / sig
                return np.nan

            def compute_calib_delta(row):
                # Calibration deltas are only meaningful post-calibration (pitch 21+)
                if row["is_calibration_phase"]:
                    return np.nan
                pt = row["pitch_type"]
                if pt in calibrated_means and not np.isnan(row.get(col_name, np.nan)):
                    mu_c = calibrated_means[pt][col_idx]
                    return row[col_name] - mu_c
                return np.nan

            df[z_col] = df.apply(compute_z, axis=1).round(4)
            df[calib_col] = df.apply(compute_calib_delta, axis=1).round(4)

        return df, calibrated_means
