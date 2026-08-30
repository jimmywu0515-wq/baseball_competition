"""
Mahalanobis Distance Anomaly Scorer (§5 Step 4)
Calculates multi-dimensional anomaly scores and decomposes feature drift contributions.
"""
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

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

class MahalanobisScorer:
    """
    Computes calibrated and raw Mahalanobis distances and feature contribution breakdowns.
    """
    def __init__(self, ridge_reg: float = 1e-4):
        self.ridge_reg = ridge_reg

    def score_pitches(self, 
                      game_pitches_df: pd.DataFrame, 
                      pitcher_id: int, 
                      game_pk: int, 
                      baseline_store: Dict[str, Any],
                      calibrated_means: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Computes anomaly score per pitch.
        """
        df = game_pitches_df.copy()
        
        m_raw_list = []
        m_calib_list = []
        contrib_rel = []
        contrib_spin = []
        contrib_mov = []
        contrib_spd = []
        dominant_feature = []

        for _, row in df.iterrows():
            pt = row["pitch_type"]
            key = f"{pitcher_id}_{game_pk}_{pt}"
            base_info = baseline_store.get(key)

            if base_info is None or pt not in calibrated_means:
                # Default baseline fallback
                m_raw_list.append(1.0)
                m_calib_list.append(1.0)
                contrib_rel.append(25.0)
                contrib_spin.append(25.0)
                contrib_mov.append(25.0)
                contrib_spd.append(25.0)
                dominant_feature.append("Normal")
                continue

            x = np.array([row.get(c, 0.0) for c in FEATURE_COLS])
            # Handle NaNs in x
            if np.isnan(x).any():
                x = np.nan_to_num(x, nan=base_info["mu_vec"])

            mu_raw = base_info["mu_vec"]
            mu_calib = calibrated_means[pt]
            prec_mat = base_info["prec_mat"]

            # Compute Raw Mahalanobis
            diff_raw = x - mu_raw
            dist_raw = np.sqrt(np.maximum(0, np.dot(np.dot(diff_raw, prec_mat), diff_raw)))
            m_raw_list.append(round(float(dist_raw), 4))

            # Compute Calibrated Mahalanobis
            diff_calib = x - mu_calib
            # Element-wise contribution approximation: diff * (prec_mat @ diff)
            contrib_vector = np.abs(diff_calib * np.dot(prec_mat, diff_calib))
            dist_calib = np.sqrt(np.maximum(0, np.sum(contrib_vector)))
            m_calib_list.append(round(float(dist_calib), 4))

            # Feature Group Contribution Breakdown
            # Indices:
            # Release: 0 (rel_x), 1 (rel_z), 2 (ext)
            # Speed: 3 (rel_spd)
            # Spin: 4 (spin_rate), 5 (spin_axis)
            # Movement: 6 (pfx_x), 7 (pfx_z), 8 (vaa)
            total_contrib = np.sum(contrib_vector) + 1e-6
            rel_pct = (contrib_vector[0] + contrib_vector[1] + contrib_vector[2]) / total_contrib * 100.0
            spd_pct = contrib_vector[3] / total_contrib * 100.0
            spin_pct = (contrib_vector[4] + contrib_vector[5]) / total_contrib * 100.0
            mov_pct = (contrib_vector[6] + contrib_vector[7] + contrib_vector[8]) / total_contrib * 100.0

            contrib_rel.append(round(float(rel_pct), 1))
            contrib_spd.append(round(float(spd_pct), 1))
            contrib_spin.append(round(float(spin_pct), 1))
            contrib_mov.append(round(float(mov_pct), 1))

            # Dominant feature
            max_idx = np.argmax(contrib_vector)
            dominant_feature.append(FEATURE_COLS[max_idx])

        df["mahalanobis_raw"] = m_raw_list
        df["mahalanobis_calibrated"] = m_calib_list
        df["contrib_release_pct"] = contrib_rel
        df["contrib_speed_pct"] = contrib_spd
        df["contrib_spin_pct"] = contrib_spin
        df["contrib_movement_pct"] = contrib_mov
        df["dominant_drift_feature"] = dominant_feature

        return df
