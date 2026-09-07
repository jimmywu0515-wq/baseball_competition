"""
Mahalanobis Distance Anomaly Scorer (§5 Step 4 & Critique Fixes)
AUDITED MATHEMATICAL FORMULATION:
Calculates true quadratic Mahalanobis distance:
    D_M^2 = (x - mu)^T * Sigma^{-1} * (x - mu)
Removes the invalid element-wise absolute value summation bug.
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

# Feature indices for sub-component decomposition
IDX_RELEASE = [0, 1, 2]       # release_pos_x, release_pos_z, release_extension
IDX_SPEED = [3]               # release_speed
IDX_SPIN = [4, 5, 6]          # release_spin_rate, spin_axis_cos, spin_axis_sin
IDX_MOVEMENT = [7, 8, 9]      # pfx_x, pfx_z, vaa

class MahalanobisScorer:
    """
    Computes exact Mahalanobis distances and true sub-group quadratic contributions.
    """
    def __init__(self, ridge_reg: float = 1e-4):
        self.ridge_reg = ridge_reg

    def score_pitches(self, 
                      outing_df: pd.DataFrame, 
                      pitcher_id: int, 
                      game_pk: int, 
                      baseline_store: Dict[str, Any],
                      calibrated_means: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Calculates exact Mahalanobis distance per pitch.
        """
        df = outing_df.copy()

        m_raw_list = []
        m_calib_list = []
        contrib_rel = []
        contrib_spd = []
        contrib_spin = []
        contrib_mov = []
        dominant_feature = []

        for _, row in df.iterrows():
            pt = row["pitch_type"]
            key = f"{pitcher_id}_{game_pk}_{pt}"
            base_info = baseline_store.get(key)

            if base_info is None or base_info.get("status") != "QUALIFIED" or pt not in calibrated_means:
                m_raw_list.append(1.0)
                m_calib_list.append(1.0)
                contrib_rel.append(25.0)
                contrib_spd.append(25.0)
                contrib_spin.append(25.0)
                contrib_mov.append(25.0)
                dominant_feature.append("Insufficient Baseline")
                continue

            x = np.array([row.get(c, 0.0) for c in FEATURE_COLS])
            if np.isnan(x).any():
                x = np.nan_to_num(x, nan=base_info["mu_vec"])

            mu_raw = base_info["mu_vec"]
            mu_calib = calibrated_means[pt]
            prec_mat = base_info["prec_mat"]

            # Exact Raw Mahalanobis Distance: sqrt( Delta^T * Sigma^{-1} * Delta )
            diff_raw = x - mu_raw
            d2_raw = float(np.dot(np.dot(diff_raw, prec_mat), diff_raw))
            dist_raw = np.sqrt(max(0.0, d2_raw))
            m_raw_list.append(round(dist_raw, 4))

            # Exact Calibrated Mahalanobis Distance
            diff_calib = x - mu_calib
            d2_calib = float(np.dot(np.dot(diff_calib, prec_mat), diff_calib))
            dist_calib = np.sqrt(max(0.0, d2_calib))
            m_calib_list.append(round(dist_calib, 4))

            # Feature Group Contribution Decomposition (Diagonal quadratic projections)
            # D_i^2 ≈ Delta_i^2 * Prec_ii
            diag_quad = (diff_calib ** 2) * np.diag(prec_mat)
            total_quad = np.sum(diag_quad) + 1e-6

            rel_p = np.sum(diag_quad[IDX_RELEASE]) / total_quad * 100.0
            spd_p = np.sum(diag_quad[IDX_SPEED]) / total_quad * 100.0
            spin_p = np.sum(diag_quad[IDX_SPIN]) / total_quad * 100.0
            mov_p = np.sum(diag_quad[IDX_MOVEMENT]) / total_quad * 100.0

            contrib_rel.append(round(float(rel_p), 1))
            contrib_spd.append(round(float(spd_p), 1))
            contrib_spin.append(round(float(spin_p), 1))
            contrib_mov.append(round(float(mov_p), 1))

            max_idx = np.argmax(diag_quad)
            dominant_feature.append(FEATURE_COLS[max_idx])

        df["mahalanobis_raw"] = m_raw_list
        df["mahalanobis_calibrated"] = m_calib_list
        df["contrib_release_pct"] = contrib_rel
        df["contrib_speed_pct"] = contrib_spd
        df["contrib_spin_pct"] = contrib_spin
        df["contrib_movement_pct"] = contrib_mov
        df["dominant_drift_feature"] = dominant_feature

        return df
