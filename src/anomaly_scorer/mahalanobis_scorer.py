"""Vectorized, exact Mahalanobis pitch anomaly scoring."""
from typing import Any, Dict

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "release_pos_x", "release_pos_z", "release_extension", "release_speed",
    "release_spin_rate", "spin_axis_cos", "spin_axis_sin", "pfx_x", "pfx_z", "vaa",
]
IDX_RELEASE = [0, 1, 2]
IDX_SPEED = [3]
IDX_SPIN = [4, 5, 6]
IDX_MOVEMENT = [7, 8, 9]


class MahalanobisScorer:
    """Compute exact distances and diagonal feature-group contributions."""

    def __init__(self, ridge_reg: float = 1e-4):
        self.ridge_reg = ridge_reg

    def score_pitches(
        self, outing_df: pd.DataFrame, pitcher_id: int, game_pk: int,
        baseline_store: Dict[str, Any], calibrated_means: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        df = outing_df.copy()
        n_rows = len(df)
        m_raw = np.full(n_rows, np.nan)
        m_calib = np.full(n_rows, np.nan)
        contributions = [np.full(n_rows, np.nan) for _ in range(4)]
        dominant = np.full(n_rows, "Unavailable: insufficient history", dtype=object)
        statuses = np.full(n_rows, "INSUFFICIENT_HISTORY", dtype=object)

        calibration = df.get(
            "is_calibration_phase", pd.Series(False, index=df.index)
        ).fillna(False).astype(bool).to_numpy()
        dominant[calibration] = "Unavailable: calibration phase"
        statuses[calibration] = "CALIBRATION_PHASE"
        pitch_types = df["pitch_type"].to_numpy()

        for pitch_type in pd.unique(df["pitch_type"].dropna()):
            positions = np.flatnonzero((pitch_types == pitch_type) & ~calibration)
            if not len(positions):
                continue
            baseline = baseline_store.get(f"{pitcher_id}_{game_pk}_{pitch_type}")
            if (baseline is None or baseline.get("status") != "QUALIFIED"
                    or pitch_type not in calibrated_means):
                status = (
                    baseline.get("status", "INSUFFICIENT_HISTORY")
                    if baseline is not None else "INSUFFICIENT_HISTORY"
                )
                if pitch_type not in calibrated_means and status == "QUALIFIED":
                    status = "INSUFFICIENT_CALIBRATION_PITCHES"
                statuses[positions] = status
                dominant[positions] = f"Unavailable: {status.lower()}"
                continue

            values = df.iloc[positions][FEATURE_COLS].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(float)
            complete = np.isfinite(values).all(axis=1)
            missing_positions = positions[~complete]
            statuses[missing_positions] = "MISSING_FEATURES"
            dominant[missing_positions] = "Unavailable: missing features"
            if not complete.any():
                continue

            valid_positions = positions[complete]
            x = values[complete]
            raw_difference = x - np.asarray(baseline["mu_vec"], dtype=float)
            calibrated_difference = x - np.asarray(calibrated_means[pitch_type], dtype=float)
            precision = np.asarray(baseline["prec_mat"], dtype=float)
            raw_squared = np.einsum("ij,jk,ik->i", raw_difference, precision, raw_difference)
            calibrated_squared = np.einsum(
                "ij,jk,ik->i", calibrated_difference, precision, calibrated_difference
            )
            m_raw[valid_positions] = np.round(np.sqrt(np.maximum(0.0, raw_squared)), 4)
            m_calib[valid_positions] = np.round(np.sqrt(np.maximum(0.0, calibrated_squared)), 4)

            diagonal_quadratic = (calibrated_difference ** 2) * np.diag(precision)
            total = diagonal_quadratic.sum(axis=1) + 1e-6
            for result, indices in zip(
                contributions, (IDX_RELEASE, IDX_SPEED, IDX_SPIN, IDX_MOVEMENT)
            ):
                result[valid_positions] = np.round(
                    diagonal_quadratic[:, indices].sum(axis=1) / total * 100.0, 1
                )
            dominant[valid_positions] = np.asarray(FEATURE_COLS, dtype=object)[
                np.argmax(diagonal_quadratic, axis=1)
            ]
            statuses[valid_positions] = "AVAILABLE"

        df["mahalanobis_raw"] = m_raw
        df["mahalanobis_calibrated"] = m_calib
        df["contrib_release_pct"] = contributions[0]
        df["contrib_speed_pct"] = contributions[1]
        df["contrib_spin_pct"] = contributions[2]
        df["contrib_movement_pct"] = contributions[3]
        df["dominant_drift_feature"] = dominant
        df["score_status"] = statuses
        df["score_available"] = df["score_status"].eq("AVAILABLE")
        return df
