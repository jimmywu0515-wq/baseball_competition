"""CUSUM change-point detection with strict score-availability isolation."""
from typing import Tuple

import numpy as np
import pandas as pd


class CUSUMDetector:
    """Apply a one-sided upper CUSUM only to pitches eligible for scoring."""

    def __init__(self, slack_k: float = 0.5, threshold_h: float = 4.0,
                 reference_mean: float = 1.0, reference_std: float = 0.5):
        self.slack_k = slack_k
        self.threshold_h = threshold_h
        self.reference_mean = reference_mean
        self.reference_std = reference_std

    def detect_game_alerts(self, game_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        order_col = "pitch_number_in_outing" if "pitch_number_in_outing" in game_df else "pitch_number_in_game"
        df = game_df.copy().sort_values(order_col).reset_index(drop=True)
        scores = pd.to_numeric(df["mahalanobis_calibrated"], errors="coerce").to_numpy()
        cusum_vals, alert_flags, alert_events = [], [], []
        statistic = 0.0

        for i, score in enumerate(scores):
            row = df.iloc[i]
            pitch_number = int(row.get("pitch_number_in_outing", row.get("pitch_number_in_game", 0)))
            eligible = (pitch_number > 20
                        and bool(row.get("score_available", np.isfinite(score)))
                        and np.isfinite(score))
            if not eligible:
                cusum_vals.append(np.nan)
                alert_flags.append(False)
                continue

            z_score = (score - self.reference_mean) / max(self.reference_std, 1e-6)
            statistic = max(0.0, statistic + z_score - self.slack_k)
            cusum_vals.append(round(statistic, 3))
            is_alert = statistic >= self.threshold_h
            alert_flags.append(is_alert)
            if is_alert:
                alert_events.append({
                    "game_pk": int(row["game_pk"]), "game_date": str(row["game_date"]),
                    "pitcher": int(row["pitcher"]),
                    "pitcher_name": str(row.get("pitcher_name", row["pitcher"])),
                    "alert_pitch_number": pitch_number, "alert_inning": int(row["inning"]),
                    "detector_type": "CUSUM", "cusum_statistic": round(statistic, 3),
                    "cusum_threshold": self.threshold_h,
                    "alert_level": "RED" if statistic >= self.threshold_h * 1.3 else "YELLOW",
                    "primary_drift_reason": str(row.get("dominant_drift_feature", "Mechanics Drift")),
                    "actual_data_source": row.get("actual_data_source", "unknown"),
                    "dataset_split": row.get("dataset_split", "unassigned"),
                })

        df["cusum_stat"] = cusum_vals
        df["is_cusum_alert"] = alert_flags
        return df, pd.DataFrame(alert_events)
