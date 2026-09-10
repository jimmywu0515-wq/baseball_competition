"""EWMA change-point detection with strict score-availability isolation."""
from typing import Tuple

import numpy as np
import pandas as pd


class EWMADetector:
    def __init__(self, ewma_lambda: float = 0.20, l_sigma: float = 2.8,
                 reference_mean: float = 1.0, reference_std: float = 0.5):
        self.ewma_lambda = ewma_lambda
        self.l_sigma = l_sigma
        self.reference_mean = reference_mean
        self.reference_std = reference_std

    def detect_game_alerts(self, game_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        order_col = "pitch_number_in_outing" if "pitch_number_in_outing" in game_df else "pitch_number_in_game"
        df = game_df.copy().sort_values(order_col).reset_index(drop=True)
        scores = pd.to_numeric(df["mahalanobis_calibrated"], errors="coerce").to_numpy()
        ewma_vals, ucl_vals, alert_flags, alert_events = [], [], [], []
        statistic, update_count = self.reference_mean, 0

        for i, score in enumerate(scores):
            row = df.iloc[i]
            pitch_number = int(row.get("pitch_number_in_outing", row.get("pitch_number_in_game", 0)))
            eligible = (pitch_number > 20
                        and bool(row.get("score_available", np.isfinite(score)))
                        and np.isfinite(score))
            if not eligible:
                ewma_vals.append(np.nan)
                ucl_vals.append(np.nan)
                alert_flags.append(False)
                continue

            update_count += 1
            lam = self.ewma_lambda
            statistic = lam * score + (1.0 - lam) * statistic
            variance_factor = (lam / (2.0 - lam)) * (1.0 - (1.0 - lam) ** (2 * update_count))
            ucl = self.reference_mean + self.l_sigma * self.reference_std * np.sqrt(max(1e-6, variance_factor))
            ewma_vals.append(round(statistic, 3))
            ucl_vals.append(round(ucl, 3))
            is_alert = statistic >= ucl
            alert_flags.append(is_alert)
            if is_alert:
                alert_events.append({
                    "game_pk": int(row["game_pk"]), "game_date": str(row["game_date"]),
                    "pitcher": int(row["pitcher"]),
                    "pitcher_name": str(row.get("pitcher_name", row["pitcher"])),
                    "alert_pitch_number": pitch_number, "alert_inning": int(row["inning"]),
                    "detector_type": "EWMA", "ewma_statistic": round(statistic, 3),
                    "ewma_upper_limit": round(ucl, 3),
                    "actual_data_source": row.get("actual_data_source", "unknown"),
                    "dataset_split": row.get("dataset_split", "unassigned"),
                })

        df["ewma_stat"] = ewma_vals
        df["ewma_ucl"] = ucl_vals
        df["is_ewma_alert"] = alert_flags
        return df, pd.DataFrame(alert_events)
