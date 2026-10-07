"""CUSUM change-point detection with strict score-availability isolation."""
from typing import Tuple
from collections.abc import Mapping

import numpy as np
import pandas as pd


def resolve_reference_std(reference_std, pitcher: int) -> float:
    """Resolve an explicit pitcher mapping; valid scalar behavior is unchanged.

    A missing mapping entry raises rather than silently borrowing another
    pitcher's scale. Callers supply explicit global fallbacks where justified.
    """
    if isinstance(reference_std, pd.Series):
        if not reference_std.index.is_unique:
            raise ValueError("Pitcher reference scales require a unique index")
        if pitcher not in reference_std.index:
            raise ValueError(f"No reference standard deviation for pitcher {pitcher}")
        value = float(reference_std.loc[pitcher])
    elif isinstance(reference_std, Mapping):
        if pitcher not in reference_std:
            raise ValueError(f"No reference standard deviation for pitcher {pitcher}")
        value = float(reference_std[pitcher])
    else:
        value = float(reference_std)
        if not np.isfinite(value):
            raise ValueError("Reference standard deviation must be finite")
        return max(value, 1e-6)  # Preserve the legacy scalar clamp.
    if not np.isfinite(value) or value <= 0:
        raise ValueError("Pitcher reference standard deviations must be positive and finite")
    return max(value, 1e-6)


class CUSUMDetector:
    """Apply a one-sided upper CUSUM only to pitches eligible for scoring."""

    def __init__(self, slack_k: float = 0.5, threshold_h: float = 4.0,
                 reference_mean: float = 1.0, reference_std: float | Mapping | pd.Series = 0.5,
                 calibration_pitches: int = 20):
        self.slack_k = slack_k
        self.threshold_h = threshold_h
        self.reference_mean = reference_mean
        self.reference_std = reference_std
        self.calibration_pitches = calibration_pitches

    def detect_game_alerts(self, game_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        order_col = "pitch_number_in_outing" if "pitch_number_in_outing" in game_df else "pitch_number_in_game"
        df = game_df.copy().sort_values(order_col).reset_index(drop=True)
        scores = pd.to_numeric(df["mahalanobis_calibrated"], errors="coerce").to_numpy()
        cusum_vals, alert_flags, alert_events = [], [], []
        statistic = 0.0

        for i, score in enumerate(scores):
            row = df.iloc[i]
            pitch_number = int(row.get("pitch_number_in_outing", row.get("pitch_number_in_game", 0)))
            past_calibration = not bool(
                row.get("is_calibration_phase", pitch_number <= self.calibration_pitches)
            )
            eligible = (past_calibration
                        and bool(row.get("score_available", np.isfinite(score)))
                        and np.isfinite(score))
            if not eligible:
                cusum_vals.append(np.nan)
                alert_flags.append(False)
                continue

            z_score = (score - self.reference_mean) / resolve_reference_std(self.reference_std, int(row["pitcher"]))
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
