"""
EWMA (Exponentially Weighted Moving Average) Control Chart Detector
"""
import logging
from typing import Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class EWMADetector:
    """
    Applies EWMA smoothing and dynamic upper control limits on anomaly scores.
    """
    def __init__(self, ewma_lambda: float = 0.20, l_sigma: float = 2.8):
        self.ewma_lambda = ewma_lambda
        self.l_sigma = l_sigma

    def detect_game_alerts(self, game_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = game_df.copy().sort_values("pitch_number_in_game").reset_index(drop=True)
        scores = df["mahalanobis_calibrated"].values

        early_scores = scores[:20] if len(scores) >= 20 else scores
        mu_0 = np.mean(early_scores) if len(early_scores) > 0 else 1.0
        sigma_0 = np.std(early_scores) if len(early_scores) > 0 and np.std(early_scores) > 0.1 else 0.5

        ewma_vals = []
        ucl_vals = []
        alert_flags = []
        alert_events = []

        z = mu_0
        lam = self.ewma_lambda

        for i, score in enumerate(scores):
            # Update EWMA
            z = lam * score + (1.0 - lam) * z
            # Dynamic control limit
            # sigma_z = sigma_0 * sqrt( (lambda / (2 - lambda)) * (1 - (1 - lambda)^(2*(i+1))) )
            var_factor = (lam / (2.0 - lam)) * (1.0 - (1.0 - lam)**(2 * (i + 1)))
            ucl = mu_0 + self.l_sigma * sigma_0 * np.sqrt(max(1e-6, var_factor))

            ewma_vals.append(round(z, 3))
            ucl_vals.append(round(ucl, 3))

            is_alert = (z >= ucl)
            alert_flags.append(is_alert)

            if is_alert:
                row = df.iloc[i]
                alert_events.append({
                    "game_pk": int(row["game_pk"]),
                    "game_date": str(row["game_date"]),
                    "pitcher": int(row["pitcher"]),
                    "pitcher_name": str(row["pitcher_name"]),
                    "alert_pitch_number": int(row["pitch_number_in_game"]),
                    "alert_inning": int(row["inning"]),
                    "detector_type": "EWMA",
                    "ewma_statistic": round(z, 3),
                    "ewma_upper_limit": round(ucl, 3),
                    "alert_level": "RED" if z >= (ucl * 1.2) else "YELLOW",
                    "primary_drift_reason": str(row.get("dominant_drift_feature", "Mechanics Drift"))
                })

        df["ewma_stat"] = ewma_vals
        df["ewma_ucl"] = ucl_vals
        df["is_ewma_alert"] = alert_flags

        return df, pd.DataFrame(alert_events)
