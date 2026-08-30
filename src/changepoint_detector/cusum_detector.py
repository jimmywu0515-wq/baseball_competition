"""
CUSUM (Cumulative Sum) Control Chart Detector (§5 Step 5)
Converts pitch-by-pitch anomaly scores into robust trend shift alarms.
"""
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CUSUMDetector:
    """
    Applies one-sided upper CUSUM on anomaly scores.
    S_0 = 0
    S_i = max(0, S_{i-1} + (anomaly_score_i - mu_0) / sigma_0 - k)
    Alert triggered when S_i >= h.
    """
    def __init__(self, slack_k: float = 0.5, threshold_h: float = 4.0):
        self.slack_k = slack_k
        self.threshold_h = threshold_h

    def detect_game_alerts(self, game_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes a single game's pitches and produces:
        1. Updated pitches dataframe with cusum statistics
        2. Alert events dataframe
        """
        df = game_df.copy().sort_values("pitch_number_in_game").reset_index(drop=True)
        
        scores = df["mahalanobis_calibrated"].values
        # Baseline reference: early game pitches (pitches 1 to 20) expected mean ≈ 1.0, std ≈ 0.5
        early_scores = scores[:20] if len(scores) >= 20 else scores
        mu_0 = np.mean(early_scores) if len(early_scores) > 0 else 1.0
        sigma_0 = np.std(early_scores) if len(early_scores) > 0 and np.std(early_scores) > 0.1 else 0.5

        cusum_vals = []
        alert_flags = []
        alert_events = []
        
        s = 0.0
        has_alerted = False

        for i, score in enumerate(scores):
            # Standardize
            z = (score - mu_0) / sigma_0
            s = max(0.0, s + (z - self.slack_k))
            cusum_vals.append(round(s, 3))

            is_alert = (s >= self.threshold_h)
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
                    "detector_type": "CUSUM",
                    "cusum_statistic": round(s, 3),
                    "cusum_threshold": self.threshold_h,
                    "alert_level": "RED" if s >= (self.threshold_h * 1.3) else "YELLOW",
                    "primary_drift_reason": str(row.get("dominant_drift_feature", "Mechanics Drift"))
                })

        df["cusum_stat"] = cusum_vals
        df["is_cusum_alert"] = alert_flags

        alerts_df = pd.DataFrame(alert_events)
        return df, alerts_df
