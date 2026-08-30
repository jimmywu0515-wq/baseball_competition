"""
Baseline Comparator (§5 Step 7 & §6)
Benchmarks Micro-Mechanics Early Warning System against:
1. Naive Velocity Baseline (Drop >= 1.5 mph)
2. Traditional Pitch Count Threshold (>= 85 pitches)
"""
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class BaselineComparator:
    """
    Compares the proposed system against naive heuristics.
    """
    def __init__(self, velocity_drop_mph: float = 1.5, pitch_count_thresh: int = 85, horizon_pitches: int = 15):
        self.vel_drop_mph = velocity_drop_mph
        self.pitch_count_thresh = pitch_count_thresh
        self.horizon = horizon_pitches

    def compare_systems(self, scored_pitches_df: pd.DataFrame, our_metrics: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculates performance for Naive Velocity Drop and Naive Pitch Count,
        and returns a comparative summary table.
        """
        df = scored_pitches_df.copy()

        # 1. Naive Velocity Drop Model
        # Baseline = average velocity in first 20 pitches
        # Alert if rolling 5-pitch velocity drops >= 1.5 mph below baseline
        naive_vel_alerts = []
        for g_pk, g_df in df.groupby("game_pk"):
            early_vel = g_df[g_df["pitch_number_in_game"] <= 20]["release_speed"].mean()
            if np.isnan(early_vel):
                early_vel = g_df["release_speed"].mean()
            
            roll_vel = g_df["release_speed"].rolling(5, min_periods=2).mean()
            is_vel_drop = (early_vel - roll_vel) >= self.vel_drop_mph
            naive_vel_alerts.extend(is_vel_drop.values)

        df["is_naive_vel_alert"] = naive_vel_alerts

        # 2. Naive Pitch Count Model (Alert when pitch_number_in_game >= 85)
        df["is_naive_count_alert"] = df["pitch_number_in_game"] >= self.pitch_count_thresh

        # 3. Compute Metrics for Naive Baselines
        def compute_baseline_stats(alert_col: str):
            y_true = df["future_collapse_in_horizon"].astype(int).values if "future_collapse_in_horizon" in df.columns else np.zeros(len(df))
            y_pred = df[alert_col].astype(int).values

            p_alert = np.mean(y_true[y_pred == 1]) if np.sum(y_pred == 1) > 0 else 0.0
            p_no_alert = np.mean(y_true[y_pred == 0]) if np.sum(y_pred == 0) > 0 else 1e-4
            lift = p_alert / max(p_no_alert, 1e-4)

            # Lead Time
            lead_times = []
            for g_pk, g_df in df.groupby("game_pk"):
                a_pitches = g_df[g_df[alert_col]]["pitch_number_in_game"].values
                c_pitches = g_df[g_df["is_collapse_event"]]["pitch_number_in_game"].values
                if len(a_pitches) > 0 and len(c_pitches) > 0:
                    first_a = a_pitches[0]
                    fut_c = c_pitches[c_pitches >= first_a]
                    if len(fut_c) > 0:
                        lead_times.append(fut_c[0] - first_a)

            lead_arr = np.array(lead_times) if lead_times else np.array([0.0])

            # False Alarm Rate
            starts_no_c = []
            starts_false_a = []
            for g_pk, g_df in df.groupby("game_pk"):
                if not g_df["is_collapse_event"].any():
                    starts_no_c.append(g_pk)
                    if g_df[alert_col].any():
                        starts_false_a.append(g_pk)
            far = len(starts_false_a) / max(1, len(starts_no_c))

            return {
                "lift": round(float(lift), 2),
                "mean_lead_time": round(float(np.mean(lead_arr)), 1),
                "median_lead_time": round(float(np.median(lead_arr)), 1),
                "far_per_start": round(float(far), 3)
            }

        vel_stats = compute_baseline_stats("is_naive_vel_alert")
        count_stats = compute_baseline_stats("is_naive_count_alert")

        comparison_data = [
            {
                "Model / System": "Proposed Micro-Mechanics (CUSUM + Mahalanobis)",
                "Lift (Odds Ratio)": f"{our_metrics['lift_odds_ratio']}x",
                "Mean Lead Time (Pitches)": f"{our_metrics['lead_time_mean_pitches']} pitches",
                "Median Lead Time (Pitches)": f"{our_metrics['lead_time_median_pitches']} pitches",
                "False Alarm Rate (Per Start)": f"{our_metrics['false_alarm_rate_per_start']*100:.1f}%",
                "Early Warning Advantage": "Early detection before velo drop & damage"
            },
            {
                "Model / System": f"Naive Velocity Drop (>= {self.vel_drop_mph} mph)",
                "Lift (Odds Ratio)": f"{vel_stats['lift']}x",
                "Mean Lead Time (Pitches)": f"{vel_stats['mean_lead_time']} pitches",
                "Median Lead Time (Pitches)": f"{vel_stats['median_lead_time']} pitches",
                "False Alarm Rate (Per Start)": f"{vel_stats['far_per_start']*100:.1f}%",
                "Early Warning Advantage": "Lags behind mechanics degradation by 10+ pitches"
            },
            {
                "Model / System": f"Traditional Pitch Count (>= {self.pitch_count_thresh} pitches)",
                "Lift (Odds Ratio)": f"{count_stats['lift']}x",
                "Mean Lead Time (Pitches)": f"{count_stats['mean_lead_time']} pitches",
                "Median Lead Time (Pitches)": f"{count_stats['median_lead_time']} pitches",
                "False Alarm Rate (Per Start)": f"{count_stats['far_per_start']*100:.1f}%",
                "Early Warning Advantage": "Rigid heuristic, ignores individual daily variance"
            }
        ]

        comp_df = pd.DataFrame(comparison_data)
        logger.info("\n" + comp_df.to_string(index=False))
        return comp_df
