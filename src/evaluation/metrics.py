"""
Model Evaluation & Causal Lead-Time Verification Metrics (§5 Step 7 & §6)
Computes Lift, Lead Time distribution, PR-AUC, Precision/Recall, and False Alarm Rates.
"""
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

class EvaluationEngine:
    """
    Computes statistical verification metrics comparing alerts vs subsequent collapse.
    """
    def __init__(self, horizon_pitches: int = 15):
        self.horizon_pitches = horizon_pitches

    def evaluate_pipeline(self, 
                          scored_pitches_df: pd.DataFrame, 
                          alerts_df: pd.DataFrame,
                          collapse_events_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation across all games.
        """
        df = scored_pitches_df.copy()
        
        # 1. Forward-looking Horizon Label:
        # For each pitch, does a collapse occur in [pitch_num, pitch_num + horizon_pitches]?
        df["future_collapse_in_horizon"] = False
        
        for g_pk, g_df in df.groupby("game_pk"):
            collapse_pitch_indices = g_df[g_df["is_collapse_event"]]["pitch_number_in_game"].values
            if len(collapse_pitch_indices) == 0:
                continue

            for idx, row in g_df.iterrows():
                p_num = row["pitch_number_in_game"]
                # Check if any collapse pitch falls within (p_num, p_num + horizon_pitches]
                in_horizon = any((c_p > p_num) and (c_p <= p_num + self.horizon_pitches) for c_p in collapse_pitch_indices)
                df.loc[idx, "future_collapse_in_horizon"] = in_horizon

        # 2. Lift / Odds Ratio
        y_true = df["future_collapse_in_horizon"].astype(int).values
        y_pred = df["is_cusum_alert"].astype(int).values
        scores = df["mahalanobis_calibrated"].values

        p_collapse_given_alert = np.mean(y_true[y_pred == 1]) if np.sum(y_pred == 1) > 0 else 0.0
        p_collapse_given_no_alert = np.mean(y_true[y_pred == 0]) if np.sum(y_pred == 0) > 0 else 1e-4
        lift = p_collapse_given_alert / max(p_collapse_given_no_alert, 1e-4)

        # 3. Precision, Recall, F1, PR-AUC
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        precisions, recalls, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(recalls, precisions)

        # 4. Lead Time Distribution (in pitches)
        lead_times = []
        for g_pk, g_df in df.groupby("game_pk"):
            alert_pitches = g_df[g_df["is_cusum_alert"]]["pitch_number_in_game"].values
            collapse_pitches = g_df[g_df["is_collapse_event"]]["pitch_number_in_game"].values
            
            if len(alert_pitches) > 0 and len(collapse_pitches) > 0:
                first_alert = alert_pitches[0]
                # First collapse that occurred AFTER the alert
                future_collapses = collapse_pitches[collapse_pitches >= first_alert]
                if len(future_collapses) > 0:
                    lead_time = future_collapses[0] - first_alert
                    lead_times.append(lead_time)

        lead_time_arr = np.array(lead_times) if lead_times else np.array([0.0])
        
        # 5. False Alarm Rate (FAR per start)
        # Fraction of starts with no collapse that had an alert
        starts_no_collapse = []
        starts_with_false_alert = []
        for g_pk, g_df in df.groupby("game_pk"):
            has_collapse = g_df["is_collapse_event"].any()
            has_alert = g_df["is_cusum_alert"].any()
            if not has_collapse:
                starts_no_collapse.append(g_pk)
                if has_alert:
                    starts_with_false_alert.append(g_pk)

        far_per_start = len(starts_with_false_alert) / max(1, len(starts_no_collapse))

        metrics_summary = {
            "sample_games_count": int(df["game_pk"].nunique()),
            "total_pitches_analyzed": len(df),
            "total_collapse_events": int(np.sum(df["is_collapse_event"])),
            "lift_odds_ratio": round(float(lift), 2),
            "p_collapse_given_alert": round(float(p_collapse_given_alert), 3),
            "p_collapse_given_no_alert": round(float(p_collapse_given_no_alert), 3),
            "pr_auc": round(float(pr_auc), 3),
            "precision": round(float(precision), 3),
            "recall": round(float(recall), 3),
            "f1_score": round(float(f1), 3),
            "lead_time_mean_pitches": round(float(np.mean(lead_time_arr)), 1),
            "lead_time_median_pitches": round(float(np.median(lead_time_arr)), 1),
            "lead_time_25pct_pitches": round(float(np.percentile(lead_time_arr, 25)), 1),
            "lead_time_75pct_pitches": round(float(np.percentile(lead_time_arr, 75)), 1),
            "false_alarm_rate_per_start": round(float(far_per_start), 3)
        }

        logger.info(f"Evaluation Complete | Lift: {lift:.2f}x | Mean Lead Time: {metrics_summary['lead_time_mean_pitches']} pitches | PR-AUC: {pr_auc:.3f}")
        return metrics_summary
