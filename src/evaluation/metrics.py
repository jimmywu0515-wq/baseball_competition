"""
Model Evaluation & Internally Consistent Verification Metrics (§5 Step 7 & Critique Fixes)
1. Single shared target: y_true_onset_in_horizon.
2. Lead Time computed ONLY for True Positive alerts.
3. Centralized False Alarm metrics: Alert-level precision, Episode recall, FAR per outing.
4. Relative Risk (Lift) properly distinguished from Odds Ratio.
"""
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)

class EvaluationEngine:
    """
    Computes rigorous statistical verification metrics comparing alerts vs subsequent collapse episodes.
    """
    def __init__(self, horizon_pitches: int = 15):
        self.horizon_pitches = horizon_pitches

    def evaluate_pipeline(self, 
                          scored_pitches_df: pd.DataFrame, 
                          alerts_df: pd.DataFrame,
                          collapse_episodes_df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Runs rigorous evaluation across all outings.
        Returns:
            metrics_summary: Dictionary of comprehensive metrics
            evaluated_df: DataFrame with verified true/false positive labels
        """
        df = scored_pitches_df.copy()

        # Exclude early calibration phase (pitches 1-20) and censored pitches from evaluation
        eval_mask = (df["pitch_number_in_outing"] > 20) & (~df.get("is_censored_followup", False))
        eval_df = df[eval_mask].copy()

        if eval_df.empty or "y_true_onset_in_horizon" not in eval_df.columns:
            logger.warning("Empty evaluation set or missing y_true_onset_in_horizon.")
            return {}, df

        y_true = eval_df["y_true_onset_in_horizon"].astype(int).values
        
        # Alert flag: CUSUM alert
        if "is_cusum_alert" in eval_df.columns:
            y_pred = eval_df["is_cusum_alert"].astype(int).values
        else:
            y_pred = (eval_df.get("mahalanobis_calibrated", 0.0) >= 2.5).astype(int).values

        # 1. Probabilities & Relative Risk (Lift)
        n_pos_pred = np.sum(y_pred == 1)
        n_neg_pred = np.sum(y_pred == 0)

        p_event_given_alert = np.mean(y_true[y_pred == 1]) if n_pos_pred > 0 else 0.0
        p_event_given_no_alert = np.mean(y_true[y_pred == 0]) if n_neg_pred > 0 else 1e-4

        # Relative Risk (often called Lift in business analytics)
        lift_relative_risk = p_event_given_alert / max(p_event_given_no_alert, 1e-4)

        # True Odds Ratio
        odds_alert = p_event_given_alert / max(1e-4, (1.0 - p_event_given_alert))
        odds_no_alert = p_event_given_no_alert / max(1e-4, (1.0 - p_event_given_no_alert))
        true_odds_ratio = odds_alert / max(1e-4, odds_no_alert)

        # 2. Precision, Recall, F1, PR-AUC, ROC-AUC
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        scores = eval_df["mahalanobis_calibrated"].values if "mahalanobis_calibrated" in eval_df.columns else y_pred
        precisions, recalls, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(recalls, precisions)
        roc_auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.5

        # 3. Internally Consistent Lead Time
        # Lead time is calculated ONLY for True Positive alerts (alert fires and episode onset occurs within horizon)
        tp_lead_times_pitches = []
        tp_lead_times_pas = []
        
        # Episode recall tracking
        episodes_detected = 0
        total_episodes = len(collapse_episodes_df) if collapse_episodes_df is not None else 0

        for (g_pk, pid), outing_group in eval_df.groupby(["game_pk", "pitcher"]):
            outing_alerts = outing_group[outing_group["is_cusum_alert"]].sort_values("pitch_number_in_outing")
            
            # Find episodes for this outing
            if collapse_episodes_df is not None and not collapse_episodes_df.empty:
                outing_eps = collapse_episodes_df[
                    (collapse_episodes_df["game_pk"] == g_pk) & 
                    (collapse_episodes_df["pitcher"] == pid)
                ]
            else:
                outing_eps = pd.DataFrame()

            for _, ep in outing_eps.iterrows():
                ep_onset_pitch = ep["onset_pitch"]
                ep_onset_pa = ep["onset_pa"]
                
                # Check for alert preceding this onset within horizon [onset - H, onset)
                valid_preceding_alerts = outing_alerts[
                    (outing_alerts["pitch_number_in_outing"] < ep_onset_pitch) &
                    (outing_alerts["pitch_number_in_outing"] >= ep_onset_pitch - self.horizon_pitches)
                ]

                if not valid_preceding_alerts.empty:
                    episodes_detected += 1
                    first_valid_alert = valid_preceding_alerts.iloc[0]
                    lead_p = ep_onset_pitch - first_valid_alert["pitch_number_in_outing"]
                    lead_pa = max(0, ep_onset_pa - first_valid_alert.get("pa_number_in_outing", ep_onset_pa))
                    tp_lead_times_pitches.append(lead_p)
                    tp_lead_times_pas.append(lead_pa)

        lead_pitches_arr = np.array(tp_lead_times_pitches) if tp_lead_times_pitches else np.array([0.0])
        lead_pas_arr = np.array(tp_lead_times_pas) if tp_lead_times_pas else np.array([0.0])
        episode_recall = episodes_detected / max(1, total_episodes)

        # 4. Rigorous False Alarm Metrics
        # Clean outings (outings with zero collapse episodes)
        clean_outings = 0
        clean_outings_with_alert = 0
        total_false_alarm_pitches = 0

        for (g_pk, pid), outing_group in eval_df.groupby(["game_pk", "pitcher"]):
            has_collapse = outing_group["y_true_onset_in_horizon"].any()
            alert_fired = outing_group["is_cusum_alert"].any()
            
            if not has_collapse:
                clean_outings += 1
                if alert_fired:
                    clean_outings_with_alert += 1
            
            total_false_alarm_pitches += np.sum((outing_group["is_cusum_alert"] == 1) & (outing_group["y_true_onset_in_horizon"] == 0))

        outing_far = clean_outings_with_alert / max(1, clean_outings)
        false_alarms_per_outing = total_false_alarm_pitches / max(1, eval_df.groupby(["game_pk", "pitcher"]).ngroups)

        metrics_summary = {
            "evaluated_outings_count": int(eval_df.groupby(["game_pk", "pitcher"]).ngroups),
            "evaluated_pitches_count": len(eval_df),
            "total_collapse_episodes": total_episodes,
            "collapse_episodes_detected": episodes_detected,
            "episode_recall": round(float(episode_recall), 3),
            "lift_relative_risk": round(float(lift_relative_risk), 2),
            "odds_ratio": round(float(true_odds_ratio), 2),
            "pitch_precision": round(float(precision), 3),
            "pitch_recall": round(float(recall), 3),
            "f1_score": round(float(f1), 3),
            "pr_auc": round(float(pr_auc), 3),
            "roc_auc": round(float(roc_auc), 3),
            "lead_time_mean_pitches": round(float(np.mean(lead_pitches_arr)), 1),
            "lead_time_median_pitches": round(float(np.median(lead_pitches_arr)), 1),
            "lead_time_mean_pas": round(float(np.mean(lead_pas_arr)), 1),
            "lead_time_median_pas": round(float(np.median(lead_pas_arr)), 1),
            "outing_false_alarm_rate": round(float(outing_far), 3),
            "false_alarms_per_outing": round(float(false_alarms_per_outing), 1)
        }

        # Keep y_true_onset_in_horizon attached to df
        df["future_collapse_in_horizon"] = df["y_true_onset_in_horizon"]

        logger.info(f"Rigorous Evaluation | Lift: {lift_relative_risk:.2f}x | Episode Recall: {episode_recall*100:.1f}% | Mean Lead Time: {metrics_summary['lead_time_mean_pitches']} pitches | Outing FAR: {outing_far*100:.1f}%")
        return metrics_summary, df
