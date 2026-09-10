"""Evaluation wrapper built on the shared warning/episode matching protocol."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

from src.evaluation.protocol import eligible_pitch_mask, evaluate_warning_predictions

logger = logging.getLogger(__name__)


class EvaluationEngine:
    def __init__(self, horizon_pitches: int = 15):
        self.horizon_pitches = horizon_pitches

    def evaluate_pipeline(
        self,
        scored_pitches_df: pd.DataFrame,
        alerts_df: Optional[pd.DataFrame],
        collapse_episodes_df: Optional[pd.DataFrame],
        alert_col: Optional[str] = None,
        evaluation_split: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        df = scored_pitches_df.copy()
        if alert_col is None:
            alert_col = (
                "is_proposed_operating_alert"
                if "is_proposed_operating_alert" in df
                else "is_cusum_alert"
            )
        if alert_col not in df:
            raise ValueError(f"Missing alert column: {alert_col}")

        metrics, warnings, matches = evaluate_warning_predictions(
            df, collapse_episodes_df, df[alert_col],
            horizon_pitches=self.horizon_pitches, split=evaluation_split
        )
        mask = eligible_pitch_mask(df, evaluation_split)
        eval_df = df.loc[mask].copy()
        y_true = eval_df["y_true_onset_in_horizon"].astype(int).to_numpy()
        predictions = eval_df[alert_col].fillna(False).astype(bool).to_numpy()
        p_alert = float(y_true[predictions].mean()) if predictions.any() else 0.0
        p_no_alert = float(y_true[~predictions].mean()) if (~predictions).any() else np.nan
        risk_ratio = p_alert / p_no_alert if np.isfinite(p_no_alert) and p_no_alert > 0 else np.nan

        score_col = "cusum_stat" if "cusum_stat" in eval_df else "mahalanobis_calibrated"
        finite = eval_df[score_col].replace([np.inf, -np.inf], np.nan).notna()
        if finite.any() and eval_df.loc[finite, "y_true_onset_in_horizon"].nunique() > 1:
            precision, recall, _ = precision_recall_curve(
                eval_df.loc[finite, "y_true_onset_in_horizon"].astype(int), eval_df.loc[finite, score_col]
            )
            pr_auc = float(auc(recall, precision))
        else:
            pr_auc = np.nan

        metrics.update({
            "risk_ratio_alert_vs_no_alert": risk_ratio,
            "pitch_pr_auc": pr_auc,
            "evaluation_split": evaluation_split or "all",
            "actual_data_sources": sorted(map(str, eval_df["actual_data_source"].dropna().unique()))
            if "actual_data_source" in eval_df else ["unknown"],
            # Compatibility aliases, now explicitly warning/event based.
            "episode_recall": metrics["episode_recall"],
            "outing_false_alarm_rate": metrics["clean_outing_false_alarm_rate"],
            "false_alarms_per_outing": metrics["false_warnings_per_outing"],
            "lift_relative_risk": risk_ratio,
            "pr_auc": pr_auc,
        })
        eval_df["future_collapse_in_horizon"] = eval_df["y_true_onset_in_horizon"]
        logger.info(
            "Evaluation split=%s | episode recall %.1f%% | warning precision %.1f%% | false warnings/outing %.3f",
            metrics["evaluation_split"], 100 * metrics["episode_recall"],
            100 * metrics["warning_precision"], metrics["false_warnings_per_outing"],
        )
        return metrics, eval_df
