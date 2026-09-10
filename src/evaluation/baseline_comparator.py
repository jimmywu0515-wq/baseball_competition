"""Leakage-safe baseline comparison under a shared event-level protocol."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.protocol import (
    assign_temporal_split,
    eligible_pitch_mask,
    evaluate_warning_predictions,
    select_operating_threshold,
)

logger = logging.getLogger(__name__)
PRIMARY_FASTBALLS = ("FF", "SI", "FC")


class BaselineComparator:
    """Compare all systems on late-2024 episodes at matched warning allowance."""

    def __init__(self, velocity_drop_mph: float = 1.5, pitch_count_thresh: int = 85,
                 horizon_pitches: int = 15, false_warnings_per_outing: float = 0.5):
        self.vel_drop_mph = velocity_drop_mph
        self.pitch_count_thresh = pitch_count_thresh
        self.horizon_pitches = horizon_pitches
        self.false_warnings_per_outing = false_warnings_per_outing

    @staticmethod
    def _velocity_drop_scores(df: pd.DataFrame) -> pd.Series:
        """Compute index-aligned velocity drop separately for FF, SI, and FC."""
        scores = pd.Series(np.nan, index=df.index, dtype=float)
        for _, outing in df.groupby(["game_pk", "pitcher"], sort=False):
            outing = outing.sort_values("pitch_number_in_outing")
            for pitch_type in PRIMARY_FASTBALLS:
                pt = outing[outing["pitch_type"].eq(pitch_type)]
                early = pt[pt["pitch_number_in_outing"].le(20)]["release_speed"].dropna()
                if len(early) < 2:
                    continue
                rolling = pt["release_speed"].rolling(5, min_periods=3).mean()
                scores.loc[pt.index] = float(early.mean()) - rolling
        return scores

    @staticmethod
    def _context_features(df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        features["pitch_number_in_outing"] = pd.to_numeric(df["pitch_number_in_outing"], errors="coerce")
        pa = pd.to_numeric(df["pa_number_in_outing"], errors="coerce")
        features["tto"] = (pa - 1) // 9 + 1
        features["inning"] = pd.to_numeric(df["inning"], errors="coerce")
        return features.fillna(0.0)

    def _add_context_scores(self, df: pd.DataFrame) -> pd.Series:
        train_mask = eligible_pitch_mask(df, "train")
        if not train_mask.any():
            raise ValueError("No eligible 2023 training pitches; contextual model cannot be fit leakage-free.")
        y_train = df.loc[train_mask, "y_true_onset_in_horizon"].astype(int)
        if y_train.nunique() < 2:
            return pd.Series(float(y_train.iloc[0]), index=df.index, dtype=float)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
        )
        features = self._context_features(df)
        model.fit(features.loc[train_mask], y_train)
        return pd.Series(model.predict_proba(features)[:, 1], index=df.index)

    @staticmethod
    def _pitch_statistics(df: pd.DataFrame, predictions: pd.Series, score_col: str,
                          split: str = "test") -> Dict[str, float]:
        mask = eligible_pitch_mask(df, split)
        part = df.loc[mask]
        y_true = part["y_true_onset_in_horizon"].astype(int).to_numpy()
        y_pred = predictions.reindex(part.index).fillna(False).astype(bool).to_numpy()
        p_alert = float(y_true[y_pred].mean()) if y_pred.any() else 0.0
        p_no_alert = float(y_true[~y_pred].mean()) if (~y_pred).any() else np.nan
        risk_ratio = p_alert / p_no_alert if np.isfinite(p_no_alert) and p_no_alert > 0 else np.nan
        finite = part[score_col].replace([np.inf, -np.inf], np.nan).notna()
        if finite.any() and part.loc[finite, "y_true_onset_in_horizon"].nunique() > 1:
            precision, recall, _ = precision_recall_curve(
                part.loc[finite, "y_true_onset_in_horizon"].astype(int), part.loc[finite, score_col]
            )
            pr_auc = float(auc(recall, precision))
        else:
            pr_auc = np.nan
        return {"risk_ratio_alert_vs_no_alert": risk_ratio, "pitch_pr_auc": pr_auc}

    def compare_systems(
        self,
        labeled_df: pd.DataFrame,
        collapse_episodes_df: Optional[pd.DataFrame] = None,
        return_details: bool = False,
    ):
        df = labeled_df.copy()
        if "dataset_split" not in df:
            df = assign_temporal_split(df)
        required = {"train", "validation", "test"}
        present = set(df["dataset_split"].dropna().unique())
        missing = required - present
        if missing:
            raise ValueError(f"Temporal evaluation requires train, validation, and test data; missing {sorted(missing)}")
        if collapse_episodes_df is None:
            collapse_episodes_df = pd.DataFrame()
        elif not collapse_episodes_df.empty and "dataset_split" not in collapse_episodes_df:
            collapse_episodes_df = assign_temporal_split(collapse_episodes_df)

        df["score_proposed_cusum"] = pd.to_numeric(df["cusum_stat"], errors="coerce")
        df["score_velocity_drop"] = self._velocity_drop_scores(df)
        df["score_pitch_count"] = pd.to_numeric(df["pitch_number_in_outing"], errors="coerce")
        df["score_contextual"] = self._add_context_scores(df)

        model_specs = [
            ("proposed", "Proposed Micro-Mechanics (CUSUM + MSI)", "score_proposed_cusum",
             "Mechanical-drift score; association is not evidence of fatigue causality."),
            ("contextual", "Contextual Model (Pitch Count + TTO + Inning)", "score_contextual",
             "Fit on 2023 only; threshold selected on early 2024."),
            ("velocity", "Pitch-Type Velocity Drop (FF/SI/FC separately)", "score_velocity_drop",
             "A contemporaneous velocity benchmark; relative risk alone does not establish timing."),
            ("pitch_count", "Traditional Pitch Count", "score_pitch_count",
             "A workload heuristic with the same validation selection rule."),
        ]

        rows = []
        metrics_by_model: Dict[str, Dict[str, Any]] = {}
        for key, display_name, score_col, interpretation in model_specs:
            threshold, validation_metrics = select_operating_threshold(
                df, collapse_episodes_df, score_col, self.false_warnings_per_outing,
                horizon_pitches=self.horizon_pitches, split="validation"
            )
            prediction_col = f"is_{key}_operating_alert"
            df[prediction_col] = df[score_col].ge(threshold) & df[score_col].notna()
            test_metrics, warnings, matches = evaluate_warning_predictions(
                df, collapse_episodes_df, df[prediction_col],
                horizon_pitches=self.horizon_pitches, split="test"
            )
            test_metrics.update(self._pitch_statistics(df, df[prediction_col], score_col))
            test_metrics["operating_threshold"] = threshold
            test_metrics["validation_false_warnings_per_outing"] = validation_metrics.get(
                "false_warnings_per_outing", np.nan
            )
            test_metrics["evaluation_split"] = "test"
            test_metrics["actual_data_sources"] = sorted(
                map(str, df.loc[eligible_pitch_mask(df, "test"), "actual_data_source"].dropna().unique())
            ) if "actual_data_source" in df else ["unknown"]
            metrics_by_model[key] = test_metrics

            def fmt(value, digits=3):
                return np.nan if not np.isfinite(value) else round(float(value), digits)

            rows.append({
                "Model / System": display_name,
                "Test Episode Recall": fmt(test_metrics["episode_recall"]),
                "Test Warning Precision": fmt(test_metrics["warning_precision"]),
                "Test False Warnings / Outing": fmt(test_metrics["false_warnings_per_outing"]),
                "Test Clean Outing FAR": fmt(test_metrics["clean_outing_false_alarm_rate"]),
                "Test Pitch PR-AUC": fmt(test_metrics["pitch_pr_auc"]),
                "Test Risk Ratio (Alert vs No Alert)": fmt(test_metrics["risk_ratio_alert_vs_no_alert"], 2),
                "Validation-Selected Threshold": fmt(threshold, 4),
                "Validation False Warnings / Outing": fmt(
                    test_metrics["validation_false_warnings_per_outing"]
                ),
                "Actual Data Source": ", ".join(test_metrics["actual_data_sources"]),
                "Interpretation": interpretation,
            })

        comparison = pd.DataFrame(rows)
        logger.info("\n%s", comparison.to_string(index=False))
        if return_details:
            return comparison, df, metrics_by_model
        return comparison
