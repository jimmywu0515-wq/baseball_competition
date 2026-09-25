"""Leakage-safe baseline comparison under a shared event-level protocol."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.protocol import (
    assign_temporal_split,
    eligible_pitch_mask,
    evaluation_opportunity_mask,
    evaluate_warning_predictions,
    select_operating_threshold,
    threshold_tradeoff_curve,
)

logger = logging.getLogger(__name__)
PRIMARY_FASTBALLS = ("FF", "SI", "FC")


class BaselineComparator:
    """Freeze operating points on validation, then compare systems on test."""

    def __init__(self, velocity_drop_mph: float = 1.5, pitch_count_thresh: int = 85,
                 horizon_pitches: int = 15, false_warnings_per_outing: float = 0.5,
                 calibration_pitches: int = 20):
        self.vel_drop_mph = velocity_drop_mph
        self.pitch_count_thresh = pitch_count_thresh
        self.horizon_pitches = horizon_pitches
        self.false_warnings_per_outing = false_warnings_per_outing
        self.calibration_pitches = calibration_pitches
        self.threshold_curves = pd.DataFrame()
        self.warning_records = pd.DataFrame()
        self.match_records = pd.DataFrame()
        self.frozen_thresholds: Dict[str, float] = {}
        self.validation_metrics: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _velocity_drop_scores(df: pd.DataFrame, calibration_pitches: int = 20) -> pd.Series:
        """Compute index-aligned velocity drop separately for FF, SI, and FC."""
        scores = pd.Series(np.nan, index=df.index, dtype=float)
        for _, outing in df.groupby(["game_pk", "pitcher"], sort=False):
            outing = outing.sort_values("pitch_number_in_outing")
            for pitch_type in PRIMARY_FASTBALLS:
                pt = outing[outing["pitch_type"].eq(pitch_type)]
                early = pt[pt["pitch_number_in_outing"].le(calibration_pitches)]["release_speed"].dropna()
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
        train_mask = evaluation_opportunity_mask(df, "train")
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
        availability = pd.Series(np.isfinite(pd.to_numeric(df[score_col], errors="coerce")), index=df.index)
        mask = eligible_pitch_mask(df, split, availability=availability)
        part = df.loc[mask]
        y_true = part["y_true_onset_in_horizon"].astype(int).to_numpy()
        y_pred = predictions.reindex(part.index).fillna(False).astype(bool).to_numpy()
        p_alert = float(y_true[y_pred].mean()) if y_pred.any() else np.nan
        p_no_alert = float(y_true[~y_pred].mean()) if (~y_pred).any() else np.nan
        if not np.isfinite(p_alert) or not np.isfinite(p_no_alert):
            risk_ratio, risk_status = np.nan, "missing_exposure_group"
        elif p_no_alert == 0:
            risk_ratio = np.inf if p_alert > 0 else np.nan
            risk_status = "infinite" if p_alert > 0 else "zero_over_zero"
        else:
            risk_ratio, risk_status = p_alert / p_no_alert, "defined"
        finite = part[score_col].replace([np.inf, -np.inf], np.nan).notna()
        if finite.any() and part.loc[finite, "y_true_onset_in_horizon"].nunique() > 1:
            precision, recall, _ = precision_recall_curve(
                part.loc[finite, "y_true_onset_in_horizon"].astype(int), part.loc[finite, score_col]
            )
            pr_auc = float(auc(recall, precision))
        else:
            pr_auc = np.nan
        return {"risk_ratio_alert_vs_no_alert": risk_ratio,
                "risk_ratio_status": risk_status, "pitch_pr_auc": pr_auc}

    def compare_systems(
        self,
        labeled_df: pd.DataFrame,
        collapse_episodes_df: Optional[pd.DataFrame] = None,
        return_details: bool = False,
        freeze_callback: Optional[Callable[[Dict[str, float]], None]] = None,
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
        df["score_velocity_drop"] = self._velocity_drop_scores(df, self.calibration_pitches)
        df["score_pitch_count"] = pd.to_numeric(df["pitch_number_in_outing"], errors="coerce")
        df["score_contextual"] = self._add_context_scores(df)

        model_specs = [
            ("proposed", "Proposed Micro-Mechanics (CUSUM + MSI)", "score_proposed_cusum",
             "Mechanical-drift score; association is not evidence of fatigue causality."),
            ("contextual", "Contextual Model (Pitch Count + TTO + Inning)", "score_contextual",
             "Fit on 2023 only; threshold selected on 2024 validation data."),
            ("velocity", "Pitch-Type Velocity Drop (FF/SI/FC separately)", "score_velocity_drop",
             "A contemporaneous velocity benchmark; relative risk alone does not establish timing."),
            ("pitch_count", "Traditional Pitch Count", "score_pitch_count",
             "A workload heuristic with the same validation selection rule."),
        ]

        # Pass 1 uses validation only. These operating points are frozen before
        # any 2025 metric is calculated.
        curve_frames = []
        curves_by_model = {}
        validation_by_model = {}
        for key, display_name, score_col, _ in model_specs:
            threshold, validation_metrics = select_operating_threshold(
                df, collapse_episodes_df, score_col, self.false_warnings_per_outing,
                horizon_pitches=self.horizon_pitches, split="validation"
            )
            self.frozen_thresholds[key] = threshold
            validation_by_model[key] = validation_metrics
            self.validation_metrics[key] = validation_metrics
            curve = threshold_tradeoff_curve(
                df, collapse_episodes_df, score_col,
                horizon_pitches=self.horizon_pitches, split="validation",
            )
            curve["model_key"] = key
            curve["Model / System"] = display_name
            curve["score_column"] = score_col
            curve["false_warning_allowance"] = self.false_warnings_per_outing
            curve["is_selected_operating_point"] = np.isclose(
                curve["threshold"], threshold, equal_nan=False
            )
            curve_frames.append(curve)
            curves_by_model[key] = curve

        if freeze_callback is not None:
            freeze_callback(dict(self.frozen_thresholds))

        rows = []
        warning_frames = []
        match_frames = []
        metrics_by_model: Dict[str, Dict[str, Any]] = {}
        for key, display_name, score_col, interpretation in model_specs:
            threshold = self.frozen_thresholds[key]
            validation_metrics = validation_by_model[key]
            prediction_col = f"is_{key}_operating_alert"
            df[prediction_col] = df[score_col].ge(threshold) & df[score_col].notna()
            test_metrics, warnings, matches = evaluate_warning_predictions(
                df, collapse_episodes_df, df[prediction_col],
                horizon_pitches=self.horizon_pitches, split="test",
                availability=pd.Series(np.isfinite(pd.to_numeric(df[score_col], errors="coerce")), index=df.index),
            )
            test_metrics.update(self._pitch_statistics(df, df[prediction_col], score_col))
            test_metrics["operating_threshold"] = threshold
            test_metrics["validation_false_warnings_per_outing"] = validation_metrics.get(
                "false_warnings_per_outing", np.nan
            )
            test_metrics["evaluation_split"] = "test"
            test_metrics["actual_data_sources"] = sorted(
                map(str, df.loc[evaluation_opportunity_mask(df, "test"), "actual_data_source"].dropna().unique())
            ) if "actual_data_source" in df else ["unknown"]
            metrics_by_model[key] = test_metrics
            if not warnings.empty:
                warning_frames.append(warnings.assign(model_key=key, model_name=display_name))
            if not matches.empty:
                match_frames.append(matches.assign(model_key=key, model_name=display_name))

            curve = curves_by_model[key]
            if not curve.empty:
                selected = curve["is_selected_operating_point"]
                curve.loc[selected, "test_episode_recall_at_frozen_threshold"] = test_metrics["episode_recall"]
                curve.loc[selected, "test_warning_precision_at_frozen_threshold"] = test_metrics["warning_precision"]
                curve.loc[selected, "test_false_warnings_per_outing_at_frozen_threshold"] = test_metrics[
                    "false_warnings_per_outing"
                ]

            def fmt(value, digits=3):
                return np.nan if not np.isfinite(value) else round(float(value), digits)

            rows.append({
                "Model Key": key,
                "Model / System": display_name,
                "Validation Episode Recall": fmt(validation_metrics.get("episode_recall", np.nan)),
                "Validation Warning Precision": fmt(validation_metrics.get("warning_precision", np.nan)),
                "Validation False Warnings / Outing": fmt(
                    validation_metrics.get("false_warnings_per_outing", np.nan)
                ),
                "Allowed Maximum False Warnings / Outing": self.false_warnings_per_outing,
                "Test Episode Recall": fmt(test_metrics["episode_recall"]),
                "Test Warning Precision": fmt(test_metrics["warning_precision"]),
                "Test False Warnings / Outing": fmt(test_metrics["false_warnings_per_outing"]),
                "Test Clean Outing FAR": fmt(test_metrics["clean_outing_false_alarm_rate"]),
                "Test Pitch PR-AUC": fmt(test_metrics["pitch_pr_auc"]),
                "Test Risk Ratio (Alert vs No Alert)": fmt(test_metrics["risk_ratio_alert_vs_no_alert"], 2),
                "Test Risk Ratio Status": test_metrics["risk_ratio_status"],
                "Validation-Selected Threshold": fmt(threshold, 4),
                "Threshold Status": "no_alert" if not np.isfinite(threshold) else "selected",
                "Test Qualified Outings": test_metrics["total_qualified_outings"],
                "Test Outings With Available Score": test_metrics["outings_with_available_score"],
                "Test Outings Without Available Score": test_metrics["outings_without_available_score"],
                "Test Pitch Scoring Coverage": fmt(test_metrics["pitch_level_scoring_coverage"]),
                "Test Outing Scoring Coverage": fmt(test_metrics["outing_level_scoring_coverage"]),
                "Test Evaluated Episodes": test_metrics["evaluated_episodes_count"],
                "Actual Data Source": ", ".join(test_metrics["actual_data_sources"]),
                "Interpretation": interpretation,
            })

        comparison = pd.DataFrame(rows)
        self.threshold_curves = pd.concat(curve_frames, ignore_index=True)
        self.warning_records = (
            pd.concat(warning_frames, ignore_index=True) if warning_frames else pd.DataFrame()
        )
        self.match_records = (
            pd.concat(match_frames, ignore_index=True) if match_frames else pd.DataFrame()
        )
        logger.info("\n%s", comparison.to_string(index=False))
        if return_details:
            return comparison, df, metrics_by_model
        return comparison
