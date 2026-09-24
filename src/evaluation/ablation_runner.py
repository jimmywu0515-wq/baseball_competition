"""Feature experiment availability and frozen-prediction sensitivity reruns."""
from __future__ import annotations

import hashlib
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

from src.evaluation.protocol import (
    eligible_pitch_mask,
    evaluate_warning_predictions,
    select_operating_threshold,
)
from src.label_builder.collapse_labels import CollapseLabelBuilder

logger = logging.getLogger(__name__)


class AblationRunner:
    def __init__(self, base_df: pd.DataFrame, collapse_episodes_df: Optional[pd.DataFrame] = None,
                 false_warnings_per_outing: float = 0.5,
                 horizon_pitches: int = 15):
        self.df = base_df.copy()
        self.episodes = collapse_episodes_df if collapse_episodes_df is not None else pd.DataFrame()
        self.false_warnings_per_outing = false_warnings_per_outing
        self.horizon_pitches = horizon_pitches

    def run_feature_ablations(self, primary_threshold: float,
                              primary_metrics: dict, protocol_sha256: str = "",
                              primary_warning_ids: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Re-evaluate the frozen full model and reject any primary parity mismatch.

        Subsets need their own historical covariance baselines. Those are not
        retained in the scored pitch frame, so they are explicitly unavailable.
        """
        required = {"score_proposed_cusum", "is_proposed_operating_alert"}
        if not required.issubset(self.df.columns):
            raise ValueError(f"Full ablation requires {sorted(required)}")
        scores = pd.to_numeric(self.df["score_proposed_cusum"], errors="coerce")
        availability = pd.Series(np.isfinite(scores), index=self.df.index)
        predicted = self.df["is_proposed_operating_alert"].fillna(False).astype(bool)
        expected_prediction = scores.ge(primary_threshold) & availability
        if not predicted.equals(expected_prediction):
            raise ValueError("Full-feature ablation predictions differ from frozen primary threshold")
        metrics, warnings, _ = evaluate_warning_predictions(
            self.df, self.episodes, predicted, horizon_pitches=self.horizon_pitches,
            split="test", availability=availability,
        )
        identity = ["game_pk", "pitcher", "warning_pitch"]
        if primary_warning_ids is not None:
            left = warnings[identity].sort_values(identity).reset_index(drop=True)
            right = primary_warning_ids[identity].sort_values(identity).reset_index(drop=True)
            if not left.equals(right):
                raise ValueError("Full-feature ablation warning IDs differ from primary")
        parity_keys = (
            "episode_recall", "warning_precision", "false_warnings_per_outing",
            "total_qualified_outings", "evaluated_episodes_count", "total_warnings",
            "false_warnings", "evaluation_opportunity_pitches",
            "score_available_evaluation_pitches", "pitch_level_scoring_coverage",
            "outing_level_scoring_coverage",
        )
        for key in parity_keys:
            actual, expected = metrics[key], primary_metrics[key]
            equal = (pd.isna(actual) and pd.isna(expected)) or np.isclose(actual, expected, rtol=0, atol=1e-12)
            if not equal:
                raise ValueError(f"Full-feature ablation parity failed for {key}: {actual} vs {expected}")
        if not (pd.isna(primary_metrics.get("operating_threshold")) and np.isinf(primary_threshold)):
            if not np.isclose(primary_threshold, primary_metrics["operating_threshold"], rtol=0, atol=1e-12):
                raise ValueError("Full-feature ablation threshold differs from primary")
        full = {
            "Feature Subset": "Full Micro-Mechanics Suite", "status": "verified",
            "reason": "Independent evaluation of frozen primary predictions",
            "protocol_sha256": protocol_sha256,
            "Test Episode Recall": metrics["episode_recall"],
            "Test Warning Precision": metrics["warning_precision"],
            "Test False Warnings / Outing": metrics["false_warnings_per_outing"],
            "Test Qualified Outings": metrics["total_qualified_outings"],
            "Test Evaluated Episodes": metrics["evaluated_episodes_count"],
            "Test Warnings": metrics["total_warnings"],
            "Test False Warnings": metrics["false_warnings"],
            "Test Pitch Scoring Coverage": metrics["pitch_level_scoring_coverage"],
            "Validation-Selected Threshold": primary_threshold if np.isfinite(primary_threshold) else np.nan,
            "warning_ids_sha256": hashlib.sha256(
                warnings[identity].sort_values(identity).to_csv(index=False).encode()
            ).hexdigest(),
        }
        rows = [full]
        for name in ("Velocity Alone", "Release Point Alone (X, Z, Extension)",
                     "Spin & Movement (rate, axis, PFX, VAA)"):
            rows.append({"Feature Subset": name, "status": "unavailable",
                         "reason": "Subset historical covariance baselines were not retained",
                         "protocol_sha256": protocol_sha256})
        return pd.DataFrame(rows)

    def run_sensitivity_analysis(self) -> pd.DataFrame:
        settings = [
            (0.400, 3, 15), (0.450, 3, 15), (0.500, 3, 15),
            (0.450, 2, 15), (0.450, 3, 10), (0.450, 3, 20),
            (0.450, 3, 25),
        ]
        rows = []
        for threshold, window_pas, horizon in settings:
            builder = CollapseLabelBuilder(
                window_pa_size=window_pas,
                blended_xwoba_threshold=threshold,
                horizon_pitches=horizon,
            )
            labeled, episodes = [], []
            for _, outing in self.df.groupby(["game_pk", "pitcher"], sort=False):
                outing_labeled, outing_episodes = builder.build_labels_for_outing(outing)
                labeled.append(outing_labeled)
                if not outing_episodes.empty:
                    episodes.append(outing_episodes)
            rerun_df = pd.concat(labeled, ignore_index=True)
            rerun_episodes = pd.concat(episodes, ignore_index=True) if episodes else pd.DataFrame()
            op_threshold, validation_metrics = select_operating_threshold(
                rerun_df, rerun_episodes, "cusum_stat", self.false_warnings_per_outing,
                horizon_pitches=horizon, split="validation"
            )
            predictions = rerun_df["cusum_stat"].ge(op_threshold) & rerun_df["cusum_stat"].notna()
            metrics, _, _ = evaluate_warning_predictions(
                rerun_df, rerun_episodes, predictions, horizon_pitches=horizon, split="test"
            )
            rows.append({
                "xwOBA Threshold": threshold,
                "Window (PA)": window_pas,
                "Horizon (Pitches)": horizon,
                "Test Episodes": metrics["total_collapse_episodes"],
                "Test Episode Recall": round(metrics["episode_recall"], 3),
                "Test Warning Precision": round(metrics["warning_precision"], 3),
                "Test False Warnings / Outing": round(metrics["false_warnings_per_outing"], 3),
                "Validation-Selected Threshold": round(op_threshold, 4) if np.isfinite(op_threshold) else np.nan,
                "Validation False Warnings / Outing": round(
                    validation_metrics.get("false_warnings_per_outing", np.nan), 3
                ),
            })
        result = pd.DataFrame(rows)
        logger.info("\n=== SENSITIVITY RESULTS ===\n%s", result.to_string(index=False))
        return result
