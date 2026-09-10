"""Feature ablations and sensitivity experiments that actually rerun evaluation."""
from __future__ import annotations

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
                 false_warnings_per_outing: float = 0.5):
        self.df = base_df.copy()
        self.episodes = collapse_episodes_df if collapse_episodes_df is not None else pd.DataFrame()
        self.false_warnings_per_outing = false_warnings_per_outing

    @staticmethod
    def _euclidean_score(df: pd.DataFrame, columns) -> pd.Series:
        values = df.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
        complete = values.notna().all(axis=1)
        result = pd.Series(np.nan, index=df.index, dtype=float)
        result.loc[complete] = np.sqrt((values.loc[complete] ** 2).sum(axis=1))
        return result

    def run_feature_ablations(self) -> pd.DataFrame:
        configs = {
            "Velocity Alone": ["z_release_speed"],
            "Release Point Alone (X, Z, Extension)": [
                "z_release_pos_x", "z_release_pos_z", "z_release_extension"
            ],
            "Spin & Movement (rate, axis, PFX, VAA)": [
                "z_release_spin_rate", "z_spin_axis_cos", "z_spin_axis_sin",
                "z_pfx_x", "z_pfx_z", "z_vaa",
            ],
        }
        working = self.df.copy()
        score_cols = {}
        for number, (name, columns) in enumerate(configs.items()):
            col = f"ablation_score_{number}"
            working[col] = self._euclidean_score(working, columns)
            score_cols[name] = col
        working["ablation_score_full"] = pd.to_numeric(
            working["mahalanobis_calibrated"], errors="coerce"
        )
        score_cols["Full Micro-Mechanics Suite"] = "ablation_score_full"

        rows = []
        test_mask = eligible_pitch_mask(working, "test")
        for name, score_col in score_cols.items():
            threshold, validation_metrics = select_operating_threshold(
                working, self.episodes, score_col, self.false_warnings_per_outing,
                horizon_pitches=15, split="validation"
            )
            predictions = working[score_col].ge(threshold) & working[score_col].notna()
            metrics, _, _ = evaluate_warning_predictions(
                working, self.episodes, predictions, horizon_pitches=15, split="test"
            )
            part = working.loc[test_mask, [score_col, "y_true_onset_in_horizon"]].dropna()
            if not part.empty and part["y_true_onset_in_horizon"].nunique() > 1:
                precision, recall, _ = precision_recall_curve(
                    part["y_true_onset_in_horizon"].astype(int), part[score_col]
                )
                pr_auc = float(auc(recall, precision))
            else:
                pr_auc = np.nan
            rows.append({
                "Feature Subset": name,
                "Test Pitch PR-AUC": round(pr_auc, 3) if np.isfinite(pr_auc) else np.nan,
                "Test Episode Recall": round(metrics["episode_recall"], 3),
                "Test Warning Precision": round(metrics["warning_precision"], 3),
                "Test False Warnings / Outing": round(metrics["false_warnings_per_outing"], 3),
                "Validation-Selected Threshold": round(threshold, 4) if np.isfinite(threshold) else np.nan,
                "Validation False Warnings / Outing": round(
                    validation_metrics.get("false_warnings_per_outing", np.nan), 3
                ),
            })
        result = pd.DataFrame(rows)
        logger.info("\n=== FEATURE ABLATION RESULTS ===\n%s", result.to_string(index=False))
        return result

    def run_sensitivity_analysis(self) -> pd.DataFrame:
        settings = [
            (0.400, 3, 15), (0.450, 3, 15), (0.500, 3, 15),
            (0.450, 2, 15), (0.450, 3, 10), (0.450, 3, 20),
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
