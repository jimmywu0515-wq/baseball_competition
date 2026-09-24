"""Feature ablations and sensitivity experiments that actually rerun evaluation.

Corrected implementation (2026-09):
- Every feature variant's anomaly score is processed through CUSUM before
  threshold selection and evaluation, matching the production system.
- Subset scores are Mahalanobis distances computed from the corresponding
  historical covariance submatrix (not Euclidean norms of z-scores).
- The full-feature reference reuses persisted production CUSUM scores and
  verifies parity with the main proposed system.
- Detector parameters are passed explicitly from configuration.
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

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


# Production feature columns and their group indices
FEATURE_COLS = [
    "release_pos_x", "release_pos_z", "release_extension", "release_speed",
    "release_spin_rate", "spin_axis_cos", "spin_axis_sin", "pfx_x", "pfx_z", "vaa",
]

ABLATION_GROUPS = {
    "Velocity Alone": [3],  # release_speed
    "Release Point Alone (X, Z, Extension)": [0, 1, 2],
    "Spin & Movement (rate, axis, PFX, VAA)": [4, 5, 6, 7, 8, 9],
}


def _get_git_metadata() -> Tuple[Optional[str], Optional[bool]]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return sha, bool(status)
    except Exception:
        return None, None


@dataclass
class AblationConfig:
    """Resolved settings for the ablation experiment."""
    cusum_slack_k: float = 0.5
    cusum_threshold_h: float = 4.5
    cusum_reference_mean: float = 1.0
    cusum_reference_std: float = 0.5
    calibration_pitches: int = 20
    horizon_pitches: int = 15
    false_warnings_per_outing: float = 0.5
    ridge_regularization: float = 1e-4
    normalization_policy: str = "none"  # "none" = raw subset Mahalanobis; fixed CUSUM params
    availability_policy: str = "variant_specific"
    distance_construction: str = "subset_mahalanobis"


@dataclass
class AblationManifest:
    """Provenance record for the ablation experiment."""
    variants: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    config: Optional[Dict[str, Any]] = None
    parent_protocol_hash: Optional[str] = None
    input_artifact: str = "fact_pitch_anomaly_scores (persisted gold)"
    git_sha: Optional[str] = None
    git_dirty: Optional[bool] = None
    output_location: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def verify_full_feature_parity(
    ablation_df: pd.DataFrame,
    expected_metrics: Dict[str, Any],
) -> bool:
    """Verify that the Full Micro-Mechanics Suite variant agrees with the production proposed system.

    Raises ValueError if there is any statistically meaningful discrepancy.
    """
    match = ablation_df[ablation_df["Feature Subset"] == "Full Micro-Mechanics Suite"]
    if match.empty:
        raise ValueError("Ablation dataframe does not contain 'Full Micro-Mechanics Suite'")
    row = match.iloc[0]

    checks = [
        ("Test Episode Recall", "episode_recall"),
        ("Test Warning Precision", "warning_precision"),
        ("Test False Warnings / Outing", "false_warnings_per_outing"),
    ]
    for abl_col, exp_key in checks:
        if exp_key in expected_metrics:
            val = float(row[abl_col])
            exp = float(expected_metrics[exp_key])
            if abs(val - exp) > 0.001 and abs(val - round(exp, 4)) > 0.001:
                raise ValueError(
                    f"Parity mismatch for {abl_col}: ablation reports {val}, "
                    f"expected {exp} (diff: {abs(val - exp):.6f})"
                )

    if "operating_threshold" in expected_metrics and expected_metrics["operating_threshold"] is not None:
        thresh_val = float(row["Validation-Selected Threshold"])
        thresh_exp = float(expected_metrics["operating_threshold"])
        if abs(thresh_val - thresh_exp) > 0.01 and abs(thresh_val - round(thresh_exp, 4)) > 0.01:
            raise ValueError(
                f"Parity mismatch for Validation-Selected Threshold: ablation reports {thresh_val}, "
                f"expected {thresh_exp} (diff: {abs(thresh_val - thresh_exp):.6f})"
            )
    return True


class AblationRunner:
    def __init__(
        self,
        base_df: pd.DataFrame,
        collapse_episodes_df: Optional[pd.DataFrame] = None,
        config: Optional[AblationConfig] = None,
        baseline_store: Optional[Dict[str, Any]] = None,
    ):
        self.df = base_df.copy()
        self.episodes = (
            collapse_episodes_df if collapse_episodes_df is not None else pd.DataFrame()
        )
        self.config = config or AblationConfig()
        self.baseline_store = baseline_store
        git_sha, git_dirty = _get_git_metadata()
        self.manifest = AblationManifest(
            config=asdict(self.config),
            git_sha=git_sha,
            git_dirty=git_dirty,
        )

    @staticmethod
    def _run_cusum_on_scores(
        df: pd.DataFrame,
        score_col: str,
        cusum_output_col: str,
        slack_k: float,
        threshold_h: float,
        reference_mean: float,
        reference_std: float,
        calibration_pitches: int,
    ) -> pd.DataFrame:
        """Run CUSUM on an anomaly score column, resetting between outings.

        This mirrors CUSUMDetector.detect_game_alerts but operates on an
        arbitrary score column and writes to a specified output column.
        Unavailable scores (NaN) do not update the CUSUM statistic.
        Calibration pitches (≤ calibration_pitches) are excluded.
        Pitch gaps and outing boundaries reset the detector.
        """
        result = df.copy()
        cusum_vals = np.full(len(result), np.nan)

        for _, outing in result.groupby(["game_pk", "pitcher"], sort=False):
            outing_sorted = outing.sort_values("pitch_number_in_outing")
            statistic = 0.0

            for idx in outing_sorted.index:
                row = result.loc[idx]
                pitch_number = int(row.get(
                    "pitch_number_in_outing",
                    row.get("pitch_number_in_game", 0),
                ))
                score = pd.to_numeric(row.get(score_col), errors="coerce")
                # Match production eligibility: pitch > calibration, score available, finite
                score_avail = bool(row.get(f"_ablation_avail_{score_col}",
                                           np.isfinite(score) if pd.notna(score) else False))
                eligible = (
                    pitch_number > calibration_pitches
                    and score_avail
                    and np.isfinite(score)
                )
                if not eligible:
                    continue

                z = (score - reference_mean) / max(reference_std, 1e-6)
                statistic = max(0.0, statistic + z - slack_k)
                cusum_vals[result.index.get_loc(idx)] = round(statistic, 3)

        result[cusum_output_col] = cusum_vals
        return result

    def _compute_subset_mahalanobis(
        self,
        df: pd.DataFrame,
        feature_indices: List[int],
        variant_name: str,
    ) -> Tuple[pd.Series, pd.Series]:
        """Compute Mahalanobis distance using the historical covariance submatrix.

        For each pitch, extract the subset of features, obtain the corresponding
        rows/columns from the pitcher×pitch-type covariance matrix, invert the
        submatrix with ridge regularization, and compute the quadratic form.

        Returns (score_series, availability_series).
        """
        subset_cols = [FEATURE_COLS[i] for i in feature_indices]
        n = len(df)
        scores = pd.Series(np.nan, index=df.index, dtype=float)
        available = pd.Series(False, index=df.index, dtype=bool)

        if self.baseline_store is None:
            logger.warning(
                "No baseline store provided; subset Mahalanobis cannot be computed. "
                "Falling back to Euclidean on z-scores for %s.",
                variant_name,
            )
            self.manifest.variants[variant_name] = {
                "distance_construction": "euclidean_zscore (fallback)",
                "limitation": "No baseline store; cannot compute subset Mahalanobis.",
            }
            return self._euclidean_fallback(df, feature_indices, variant_name)

        calibration = df.get(
            "is_calibration_phase", pd.Series(False, index=df.index)
        ).fillna(False).astype(bool)

        for (game_pk, pitcher), outing in df.groupby(
            ["game_pk", "pitcher"], sort=False
        ):
            outing_mask = (df["game_pk"] == game_pk) & (df["pitcher"] == pitcher)
            for pitch_type in df.loc[outing_mask, "pitch_type"].dropna().unique():
                key = f"{pitcher}_{game_pk}_{pitch_type}"
                baseline = self.baseline_store.get(key)
                if baseline is None or baseline.get("status") != "QUALIFIED":
                    continue

                pt_mask = outing_mask & (df["pitch_type"] == pitch_type) & ~calibration
                if not pt_mask.any():
                    continue

                # Extract feature values
                values = df.loc[pt_mask, subset_cols].apply(
                    pd.to_numeric, errors="coerce"
                ).to_numpy(float)
                complete = np.isfinite(values).all(axis=1)
                if not complete.any():
                    continue

                # Get subset covariance and mean from the full baseline
                full_mu = np.asarray(baseline["mu_vec"], dtype=float)
                full_cov = np.asarray(baseline["cov_mat"], dtype=float)

                subset_mu = full_mu[feature_indices]
                subset_cov = full_cov[np.ix_(feature_indices, feature_indices)]

                # Re-regularize and invert the *submatrix* (not a slice of prec_mat)
                reg_cov = subset_cov + np.eye(len(feature_indices)) * self.config.ridge_regularization
                try:
                    subset_prec = np.linalg.inv(reg_cov)
                except np.linalg.LinAlgError:
                    subset_prec = np.linalg.pinv(reg_cov)

                # Compute Mahalanobis distance using calibrated mean if available
                # For ablation consistency, use the historical mean (not calibrated)
                # since we don't have subset-specific calibrated means
                valid_idx = pt_mask & pd.Series(complete, index=df.loc[pt_mask].index).reindex(df.index, fill_value=False)
                valid_values = values[complete]
                diff = valid_values - subset_mu
                squared = np.einsum("ij,jk,ik->i", diff, subset_prec, diff)
                dist = np.sqrt(np.maximum(0.0, squared))

                scores.loc[valid_idx] = np.round(dist, 4)
                available.loc[valid_idx] = True

        return scores, available

    def _euclidean_fallback(
        self,
        df: pd.DataFrame,
        feature_indices: List[int],
        variant_name: str,
    ) -> Tuple[pd.Series, pd.Series]:
        """Fallback: Euclidean norm of z-scored features."""
        z_prefix = "z_"
        z_cols = [f"{z_prefix}{FEATURE_COLS[i]}" for i in feature_indices]
        available_cols = [c for c in z_cols if c in df.columns]
        if not available_cols:
            return (
                pd.Series(np.nan, index=df.index, dtype=float),
                pd.Series(False, index=df.index, dtype=bool),
            )
        values = df[available_cols].apply(pd.to_numeric, errors="coerce")
        complete = values.notna().all(axis=1)
        calibration = df.get(
            "is_calibration_phase", pd.Series(False, index=df.index)
        ).fillna(False).astype(bool)
        complete = complete & ~calibration

        scores = pd.Series(np.nan, index=df.index, dtype=float)
        scores.loc[complete] = np.sqrt((values.loc[complete] ** 2).sum(axis=1))
        return scores, complete

    def run_feature_ablations(self) -> Tuple[pd.DataFrame, AblationManifest]:
        """Run the corrected feature ablation experiment.

        Every variant's anomaly score goes through CUSUM, with threshold
        selection on 2024 validation data and evaluation on 2025 test data.
        The full-feature reference reuses persisted production CUSUM scores.
        """
        working = self.df.copy()
        cfg = self.config
        results_rows = []

        # --- Full-feature reference: reuse persisted cusum_stat ---
        full_score_col = "cusum_stat"
        full_cusum_col = full_score_col  # Already CUSUM output
        variant_name = "Full Micro-Mechanics Suite"

        if full_score_col not in working.columns:
            raise ValueError(
                "Persisted cusum_stat column not found. The full-feature reference "
                "requires the production CUSUM scores."
            )

        # Verify production score availability
        if "score_available" in working.columns:
            working[f"_ablation_avail_{full_cusum_col}"] = working["score_available"].fillna(False)
        else:
            working[f"_ablation_avail_{full_cusum_col}"] = working[full_cusum_col].notna()

        full_threshold, full_val_metrics = select_operating_threshold(
            working, self.episodes, full_cusum_col,
            cfg.false_warnings_per_outing,
            horizon_pitches=cfg.horizon_pitches,
            split="validation",
        )
        full_predictions = working[full_cusum_col].ge(full_threshold) & working[full_cusum_col].notna()
        full_test_metrics, _, _ = evaluate_warning_predictions(
            working, self.episodes, full_predictions,
            horizon_pitches=cfg.horizon_pitches, split="test",
        )

        # PR-AUC on test set
        test_mask = eligible_pitch_mask(working, "test")
        part = working.loc[test_mask, [full_cusum_col, "y_true_onset_in_horizon"]].dropna()
        if not part.empty and part["y_true_onset_in_horizon"].nunique() > 1:
            precision_arr, recall_arr, _ = precision_recall_curve(
                part["y_true_onset_in_horizon"].astype(int), part[full_cusum_col]
            )
            full_pr_auc = float(auc(recall_arr, precision_arr))
        else:
            full_pr_auc = np.nan

        # Compute coverage for full-feature
        test_pitches = working.loc[test_mask]
        full_avail = working.loc[test_mask, f"_ablation_avail_{full_cusum_col}"]
        full_pitch_coverage = float(full_avail.sum() / len(test_pitches)) if len(test_pitches) else 0.0
        full_outing_keys = test_pitches[["game_pk", "pitcher"]].drop_duplicates()
        scored_outings = test_pitches.loc[full_avail, ["game_pk", "pitcher"]].drop_duplicates()
        full_outing_coverage = float(len(scored_outings) / len(full_outing_keys)) if len(full_outing_keys) else 0.0

        results_rows.append({
            "Feature Subset": variant_name,
            "Score Method": "mahalanobis_calibrated → CUSUM (production reuse)",
            "Validation-Selected Threshold": (
                round(full_threshold, 4) if np.isfinite(full_threshold) else np.nan
            ),
            "Validation Recall": round(full_val_metrics.get("episode_recall", 0.0), 4),
            "Validation Precision": round(full_val_metrics.get("warning_precision", 0.0), 4),
            "Validation False Warnings / Outing": round(
                full_val_metrics.get("false_warnings_per_outing", 0.0), 4
            ),
            "Allowed Max FW/Outing": cfg.false_warnings_per_outing,
            "Test Episode Recall": round(full_test_metrics["episode_recall"], 4),
            "Test Warning Precision": round(full_test_metrics["warning_precision"], 4),
            "Test False Warnings / Outing": round(full_test_metrics["false_warnings_per_outing"], 4),
            "Test Pitch PR-AUC": round(full_pr_auc, 4) if np.isfinite(full_pr_auc) else np.nan,
            "Test Evaluated Outings": full_test_metrics["evaluated_outings_count"],
            "Test Evaluated Episodes": full_test_metrics["total_collapse_episodes"],
            "Test Warnings": full_test_metrics["total_warnings"],
            "Test False Warnings": full_test_metrics["false_warnings"],
            "Test Pitch Coverage": round(full_pitch_coverage, 4),
            "Test Outing Coverage": round(full_outing_coverage, 4),
            "Actual Data Source": working.get(
                "actual_data_source", pd.Series(["unknown"])
            ).dropna().iloc[0] if "actual_data_source" in working else "unknown",
        })

        self.manifest.variants[variant_name] = {
            "feature_columns": FEATURE_COLS,
            "feature_indices": list(range(len(FEATURE_COLS))),
            "distance_construction": "mahalanobis_calibrated → CUSUM (production reuse)",
            "cusum_parameters": {
                "slack_k": cfg.cusum_slack_k,
                "threshold_h": cfg.cusum_threshold_h,
                "reference_mean": cfg.cusum_reference_mean,
                "reference_std": cfg.cusum_reference_std,
            },
            "normalization": cfg.normalization_policy,
            "availability": cfg.availability_policy,
            "validation_threshold": float(full_threshold) if np.isfinite(full_threshold) else None,
        }

        # --- Subset variants ---
        for group_name, feature_indices in ABLATION_GROUPS.items():
            subset_cols = [FEATURE_COLS[i] for i in feature_indices]
            score_col = f"_ablation_score_{group_name}"
            cusum_col = f"_ablation_cusum_{group_name}"

            # Compute subset anomaly score
            if self.baseline_store is not None:
                scores, avail = self._compute_subset_mahalanobis(
                    working, feature_indices, group_name,
                )
                distance_method = "subset_mahalanobis"
            else:
                scores, avail = self._euclidean_fallback(
                    working, feature_indices, group_name,
                )
                distance_method = "euclidean_zscore (no baseline store)"

            working[score_col] = scores
            working[f"_ablation_avail_{score_col}"] = avail

            # Run CUSUM on subset scores
            working = self._run_cusum_on_scores(
                working,
                score_col=score_col,
                cusum_output_col=cusum_col,
                slack_k=cfg.cusum_slack_k,
                threshold_h=cfg.cusum_threshold_h,
                reference_mean=cfg.cusum_reference_mean,
                reference_std=cfg.cusum_reference_std,
                calibration_pitches=cfg.calibration_pitches,
            )

            # Mark CUSUM availability
            working[f"_ablation_avail_{cusum_col}"] = working[cusum_col].notna()

            # Select threshold on validation
            threshold, val_metrics = select_operating_threshold(
                working, self.episodes, cusum_col,
                cfg.false_warnings_per_outing,
                horizon_pitches=cfg.horizon_pitches,
                split="validation",
            )

            # Evaluate on test
            predictions = working[cusum_col].ge(threshold) & working[cusum_col].notna()
            test_metrics, _, _ = evaluate_warning_predictions(
                working, self.episodes, predictions,
                horizon_pitches=cfg.horizon_pitches, split="test",
            )

            # PR-AUC on test
            part = working.loc[test_mask, [cusum_col, "y_true_onset_in_horizon"]].dropna()
            if not part.empty and part["y_true_onset_in_horizon"].nunique() > 1:
                precision_arr, recall_arr, _ = precision_recall_curve(
                    part["y_true_onset_in_horizon"].astype(int), part[cusum_col]
                )
                pr_auc = float(auc(recall_arr, precision_arr))
            else:
                pr_auc = np.nan

            # Coverage
            subset_avail = working.loc[test_mask, f"_ablation_avail_{score_col}"]
            pitch_cov = float(subset_avail.sum() / len(test_pitches)) if len(test_pitches) else 0.0
            scored_out = test_pitches.loc[subset_avail, ["game_pk", "pitcher"]].drop_duplicates()
            outing_cov = float(len(scored_out) / len(full_outing_keys)) if len(full_outing_keys) else 0.0

            results_rows.append({
                "Feature Subset": group_name,
                "Score Method": f"{distance_method} → CUSUM",
                "Validation-Selected Threshold": (
                    round(threshold, 4) if np.isfinite(threshold) else np.nan
                ),
                "Validation Recall": round(val_metrics.get("episode_recall", 0.0), 4),
                "Validation Precision": round(val_metrics.get("warning_precision", 0.0), 4),
                "Validation False Warnings / Outing": round(
                    val_metrics.get("false_warnings_per_outing", 0.0), 4
                ),
                "Allowed Max FW/Outing": cfg.false_warnings_per_outing,
                "Test Episode Recall": round(test_metrics["episode_recall"], 4),
                "Test Warning Precision": round(test_metrics["warning_precision"], 4),
                "Test False Warnings / Outing": round(test_metrics["false_warnings_per_outing"], 4),
                "Test Pitch PR-AUC": round(pr_auc, 4) if np.isfinite(pr_auc) else np.nan,
                "Test Evaluated Outings": test_metrics["evaluated_outings_count"],
                "Test Evaluated Episodes": test_metrics["total_collapse_episodes"],
                "Test Warnings": test_metrics["total_warnings"],
                "Test False Warnings": test_metrics["false_warnings"],
                "Test Pitch Coverage": round(pitch_cov, 4),
                "Test Outing Coverage": round(outing_cov, 4),
                "Actual Data Source": working.get(
                    "actual_data_source", pd.Series(["unknown"])
                ).dropna().iloc[0] if "actual_data_source" in working else "unknown",
            })

            self.manifest.variants[group_name] = {
                "feature_columns": subset_cols,
                "feature_indices": feature_indices,
                "distance_construction": distance_method,
                "cusum_parameters": {
                    "slack_k": cfg.cusum_slack_k,
                    "threshold_h": cfg.cusum_threshold_h,
                    "reference_mean": cfg.cusum_reference_mean,
                    "reference_std": cfg.cusum_reference_std,
                },
                "normalization": cfg.normalization_policy,
                "availability": cfg.availability_policy,
                "validation_threshold": float(threshold) if np.isfinite(threshold) else None,
            }

        result = pd.DataFrame(results_rows)
        logger.info("\n=== CORRECTED FEATURE ABLATION RESULTS ===\n%s", result.to_string(index=False))
        return result, self.manifest

    def run_sensitivity_analysis(self) -> pd.DataFrame:
        """Re-run label definitions and evaluate under different thresholds.

        This method is unchanged in its approach: it re-labels and evaluates
        using cusum_stat, which is the correct production score column.
        """
        cfg = self.config
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
                rerun_df, rerun_episodes, "cusum_stat", cfg.false_warnings_per_outing,
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
