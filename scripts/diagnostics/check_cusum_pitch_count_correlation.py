"""Explore CUSUM accumulation using 2023 training and 2024 validation only.

Usage: python scripts/diagnostics/check_cusum_pitch_count_correlation.py --variant all

Reads the existing real-data Parquet warehouse without opening or changing DuckDB.
Fits a separate reference for each feature variant from finite, available,
post-calibration TRAIN scores. Follow-up censoring is deliberately irrelevant to
reference fitting: it governs outcome evaluation, not detector input eligibility.
Selects original and recalibrated thresholds on validation with the production
warning evaluator and false-warning allowance. These are development results,
not an independent performance estimate. No 2025 rows are loaded or evaluated.

The shuffle control permutes available post-calibration scores within each outing
and pitch type, preserving score distributions, workload, pitch-type sequence,
and unavailable positions. Each control uses the corresponding unshuffled
validation threshold, without retuning. It disrupts mechanical timing but cannot
remove information in outing-level score distributions. Its ranges are descriptive
shuffle ranges, not confidence intervals or formal significance tests.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configuration import load_project_config
from src.evaluation.ablation_runner import ABLATION_GROUPS, FEATURE_COLS, AblationConfig, AblationRunner
from src.evaluation.protocol import (
    eligible_pitch_mask, evaluate_warning_predictions, select_operating_threshold,
)

logger = logging.getLogger(__name__)
VARIANTS = {
    "full": "Full Micro-Mechanics Suite",
    "velocity": "Velocity Alone",
    "release_point": "Release Point Alone (X, Z, Extension)",
    "spin_movement": "Spin & Movement (rate, axis, PFX, VAA)",
}
METRICS = ("episode_recall", "warning_precision", "false_warnings_per_outing")
EVALUATION_COLUMNS = [
    "game_pk", "pitcher", "pitch_number_in_outing", "pa_number_in_outing",
    "game_date", "dataset_split", "actual_data_source", "is_calibration_phase",
    "is_censored_followup", "score_available",
]


def scoring_mask(df: pd.DataFrame, score_col: str, calibration_pitches: int) -> pd.Series:
    """Detector inputs, including pitches censored only from outcome evaluation."""
    mask = np.isfinite(pd.to_numeric(df[score_col], errors="coerce"))
    mask &= df["pitch_number_in_outing"].gt(calibration_pitches)
    mask &= ~df["is_calibration_phase"].fillna(True).astype(bool)
    mask &= df["score_available"].fillna(False).astype(bool)
    return mask


def estimate_reference_from_train(
    df: pd.DataFrame, score_col: str, calibration_pitches: int,
) -> dict[str, Any]:
    mask = scoring_mask(df, score_col, calibration_pitches) & df["dataset_split"].eq("train")
    vals = pd.to_numeric(df.loc[mask, score_col], errors="coerce")
    if len(vals) < 30:
        raise ValueError(f"Only {len(vals)} eligible train scores; at least 30 are required")
    mean, std = float(vals.mean()), float(vals.std(ddof=1))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 1e-6:
        raise ValueError("Training reference must have finite mean and nonzero standard deviation")
    return {
        "mean": mean, "std": std, "n_pitches": len(vals),
        "n_outings": len(df.loc[mask, ["game_pk", "pitcher"]].drop_duplicates()),
        "weighting": "equal weight per available post-calibration train pitch",
        "followup_censored_pitches_included": True,
    }


def correlations(df: pd.DataFrame, score_col: str, split: str) -> dict[str, Any]:
    availability = np.isfinite(pd.to_numeric(df[score_col], errors="coerce"))
    availability &= df["score_available"].fillna(False).astype(bool)
    mask = eligible_pitch_mask(df, split, availability=availability)
    sub = df.loc[mask, ["game_pk", "pitcher", "pitch_number_in_outing", score_col]]
    pooled = np.nan
    if len(sub) >= 10 and sub[score_col].nunique() > 1 and sub.pitch_number_in_outing.nunique() > 1:
        pooled = float(spearmanr(sub[score_col], sub.pitch_number_in_outing).statistic)
    rhos = []
    excluded = 0
    for _, outing in sub.groupby(["game_pk", "pitcher"], sort=False):
        if len(outing) < 10 or outing[score_col].nunique() < 3:
            excluded += 1
            continue
        rho = float(spearmanr(outing[score_col], outing.pitch_number_in_outing).statistic)
        if np.isfinite(rho):
            rhos.append(rho)
        else:
            excluded += 1
    return {
        "n_pitches": len(sub), "pooled_rho": pooled,
        "n_outings": len(rhos), "excluded_outings": excluded,
        "median_within_outing_rho": float(np.median(rhos)) if rhos else np.nan,
        "p25": float(np.quantile(rhos, .25)) if rhos else np.nan,
        "p75": float(np.quantile(rhos, .75)) if rhos else np.nan,
        "fraction_above_0_9": float(np.mean(np.asarray(rhos) > .9)) if rhos else np.nan,
    }


def run_cusum(
    df: pd.DataFrame, score_col: str, cfg: AblationConfig, mean: float, std: float,
) -> pd.DataFrame:
    working = df.copy()
    working[f"_ablation_avail_{score_col}"] = scoring_mask(df, score_col, cfg.calibration_pitches)
    return AblationRunner._run_cusum_on_scores(
        working, score_col=score_col, cusum_output_col="_diagnostic_cusum",
        slack_k=cfg.cusum_slack_k, threshold_h=cfg.cusum_threshold_h,
        reference_mean=mean, reference_std=std, calibration_pitches=cfg.calibration_pitches,
    )


def evaluate_at_threshold(
    df: pd.DataFrame, episodes: pd.DataFrame, score_col: str,
    threshold: float, cfg: AblationConfig,
) -> dict[str, Any]:
    df = df.loc[df.dataset_split.eq("validation"), list(dict.fromkeys([*EVALUATION_COLUMNS, score_col]))]
    available = np.isfinite(pd.to_numeric(df[score_col], errors="coerce"))
    predictions = df[score_col].ge(threshold) & available
    metrics, _, _ = evaluate_warning_predictions(
        df, episodes, predictions, horizon_pitches=cfg.horizon_pitches,
        split="validation", availability=available,
    )
    return metrics


def select_and_evaluate(
    df: pd.DataFrame, episodes: pd.DataFrame, score_col: str, cfg: AblationConfig,
) -> tuple[float, dict[str, Any]]:
    df = df.loc[df.dataset_split.eq("validation"), list(dict.fromkeys([*EVALUATION_COLUMNS, score_col]))]
    threshold, metrics = select_operating_threshold(
        df, episodes, score_col, cfg.false_warnings_per_outing,
        horizon_pitches=cfg.horizon_pitches, split="validation",
    )
    return threshold, {
        "selected_threshold": threshold,
        "threshold_status": "selected" if np.isfinite(threshold) else "no_alert",
        "validation_metrics": metrics,
    }


def shuffle_scores(
    df: pd.DataFrame, score_col: str, calibration_pitches: int, seed: int,
) -> pd.DataFrame:
    """Leave calibration, missingness, pitch types and labels in their original positions."""
    out = df.copy()
    eligible = scoring_mask(df, score_col, calibration_pitches)
    rng = np.random.default_rng(seed)
    for _, group in df.loc[eligible].groupby(["game_pk", "pitcher", "pitch_type"], sort=True):
        ordered = group.sort_values("pitch_number_in_outing")
        out.loc[ordered.index, score_col] = rng.permutation(ordered[score_col].to_numpy())
    return out


def shuffle_summary(replicates: list[dict[str, Any]], observed: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for metric in METRICS:
        vals = np.asarray([r[metric] for r in replicates], dtype=float)
        vals = vals[np.isfinite(vals)]
        actual = observed[metric]
        summary[metric] = {
            "n_defined": len(vals), "observed": actual,
            "median": float(np.median(vals)) if len(vals) else np.nan,
            "p025": float(np.quantile(vals, .025)) if len(vals) else np.nan,
            "p975": float(np.quantile(vals, .975)) if len(vals) else np.nan,
            "fraction_at_least_observed": (
                float(np.mean(vals >= actual)) if len(vals) and np.isfinite(actual) else np.nan
            ),
        }
    return summary


def compare_variant(
    df: pd.DataFrame, episodes: pd.DataFrame, score_col: str,
    cfg: AblationConfig, shuffle_seeds: list[int], check_persisted: bool = False,
) -> dict[str, Any]:
    reference = estimate_reference_from_train(df, score_col, cfg.calibration_pitches)
    # CUSUM resets per outing. Training contributes reference values only.
    columns = list(dict.fromkeys([*EVALUATION_COLUMNS, "pitch_type", score_col] +
                                (["cusum_stat"] if check_persisted else [])))
    validation = df.loc[df.dataset_split.eq("validation"), columns].copy()
    references = {
        "original": (cfg.cusum_reference_mean, cfg.cusum_reference_std),
        "recalibrated": (reference["mean"], reference["std"]),
    }
    report: dict[str, Any] = {
        "empirical_train_reference": reference,
        "raw_score_correlation": {s: correlations(df, score_col, s) for s in ("train", "validation")},
    }
    thresholds = {}
    for name, (mean, std) in references.items():
        scored = run_cusum(validation, score_col, cfg, mean, std)
        if name == "original" and check_persisted:
            if not np.allclose(scored._diagnostic_cusum, scored.cusum_stat, rtol=0, atol=1e-12, equal_nan=True):
                raise ValueError("Recomputed original CUSUM differs from persisted production scores")
            report["production_cusum_parity"] = "verified on every validation pitch"
        threshold, result = select_and_evaluate(scored, episodes, "_diagnostic_cusum", cfg)
        thresholds[name] = threshold
        result["reference"] = {"mean": mean, "std": std}
        result["cusum_pitch_count_correlation"] = correlations(scored, "_diagnostic_cusum", "validation")
        mask = scoring_mask(validation, score_col, cfg.calibration_pitches)
        increments = (validation.loc[mask, score_col] - mean) / std - cfg.cusum_slack_k
        result["fraction_positive_increments"] = float((increments > 0).mean())
        result["mean_increment_before_zero_floor"] = float(increments.mean())
        report[name] = result
        logger.info("%s: threshold=%s; recall=%.4f; precision=%.4f; false warnings/outing=%.4f",
                    name, threshold, *(result["validation_metrics"][m] for m in METRICS))
    replicates: dict[str, list[dict[str, Any]]] = {name: [] for name in references}
    for i, seed in enumerate(shuffle_seeds):
        shuffled = shuffle_scores(validation, score_col, cfg.calibration_pitches, seed)
        for name, (mean, std) in references.items():
            scored = run_cusum(shuffled, score_col, cfg, mean, std)
            metrics = evaluate_at_threshold(scored, episodes, "_diagnostic_cusum", thresholds[name], cfg)
            replicates[name].append({"seed": seed, **metrics})
        logger.info("Shuffle %d/%d complete", i + 1, len(shuffle_seeds))
    report["shuffle_control"] = {
        "grouping": ["game_pk", "pitcher", "pitch_type"],
        "threshold_policy": "fixed unshuffled validation threshold for each reference; never reselected",
        "interpretation": "Descriptive timing control; preserves outing distributions and workload. Ranges are not confidence intervals or formal p-values.",
        "results": {name: {
            "summary": shuffle_summary(rows, report[name]["validation_metrics"]),
            "replicates": rows,
        } for name, rows in replicates.items()},
    }
    report["recalibrated_minus_original"] = {
        m: report["recalibrated"]["validation_metrics"][m] - report["original"]["validation_metrics"][m]
        for m in METRICS
    }
    return report


def _load_parquet(root: Path, relative: str, predicate: Any) -> pd.DataFrame:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing local warehouse file: {path}. Run the real-data pipeline first.")
    return ds.dataset(path, format="parquet").to_table(filter=predicate).to_pandas()


def _validate_provenance(df: pd.DataFrame, manifest: dict[str, Any], name: str) -> None:
    if df.empty:
        raise ValueError(f"{name} contains no train/validation rows")
    for column in ("run_id", "protocol_sha256", "actual_data_source"):
        if column not in df or df[column].isna().any() or set(df[column].unique()) != {manifest[column]}:
            raise ValueError(f"{name} {column} does not match the published protocol manifest")
    dates = pd.to_datetime(df.game_date)
    config = manifest["resolved_runtime"]["resolved_config"]["evaluation"]
    valid = ((df.dataset_split.eq("train") & dates.le(config["train_end"])) |
             (df.dataset_split.eq("validation") & dates.gt(config["train_end"]) & dates.le(config["validation_end"])))
    if not valid.all():
        raise ValueError(f"{name} contains dates inconsistent with the train/validation protocol")


def episodes_from_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Recover production episode boundaries from persisted active-episode flags.

    fact_collapse_labels contains pitch rows, not episode records. An episode is
    one contiguous run of is_collapse_event in an outing. No label thresholds or
    outcomes are changed here; verify against the stored per-outing episode count.
    """
    keys = ["game_pk", "pitcher"]
    identity = keys + ["pitch_number_in_outing"]
    if labels.duplicated(identity).any():
        raise ValueError("Duplicate pitch identities in collapse labels")
    ordered = labels.sort_values(identity).copy()
    if ordered.is_collapse_event.isna().any():
        raise ValueError("Missing active-episode flags in persisted collapse labels")
    active = ordered.is_collapse_event.astype(bool)
    previous = active.groupby([ordered[k] for k in keys]).shift(fill_value=False)
    starts = active & ~previous
    ordered["episode_id"] = starts.groupby([ordered[k] for k in keys]).cumsum()
    counts = ordered.groupby(keys).collapse_episodes_count_in_game
    expected = counts.first()
    actual = starts.groupby([ordered[k] for k in keys]).sum()
    if (counts.nunique() != 1).any() or not actual.eq(expected).all():
        raise ValueError("Reconstructed episodes do not match persisted per-outing episode counts")
    return ordered.loc[active].groupby(keys + ["episode_id"], as_index=False).agg(
        onset_pitch=("pitch_number_in_outing", "first"),
        end_pitch=("pitch_number_in_outing", "last"),
        onset_pa=("pa_number_in_outing", "first"),
        end_pa=("pa_number_in_outing", "last"),
        game_date=("game_date", "first"),
        dataset_split=("dataset_split", "first"),
        actual_data_source=("actual_data_source", "first"),
    )


def run_check(
    variant: str = "full", shuffles: int = 20, seed: int = 20261001,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    if variant not in {*VARIANTS, "all"} or shuffles < 1 or seed < 0:
        raise ValueError("Choose a supported variant, at least one shuffle, and a nonnegative seed")
    config = load_project_config(root / "config/config.yaml")
    manifest_path = root / "outputs/real_data/protocol_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Published real-data protocol manifest is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["actual_data_source"] != "mlb_statcast":
        raise ValueError("This diagnostic requires the real Statcast warehouse")
    if config != manifest["resolved_runtime"]["resolved_config"]:
        raise ValueError("Current configuration differs from the persisted run; restore its configuration first")
    predicate = ds.field("dataset_split").isin(["train", "validation"])
    scored = _load_parquet(root, "data/gold/fact_pitch_anomaly_scores.parquet", predicate)
    labels = _load_parquet(root, "data/gold/fact_collapse_labels.parquet", predicate)
    _validate_provenance(scored, manifest, "scores")
    _validate_provenance(labels, manifest, "labels")
    identity = ["game_pk", "pitcher", "pitch_number_in_outing"]
    if (scored.duplicated(identity).any() or labels.duplicated(identity).any() or
            set(map(tuple, scored[identity].to_numpy())) != set(map(tuple, labels[identity].to_numpy()))):
        raise ValueError("Score and collapse-label pitch identities do not match")
    episodes = episodes_from_labels(labels)
    cfg = AblationConfig(
        cusum_slack_k=float(config["changepoint"]["cusum_slack_k"]),
        cusum_threshold_h=float(config["changepoint"]["cusum_internal_alert_h"]),
        cusum_reference_mean=float(config["changepoint"]["cusum_reference_mean"]),
        cusum_reference_std=float(config["changepoint"]["cusum_reference_std"]),
        calibration_pitches=int(config["baseline"]["intra_game_calibration_pitches"]),
        horizon_pitches=int(config["evaluation"]["warning_matching_horizon_pitches"]),
        false_warnings_per_outing=float(config["evaluation"]["false_warnings_per_outing"]),
        ridge_regularization=float(config["anomaly"]["ridge_regularization"]),
        shrinkage_lambda=float(config["baseline"]["shrinkage_lambda"]),
    )
    variants = list(VARIANTS) if variant == "all" else [variant]
    baseline_store = None
    if any(v != "full" for v in variants):
        from src.baseline_builder.historical_baseline import BaselineBuilder
        from src.feature_engineering.mechanics_features import compute_kinematics_and_vaa
        from src.feature_engineering.rolling_stats import compute_rolling_features
        logger.info("Rebuilding historical subset baselines from pre-2025 qualified pitches")
        cutoff = pa.scalar(pd.Timestamp(config["evaluation"]["validation_end"]), type=pa.timestamp("ns"))
        qualified = _load_parquet(root, "data/silver/stg_qualified_pitches.parquet",
                                  ds.field("game_date").cast(pa.timestamp("ns")) <= cutoff)
        if qualified.empty or set(qualified.actual_data_source.dropna().unique()) != {"mlb_statcast"}:
            raise ValueError("Qualified train/validation Statcast pitches are required for subsets")
        features = compute_rolling_features(compute_kinematics_and_vaa(qualified))
        if set(map(tuple, features[identity].to_numpy())) != set(map(tuple, scored[identity].to_numpy())):
            raise ValueError("Qualified silver pitches do not match the scored gold population")
        baseline = config["baseline"]
        _, baseline_store = BaselineBuilder(
            historical_window_starts=int(baseline["historical_window_starts"]),
            min_prior_starts=int(baseline["min_prior_starts"]),
            min_pitches_for_baseline=int(baseline["min_pitches_for_baseline"]),
            ridge_reg=cfg.ridge_regularization,
        ).build_baselines(features)
    columns = list(dict.fromkeys([*EVALUATION_COLUMNS, "pitch_type", "mahalanobis_calibrated", "cusum_stat"] +
                                (FEATURE_COLS if baseline_store is not None else [])))
    scored = scored.loc[:, columns].copy()
    runner = AblationRunner(scored, episodes, config=cfg, baseline_store=baseline_store)
    shuffle_seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(shuffles)]
    source_paths = [Path(__file__), PROJECT_ROOT / "src/evaluation/ablation_runner.py",
                    PROJECT_ROOT / "src/evaluation/protocol.py", PROJECT_ROOT / "src/baseline_builder/historical_baseline.py",
                    PROJECT_ROOT / "src/feature_engineering/mechanics_features.py", PROJECT_ROOT / "src/feature_engineering/rolling_stats.py"]
    report: dict[str, Any] = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_run_id": manifest["run_id"], "parent_protocol_sha256": manifest["protocol_sha256"],
        "actual_data_source": manifest["actual_data_source"],
        "prior_2025_exposure_disclosure": manifest["prior_2025_exposure_disclosure"],
        "config_used": asdict(cfg),
        "experiment": {
            "variants": variants, "seed": seed, "shuffle_seeds": shuffle_seeds,
            "reference_fit_split": "train", "threshold_selection_split": "validation",
            "evaluated_split": "validation", "test_rows_loaded": False,
            "episode_source": "Runs of persisted is_collapse_event; counts verified against collapse_episodes_count_in_game",
            "interpretation": "Exploratory development comparison on the threshold-selection split; no independent performance claim. Freeze any later test experiment before evaluating 2025.",
            "source_sha256": {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
            "git_sha": runner.manifest.git_sha, "git_dirty": runner.manifest.git_dirty,
        },
        "variants": {},
    }
    count_df = scored.assign(_diagnostic_pitch_count=scored.pitch_number_in_outing.astype(float))
    _, report["pitch_count_baseline"] = select_and_evaluate(count_df, episodes, "_diagnostic_pitch_count", cfg)
    for key in variants:
        logger.info("Checking variant: %s", key)
        working = scored.copy()
        score_col = "mahalanobis_calibrated"
        if key != "full":
            score_col = "_diagnostic_raw"
            scores, availability = runner._compute_subset_mahalanobis(working, ABLATION_GROUPS[VARIANTS[key]])
            working[score_col] = scores
            working["score_available"] = availability
        report["variants"][key] = {
            "label": VARIANTS[key],
            **compare_variant(working, episodes, score_col, cfg, shuffle_seeds, check_persisted=key == "full"),
        }
    return json_safe(report)


def json_safe(value: Any) -> Any:
    """Keep undefined values as JSON null, never NaN/Infinity or numeric strings."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="full")
    parser.add_argument("--shuffles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/diagnostics/cusum_pitch_count_check.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    report = run_check(args.variant, args.shuffles, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    logger.info("Wrote diagnostic report to %s", args.out)


if __name__ == "__main__":
    main()
