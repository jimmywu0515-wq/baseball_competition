"""End-to-end MLB mechanics evaluation with temporal holdout and provenance."""
from __future__ import annotations

import json
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.anomaly_scorer.health_index import compute_mechanics_stability_index
from src.anomaly_scorer.mahalanobis_scorer import MahalanobisScorer
from src.baseline_builder.historical_baseline import BaselineBuilder
from src.baseline_builder.shrinkage_calibrator import ShrinkageCalibrator
from src.changepoint_detector.cusum_detector import CUSUMDetector
from src.changepoint_detector.ewma_detector import EWMADetector
from src.configuration import load_project_config
from src.data_ingest.cohort import resolve_cohort
from src.data_ingest.qualify_filter import QualifyFilter
from src.data_ingest.statcast_loader import StatcastLoader
from src.evaluation.ablation_runner import AblationRunner, AblationConfig
from src.evaluation.baseline_comparator import BaselineComparator
from src.evaluation.bootstrap import paired_bootstrap_confidence_intervals
from src.evaluation.lead_time import fixed_warning_horizon_analysis
from src.evaluation.protocol import (
    assign_historical_temporal_split,
    assign_temporal_split,
    eligible_pitch_mask,
    evaluation_opportunity_mask,
    evaluate_warning_predictions,
    warning_events,
)
from src.feature_engineering.mechanics_features import compute_kinematics_and_vaa
from src.feature_engineering.rolling_stats import compute_rolling_features
from src.label_builder.collapse_labels import CollapseLabelBuilder
from src.reporting import render_validation_report
from src.presentation import validate_release_artifacts
from src.storage.adapter import StorageManager
from src.storage.integrity import (
    audit_persisted_warehouse,
    audit_pipeline_frames,
    prepare_raw_pitch_data,
)
from src.visualization.case_plots import CaseStudyVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Pipeline")


MODEL_PREDICTIONS = {
    "proposed": "is_proposed_operating_alert",
    "contextual": "is_contextual_operating_alert",
    "velocity": "is_velocity_operating_alert",
    "pitch_count": "is_pitch_count_operating_alert",
}

MODEL_AVAILABILITY = {
    "proposed": "score_proposed_cusum",
    "contextual": "score_contextual",
    "velocity": "score_velocity_drop",
    "pitch_count": "score_pitch_count",
}


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def _resolve_algorithm_components(config: dict) -> dict:
    """Build production algorithms and labels from the resolved project config."""
    baseline, labels = config["baseline"], config["labels"]
    evaluation, changepoint = config["evaluation"], config["changepoint"]
    calibration_pitches = int(baseline["intra_game_calibration_pitches"])
    cusum = CUSUMDetector(
        slack_k=float(changepoint["cusum_slack_k"]),
        threshold_h=float(changepoint["cusum_internal_alert_h"]),
        reference_mean=float(changepoint["cusum_reference_mean"]),
        reference_std=float(changepoint["cusum_reference_std"]),
        calibration_pitches=calibration_pitches,
    )
    ewma = EWMADetector(
        ewma_lambda=float(changepoint["ewma_lambda"]),
        l_sigma=float(changepoint["ewma_control_limit_sigma"]),
        reference_mean=float(changepoint["ewma_reference_mean"]),
        reference_std=float(changepoint["ewma_reference_std"]),
        calibration_pitches=calibration_pitches,
    )
    label_builder = CollapseLabelBuilder(
        window_pa_size=int(labels["window_pa_size"]),
        blended_xwoba_threshold=float(labels["blended_xwoba_threshold"]),
        min_barrels_in_window=int(labels["min_barrels_in_window"]),
        min_bb_hbp_in_window=int(labels["min_bb_hbp_in_window"]),
        horizon_pitches=int(labels["prediction_horizon_pitches"]),
        horizon_pas=int(labels["prediction_horizon_pas"]),
    )
    comparator = BaselineComparator(
        velocity_drop_mph=float(evaluation["naive_velocity_drop_mph"]),
        pitch_count_thresh=int(evaluation["naive_pitch_count_threshold"]),
        horizon_pitches=int(evaluation["warning_matching_horizon_pitches"]),
        false_warnings_per_outing=float(evaluation["false_warnings_per_outing"]),
        calibration_pitches=calibration_pitches,
    )
    return {
        "msi_decay_alpha": float(config["anomaly"]["health_index_decay_alpha"]),
        "calibration_pitches": calibration_pitches,
        "cusum": cusum, "ewma": ewma,
        "label_builder": label_builder, "comparator": comparator,
    }


def _missing_data_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "release_speed", "release_pos_x", "release_pos_z", "release_extension",
        "release_spin_rate", "spin_axis", "pfx_x", "pfx_z",
    ]
    rows = []
    dates = pd.to_datetime(raw_df["game_date"])
    for season, season_df in raw_df.assign(_season=dates.dt.year).groupby("_season"):
        for column in columns:
            missing = int(season_df[column].isna().sum())
            rows.append({
                "season": int(season), "column": column,
                "row_count": int(len(season_df)), "missing_count": missing,
                "missing_fraction": float(missing / len(season_df)),
                "actual_data_source": season_df["actual_data_source"].iloc[0],
            })
    return pd.DataFrame(rows)


def _pitcher_model_evaluation(
    evaluated_df: pd.DataFrame, episodes_df: pd.DataFrame, actual_source: str,
    horizon_pitches: int,
) -> pd.DataFrame:
    """Expose between-pitcher heterogeneity under the same frozen predictions."""
    rows = []
    test = evaluated_df[evaluated_df["dataset_split"].eq("test")]
    for pitcher, pitcher_df in test.groupby("pitcher", sort=True):
        pitcher_episodes = episodes_df[
            episodes_df["pitcher"].eq(pitcher) & episodes_df["dataset_split"].eq("test")
        ] if not episodes_df.empty else pd.DataFrame()
        name = pitcher_df["pitcher_name"].dropna().iloc[0] if pitcher_df["pitcher_name"].notna().any() else str(pitcher)
        for model_key, prediction_col in MODEL_PREDICTIONS.items():
            availability = pd.Series(
                np.isfinite(pd.to_numeric(pitcher_df[MODEL_AVAILABILITY[model_key]], errors="coerce")),
                index=pitcher_df.index,
            )
            metrics, _, _ = evaluate_warning_predictions(
                pitcher_df, pitcher_episodes, pitcher_df[prediction_col],
                horizon_pitches=horizon_pitches, split="test",
                availability=availability,
            )
            rows.append({
                "pitcher": int(pitcher), "pitcher_name": name, "model_key": model_key,
                "evaluated_outings": metrics["evaluated_outings_count"],
                "evaluated_pitches": metrics["evaluated_pitches_count"],
                "collapse_episodes": metrics["total_collapse_episodes"],
                "episode_recall": metrics["episode_recall"],
                "warning_precision": metrics["warning_precision"],
                "false_warnings_per_outing": metrics["false_warnings_per_outing"],
                "actual_data_source": actual_source,
            })
    return pd.DataFrame(rows)


def _write_protocol_manifest(
    output_dir: Path,
    actual_source: str,
    frozen_thresholds: dict,
    validation_metrics: dict,
    resolved_runtime: dict,
    run_id: str,
) -> dict:
    selected_models = {}
    for model_key, threshold in frozen_thresholds.items():
        metrics = validation_metrics.get(model_key, {})
        selected_models[model_key] = {
            "threshold_status": "no_alert" if not np.isfinite(threshold) else "selected",
            "validation_selected_operating_threshold": float(threshold) if np.isfinite(threshold) else None,
            "validation_episode_recall": metrics.get("episode_recall"),
            "validation_warning_precision": metrics.get("warning_precision"),
            "validation_false_warnings_per_outing": metrics.get("false_warnings_per_outing"),
            "allowed_maximum_false_warnings_per_outing": resolved_runtime["threshold_selection"]["allowed_maximum_false_warnings_per_outing"],
        }
    resolved = {**resolved_runtime, "selected_models": selected_models}
    canonical_fields = {key: value for key, value in resolved.items() if key not in {"source_control", "locations"}}
    canonical = json.dumps(_json_ready(canonical_fields), sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "protocol_name": (
            f"train_to_{resolved_runtime.get('temporal_protocol', {}).get('training_end', 'unknown')}"
            f"_validation_to_{resolved_runtime.get('temporal_protocol', {}).get('validation_end', 'unknown')}"
            f"_test_from_{resolved_runtime.get('temporal_protocol', {}).get('test_start', 'unknown')}"
        ),
        "protocol_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "protocol_hash_fields": sorted(canonical_fields),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_data_source": actual_source,
        "baseline_update_policy": f"For every outing, use at most {resolved_runtime['baseline']['historical_window_starts']} most recent qualified outings with dates strictly before the scoring date; 2025 baselines may update only from completed prior outings.",
        "threshold_policy": "Select on 2024 only; keep fixed for every 2025 result and bootstrap replicate.",
        "prior_2025_exposure_disclosure": "Exploratory 2025 metrics were generated in this workspace before this revised protocol. The current result is therefore a locked retrospective holdout evaluation, not a pristine first look." if actual_source == "mlb_statcast" else "Not applicable to simulation.",
        "resolved_runtime": resolved,
    }
    with (output_dir / "protocol_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(manifest), handle, indent=2, allow_nan=False)
    return manifest


def _promote_staged_outputs(stage_dir: Path, published_dir: Path, run_id: str) -> None:
    """Validate the completed release, then atomically promote it with rollback."""
    validate_release_artifacts(stage_dir)
    backup_dir = published_dir.with_name(f".{published_dir.name}.{run_id}.previous")
    if published_dir.exists():
        os.replace(published_dir, backup_dir)
    try:
        os.replace(stage_dir, published_dir)
    except Exception:
        if backup_dir.exists():
            os.replace(backup_dir, published_dir)
        raise
    if backup_dir.exists():
        logger.info("Previous published output retained at %s", backup_dir)


def _git_state(base_path: Path) -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=base_path, check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=base_path, check=True, capture_output=True, text=True).stdout.strip())
        return {"commit_sha": sha, "worktree_dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit_sha": None, "worktree_dirty": None}


def _source_digest(base_path: Path) -> str:
    """Hash the code and resolved configuration actually used by this run."""
    digest = hashlib.sha256()
    paths = sorted((project_root / "src").rglob("*.py")) + sorted((project_root / "scripts").rglob("*.py"))
    for path in paths:
        digest.update(path.relative_to(project_root).as_posix().encode())
        digest.update(path.read_bytes())
    digest.update(b"config/config.yaml")
    digest.update((base_path / "config" / "config.yaml").read_bytes())
    return digest.hexdigest()


def _plot_threshold_curves(curves: pd.DataFrame, output_path: Path) -> None:
    if curves.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, group in curves.groupby("Model / System", sort=False):
        ordered = group.sort_values("false_warnings_per_outing")
        ax.plot(
            ordered["false_warnings_per_outing"], ordered["episode_recall"],
            marker=".", alpha=0.7, label=model_name,
        )
        selected = group[group["is_selected_operating_point"]]
        if not selected.empty:
            row = selected.iloc[0]
            ax.scatter(row["false_warnings_per_outing"], row["episode_recall"], marker="*", s=180)
            if pd.notna(row.get("test_false_warnings_per_outing_at_frozen_threshold")):
                ax.scatter(
                    row["test_false_warnings_per_outing_at_frozen_threshold"],
                    row["test_episode_recall_at_frozen_threshold"], marker="X", s=100,
                )
    allowance = float(curves["false_warning_allowance"].iloc[0])
    ax.axvline(allowance, color="black", linestyle="--", label=f"allowance = {allowance:.2f}")
    ax.set(
        xlabel="False warnings per outing",
        ylabel="Episode recall",
        title="2024 validation trade-offs; stars selected, X marks frozen 2025 result",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _build_case_studies(df, episodes, output_dir, horizon_pitches,
                        msi_decay_alpha, calibration_pitches, collapse_xwoba_threshold):
    visualizer = CaseStudyVisualizer(
        output_dir=str(output_dir), msi_decay_alpha=msi_decay_alpha,
        calibration_pitches=calibration_pitches,
        collapse_xwoba_threshold=collapse_xwoba_threshold,
    )
    _, warnings, matches = evaluate_warning_predictions(
        df, episodes, df["is_proposed_operating_alert"], horizon_pitches=horizon_pitches, split="test",
        availability=pd.Series(np.isfinite(pd.to_numeric(df["score_proposed_cusum"], errors="coerce")), index=df.index),
    )
    test_df = df[df["dataset_split"].eq("test")]
    warnings = warnings.loc[warnings["is_evaluable_warning"]].copy()
    episode_keys = set()
    if episodes is not None and not episodes.empty:
        test_eps = episodes[episodes["dataset_split"].eq("test")]
        episode_keys = set(map(tuple, test_eps[["game_pk", "pitcher"]].drop_duplicates().values))
    warning_keys = set()
    if not warnings.empty:
        warning_keys = set(map(tuple, warnings[["game_pk", "pitcher"]].drop_duplicates().values))
    matched_keys = set()
    if not matches.empty:
        matched_keys = set(map(tuple, matches[["game_pk", "pitcher"]].drop_duplicates().values))

    categories = {"TP": None, "FP": None, "FN": None, "TN": None}
    for key, outing in test_df.groupby(["game_pk", "pitcher"], sort=True):
        if key in matched_keys and categories["TP"] is None:
            categories["TP"] = outing
        elif key in warning_keys and key not in matched_keys and categories["FP"] is None:
            categories["FP"] = outing
        elif key in episode_keys and key not in matched_keys and categories["FN"] is None:
            categories["FN"] = outing
        elif key not in warning_keys and key not in episode_keys and categories["TN"] is None:
            categories["TN"] = outing

    definitions = {
        "TP": ("True Positive (Timely Warning)", "case_study_1_true_positive.png"),
        "FP": ("False Positive (Unmatched Warning)", "case_study_2_false_positive.png"),
        "FN": ("False Negative (Missed Episode)", "case_study_3_false_negative.png"),
        "TN": ("True Negative (Clean Outing)", "case_study_4_true_negative.png"),
    }
    evidence_rows = []
    for category, outing in categories.items():
        if outing is None:
            continue
        game_pk = outing["game_pk"].iloc[0]
        pitcher = outing["pitcher"].iloc[0]
        evidence = {
            "case_type": category,
            "game_pk": game_pk,
            "game_date": outing["game_date"].iloc[0],
            "pitcher": pitcher,
            "pitcher_name": outing.get("pitcher_name", pd.Series([pitcher])).iloc[0],
            "actual_data_source": outing.get("actual_data_source", pd.Series(["unknown"])).iloc[0],
            "warning_pitch": None,
            "episode_onset_pitch": None,
            "episode_end_pitch": None,
            "lead_time_pitches": None,
        }
        outing_warnings = warnings[(warnings["game_pk"] == game_pk) & (warnings["pitcher"] == pitcher)] if not warnings.empty else pd.DataFrame()
        outing_matches = matches[(matches["game_pk"] == game_pk) & (matches["pitcher"] == pitcher)] if not matches.empty else pd.DataFrame()
        if not outing_matches.empty:
            first = outing_matches.sort_values("warning_pitch").iloc[0]
            evidence.update(warning_pitch=int(first["warning_pitch"]),
                            episode_onset_pitch=int(first["onset_pitch"]),
                            lead_time_pitches=int(first["lead_time_pitches"]))
            selected_episode = episodes.loc[
                episodes["game_pk"].eq(game_pk) & episodes["pitcher"].eq(pitcher) &
                episodes["onset_pitch"].eq(first["onset_pitch"])
            ]
            if not selected_episode.empty and "end_pitch" in selected_episode:
                evidence["episode_end_pitch"] = int(selected_episode["end_pitch"].iloc[0])
            note = f"Warning at pitch {first['warning_pitch']}; matched episode at pitch {first['onset_pitch']} ({first['lead_time_pitches']}-pitch lead)."
        elif not outing_warnings.empty:
            pitch = int(outing_warnings["warning_pitch"].min())
            evidence["warning_pitch"] = pitch
            note = f"Warning at pitch {pitch}; no episode onset matched within {horizon_pitches} pitches."
        elif (game_pk, pitcher) in episode_keys:
            eps = episodes[(episodes["game_pk"] == game_pk) & (episodes["pitcher"] == pitcher)].sort_values("onset_pitch")
            onset = int(eps["onset_pitch"].iloc[0])
            evidence["episode_onset_pitch"] = onset
            if "end_pitch" in eps:
                evidence["episode_end_pitch"] = int(eps["end_pitch"].iloc[0])
            note = f"Episode onset at pitch {onset}; no warning matched in the preceding {horizon_pitches} pitches."
        else:
            note = "No warning and no evaluable collapse episode in this outing."
        title, filename = definitions[category]
        visualizer.plot_case_study(
            outing, title, note, filename,
            warning_pitch=evidence["warning_pitch"],
            episode_start=evidence["episode_onset_pitch"],
            episode_end=evidence["episode_end_pitch"],
        )
        evidence_rows.append(evidence)

    evidence_df = pd.DataFrame(evidence_rows)
    if not evidence_df.empty:
        evidence_df.to_csv(Path(output_dir) / "case_study_index.csv", index=False)
    return evidence_df


def run_pipeline(
    use_real_data: bool = True,
    base_dir: str = str(project_root),
    start_dt: str = None,
    end_dt: str = None,
):
    base_path = Path(base_dir)
    config = load_project_config(base_path / "config" / "config.yaml")
    components = _resolve_algorithm_components(config)
    start_dt = start_dt or config["ingestion"]["start_date"]
    end_dt = end_dt or config["ingestion"]["end_date"]
    published_dir = base_path / "outputs" / ("real_data" if use_real_data else "simulation")
    run_id = uuid.uuid4().hex
    output_dir = base_path / "outputs" / ".staging" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    sm = StorageManager(base_dir=str(base_path), namespace=None if use_real_data else "simulation")
    loader = StatcastLoader(cache_dir=str(base_path / "data" / "raw"))
    requested_mode = "mlb_statcast" if use_real_data else "simulation_benchmark"
    cohort_selection_audit = pd.DataFrame()
    ingestion_segment_audit = pd.DataFrame()

    if use_real_data:
        cohort_selection_audit = resolve_cohort(
            config["ingestion"], base_path / "data" / "raw" / "cohort"
        )
        selected_cohort = cohort_selection_audit[cohort_selection_audit["selected"]].copy()
        raw_df = loader.fetch_real_pitchers_statcast(
            pitcher_ids=selected_cohort["pitcher"].astype(int).tolist(),
            start_dt=start_dt, end_dt=end_dt, strict=True,
            pitcher_names=selected_cohort.set_index("pitcher")["pitcher_name"].to_dict(),
            verify_empty_seasons=True,
        )
        ingestion_segment_audit = loader.last_segment_audit.copy()
        actual_source = "mlb_statcast"
        required_years = list(range(pd.Timestamp(start_dt).year, pd.Timestamp(end_dt).year + 1))
    else:
        raw_df = loader.generate_simulation_benchmark(num_pitchers=5, starts_per_pitcher=30)
        actual_source = "simulation_benchmark"
        start_dt = str(pd.to_datetime(raw_df["game_date"]).min().date())
        end_dt = str(pd.to_datetime(raw_df["game_date"]).max().date())
        required_years = sorted(pd.to_datetime(raw_df["game_date"]).dt.year.unique().tolist())

    raw_df["actual_data_source"] = actual_source
    raw_df["requested_data_mode"] = requested_mode
    raw_df = prepare_raw_pitch_data(
        raw_df, actual_source, start_dt, end_dt, required_years=required_years
    )
    raw_df = assign_temporal_split(raw_df, config["evaluation"]["train_end"],
                                   config["evaluation"]["validation_end"],
                                   config["evaluation"]["test_start"])
    for audit_frame in (cohort_selection_audit, ingestion_segment_audit):
        audit_frame["actual_data_source"] = actual_source

    qualify_config = config["qualify"]
    qualifier = QualifyFilter(
        min_pitches_per_game=int(qualify_config["min_pitches_per_game"]),
        min_starts_for_cohort=int(qualify_config["min_starts_for_cohort"]),
        cohort_eligibility_end=qualify_config["cohort_eligibility_end"],
        max_missing_mechanics_pct=float(qualify_config["max_missing_mechanics_pct"]),
        top_n_pitch_types=int(qualify_config["top_n_pitch_types"]),
    )
    qualified_df, dim_pitchers, dim_games = qualifier.filter_qualified_games(raw_df)
    if qualified_df.empty:
        raise RuntimeError("No outings passed the documented qualification policy.")

    features = compute_rolling_features(compute_kinematics_and_vaa(qualified_df))
    baseline_config = config["baseline"]
    baseline_builder = BaselineBuilder(
        historical_window_starts=int(baseline_config["historical_window_starts"]),
        min_prior_starts=int(baseline_config["min_prior_starts"]),
        min_pitches_for_baseline=int(baseline_config["min_pitches_for_baseline"]),
        ridge_reg=float(config["anomaly"]["ridge_regularization"]),
    )
    baseline_df, baseline_store = baseline_builder.build_baselines(features)

    calibrator = ShrinkageCalibrator(
        intra_game_calibration_pitches=int(baseline_config["intra_game_calibration_pitches"]),
        shrinkage_lambda=float(baseline_config["shrinkage_lambda"]),
    )
    calibrated, means = [], {}
    for (pitcher, game_pk), outing in features.groupby(["pitcher", "game_pk"], sort=False):
        result, outing_means = calibrator.calibrate_outing_pitches(
            outing, pitcher, game_pk, baseline_store
        )
        calibrated.append(result)
        means[(pitcher, game_pk)] = outing_means
    calibrated_df = pd.concat(calibrated, ignore_index=True)

    scorer = MahalanobisScorer()
    scored = []
    for (pitcher, game_pk), outing in calibrated_df.groupby(["pitcher", "game_pk"], sort=False):
        scored.append(scorer.score_pitches(
            outing, pitcher, game_pk, baseline_store, means[(pitcher, game_pk)]
        ))
    scored_df = pd.concat(scored, ignore_index=True)
    scored_df["health_index"] = compute_mechanics_stability_index(
        scored_df["mahalanobis_calibrated"].to_numpy(), decay_alpha=components["msi_decay_alpha"]
    )

    cusum, ewma, detected = components["cusum"], components["ewma"], []
    for _, outing in scored_df.groupby(["pitcher", "game_pk"], sort=False):
        with_cusum, _ = cusum.detect_game_alerts(outing)
        with_ewma, _ = ewma.detect_game_alerts(with_cusum)
        detected.append(with_ewma)
    detected_df = pd.concat(detected, ignore_index=True)

    label_builder = components["label_builder"]
    labeled, episode_frames = [], []
    for _, outing in detected_df.groupby(["pitcher", "game_pk"], sort=False):
        outing_labeled, outing_episodes = label_builder.build_labels_for_outing(outing)
        labeled.append(outing_labeled)
        if not outing_episodes.empty:
            episode_frames.append(outing_episodes)
    labeled_df = pd.concat(labeled, ignore_index=True)
    episodes_df = pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame()

    evaluation_config = config["evaluation"]
    matching_horizon = int(evaluation_config["warning_matching_horizon_pitches"])
    comparator = components["comparator"]
    manifest_holder = {}
    selected_count = int(cohort_selection_audit["selected"].sum()) if not cohort_selection_audit.empty else int(dim_pitchers["pitcher"].nunique())
    cohort_seed = config["ingestion"].get("cohort_selection", {}).get("seed")
    resolved_runtime = {
        "source_control": _git_state(base_path),
        "source_digest_sha256": _source_digest(base_path),
        "resolved_config": config,
        "cohort": {"identifier": f"pretest_{selected_count}_seed_{cohort_seed}" if cohort_seed is not None else f"{actual_source}_{selected_count}", "selected_size": selected_count, "qualified_size": int(dim_pitchers["pitcher"].nunique()), "selection_seed": cohort_seed, "selection_seasons": config["ingestion"].get("cohort_selection", {}).get("seasons")},
        "random_seeds": {"cohort_selection": cohort_seed, "contextual_logistic_regression": 42, "bootstrap": int(evaluation_config["bootstrap_seed"])},
        "temporal_protocol": {"training_end": evaluation_config["train_end"], "validation_end": evaluation_config["validation_end"], "test_start": evaluation_config["test_start"], "test_end": end_dt},
        "qualification": {"minimum_pitches_per_outing": int(qualify_config["min_pitches_per_game"]), "minimum_pretest_starts": int(qualify_config["min_starts_for_cohort"]), "cohort_eligibility_end": qualify_config["cohort_eligibility_end"], "maximum_missing_mechanics_fraction": float(qualify_config["max_missing_mechanics_pct"]), "top_pitch_types": int(qualify_config["top_n_pitch_types"])},
        "baseline": {"historical_window_starts": int(baseline_config["historical_window_starts"]), "minimum_prior_starts": int(baseline_config["min_prior_starts"]), "minimum_pitch_type_observations": int(baseline_config["min_pitches_for_baseline"]), "calibration_pitches": int(baseline_config["intra_game_calibration_pitches"]), "shrinkage_lambda": float(baseline_config["shrinkage_lambda"])},
        "anomaly_scoring": {"method": config["anomaly"]["method"], "ridge_regularization": float(config["anomaly"]["ridge_regularization"]), "msi_decay_alpha": components["msi_decay_alpha"]},
        "detectors": {"cusum": {"slack_k": cusum.slack_k, "internal_alert_h": cusum.threshold_h, "reference_mean": cusum.reference_mean, "reference_std": cusum.reference_std}, "ewma": {"lambda": ewma.ewma_lambda, "control_limit_sigma": ewma.l_sigma, "reference_mean": ewma.reference_mean, "reference_std": ewma.reference_std}},
        "horizons": {"label_pitches": label_builder.horizon_pitches, "label_plate_appearances": label_builder.horizon_pas, "warning_matching_pitches": matching_horizon, "sensitivity_matching_pitches": list(map(int, evaluation_config["sensitivity_horizons"]))},
        "labels": {"window_pa_size": label_builder.window_pa_size, "blended_xwoba_threshold": label_builder.blended_xwoba_threshold, "min_barrels_in_window": label_builder.min_barrels_in_window, "min_bb_hbp_in_window": label_builder.min_bb_hbp_in_window},
        "bootstrap": {"samples": int(evaluation_config["bootstrap_samples"]), "seed": int(evaluation_config["bootstrap_seed"]), "primary_resampling_unit": "outing", "sensitivity_resampling_unit": "pitcher"},
        "threshold_selection": {"allowed_maximum_false_warnings_per_outing": float(evaluation_config["false_warnings_per_outing"])},
        "locations": {"output_directory": str(published_dir.resolve()), "warehouse": str(Path(sm.db_path).resolve())},
    }

    def freeze_primary_protocol(thresholds):
        manifest_holder["manifest"] = _write_protocol_manifest(
            output_dir, actual_source, thresholds, comparator.validation_metrics,
            resolved_runtime, run_id,
        )

    comparison_df, evaluated_df, model_metrics = comparator.compare_systems(
        labeled_df, episodes_df, return_details=True,
        freeze_callback=freeze_primary_protocol,
    )
    comparison_df["Actual Data Source"] = actual_source
    comparison_df["Experiment"] = "Primary frozen 2025 test"

    # Preserve the earlier late-2024 analysis as a separately labeled historical
    # experiment. It is never mixed into the primary 2025 comparison.
    historical_labeled = assign_historical_temporal_split(labeled_df, **config["historical_experiment"])
    historical_episodes = (
        assign_historical_temporal_split(episodes_df, **config["historical_experiment"]) if not episodes_df.empty else episodes_df
    )
    historical_comparator = BaselineComparator(
        velocity_drop_mph=float(evaluation_config["naive_velocity_drop_mph"]),
        pitch_count_thresh=int(evaluation_config["naive_pitch_count_threshold"]),
        horizon_pitches=matching_horizon,
        false_warnings_per_outing=float(evaluation_config["false_warnings_per_outing"]),
        calibration_pitches=components["calibration_pitches"],
    )
    historical_comparison, _, _ = historical_comparator.compare_systems(
        historical_labeled, historical_episodes, return_details=True
    )
    historical_comparison["Actual Data Source"] = actual_source
    historical_comparison["Experiment"] = "Historical July-December 2024 test"

    protocol_manifest = manifest_holder["manifest"]
    metrics_summary = dict(model_metrics["proposed"])
    test_pitchers_count = int(
        evaluated_df.loc[eligible_pitch_mask(evaluated_df, "test"), "pitcher"].nunique()
    )
    metrics_summary.update({
        "requested_data_mode": requested_mode,
        "actual_data_source": actual_source,
        "split_definition": {
            "train": f"through {evaluation_config['train_end']}",
            "validation": f"after {evaluation_config['train_end']} through {evaluation_config['validation_end']}",
            "test": f"from {evaluation_config['test_start']} through {end_dt}",
        },
        "data_coverage": {
            "requested_start": start_dt,
            "requested_end": end_dt,
            "observed_start": str(pd.to_datetime(raw_df["game_date"]).min().date()),
            "observed_end": str(pd.to_datetime(raw_df["game_date"]).max().date()),
            "seasons_present": required_years,
        },
        "false_alarm_allowance": (
            f"{float(evaluation_config['false_warnings_per_outing'])} distinct false warnings "
            "per validation outing, used as an operational ceiling."
        ),
        "minimum_pitches_per_outing": int(qualify_config["min_pitches_per_game"]),
        "outing_filter_limitation": (
            f"The >={int(qualify_config['min_pitches_per_game'])}-pitch filter is retrospective "
            "because final outing length is unknown live."
        ),
        "cohort_size": int(dim_pitchers["pitcher"].nunique()),
        "test_pitchers_count": test_pitchers_count,
        "preselected_pitchers_count": int(cohort_selection_audit["selected"].sum())
        if not cohort_selection_audit.empty else int(dim_pitchers["pitcher"].nunique()),
        "qualified_outings": int(len(dim_games)),
        "qualified_pitches": int(len(qualified_df)),
        "protocol_sha256": protocol_manifest["protocol_sha256"],
        "run_id": run_id,
        "prior_2025_exposure_disclosure": protocol_manifest["prior_2025_exposure_disclosure"],
    })

    alerts_df = warning_events(
        evaluated_df, evaluated_df["is_proposed_operating_alert"],
        availability=pd.Series(np.isfinite(pd.to_numeric(evaluated_df["score_proposed_cusum"], errors="coerce")), index=evaluated_df.index),
    )
    if not alerts_df.empty:
        alerts_df = alerts_df.rename(columns={"warning_pitch": "alert_pitch_number"})
        alerts_df["detector_type"] = "CUSUM_VALIDATION_SELECTED"
        alerts_df["operating_threshold"] = metrics_summary["operating_threshold"]

    threshold_curves = comparator.threshold_curves.copy()
    threshold_curves["actual_data_source"] = actual_source
    _plot_threshold_curves(threshold_curves, output_dir / "validation_threshold_tradeoffs.png")

    bootstrap_outing = paired_bootstrap_confidence_intervals(
        evaluated_df, episodes_df, MODEL_PREDICTIONS,
        horizon_pitches=matching_horizon,
        n_bootstrap=int(evaluation_config["bootstrap_samples"]),
        seed=int(evaluation_config["bootstrap_seed"]),
        cluster_by_pitcher=False,
        model_availability=MODEL_AVAILABILITY,
    )
    bootstrap_pitcher = paired_bootstrap_confidence_intervals(
        evaluated_df, episodes_df, MODEL_PREDICTIONS,
        horizon_pitches=matching_horizon,
        n_bootstrap=int(evaluation_config["bootstrap_samples"]),
        seed=int(evaluation_config["bootstrap_seed"]),
        cluster_by_pitcher=True,
        model_availability=MODEL_AVAILABILITY,
    )
    bootstrap_results = pd.concat([bootstrap_outing, bootstrap_pitcher], ignore_index=True)
    bootstrap_results["actual_data_source"] = actual_source

    lead_time_sensitivity, matched_records, lead_time_distribution = fixed_warning_horizon_analysis(
        evaluated_df, episodes_df, MODEL_PREDICTIONS,
        horizons=evaluation_config["sensitivity_horizons"], split="test",
        model_availability=MODEL_AVAILABILITY,
    )
    for result in (matched_records, lead_time_distribution):
        if not result.empty and "actual_data_source" not in result:
            result["actual_data_source"] = actual_source

    changepoint_config = config["changepoint"]
    ablation_config = AblationConfig(
        cusum_slack_k=float(cusum.slack_k),
        cusum_threshold_h=float(cusum.threshold_h),
        cusum_reference_mean=float(cusum.reference_mean),
        cusum_reference_std=float(cusum.reference_std),
        calibration_pitches=int(components["calibration_pitches"]),
        horizon_pitches=int(evaluation_config["warning_matching_horizon_pitches"]),
        false_warnings_per_outing=float(evaluation_config["false_warnings_per_outing"]),
        ridge_regularization=float(config["anomaly"]["ridge_regularization"]),
        shrinkage_lambda=float(baseline_config["shrinkage_lambda"]),
    )
    ablation_runner = AblationRunner(
        evaluated_df, episodes_df,
        config=ablation_config,
        baseline_store=baseline_store,
    )
    ablations, ablation_manifest = ablation_runner.run_feature_ablations(
        frozen_threshold=comparator.frozen_thresholds["proposed"],
        expected_metrics=model_metrics["proposed"],
    )
    ablation_manifest.parent_protocol_hash = protocol_manifest.get("protocol_sha256")
    ablation_manifest.run_id = run_id
    sensitivity = AblationRunner(
        evaluated_df, episodes_df, config=ablation_config, baseline_store=baseline_store,
    ).run_sensitivity_analysis()
    for result in (ablations, sensitivity):
        result["Actual Data Source"] = actual_source

    missing_data_summary = _missing_data_summary(raw_df)
    excluded_pitchers = qualifier.last_excluded_pitchers.copy()
    excluded_outings = qualifier.last_excluded_outings.copy()
    for result in (excluded_pitchers, excluded_outings):
        result["actual_data_source"] = actual_source
    unavailable_scores = (
        evaluated_df[~evaluated_df["score_available"].fillna(False)]
        .assign(season=lambda frame: pd.to_datetime(frame["game_date"]).dt.year)
        .groupby(["season", "dataset_split", "score_status", "pitch_type", "actual_data_source"],
                 dropna=False)
        .size().reset_index(name="pitch_count")
    )
    pitcher_model_evaluation = _pitcher_model_evaluation(
        evaluated_df, episodes_df, actual_source,
        matching_horizon,
    )

    coverage_rows = []
    for model_key, row in model_metrics.items():
        coverage_rows.append({
            "model_key": model_key,
            "total_qualified_test_outings": row["total_qualified_outings"],
            "outings_with_available_score": row["outings_with_available_score"],
            "outings_without_available_score": row["outings_without_available_score"],
            "evaluation_opportunity_pitches": row["evaluation_opportunity_pitches"],
            "score_available_evaluation_pitches": row["score_available_evaluation_pitches"],
            "pitch_level_scoring_coverage": row["pitch_level_scoring_coverage"],
            "outing_level_scoring_coverage": row["outing_level_scoring_coverage"],
            "evaluated_episodes": row["evaluated_episodes_count"],
            "episode_recall_denominator": row["headline_denominators"]["episode_recall"],
            "warning_precision_denominator": row["headline_denominators"]["warning_precision"],
            "false_warnings_per_outing_denominator": row["headline_denominators"]["false_warnings_per_outing"],
            "actual_data_source": actual_source,
        })
    evaluation_coverage = pd.DataFrame(coverage_rows)

    frames = {
        "raw_statcast_pitches": raw_df,
        "stg_qualified_pitches": qualified_df,
        "dim_pitchers": dim_pitchers,
        "dim_games": dim_games,
        "feat_pitcher_pitchtype_baseline": baseline_df,
        "feat_pitch_level_features": calibrated_df,
        "fact_pitch_anomaly_scores": evaluated_df,
        "fact_collapse_labels": evaluated_df,
        "fact_alert_events": alerts_df,
        "mart_model_evaluation": comparison_df,
        "mart_historical_2024_evaluation": historical_comparison,
        "mart_threshold_tradeoffs": threshold_curves,
        "mart_bootstrap_confidence_intervals": bootstrap_results,
        "mart_lead_time_sensitivity": lead_time_sensitivity,
        "fact_matched_warning_episodes": matched_records,
        "mart_lead_time_distribution": lead_time_distribution,
        "mart_missing_data_summary": missing_data_summary,
        "audit_excluded_pitchers": excluded_pitchers,
        "audit_excluded_outings": excluded_outings,
        "audit_unavailable_scores": unavailable_scores,
        "audit_cohort_selection": cohort_selection_audit,
        "audit_ingestion_segments": ingestion_segment_audit,
        "mart_pitcher_model_evaluation": pitcher_model_evaluation,
        "mart_evaluation_coverage": evaluation_coverage,
    }
    for frame in frames.values():
        if "actual_data_source" not in frame and "Actual Data Source" not in frame:
            frame["actual_data_source"] = actual_source
    for name, frame in frames.items():
        if name.startswith(("fact_", "mart_", "audit_")):
            frame["run_id"] = run_id
            frame["protocol_sha256"] = protocol_manifest["protocol_sha256"]
    integrity_report = audit_pipeline_frames(frames, actual_source, required_years)
    frames["warehouse_integrity_report"] = integrity_report
    layer_by_table = {
        "raw_statcast_pitches": "raw",
        "stg_qualified_pitches": "silver",
        "dim_pitchers": "silver",
        "dim_games": "silver",
        "feat_pitcher_pitchtype_baseline": "silver",
        "feat_pitch_level_features": "silver",
        "fact_pitch_anomaly_scores": "gold",
        "fact_collapse_labels": "gold",
        "fact_alert_events": "gold",
        "mart_model_evaluation": "gold",
        "mart_historical_2024_evaluation": "gold",
        "mart_threshold_tradeoffs": "gold",
        "mart_bootstrap_confidence_intervals": "gold",
        "mart_lead_time_sensitivity": "gold",
        "fact_matched_warning_episodes": "gold",
        "mart_lead_time_distribution": "gold",
        "mart_missing_data_summary": "gold",
        "audit_excluded_pitchers": "gold",
        "audit_excluded_outings": "gold",
        "audit_unavailable_scores": "gold",
        "audit_cohort_selection": "gold",
        "audit_ingestion_segments": "gold",
        "mart_pitcher_model_evaluation": "gold",
        "mart_evaluation_coverage": "gold",
        "warehouse_integrity_report": "gold",
    }
    output_frames = {
        "model_comparison.csv": comparison_df,
        "historical_2024_model_comparison.csv": historical_comparison,
        "validation_threshold_tradeoffs.csv": threshold_curves,
        "bootstrap_confidence_intervals.csv": bootstrap_results,
        "lead_time_sensitivity.csv": lead_time_sensitivity,
        "matched_warning_episodes.csv": matched_records,
        "lead_time_distribution.csv": lead_time_distribution,
        "missing_data_summary.csv": missing_data_summary,
        "excluded_pitchers.csv": excluded_pitchers,
        "excluded_outings.csv": excluded_outings,
        "unavailable_scores.csv": unavailable_scores,
        "ablation_results.csv": ablations,
        "sensitivity_results.csv": sensitivity,
        "warehouse_integrity_report.csv": integrity_report,
        "cohort_selection.csv": cohort_selection_audit,
        "ingestion_segments.csv": ingestion_segment_audit,
        "pitcher_model_evaluation.csv": pitcher_model_evaluation,
        "evaluation_coverage.csv": evaluation_coverage,
    }
    for filename, frame in output_frames.items():
        frame["run_id"] = run_id
        frame["protocol_sha256"] = protocol_manifest["protocol_sha256"]
        frame.to_csv(output_dir / filename, index=False)
    case_index = _build_case_studies(
        evaluated_df, episodes_df, output_dir / "case_studies", matching_horizon,
        components["msi_decay_alpha"], components["calibration_pitches"],
        float(config["labels"]["blended_xwoba_threshold"]),
    )
    case_index["run_id"] = run_id
    case_index["protocol_sha256"] = protocol_manifest["protocol_sha256"]
    case_index.to_csv(output_dir / "case_studies" / "case_study_index.csv", index=False)

    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(metrics_summary), handle, indent=2, allow_nan=False)
    with (output_dir / "ablation_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(ablation_manifest.to_dict()), handle, indent=2)
    render_validation_report(output_dir)
    validate_release_artifacts(output_dir)
    sm.save_tables_atomically([
        (frame, name, layer_by_table[name]) for name, frame in frames.items()
    ])
    persisted_integrity = audit_persisted_warehouse(sm, actual_source, required_years)
    sm.save_tables_atomically([
        (persisted_integrity, "warehouse_integrity_report", "gold")
    ])
    _promote_staged_outputs(output_dir, published_dir, run_id)
    output_dir = published_dir
    sm.close()
    logger.info("Pipeline complete. Requested=%s actual=%s", requested_mode, actual_source)
    return metrics_summary, comparison_df, ablations


if __name__ == "__main__":
    run_pipeline(use_real_data=True)
