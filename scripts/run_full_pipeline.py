"""End-to-end MLB mechanics evaluation with temporal holdout and provenance."""
from __future__ import annotations

import json
import hashlib
import logging
import os
import sys
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
    evaluate_warning_predictions,
    warning_events,
)
from src.feature_engineering.mechanics_features import compute_kinematics_and_vaa
from src.feature_engineering.rolling_stats import compute_rolling_features
from src.label_builder.collapse_labels import CollapseLabelBuilder
from src.reporting import render_validation_report
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
            metrics, _, _ = evaluate_warning_predictions(
                pitcher_df, pitcher_episodes, pitcher_df[prediction_col],
                horizon_pitches=horizon_pitches, split="test",
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
    config: dict,
    actual_source: str,
    frozen_thresholds: dict,
) -> dict:
    frozen_settings = {
        "ingestion": config["ingestion"],
        "qualify": config["qualify"],
        "baseline": config["baseline"],
        "features": config["features"],
        "anomaly": config["anomaly"],
        "changepoint": config["changepoint"],
        "labels": config["labels"],
        "evaluation": config["evaluation"],
        "frozen_validation_thresholds": frozen_thresholds,
    }
    canonical = json.dumps(_json_ready(frozen_settings), sort_keys=True, separators=(",", ":"))
    manifest = {
        "protocol_name": "2023_train_2024_validation_2025_test",
        "protocol_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_data_source": actual_source,
        "baseline_update_policy": (
            "For every outing, use at most the 12 most recent qualified outings with dates "
            "strictly before the scoring date; 2025 baselines may update only from completed prior outings."
        ),
        "threshold_policy": "Select on 2024 only; keep fixed for every 2025 result and bootstrap replicate.",
        "prior_2025_exposure_disclosure": (
            "Exploratory 2025 metrics were generated in this workspace before this revised protocol. "
            "The current result is therefore a locked retrospective holdout evaluation, not a pristine first look."
            if actual_source == "mlb_statcast" else "Not applicable to simulation."
        ),
        "frozen_settings": frozen_settings,
    }
    with (output_dir / "protocol_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(manifest), handle, indent=2)
    return manifest


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


def _build_case_studies(df, episodes, output_dir):
    visualizer = CaseStudyVisualizer(output_dir=str(output_dir))
    metrics, warnings, matches = evaluate_warning_predictions(
        df, episodes, df["is_proposed_operating_alert"], horizon_pitches=15, split="test"
    )
    test_df = df[df["dataset_split"].eq("test")]
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
            "lead_time_pitches": None,
        }
        outing_warnings = warnings[(warnings["game_pk"] == game_pk) & (warnings["pitcher"] == pitcher)] if not warnings.empty else pd.DataFrame()
        outing_matches = matches[(matches["game_pk"] == game_pk) & (matches["pitcher"] == pitcher)] if not matches.empty else pd.DataFrame()
        if not outing_matches.empty:
            first = outing_matches.sort_values("warning_pitch").iloc[0]
            evidence.update(warning_pitch=int(first["warning_pitch"]),
                            episode_onset_pitch=int(first["onset_pitch"]),
                            lead_time_pitches=int(first["lead_time_pitches"]))
            note = f"Warning at pitch {first['warning_pitch']}; matched episode at pitch {first['onset_pitch']} ({first['lead_time_pitches']}-pitch lead)."
        elif not outing_warnings.empty:
            pitch = int(outing_warnings["warning_pitch"].min())
            evidence["warning_pitch"] = pitch
            note = f"Warning at pitch {pitch}; no episode onset matched within 15 pitches."
        elif (game_pk, pitcher) in episode_keys:
            eps = episodes[(episodes["game_pk"] == game_pk) & (episodes["pitcher"] == pitcher)].sort_values("onset_pitch")
            onset = int(eps["onset_pitch"].iloc[0])
            evidence["episode_onset_pitch"] = onset
            note = f"Episode onset at pitch {onset}; no warning matched in the preceding 15 pitches."
        else:
            note = "No warning and no evaluable collapse episode in this outing."
        title, filename = definitions[category]
        visualizer.plot_case_study(outing, title, note, filename)
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
    start_dt = start_dt or config["ingestion"]["start_date"]
    end_dt = end_dt or config["ingestion"]["end_date"]
    output_dir = base_path / "outputs" / ("real_data" if use_real_data else "simulation")
    output_dir.mkdir(parents=True, exist_ok=True)
    sm = StorageManager(base_dir=str(base_path))
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
    raw_df = assign_temporal_split(raw_df)
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
        scored_df["mahalanobis_calibrated"].to_numpy()
    )

    cusum, ewma, detected = CUSUMDetector(), EWMADetector(), []
    for _, outing in scored_df.groupby(["pitcher", "game_pk"], sort=False):
        with_cusum, _ = cusum.detect_game_alerts(outing)
        with_ewma, _ = ewma.detect_game_alerts(with_cusum)
        detected.append(with_ewma)
    detected_df = pd.concat(detected, ignore_index=True)

    label_builder = CollapseLabelBuilder(horizon_pitches=15)
    labeled, episode_frames = [], []
    for _, outing in detected_df.groupby(["pitcher", "game_pk"], sort=False):
        outing_labeled, outing_episodes = label_builder.build_labels_for_outing(outing)
        labeled.append(outing_labeled)
        if not outing_episodes.empty:
            episode_frames.append(outing_episodes)
    labeled_df = pd.concat(labeled, ignore_index=True)
    episodes_df = pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame()

    evaluation_config = config["evaluation"]
    comparator = BaselineComparator(
        horizon_pitches=int(evaluation_config["prediction_horizon_pitches"]),
        false_warnings_per_outing=float(evaluation_config["false_warnings_per_outing"]),
    )
    manifest_holder = {}

    def freeze_primary_protocol(thresholds):
        manifest_holder["manifest"] = _write_protocol_manifest(
            output_dir, config, actual_source, thresholds
        )

    comparison_df, evaluated_df, model_metrics = comparator.compare_systems(
        labeled_df, episodes_df, return_details=True,
        freeze_callback=freeze_primary_protocol,
    )
    comparison_df["Actual Data Source"] = actual_source
    comparison_df["Experiment"] = "Primary frozen 2025 test"

    # Preserve the earlier late-2024 analysis as a separately labeled historical
    # experiment. It is never mixed into the primary 2025 comparison.
    historical_labeled = assign_historical_temporal_split(labeled_df)
    historical_episodes = (
        assign_historical_temporal_split(episodes_df) if not episodes_df.empty else episodes_df
    )
    historical_comparator = BaselineComparator(
        horizon_pitches=int(evaluation_config["prediction_horizon_pitches"]),
        false_warnings_per_outing=float(evaluation_config["false_warnings_per_outing"]),
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
            "train": "through 2023-12-31",
            "validation": "2024-01-01 through 2024-12-31",
            "test": "2025-01-01 through 2025-09-30",
        },
        "data_coverage": {
            "requested_start": start_dt,
            "requested_end": end_dt,
            "observed_start": str(pd.to_datetime(raw_df["game_date"]).min().date()),
            "observed_end": str(pd.to_datetime(raw_df["game_date"]).max().date()),
            "seasons_present": required_years,
        },
        "false_alarm_allowance": (
            "0.5 distinct false warnings per validation outing; chosen as an operational "
            "ceiling of approximately one unmatched warning every two starts."
        ),
        "minimum_pitches_per_outing": int(qualify_config["min_pitches_per_game"]),
        "outing_filter_limitation": (
            "The >=50-pitch filter is retrospective because final outing length is unknown live."
        ),
        "cohort_size": int(dim_pitchers["pitcher"].nunique()),
        "test_pitchers_count": test_pitchers_count,
        "preselected_pitchers_count": int(cohort_selection_audit["selected"].sum())
        if not cohort_selection_audit.empty else int(dim_pitchers["pitcher"].nunique()),
        "qualified_outings": int(len(dim_games)),
        "qualified_pitches": int(len(qualified_df)),
        "protocol_sha256": protocol_manifest["protocol_sha256"],
        "prior_2025_exposure_disclosure": protocol_manifest["prior_2025_exposure_disclosure"],
    })

    alerts_df = warning_events(evaluated_df, evaluated_df["is_proposed_operating_alert"])
    if not alerts_df.empty:
        alerts_df = alerts_df.rename(columns={"warning_pitch": "alert_pitch_number"})
        alerts_df["detector_type"] = "CUSUM_VALIDATION_SELECTED"
        alerts_df["operating_threshold"] = metrics_summary["operating_threshold"]

    threshold_curves = comparator.threshold_curves.copy()
    threshold_curves["actual_data_source"] = actual_source
    _plot_threshold_curves(threshold_curves, output_dir / "validation_threshold_tradeoffs.png")

    bootstrap_outing = paired_bootstrap_confidence_intervals(
        evaluated_df, episodes_df, MODEL_PREDICTIONS,
        horizon_pitches=int(evaluation_config["prediction_horizon_pitches"]),
        n_bootstrap=int(evaluation_config["bootstrap_samples"]),
        seed=int(evaluation_config["bootstrap_seed"]),
        cluster_by_pitcher=False,
    )
    bootstrap_pitcher = paired_bootstrap_confidence_intervals(
        evaluated_df, episodes_df, MODEL_PREDICTIONS,
        horizon_pitches=int(evaluation_config["prediction_horizon_pitches"]),
        n_bootstrap=int(evaluation_config["bootstrap_samples"]),
        seed=int(evaluation_config["bootstrap_seed"]),
        cluster_by_pitcher=True,
    )
    bootstrap_results = pd.concat([bootstrap_outing, bootstrap_pitcher], ignore_index=True)
    bootstrap_results["actual_data_source"] = actual_source

    lead_time_sensitivity, matched_records, lead_time_distribution = fixed_warning_horizon_analysis(
        evaluated_df, episodes_df, MODEL_PREDICTIONS,
        horizons=evaluation_config["sensitivity_horizons"], split="test",
    )
    for result in (matched_records, lead_time_distribution):
        if not result.empty and "actual_data_source" not in result:
            result["actual_data_source"] = actual_source

    changepoint_config = config["changepoint"]
    ablation_config = AblationConfig(
        cusum_slack_k=float(changepoint_config["cusum_slack_k"]),
        cusum_threshold_h=float(changepoint_config["cusum_threshold_h"]),
        cusum_reference_mean=1.0,
        cusum_reference_std=0.5,
        calibration_pitches=int(config["baseline"]["intra_game_calibration_pitches"]),
        horizon_pitches=int(evaluation_config["prediction_horizon_pitches"]),
        false_warnings_per_outing=float(evaluation_config["false_warnings_per_outing"]),
        ridge_regularization=float(config["anomaly"]["ridge_regularization"]),
    )
    ablation_runner = AblationRunner(
        evaluated_df, episodes_df,
        config=ablation_config,
        baseline_store=baseline_store,
    )
    ablations, ablation_manifest = ablation_runner.run_feature_ablations()
    ablation_manifest.parent_protocol_hash = protocol_manifest.get("protocol_sha256")
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
        int(evaluation_config["prediction_horizon_pitches"]),
    )

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
    }
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
        "warehouse_integrity_report": "gold",
    }
    sm.save_tables_atomically([
        (frame, name, layer_by_table[name]) for name, frame in frames.items()
    ])
    integrity_report = audit_persisted_warehouse(sm, actual_source, required_years)
    sm.save_tables_atomically([
        (integrity_report, "warehouse_integrity_report", "gold")
    ])

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
    }
    for filename, frame in output_frames.items():
        frame.to_csv(output_dir / filename, index=False)
    case_index = _build_case_studies(evaluated_df, episodes_df, output_dir / "case_studies")

    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(metrics_summary), handle, indent=2)
    with (output_dir / "ablation_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(ablation_manifest.to_dict()), handle, indent=2)
    render_validation_report(output_dir)
    logger.info("Pipeline complete. Requested=%s actual=%s", requested_mode, actual_source)
    return metrics_summary, comparison_df, ablations


if __name__ == "__main__":
    run_pipeline(use_real_data=True)
