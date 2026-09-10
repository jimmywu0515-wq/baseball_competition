"""End-to-end MLB mechanics evaluation with temporal holdout and provenance."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

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
from src.data_ingest.qualify_filter import QualifyFilter
from src.data_ingest.statcast_loader import REAL_PITCHER_COHORT, StatcastLoader
from src.evaluation.ablation_runner import AblationRunner
from src.evaluation.baseline_comparator import BaselineComparator
from src.evaluation.protocol import (
    assign_temporal_split,
    evaluate_warning_predictions,
    warning_events,
)
from src.feature_engineering.mechanics_features import compute_kinematics_and_vaa
from src.feature_engineering.rolling_stats import compute_rolling_features
from src.label_builder.collapse_labels import CollapseLabelBuilder
from src.storage.adapter import StorageManager
from src.visualization.case_plots import CaseStudyVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Pipeline")


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


def run_pipeline(use_real_data: bool = True, base_dir: str = str(project_root)):
    base_path = Path(base_dir)
    (base_path / "outputs").mkdir(parents=True, exist_ok=True)
    sm = StorageManager(base_dir=str(base_path))
    loader = StatcastLoader(cache_dir=str(base_path / "data" / "raw"))
    requested_mode = "mlb_statcast" if use_real_data else "simulation_benchmark"

    if use_real_data:
        raw_df = loader.fetch_real_pitchers_statcast(
            pitcher_ids=[669203, 554430, 592332, 605400, 657277, 519242],
            start_dt="2023-03-30", end_dt="2024-09-30",
        )
        if raw_df.empty:
            logger.warning("Statcast returned no rows; using an explicitly labeled simulation benchmark.")
            raw_df = loader.generate_simulation_benchmark(num_pitchers=5, starts_per_pitcher=30)
            actual_source = "simulation_benchmark"
        else:
            actual_source = "mlb_statcast"
    else:
        raw_df = loader.generate_simulation_benchmark(num_pitchers=5, starts_per_pitcher=30)
        actual_source = "simulation_benchmark"

    raw_df["actual_data_source"] = actual_source
    raw_df["requested_data_mode"] = requested_mode
    raw_df = assign_temporal_split(raw_df)
    sm.save_table(raw_df, "raw_statcast_pitches", layer="raw")

    qualifier = QualifyFilter(min_pitches_per_game=50, min_starts_per_season=10,
                              max_missing_mechanics_pct=0.05)
    qualified_df, dim_pitchers, dim_games = qualifier.filter_qualified_games(raw_df)
    if qualified_df.empty:
        raise RuntimeError("No outings passed the documented qualification policy.")
    for table, name in [(qualified_df, "stg_qualified_pitches"),
                        (dim_pitchers, "dim_pitchers"), (dim_games, "dim_games")]:
        sm.save_table(table, name, layer="silver")

    features = compute_rolling_features(compute_kinematics_and_vaa(qualified_df))
    baseline_builder = BaselineBuilder(historical_window_starts=12, min_prior_starts=5,
                                       min_pitches_for_baseline=40)
    baseline_df, baseline_store = baseline_builder.build_baselines(features)
    sm.save_table(baseline_df, "feat_pitcher_pitchtype_baseline", layer="silver")

    calibrator = ShrinkageCalibrator(intra_game_calibration_pitches=20, shrinkage_lambda=0.35)
    calibrated, means = [], {}
    for (pitcher, game_pk), outing in features.groupby(["pitcher", "game_pk"], sort=False):
        result, outing_means = calibrator.calibrate_outing_pitches(
            outing, pitcher, game_pk, baseline_store
        )
        calibrated.append(result)
        means[(pitcher, game_pk)] = outing_means
    calibrated_df = pd.concat(calibrated, ignore_index=True)
    sm.save_table(calibrated_df, "feat_pitch_level_features", layer="silver")

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

    comparator = BaselineComparator(false_warnings_per_outing=0.5)
    comparison_df, evaluated_df, model_metrics = comparator.compare_systems(
        labeled_df, episodes_df, return_details=True
    )
    metrics_summary = dict(model_metrics["proposed"])
    metrics_summary.update({
        "requested_data_mode": requested_mode,
        "actual_data_source": actual_source,
        "split_definition": {
            "train": "through 2023-12-31",
            "validation": "2024-01-01 through 2024-06-30",
            "test": "from 2024-07-01",
        },
        "false_alarm_allowance": "0.5 distinct false warnings per validation outing",
        "minimum_pitches_per_outing": 50,
    })

    alerts_df = warning_events(evaluated_df, evaluated_df["is_proposed_operating_alert"])
    if not alerts_df.empty:
        alerts_df = alerts_df.rename(columns={"warning_pitch": "alert_pitch_number"})
        alerts_df["detector_type"] = "CUSUM_VALIDATION_SELECTED"
        alerts_df["operating_threshold"] = metrics_summary["operating_threshold"]

    sm.save_table(evaluated_df, "fact_pitch_anomaly_scores", layer="gold")
    sm.save_table(evaluated_df, "fact_collapse_labels", layer="gold")
    sm.save_table(alerts_df, "fact_alert_events", layer="gold")
    sm.save_table(comparison_df, "mart_model_evaluation", layer="gold")

    ablations = AblationRunner(evaluated_df, episodes_df).run_feature_ablations()
    sensitivity = AblationRunner(evaluated_df, episodes_df).run_sensitivity_analysis()
    for result in (ablations, sensitivity):
        result["Actual Data Source"] = actual_source
    ablations.to_csv(base_path / "outputs" / "ablation_results.csv", index=False)
    sensitivity.to_csv(base_path / "outputs" / "sensitivity_results.csv", index=False)
    case_index = _build_case_studies(evaluated_df, episodes_df, base_path / "outputs" / "case_studies")

    with open(base_path / "outputs" / "metrics_summary.json", "w", encoding="utf-8") as handle:
        json.dump(_json_ready(metrics_summary), handle, indent=2)
    with open(base_path / "outputs" / "validation_report.md", "w", encoding="utf-8") as handle:
        handle.write(
            "# Temporal holdout validation report\n\n"
            f"- Requested mode: `{requested_mode}`\n"
            f"- Actual data source: `{actual_source}`\n"
            "- Train: 2023; threshold validation: January-June 2024; held-out test: July 2024 onward.\n"
            "- Every model uses the same eligible pitches, one-to-one 15-pitch event matching, and "
            "a threshold selected under the same validation allowance of 0.5 distinct false warnings per outing.\n\n"
            "## Held-out model comparison\n\n" + comparison_df.to_markdown(index=False) +
            "\n\n## Feature ablations\n\n" + ablations.to_markdown(index=False) +
            "\n\n## Sensitivity reruns\n\n" + sensitivity.to_markdown(index=False) +
            "\n\n## Traceable case-study index\n\n" +
            (case_index.to_markdown(index=False) if not case_index.empty else "No complete case set was available.")
        )
    logger.info("Pipeline complete. Requested=%s actual=%s", requested_mode, actual_source)
    return metrics_summary, comparison_df, ablations


if __name__ == "__main__":
    run_pipeline(use_real_data=True)
