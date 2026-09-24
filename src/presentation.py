"""Presentation-layer loading and cross-artifact validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def load_protocol_manifest(output_dir: Path) -> Dict[str, Any]:
    path = Path(output_dir) / "protocol_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Frozen run manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2 or not manifest.get("run_id") or not manifest.get("resolved_runtime"):
        raise ValueError(f"Incompatible run manifest at {path}; regenerate with the canonical pipeline")
    return manifest


def validate_operating_threshold_artifacts(
    manifest: Dict[str, Any], comparison: pd.DataFrame
) -> None:
    """Fail when presentation thresholds disagree with the frozen run manifest."""
    if manifest.get("schema_version") != 2:
        raise ValueError("Incompatible run manifest version; regenerate the complete artifact set")
    selected = manifest.get("resolved_runtime", {}).get("selected_models", {})
    required_columns = {"Model Key", "Validation-Selected Threshold"}
    if not required_columns.issubset(comparison.columns):
        raise ValueError(
            "Model comparison lacks threshold provenance columns: "
            f"{sorted(required_columns - set(comparison.columns))}"
        )
    for _, row in comparison.iterrows():
        model_key = row["Model Key"]
        displayed = row["Validation-Selected Threshold"]
        if model_key not in selected:
            raise ValueError(f"Model {model_key!r} is absent from the frozen run manifest.")
        model = selected[model_key]
        frozen = model.get("validation_selected_operating_threshold")
        status = model.get("threshold_status")
        if status == "no_alert" and frozen is None and pd.isna(displayed):
            continue
        if status != "selected" or frozen is None or pd.isna(displayed) or not np.isclose(
            float(displayed), float(frozen), rtol=0.0, atol=5e-5
        ):
            raise ValueError(
                f"Operating threshold mismatch for {model_key}: dashboard/report={displayed}, "
                f"manifest={frozen}."
            )


def selected_model_settings(manifest: Dict[str, Any], model_key: str) -> Dict[str, Any]:
    resolved = manifest.get("resolved_runtime", {})
    model = dict(resolved.get("selected_models", {}).get(model_key, {}))
    model["internal_detector_parameters"] = resolved.get("detectors", {})
    model["matching_horizon_pitches"] = resolved.get("horizons", {}).get(
        "warning_matching_pitches"
    )
    return model


def validate_release_artifacts(output_dir: Path) -> Dict[str, Any]:
    """Check a staged release before it can replace a published output set."""
    output_dir = Path(output_dir)
    manifest = load_protocol_manifest(output_dir)
    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(output_dir / "model_comparison.csv")
    coverage = pd.read_csv(output_dir / "evaluation_coverage.csv")
    bootstrap = pd.read_csv(output_dir / "bootstrap_confidence_intervals.csv")
    ablation = pd.read_csv(output_dir / "ablation_results.csv")
    validate_operating_threshold_artifacts(manifest, comparison)
    run_id, protocol = manifest["run_id"], manifest["protocol_sha256"]
    if metrics.get("run_id") != run_id or metrics.get("protocol_sha256") != protocol:
        raise ValueError("Metrics summary has a different run or protocol identity")
    for name, frame in (("comparison", comparison), ("coverage", coverage),
                        ("bootstrap", bootstrap), ("ablation", ablation)):
        if frame.empty or not {"run_id", "protocol_sha256"}.issubset(frame):
            raise ValueError(f"{name} lacks required release identity")
        if set(frame["run_id"]) != {run_id} or set(frame["protocol_sha256"]) != {protocol}:
            raise ValueError(f"{name} belongs to a different run or protocol")
    for path in output_dir.glob("*.csv"):
        frame = pd.read_csv(path)
        if not {"run_id", "protocol_sha256"}.issubset(frame.columns):
            raise ValueError(f"Derived artifact {path.name} lacks run and protocol identity")
        if not frame.empty and (set(frame["run_id"]) != {run_id} or set(frame["protocol_sha256"]) != {protocol}):
            raise ValueError(f"Derived artifact {path.name} has a different run or protocol")
    case_index = output_dir / "case_studies" / "case_study_index.csv"
    if case_index.exists():
        frame = pd.read_csv(case_index)
        if not {"run_id", "protocol_sha256"}.issubset(frame.columns):
            raise ValueError("Case-study index lacks release identity")
    primary = coverage.loc[coverage["model_key"].eq("proposed")]
    if len(primary) != 1:
        raise ValueError("Coverage must contain one proposed-model row")
    primary = primary.iloc[0]
    for column, key in (("total_qualified_test_outings", "total_qualified_outings"),
                        ("outings_with_available_score", "outings_with_available_score"),
                        ("outings_without_available_score", "outings_without_available_score"),
                        ("evaluated_episodes", "evaluated_episodes_count")):
        if int(primary[column]) != int(metrics[key]):
            raise ValueError(f"Coverage and primary metric disagree on {column}")
    if int(primary["outings_with_available_score"] + primary["outings_without_available_score"]) != int(primary["total_qualified_test_outings"]):
        raise ValueError("Available and unavailable outings do not reconcile")
    for _, row in coverage.iterrows():
        if int(row["total_qualified_test_outings"]) != int(primary["total_qualified_test_outings"]) or int(row["evaluated_episodes"]) != int(primary["evaluated_episodes"]):
            raise ValueError("Models do not share outing and episode populations")
        if int(row["evaluation_opportunity_pitches"]) != int(primary["evaluation_opportunity_pitches"]):
            raise ValueError("Models do not share protocol-defined pitch opportunities")
        if not 0 <= int(row["score_available_evaluation_pitches"]) <= int(row["evaluation_opportunity_pitches"]):
            raise ValueError("Available pitch count exceeds protocol opportunities")
        pitch_rate = (row["score_available_evaluation_pitches"] / row["evaluation_opportunity_pitches"]
                      if row["evaluation_opportunity_pitches"] else np.nan)
        outing_rate = (row["outings_with_available_score"] / row["total_qualified_test_outings"]
                       if row["total_qualified_test_outings"] else np.nan)
        if not ((pd.isna(pitch_rate) and pd.isna(row["pitch_level_scoring_coverage"])) or
                np.isclose(pitch_rate, row["pitch_level_scoring_coverage"], rtol=0, atol=1e-12)):
            raise ValueError("Pitch coverage numerator and denominator disagree")
        if not ((pd.isna(outing_rate) and pd.isna(row["outing_level_scoring_coverage"])) or
                np.isclose(outing_rate, row["outing_level_scoring_coverage"], rtol=0, atol=1e-12)):
            raise ValueError("Outing coverage numerator and denominator disagree")
    for metric, numerator, denominator in (("episode_recall", "collapse_episodes_detected", "evaluated_episodes_count"),
                                           ("warning_precision", "total_warnings", "total_warnings"),
                                           ("false_warnings_per_outing", "false_warnings", "total_qualified_outings")):
        if metric == "warning_precision":
            numerator_value = metrics["total_warnings"] - metrics["false_warnings"]
        else:
            numerator_value = metrics[numerator]
        expected = numerator_value / metrics[denominator] if metrics[denominator] else None
        observed = metrics[metric]
        if not ((expected is None and observed is None) or
                (expected is not None and observed is not None and np.isclose(expected, observed, rtol=0, atol=1e-12))):
            raise ValueError(f"Reported {metric} does not match its numerator and denominator")
    point = bootstrap.loc[bootstrap["comparison"].eq("proposed") & bootstrap["resampling_unit"].eq("outing")]
    for metric in ("episode_recall", "warning_precision", "false_warnings_per_outing"):
        row = point.loc[point["metric"].eq(metric)]
        if len(row) != 1:
            raise ValueError(f"Missing bootstrap point estimate for {metric}")
        estimate = row["estimate"].iloc[0]
        observed = metrics[metric]
        if not ((pd.isna(estimate) and observed is None) or
                (observed is not None and np.isclose(estimate, observed, rtol=0, atol=1e-12))):
            raise ValueError(f"Bootstrap point estimate disagrees for {metric}")
    full = ablation.loc[ablation["Feature Subset"].eq("Full Micro-Mechanics Suite")]
    if len(full) != 1 or full["status"].iloc[0] != "verified":
        raise ValueError("Full-feature ablation has not passed parity")
    for column, key in (("Test Episode Recall", "episode_recall"),
                        ("Test Warning Precision", "warning_precision"),
                        ("Test False Warnings / Outing", "false_warnings_per_outing"),
                        ("Test Qualified Outings", "total_qualified_outings"),
                        ("Test Evaluated Episodes", "evaluated_episodes_count"),
                        ("Test Warnings", "total_warnings"),
                        ("Test False Warnings", "false_warnings")):
        value, expected = full[column].iloc[0], metrics[key]
        if not ((pd.isna(value) and expected is None) or
                (expected is not None and np.isclose(value, expected, rtol=0, atol=1e-12))):
            raise ValueError(f"Full-feature ablation parity failed for {column}")
    frozen = manifest["resolved_runtime"]["selected_models"]["proposed"]
    ablation_threshold = full["Validation-Selected Threshold"].iloc[0]
    if frozen["threshold_status"] == "no_alert":
        if pd.notna(ablation_threshold):
            raise ValueError("Full-feature ablation no-alert threshold mismatch")
    elif not np.isclose(ablation_threshold, frozen["validation_selected_operating_threshold"], rtol=0, atol=1e-12):
        raise ValueError("Full-feature ablation frozen threshold mismatch")
    if set(comparison["Actual Data Source"]) != {manifest["actual_data_source"]}:
        raise ValueError("Real and simulated output sources are mixed")
    return manifest
