"""Central, validated project configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_project_config(path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = path or Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    # Strict canonical names: reject historical aliases even when a new key is
    # also present, so a stale edit cannot silently change the wrong setting.
    old_keys = {
        "anomaly": ("health_index_alpha", "msi_decay_alpha"),
        "changepoint": ("cusum_threshold_h", "cusum_alert_h"),
        "evaluation": ("matching_horizon_pitches", "prediction_horizon_pitches"),
    }
    for section, names in old_keys.items():
        for name in names:
            if name in config[section]:
                raise ValueError(f"Obsolete configuration key {section}.{name}; use the canonical config field")
    baseline = config["baseline"]
    minimum = int(baseline["min_prior_starts"])
    window = int(baseline["historical_window_starts"])
    if minimum > window:
        raise ValueError(
            f"baseline.min_prior_starts ({minimum}) cannot exceed "
            f"baseline.historical_window_starts ({window})."
        )
    if minimum <= 0 or window <= 0 or int(baseline["min_pitches_for_baseline"]) <= 0:
        raise ValueError("Historical baseline sample counts must be positive")
    if not 0 <= float(baseline["shrinkage_lambda"]) <= 1:
        raise ValueError("baseline.shrinkage_lambda must be in [0, 1]")
    if int(config["qualify"]["min_pitches_per_game"]) <= 0 or int(config["qualify"]["min_starts_for_cohort"]) <= 0:
        raise ValueError("Qualification pitch and outing minima must be positive")
    if not 0 <= float(config["qualify"]["max_missing_mechanics_pct"]) <= 1:
        raise ValueError("qualify.max_missing_mechanics_pct must be in [0, 1]")
    seasons = [int(year) for year in config["ingestion"]["required_seasons"]]
    if sorted(set(seasons)) != seasons:
        raise ValueError("ingestion.required_seasons must be unique and sorted.")
    if not config["ingestion"].get("pitcher_ids"):
        raise ValueError("ingestion.pitcher_ids must contain at least one pitcher.")
    selection = config["ingestion"].get("cohort_selection")
    if selection:
        if any(int(year) >= 2025 for year in selection["seasons"]):
            raise ValueError("cohort_selection.seasons must contain only pre-2025 seasons.")
        if int(selection["target_size"]) < len(set(config["ingestion"]["pitcher_ids"])):
            raise ValueError("cohort_selection.target_size cannot be smaller than retained pitcher_ids.")
    label_horizon = int(config["labels"]["prediction_horizon_pitches"])
    matching_horizon = int(config["evaluation"]["warning_matching_horizon_pitches"])
    if label_horizon <= 0 or matching_horizon <= 0:
        raise ValueError("Label and warning-matching horizons must be positive.")
    if int(config["labels"]["prediction_horizon_pas"]) <= 0:
        raise ValueError("labels.prediction_horizon_pas must be positive")
    if int(config["labels"]["window_pa_size"]) <= 0 or not 0 <= float(config["labels"]["blended_xwoba_threshold"]) <= 1:
        raise ValueError("Label window must be positive and xwOBA threshold must be in [0, 1]")
    if not all(int(value) > 0 for value in config["evaluation"]["sensitivity_horizons"]):
        raise ValueError("evaluation.sensitivity_horizons must be positive")
    if int(config["evaluation"]["bootstrap_samples"]) <= 0:
        raise ValueError("evaluation.bootstrap_samples must be positive")
    if not 0 <= float(config["evaluation"]["false_warnings_per_outing"]):
        raise ValueError("evaluation.false_warnings_per_outing cannot be negative")
    if not (config["evaluation"]["train_end"] < config["evaluation"]["validation_end"] < config["evaluation"]["test_start"]):
        raise ValueError("Training, validation, and test dates must be chronological")
    if int(baseline["intra_game_calibration_pitches"]) < 0:
        raise ValueError("baseline.intra_game_calibration_pitches cannot be negative.")
    if int(baseline["intra_game_calibration_pitches"]) >= int(config["qualify"]["min_pitches_per_game"]):
        raise ValueError("Calibration length must leave at least one pitch in a qualified outing")
    if not 0.0 < float(config["anomaly"]["health_index_decay_alpha"]):
        raise ValueError("anomaly.health_index_decay_alpha must be positive.")
    if float(config["anomaly"]["ridge_regularization"]) < 0:
        raise ValueError("anomaly.ridge_regularization cannot be negative")
    detector = config["changepoint"]
    for name in ("cusum_reference_std", "ewma_reference_std", "cusum_internal_alert_h", "ewma_control_limit_sigma"):
        if float(detector[name]) <= 0:
            raise ValueError(f"changepoint.{name} must be positive")
    if float(detector["cusum_slack_k"]) < 0 or not 0 < float(detector["ewma_lambda"]) <= 1:
        raise ValueError("CUSUM slack must be nonnegative and EWMA lambda must be in (0, 1]")
    return config
