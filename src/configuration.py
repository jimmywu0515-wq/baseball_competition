"""Central, validated project configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_project_config(path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = path or Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    baseline = config["baseline"]
    minimum = int(baseline["min_prior_starts"])
    window = int(baseline["historical_window_starts"])
    if minimum > window:
        raise ValueError(
            f"baseline.min_prior_starts ({minimum}) cannot exceed "
            f"baseline.historical_window_starts ({window})."
        )
    seasons = [int(year) for year in config["ingestion"]["required_seasons"]]
    if sorted(set(seasons)) != seasons:
        raise ValueError("ingestion.required_seasons must be unique and sorted.")
    if not config["ingestion"].get("pitcher_ids"):
        raise ValueError("ingestion.pitcher_ids must contain at least one pitcher.")
    return config
