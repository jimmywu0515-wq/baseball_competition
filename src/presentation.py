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
    return json.loads(path.read_text(encoding="utf-8"))


def validate_operating_threshold_artifacts(
    manifest: Dict[str, Any], comparison: pd.DataFrame
) -> None:
    """Fail when presentation thresholds disagree with the frozen run manifest."""
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
        frozen = selected[model_key].get("validation_selected_operating_threshold")
        if frozen is None and pd.isna(displayed):
            continue
        if frozen is None or pd.isna(displayed) or not np.isclose(
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
