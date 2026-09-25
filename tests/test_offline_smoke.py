"""Small, isolated synthetic run through the production pipeline."""
from pathlib import Path
import json
import shutil
import uuid

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.run_full_pipeline import run_pipeline
from src.data_ingest.statcast_loader import StatcastLoader
from src.presentation import validate_release_artifacts
from src.anomaly_scorer.health_index import compute_mechanics_stability_index


def test_synthetic_pipeline_release(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    temp_root = root / "outputs" / f".smoke_{uuid.uuid4().hex}"
    assert temp_root.resolve().is_relative_to((root / "outputs").resolve())
    (temp_root / "config").mkdir(parents=True)
    config = yaml.safe_load((root / "config" / "config.yaml").read_text(encoding="utf-8"))
    config["evaluation"]["bootstrap_samples"] = 24
    (temp_root / "config" / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    original = StatcastLoader.generate_simulation_benchmark
    monkeypatch.setattr(
        StatcastLoader, "generate_simulation_benchmark",
        lambda self, **kwargs: original(self, num_pitchers=1, starts_per_pitcher=24),
    )
    try:
        run_pipeline(use_real_data=False, base_dir=str(temp_root))
        published = temp_root / "outputs" / "simulation"
        manifest = validate_release_artifacts(published)
        assert manifest["actual_data_source"] == "simulation_benchmark"
        assert (published / "validation_report.md").exists()
        assert pd.read_csv(published / "evaluation_coverage.csv")["total_qualified_test_outings"].min() > 0
        scored = pd.read_parquet(temp_root / "data" / "simulation" / "gold" /
                                 "fact_pitch_anomaly_scores.parquet")
        assert set(scored["run_id"]) == {manifest["run_id"]}
        assert set(scored["protocol_sha256"]) == {manifest["protocol_sha256"]}
        np.testing.assert_allclose(
            scored["health_index"].to_numpy(),
            compute_mechanics_stability_index(
                scored["mahalanobis_calibrated"].to_numpy(),
                decay_alpha=config["anomaly"]["health_index_decay_alpha"],
            ), equal_nan=True,
        )
        ablation_path = published / "ablation_manifest.json"
        ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
        ablation["parent_protocol_hash"] = "stale"
        ablation_path.write_text(json.dumps(ablation), encoding="utf-8")
        with pytest.raises(ValueError, match="Ablation manifest"):
            validate_release_artifacts(published)
    finally:
        try:
            shutil.rmtree(temp_root)
        except PermissionError:
            # Windows may retain a short-lived DuckDB handle until pytest exits.
            pass
