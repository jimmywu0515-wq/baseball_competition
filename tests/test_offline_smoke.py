"""Small, isolated synthetic run through the production pipeline."""
from pathlib import Path
import shutil
import uuid

import pandas as pd
import yaml

from scripts.run_full_pipeline import run_pipeline
from src.data_ingest.statcast_loader import StatcastLoader
from src.presentation import validate_release_artifacts


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
    finally:
        try:
            shutil.rmtree(temp_root)
        except PermissionError:
            # Windows may retain a short-lived DuckDB handle until pytest exits.
            pass
