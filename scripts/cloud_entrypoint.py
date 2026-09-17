"""Cloud Run Job entrypoint with durable prepare and evaluate stages."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_expanded_cohort import prepare_expanded_cohort
from scripts.run_cloud_elt import run_cloud_elt
from src.storage.gcs_adapter import GCSLakeManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CloudEntrypoint")


def _lake() -> GCSLakeManager:
    project_id = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    bucket_name = os.environ.get("GCS_BUCKET_NAME")
    if not project_id or not bucket_name:
        raise RuntimeError("GCP_PROJECT_ID and GCS_BUCKET_NAME are required.")
    return GCSLakeManager(bucket_name=bucket_name, project_id=project_id, strict=True)


def prepare(lake: GCSLakeManager) -> None:
    # Existing objects make this stage resumable: the loader validates and reuses
    # cached pitcher-season segments instead of redownloading them.
    restored = lake.download_directory("cache/raw", ROOT / "data" / "raw")
    logger.info("Restored %s cached raw files before preparation", restored)
    raw_dir = ROOT / "data" / "raw"
    try:
        prepare_expanded_cohort(ROOT)
    finally:
        # Preserve every completed segment even when a later API request fails.
        # A Cloud Run retry (or later manual execution) resumes from these files.
        cache_files = sorted(raw_dir.glob("real_statcast_*"))
        cache_files.extend(
            sorted(path for path in (raw_dir / "cohort").rglob("*") if path.is_file())
        )
        for local_path in cache_files:
            lake.upload_file(local_path, f"cache/raw/{local_path.relative_to(raw_dir).as_posix()}")
        cached = len(cache_files)
        audits = lake.upload_directory(ROOT / "outputs" / "cohort_expansion", "cache/audits")
        logger.info("Preparation checkpoint: %s cache files, %s audits", cached, audits)


def evaluate(lake: GCSLakeManager) -> None:
    restored = lake.download_directory("cache/raw", ROOT / "data" / "raw")
    segment_files = list((ROOT / "data" / "raw").glob("real_statcast_*"))
    if restored == 0 or not segment_files:
        raise RuntimeError(
            "No prepared Statcast cache exists in GCS. Execute the prepare job first."
        )
    logger.info("Restored %s cache files; starting canonical holdout evaluation", restored)
    run_cloud_elt(use_real_data=True)


def main() -> None:
    stage = os.environ.get("PIPELINE_STAGE", "evaluate").strip().lower()
    if stage not in {"prepare", "evaluate", "all"}:
        raise ValueError("PIPELINE_STAGE must be prepare, evaluate, or all")
    lake = _lake()
    if stage in {"prepare", "all"}:
        prepare(lake)
    if stage in {"evaluate", "all"}:
        evaluate(lake)


if __name__ == "__main__":
    main()
