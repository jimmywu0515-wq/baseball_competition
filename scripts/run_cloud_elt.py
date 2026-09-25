"""Cloud publishing wrapper around the canonical local/holdout pipeline."""
from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.run_full_pipeline import run_pipeline
from src.storage.adapter import StorageManager
from src.storage.gcs_adapter import GCSLakeManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CloudELT")

PROJECT_ID = os.environ.get("GCP_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
DATASET_ID = os.environ.get("BIGQUERY_DATASET", "baseball_analytics")
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME") or (
    f"{PROJECT_ID}-baseball-lakehouse" if PROJECT_ID else None
)


def clean_df_for_bigquery(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9_]", "_", str(c))).strip("_").lower()
                      for c in result.columns]
    for column in [c for c in result if "date" in c]:
        try:
            result[column] = pd.to_datetime(result[column]).dt.date
        except Exception:
            pass
    return result


def write_to_bigquery(df: pd.DataFrame, table_name: str):
    if df.empty:
        return
    from google.cloud import bigquery
    client = bigquery.Client(project=PROJECT_ID)
    destination = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"
    config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE, autodetect=True
    )
    client.load_table_from_dataframe(
        clean_df_for_bigquery(df), destination, job_config=config
    ).result()
    logger.info("Loaded %s into %s", len(df), destination)


def run_cloud_elt(use_real_data: bool = True):
    """Run the exact canonical pipeline, then publish its provenance-bearing tables."""
    if not PROJECT_ID or not BUCKET_NAME:
        raise RuntimeError("GCP_PROJECT_ID and GCS_BUCKET_NAME must be set for a cloud run.")
    metrics, _, _ = run_pipeline(use_real_data=use_real_data, base_dir=str(project_root))
    logger.info("Canonical run actual source: %s", metrics.get("actual_data_source"))

    storage = StorageManager(base_dir=str(project_root), namespace=None if use_real_data else "simulation")
    gcs = GCSLakeManager(bucket_name=BUCKET_NAME, project_id=PROJECT_ID, strict=True)
    tables = [
        ("raw_statcast_pitches", "raw"),
        ("stg_qualified_pitches", "silver"),
        ("dim_pitchers", "silver"),
        ("dim_games", "silver"),
        ("feat_pitcher_pitchtype_baseline", "silver"),
        ("feat_pitch_level_features", "silver"),
        ("fact_pitch_anomaly_scores", "gold"),
        ("fact_alert_events", "gold"),
        ("fact_collapse_labels", "gold"),
        ("mart_model_evaluation", "gold"),
        ("mart_historical_2024_evaluation", "gold"),
        ("mart_threshold_tradeoffs", "gold"),
        ("mart_bootstrap_confidence_intervals", "gold"),
        ("mart_lead_time_sensitivity", "gold"),
        ("fact_matched_warning_episodes", "gold"),
        ("mart_lead_time_distribution", "gold"),
        ("mart_missing_data_summary", "gold"),
        ("audit_excluded_pitchers", "gold"),
        ("audit_excluded_outings", "gold"),
        ("audit_unavailable_scores", "gold"),
        ("audit_cohort_selection", "gold"),
        ("audit_ingestion_segments", "gold"),
        ("mart_pitcher_model_evaluation", "gold"),
        ("mart_evaluation_coverage", "gold"),
        ("warehouse_integrity_report", "gold"),
    ]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for table_name, layer in tables:
        frame = storage.load_table(table_name, layer=layer)
        if frame.empty:
            continue
        gcs.upload_parquet_to_lake(frame, f"warehouse/{layer}/{table_name}/latest.parquet")
        write_to_bigquery(frame, table_name)
    result_count = gcs.upload_directory(
        project_root / "outputs" / ("real_data" if use_real_data else "simulation"),
        f"results/{'real_data' if use_real_data else 'simulation'}/{timestamp}",
    )
    logger.info("Uploaded %s result artifacts to the versioned GCS result path", result_count)
    logger.info("Cloud ELT complete using the same split, calibration, and evaluation path.")


if __name__ == "__main__":
    simulation_only = os.environ.get("SIMULATION_BENCHMARK", "0") == "1"
    run_cloud_elt(use_real_data=not simulation_only)
