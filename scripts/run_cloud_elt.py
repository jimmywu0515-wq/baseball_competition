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

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "project-f677f84f-db22-4976-96b")
DATASET_ID = os.environ.get("BIGQUERY_DATASET", "baseball_analytics")
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", f"{PROJECT_ID}-baseball-lakehouse")


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


def write_to_bigquery_if_possible(df: pd.DataFrame, table_name: str):
    if df.empty:
        return
    try:
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
    except Exception as exc:
        logger.error("BigQuery load failed for %s: %s", table_name, exc)


def run_cloud_elt(use_real_data: bool = True):
    """Run the exact canonical pipeline, then publish its provenance-bearing tables."""
    metrics, _, _ = run_pipeline(use_real_data=use_real_data, base_dir=str(project_root))
    logger.info("Canonical run actual source: %s", metrics.get("actual_data_source"))

    storage = StorageManager(base_dir=str(project_root))
    gcs = GCSLakeManager(bucket_name=BUCKET_NAME, project_id=PROJECT_ID)
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
    ]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for table_name, layer in tables:
        frame = storage.load_table(table_name, layer=layer)
        if frame.empty:
            continue
        gcs.upload_parquet_to_lake(frame, f"{layer}/{table_name}/batch_{timestamp}.parquet")
        write_to_bigquery_if_possible(frame, table_name)
    logger.info("Cloud ELT complete using the same split, calibration, and evaluation path.")


if __name__ == "__main__":
    simulation_only = os.environ.get("SIMULATION_BENCHMARK", "0") == "1"
    run_cloud_elt(use_real_data=not simulation_only)
