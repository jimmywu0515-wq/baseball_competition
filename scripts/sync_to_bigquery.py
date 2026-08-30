"""
Data Lakehouse Sync Script: Local Parquet -> GCP BigQuery
Uploads all 5 Medallion layers into BigQuery dataset 'baseball_analytics'.
"""
import os
import sys
import logging
from pathlib import Path
import pandas as pd

# Ensure storage is loadable
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.storage.adapter import StorageManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "project-f677f84f-db22-4976-96b")
DATASET = os.environ.get("BIGQUERY_DATASET", "baseball_analytics")

def sync_all_tables_to_bigquery():
    try:
        from google.cloud import bigquery
    except ImportError:
        logger.error("google-cloud-bigquery is not installed. Run: pip install google-cloud-bigquery")
        return

    client = bigquery.Client(project=PROJECT_ID)
    sm = StorageManager(base_dir=str(project_root))
    
    tables_to_sync = [
        ("raw_statcast_pitches", "raw"),
        ("stg_qualified_pitches", "silver"),
        ("dim_pitchers", "silver"),
        ("dim_games", "silver"),
        ("feat_pitcher_pitchtype_baseline", "silver"),
        ("feat_pitch_level_features", "silver"),
        ("fact_pitch_anomaly_scores", "gold"),
        ("fact_alert_events", "gold"),
        ("fact_collapse_labels", "gold"),
        ("mart_model_evaluation", "gold")
    ]

    logger.info(f"Starting sync of 10 Medallion Lakehouse tables to BigQuery `{PROJECT_ID}.{DATASET}`...")

    for table_name, layer in tables_to_sync:
        df = sm.load_table(table_name, layer=layer)
        if df.empty:
            logger.warning(f"Table {table_name} is empty locally. Skipping.")
            continue

        dest_table = f"{PROJECT_ID}.{DATASET}.{table_name}"
        logger.info(f"Uploading {len(df)} rows to BigQuery table `{dest_table}`...")
        
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        
        try:
            job = client.load_table_from_dataframe(df, dest_table, job_config=job_config)
            job.result() # Wait for job to finish
            logger.info(f" Successfully loaded `{table_name}` ({len(df)} rows).")
        except Exception as e:
            logger.error(f"Failed to load `{table_name}` to BigQuery: {e}")

    logger.info("BigQuery Sync Completed Successfully!")

if __name__ == "__main__":
    sync_all_tables_to_bigquery()
