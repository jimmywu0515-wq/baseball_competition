"""
Production Cloud ELT/ETL Pipeline for Baseball Fatigue & Collapse System
Workflow:
1. Extract (Statcast Ingest)
2. Load to Lake (Upload Raw Parquet directly to GCS Data Lake: gs://{bucket}/raw/)
3. Transform & Cleaning (Qualify Filters, VAA Kinematics, Rolling Features)
4. Model & Detection (Baselines with No-Lookahead, Mahalanobis/Autoencoder, CUSUM/EWMA)
5. Label Isolation (3-PA Rolling Collapse Labels)
6. Load to Warehouse (Direct write to BigQuery partitioned tables)
"""
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

# Path setup
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.storage.adapter import StorageManager
from src.storage.gcs_adapter import GCSLakeManager
from src.data_ingest.statcast_loader import StatcastLoader
from src.data_ingest.qualify_filter import QualifyFilter
from src.feature_engineering.mechanics_features import compute_kinematics_and_vaa
from src.feature_engineering.rolling_stats import compute_rolling_features
from src.baseline_builder.historical_baseline import BaselineBuilder
from src.baseline_builder.shrinkage_calibrator import ShrinkageCalibrator
from src.anomaly_scorer.mahalanobis_scorer import MahalanobisScorer
from src.anomaly_scorer.autoencoder_scorer import AutoencoderScorer
from src.anomaly_scorer.health_index import compute_pitcher_health_index
from src.changepoint_detector.cusum_detector import CUSUMDetector
from src.changepoint_detector.ewma_detector import EWMADetector
from src.label_builder.collapse_labels import CollapseLabelBuilder
from src.evaluation.metrics import EvaluationEngine
from src.evaluation.baseline_comparator import BaselineComparator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CloudELT")

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "project-f677f84f-db22-4976-96b")
DATASET_ID = os.environ.get("BIGQUERY_DATASET", "baseball_analytics")
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", f"{PROJECT_ID}-baseball-lakehouse")

def write_to_bigquery_if_possible(df: pd.DataFrame, table_name: str):
    """Writes dataframe to BigQuery if credentials/client are available."""
    try:
        from google.cloud import bigquery
        client = bigquery.Client(project=PROJECT_ID)
        dest_table = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"
        job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
        job = client.load_table_from_dataframe(df, dest_table, job_config=job_config)
        job.result()
        logger.info(f"[BigQuery Warehouse] Loaded {len(df)} rows into `{dest_table}`")
    except Exception as e:
        logger.warning(f"[BigQuery] Could not load to BigQuery table {table_name}: {e}")

def run_cloud_elt():
    logger.info("=" * 70)
    logger.info("STARTING CLOUD END-TO-END ELT/ETL PIPELINE")
    logger.info(f"Target Project: {PROJECT_ID} | Lake: gs://{BUCKET_NAME} | Warehouse: {DATASET_ID}")
    logger.info("=" * 70)

    gcs = GCSLakeManager(bucket_name=BUCKET_NAME, project_id=PROJECT_ID)
    sm = StorageManager(base_dir=str(project_root))

    # -------------------------------------------------------------
    # 1. EXTRACT: Fetch raw tracking data from Statcast
    # -------------------------------------------------------------
    logger.info("\n>>> PHASE 1: EXTRACT (Statcast High-Frequency Ingest)")
    loader = StatcastLoader()
    raw_df = loader.generate_realistic_sample_data(num_pitchers=5, starts_per_pitcher=16)
    
    # -------------------------------------------------------------
    # 2. LOAD TO LAKE: Stream raw data into GCS Bucket (Data Lake)
    # -------------------------------------------------------------
    logger.info("\n>>> PHASE 2: LOAD TO LAKE (GCS Raw Data Lake)")
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_lake_path = f"raw/statcast_pitches/batch_{now_str}.parquet"
    gcs.upload_parquet_to_lake(raw_df, raw_lake_path)
    sm.save_table(raw_df, "raw_statcast_pitches", layer="raw")
    write_to_bigquery_if_possible(raw_df, "raw_statcast_pitches")

    # -------------------------------------------------------------
    # 3. TRANSFORM: Data Cleaning & §3 Qualify Filter
    # -------------------------------------------------------------
    logger.info("\n>>> PHASE 3: TRANSFORM - Cleaning & Qualify Screening")
    qualifier = QualifyFilter(min_pitches_per_game=50, min_starts_per_season=10, max_missing_mechanics_pct=0.05)
    qualified_df, dim_pitchers, dim_games = qualifier.filter_qualified_games(raw_df)
    
    # Save clean Silver layers
    gcs.upload_parquet_to_lake(qualified_df, "silver/stg_qualified_pitches.parquet")
    sm.save_table(qualified_df, "stg_qualified_pitches", layer="silver")
    sm.save_table(dim_pitchers, "dim_pitchers", layer="silver")
    sm.save_table(dim_games, "dim_games", layer="silver")
    write_to_bigquery_if_possible(qualified_df, "stg_qualified_pitches")
    write_to_bigquery_if_possible(dim_pitchers, "dim_pitchers")
    write_to_bigquery_if_possible(dim_games, "dim_games")

    # -------------------------------------------------------------
    # 4. TRANSFORM: Mechanics Features, VAA & Rolling Dispersion
    # -------------------------------------------------------------
    logger.info("\n>>> PHASE 4: TRANSFORM - Level 0 Kinematics & Level 1 Rolling Features")
    feat_df = compute_kinematics_and_vaa(qualified_df)
    feat_df = compute_rolling_features(feat_df)

    # -------------------------------------------------------------
    # 5. MODELING: Personal Baselines & Empirical Bayes Calibration
    # -------------------------------------------------------------
    logger.info("\n>>> PHASE 5: MODELING - Rolling Prior Baselines & Shrinkage Calibration")
    b_builder = BaselineBuilder(historical_window_starts=12, min_pitches_for_baseline=30)
    baseline_df, baseline_store = b_builder.build_baselines(feat_df)
    sm.save_table(baseline_df, "feat_pitcher_pitchtype_baseline", layer="silver")
    write_to_bigquery_if_possible(baseline_df, "feat_pitcher_pitchtype_baseline")

    calibrator = ShrinkageCalibrator(intra_game_calibration_pitches=20, shrinkage_lambda=0.35)
    calib_dfs = []
    calib_means_store = {}
    for (p_id, g_pk), g_df in feat_df.groupby(["pitcher", "game_pk"]):
        c_df, c_means = calibrator.calibrate_game_pitches(g_df, p_id, g_pk, baseline_store)
        calib_dfs.append(c_df)
        calib_means_store[(p_id, g_pk)] = c_means
    features_calib_df = pd.concat(calib_dfs, ignore_index=True)
    sm.save_table(features_calib_df, "feat_pitch_level_features", layer="silver")
    write_to_bigquery_if_possible(features_calib_df, "feat_pitch_level_features")

    # -------------------------------------------------------------
    # 6. ANOMALY DETECTION: Mahalanobis & CUSUM Alert Engine
    # -------------------------------------------------------------
    logger.info("\n>>> PHASE 6: ANOMALY DETECTION - Mahalanobis, Health Index & CUSUM")
    m_scorer = MahalanobisScorer()
    scored_dfs = []
    for (p_id, g_pk), g_df in features_calib_df.groupby(["pitcher", "game_pk"]):
        scored_g = m_scorer.score_pitches(g_df, p_id, g_pk, baseline_store, calib_means_store.get((p_id, g_pk), {}))
        scored_dfs.append(scored_g)
    scored_df = pd.concat(scored_dfs, ignore_index=True)

    # Autoencoder & Health Index (0-100)
    ae = AutoencoderScorer()
    ae.fit(features_calib_df)
    scored_df["autoencoder_recon_loss"] = ae.score(scored_df)
    scored_df["health_index"] = compute_pitcher_health_index(scored_df["mahalanobis_calibrated"].values)

    # CUSUM / EWMA Detectors
    cusum = CUSUMDetector(slack_k=0.5, threshold_h=4.0)
    ewma = EWMADetector(ewma_lambda=0.20, l_sigma=2.8)
    final_dfs = []
    alerts = []
    for g_pk, g_df in scored_df.groupby("game_pk"):
        g_c, g_a = cusum.detect_game_alerts(g_df)
        g_full, _ = ewma.detect_game_alerts(g_c)
        final_dfs.append(g_full)
        if not g_a.empty:
            alerts.append(g_a)
    final_scored_df = pd.concat(final_dfs, ignore_index=True)
    alerts_df = pd.concat(alerts, ignore_index=True) if alerts else pd.DataFrame()

    sm.save_table(final_scored_df, "fact_pitch_anomaly_scores", layer="gold")
    sm.save_table(alerts_df, "fact_alert_events", layer="gold")
    write_to_bigquery_if_possible(final_scored_df, "fact_pitch_anomaly_scores")
    write_to_bigquery_if_possible(alerts_df, "fact_alert_events")

    # -------------------------------------------------------------
    # 7. ISOLATED LABELS & EVALUATION: 3-PA Windows & Lift Analysis
    # -------------------------------------------------------------
    logger.info("\n>>> PHASE 7: ISOLATED GROUND TRUTH LABELS & BENCHMARKS")
    l_builder = CollapseLabelBuilder(window_pa_size=3, blended_xwoba_threshold=0.450)
    labeled_dfs = []
    c_summaries = []
    for g_pk, g_df in final_scored_df.groupby("game_pk"):
        l_df, c_sum = l_builder.build_labels_for_game(g_df)
        labeled_dfs.append(l_df)
        if not c_sum.empty:
            c_summaries.append(c_sum)
    final_labeled_df = pd.concat(labeled_dfs, ignore_index=True)
    c_sum_df = pd.concat(c_summaries, ignore_index=True) if c_summaries else pd.DataFrame()
    sm.save_table(final_labeled_df, "fact_collapse_labels", layer="gold")
    write_to_bigquery_if_possible(final_labeled_df, "fact_collapse_labels")

    # Evaluation Mart
    eval_eng = EvaluationEngine(horizon_pitches=15)
    metrics = eval_eng.evaluate_pipeline(final_labeled_df, alerts_df, c_sum_df)
    comp = BaselineComparator(velocity_drop_mph=1.5, pitch_count_thresh=85, horizon_pitches=15)
    comp_df = comp.compare_systems(final_labeled_df, metrics)
    sm.save_table(comp_df, "mart_model_evaluation", layer="gold")
    write_to_bigquery_if_possible(comp_df, "mart_model_evaluation")

    logger.info("=" * 70)
    logger.info("CLOUD ELT PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)

if __name__ == "__main__":
    run_cloud_elt()
