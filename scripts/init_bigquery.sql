-- =====================================================================
-- GCP BigQuery Table DDL for Baseball Lakehouse (Medallion Architecture)
-- =====================================================================

-- 1. Bronze Layer: Raw Ingest Table
CREATE TABLE IF NOT EXISTS `baseball_analytics.raw_statcast_pitches`
(
  game_pk INT64 NOT NULL,
  game_date DATE NOT NULL,
  pitcher INT64 NOT NULL,
  batter INT64,
  pitcher_name STRING,
  pitch_type STRING,
  release_speed FLOAT64,
  release_pos_x FLOAT64,
  release_pos_y FLOAT64,
  release_pos_z FLOAT64,
  release_spin_rate FLOAT64,
  spin_axis FLOAT64,
  pfx_x FLOAT64,
  pfx_z FLOAT64,
  plate_x FLOAT64,
  plate_z FLOAT64,
  vx0 FLOAT64,
  vy0 FLOAT64,
  vz0 FLOAT64,
  ax FLOAT64,
  ay FLOAT64,
  az FLOAT64,
  release_extension FLOAT64,
  events STRING,
  description STRING,
  balls INT64,
  strikes INT64,
  outs_when_up INT64,
  inning INT64,
  at_bat_number INT64,
  pitch_number INT64,
  woba_value FLOAT64,
  woba_denom FLOAT64,
  launch_speed FLOAT64,
  launch_angle FLOAT64,
  estimated_woba_using_speedangle FLOAT64
)
PARTITION BY game_date
CLUSTER BY pitcher, pitch_type;

-- 2. Silver Layer: Qualified Pitches
CREATE TABLE IF NOT EXISTS `baseball_analytics.stg_qualified_pitches`
(
  game_pk INT64 NOT NULL,
  game_date DATE NOT NULL,
  pitcher INT64 NOT NULL,
  pitcher_name STRING,
  p_throws STRING,
  inning INT64,
  at_bat_number INT64,
  pitch_number_in_game INT64,
  pitch_type STRING,
  is_primary_pitch_type BOOL,
  release_speed FLOAT64,
  release_pos_x FLOAT64,
  release_pos_z FLOAT64,
  release_extension FLOAT64,
  release_spin_rate FLOAT64,
  spin_axis FLOAT64,
  pfx_x FLOAT64,
  pfx_z FLOAT64,
  vaa FLOAT64,
  estimated_woba_using_speedangle FLOAT64
)
PARTITION BY game_date
CLUSTER BY pitcher, pitch_type;

-- 3. Silver Layer: Historical Rolling Baselines
CREATE TABLE IF NOT EXISTS `baseball_analytics.feat_pitcher_pitchtype_baseline`
(
  pitcher INT64 NOT NULL,
  pitch_type STRING NOT NULL,
  as_of_game_pk INT64 NOT NULL,
  as_of_date DATE NOT NULL,
  window_start_date DATE,
  window_games_count INT64,
  pitches_count INT64,
  mean_release_pos_x FLOAT64,
  std_release_pos_x FLOAT64,
  mean_release_pos_z FLOAT64,
  std_release_pos_z FLOAT64,
  mean_release_extension FLOAT64,
  std_release_extension FLOAT64,
  mean_release_speed FLOAT64,
  std_release_speed FLOAT64,
  mean_release_spin_rate FLOAT64,
  std_release_spin_rate FLOAT64,
  mean_spin_axis FLOAT64,
  std_spin_axis FLOAT64,
  mean_pfx_x FLOAT64,
  std_pfx_x FLOAT64,
  mean_pfx_z FLOAT64,
  std_pfx_z FLOAT64,
  mean_vaa FLOAT64,
  std_vaa FLOAT64,
  covariance_matrix_json STRING,
  precision_matrix_json STRING
)
PARTITION BY as_of_date
CLUSTER BY pitcher, pitch_type;

-- 4. Gold Layer: Pitch Anomaly Scores & Health Index
CREATE TABLE IF NOT EXISTS `baseball_analytics.fact_pitch_anomaly_scores`
(
  game_pk INT64 NOT NULL,
  game_date DATE NOT NULL,
  pitcher INT64 NOT NULL,
  pitcher_name STRING,
  pitch_number_in_game INT64,
  inning INT64,
  at_bat_number INT64,
  pitch_type STRING,
  mahalanobis_raw FLOAT64,
  mahalanobis_calibrated FLOAT64,
  autoencoder_recon_loss FLOAT64,
  health_index FLOAT64,
  dominant_drift_feature STRING,
  contrib_release_pct FLOAT64,
  contrib_spin_pct FLOAT64,
  contrib_movement_pct FLOAT64,
  contrib_speed_pct FLOAT64,
  cusum_stat FLOAT64,
  is_cusum_alert BOOL,
  ewma_stat FLOAT64,
  is_ewma_alert BOOL
)
PARTITION BY game_date
CLUSTER BY pitcher, game_pk;

-- 5. Gold Layer: Alert Events
CREATE TABLE IF NOT EXISTS `baseball_analytics.fact_alert_events`
(
  game_pk INT64 NOT NULL,
  game_date DATE NOT NULL,
  pitcher INT64 NOT NULL,
  pitcher_name STRING,
  alert_pitch_number INT64,
  alert_inning INT64,
  detector_type STRING,
  cusum_statistic FLOAT64,
  cusum_threshold FLOAT64,
  alert_level STRING,
  primary_drift_reason STRING
)
PARTITION BY game_date
CLUSTER BY pitcher;

-- 6. Gold Layer: Isolated Ground Truth Collapse Labels
CREATE TABLE IF NOT EXISTS `baseball_analytics.fact_collapse_labels`
(
  game_pk INT64 NOT NULL,
  game_date DATE NOT NULL,
  pitcher INT64 NOT NULL,
  pitcher_name STRING,
  at_bat_number INT64,
  pitch_number_in_game INT64,
  inning INT64,
  window_pa_count INT64,
  window_blended_xwoba FLOAT64,
  window_barrels_count INT64,
  window_bb_hbp_count INT64,
  is_collapse_event BOOL,
  collapse_reason STRING
)
PARTITION BY game_date
CLUSTER BY pitcher, game_pk;
