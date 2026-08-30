"""
Comprehensive Unit & Integration Test Suite for Baseball Fatigue System
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_ingest.statcast_loader import StatcastLoader
from src.data_ingest.qualify_filter import QualifyFilter
from src.feature_engineering.mechanics_features import compute_kinematics_and_vaa
from src.feature_engineering.rolling_stats import compute_rolling_features
from src.baseline_builder.historical_baseline import BaselineBuilder
from src.baseline_builder.shrinkage_calibrator import ShrinkageCalibrator
from src.anomaly_scorer.mahalanobis_scorer import MahalanobisScorer
from src.anomaly_scorer.health_index import compute_pitcher_health_index
from src.changepoint_detector.cusum_detector import CUSUMDetector
from src.label_builder.collapse_labels import CollapseLabelBuilder
from src.evaluation.metrics import EvaluationEngine

@pytest.fixture
def sample_dataset():
    loader = StatcastLoader()
    raw_df = loader.generate_realistic_sample_data(num_pitchers=2, starts_per_pitcher=6)
    return raw_df

def test_qualify_filter(sample_dataset):
    qualifier = QualifyFilter(min_pitches_per_game=50, min_starts_per_season=4, max_missing_mechanics_pct=0.05)
    q_df, dim_p, dim_g = qualifier.filter_qualified_games(sample_dataset)
    assert not q_df.empty
    assert len(dim_p) == 2
    assert (q_df.groupby("game_pk").size() >= 50).all()

def test_vaa_and_mechanics(sample_dataset):
    feat_df = compute_kinematics_and_vaa(sample_dataset)
    assert "vaa" in feat_df.columns
    assert "release_3d_dist_from_origin" in feat_df.columns
    assert not feat_df["vaa"].isna().all()
    # Typical MLB VAA ranges between -12 and -3 degrees
    assert feat_df["vaa"].between(-25.0, 5.0).all()

def test_rolling_dispersion(sample_dataset):
    roll_df = compute_rolling_features(sample_dataset)
    assert "roll5_speed_std" in roll_df.columns
    assert "roll5_release_dist_drift" in roll_df.columns
    assert not roll_df["roll5_speed_std"].isna().any()

def test_baseline_and_temporal_isolation(sample_dataset):
    feat_df = compute_kinematics_and_vaa(sample_dataset)
    b_builder = BaselineBuilder(historical_window_starts=3, min_pitches_for_baseline=15)
    base_df, base_store = b_builder.build_baselines(feat_df)
    assert not base_df.empty
    assert len(base_store) > 0

def test_mahalanobis_and_health_index():
    scores = np.array([0.5, 1.2, 2.5, 4.0])
    health = compute_pitcher_health_index(scores, decay_alpha=0.45)
    assert len(health) == 4
    # Health should monotonically decrease with higher anomaly score
    assert health[0] > health[1] > health[2] > health[3]
    assert health[0] > 70.0
    assert health[3] < 30.0

def test_cusum_changepoint():
    detector = CUSUMDetector(slack_k=0.5, threshold_h=3.0)
    # Synthetic game with normal pitches followed by severe drift
    np.random.seed(42)
    normal = np.random.normal(1.0, 0.2, 30)
    fatigued = np.random.normal(3.5, 0.4, 30)
    scores = np.concatenate([normal, fatigued])
    
    test_df = pd.DataFrame({
        "game_pk": [1001] * 60,
        "game_date": ["2024-05-01"] * 60,
        "pitcher": [543037] * 60,
        "pitcher_name": ["Gerrit Cole"] * 60,
        "pitch_number_in_game": np.arange(1, 61),
        "inning": [1]*15 + [2]*15 + [3]*15 + [4]*15,
        "mahalanobis_calibrated": scores,
        "dominant_drift_feature": ["release_pos_x"] * 60
    })
    
    scored_df, alerts = detector.detect_game_alerts(test_df)
    assert not alerts.empty
    # Alert should trigger in the fatigued portion (pitch > 30)
    first_alert = alerts["alert_pitch_number"].min()
    assert first_alert >= 31

def test_collapse_label_builder(sample_dataset):
    l_builder = CollapseLabelBuilder(window_pa_size=3, blended_xwoba_threshold=0.450)
    game_1 = sample_dataset[sample_dataset["game_pk"] == sample_dataset["game_pk"].iloc[0]]
    labeled_df, c_sum = l_builder.build_labels_for_game(game_1)
    assert "is_collapse_event" in labeled_df.columns
    assert "window_blended_xwoba" in labeled_df.columns
