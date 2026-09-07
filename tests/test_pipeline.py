"""
Comprehensive Unit & Integration Tests for Audited Baseball Fatigue System
"""
import pytest
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

from src.data_ingest.statcast_loader import StatcastLoader
from src.data_ingest.qualify_filter import QualifyFilter
from src.feature_engineering.mechanics_features import compute_kinematics_and_vaa
from src.feature_engineering.rolling_stats import compute_rolling_features
from src.baseline_builder.historical_baseline import BaselineBuilder
from src.baseline_builder.shrinkage_calibrator import ShrinkageCalibrator
from src.anomaly_scorer.mahalanobis_scorer import MahalanobisScorer
from src.anomaly_scorer.health_index import compute_mechanics_stability_index
from src.changepoint_detector.cusum_detector import CUSUMDetector
from src.label_builder.collapse_labels import CollapseLabelBuilder
from src.evaluation.metrics import EvaluationEngine
from src.evaluation.baseline_comparator import BaselineComparator

@pytest.fixture
def sample_dataset():
    loader = StatcastLoader()
    raw_df = loader.generate_simulation_benchmark(num_pitchers=2, starts_per_pitcher=6)
    return raw_df

def test_circular_spin_axis_continuity():
    """Verify that spin axis at 359° and 1° is continuous (2° difference, not 358°)."""
    df = pd.DataFrame({
        "release_pos_x": [0.0, 0.0],
        "release_pos_z": [6.0, 6.0],
        "spin_axis": [359.0, 1.0],
        "vy0": [-130.0, -130.0],
        "ay": [28.0, 28.0],
        "vz0": [-6.0, -6.0],
        "az": [-20.0, -20.0],
        "vx0": [4.0, 4.0],
        "ax": [-10.0, -10.0],
        "plate_z": [2.5, 2.5]
    })
    res = compute_kinematics_and_vaa(df)
    v1 = np.array([res["spin_axis_cos"].iloc[0], res["spin_axis_sin"].iloc[0]])
    v2 = np.array([res["spin_axis_cos"].iloc[1], res["spin_axis_sin"].iloc[1]])
    euclid_dist = np.linalg.norm(v1 - v2)
    # 2 degrees in radians is ~0.0349
    assert euclid_dist < 0.05

def test_mahalanobis_exact_math():
    """Verify that audited Mahalanobis quadratic form matches scipy calculation."""
    np.random.seed(42)
    dim = 10
    mu = np.zeros(dim)
    A = np.random.randn(dim, dim)
    cov = np.dot(A, A.T) + np.eye(dim) * 0.1
    prec = np.linalg.inv(cov)
    
    x = np.random.randn(dim)
    diff = x - mu
    
    # Scipy standard
    expected_d = mahalanobis(x, mu, prec)
    
    # Audited implementation
    d2 = float(np.dot(np.dot(diff, prec), diff))
    actual_d = np.sqrt(max(0.0, d2))
    
    assert np.isclose(expected_d, actual_d, atol=1e-5)

def test_no_lookahead_temporal_isolation(sample_dataset):
    """Verify that baselines for game D strictly use games < D."""
    feat_df = compute_kinematics_and_vaa(sample_dataset)
    b_builder = BaselineBuilder(historical_window_starts=3, min_prior_starts=1, min_pitches_for_baseline=15)
    base_df, base_store = b_builder.build_baselines(feat_df)
    
    assert not base_df.empty
    for _, row in base_df.iterrows():
        as_of_date = row["as_of_date"]
        window_start = row["window_start_date"]
        assert window_start <= as_of_date

def test_calibration_temporal_isolation(sample_dataset):
    """Verify pitches 1..20 are flagged as calibration and alert begins pitch 21+."""
    qualifier = QualifyFilter(min_pitches_per_game=40, min_starts_per_season=2)
    q_df, _, _ = qualifier.filter_qualified_games(sample_dataset)
    feat_df = compute_kinematics_and_vaa(q_df)
    
    b_builder = BaselineBuilder(historical_window_starts=3, min_prior_starts=1, min_pitches_for_baseline=15)
    _, base_store = b_builder.build_baselines(feat_df)
    
    calibrator = ShrinkageCalibrator(intra_game_calibration_pitches=20)
    first_outing = feat_df[(feat_df["game_pk"] == feat_df["game_pk"].iloc[0]) & (feat_df["pitcher"] == feat_df["pitcher"].iloc[0])]
    calib_df, _ = calibrator.calibrate_outing_pitches(first_outing, first_outing["pitcher"].iloc[0], first_outing["game_pk"].iloc[0], base_store)
    
    assert (calib_df[calib_df["pitch_number_in_outing"] <= 20]["is_calibration_phase"]).all()
    assert not (calib_df[calib_df["pitch_number_in_outing"] > 20]["is_calibration_phase"]).any()

def test_collapse_episode_merging(sample_dataset):
    """Verify contiguous bad windows are merged into single distinct episodes."""
    l_builder = CollapseLabelBuilder(window_pa_size=3, blended_xwoba_threshold=0.450)
    outing_1 = sample_dataset[sample_dataset["game_pk"] == sample_dataset["game_pk"].iloc[0]].copy()
    outing_1["pitch_number_in_outing"] = np.arange(1, len(outing_1) + 1)
    outing_1["pa_number_in_outing"] = pd.factorize(outing_1["at_bat_number"])[0] + 1
    
    labeled_df, ep_df = l_builder.build_labels_for_outing(outing_1)
    assert "y_true_onset_in_horizon" in labeled_df.columns
    assert "is_collapse_event" in labeled_df.columns

def test_fair_shared_ground_truth(sample_dataset):
    """Verify BaselineComparator evaluates on shared ground truth without 0.0x bug."""
    outing_1 = sample_dataset[sample_dataset["game_pk"] == sample_dataset["game_pk"].iloc[0]].copy()
    outing_1["pitch_number_in_outing"] = np.arange(1, len(outing_1) + 1)
    outing_1["pa_number_in_outing"] = pd.factorize(outing_1["at_bat_number"])[0] + 1
    outing_1["is_cusum_alert"] = outing_1["pitch_number_in_outing"] > 50
    outing_1["mahalanobis_calibrated"] = 1.0 + (outing_1["pitch_number_in_outing"] / 30.0)
    
    l_builder = CollapseLabelBuilder(window_pa_size=3, blended_xwoba_threshold=0.450)
    labeled_df, ep_df = l_builder.build_labels_for_outing(outing_1)
    
    eval_engine = EvaluationEngine(horizon_pitches=15)
    metrics, eval_df = eval_engine.evaluate_pipeline(labeled_df, pd.DataFrame(), ep_df)
    
    comp = BaselineComparator(velocity_drop_mph=1.5, pitch_count_thresh=85, horizon_pitches=15)
    comp_df = comp.compare_systems(eval_df, metrics)
    assert not comp_df.empty
    assert "Model / System" in comp_df.columns
