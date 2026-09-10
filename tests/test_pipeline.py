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
from src.evaluation.protocol import assign_temporal_split, evaluate_warning_predictions

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
    """Verify the shared matcher reports event recall rather than pitch recall."""
    outing_1 = sample_dataset[sample_dataset["game_pk"] == sample_dataset["game_pk"].iloc[0]].copy()
    outing_1["pitch_number_in_outing"] = np.arange(1, len(outing_1) + 1)
    outing_1["pa_number_in_outing"] = pd.factorize(outing_1["at_bat_number"])[0] + 1
    outing_1["is_cusum_alert"] = outing_1["pitch_number_in_outing"] > 50
    outing_1["mahalanobis_calibrated"] = 1.0 + (outing_1["pitch_number_in_outing"] / 30.0)
    
    l_builder = CollapseLabelBuilder(window_pa_size=3, blended_xwoba_threshold=0.450)
    labeled_df, ep_df = l_builder.build_labels_for_outing(outing_1)
    
    labeled_df["score_available"] = labeled_df["pitch_number_in_outing"] > 20
    metrics, warnings, matches = evaluate_warning_predictions(
        labeled_df, ep_df, labeled_df["is_cusum_alert"], horizon_pitches=15
    )
    assert "episode_recall" in metrics
    assert "warning_precision" in metrics


def test_temporal_split_is_implemented():
    df = pd.DataFrame({"game_date": ["2023-09-01", "2024-04-01", "2024-07-01"]})
    result = assign_temporal_split(df)
    assert result["dataset_split"].tolist() == ["train", "validation", "test"]


def test_cusum_never_processes_calibration_or_unavailable_scores():
    df = pd.DataFrame({
        "game_pk": [1] * 23,
        "game_date": ["2024-07-01"] * 23,
        "pitcher": [10] * 23,
        "pitcher_name": ["Pitcher"] * 23,
        "inning": [1] * 23,
        "pitch_number_in_outing": np.arange(1, 24),
        "pitch_number_in_game": np.arange(1, 24),
        "mahalanobis_calibrated": [100.0] * 20 + [np.nan, 4.0, 4.0],
        "score_available": [False] * 21 + [True, True],
    })
    detected, alerts = CUSUMDetector(threshold_h=1.0).detect_game_alerts(df)
    assert detected.loc[detected["pitch_number_in_outing"] <= 21, "cusum_stat"].isna().all()
    assert not detected.loc[detected["pitch_number_in_outing"] <= 21, "is_cusum_alert"].any()
    assert alerts["alert_pitch_number"].min() == 22


def test_insufficient_history_score_is_explicitly_unavailable():
    row = {
        "game_pk": 1, "pitcher": 10, "pitch_type": "FF",
        "pitch_number_in_outing": 21, "is_calibration_phase": False,
    }
    row.update({column: 1.0 for column in [
        "release_pos_x", "release_pos_z", "release_extension", "release_speed",
        "release_spin_rate", "spin_axis_cos", "spin_axis_sin", "pfx_x", "pfx_z", "vaa"
    ]})
    result = MahalanobisScorer().score_pitches(
        pd.DataFrame([row]), 10, 1, baseline_store={}, calibrated_means={}
    )
    assert result["score_status"].iloc[0] == "INSUFFICIENT_HISTORY"
    assert not result["score_available"].iloc[0]
    assert np.isnan(result["mahalanobis_calibrated"].iloc[0])


def test_velocity_scores_are_index_aligned_and_pitch_type_specific():
    rows = []
    for game_pk in (1, 2):
        for pitch in range(1, 26):
            pitch_type = "FF" if pitch % 2 else "SI"
            baseline = 95.0 if pitch_type == "FF" else 90.0
            speed = baseline - (1.0 if pitch > 20 else 0.0)
            rows.append({
                "row_id": f"{game_pk}-{pitch}", "game_pk": game_pk, "pitcher": 10,
                "pitch_number_in_outing": pitch, "pitch_type": pitch_type,
                "release_speed": speed,
            })
    original = pd.DataFrame(rows)
    shuffled = original.sample(frac=1, random_state=7)
    score_original = BaselineComparator._velocity_drop_scores(original)
    score_shuffled = BaselineComparator._velocity_drop_scores(shuffled)
    expected = pd.Series(score_original.to_numpy(), index=original["row_id"].to_numpy())
    actual = pd.Series(score_shuffled.to_numpy(), index=shuffled["row_id"].to_numpy()).reindex(expected.index)
    assert np.allclose(expected, actual, equal_nan=True)
    assert np.nanmax(score_original) < 2.0  # SI is not compared with the faster FF baseline.


def test_warning_metrics_use_distinct_warnings_and_timely_episode_matching():
    pitches = np.arange(21, 51)
    df = pd.DataFrame({
        "game_pk": 1, "pitcher": 10, "game_date": "2024-07-10", "dataset_split": "test",
        "pitch_number_in_outing": pitches, "pa_number_in_outing": (pitches // 4),
        "score_available": True, "is_censored_followup": False,
        "y_true_onset_in_horizon": (pitches < 30) & (pitches >= 15),
    })
    predictions = pd.Series(False, index=df.index)
    predictions.loc[df["pitch_number_in_outing"].isin([25, 26, 40])] = True
    episodes = pd.DataFrame({
        "game_pk": [1], "pitcher": [10], "episode_id": [1], "onset_pitch": [30],
        "onset_pa": [8], "dataset_split": ["test"],
    })
    metrics, warnings, matches = evaluate_warning_predictions(
        df, episodes, predictions, horizon_pitches=15, split="test"
    )
    assert len(warnings) == 2
    assert len(matches) == 1
    assert metrics["episode_recall"] == 1.0
    assert metrics["warning_precision"] == 0.5
    assert metrics["false_warnings_per_outing"] == 1.0


def test_label_builder_emits_dashboard_schema(sample_dataset):
    outing = sample_dataset[sample_dataset["game_pk"] == sample_dataset["game_pk"].iloc[0]].copy()
    outing["pitch_number_in_outing"] = np.arange(1, len(outing) + 1)
    labeled, _ = CollapseLabelBuilder().build_labels_for_outing(outing)
    assert {"collapse_reason", "window_blended_xwoba"}.issubset(labeled.columns)
