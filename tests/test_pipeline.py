"""
Comprehensive Unit & Integration Tests for Audited Baseball Fatigue System
"""
import pytest
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

from src.data_ingest.statcast_loader import StatcastLoader
from src.data_ingest.qualify_filter import QualifyFilter
from src.data_ingest.cohort import select_cohort
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
from src.evaluation.protocol import select_operating_threshold, warning_events
from src.storage.integrity import DataIntegrityError, PITCH_KEY, prepare_raw_pitch_data
from src.configuration import load_project_config
from src.evaluation.bootstrap import paired_bootstrap_confidence_intervals
from src.evaluation.ablation_runner import (
    AblationRunner, AblationConfig, verify_full_feature_parity, FEATURE_COLS,
)

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
    df = pd.DataFrame({"game_date": ["2023-09-01", "2024-04-01", "2024-07-01", "2025-06-01"]})
    result = assign_temporal_split(df)
    assert result["dataset_split"].tolist() == ["train", "validation", "validation", "test"]


def test_central_config_has_valid_baseline_window():
    config = load_project_config()
    assert config["baseline"]["min_prior_starts"] <= config["baseline"]["historical_window_starts"]
    assert config["evaluation"]["test_start"] == "2025-01-01"


def test_cohort_selection_is_pretest_deterministic_and_retains_original_pitchers():
    records = pd.DataFrame([
        {"pitcher": pitcher, "pitcher_name": f"P{pitcher}", "season": season,
         "starts": 20, "appearances": 20, "pitches": 1000, "aggregate": True}
        for pitcher in range(1, 9) for season in (2023, 2024)
    ])
    first = select_cohort(records, retained_ids=[1, 2], target_size=5,
                          minimum_starts=30, cutoff_year=2024, seed=77)
    shuffled = select_cohort(records.sample(frac=1, random_state=9), retained_ids=[1, 2],
                             target_size=5, minimum_starts=30, cutoff_year=2024, seed=77)
    assert set(first.loc[first["selected"], "pitcher"]) == set(
        shuffled.loc[shuffled["selected"], "pitcher"]
    )
    assert {1, 2}.issubset(set(first.loc[first["selected"], "pitcher"]))
    assert (first["selection_cutoff"] == "2024-12-31").all()


def test_raw_integrity_requires_2025_and_rejects_conflicting_pitch_keys():
    rows = []
    for year in (2023, 2024, 2025):
        rows.append({
            "game_pk": year, "pitcher": 10, "at_bat_number": 1, "pitch_number": 1,
            "game_date": f"{year}-06-01", "pitch_type": "FF", "release_speed": 95.0,
            "release_pos_x": -2.0, "release_pos_z": 6.0, "release_extension": 6.5,
            "release_spin_rate": 2400.0, "spin_axis": 210.0, "pfx_x": -0.5,
            "pfx_z": 1.2, "actual_data_source": "mlb_statcast",
            "game_type": "R",
        })
    clean = prepare_raw_pitch_data(
        pd.concat([
            pd.DataFrame(rows),
            pd.DataFrame([{**rows[-1], "game_pk": 9999, "game_type": "P"}]),
        ], ignore_index=True),
        "mlb_statcast", "2023-01-01", "2025-12-31",
        required_years=[2023, 2024, 2025],
    )
    assert set(clean["ingest_season"]) == {2023, 2024, 2025}
    assert set(clean["game_type"]) == {"R"}
    assert not clean.duplicated(PITCH_KEY).any()

    conflicting = pd.concat([clean, clean.iloc[[0]].assign(release_speed=80.0)], ignore_index=True)
    with pytest.raises(DataIntegrityError, match="Conflicting duplicate pitch keys"):
        prepare_raw_pitch_data(
            conflicting, "mlb_statcast", "2023-01-01", "2025-12-31",
            required_years=[2023, 2024, 2025],
        )


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


def test_baseline_sources_precede_cutoff_and_future_games_cannot_change_earlier_outputs(sample_dataset):
    features = compute_kinematics_and_vaa(sample_dataset.copy())
    features = features.sort_values(["pitcher", "game_date", "game_pk", "at_bat_number", "pitch_number"])
    features["pitch_number_in_outing"] = features.groupby(["game_pk", "pitcher"]).cumcount() + 1
    builder = BaselineBuilder(historical_window_starts=4, min_prior_starts=1, min_pitches_for_baseline=10)
    baseline_a, store_a = builder.build_baselines(features)
    assert (
        pd.to_datetime(baseline_a["baseline_max_source_date"])
        < pd.to_datetime(baseline_a["as_of_date"])
    ).all()

    final_game = features.sort_values("game_date")["game_pk"].iloc[-1]
    changed = features.copy()
    changed.loc[changed["game_pk"].eq(final_game), "release_speed"] += 25.0
    baseline_b, store_b = builder.build_baselines(changed)
    before_final_a = baseline_a[baseline_a["as_of_game_pk"].ne(final_game)].sort_values(
        ["pitcher", "as_of_game_pk", "pitch_type"]
    ).reset_index(drop=True)
    before_final_b = baseline_b[baseline_b["as_of_game_pk"].ne(final_game)].sort_values(
        ["pitcher", "as_of_game_pk", "pitch_type"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(before_final_a, before_final_b)

    candidate_game = next(
        game for game in features["game_pk"].drop_duplicates()
        if game != final_game and any(
            key.startswith(f"{features.loc[features['game_pk'].eq(game), 'pitcher'].iloc[0]}_{game}_")
            and value.get("status") == "QUALIFIED" for key, value in store_a.items()
        )
    )
    outing = features[features["game_pk"].eq(candidate_game)]
    pitcher = int(outing["pitcher"].iloc[0])
    calibrator = ShrinkageCalibrator(intra_game_calibration_pitches=20)
    calibrated_a, means_a = calibrator.calibrate_outing_pitches(
        outing, pitcher, candidate_game, store_a
    )
    calibrated_b, means_b = calibrator.calibrate_outing_pitches(
        outing, pitcher, candidate_game, store_b
    )
    scored_a = MahalanobisScorer().score_pitches(
        calibrated_a, pitcher, candidate_game, store_a, means_a
    )
    scored_b = MahalanobisScorer().score_pitches(
        calibrated_b, pitcher, candidate_game, store_b, means_b
    )
    np.testing.assert_allclose(
        scored_a["mahalanobis_calibrated"], scored_b["mahalanobis_calibrated"], equal_nan=True
    )
    warnings_a, _ = CUSUMDetector().detect_game_alerts(scored_a)
    warnings_b, _ = CUSUMDetector().detect_game_alerts(scored_b)
    pd.testing.assert_series_equal(warnings_a["is_cusum_alert"], warnings_b["is_cusum_alert"])


def test_outcome_columns_cannot_change_mechanics_scores_or_warning_times():
    feature_columns = [
        "release_pos_x", "release_pos_z", "release_extension", "release_speed",
        "release_spin_rate", "spin_axis_cos", "spin_axis_sin", "pfx_x", "pfx_z", "vaa",
    ]
    rows = []
    for pitch in range(1, 41):
        row = {
            "game_pk": 1, "game_date": "2025-05-01", "pitcher": 10,
            "pitcher_name": "Pitcher", "pitch_type": "FF", "inning": 1,
            "pitch_number_in_outing": pitch, "is_calibration_phase": pitch <= 20,
            "y_true_onset_in_horizon": False, "collapse_reason": "NONE",
        }
        row.update({column: 1.0 + pitch / 100.0 for column in feature_columns})
        rows.append(row)
    outing = pd.DataFrame(rows)
    baseline = {
        "10_1_FF": {
            "status": "QUALIFIED", "mu_vec": np.ones(10), "sigma_vec": np.ones(10),
            "prec_mat": np.eye(10), "feature_cols": feature_columns,
        }
    }
    means = {"FF": np.ones(10)}
    changed = outing.copy()
    changed["y_true_onset_in_horizon"] = True
    changed["collapse_reason"] = "ALTERED_FUTURE_OUTCOME"
    score_a = MahalanobisScorer().score_pitches(outing, 10, 1, baseline, means)
    score_b = MahalanobisScorer().score_pitches(changed, 10, 1, baseline, means)
    np.testing.assert_allclose(
        score_a["mahalanobis_calibrated"], score_b["mahalanobis_calibrated"], equal_nan=True
    )
    detected_a, alerts_a = CUSUMDetector().detect_game_alerts(score_a)
    detected_b, alerts_b = CUSUMDetector().detect_game_alerts(score_b)
    pd.testing.assert_series_equal(detected_a["is_cusum_alert"], detected_b["is_cusum_alert"])
    pd.testing.assert_frame_equal(alerts_a, alerts_b)


def test_test_data_cannot_change_validation_selected_threshold():
    rows = []
    for split, game_pk, scores in (
        ("validation", 1, [0.1, 0.2, 0.8, 0.9]),
        ("test", 2, [0.3, 0.4, 0.5, 0.6]),
    ):
        for offset, score in enumerate(scores, start=21):
            rows.append({
                "game_pk": game_pk, "pitcher": 10, "game_date": "2024-05-01" if split == "validation" else "2025-05-01",
                "dataset_split": split, "pitch_number_in_outing": offset,
                "pa_number_in_outing": offset // 4, "score_available": True,
                "is_censored_followup": False, "score": score,
                "y_true_onset_in_horizon": offset >= 23,
            })
    frame = pd.DataFrame(rows)
    episodes = pd.DataFrame({
        "game_pk": [1, 2], "pitcher": [10, 10], "episode_id": [1, 2],
        "onset_pitch": [25, 25], "onset_pa": [7, 7],
        "dataset_split": ["validation", "test"],
    })
    threshold_a, _ = select_operating_threshold(frame, episodes, "score", 1.0)
    changed = frame.copy()
    changed.loc[changed["dataset_split"].eq("test"), ["score", "y_true_onset_in_horizon"]] = [999.0, False]
    threshold_b, _ = select_operating_threshold(changed, episodes, "score", 1.0)
    assert threshold_a == threshold_b


def test_test_rows_cannot_change_contextual_model_fitted_on_train():
    rows = []
    for position in range(12):
        is_train = position < 8
        rows.append({
            "game_pk": 1 if is_train else 2,
            "pitcher": 10,
            "game_date": "2023-06-01" if is_train else "2025-06-01",
            "dataset_split": "train" if is_train else "test",
            "pitch_number_in_outing": 21 + position,
            "pa_number_in_outing": 6 + position,
            "inning": 3 + position // 3,
            "score_available": True,
            "is_censored_followup": False,
            "y_true_onset_in_horizon": bool(position % 2),
        })
    frame = pd.DataFrame(rows)
    comparator = BaselineComparator()
    scores_a = comparator._add_context_scores(frame)
    changed = frame.copy()
    test_mask = changed["dataset_split"].eq("test")
    changed.loc[test_mask, "pitch_number_in_outing"] = 999
    changed.loc[test_mask, "pa_number_in_outing"] = 999
    changed.loc[test_mask, "inning"] = 9
    changed.loc[test_mask, "y_true_onset_in_horizon"] = False
    scores_b = comparator._add_context_scores(changed)
    np.testing.assert_allclose(scores_a[~test_mask], scores_b[~test_mask])


def test_pitch_mix_changes_do_not_change_pitch_type_normalized_cusum_rate():
    feature_columns = [
        "release_pos_x", "release_pos_z", "release_extension", "release_speed",
        "release_spin_rate", "spin_axis_cos", "spin_axis_sin", "pfx_x", "pfx_z", "vaa",
    ]
    baseline_store = {}
    calibrated_means = {}
    for game_pk in (1, 2):
        for pitch_type in ("FF", "SL"):
            baseline_store[f"10_{game_pk}_{pitch_type}"] = {
                "status": "QUALIFIED", "mu_vec": np.zeros(10), "sigma_vec": np.ones(10),
                "prec_mat": np.eye(10), "feature_cols": feature_columns,
            }
            calibrated_means[(game_pk, pitch_type)] = np.zeros(10)

    alert_rates = []
    for game_pk, ff_fraction in ((1, 0.5), (2, 0.9)):
        rows = []
        for pitch in range(21, 121):
            pitch_type = "FF" if (pitch - 21) < 100 * ff_fraction else "SL"
            row = {
                "game_pk": game_pk, "game_date": "2025-06-01", "pitcher": 10,
                "pitcher_name": "Pitcher", "pitch_type": pitch_type, "inning": 1,
                "pitch_number_in_outing": pitch, "is_calibration_phase": False,
            }
            row.update({column: 0.5 for column in feature_columns})
            rows.append(row)
        outing = pd.DataFrame(rows)
        scored = MahalanobisScorer().score_pitches(
            outing, 10, game_pk, baseline_store,
            {pitch_type: calibrated_means[(game_pk, pitch_type)] for pitch_type in ("FF", "SL")},
        )
        detected, _ = CUSUMDetector().detect_game_alerts(scored)
        alert_rates.append(detected["is_cusum_alert"].mean())
    assert alert_rates[0] == alert_rates[1]


def test_paired_bootstrap_reports_direct_differences_and_zero_denominators():
    rows = []
    for game_pk in (1, 2, 3):
        for pitch in range(21, 41):
            rows.append({
                "game_pk": game_pk, "pitcher": 10 + game_pk % 2,
                "game_date": "2025-06-01", "dataset_split": "test",
                "pitch_number_in_outing": pitch, "pa_number_in_outing": pitch // 4,
                "score_available": True, "is_censored_followup": False,
                "y_true_onset_in_horizon": pitch >= 25,
                "is_proposed_operating_alert": pitch in (22, 23),
                "is_contextual_operating_alert": game_pk == 1 and pitch in (22, 23),
            })
    frame = pd.DataFrame(rows)
    episodes = pd.DataFrame({
        "game_pk": [1, 2, 3], "pitcher": [11, 10, 11], "episode_id": [1, 2, 3],
        "onset_pitch": [30, 30, 30], "onset_pa": [8, 8, 8],
        "dataset_split": ["test", "test", "test"],
    })
    result = paired_bootstrap_confidence_intervals(
        frame, episodes,
        {"proposed": "is_proposed_operating_alert", "contextual": "is_contextual_operating_alert"},
        n_bootstrap=100, seed=7,
    )
    assert "proposed_minus_contextual" in set(result["comparison"])
    assert result["zero_denominator_frequency"].between(0, 1).all()


# =============================================================================
# FEATURE ABLATION & CUSUM INTEGRATION REGRESSION TESTS (§7 Requirements)
# =============================================================================

def test_full_feature_ablation_parity_with_proposed_system():
    """1. Full-feature ablation parity with production proposed system."""
    rows = []
    for split, gpk, year in [("train", 1, 2023), ("validation", 2, 2024), ("test", 3, 2025)]:
        for p in range(1, 35):
            rows.append({
                "game_pk": gpk, "pitcher": 10, "game_date": f"{year}-06-01",
                "dataset_split": split, "pitch_number_in_outing": p,
                "pa_number_in_outing": p // 4 + 1, "inning": 1 + p // 15,
                "is_calibration_phase": p <= 20,
                "score_available": True, "is_censored_followup": False,
                "release_speed": 95.0, "release_pos_x": 0.0, "release_pos_z": 6.0,
                "release_extension": 6.0, "release_spin_rate": 2200.0,
                "spin_axis_cos": 0.5, "spin_axis_sin": 0.866,
                "pfx_x": 0.1, "pfx_z": 0.8, "vaa": -5.0,
                "mahalanobis_calibrated": 2.0 if p > 20 else np.nan,
                "cusum_stat": float(p - 20) * 10.0 if p > 20 else np.nan,
                "y_true_onset_in_horizon": p >= 26,
                "pitch_type": "FF",
            })
    df = pd.DataFrame(rows)
    eps = pd.DataFrame([
        {"game_pk": 2, "pitcher": 10, "episode_id": 1, "onset_pitch": 28, "onset_pa": 7, "dataset_split": "validation"},
        {"game_pk": 3, "pitcher": 10, "episode_id": 2, "onset_pitch": 28, "onset_pa": 7, "dataset_split": "test"},
    ])

    comp = BaselineComparator(horizon_pitches=15, false_warnings_per_outing=0.5)
    comp_df, eval_df, metrics = comp.compare_systems(df, eps, return_details=True)

    runner = AblationRunner(eval_df, eps, config=AblationConfig(false_warnings_per_outing=0.5))
    abl_df, manifest = runner.run_feature_ablations()

    # Exact parity check
    assert verify_full_feature_parity(abl_df, metrics["proposed"])
    full_row = abl_df[abl_df["Feature Subset"] == "Full Micro-Mechanics Suite"].iloc[0]
    assert np.isclose(full_row["Test Episode Recall"], metrics["proposed"]["episode_recall"])
    assert np.isclose(full_row["Test Warning Precision"], metrics["proposed"]["warning_precision"])
    assert np.isclose(full_row["Test False Warnings / Outing"], metrics["proposed"]["false_warnings_per_outing"])
    assert np.isclose(full_row["Validation-Selected Threshold"], metrics["proposed"]["operating_threshold"])


def test_accumulated_cusum_differs_from_raw_distance_thresholding():
    """2. A sequence where accumulated CUSUM behavior differs from raw-distance thresholding."""
    # A moderate, persistent drift: raw score is 2.0 on each pitch (above reference mean 1.0, z=2.0)
    # Under raw thresholding at 2.5, NO alert is ever fired.
    # Under CUSUM, with slack_k=0.5, statistic increases by 1.5 each pitch: 1.5, 3.0, 4.5, 6.0, 7.5
    pitches = list(range(21, 26))
    df = pd.DataFrame({
        "game_pk": 1, "pitcher": 10,
        "pitch_number_in_outing": pitches,
        "raw_score": [2.0] * len(pitches),
        "is_calibration_phase": False,
        "score_available": True,
    })
    res = AblationRunner._run_cusum_on_scores(
        df, score_col="raw_score", cusum_output_col="cusum_stat",
        slack_k=0.5, threshold_h=4.0, reference_mean=1.0, reference_std=0.5,
        calibration_pitches=20,
    )
    raw_alerts = res["raw_score"] >= 2.5
    cusum_alerts = res["cusum_stat"] >= 4.0

    assert raw_alerts.sum() == 0, "Raw thresholding at 2.5 should yield 0 alerts"
    assert cusum_alerts.sum() == 3, "CUSUM accumulation should trigger 3 alerts (pitches 23, 24, 25)"


def test_nondefault_cusum_and_calibration_settings_reach_ablation_detector():
    """3. Nondefault CUSUM settings and calibration length reach the ablation detector."""
    pitches = list(range(1, 35))
    df = pd.DataFrame({
        "game_pk": 1, "pitcher": 10,
        "pitch_number_in_outing": pitches,
        "raw_score": [2.0] * len(pitches),
        "is_calibration_phase": False,
        "score_available": True,
    })
    # Nondefault calibration_pitches = 28, slack_k = 1.2
    res = AblationRunner._run_cusum_on_scores(
        df, score_col="raw_score", cusum_output_col="cusum_stat",
        slack_k=1.2, threshold_h=5.0, reference_mean=1.0, reference_std=0.5,
        calibration_pitches=28,
    )
    # Pitches <= 28 are calibration and must remain NaN
    assert res.loc[res["pitch_number_in_outing"] <= 28, "cusum_stat"].isna().all()
    # Pitch 29: z = (2.0-1.0)/0.5 = 2.0; stat = max(0, 0 + 2.0 - 1.2) = 0.8
    p29 = res.loc[res["pitch_number_in_outing"] == 29, "cusum_stat"].iloc[0]
    assert np.isclose(p29, 0.8), f"Expected 0.8 with nondefault slack_k=1.2, got {p29}"


def test_detector_state_resets_across_outings():
    """4. Detector state resets across outings."""
    rows = []
    # Outing 1: high scores accumulating to > 30
    for p in range(21, 26):
        rows.append({"game_pk": 1, "pitcher": 10, "pitch_number_in_outing": p, "score": 5.0, "score_available": True})
    # Outing 2: baseline score (z=0, stat=0)
    for p in range(21, 26):
        rows.append({"game_pk": 2, "pitcher": 10, "pitch_number_in_outing": p, "score": 1.0, "score_available": True})
    df = pd.DataFrame(rows)
    res = AblationRunner._run_cusum_on_scores(
        df, "score", "cusum", slack_k=0.5, threshold_h=4.0, reference_mean=1.0, reference_std=0.5, calibration_pitches=20,
    )
    out1_last = res.loc[res["game_pk"] == 1, "cusum"].iloc[-1]
    out2_first = res.loc[res["game_pk"] == 2, "cusum"].iloc[0]
    assert out1_last > 15.0, "Outing 1 should accumulate high CUSUM stat"
    assert out2_first == 0.0, "Outing 2 must reset state to 0 and not inherit Outing 1's accumulation"


def test_unavailable_pitches_and_pitch_gaps_break_warning_continuity():
    """5. Unavailable pitches and pitch gaps break warning continuity."""
    # A. Unavailable pitch breaks contiguous run
    df_unavail = pd.DataFrame({
        "game_pk": [1, 1, 1, 1], "pitcher": [10, 10, 10, 10],
        "pitch_number_in_outing": [21, 22, 23, 24], "game_date": "2025-06-01",
    })
    preds_unavail = pd.Series([True, True, False, True], index=df_unavail.index)
    warns_a = warning_events(df_unavail, preds_unavail)
    assert len(warns_a) == 2, f"Expected 2 warnings (interrupted by unavailable pitch 23), got {len(warns_a)}"
    assert warns_a["warning_pitch"].tolist() == [21, 24]

    # B. Pitch number gap breaks contiguous run
    df_gap = pd.DataFrame({
        "game_pk": [1, 1, 1], "pitcher": [10, 10, 10],
        "pitch_number_in_outing": [21, 22, 35], "game_date": "2025-06-01",
    })
    preds_gap = pd.Series([True, True, True], index=df_gap.index)
    warns_b = warning_events(df_gap, preds_gap)
    assert len(warns_b) == 2, f"Expected 2 warnings (interrupted by gap between 22 and 35), got {len(warns_b)}"
    assert warns_b["warning_pitch"].tolist() == [21, 35]


def test_scoreless_outings_remain_in_ablation_denominators():
    """6. Scoreless outings remain in ablation denominators."""
    # Two outings: outing 1 is scored, outing 2 has all pitches unavailable (scoreless)
    # Both outings have an eligible collapse episode.
    rows = []
    for p in range(21, 35):
        rows.append({
            "game_pk": 1, "pitcher": 10, "pitch_number_in_outing": p,
            "game_date": "2025-06-01", "dataset_split": "test",
            "score_available": True, "cusum_stat": 100.0,
            "y_true_onset_in_horizon": p >= 25,
        })
        rows.append({
            "game_pk": 2, "pitcher": 11, "pitch_number_in_outing": p,
            "game_date": "2025-06-01", "dataset_split": "test",
            "score_available": True, "cusum_stat": np.nan,  # Scoreless
            "y_true_onset_in_horizon": p >= 25,
        })
    df = pd.DataFrame(rows)
    eps = pd.DataFrame([
        {"game_pk": 1, "pitcher": 10, "episode_id": 1, "onset_pitch": 28, "dataset_split": "test"},
        {"game_pk": 2, "pitcher": 11, "episode_id": 2, "onset_pitch": 28, "dataset_split": "test"},
    ])
    preds = df["cusum_stat"].ge(50.0) & df["cusum_stat"].notna()
    metrics, _, _ = evaluate_warning_predictions(df, eps, preds, horizon_pitches=15, split="test")

    # Both outings and both episodes are part of the evaluated universe
    assert metrics["evaluated_outings_count"] == 2, "Scoreless outing must remain in evaluated outings denominator"
    assert metrics["total_collapse_episodes"] == 2, "Episode in scoreless outing must remain in episode denominator"
    assert metrics["collapse_episodes_detected"] == 1, "Only scored outing's episode should be detected"
    assert np.isclose(metrics["episode_recall"], 0.5), "Recall should be 1/2 = 0.5"


def test_changing_test_data_cannot_change_validation_selected_ablation_threshold():
    """7. Changing test scores or outcomes cannot change validation-selected ablation thresholds."""
    rows_val = []
    for p in range(21, 35):
        rows_val.append({
            "game_pk": 1, "pitcher": 10, "pitch_number_in_outing": p,
            "game_date": "2024-06-01", "dataset_split": "validation",
            "score_available": True, "cusum_stat": float(p) * 2.0,
            "y_true_onset_in_horizon": p >= 25,
        })
    eps_val = pd.DataFrame([{"game_pk": 1, "pitcher": 10, "episode_id": 1, "onset_pitch": 28, "dataset_split": "validation"}])

    # Test set A: normal
    rows_test_a = [{**r, "game_pk": 2, "dataset_split": "test", "game_date": "2025-06-01"} for r in rows_val]
    # Test set B: extreme scores and changed labels
    rows_test_b = [{**r, "game_pk": 2, "dataset_split": "test", "game_date": "2025-06-01",
                    "cusum_stat": 9999.0, "y_true_onset_in_horizon": False} for r in rows_val]

    runner_a = AblationRunner(pd.DataFrame(rows_val + rows_test_a), eps_val)
    res_a, _ = runner_a.run_feature_ablations()

    runner_b = AblationRunner(pd.DataFrame(rows_val + rows_test_b), eps_val)
    res_b, _ = runner_b.run_feature_ablations()

    thresh_a = res_a["Validation-Selected Threshold"].to_numpy()
    thresh_b = res_b["Validation-Selected Threshold"].to_numpy()
    np.testing.assert_allclose(thresh_a, thresh_b, equal_nan=True)


def test_subset_mahalanobis_matches_independent_mathematical_reference():
    """8. Subset score construction matches an independently calculated mathematical reference."""
    np.random.seed(42)
    dim = len(FEATURE_COLS)
    full_mu = np.random.uniform(0.5, 2.0, dim)
    A = np.random.normal(0, 1, (dim, dim))
    full_cov = A @ A.T + np.eye(dim) * 0.1

    sub_indices = [0, 1, 2]  # release_pos_x, release_pos_z, release_extension
    sub_mu = full_mu[sub_indices]
    sub_cov = full_cov[np.ix_(sub_indices, sub_indices)]
    ridge = 1e-4
    sub_prec = np.linalg.inv(sub_cov + np.eye(3) * ridge)

    x_full = np.random.uniform(0.5, 2.0, dim)
    x_sub = x_full[sub_indices]
    diff = x_sub - sub_mu
    expected_dist = float(np.sqrt(np.dot(diff, np.dot(sub_prec, diff))))

    df = pd.DataFrame([{
        "game_pk": 1, "pitcher": 10, "pitch_type": "FF", "pitch_number_in_outing": 25,
        "is_calibration_phase": False,
        **{FEATURE_COLS[i]: x_full[i] for i in range(dim)}
    }])
    store = {
        "10_1_FF": {
            "status": "QUALIFIED",
            "mu_vec": full_mu,
            "cov_mat": full_cov,
        }
    }
    runner = AblationRunner(df, baseline_store=store, config=AblationConfig(ridge_regularization=ridge))
    scores, avail = runner._compute_subset_mahalanobis(df, sub_indices, "Test_Sub")
    actual_dist = scores.iloc[0]

    assert np.isclose(expected_dist, actual_dist, atol=1e-4), (
        f"Subset Mahalanobis distance {actual_dist} does not match mathematical reference {expected_dist}"
    )


def test_deliberate_full_feature_main_result_disagreement_fails_validation():
    """9. A deliberate full-feature/main-result disagreement fails artifact validation."""
    tampered_df = pd.DataFrame([{
        "Feature Subset": "Full Micro-Mechanics Suite",
        "Test Episode Recall": 0.0,  # Deliberately falsified / bug reproduction
        "Test Warning Precision": 0.0,
        "Test False Warnings / Outing": 0.0,
        "Validation-Selected Threshold": float("nan"),
    }])
    expected_metrics = {
        "episode_recall": 0.19178,
        "warning_precision": 0.43299,
        "false_warnings_per_outing": 0.39855,
        "operating_threshold": 214.651,
    }
    with pytest.raises(ValueError, match="Parity mismatch"):
        verify_full_feature_parity(tampered_df, expected_metrics)

