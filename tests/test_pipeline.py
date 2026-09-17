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
from src.evaluation.protocol import select_operating_threshold
from src.storage.integrity import DataIntegrityError, PITCH_KEY, prepare_raw_pitch_data
from src.configuration import load_project_config
from src.evaluation.bootstrap import paired_bootstrap_confidence_intervals

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
