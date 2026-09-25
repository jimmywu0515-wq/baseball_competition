"""Regressions for failures found in the 2026 project review."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.ablation_runner import AblationRunner, verify_full_feature_parity
from src.evaluation.baseline_comparator import BaselineComparator
from src.feature_engineering.mechanics_features import compute_kinematics_and_vaa
from src.feature_engineering.rolling_stats import compute_rolling_features


def test_full_feature_parity_rejects_missing_values():
    row = pd.DataFrame([{
        "Feature Subset": "Full Micro-Mechanics Suite",
        "Test Episode Recall": np.nan,
        "Test Warning Precision": 0.5,
        "Test False Warnings / Outing": 0.1,
    }])
    with pytest.raises(ValueError, match="Parity mismatch"):
        verify_full_feature_parity(row, {
            "episode_recall": 0.2,
            "warning_precision": 0.5,
            "false_warnings_per_outing": 0.1,
        })


def test_subset_cusum_resets_by_outing_and_skips_unavailable_pitches():
    frame = pd.DataFrame({
        "game_pk": [1, 1, 1, 2], "pitcher": [10, 10, 10, 10],
        "pitch_number_in_outing": [21, 22, 23, 21],
        "subset_score": [2.0, np.nan, 2.0, 2.0],
    })
    scored = AblationRunner._run_cusum_on_scores(
        frame, "subset_score", "subset_cusum",
        slack_k=0.5, threshold_h=4.5, reference_mean=1.0,
        reference_std=0.5, calibration_pitches=20,
    )
    np.testing.assert_allclose(
        scored["subset_cusum"].to_numpy(), [1.5, np.nan, 3.0, 1.5],
        equal_nan=True,
    )


def test_missing_measurements_stay_missing_and_timing_is_not_fabricated():
    frame = pd.DataFrame({
        "game_pk": [1, 1], "pitcher": [10, 10],
        "pitch_number_in_outing": [1, 2],
        "release_pos_x": [np.nan, 1.0],
        "release_pos_z": [np.nan, 5.0],
        "release_speed": [90.0, 91.0],
        "spin_axis": [np.nan, 180.0],
        "vy0": [np.nan, -130.0], "ay": [np.nan, 28.0],
        "vz0": [np.nan, -6.0], "az": [np.nan, -20.0],
        "vx0": [np.nan, 4.0], "ax": [np.nan, -10.0],
    })
    features = compute_rolling_features(compute_kinematics_and_vaa(frame))
    assert features.loc[0, ["spin_axis_cos", "spin_axis_sin", "vaa", "haa",
                            "release_3d_dist_from_origin"]].isna().all()
    assert features["pitch_interval_sec"].isna().all()


def test_rolling_spin_dispersion_respects_angle_wraparound():
    frame = pd.DataFrame({
        "game_pk": [1, 1], "pitcher": [10, 10],
        "pitch_number_in_outing": [1, 2],
        "release_speed": [90.0, 90.0],
        "release_pos_x": [1.0, 1.0], "release_pos_z": [5.0, 5.0],
        "spin_axis": [359.0, 1.0],
    })
    result = compute_rolling_features(frame)
    assert pd.isna(result.loc[0, "roll5_spin_axis_shift"])
    assert result.loc[1, "roll5_spin_axis_shift"] < 2.0


def test_comparison_can_report_no_velocity_score():
    rows = []
    for split, date, game in (("train", "2023-06-01", 1),
                              ("validation", "2024-06-01", 2),
                              ("test", "2025-06-01", 3)):
        for pitch in range(21, 25):
            rows.append({
                "game_pk": game, "pitcher": 10, "pitcher_name": "P",
                "game_date": date, "dataset_split": split,
                "pitch_number_in_outing": pitch, "pa_number_in_outing": pitch // 4,
                "inning": 3, "pitch_type": "SL", "release_speed": 85.0,
                "cusum_stat": float(pitch), "score_available": True,
                "is_calibration_phase": False, "is_censored_followup": False,
                "y_true_onset_in_horizon": pitch % 2 == 0,
                "actual_data_source": "simulation_benchmark",
            })
    comparison = BaselineComparator().compare_systems(pd.DataFrame(rows), pd.DataFrame())
    velocity = comparison.loc[comparison["Model Key"].eq("velocity")].iloc[0]
    assert velocity["Threshold Status"] == "no_alert"
    assert velocity["Test Outings With Available Score"] == 0
