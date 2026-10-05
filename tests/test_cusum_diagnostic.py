"""Scientific safeguards for the exploratory CUSUM diagnostic."""
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.diagnostics.check_cusum_pitch_count_correlation import (
    compare_variant, episodes_from_labels, estimate_reference_from_train, json_safe, run_check,
    run_cusum, select_and_evaluate, shuffle_scores,
)
from src.changepoint_detector.cusum_detector import CUSUMDetector
from src.configuration import load_project_config
from src.evaluation.ablation_runner import AblationConfig
from src.evaluation.protocol import select_operating_threshold


@pytest.fixture
def warehouse_root():
    # Use ordinary directory permissions: pytest's restricted temp directories
    # are inaccessible under this Windows workspace's managed permission profile.
    outputs = Path(__file__).resolve().parents[1] / "outputs"
    root = outputs / f".smoke_cusum_{uuid.uuid4().hex}"
    root.mkdir()
    try:
        yield root
    finally:
        assert root.resolve().parent == outputs.resolve()
        # Windows/OneDrive can retain directory handles after Parquet reads.
        # A locked temporary directory must not turn scientific assertions into
        # a teardown failure; these uniquely named fixtures are ignored by Git.
        shutil.rmtree(root, ignore_errors=True)


def rows():
    frame = pd.DataFrame([
        {"game_pk": game, "pitcher": 7, "pitch_number_in_outing": pitch,
         "pitch_type": "FF" if pitch % 2 else "SL", "inning": 3,
         "pa_number_in_outing": pitch // 4, "dataset_split": split,
         "game_date": pd.Timestamp(date), "is_calibration_phase": pitch <= 20,
         "is_censored_followup": pitch > 55, "score_available": pitch > 20,
         "mahalanobis_calibrated": 1.5 + (pitch % 7) / 5 if pitch > 20 else np.nan,
         "actual_data_source": "mlb_statcast", "run_id": "diagnostic_fixture",
         "protocol_sha256": "fixture_hash"}
        for game, split, date in [(1, "train", "2023-06-01"),
                                  (2, "validation", "2024-06-01"),
                                  (3, "test", "2025-06-01")]
        for pitch in range(1, 61)
    ])
    # Noncontiguous indices expose positional/index alignment mistakes.
    frame.index = np.arange(len(frame)) * 3 + 11
    return frame


def test_reference_uses_train_detector_inputs_without_outcome_censoring():
    frame = rows()
    frame.loc[frame.pitch_number_in_outing.eq(21), "score_available"] = False
    reference = estimate_reference_from_train(frame, "mahalanobis_calibrated", 20)
    expected = frame.loc[frame.dataset_split.eq("train") & frame.pitch_number_in_outing.gt(21),
                         "mahalanobis_calibrated"]
    assert reference["n_pitches"] == 39  # includes late pitches censored from evaluation
    assert reference["mean"] == pytest.approx(expected.mean())
    assert reference["std"] == pytest.approx(expected.std(ddof=1))
    altered = frame.copy()
    altered.loc[~altered.dataset_split.eq("train"), "mahalanobis_calibrated"] = 1e12
    assert estimate_reference_from_train(altered, "mahalanobis_calibrated", 20) == reference
    altered.loc[altered.dataset_split.eq("train"), "mahalanobis_calibrated"] = np.inf
    with pytest.raises(ValueError, match="Only 0"):
        estimate_reference_from_train(altered, "mahalanobis_calibrated", 20)
    altered.loc[altered.dataset_split.eq("train"), "mahalanobis_calibrated"] = 3.0
    with pytest.raises(ValueError, match="nonzero"):
        estimate_reference_from_train(altered, "mahalanobis_calibrated", 20)


def test_cusum_matches_production_across_missing_scores_and_boundaries():
    frame = rows()
    frame.loc[frame.pitch_number_in_outing.eq(30), "score_available"] = False
    frame.loc[frame.pitch_number_in_outing.eq(35), "mahalanobis_calibrated"] = np.nan
    cfg = AblationConfig()
    actual = run_cusum(frame, "mahalanobis_calibrated", cfg, 1.0, .5)
    for _, outing in frame.groupby(["game_pk", "pitcher"]):
        expected, _ = CUSUMDetector().detect_game_alerts(outing)
        np.testing.assert_allclose(actual.loc[outing.index, "_diagnostic_cusum"],
                                   expected.cusum_stat, equal_nan=True)


def test_shuffle_preserves_type_distributions_positions_labels_and_reproducibility():
    frame = rows()
    frame.loc[frame.pitch_number_in_outing.eq(25), "score_available"] = False
    shuffled = shuffle_scores(frame, "mahalanobis_calibrated", 20, 123)
    pd.testing.assert_frame_equal(shuffled, shuffle_scores(frame, "mahalanobis_calibrated", 20, 123))
    other_columns = frame.columns.drop("mahalanobis_calibrated")
    pd.testing.assert_frame_equal(frame[other_columns], shuffled[other_columns])
    pd.testing.assert_series_equal(frame.mahalanobis_calibrated.isna(), shuffled.mahalanobis_calibrated.isna())
    pd.testing.assert_series_equal(frame.loc[~frame.score_available, "mahalanobis_calibrated"],
                                   shuffled.loc[~frame.score_available, "mahalanobis_calibrated"])
    for _, group in frame.groupby(["game_pk", "pitcher", "pitch_type"]):
        np.testing.assert_allclose(np.sort(group.mahalanobis_calibrated),
                                   np.sort(shuffled.loc[group.index, "mahalanobis_calibrated"]), equal_nan=True)
    assert not frame.mahalanobis_calibrated.equals(shuffled.mahalanobis_calibrated)


def test_no_alert_threshold_and_scoreless_outing_keep_denominators():
    frame = rows()
    extra = frame.loc[frame.dataset_split.eq("validation")].copy()
    extra["game_pk"] = 4
    extra["score_available"] = False
    extra["mahalanobis_calibrated"] = np.nan
    frame = pd.concat([frame, extra], ignore_index=True)
    episodes = pd.DataFrame({"game_pk": [4], "pitcher": [7], "onset_pitch": [40],
                             "episode_id": [1], "dataset_split": ["validation"]})
    result = compare_variant(frame, episodes, "mahalanobis_calibrated",
                             AblationConfig(false_warnings_per_outing=0), [123])
    for name in ["original", "recalibrated"]:
        metrics = result[name]["validation_metrics"]
        assert metrics["total_qualified_outings"] == 2
        assert metrics["total_collapse_episodes"] == 1
        assert metrics["outings_without_available_score"] == 1
        assert metrics["episode_recall"] == 0
        assert result[name]["threshold_status"] == "no_alert"
        assert json_safe(result[name])["selected_threshold"] is None
        assert result["shuffle_control"]["results"][name]["replicates"][0]["total_warnings"] == 0
    json.dumps(json_safe(result), allow_nan=False)


def make_warehouse(root):
    config = load_project_config()
    (root / "config").mkdir()
    (root / "config/config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest = {
        "run_id": "diagnostic_fixture", "protocol_sha256": "fixture_hash",
        "actual_data_source": "mlb_statcast", "prior_2025_exposure_disclosure": "Previously inspected",
        "resolved_runtime": {"resolved_config": config},
    }
    (root / "outputs/real_data").mkdir(parents=True)
    (root / "outputs/real_data/protocol_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "data/gold").mkdir(parents=True)
    frame = rows()
    scored = run_cusum(frame, "mahalanobis_calibrated", AblationConfig(), 1.0, .5)
    frame["cusum_stat"] = scored._diagnostic_cusum
    frame["is_collapse_event"] = frame.pitch_number_in_outing.between(40, 44)
    frame["collapse_episodes_count_in_game"] = 1
    frame.to_parquet(root / "data/gold/fact_pitch_anomaly_scores.parquet", index=False)
    frame.to_parquet(root / "data/gold/fact_collapse_labels.parquet", index=False)
    return frame


def test_end_to_end_excludes_test_rows_and_does_not_modify_warehouse(warehouse_root):
    tmp_path = warehouse_root
    frame = make_warehouse(tmp_path)
    files = list((tmp_path / "data").rglob("*parquet"))
    original_bytes = {path: path.read_bytes() for path in files}
    result = run_check(shuffles=1, root=tmp_path)
    assert result["experiment"]["test_rows_loaded"] is False
    assert result["variants"]["full"]["production_cusum_parity"].startswith("verified")
    assert all(path.read_bytes() == value for path, value in original_bytes.items())
    assert not (tmp_path / "data/baseball_warehouse.duckdb").exists()
    # Alter every test score and label: neither training nor validation may change.
    test = frame.dataset_split.eq("test")
    frame.loc[test, ["mahalanobis_calibrated", "cusum_stat"]] = 1e12
    frame.to_parquet(tmp_path / "data/gold/fact_pitch_anomaly_scores.parquet", index=False)
    episode_path = tmp_path / "data/gold/fact_collapse_labels.parquet"
    labels = pd.read_parquet(episode_path)
    labels.loc[labels.dataset_split.eq("test"), "is_collapse_event"] = False
    labels.loc[labels.dataset_split.eq("test"), "collapse_episodes_count_in_game"] = 0
    labels.to_parquet(episode_path, index=False)
    after = run_check(shuffles=1, root=tmp_path)
    assert result["variants"] == after["variants"]
    assert result["pitch_count_baseline"] == after["pitch_count_baseline"]
    with pytest.raises(ValueError, match="differs from persisted"):
        frame.loc[frame.dataset_split.eq("validation"), "cusum_stat"] += 1
        frame.to_parquet(tmp_path / "data/gold/fact_pitch_anomaly_scores.parquet", index=False)
        run_check(shuffles=1, root=tmp_path)


def test_missing_warehouse_fails_without_creating_data(warehouse_root):
    tmp_path = warehouse_root
    make_warehouse(tmp_path)
    (tmp_path / "data/gold/fact_pitch_anomaly_scores.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="Missing local warehouse"):
        run_check(shuffles=1, root=tmp_path)
    assert not (tmp_path / "data/baseball_warehouse.duckdb").exists()


def test_episode_reconstruction_preserves_multiple_runs_and_checks_counts():
    labels = rows()
    labels["is_collapse_event"] = (labels.pitch_number_in_outing.between(4, 10) |
                                    labels.pitch_number_in_outing.between(40, 44))
    labels["collapse_episodes_count_in_game"] = 2
    episodes = episodes_from_labels(labels)
    for _, outing in episodes.groupby(["game_pk", "pitcher"]):
        assert outing.onset_pitch.tolist() == [4, 40]
        assert outing.end_pitch.tolist() == [10, 44]
        assert outing.episode_id.tolist() == [1, 2]
    labels["collapse_episodes_count_in_game"] = 1
    with pytest.raises(ValueError, match="per-outing episode counts"):
        episodes_from_labels(labels)


def test_narrow_evaluator_input_matches_full_production_input():
    frame = rows()
    frame["irrelevant_feature"] = np.arange(len(frame))
    episodes = pd.DataFrame({"game_pk": [2], "pitcher": [7], "onset_pitch": [40],
                             "episode_id": [1], "dataset_split": ["validation"]})
    cfg = AblationConfig()
    expected_threshold, expected_metrics = select_operating_threshold(
        frame, episodes, "mahalanobis_calibrated", cfg.false_warnings_per_outing,
        horizon_pitches=cfg.horizon_pitches, split="validation",
    )
    threshold, result = select_and_evaluate(frame, episodes, "mahalanobis_calibrated", cfg)
    assert threshold == expected_threshold
    assert json_safe(result["validation_metrics"]) == json_safe(expected_metrics)
