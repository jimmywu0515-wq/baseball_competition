"""Regression examples for the shared scientific evaluation contract."""
import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path
import shutil
import uuid

from scripts.run_full_pipeline import _promote_staged_outputs, _write_protocol_manifest
from src.evaluation.ablation_runner import AblationRunner
from src.evaluation.bootstrap import paired_bootstrap_confidence_intervals
from src.evaluation.protocol import (
    build_evaluation_universe, eligible_pitch_mask, evaluate_warning_predictions,
    evaluation_opportunity_mask, warning_events, select_operating_threshold,
)
from src.presentation import load_protocol_manifest
from src.configuration import load_project_config


def _rows(games=(1, 2), pitches=(21, 22, 23, 24)):
    frame = pd.DataFrame([
        {"game_pk": game, "pitcher": 7, "pitch_number_in_outing": pitch,
         "pa_number_in_outing": pitch // 4, "dataset_split": "test",
         "is_calibration_phase": False, "is_censored_followup": False,
         "score_available": game == 1, "game_date": "2025-06-01"}
        for game in games for pitch in pitches
    ])
    frame.index = pd.Index([91, 4, 77, 12, 101, 8, 53, 30][:len(frame)])
    return frame


def test_index_alignment_and_boolean_availability():
    frame = _rows()
    availability = frame["score_available"].sample(frac=1, random_state=7)
    mask = eligible_pitch_mask(frame, "test", availability=availability)
    assert mask.index.equals(frame.index)
    assert mask.sum() == 4
    assert evaluation_opportunity_mask(frame, "test").sum() == 8
    assert len(build_evaluation_universe(frame, "test")) == 2
    with pytest.raises(ValueError, match="Boolean"):
        eligible_pitch_mask(frame, availability=pd.Series(np.arange(len(frame)), index=frame.index))
    with pytest.raises(ValueError, match="missing pitch-row"):
        eligible_pitch_mask(frame, availability=availability.iloc[:-1])
    duplicate = frame.copy()
    duplicate.iloc[1, duplicate.columns.get_loc("pitch_number_in_outing")] = 21
    with pytest.raises(ValueError, match="Duplicate pitch identity"):
        build_evaluation_universe(duplicate)


def test_scoreless_outing_episode_and_unmatched_warning():
    frame = _rows()
    flagged = pd.Series(False, index=frame.index)
    flagged.loc[frame[frame["game_pk"].eq(1) & frame["pitch_number_in_outing"].eq(21)].index] = True
    episodes = pd.DataFrame({"game_pk": [2], "pitcher": [7], "episode_id": [1],
                             "onset_pitch": [23], "dataset_split": ["test"]})
    metrics, warnings, matches = evaluate_warning_predictions(frame, episodes, flagged, split="test")
    assert metrics["total_qualified_outings"] == 2
    assert metrics["total_collapse_episodes"] == 1
    assert metrics["episode_recall"] == 0
    assert metrics["false_warnings_per_outing"] == 0.5
    assert metrics["outings_without_available_score"] == 1
    assert len(warnings) == 1 and matches.empty
    frame["pred"] = flagged
    frame["y_true_onset_in_horizon"] = frame["game_pk"].eq(2) & frame["pitch_number_in_outing"].lt(23)
    result = paired_bootstrap_confidence_intervals(
        frame, episodes, {"proposed": "pred"}, n_bootstrap=30, seed=4,
        model_availability={"proposed": "score_available"},
    )
    estimates = result.set_index("metric")["estimate"]
    assert estimates["false_warnings_per_outing"] == 0.5
    assert estimates["episode_recall"] == 0


def test_raw_warnings_ignore_followup_label_and_censor_only_at_evaluation():
    frame = _rows(games=(1,))
    frame["score_available"] = True
    frame.loc[frame["pitch_number_in_outing"].eq(23), "is_censored_followup"] = True
    flags = pd.Series([True, False, True, True], index=frame.index)
    raw = warning_events(frame, flags)
    assert raw["warning_pitch"].tolist() == [21, 23]
    metrics, warnings, _ = evaluate_warning_predictions(frame, pd.DataFrame(), flags)
    assert warnings["warning_pitch"].tolist() == [21, 23]
    assert metrics["master_warning_count"] == 2
    assert metrics["total_warnings"] == 1
    assert metrics["censored_warnings"] == 1
    frame["is_censored_followup"] = False
    assert warning_events(frame, flags)["warning_pitch"].tolist() == [21, 23]


def test_strict_future_horizon_and_one_to_one_matching():
    frame = _rows(games=(1,), pitches=(21, 22, 23, 24))
    frame["score_available"] = True
    frame.loc[frame["pitch_number_in_outing"].isin([22, 24]), "score_available"] = False
    flags = pd.Series(True, index=frame.index)
    episodes = pd.DataFrame({"game_pk": [1, 1, 1], "pitcher": [7, 7, 7],
                             "episode_id": [1, 2, 3], "onset_pitch": [21, 23, 25],
                             "dataset_split": ["test"] * 3})
    metrics, _, matches = evaluate_warning_predictions(frame, episodes, flags, horizon_pitches=1)
    assert matches.empty  # starts at 21 and 23; later onsets exceed a one-pitch horizon
    assert metrics["episode_recall"] == 0
    metrics, _, matches = evaluate_warning_predictions(frame, episodes, flags, horizon_pitches=2)
    assert matches["onset_pitch"].tolist() == [23, 25]
    assert metrics["episode_recall"] == pytest.approx(2 / 3)


def test_empty_schema_and_no_alert_threshold():
    frame = pd.DataFrame(columns=["game_pk", "pitcher", "pitch_number_in_outing", "dataset_split"])
    metrics, warnings, matches = evaluate_warning_predictions(frame, pd.DataFrame(), pd.Series(dtype=bool), split="test")
    assert metrics["total_qualified_outings"] == 0
    assert np.isnan(metrics["episode_recall"])
    assert np.isnan(metrics["warning_precision"])
    assert np.isnan(metrics["false_warnings_per_outing"])
    assert {"game_pk", "pitcher", "warning_pitch", "is_evaluable_warning"}.issubset(warnings.columns)
    assert {"episode_id", "onset_pitch", "lead_time_pitches"}.issubset(matches.columns)
    scored = _rows(games=(1,))
    scored["score"] = np.nan
    threshold, selected = select_operating_threshold(scored, pd.DataFrame(), "score", 0.5, split="test")
    assert np.isinf(threshold)
    assert selected["total_warnings"] == 0


def test_ablation_mismatch_and_failed_promotion_preserve_published():
    frame = _rows(games=(1,))
    frame["score_proposed_cusum"] = [0, 1, 2, 3]
    frame["is_proposed_operating_alert"] = [False, False, True, True]
    runner = AblationRunner(frame)
    metrics, _, _ = evaluate_warning_predictions(frame, pd.DataFrame(), frame["is_proposed_operating_alert"],
                                                  availability=pd.Series(True, index=frame.index))
    metrics["operating_threshold"] = 2.0
    wrong = dict(metrics, total_qualified_outings=3)
    with pytest.raises(ValueError, match="parity failed"):
        runner.run_feature_ablations(2.0, wrong)
    root = Path(__file__).resolve().parents[1] / "outputs"
    temp_root = root / f".promotion_test_{uuid.uuid4().hex}"
    assert temp_root.resolve().is_relative_to(root.resolve())
    published = temp_root / "published"
    published.mkdir(parents=True)
    (published / "marker.txt").write_text("previous", encoding="utf-8")
    stage = temp_root / "stage"
    stage.mkdir()
    with pytest.raises(FileNotFoundError):
        _promote_staged_outputs(stage, published, "bad")
    assert (published / "marker.txt").read_text(encoding="utf-8") == "previous"
    (stage / "protocol_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Incompatible"):
        load_protocol_manifest(stage)
    shutil.rmtree(temp_root)


def test_no_alert_manifest_round_trip():
    root = Path(__file__).resolve().parents[1] / "outputs"
    temp_root = root / f".promotion_test_{uuid.uuid4().hex}"
    assert temp_root.resolve().is_relative_to(root.resolve())
    temp_root.mkdir()
    try:
        manifest = _write_protocol_manifest(
            temp_root, "simulation_benchmark", {"proposed": np.inf},
            {"proposed": {"episode_recall": np.nan, "warning_precision": np.nan,
                          "false_warnings_per_outing": 0.0}},
            {"threshold_selection": {"allowed_maximum_false_warnings_per_outing": 0.5},
             "baseline": {"historical_window_starts": 12}}, "test-run",
        )
        raw = (temp_root / "protocol_manifest.json").read_text(encoding="utf-8")
        assert "Infinity" not in raw and "NaN" not in raw
        selected = load_protocol_manifest(temp_root)["resolved_runtime"]["selected_models"]["proposed"]
        assert selected["threshold_status"] == "no_alert"
        assert selected["validation_selected_operating_threshold"] is None
    finally:
        shutil.rmtree(temp_root)


def test_obsolete_and_invalid_config_are_rejected():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config" / "config.yaml").read_text(encoding="utf-8"))
    temp_root = root / "outputs" / f".promotion_test_{uuid.uuid4().hex}"
    assert temp_root.resolve().is_relative_to((root / "outputs").resolve())
    temp_root.mkdir()
    path = temp_root / "config.yaml"
    try:
        config["evaluation"]["matching_horizon_pitches"] = 10
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with pytest.raises(ValueError, match="Obsolete configuration key"):
            load_project_config(path)
        del config["evaluation"]["matching_horizon_pitches"]
        config["baseline"]["intra_game_calibration_pitches"] = config["qualify"]["min_pitches_per_game"]
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with pytest.raises(ValueError, match="leave at least one pitch"):
            load_project_config(path)
    finally:
        shutil.rmtree(temp_root)
