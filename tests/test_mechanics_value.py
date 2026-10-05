"""Leakage, timing, and matched-population safeguards for the development study."""
import numpy as np
import pandas as pd
import pytest
import json
from pathlib import Path
import shutil
import uuid
import yaml

from scripts.diagnostics.check_cusum_pitch_count_correlation import episodes_from_labels, json_safe
from src.evaluation.mechanics_value import (
    audit_labels, development_roles, fit_matched_models, mechanics_features, select_from_curve,
)
from src.configuration import load_project_config


def rows():
    rng = np.random.default_rng(9)
    rows = []
    for game, date, split in [(1, "2023-05-01", "train"), (2, "2023-07-01", "train"),
                               (3, "2024-05-01", "validation"), (4, "2024-08-01", "validation")]:
        for pitch in range(1, 61):
            rows.append({
                "game_pk": game, "pitcher": 7, "pitcher_name": "Fixture", "batter": 10,
                "game_date": pd.Timestamp(date), "dataset_split": split, "actual_data_source": "mlb_statcast",
                "pitch_number_in_outing": pitch, "pa_number_in_outing": (pitch - 1) // 4 + 1,
                "pitch_type": "FF" if pitch % 2 else "SL", "inning": (pitch - 1) // 12 + 1,
                "is_calibration_phase": pitch <= 20, "score_available": pitch > 20,
                "is_censored_followup": pitch > 55, "mahalanobis_calibrated": 1 + rng.random() * 3,
                "z_release_speed": rng.normal(), "z_release_pos_x": rng.normal(),
                "z_release_pos_z": rng.normal(), "z_release_extension": rng.normal(),
                "y_true_onset_in_horizon": (pitch + game) % 5 < 2,
                "events": "walk" if pitch % 4 == 0 else None,
                "estimated_woba_using_speedangle": np.nan, "launch_speed": np.nan, "launch_angle": np.nan,
                "window_blended_xwoba": .50, "collapse_reason": "xwOBA",
                "is_collapse_event": 5 <= pitch <= 8 or 37 <= pitch <= 40,
                "collapse_episodes_count_in_game": 2,
            })
    frame = pd.DataFrame(rows)
    frame.index = np.arange(len(frame)) * 7 + 2
    return frame


def test_development_split_rejects_test_and_respects_boundary():
    frame = rows()
    result = development_roles(frame)
    assert result.loc[result.game_pk.eq(3), "dataset_split"].eq("selection").all()
    assert result.loc[result.game_pk.eq(4), "dataset_split"].eq("assessment").all()
    frame.loc[frame.game_pk.eq(3), "game_date"] = pd.Timestamp("2024-06-30")
    assert development_roles(frame).loc[frame.game_pk.eq(3), "dataset_split"].eq("selection").all()
    frame.loc[frame.game_pk.eq(4), "dataset_split"] = "test"
    with pytest.raises(ValueError, match="no test rows"):
        development_roles(frame)


def test_features_are_causal_and_isolated_from_labels_and_other_outings():
    frame = rows()
    expected = mechanics_features(frame)
    assert expected.loc[frame.pitch_number_in_outing.le(20)].isna().all().all()
    changed = frame.copy()
    late = changed.pitch_number_in_outing.gt(40)
    changed.loc[late, ["mahalanobis_calibrated", "z_release_speed"]] = 1e9
    changed["y_true_onset_in_horizon"] = ~changed.y_true_onset_in_horizon
    changed["is_collapse_event"] = ~changed.is_collapse_event
    actual = mechanics_features(changed)
    pd.testing.assert_frame_equal(expected.loc[~late], actual.loc[~late])
    alone = mechanics_features(frame.loc[frame.game_pk.eq(4)])
    pd.testing.assert_frame_equal(expected.loc[alone.index], alone)
    shuffled = mechanics_features(frame.sample(frac=1, random_state=6)).reindex(frame.index)
    pd.testing.assert_frame_equal(expected, shuffled)


def test_fitting_uses_train_only_and_exactly_matched_availability():
    frame = development_roles(rows())
    frame.loc[frame.pitch_number_in_outing.eq(25), "z_release_speed"] = np.nan
    expected, metadata = fit_matched_models(frame)
    assert expected.study_score_context.notna().equals(expected.study_score_proposed.notna())
    assert expected.study_score_pitch_count.notna().equals(expected.study_score_proposed.notna())
    assert expected.loc[frame.pitch_number_in_outing.eq(25), "study_score_context"].isna().all()
    changed = frame.copy()
    future = changed.dataset_split.eq("assessment")
    changed.loc[future, "mahalanobis_calibrated"] = 1e9
    changed.loc[future, "z_release_speed"] = -1e9
    changed.loc[future, "y_true_onset_in_horizon"] = ~changed.loc[future, "y_true_onset_in_horizon"]
    actual, changed_metadata = fit_matched_models(changed)
    assert metadata == changed_metadata
    pd.testing.assert_frame_equal(
        expected.loc[~future, ["study_score_context", "study_score_proposed"]],
        actual.loc[~future, ["study_score_context", "study_score_proposed"]],
    )


def test_audit_retains_unwarnable_episodes_and_separates_onset_from_confirmation():
    frame = rows()
    frame.loc[frame.game_pk.eq(2), "score_available"] = False
    episodes = episodes_from_labels(frame)
    report, audit, cases, context = audit_labels(frame, episodes, calibration_pitches=20, horizon=15)
    assert len(audit) == 8
    early = audit.loc[audit.onset_pitch.eq(5)]
    assert early.protocol_warning_opportunities.eq(0).all()
    assert early.onset_window_pas.eq(2).all()
    assert early.label_confirmed_pitch.eq(8).all()
    assert early.recognition_delay_pitches.eq(3).all()
    late = audit.loc[audit.onset_pitch.eq(37)]
    assert late.protocol_warning_opportunities.eq(15).all()
    assert late.loc[late.game_pk.eq(2), "mechanics_warning_opportunities"].eq(0).all()
    assert report["splits"]["train"]["episodes"] == 4
    assert report["splits"]["train"]["no_protocol_warning_opportunity"] == 2
    assert report["splits"]["train"]["no_mechanics_warning_opportunity"] == 3
    assert set(context.case_id) == set(cases.case_id)
    _, _, repeated, _ = audit_labels(frame, episodes, 20, 15)
    pd.testing.assert_frame_equal(cases, repeated)
    json_safe(report)


def test_curve_selection_uses_budget_and_production_tie_breaks():
    curve = pd.DataFrame({
        "threshold": [np.inf, 3, 2, 1, .5],
        "episode_recall": [0, .2, .2, .2, .8],
        "warning_precision": [np.nan, .5, .7, .7, .8],
        "false_warnings_per_outing": [0, .2, .3, .25, .6],
    })
    assert select_from_curve(curve, .5).threshold == 1
    assert np.isinf(select_from_curve(curve, 0).threshold)


def test_real_entrypoint_excludes_2025_and_preserves_source_tables(monkeypatch):
    from scripts.diagnostics import run_mechanics_value_study as study
    outputs = Path(__file__).resolve().parents[1] / "outputs"
    root = outputs / f".smoke_mechanics_value_{uuid.uuid4().hex}"
    root.mkdir()
    try:
        config = load_project_config()
        (root / "config").mkdir()
        (root / "config/config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        (root / "outputs/real_data").mkdir(parents=True)
        manifest = {"run_id": "fixture", "protocol_sha256": "fixture_hash", "actual_data_source": "mlb_statcast",
                    "prior_2025_exposure_disclosure": "Previously inspected", "resolved_runtime": {"resolved_config": config}}
        (root / "outputs/real_data/protocol_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        frame = rows()
        future = frame.loc[frame.game_pk.eq(4)].copy()
        future["game_pk"] = 5
        future["dataset_split"] = "test"
        future["game_date"] = pd.Timestamp("2025-06-01")
        future["mahalanobis_calibrated"] = 1e12
        frame = pd.concat([frame, future], ignore_index=True).assign(run_id="fixture", protocol_sha256="fixture_hash")
        (root / "data/gold").mkdir(parents=True)
        sources = []
        for name in ["fact_pitch_anomaly_scores", "fact_collapse_labels"]:
            path = root / "data/gold" / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            sources.append((path, path.read_bytes()))
        original_curve = study.threshold_tradeoff_curve

        def checked_curve(frame, *args, **kwargs):
            assert frame.dataset_split.eq("selection").all()
            assert pd.to_datetime(frame.game_date).max() <= pd.Timestamp("2024-06-30")
            return original_curve(frame, *args, **kwargs)

        monkeypatch.setattr(study, "threshold_tradeoff_curve", checked_curve)
        report = study.run_study(root / "results", bootstrap_samples=20, root=root)
        assert report["plan"]["test_rows_loaded"] is False
        assert report["label_audit"]["splits"]["validation"]["qualified_outings"] == 2
        assert len(report["comparison"]) == 9
        for row in report["comparison"]:
            assert row["total_qualified_outings"] == 1
            assert row["total_collapse_episodes"] == 2
            assert row["selection_false_warnings_per_outing"] <= row["selection_budget"]
        assert all(path.read_bytes() == before for path, before in sources)
        assert not (root / "data/baseball_warehouse.duckdb").exists()
        json.loads((root / "results/mechanics_value_report.json").read_text(),
                   parse_constant=lambda value: pytest.fail(f"Nonstandard JSON: {value}"))
    finally:
        assert root.resolve().parent == outputs.resolve()
        shutil.rmtree(root, ignore_errors=True)
