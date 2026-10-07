"""Guard warning units, direction ambiguity, training isolation, and provenance."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.diagnostics.directional_scale_diagnostics import (
    ROOT, directional_diagnostic, hypothesis_signs, matched_background, training_dispersion,
)
from src.baseline_builder.dispersion_shrinkage import fit_pitcher_dispersion_scales, shrinkage_weight
from src.changepoint_detector.cusum_detector import CUSUMDetector, resolve_reference_std
from src.evaluation.ablation_runner import AblationRunner


def plan():
    return json.loads((ROOT / "outputs/real_data/diagnostics/directional_scale_plan.json").read_text())


def fixture(split="validation"):
    rows = []
    year = 2023 if split == "train" else 2024
    for pitcher in [1,2]:
        for game in range(1,7):
            for pitch in range(1,51):
                rows.append({
                    "game_pk": pitcher*100+game, "pitcher": pitcher, "pitcher_name": f"Fixture {pitcher}",
                    "game_date": pd.Timestamp(f"{year}-06-{game:02d}"), "dataset_split": split,
                    "actual_data_source": "mlb_statcast", "pitch_number_in_outing": pitch,
                    "pa_number_in_outing": (pitch-1)//4+1, "pitch_type": "FF",
                    "is_calibration_phase": pitch<=20, "score_available": pitch>20,
                    "is_censored_followup": pitch>45, "is_collapse_event": 30<=pitch<=35,
                    "y_true_onset_in_horizon": 22<=pitch<=25,
                    "mahalanobis_calibrated": 3 + (.2 if pitcher==1 else 2)*np.sin(pitch),
                    "cusum_stat": 300.0 if pitch>=25 else 0.0,
                    "dominant_drift_feature": "release_speed",
                    "calib_delta_release_speed": 1.0 if pitcher==1 else -1.0,
                    "calib_delta_release_pos_z": .1, "calib_delta_release_extension": .1,
                    "calib_delta_release_spin_rate": 10.0,
                })
    frame=pd.DataFrame(rows)
    frame.index=np.arange(len(frame))*7+3
    return frame


def warning_fixture():
    frame=fixture().loc[lambda d:d.game_pk.isin([101,201])].copy()
    frame["is_collapse_event"]=False
    frame["y_true_onset_in_horizon"]=False
    alerts=frame.loc[frame.pitch_number_in_outing.eq(25),["game_pk","pitcher","dataset_split"]].copy()
    alerts["alert_pitch_number"]=25
    alerts["operating_threshold"]=200.0
    episodes=pd.DataFrame(columns=["game_pk","pitcher","dataset_split","onset_pitch","onset_pa"])
    return frame,episodes,alerts


def test_raw_signed_directions_preserve_unknown_zero_and_outcome_isolation():
    frame=fixture().iloc[:5].copy()
    frame["dominant_drift_feature"]=["release_speed","release_spin_rate","vaa","release_extension","release_pos_z"]
    frame.loc[frame.index[3],"calib_delta_release_extension"]=0.0
    frame.loc[frame.index[4],"calib_delta_release_pos_z"]=np.nan
    signed=hypothesis_signs(frame)
    assert signed.direction_status.tolist()==["positive","positive","unclassified_dominant_feature","neutral_or_rounded_zero","missing_delta"]
    frame["is_collapse_event"]=~frame.is_collapse_event
    frame["y_true_onset_in_horizon"]=~frame.y_true_onset_in_horizon
    pd.testing.assert_series_equal(signed.hypothesized_positive,hypothesis_signs(frame).hypothesized_positive)


def test_background_matches_pitcher_type_feature_and_workload_band():
    frame=fixture().loc[lambda d:d.game_pk.isin([101,201])]
    signed=hypothesis_signs(frame)
    warnings=signed.loc[signed.pitch_number_in_outing.eq(25)]
    matched=matched_background(warnings,signed)
    assert matched.baseline_positive_share.tolist()==[1.0,0.0]
    assert matched.baseline_classifiable_pitches.eq(10).all()


def test_warning_starts_are_primary_and_persisted_alert_parity_is_enforced():
    frame,episodes,alerts=warning_fixture()
    report,rows=directional_diagnostic(frame,episodes,alerts,200,15,plan())
    val=report["splits"]["validation"]
    assert val["clean_false_warning_starts"]["rows"]==2
    assert val["clean_false_warning_starts"]["positive_sign_share"]==.5
    assert val["clean_warning_run_pitches_within_horizon"]["rows"]==32
    assert len(rows)==2
    assert not report["proceed_to_directional_production"]
    alerts.loc[alerts.index[0],"alert_pitch_number"]=26
    with pytest.raises(ValueError,match="warning starts differ"):
        directional_diagnostic(frame,episodes,alerts,200,15,plan())


def test_no_warnings_and_post2024_rows_cannot_trigger_a_production_gate():
    frame,episodes,alerts=warning_fixture()
    frame["cusum_stat"]=0.0
    report,_=directional_diagnostic(frame,episodes,alerts.iloc[:0],200,15,plan())
    assert not report["proceed_to_directional_production"]
    assert report["splits"]["validation"]["clean_false_warning_starts"]["classifiable_rows"]==0
    frame["dataset_split"]="test"
    with pytest.raises(ValueError,match="test rows"):
        directional_diagnostic(frame,episodes,alerts,200,15,plan())


def test_dispersion_is_train_only_and_strict_sensitivity_excludes_active_and_censored():
    frame=fixture("train")
    primary,table=training_dispersion(frame,plan())
    strict,strict_table=training_dispersion(frame,plan(),strict=True)
    assert table.n_eligible_pitches.eq(156).all()
    assert strict_table.n_eligible_pitches.eq(90).all()
    assert primary["passes_ratio_gate"] and strict["passes_ratio_gate"]
    assert primary["max_to_min_std_ratio"]==pytest.approx(10.0)
    frame.loc[frame.index[-1],"dataset_split"]="validation"
    with pytest.raises(ValueError,match="2023 train"):
        training_dispersion(frame,plan())
    frame["dataset_split"]="train"
    frame.loc[frame.index[-1],"game_date"]=pd.Timestamp("2025-01-01")
    with pytest.raises(ValueError,match="2023 train"):
        training_dispersion(frame,plan())


def test_dispersion_outing_floor_uses_actual_clean_scoring_rows():
    frame=fixture("train")
    frame.loc[frame.pitcher.eq(1)&frame.game_pk.ge(105),"score_available"]=False
    with pytest.raises(ValueError,match="two qualifying pitchers"):
        training_dispersion(frame,plan())


def test_shrinkage_weight_uses_pitch_count_and_has_the_declared_limits():
    assert shrinkage_weight(0,100)==0
    assert shrinkage_weight(100,100)==.5
    assert 0<shrinkage_weight(10,100)<shrinkage_weight(50,100)<.5
    assert shrinkage_weight(1000000,100)>.999
    for invalid in [0,-1,np.nan,np.inf]:
        with pytest.raises(ValueError):
            shrinkage_weight(10,invalid)


def test_frozen_scale_fitter_rejects_validation_and_records_train_only_prior():
    frame=fixture("train")
    table,metadata=fit_pitcher_dispersion_scales(frame)
    assert metadata["prior_pitches"]==156
    assert table.shrinkage_weight.eq(.5).all()
    assert table.loc[table.pitcher.eq(1),"scale_used"].iloc[0]<1
    assert table.loc[table.pitcher.eq(2),"scale_used"].iloc[0]>1
    assert metadata["unseen_or_insufficient_training_fallback_scale"]==1
    changed=frame.copy()
    changed.loc[changed.index[-1],"dataset_split"]="test"
    with pytest.raises(ValueError,match="2023 train"):
        fit_pitcher_dispersion_scales(changed)
    changed=frame.copy()
    changed.loc[changed.index[-1],"game_date"]=pd.Timestamp("2024-01-01")
    with pytest.raises(ValueError,match="2023 train"):
        fit_pitcher_dispersion_scales(changed)


def test_scalar_cusum_matches_frozen_values_and_constant_mapping_in_both_paths():
    frame=fixture().loc[lambda d:d.game_pk.eq(101)&d.pitch_number_in_outing.between(20,24)].copy()
    frame["inning"]=1
    frame["mahalanobis_calibrated"]=[1e9,2,2.5,np.nan,1]
    expected=np.array([np.nan,1.5,4,np.nan,3.5])
    scalar,alerts=CUSUMDetector(threshold_h=2).detect_game_alerts(frame)
    mapped,mapped_alerts=CUSUMDetector(threshold_h=2,reference_std={1:.5}).detect_game_alerts(frame)
    np.testing.assert_array_equal(scalar.cusum_stat.to_numpy(),expected)
    pd.testing.assert_frame_equal(scalar,mapped)
    pd.testing.assert_frame_equal(alerts,mapped_alerts)
    reference=AblationRunner._run_cusum_on_scores(frame,"mahalanobis_calibrated","result",.5,2,1,.5,20)
    np.testing.assert_array_equal(reference.result.to_numpy(),expected)
    same=AblationRunner._run_cusum_on_scores(frame,"mahalanobis_calibrated","result",.5,2,1,pd.Series({1:.5}),20)
    pd.testing.assert_frame_equal(reference,same)


def test_pitcher_mapping_is_shared_and_invalid_or_missing_scales_raise():
    frame=fixture().loc[lambda d:d.game_pk.isin([101,201])&d.pitch_number_in_outing.between(21,26)].copy()
    frame["inning"]=1
    references={1:.5,2:2.0}
    generic=AblationRunner._run_cusum_on_scores(frame,"mahalanobis_calibrated","result",.5,2,1,references,20)
    for pitcher,outing in frame.groupby("pitcher"):
        detected,_=CUSUMDetector(threshold_h=2,reference_std=references).detect_game_alerts(outing)
        np.testing.assert_array_equal(generic.loc[outing.index,"result"],detected.cusum_stat)
    for references in [{1:-1},{1:np.nan},{2:.5},pd.Series([.5,.7],index=[1,1])]:
        with pytest.raises(ValueError):
            resolve_reference_std(references,1)
