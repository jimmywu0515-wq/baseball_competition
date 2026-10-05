"""Development-only label audit and matched context/mechanics comparison.

This module does not change production labels, configuration, or thresholds.
All feature calculations use current/past pitches; outcome columns enter only
the audit and supervised training target. The paired models share training rows
and scoring availability. Predicted values are ranking scores, not calibrated
probabilities (the classifiers use balanced class weights).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.baseline_comparator import BaselineComparator
from src.evaluation.protocol import eligible_pitch_mask, evaluation_opportunity_mask
from src.label_builder.collapse_labels import WOBA_WEIGHTS

logger = logging.getLogger(__name__)
KEYS = ["game_pk", "pitcher"]
CONTEXT_FEATURES = ["pitch_number_in_outing", "tto", "inning"]
MECHANICS_FEATURES = [
    "log_distance", "recent_log_distance", "distance_trend",
    "velocity_z", "release_z_magnitude",
]


def development_roles(frame: pd.DataFrame, selection_end: str = "2024-06-30") -> pd.DataFrame:
    """Assign a chronological threshold-selection/assessment split inside 2024."""
    result = frame.copy()
    if not result.dataset_split.isin(["train", "validation"]).all():
        raise ValueError("Development study accepts only train/validation rows; no test rows")
    dates = pd.to_datetime(result.game_date)
    if (dates.dt.year > 2024).any() or dates.isna().any():
        raise ValueError("Development study must exclude post-2024 and invalid dates")
    result["original_dataset_split"] = result.dataset_split
    validation = result.dataset_split.eq("validation")
    result.loc[validation & dates.le(selection_end), "dataset_split"] = "selection"
    result.loc[validation & dates.gt(selection_end), "dataset_split"] = "assessment"
    return result


def mechanics_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Five fixed causal features, with trends grouped by outing and pitch type.

    The trailing windows contain the last five appearances of that pitch type,
    including the current pitch. Unavailable scores remain missing in those
    windows. A trend is zero until a previous five-position mean exists.
    Historical z scores are normalized against prior outings, as persisted by
    ShrinkageCalibrator; they are not calibrated against future observations.
    """
    if not frame.index.is_unique:
        raise ValueError("Feature rows require unique indices")
    ordered = frame.sort_values(KEYS + ["pitch_number_in_outing"]).copy()
    active = ordered.score_available.fillna(False).astype(bool) & ~ordered.is_calibration_phase.fillna(True).astype(bool)
    distance = pd.to_numeric(ordered.mahalanobis_calibrated, errors="coerce")
    distance = distance.where(active & np.isfinite(distance) & distance.ge(0))
    features = pd.DataFrame(index=ordered.index)
    features["log_distance"] = np.log1p(distance)
    grouping = [ordered[c] for c in KEYS + ["pitch_type"]]
    recent = features.log_distance.groupby(grouping).transform(lambda s: s.rolling(5, min_periods=1).mean())
    previous = recent.groupby(grouping).shift(5)
    features["recent_log_distance"] = recent
    features["distance_trend"] = (recent - previous).fillna(0.0)
    features["velocity_z"] = pd.to_numeric(ordered.z_release_speed, errors="coerce")
    release = ordered[["z_release_pos_x", "z_release_pos_z", "z_release_extension"]].apply(pd.to_numeric, errors="coerce")
    features["release_z_magnitude"] = np.sqrt(release.pow(2).sum(axis=1, min_count=3) / 3)
    features.loc[~active] = np.nan
    return features.replace([np.inf, -np.inf], np.nan).reindex(frame.index)


def fit_matched_models(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit fixed logistic models to identical train rows; freeze all preprocessing."""
    context = BaselineComparator._context_features(frame)
    mechanics = mechanics_features(frame)
    available = mechanics.notna().all(axis=1) & context.notna().all(axis=1)
    train = evaluation_opportunity_mask(frame, "train") & available
    if train.sum() < 30 or frame.loc[train, "y_true_onset_in_horizon"].nunique() < 2:
        raise ValueError("At least 30 common training pitches and both outcome classes are required")
    # Limit exceptional baseline/outlier scores using TRAIN quantiles only.
    lower = mechanics.loc[train].quantile(.005)
    upper = mechanics.loc[train].quantile(.995)
    mechanics = mechanics.clip(lower=lower, upper=upper, axis=1)
    combined = pd.concat([context, mechanics], axis=1)
    result = frame.copy()
    result["study_available"] = available
    metadata: dict[str, Any] = {
        "training_pitches": int(train.sum()),
        "training_outings": len(frame.loc[train, KEYS].drop_duplicates()),
        "training_positive_rate": float(frame.loc[train, "y_true_onset_in_horizon"].mean()),
        "mechanics_clip_lower": lower.to_dict(), "mechanics_clip_upper": upper.to_dict(),
        "models": {},
    }
    for key, x in [("context", context), ("proposed", combined)]:
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=1.0, class_weight="balanced", random_state=42, max_iter=1000,
        ))
        model.fit(x.loc[train], frame.loc[train, "y_true_onset_in_horizon"].astype(int))
        classifier = model.named_steps["logisticregression"]
        if classifier.n_iter_[0] >= 1000:
            raise RuntimeError(f"{key} did not converge")
        result[f"study_score_{key}"] = np.nan
        result.loc[available, f"study_score_{key}"] = model.predict_proba(x.loc[available])[:, 1]
        scaler = model.named_steps["standardscaler"]
        metadata["models"][key] = {
            "features": list(x.columns), "coefficients": classifier.coef_[0].tolist(),
            "intercept": float(classifier.intercept_[0]),
            "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "iterations": int(classifier.n_iter_[0]),
        }
    result["study_score_pitch_count"] = result.pitch_number_in_outing.where(available).astype(float)
    return result, metadata


def select_from_curve(curve: pd.DataFrame, budget: float) -> pd.Series:
    """Apply production recall/precision/false-warning tie-breaks to a saved curve."""
    allowed = curve.loc[curve.false_warnings_per_outing.le(budget + 1e-12)].copy()
    if allowed.empty:
        raise ValueError("No feasible threshold (a no-alert candidate must be included)")
    allowed["_recall"] = allowed.episode_recall.fillna(-1)
    allowed["_precision"] = allowed.warning_precision.fillna(-1)
    return allowed.sort_values(
        ["_recall", "_precision", "false_warnings_per_outing"],
        ascending=[False, False, True], kind="stable",
    ).iloc[0]


def audit_labels(
    frame: pd.DataFrame, episodes: pd.DataFrame, calibration_pitches: int,
    horizon: int, window_pas: int = 3, seed: int = 20261002,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit the existing endpoint without changing it or excluding hard episodes.

    Opportunity counts indicate only whether at least one valid warning pitch
    exists; they are not an achievable one-to-one-matching recall ceiling.
    The PA context file supports baseball review, not automatic fatigue diagnosis.
    """
    ordered = frame.sort_values(KEYS + ["pitch_number_in_outing"])
    pa_keys = KEYS + ["pa_number_in_outing"]
    pa = ordered.groupby(pa_keys, sort=False).tail(1).copy()
    pa["first_pitch_in_pa"] = ordered.groupby(pa_keys).pitch_number_in_outing.min().reindex(
        pd.MultiIndex.from_frame(pa[pa_keys])).to_numpy()
    pa["pa_ordinal"] = pa.groupby(KEYS).cumcount() + 1
    xwoba = pd.to_numeric(pa.estimated_woba_using_speedangle, errors="coerce")
    event_weight = pa.events.fillna("").str.lower().map(WOBA_WEIGHTS).fillna(0.0)
    pa["pa_blended_xwoba"] = xwoba.fillna(event_weight)
    # The persisted column can also contain values for walks/HBP. Do not
    # misdescribe every finite provider value as a batted-ball contact estimate.
    pa["xwoba_source"] = np.where(xwoba.notna(), "persisted_estimated_woba_value", "event_weight_fallback")
    pa["onset_window_pas"] = pa.pa_ordinal.clip(upper=window_pas)

    onset_pa = pa[pa_keys + ["pitch_number_in_outing", "events", "window_blended_xwoba",
                             "collapse_reason", "pa_blended_xwoba", "xwoba_source", "onset_window_pas"]].rename(columns={
        "pa_number_in_outing": "onset_pa", "pitch_number_in_outing": "label_confirmed_pitch",
        "events": "onset_pa_event", "pa_blended_xwoba": "onset_pa_blended_xwoba",
    })
    audit = episodes.merge(onset_pa, on=KEYS + ["onset_pa"], how="left", validate="one_to_one")
    if audit.label_confirmed_pitch.isna().any():
        raise ValueError("An episode onset has no matching PA outcome")
    audit["duration_pitches"] = audit.end_pitch - audit.onset_pitch + 1
    audit["duration_pas"] = audit.end_pa - audit.onset_pa + 1
    audit["recognition_delay_pitches"] = audit.label_confirmed_pitch - audit.onset_pitch
    audit["pre_onset_pas_in_window"] = audit.onset_window_pas - 1
    audit["onset_during_calibration"] = audit.onset_pitch.le(calibration_pitches)
    protocol = evaluation_opportunity_mask(frame)
    mechanical = eligible_pitch_mask(frame)
    opportunities = {}
    for key, outing in frame.groupby(KEYS, sort=False):
        opportunities[key] = (
            np.sort(outing.loc[protocol.loc[outing.index], "pitch_number_in_outing"].to_numpy()),
            np.sort(outing.loc[mechanical.loc[outing.index], "pitch_number_in_outing"].to_numpy()),
        )
    protocol_counts, mechanics_counts = [], []
    for row in audit.itertuples():
        counts = []
        for pitches in opportunities[(row.game_pk, row.pitcher)]:
            counts.append(int(np.searchsorted(pitches, row.onset_pitch, side="left") -
                              np.searchsorted(pitches, row.onset_pitch - horizon, side="left")))
        protocol_counts.append(counts[0])
        mechanics_counts.append(counts[1])
    audit["protocol_warning_opportunities"] = protocol_counts
    audit["mechanics_warning_opportunities"] = mechanics_counts
    report: dict[str, Any] = {
        "interpretation": "Existing performance-stretch labels; frequency alone neither validates nor invalidates a collapse interpretation.",
        "onset_semantics": "Onset is backdated to the first pitch of the PA completing a bad rolling window; confirmation is at that PA's last pitch. Earlier PAs in the window precede onset.",
        "opportunity_semantics": "Opportunity counts are structural availability diagnostics, not a one-to-one-matching recall ceiling. Original denominators are retained.",
        "splits": {},
    }
    for split, pitches in frame.groupby("dataset_split", sort=False):
        eps = audit.loc[audit.dataset_split.eq(split)]
        n_outings = len(pitches[KEYS].drop_duplicates())
        bad_outings = len(eps[KEYS].drop_duplicates())
        report["splits"][split] = {
            "qualified_outings": n_outings, "outings_with_episode": bad_outings,
            "clean_outings": n_outings - bad_outings,
            "fraction_outings_with_episode": bad_outings / n_outings,
            "episodes": len(eps), "episodes_per_outing": len(eps) / n_outings,
            "onsets_during_calibration": int(eps.onset_during_calibration.sum()),
            "no_protocol_warning_opportunity": int(eps.protocol_warning_opportunities.eq(0).sum()),
            "no_mechanics_warning_opportunity": int(eps.mechanics_warning_opportunities.eq(0).sum()),
            "onsets_using_incomplete_pa_window": int(eps.onset_window_pas.lt(window_pas).sum()),
            "median_duration_pas": float(eps.duration_pas.median()),
            "median_onset_pitch": float(eps.onset_pitch.median()),
            "median_recognition_delay_pitches": float(eps.recognition_delay_pitches.median()),
            "median_onset_window_xwoba": float(eps.window_blended_xwoba.median()),
            "reason_counts": eps.collapse_reason.fillna("unknown").value_counts().to_dict(),
        }
    # A deterministic stratified review sample, selected independently of model predictions.
    samples = []
    for _, group in audit.groupby(["dataset_split", "collapse_reason", "onset_during_calibration"], dropna=False, sort=True):
        samples.append(group.sample(n=min(2, len(group)), random_state=seed))
    cases = pd.concat(samples, ignore_index=True) if samples else audit.iloc[:0].copy()
    contexts = []
    for case_id, row in enumerate(cases.itertuples(), 1):
        case_pa = pa.loc[pa.game_pk.eq(row.game_pk) & pa.pitcher.eq(row.pitcher) &
                         pa.pa_number_in_outing.between(max(1, row.onset_pa - window_pas), row.end_pa + 2)].copy()
        columns = pa_keys + ["game_date", "pitcher_name", "batter", "events", "first_pitch_in_pa",
                             "pitch_number_in_outing", "pa_blended_xwoba", "xwoba_source", "window_blended_xwoba",
                             "collapse_reason", "is_collapse_event", "launch_speed", "launch_angle"]
        case_pa = case_pa.loc[:, columns]
        case_pa["case_id"] = case_id
        case_pa["episode_id"] = row.episode_id
        contexts.append(case_pa)
    cases["case_id"] = np.arange(1, len(cases) + 1)
    cases["review_assessment"] = "pending baseball review"
    return report, audit, cases, pd.concat(contexts, ignore_index=True) if contexts else pd.DataFrame()
