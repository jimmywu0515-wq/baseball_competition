"""Shared temporal split, eligibility, warning, and event-matching protocol."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


TRAIN_END = pd.Timestamp("2023-12-31")
VALIDATION_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")


def assign_temporal_split(
    df: pd.DataFrame,
    train_end: Any = TRAIN_END,
    validation_end: Any = VALIDATION_END,
    test_start: Any = TEST_START,
) -> pd.DataFrame:
    """Assign the frozen primary protocol: 2023 train, 2024 validation, 2025 test."""
    result = df.copy()
    dates = pd.to_datetime(result["game_date"], errors="coerce")
    train_end = pd.Timestamp(train_end)
    validation_end = pd.Timestamp(validation_end)
    test_start = pd.Timestamp(test_start)
    result["dataset_split"] = np.select(
        [dates <= train_end, (dates > train_end) & (dates <= validation_end), dates >= test_start],
        ["train", "validation", "test"],
        default="unassigned",
    )
    result.loc[dates.isna(), "dataset_split"] = "unassigned"
    return result


def assign_historical_temporal_split(
    df: pd.DataFrame,
    train_end: Any = TRAIN_END,
    validation_end: Any = "2024-06-30",
    test_start: Any = "2024-07-01",
    test_end: Any = "2024-12-31",
) -> pd.DataFrame:
    """Preserve the earlier 2023 / early-2024 / late-2024 historical experiment."""
    result = df.copy()
    dates = pd.to_datetime(result["game_date"], errors="coerce")
    train_end = pd.Timestamp(train_end)
    validation_end = pd.Timestamp(validation_end)
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)
    result["dataset_split"] = np.select(
        [
            dates <= train_end,
            (dates > train_end) & (dates <= validation_end),
            (dates >= test_start) & (dates <= test_end),
        ],
        ["train", "validation", "test"],
        default="unassigned",
    )
    result.loc[dates.isna(), "dataset_split"] = "unassigned"
    return result


def evaluation_opportunity_mask(df: pd.DataFrame, split: Optional[str] = None) -> pd.Series:
    """Index-aligned evaluation opportunities, excluding calibration and censored follow-up.

    Requires a unique row index. Optional phase flags default to all False; when
    a split is requested, ``dataset_split`` is required. No model score is read.
    An empty frame returns an empty Boolean Series with the same index.
    """
    _validate_pitch_rows(df, split)
    mask = pd.Series(True, index=df.index, dtype=bool)
    if "is_calibration_phase" in df:
        mask &= ~df["is_calibration_phase"].fillna(True).astype(bool)
    if "is_censored_followup" in df:
        mask &= ~df["is_censored_followup"].fillna(True).astype(bool)
    if split is not None and "dataset_split" in df:
        mask &= df["dataset_split"].eq(split)
    return mask


def eligible_pitch_mask(
    df: pd.DataFrame,
    split: Optional[str] = None,
    availability: Optional[pd.Series] = None,
) -> pd.Series:
    """Intersect opportunities with index-aligned Boolean availability.

    A missing availability argument uses Boolean ``score_available`` when present,
    otherwise finite ``mahalanobis_calibrated`` scores, otherwise all opportunities.
    Numeric scores must be converted to a finite mask by the caller. Missing mask
    values mean unavailable; duplicate or unaligned indices raise ValueError.
    """
    mask = evaluation_opportunity_mask(df, split)
    if availability is not None:
        mask &= _aligned_boolean(availability, df.index, "availability")
    elif "score_available" in df:
        mask &= _aligned_boolean(df["score_available"], df.index, "score_available")
    elif "mahalanobis_calibrated" in df:
        mask &= np.isfinite(pd.to_numeric(df["mahalanobis_calibrated"], errors="coerce"))
    return mask


def _aligned_boolean(values: pd.Series, index: pd.Index, name: str) -> pd.Series:
    if not isinstance(values, pd.Series) or not values.index.is_unique or not index.is_unique:
        raise ValueError(f"{name} must be an index-aligned Series with unique indices")
    if not index.isin(values.index).all():
        raise ValueError(f"{name} is missing pitch-row indices")
    aligned = values.reindex(index)
    if not (pd.api.types.is_bool_dtype(aligned.dtype) or aligned.dropna().map(lambda x: isinstance(x, (bool, np.bool_))).all()):
        raise ValueError(f"{name} must contain Boolean values, not numeric scores")
    return aligned.fillna(False).astype(bool)


def _validate_pitch_rows(df: pd.DataFrame, split: Optional[str] = None) -> None:
    if not df.index.is_unique:
        raise ValueError("Pitch-row index must be unique")
    required = {"game_pk", "pitcher", "pitch_number_in_outing"}
    if split is not None:
        required.add("dataset_split")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pitch rows missing required columns: {sorted(missing)}")
    check = df.loc[df["dataset_split"].eq(split)] if split is not None else df
    if check.empty:
        return
    if check[list(required - {"dataset_split"})].isna().any().any():
        raise ValueError("Outing identifiers and pitch numbers cannot be missing")
    number = pd.to_numeric(check["pitch_number_in_outing"], errors="coerce")
    if number.isna().any() or (number <= 0).any() or (number % 1 != 0).any():
        raise ValueError("pitch_number_in_outing must contain positive integers")
    if check.duplicated(["game_pk", "pitcher", "pitch_number_in_outing"]).any():
        raise ValueError("Duplicate pitch identity within an outing")


def build_evaluation_universe(df: pd.DataFrame, split: Optional[str] = None) -> pd.DataFrame:
    """One row per qualified (game_pk, pitcher) in ``split``, before score filtering.

    Requires pitch identity columns and a unique row index. The result has a fresh
    RangeIndex and stable columns game_pk, pitcher, plus available date/split fields.
    Empty input returns those columns with zero rows.
    """
    _validate_pitch_rows(df, split)
    work = df
    if split is not None and "dataset_split" in work:
        work = work[work["dataset_split"].eq(split)]
    columns = ["game_pk", "pitcher"]
    optional = [column for column in ("game_date", "dataset_split") if column in work]
    for column in optional:
        if work.groupby(columns, dropna=False)[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Conflicting {column} within a qualified outing")
    return work[columns + optional].drop_duplicates(columns).reset_index(drop=True)


def episodes_for_split(episodes_df: Optional[pd.DataFrame], split: Optional[str]) -> pd.DataFrame:
    if episodes_df is None or episodes_df.empty:
        return pd.DataFrame()
    result = episodes_df.copy()
    if split is not None and "dataset_split" in result:
        result = result[result["dataset_split"].eq(split)]
    return result


def warning_events(
    df: pd.DataFrame,
    predictions: pd.Series,
    availability: Optional[pd.Series] = None,
    split: Optional[str] = None,
) -> pd.DataFrame:
    """Return starts of adjacent valid flagged runs on the original pitch sequence.

    Requires unique indexed pitch rows and index-aligned Boolean predictions and
    availability. Calibration and unavailable rows break runs; follow-up censoring
    is deliberately ignored here. Output has a fresh RangeIndex and stable columns
    even when no warnings occur. Warning identity is (game_pk, pitcher, warning_pitch).
    """
    columns = [
        "game_pk", "pitcher", "warning_pitch", "warning_pa", "game_date",
        "dataset_split", "actual_data_source",
    ]
    _validate_pitch_rows(df, split)
    if df.empty:
        return pd.DataFrame(columns=columns)

    # This function is called for every threshold and every bootstrap model.
    # Keep the exact index-aligned semantics while avoiding a Python loop over
    # every outing and warning.
    work = df.copy()
    available = _aligned_boolean(availability, df.index, "availability") if availability is not None else (
        _aligned_boolean(df["score_available"], df.index, "score_available") if "score_available" in df else
        pd.Series(True, index=df.index, dtype=bool)
    )
    if "is_calibration_phase" in work:
        available &= ~work["is_calibration_phase"].fillna(True).astype(bool)
    if split is not None:
        available &= work["dataset_split"].eq(split)
    work["_prediction"] = (
        _aligned_boolean(predictions, df.index, "predictions") & available
    ).to_numpy()
    work = work.sort_values(
        ["game_pk", "pitcher", "pitch_number_in_outing"], kind="stable"
    )
    groups = work.groupby(["game_pk", "pitcher"], sort=False)
    previous = groups["_prediction"].shift(fill_value=False)
    pitch_number = pd.to_numeric(work["pitch_number_in_outing"], errors="coerce")
    previous_pitch = groups["pitch_number_in_outing"].shift()
    consecutive_pitch = pitch_number.eq(pd.to_numeric(previous_pitch, errors="coerce") + 1)
    starts = work["_prediction"] & (~previous | ~consecutive_pitch)
    warnings = work.loc[starts].copy()
    if warnings.empty:
        return pd.DataFrame(columns=columns)

    warnings["warning_pitch"] = pd.to_numeric(
        warnings["pitch_number_in_outing"], errors="raise"
    ).astype(int)
    if "pa_number_in_outing" in warnings:
        warnings["warning_pa"] = pd.to_numeric(
            warnings["pa_number_in_outing"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        warnings["warning_pa"] = 0
    if "game_date" not in warnings:
        warnings["game_date"] = pd.NaT
    if "dataset_split" not in warnings:
        warnings["dataset_split"] = None
    if "actual_data_source" not in warnings:
        warnings["actual_data_source"] = "unknown"
    return warnings[columns].reset_index(drop=True)


def evaluate_warning_predictions(
    df: pd.DataFrame,
    episodes_df: Optional[pd.DataFrame],
    predictions: pd.Series,
    horizon_pitches: int = 15,
    split: Optional[str] = None,
    availability: Optional[pd.Series] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Evaluate common outings/episodes and return (metrics, warnings, matches).

    Pitch rows require unique identity, split, and ordered pitch number; episodes
    require game_pk, pitcher, onset_pitch and optional episode_id/onset_pa. All
    model availability is an index-aligned Boolean mask. Raw warning starts are
    retained before follow-up censoring; only evaluable warnings enter precision
    and false-warning rates. Empty warnings/matches retain stable columns.
    """
    if horizon_pitches <= 0:
        raise ValueError("horizon_pitches must be positive")
    _validate_pitch_rows(df, split)
    predictions = _aligned_boolean(predictions, df.index, "predictions")
    if availability is not None:
        availability = _aligned_boolean(availability, df.index, "availability")
    split_mask = pd.Series(True, index=df.index)
    if split is not None and "dataset_split" in df:
        split_mask &= df["dataset_split"].eq(split)
    split_df = df.loc[split_mask].copy()
    universe = build_evaluation_universe(df, split)
    opportunity_mask = evaluation_opportunity_mask(df, split)
    available_mask = eligible_pitch_mask(df, split, availability=availability)
    eval_df = df.loc[available_mask].copy()
    warnings = warning_events(
        split_df,
        predictions.loc[split_df.index],
        availability=availability.loc[split_df.index] if availability is not None else None,
    )
    episodes = episodes_for_split(episodes_df, split)
    if not episodes.empty:
        required = {"game_pk", "pitcher", "onset_pitch"}
        if not required.issubset(episodes.columns):
            raise ValueError(f"Episodes missing columns: {sorted(required - set(episodes.columns))}")
        if episodes[list(required)].isna().any().any():
            raise ValueError("Episode outing identifiers and onset cannot be missing")
        identity = ["game_pk", "pitcher", "episode_id"] if "episode_id" in episodes else ["game_pk", "pitcher", "onset_pitch"]
        if episodes.duplicated(identity).any():
            raise ValueError("Duplicate episode identity")
        episodes = episodes.merge(
            universe[["game_pk", "pitcher"]], on=["game_pk", "pitcher"], how="inner"
        )

    # Follow-up censoring affects evaluability, never the raw warning timestamp.
    if not warnings.empty:
        censored = split_df.set_index(["game_pk", "pitcher", "pitch_number_in_outing"])["is_censored_followup"] if "is_censored_followup" in split_df else None
        warnings["is_evaluable_warning"] = (
            ~pd.Series([bool(censored.loc[(r.game_pk, r.pitcher, r.warning_pitch)]) for r in warnings.itertuples()], index=warnings.index)
            if censored is not None else True
        )
    else:
        warnings["is_evaluable_warning"] = pd.Series(dtype=bool)
    evaluable_warnings = warnings.loc[warnings["is_evaluable_warning"]]
    matches = []
    matched_episode_keys = set()
    matched_warning_indices = set()
    episode_groups = {
        key: group.sort_values("onset_pitch")
        for key, group in episodes.groupby(["game_pk", "pitcher"], sort=False)
    } if not episodes.empty else {}
    for outing_key, outing_warnings in evaluable_warnings.groupby(["game_pk", "pitcher"], sort=False):
        outing_episodes = episode_groups.get(outing_key)
        if outing_episodes is None:
            continue
        matched_local = np.zeros(len(outing_episodes), dtype=bool)
        onsets = outing_episodes["onset_pitch"].to_numpy()
        for warning_idx, warning in outing_warnings.sort_values("warning_pitch").iterrows():
            candidate_positions = np.flatnonzero(
                (~matched_local)
                & (onsets > warning["warning_pitch"])
                & (onsets <= warning["warning_pitch"] + horizon_pitches)
            )
            if not len(candidate_positions):
                continue
            position = int(candidate_positions[0])
            matched_local[position] = True
            episode_idx = outing_episodes.index[position]
            episode = outing_episodes.iloc[position]
            episode_key = (episode["game_pk"], episode["pitcher"], episode.get("episode_id", episode_idx))
            matched_episode_keys.add(episode_key)
            matched_warning_indices.add(warning_idx)
            matches.append({
                **warning.to_dict(), "episode_id": episode.get("episode_id"),
                "onset_pitch": int(episode["onset_pitch"]),
                "onset_pa": int(episode.get("onset_pa", 0)),
                "lead_time_pitches": int(episode["onset_pitch"] - warning["warning_pitch"]),
                "lead_time_pas": max(0, int(episode.get("onset_pa", 0) - warning["warning_pa"])),
            })

    match_df = pd.DataFrame(matches, columns=list(warnings.columns) + [
        "episode_id", "onset_pitch", "onset_pa", "lead_time_pitches", "lead_time_pas"
    ])
    if not warnings.empty:
        warnings = warnings.copy()
        warnings["is_matched_warning"] = warnings.index.isin(matched_warning_indices)

    outing_keys = universe[["game_pk", "pitcher"]]
    total_outings = len(outing_keys)
    episode_outing_keys = set()
    if not episodes.empty:
        episode_outing_keys = set(map(tuple, episodes[["game_pk", "pitcher"]].drop_duplicates().values))
    warning_outing_keys = set()
    if not warnings.empty:
        warning_outing_keys = set(map(tuple, evaluable_warnings[["game_pk", "pitcher"]].drop_duplicates().values))
    clean_outing_keys = set(map(tuple, outing_keys.values)) - episode_outing_keys
    clean_with_warning = clean_outing_keys & warning_outing_keys

    total_warnings = len(evaluable_warnings)
    matched_warnings = len(matched_warning_indices)
    false_warnings = total_warnings - matched_warnings
    total_episodes = len(episodes)
    leads = match_df.get("lead_time_pitches", pd.Series(dtype=float))
    lead_pas = match_df.get("lead_time_pas", pd.Series(dtype=float))

    available_outings = eval_df[["game_pk", "pitcher"]].drop_duplicates()
    opportunity_count = int(opportunity_mask.sum())
    available_count = int(available_mask.sum())
    metrics = {
        "evaluated_outings_count": int(total_outings),
        "evaluated_pitches_count": available_count,
        "total_qualified_outings": int(total_outings),
        "outings_with_available_score": int(len(available_outings)),
        "outings_without_available_score": int(total_outings - len(available_outings)),
        "evaluation_opportunity_pitches": opportunity_count,
        "score_available_evaluation_pitches": available_count,
        "pitch_level_scoring_coverage": (
            float(available_count / opportunity_count) if opportunity_count else np.nan
        ),
        "outing_level_scoring_coverage": (
            float(len(available_outings) / total_outings) if total_outings else np.nan
        ),
        "total_collapse_episodes": int(total_episodes),
        "evaluated_episodes_count": int(total_episodes),
        "collapse_episodes_detected": int(len(matched_episode_keys)),
        "episode_recall": float(len(matched_episode_keys) / total_episodes) if total_episodes else np.nan,
        "warning_precision": float(matched_warnings / total_warnings) if total_warnings else np.nan,
        "master_warning_count": int(len(warnings)),
        "censored_warnings": int(len(warnings) - total_warnings),
        "total_warnings": int(total_warnings),
        "false_warnings": int(false_warnings),
        "false_warnings_per_outing": float(false_warnings / total_outings) if total_outings else np.nan,
        "clean_outing_false_alarm_rate": (
            float(len(clean_with_warning) / len(clean_outing_keys)) if clean_outing_keys else np.nan
        ),
        "lead_time_mean_pitches": float(leads.mean()) if len(leads) else np.nan,
        "lead_time_median_pitches": float(leads.median()) if len(leads) else np.nan,
        "lead_time_mean_pas": float(lead_pas.mean()) if len(lead_pas) else np.nan,
        "lead_time_median_pas": float(lead_pas.median()) if len(lead_pas) else np.nan,
        "headline_denominators": {
            "episode_recall": int(total_episodes),
            "warning_precision": int(total_warnings),
            "false_warnings_per_outing": int(total_outings),
            "clean_outing_false_alarm_rate": int(len(clean_outing_keys)),
            "pitch_level_scoring_coverage": opportunity_count,
            "outing_level_scoring_coverage": total_outings,
        },
    }
    return metrics, warnings, match_df


def select_operating_threshold(
    df: pd.DataFrame,
    episodes_df: Optional[pd.DataFrame],
    score_col: str,
    false_warnings_per_outing: float,
    horizon_pitches: int = 15,
    split: str = "validation",
) -> Tuple[float, Dict[str, Any]]:
    """Select the best recall threshold under a shared false-warning allowance."""
    score_available = df[score_col].replace([np.inf, -np.inf], np.nan).notna()
    eligible = df.loc[eligible_pitch_mask(df, split, availability=score_available)]
    finite = eligible[score_col].replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        predictions = pd.Series(False, index=df.index)
        metrics, _, _ = evaluate_warning_predictions(
            df, episodes_df, predictions, horizon_pitches=horizon_pitches,
            split=split, availability=score_available,
        )
        return np.inf, metrics

    candidates = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 31)))
    candidates = np.r_[np.inf, candidates[::-1]]
    best_threshold = np.inf
    best_metrics: Dict[str, Any] = {}
    best_key = (-1.0, -1.0, -np.inf)
    for threshold in candidates:
        predictions = df[score_col].ge(threshold) & df[score_col].notna()
        metrics, _, _ = evaluate_warning_predictions(
            df, episodes_df, predictions, horizon_pitches=horizon_pitches, split=split,
            availability=score_available,
        )
        if metrics["false_warnings_per_outing"] > false_warnings_per_outing + 1e-12:
            continue
        key = (
            -1.0 if np.isnan(metrics["episode_recall"]) else metrics["episode_recall"],
            -1.0 if np.isnan(metrics["warning_precision"]) else metrics["warning_precision"],
            -metrics["false_warnings_per_outing"],
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def threshold_tradeoff_curve(
    df: pd.DataFrame,
    episodes_df: Optional[pd.DataFrame],
    score_col: str,
    horizon_pitches: int = 15,
    split: str = "validation",
    candidates: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Evaluate every candidate threshold without selecting on test data."""
    score_available = df[score_col].replace([np.inf, -np.inf], np.nan).notna()
    eligible = df.loc[eligible_pitch_mask(df, split, availability=score_available)]
    finite = eligible[score_col].replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return pd.DataFrame(columns=[
            "threshold", "episode_recall", "warning_precision", "false_warnings_per_outing",
        ])
    if candidates is None:
        candidates = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 31)))
        candidates = np.r_[np.inf, candidates[::-1]]
    rows = []
    for threshold in candidates:
        predictions = df[score_col].ge(threshold) & df[score_col].notna()
        metrics, _, _ = evaluate_warning_predictions(
            df, episodes_df, predictions, horizon_pitches=horizon_pitches, split=split,
            availability=score_available,
        )
        rows.append({"threshold": float(threshold), **metrics})
    return pd.DataFrame(rows)
