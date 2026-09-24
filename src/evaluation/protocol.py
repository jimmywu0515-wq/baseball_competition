"""Shared temporal split, eligibility, warning, and event-matching protocol."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


TRAIN_END = pd.Timestamp("2023-12-31")
VALIDATION_END = pd.Timestamp("2024-12-31")
TEST_START = pd.Timestamp("2025-01-01")


def assign_temporal_split(df: pd.DataFrame) -> pd.DataFrame:
    """Assign the frozen primary protocol: 2023 train, 2024 validation, 2025 test."""
    result = df.copy()
    dates = pd.to_datetime(result["game_date"], errors="coerce")
    result["dataset_split"] = np.select(
        [dates <= TRAIN_END, (dates > TRAIN_END) & (dates <= VALIDATION_END), dates >= TEST_START],
        ["train", "validation", "test"],
        default="unassigned",
    )
    result.loc[dates.isna(), "dataset_split"] = "unassigned"
    return result


def assign_historical_temporal_split(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve the earlier 2023 / early-2024 / late-2024 historical experiment."""
    result = df.copy()
    dates = pd.to_datetime(result["game_date"], errors="coerce")
    result["dataset_split"] = np.select(
        [
            dates <= TRAIN_END,
            (dates > TRAIN_END) & (dates <= pd.Timestamp("2024-06-30")),
            (dates >= pd.Timestamp("2024-07-01")) & (dates <= pd.Timestamp("2024-12-31")),
        ],
        ["train", "validation", "test"],
        default="unassigned",
    )
    result.loc[dates.isna(), "dataset_split"] = "unassigned"
    return result


def eligible_pitch_mask(df: pd.DataFrame, split: Optional[str] = None) -> pd.Series:
    """Return the single scoring/evaluation/display eligibility policy."""
    mask = pd.Series(True, index=df.index, dtype=bool)
    if "pitch_number_in_outing" in df:
        mask &= df["pitch_number_in_outing"] > 20
    if "is_calibration_phase" in df:
        mask &= ~df["is_calibration_phase"].fillna(True).astype(bool)
    if "score_available" in df:
        mask &= df["score_available"].fillna(False).astype(bool)
    elif "mahalanobis_calibrated" in df:
        mask &= df["mahalanobis_calibrated"].notna()
    if "is_censored_followup" in df:
        mask &= ~df["is_censored_followup"].fillna(True).astype(bool)
    if split is not None and "dataset_split" in df:
        mask &= df["dataset_split"].eq(split)
    return mask


def episodes_for_split(episodes_df: Optional[pd.DataFrame], split: Optional[str]) -> pd.DataFrame:
    if episodes_df is None or episodes_df.empty:
        return pd.DataFrame()
    result = episodes_df.copy()
    if split is not None and "dataset_split" in result:
        result = result[result["dataset_split"].eq(split)]
    return result


def warning_events(df: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
    """Collapse consecutive flagged pitches into distinct warning events."""
    columns = [
        "game_pk", "pitcher", "warning_pitch", "warning_pa", "game_date",
        "dataset_split", "actual_data_source",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    # This function is called for every threshold and every bootstrap model.
    # Keep the exact index-aligned semantics while avoiding a Python loop over
    # every outing and warning.
    work = df.copy()
    work["_prediction"] = predictions.reindex(df.index).fillna(False).astype(bool).to_numpy()
    work = work.sort_values(
        ["game_pk", "pitcher", "pitch_number_in_outing"], kind="stable"
    )
    prev_prediction = work.groupby(["game_pk", "pitcher"], sort=False)["_prediction"].shift(
        fill_value=False
    )
    if "pitch_number_in_outing" in work.columns:
        prev_pitch = work.groupby(["game_pk", "pitcher"], sort=False)["pitch_number_in_outing"].shift()
        curr_pitch = pd.to_numeric(work["pitch_number_in_outing"], errors="coerce")
        is_contiguous = (curr_pitch - prev_pitch) == 1
        previous = prev_prediction & is_contiguous
    else:
        previous = prev_prediction
    starts = work["_prediction"] & ~previous
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
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Evaluate warnings and episodes with one-to-one, timely event matching."""
    eval_df = df.loc[eligible_pitch_mask(df, split)].copy()
    eval_pred = predictions.reindex(eval_df.index).fillna(False).astype(bool)
    warnings = warning_events(eval_df, eval_pred)
    episodes = episodes_for_split(episodes_df, split)

    if not episodes.empty:
        valid_outings = eval_df[["game_pk", "pitcher"]].drop_duplicates()
        episodes = episodes.merge(valid_outings, on=["game_pk", "pitcher"], how="inner")
        # Do not count an episode when its entire warning window occurred during
        # calibration or otherwise had no available, uncensored score.
        pitch_numbers_by_outing = {
            key: group["pitch_number_in_outing"].to_numpy()
            for key, group in eval_df.groupby(["game_pk", "pitcher"], sort=False)
        }
        evaluable_episode_indices = []
        for episode_idx, episode in episodes.iterrows():
            pitches = pitch_numbers_by_outing.get((episode["game_pk"], episode["pitcher"]), np.array([]))
            if np.any((pitches < episode["onset_pitch"]) &
                      (pitches >= episode["onset_pitch"] - horizon_pitches)):
                evaluable_episode_indices.append(episode_idx)
        episodes = episodes.loc[evaluable_episode_indices]

    matches = []
    matched_episode_keys = set()
    matched_warning_indices = set()
    episode_groups = {
        key: group.sort_values("onset_pitch")
        for key, group in episodes.groupby(["game_pk", "pitcher"], sort=False)
    } if not episodes.empty else {}
    for outing_key, outing_warnings in warnings.groupby(["game_pk", "pitcher"], sort=False):
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

    match_df = pd.DataFrame(matches)
    if not warnings.empty:
        warnings = warnings.copy()
        warnings["is_matched_warning"] = warnings.index.isin(matched_warning_indices)

    outing_keys = eval_df[["game_pk", "pitcher"]].drop_duplicates()
    total_outings = len(outing_keys)
    episode_outing_keys = set()
    if not episodes.empty:
        episode_outing_keys = set(map(tuple, episodes[["game_pk", "pitcher"]].drop_duplicates().values))
    warning_outing_keys = set()
    if not warnings.empty:
        warning_outing_keys = set(map(tuple, warnings[["game_pk", "pitcher"]].drop_duplicates().values))
    clean_outing_keys = set(map(tuple, outing_keys.values)) - episode_outing_keys
    clean_with_warning = clean_outing_keys & warning_outing_keys

    total_warnings = len(warnings)
    matched_warnings = len(matched_warning_indices)
    false_warnings = total_warnings - matched_warnings
    total_episodes = len(episodes)
    leads = match_df.get("lead_time_pitches", pd.Series(dtype=float))
    lead_pas = match_df.get("lead_time_pas", pd.Series(dtype=float))

    metrics = {
        "evaluated_outings_count": int(total_outings),
        "evaluated_pitches_count": int(len(eval_df)),
        "total_collapse_episodes": int(total_episodes),
        "collapse_episodes_detected": int(len(matched_episode_keys)),
        "episode_recall": float(len(matched_episode_keys) / total_episodes) if total_episodes else 0.0,
        "warning_precision": float(matched_warnings / total_warnings) if total_warnings else 0.0,
        "total_warnings": int(total_warnings),
        "false_warnings": int(false_warnings),
        "false_warnings_per_outing": float(false_warnings / total_outings) if total_outings else 0.0,
        "clean_outing_false_alarm_rate": (
            float(len(clean_with_warning) / len(clean_outing_keys)) if clean_outing_keys else 0.0
        ),
        "lead_time_mean_pitches": float(leads.mean()) if len(leads) else np.nan,
        "lead_time_median_pitches": float(leads.median()) if len(leads) else np.nan,
        "lead_time_mean_pas": float(lead_pas.mean()) if len(lead_pas) else np.nan,
        "lead_time_median_pas": float(lead_pas.median()) if len(lead_pas) else np.nan,
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
    eligible = df.loc[eligible_pitch_mask(df, split)]
    finite = eligible[score_col].replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return np.inf, {}

    candidates = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 31)))
    candidates = np.r_[np.inf, candidates[::-1]]
    best_threshold = np.inf
    best_metrics: Dict[str, Any] = {}
    best_key = (-1.0, -1.0, -np.inf)
    for threshold in candidates:
        predictions = df[score_col].ge(threshold) & df[score_col].notna()
        metrics, _, _ = evaluate_warning_predictions(
            df, episodes_df, predictions, horizon_pitches=horizon_pitches, split=split
        )
        if metrics["false_warnings_per_outing"] > false_warnings_per_outing + 1e-12:
            continue
        key = (
            metrics["episode_recall"],
            metrics["warning_precision"],
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
    eligible = df.loc[eligible_pitch_mask(df, split)]
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
            df, episodes_df, predictions, horizon_pitches=horizon_pitches, split=split
        )
        rows.append({"threshold": float(threshold), **metrics})
    return pd.DataFrame(rows)
