"""Lead-time sensitivity with frozen scores, thresholds, and warning times."""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from src.evaluation.protocol import eligible_pitch_mask, episodes_for_split, warning_events


def _match_fixed_warnings(
    warnings: pd.DataFrame,
    episodes: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    matches = []
    episode_groups = {
        key: group.sort_values("onset_pitch")
        for key, group in episodes.groupby(["game_pk", "pitcher"], sort=False)
    } if not episodes.empty else {}
    for outing_key, outing_warnings in warnings.groupby(["game_pk", "pitcher"], sort=False):
        outing_episodes = episode_groups.get(outing_key)
        if outing_episodes is None:
            continue
        used = np.zeros(len(outing_episodes), dtype=bool)
        onsets = outing_episodes["onset_pitch"].to_numpy()
        for _, warning in outing_warnings.sort_values("warning_pitch").iterrows():
            candidates = np.flatnonzero(
                (~used) & (onsets > warning["warning_pitch"])
                & (onsets <= warning["warning_pitch"] + horizon)
            )
            if not len(candidates):
                continue
            position = int(candidates[0])
            used[position] = True
            episode = outing_episodes.iloc[position]
            matches.append({
                **warning.to_dict(),
                "episode_id": episode.get("episode_id"),
                "onset_pitch": int(episode["onset_pitch"]),
                "onset_pa": int(episode.get("onset_pa", 0)),
                "lead_time_pitches": int(episode["onset_pitch"] - warning["warning_pitch"]),
                "matching_horizon": int(horizon),
            })
    return pd.DataFrame(matches)


def fixed_warning_horizon_analysis(
    df: pd.DataFrame,
    episodes_df: Optional[pd.DataFrame],
    model_predictions: Dict[str, str],
    horizons: Iterable[int] = (10, 15, 20, 25),
    split: str = "test",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Change only event-matching horizon; warnings and thresholds remain fixed."""
    horizons = sorted(set(map(int, horizons)))
    max_horizon = max(horizons)
    # Score/calibration eligibility is fixed. Horizon-specific censoring is
    # applied below instead of inheriting the default 15-pitch label mask.
    eligibility_frame = df.copy()
    eligibility_frame["is_censored_followup"] = False
    eligible = eligibility_frame.loc[eligible_pitch_mask(eligibility_frame, split)].copy()
    episodes = episodes_for_split(episodes_df, split)
    valid_outings = eligible[["game_pk", "pitcher"]].drop_duplicates()
    if not episodes.empty:
        episodes = episodes.merge(valid_outings, on=["game_pk", "pitcher"], how="inner")
    outing_end = df.groupby(["game_pk", "pitcher"])["pitch_number_in_outing"].max().to_dict()

    result_rows = []
    match_frames = []
    distribution_rows = []
    eligible["available_followup_pitches"] = [
        max(0, int(outing_end.get((row.game_pk, row.pitcher), row.pitch_number_in_outing)
                   - row.pitch_number_in_outing))
        for row in eligible.itertuples()
    ]

    def evaluable_episodes(pitch_population: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if episodes.empty:
            return episodes
        pitches_by_outing = {
            key: group["pitch_number_in_outing"].to_numpy()
            for key, group in pitch_population.groupby(["game_pk", "pitcher"], sort=False)
        }
        keep = []
        for idx, episode in episodes.iterrows():
            pitches = pitches_by_outing.get((episode["game_pk"], episode["pitcher"]), np.array([]))
            if np.any((pitches < episode["onset_pitch"]) &
                      (pitches >= episode["onset_pitch"] - horizon)):
                keep.append(idx)
        return episodes.loc[keep]

    common_pitch_population = eligible[eligible["available_followup_pitches"].ge(max_horizon)]
    common_episode_population = evaluable_episodes(common_pitch_population, max_horizon)

    for model_key, prediction_col in model_predictions.items():
        fixed_warnings = warning_events(eligible, eligible[prediction_col])
        if not fixed_warnings.empty:
            fixed_warnings["available_followup_pitches"] = [
                max(0, int(outing_end.get((row.game_pk, row.pitcher), row.warning_pitch) - row.warning_pitch))
                for row in fixed_warnings.itertuples()
            ]
        common_warnings = fixed_warnings[
            fixed_warnings.get("available_followup_pitches", pd.Series(dtype=int)).ge(max_horizon)
        ] if not fixed_warnings.empty else fixed_warnings
        for horizon in horizons:
            complete = fixed_warnings[
                fixed_warnings["available_followup_pitches"].ge(horizon)
            ] if not fixed_warnings.empty else fixed_warnings
            complete_pitch_population = eligible[eligible["available_followup_pitches"].ge(horizon)]
            horizon_episodes = evaluable_episodes(complete_pitch_population, horizon)
            matches = _match_fixed_warnings(complete, horizon_episodes, horizon)
            common_matches = _match_fixed_warnings(
                common_warnings, common_episode_population, horizon
            )
            matched_warning_count = len(matches)
            result_rows.append({
                "model_key": model_key,
                "matching_horizon": horizon,
                "frozen_warning_count": int(len(fixed_warnings)),
                "complete_followup_warning_count": int(len(complete)),
                "censored_warning_count": int(len(fixed_warnings) - len(complete)),
                "followup_coverage": float(len(complete) / len(fixed_warnings)) if len(fixed_warnings) else np.nan,
                "episode_count": int(len(horizon_episodes)),
                "matched_episode_count": int(len(matches)),
                "episode_recall_complete_followup": (
                    float(len(matches) / len(horizon_episodes)) if len(horizon_episodes) else np.nan
                ),
                "warning_precision_complete_followup": float(matched_warning_count / len(complete)) if len(complete) else np.nan,
                "common_followup_warning_count": int(len(common_warnings)),
                "common_followup_episode_count": int(len(common_episode_population)),
                "common_followup_matched_episode_count": int(len(common_matches)),
                "common_followup_episode_recall": (
                    float(len(common_matches) / len(common_episode_population))
                    if len(common_episode_population) else np.nan
                ),
                "actual_data_source": ",".join(sorted(map(str, eligible.get(
                    "actual_data_source", pd.Series(["unknown"])
                ).dropna().unique()))),
            })
            if not matches.empty:
                match_frames.append(matches.assign(model_key=model_key, analysis_population="complete_followup"))
                for lead, count in matches["lead_time_pitches"].value_counts().sort_index().items():
                    distribution_rows.append({
                        "model_key": model_key,
                        "matching_horizon": horizon,
                        "lead_time_pitches": int(lead),
                        "matched_count": int(count),
                    })
    return (
        pd.DataFrame(result_rows),
        pd.concat(match_frames, ignore_index=True) if match_frames else pd.DataFrame(),
        pd.DataFrame(distribution_rows),
    )
