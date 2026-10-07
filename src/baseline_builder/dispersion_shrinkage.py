"""Fit frozen pitcher dispersion scales from completed 2023 training data.

These parameters may be applied only AFTER their training cutoff. They cannot
be replayed onto earlier training outings as if they were available at the time.
Non-pre-onset is an outcome-based reference population, not a fatigue diagnosis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def shrinkage_weight(n_pitches: int, prior_pitches: float) -> float:
    """Prior pitch count gives 50% shrinkage at n == prior_pitches."""
    if n_pitches < 0 or not np.isfinite(prior_pitches) or prior_pitches <= 0:
        raise ValueError("Require a nonnegative sample size and positive finite prior pitch count")
    return n_pitches / (n_pitches + prior_pitches)


def fit_pitcher_dispersion_scales(
    train: pd.DataFrame, score_col: str = "mahalanobis_calibrated",
    prior_pitches: float | None = None, min_outings: int = 5,
    min_pitches: int = 30, calibration_pitches: int = 20,
    strict_clean: bool = False, training_end: str = "2023-12-31",
) -> tuple[pd.DataFrame, dict]:
    """Return pitcher scales plus the train-only pooled reference parameters.

    No validation/test rows are accepted. The default prior is the median
    eligible pitch count among qualifying pitchers, determined from train only.
    Callers must use scale 1 for pitchers without qualifying training scores.
    """
    dates = pd.to_datetime(train.game_date, errors="coerce")
    if (not train.dataset_split.eq("train").all() or dates.isna().any()
            or dates.dt.year.ne(2023).any() or dates.gt(training_end).any()):
        raise ValueError("Dispersion scales accept 2023 train rows only")
    if min_outings < 1 or min_pitches < 2 or calibration_pitches < 0:
        raise ValueError("Invalid eligibility limits")
    if train.y_true_onset_in_horizon.isna().any():
        raise ValueError("Missing training onset targets cannot identify the reference population")
    values = pd.to_numeric(train[score_col], errors="coerce")
    mask = (np.isfinite(values) & train.score_available.fillna(False).astype(bool)
            & ~train.is_calibration_phase.fillna(True).astype(bool)
            & train.pitch_number_in_outing.gt(calibration_pitches)
            & ~train.y_true_onset_in_horizon.astype(bool))
    if strict_clean:
        mask &= (~train.is_collapse_event.fillna(True).astype(bool)
                 & ~train.is_censored_followup.fillna(True).astype(bool))
    selected = train.loc[mask, ["pitcher", "game_pk"]].copy()
    selected["score"] = values.loc[mask]
    stats = selected.groupby("pitcher").agg(
        pitcher_std=("score", "std"), n_eligible_pitches=("score", "size"),
        n_eligible_outings=("game_pk", "nunique"),
    )
    stats = stats.loc[(stats.n_eligible_outings >= min_outings) & (stats.n_eligible_pitches >= min_pitches)
                      & np.isfinite(stats.pitcher_std) & stats.pitcher_std.gt(1e-6)].copy()
    if len(stats) < 2:
        raise ValueError("At least two pitchers need qualifying training dispersion")
    pooled = selected.loc[selected.pitcher.isin(stats.index), "score"]
    global_std = float(pooled.std(ddof=1))
    if not np.isfinite(global_std) or global_std <= 1e-6:
        raise ValueError("Pooled training dispersion must be finite and nonzero")
    resolved_prior = float(stats.n_eligible_pitches.median() if prior_pitches is None else prior_pitches)
    stats["raw_scale"] = stats.pitcher_std / global_std
    stats["shrinkage_weight"] = [shrinkage_weight(int(n), resolved_prior) for n in stats.n_eligible_pitches]
    stats["scale_used"] = stats.shrinkage_weight * stats.raw_scale + (1 - stats.shrinkage_weight)
    metadata = {
        "score_column": score_col, "training_start": dates.min().date().isoformat(),
        "training_end": training_end, "global_mean": float(pooled.mean()), "global_std": global_std,
        "prior_pitches": resolved_prior, "qualifying_pitchers": len(stats),
        "training_pitches": len(pooled), "strict_clean": strict_clean,
        "unseen_or_insufficient_training_fallback_scale": 1.0,
        "application_policy": "Frozen parameters only after the training-end cutoff; no retrospectively fitted train scores",
    }
    return stats.reset_index(), metadata
