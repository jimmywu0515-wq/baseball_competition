"""Paired test-outing bootstrap confidence intervals."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.evaluation.protocol import (
    build_evaluation_universe,
    eligible_pitch_mask,
    evaluate_warning_predictions,
)


METRICS = (
    "episode_recall",
    "warning_precision",
    "false_warnings_per_outing",
    "risk_ratio_alert_vs_no_alert",
)


def _outing_sufficient_statistics(
    df: pd.DataFrame,
    episodes: pd.DataFrame,
    model_predictions: Dict[str, str],
    split: str,
    horizon_pitches: int,
    model_availability: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    rows = []
    split_df = df[df["dataset_split"].eq(split)]
    universe = build_evaluation_universe(df, split)
    split_episodes = episodes[episodes["dataset_split"].eq(split)] if not episodes.empty else episodes
    grouped = split_df.groupby(["game_pk", "pitcher"], sort=False)
    for game_pk, pitcher in universe[["game_pk", "pitcher"]].itertuples(index=False, name=None):
        outing = grouped.get_group((game_pk, pitcher))
        outing_episodes = split_episodes[
            split_episodes["game_pk"].eq(game_pk) & split_episodes["pitcher"].eq(pitcher)
        ] if not split_episodes.empty else pd.DataFrame()
        for model_key, prediction_col in model_predictions.items():
            availability_col = (model_availability or {}).get(model_key)
            availability = (
                outing[availability_col].notna()
                if availability_col and availability_col in outing
                else None
            )
            metrics, warnings, _ = evaluate_warning_predictions(
                outing, outing_episodes, outing[prediction_col],
                horizon_pitches=horizon_pitches, split=split,
                availability=availability,
            )
            eligible = outing.loc[eligible_pitch_mask(outing, availability=availability)]
            pred = eligible[prediction_col].fillna(False).astype(bool)
            outcome = eligible["y_true_onset_in_horizon"].fillna(False).astype(bool)
            rows.append({
                "game_pk": game_pk,
                "pitcher": pitcher,
                "model_key": model_key,
                "episode_detected_n": metrics["collapse_episodes_detected"],
                "episode_n": metrics["total_collapse_episodes"],
                "matched_warning_n": int(warnings.get("is_matched_warning", pd.Series(dtype=bool)).sum()),
                "warning_n": metrics["total_warnings"],
                "false_warning_n": metrics["false_warnings"],
                "outing_n": 1,
                "alert_positive_n": int((outcome & pred).sum()),
                "alert_pitch_n": int(pred.sum()),
                "no_alert_positive_n": int((outcome & ~pred).sum()),
                "no_alert_pitch_n": int((~pred).sum()),
            })
    return pd.DataFrame(rows)


def _metrics_from_totals(totals: np.ndarray, columns: Dict[str, int]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    def ratio(numerator: str, denominator: str) -> Tuple[np.ndarray, np.ndarray]:
        den = totals[:, columns[denominator]]
        num = totals[:, columns[numerator]]
        invalid = den == 0
        values = np.divide(num, den, out=np.full(len(den), np.nan), where=~invalid)
        return values, invalid

    recall, recall_zero = ratio("episode_detected_n", "episode_n")
    precision, precision_zero = ratio("matched_warning_n", "warning_n")
    false_rate, outing_zero = ratio("false_warning_n", "outing_n")
    alert_risk, alert_zero = ratio("alert_positive_n", "alert_pitch_n")
    no_alert_risk, no_alert_zero = ratio("no_alert_positive_n", "no_alert_pitch_n")
    risk_invalid = alert_zero | no_alert_zero | (no_alert_risk == 0)
    risk_ratio = np.divide(
        alert_risk, no_alert_risk, out=np.full(len(alert_risk), np.nan), where=~risk_invalid
    )
    values = np.column_stack([recall, precision, false_rate, risk_ratio])
    invalid = {
        "episode_recall": recall_zero,
        "warning_precision": precision_zero,
        "false_warnings_per_outing": outing_zero,
        "risk_ratio_alert_vs_no_alert": risk_invalid,
    }
    return values, invalid


def paired_bootstrap_confidence_intervals(
    df: pd.DataFrame,
    episodes: Optional[pd.DataFrame],
    model_predictions: Dict[str, str],
    horizon_pitches: int = 15,
    split: str = "test",
    n_bootstrap: int = 1000,
    seed: int = 2025,
    cluster_by_pitcher: bool = False,
    model_availability: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Use the same resampled outings or pitchers for every compared model."""
    episodes = episodes if episodes is not None else pd.DataFrame()
    stats = _outing_sufficient_statistics(
        df, episodes, model_predictions, split=split, horizon_pitches=horizon_pitches,
        model_availability=model_availability,
    )
    if stats.empty:
        return pd.DataFrame()

    value_columns = [
        "episode_detected_n", "episode_n", "matched_warning_n", "warning_n",
        "false_warning_n", "outing_n", "alert_positive_n", "alert_pitch_n",
        "no_alert_positive_n", "no_alert_pitch_n",
    ]
    column_index = {name: idx for idx, name in enumerate(value_columns)}
    models = list(model_predictions)
    unit_column = "pitcher" if cluster_by_pitcher else "game_pk_pitcher"
    stats["game_pk_pitcher"] = stats["game_pk"].astype(str) + ":" + stats["pitcher"].astype(str)
    units = sorted(stats[unit_column].unique())
    arrays = {}
    point_totals = {}
    point_values = {}
    for model in models:
        model_stats = stats[stats["model_key"].eq(model)]
        aggregate = model_stats.groupby(unit_column)[value_columns].sum().reindex(units, fill_value=0)
        arrays[model] = aggregate.to_numpy(dtype=float)
        point_totals[model] = arrays[model].sum(axis=0, keepdims=True)
        prediction_col = model_predictions[model]
        availability_col = (model_availability or {}).get(model)
        availability = (
            df[availability_col].notna()
            if availability_col and availability_col in df
            else None
        )
        ordinary, _, _ = evaluate_warning_predictions(
            df, episodes, df[prediction_col], split=split,
            horizon_pitches=horizon_pitches, availability=availability,
        )
        point, _ = _metrics_from_totals(point_totals[model], column_index)
        point[0, 0] = ordinary["episode_recall"]
        point[0, 1] = ordinary["warning_precision"]
        point[0, 2] = ordinary["false_warnings_per_outing"]
        point_values[model] = point

    rng = np.random.default_rng(seed)
    sampled_positions = rng.integers(0, len(units), size=(n_bootstrap, len(units)))
    draws_by_model = {}
    invalid_by_model = {}
    rows = []
    for model in models:
        totals = arrays[model][sampled_positions].sum(axis=1)
        draws, invalid = _metrics_from_totals(totals, column_index)
        point = point_values[model]
        draws_by_model[model] = draws
        invalid_by_model[model] = invalid
        for metric_idx, metric in enumerate(METRICS):
            finite = draws[:, metric_idx][np.isfinite(draws[:, metric_idx])]
            rows.append({
                "resampling_unit": "pitcher" if cluster_by_pitcher else "outing",
                "comparison": model,
                "metric": metric,
                "estimate": float(point[0, metric_idx]),
                "ci_lower_95": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
                "ci_upper_95": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
                "zero_denominator_frequency": float(invalid[metric].mean()),
                "bootstrap_samples": n_bootstrap,
                "cluster_count": len(units),
            })

    proposed = "proposed"
    if proposed in draws_by_model:
        for competitor in [model for model in models if model != proposed]:
            for metric_idx, metric in enumerate(METRICS):
                difference = draws_by_model[proposed][:, metric_idx] - draws_by_model[competitor][:, metric_idx]
                finite = difference[np.isfinite(difference)]
                point_proposed = point_values[proposed]
                point_competitor = point_values[competitor]
                rows.append({
                    "resampling_unit": "pitcher" if cluster_by_pitcher else "outing",
                    "comparison": f"proposed_minus_{competitor}",
                    "metric": metric,
                    "estimate": float(point_proposed[0, metric_idx] - point_competitor[0, metric_idx]),
                    "ci_lower_95": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
                    "ci_upper_95": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
                    "zero_denominator_frequency": float((~np.isfinite(difference)).mean()),
                    "bootstrap_samples": n_bootstrap,
                    "cluster_count": len(units),
                })
    return pd.DataFrame(rows)
