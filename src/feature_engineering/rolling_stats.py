"""
Rolling Trend and Dispersion Features (Level 1 Features)
Computes in-outing rolling statistics, circular spin deviations, and velocity dispersion.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def circular_angular_diff(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """Computes minimum circular difference between angles in [0, 360)."""
    diff = np.abs(a1 - a2) % 360.0
    return np.minimum(diff, 360.0 - diff)


def _rolling_circular_std(series: pd.Series, window: int, minimum: int) -> pd.Series:
    """Circular standard deviation in degrees, preserving insufficient data as NaN."""
    radians = np.deg2rad(pd.to_numeric(series, errors="coerce"))
    mean_sin = np.sin(radians).rolling(window, min_periods=minimum).mean()
    mean_cos = np.cos(radians).rolling(window, min_periods=minimum).mean()
    resultant = np.sqrt(mean_sin**2 + mean_cos**2).clip(lower=1e-12, upper=1.0)
    return np.rad2deg(np.sqrt(-2.0 * np.log(resultant))).clip(upper=180.0)

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Level 1 rolling window features strictly grouped by (game_pk, pitcher).
    No intra-game lookahead (strictly backwards-looking rolling window).
    """
    res = df.copy()
    
    if "pitch_number_in_outing" not in res.columns:
        if "pitch_number_in_game" in res.columns:
            res["pitch_number_in_outing"] = res["pitch_number_in_game"]
        else:
            res["pitch_number_in_outing"] = res.groupby(["game_pk", "pitcher"]).cumcount() + 1
            res["pitch_number_in_game"] = res["pitch_number_in_outing"]

    res = res.sort_values(by=["game_pk", "pitcher", "pitch_number_in_outing"]).reset_index(drop=True)
    group_keys = ["game_pk", "pitcher"]

    # 1. Rolling velocity standard deviation (velocity volatility)
    res["roll5_speed_std"] = (
        res.groupby(group_keys)["release_speed"]
        .transform(lambda s: s.rolling(5, min_periods=2).std())
        .round(3)
    )
    res["roll10_speed_std"] = (
        res.groupby(group_keys)["release_speed"]
        .transform(lambda s: s.rolling(10, min_periods=3).std())
        .round(3)
    )

    # 2. Rolling release position 3D drift (Release X & Release Z standard deviation)
    res["roll5_rel_x_std"] = (
        res.groupby(group_keys)["release_pos_x"]
        .transform(lambda s: s.rolling(5, min_periods=2).std())
        .round(4)
    )
    res["roll5_rel_z_std"] = (
        res.groupby(group_keys)["release_pos_z"]
        .transform(lambda s: s.rolling(5, min_periods=2).std())
        .round(4)
    )
    res["roll5_release_dist_drift"] = np.sqrt(res["roll5_rel_x_std"]**2 + res["roll5_rel_z_std"]**2).round(4)

    res["roll10_rel_x_std"] = (
        res.groupby(group_keys)["release_pos_x"]
        .transform(lambda s: s.rolling(10, min_periods=3).std())
        .round(4)
    )
    res["roll10_rel_z_std"] = (
        res.groupby(group_keys)["release_pos_z"]
        .transform(lambda s: s.rolling(10, min_periods=3).std())
        .round(4)
    )
    res["roll10_release_dist_drift"] = np.sqrt(res["roll10_rel_x_std"]**2 + res["roll10_rel_z_std"]**2).round(4)

    # 3. Circular spin axis shift
    # Use circular dispersion so 359 and 1 degrees are close.
    res["roll5_spin_axis_shift"] = (
        res.groupby(group_keys)["spin_axis"]
        .transform(lambda s: _rolling_circular_std(s, 5, 2))
        .round(3)
    )
    res["roll10_spin_axis_shift"] = (
        res.groupby(group_keys)["spin_axis"]
        .transform(lambda s: _rolling_circular_std(s, 10, 3))
        .round(3)
    )

    # 4. Outing pitch tempo estimation
    if "pitch_interval_sec" not in res.columns:
        res["pitch_interval_sec"] = np.nan

    return res
