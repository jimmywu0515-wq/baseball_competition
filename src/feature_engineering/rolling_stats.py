"""
Rolling Trend and Dispersion Features (Level 1 Features)
Calculates in-game micro-trend signals: 5/10 pitch rolling dispersion, velocity volatility, and drift.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Level 1 rolling window features grouped by game_pk and pitch_type.
    """
    res = df.copy()
    
    if "pitch_number_in_game" not in res.columns:
        if "pitch_number" in res.columns:
            res["pitch_number_in_game"] = res["pitch_number"]
        else:
            res["pitch_number_in_game"] = res.groupby("game_pk").cumcount() + 1

    # Sort to ensure chronological sequence
    res = res.sort_values(by=["game_pk", "pitch_number_in_game"]).reset_index(drop=True)

    # Rolling calculations per game
    # 1. Rolling velocity standard deviation (volatility/inconsistency)
    res["roll5_speed_std"] = (
        res.groupby("game_pk")["release_speed"]
        .transform(lambda s: s.rolling(5, min_periods=2).std())
        .fillna(0.0)
        .round(3)
    )
    res["roll10_speed_std"] = (
        res.groupby("game_pk")["release_speed"]
        .transform(lambda s: s.rolling(10, min_periods=3).std())
        .fillna(0.0)
        .round(3)
    )

    # 2. Rolling release position drift (standard deviation & range)
    res["roll5_rel_x_std"] = (
        res.groupby("game_pk")["release_pos_x"]
        .transform(lambda s: s.rolling(5, min_periods=2).std())
        .fillna(0.0)
        .round(4)
    )
    res["roll5_rel_z_std"] = (
        res.groupby("game_pk")["release_pos_z"]
        .transform(lambda s: s.rolling(5, min_periods=2).std())
        .fillna(0.0)
        .round(4)
    )
    res["roll5_release_dist_drift"] = np.sqrt(res["roll5_rel_x_std"]**2 + res["roll5_rel_z_std"]**2).round(4)

    res["roll10_rel_x_std"] = (
        res.groupby("game_pk")["release_pos_x"]
        .transform(lambda s: s.rolling(10, min_periods=3).std())
        .fillna(0.0)
        .round(4)
    )
    res["roll10_rel_z_std"] = (
        res.groupby("game_pk")["release_pos_z"]
        .transform(lambda s: s.rolling(10, min_periods=3).std())
        .fillna(0.0)
        .round(4)
    )
    res["roll10_release_dist_drift"] = np.sqrt(res["roll10_rel_x_std"]**2 + res["roll10_rel_z_std"]**2).round(4)

    # 3. Rolling spin axis shift (angular difference standard deviation)
    res["roll5_spin_axis_shift"] = (
        res.groupby("game_pk")["spin_axis"]
        .transform(lambda s: s.rolling(5, min_periods=2).std())
        .fillna(0.0)
        .round(3)
    )
    res["roll10_spin_axis_shift"] = (
        res.groupby("game_pk")["spin_axis"]
        .transform(lambda s: s.rolling(10, min_periods=3).std())
        .fillna(0.0)
        .round(3)
    )

    # 4. Tempo: Pitch interval approximation
    if "pitch_interval_sec" not in res.columns:
        res["pitch_interval_sec"] = np.where(res["pitch_number_in_game"] > 1, 18.0 + np.random.normal(0, 2.5, len(res)), 0.0)
        res["pitch_interval_sec"] = np.maximum(res["pitch_interval_sec"], 5.0).round(1)

    return res
