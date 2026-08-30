"""
Data Qualification Module (§3 Qualify Standards)
Applies role filters, single-game thresholds, season consistency, and pitch-type breakdown.
"""
import logging
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class QualifyFilter:
    """
    Filters raw pitch data according to the research proposal specifications:
    - Starting Pitcher only
    - Single-game pitch count >= 50
    - Missing mechanics rate < 5% per game
    - Minimum qualified starts per pitcher (e.g. >= 10-15 starts)
    - Identify top N primary pitch types per pitcher
    """
    def __init__(self, 
                 min_pitches_per_game: int = 50,
                 min_starts_per_season: int = 10,
                 max_missing_mechanics_pct: float = 0.05,
                 top_n_pitch_types: int = 3):
        self.min_pitches_per_game = min_pitches_per_game
        self.min_starts_per_season = min_starts_per_season
        self.max_missing_mechanics_pct = max_missing_mechanics_pct
        self.top_n_pitch_types = top_n_pitch_types

    def filter_qualified_games(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executes qualify screening and produces:
        1. stg_qualified_pitches (filtered pitches)
        2. dim_pitchers (qualified pitcher metadata & primary pitch types)
        3. dim_games (qualified game metadata)
        """
        if raw_df.empty:
            logger.warning("Empty raw DataFrame passed to QualifyFilter.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        df = raw_df.copy()
        
        # 1. Identify Starting Pitchers (Pitcher of the first pitch of the game in Inning 1)
        first_pitches = df.sort_values(by=["game_pk", "inning", "pitch_number"]).groupby("game_pk").first().reset_index()
        starter_dict = dict(zip(first_pitches["game_pk"], first_pitches["pitcher"]))
        
        # Filter for starter pitches only
        df["is_starter"] = df.apply(lambda row: starter_dict.get(row["game_pk"]) == row["pitcher"], axis=1)
        df_sp = df[df["is_starter"]].copy()
        
        # 2. Single-game checks: Pitch count >= 50 and Missing Data < 5%
        game_stats = df_sp.groupby(["pitcher", "pitcher_name", "game_pk", "game_date"]).agg(
            total_pitches=("pitch_number", "count"),
            missing_rel_x=("release_pos_x", lambda s: s.isna().mean()),
            missing_spin_axis=("spin_axis", lambda s: s.isna().mean())
        ).reset_index()

        qualified_games_mask = (
            (game_stats["total_pitches"] >= self.min_pitches_per_game) &
            (game_stats["missing_rel_x"] <= self.max_missing_mechanics_pct) &
            (game_stats["missing_spin_axis"] <= self.max_missing_mechanics_pct)
        )
        qualified_games_df = game_stats[qualified_games_mask].copy()
        logger.info(f"Qualified {len(qualified_games_df)} of {len(game_stats)} starter games (pitch count >= {self.min_pitches_per_game}, missing < 5%).")

        # 3. Season-level check: Pitcher starts count >= min_starts_per_season
        pitcher_starts = qualified_games_df.groupby(["pitcher", "pitcher_name"]).agg(
            qualified_starts=("game_pk", "count")
        ).reset_index()

        qualified_pitchers = pitcher_starts[pitcher_starts["qualified_starts"] >= self.min_starts_per_season]
        qualified_pitcher_ids = set(qualified_pitchers["pitcher"])
        logger.info(f"Qualified {len(qualified_pitchers)} pitchers with >= {self.min_starts_per_season} qualified starts.")

        final_game_pks = set(qualified_games_df[qualified_games_df["pitcher"].isin(qualified_pitcher_ids)]["game_pk"])

        # Filter pitch-level dataset
        qualified_pitches = df_sp[
            (df_sp["pitcher"].isin(qualified_pitcher_ids)) &
            (df_sp["game_pk"].isin(final_game_pks))
        ].copy()

        # Sort pitches in chronological order
        qualified_pitches = qualified_pitches.sort_values(
            by=["pitcher", "game_date", "game_pk", "pitch_number"]
        ).reset_index(drop=True)

        # 4. Identify Primary Pitch Types for each pitcher
        pitch_counts = qualified_pitches.groupby(["pitcher", "pitch_type"]).size().reset_index(name="count")
        pitch_counts = pitch_counts.sort_values(["pitcher", "count"], ascending=[True, False])
        top_pitches = pitch_counts.groupby("pitcher").head(self.top_n_pitch_types)
        
        primary_pitch_map: Dict[int, List[str]] = {}
        for p_id, grp in top_pitches.groupby("pitcher"):
            primary_pitch_map[p_id] = grp["pitch_type"].tolist()

        qualified_pitches["is_primary_pitch_type"] = qualified_pitches.apply(
            lambda r: r["pitch_type"] in primary_pitch_map.get(r["pitcher"], []), axis=1
        )
        
        # Ensure pitch_number_in_game is continuous 1..N
        qualified_pitches["pitch_number_in_game"] = (
            qualified_pitches.groupby("game_pk").cumcount() + 1
        )

        # Build dimension tables
        dim_pitchers = qualified_pitchers.copy()
        dim_pitchers["primary_pitch_types"] = dim_pitchers["pitcher"].map(lambda pid: ",".join(primary_pitch_map.get(pid, [])))
        
        dim_games = qualified_games_df[qualified_games_df["game_pk"].isin(final_game_pks)].copy()

        logger.info(f"Final Qualified Dataset: {len(qualified_pitches)} pitches across {len(dim_games)} games.")
        return qualified_pitches, dim_pitchers, dim_games
