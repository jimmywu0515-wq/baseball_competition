"""
Data Qualification & Outing Sequencing Module (§3 & Critique Fixes)
Handles:
1. Both Home and Away starting pitcher identification.
2. Chronological sorting by (game_pk, at_bat_number, pitch_number).
3. Outing-level pitch and PA counters.
4. Right-censored follow-up marking.
"""
import logging
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class QualifyFilter:
    """
    Filters raw pitch data according to strict starter qualification rules
    and enforces correct chronological outing indexing.
    """
    def __init__(self, 
                 min_pitches_per_game: int = 50,
                 min_starts_per_season: int = None,
                 min_starts_for_cohort: int = 10,
                 cohort_eligibility_end: str = "2024-12-31",
                 max_missing_mechanics_pct: float = 0.05,
                 top_n_pitch_types: int = 3,
                 horizon_pas: int = 3):
        self.min_pitches_per_game = min_pitches_per_game
        # The legacy argument remains as an alias for callers/tests, but the
        # count is now applied once using only pre-test information.
        self.min_starts_for_cohort = int(
            min_starts_per_season if min_starts_per_season is not None else min_starts_for_cohort
        )
        self.cohort_eligibility_end = pd.Timestamp(cohort_eligibility_end)
        self.max_missing_mechanics_pct = max_missing_mechanics_pct
        self.top_n_pitch_types = top_n_pitch_types
        self.horizon_pas = horizon_pas
        self.last_excluded_pitchers = pd.DataFrame()
        self.last_excluded_outings = pd.DataFrame()

    def filter_qualified_games(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executes qualify screening and produces:
        1. stg_qualified_pitches (chronologically sorted with outing counters)
        2. dim_pitchers (metadata & qualified starts count)
        3. dim_games (qualified outings metadata)
        """
        if raw_df.empty:
            logger.warning("Empty raw DataFrame passed to QualifyFilter.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        df = raw_df.copy()

        # 1. Standardize types and column existence
        if "inning_topbot" not in df.columns:
            df["inning_topbot"] = "Top"
        if "at_bat_number" not in df.columns:
            df["at_bat_number"] = df.groupby(["game_pk", "inning"]).cumcount() // 4 + 1
        if "pitch_number" not in df.columns:
            df["pitch_number"] = df.groupby(["game_pk", "at_bat_number"]).cumcount() + 1

        # 2. Correct Chronological Sorting by Game, Plate Appearance, and Pitch in PA
        df = df.sort_values(
            by=["game_pk", "inning", "at_bat_number", "pitch_number"], 
            ascending=[True, True, True, True]
        ).reset_index(drop=True)

        # 3. Proper Starter Identification (Both Home and Away starters!)
        # Pitcher who starts Inning 1 Top = Away Starter
        # Pitcher who starts Inning 1 Bot = Home Starter
        starter_keys = set()
        for g_pk, g_group in df.groupby("game_pk"):
            # Away starter
            top1 = g_group[(g_group["inning"] == 1) & (g_group["inning_topbot"].str.lower().str.startswith("top"))]
            if not top1.empty:
                away_sp = top1.iloc[0]["pitcher"]
                starter_keys.add((g_pk, away_sp))
            # Home starter
            bot1 = g_group[(g_group["inning"] == 1) & (g_group["inning_topbot"].str.lower().str.startswith("bot"))]
            if not bot1.empty:
                home_sp = bot1.iloc[0]["pitcher"]
                starter_keys.add((g_pk, home_sp))
            # No fallback for outings that begin after inning one: those are
            # relief appearances, not starters, in pitcher-scoped Statcast data.

        df["is_starter"] = pd.MultiIndex.from_frame(df[["game_pk", "pitcher"]]).isin(starter_keys)
        df_sp = df[df["is_starter"]].copy()

        # 4. Compute accurate Outing-Level Sequence Counters per (game_pk, pitcher)
        df_sp["pitch_number_in_outing"] = df_sp.groupby(["game_pk", "pitcher"]).cumcount() + 1
        df_sp["pitch_number_in_game"] = df_sp["pitch_number_in_outing"] # Alias for consistency
        df_sp["pa_number_in_outing"] = df_sp.groupby(["game_pk", "pitcher"])["at_bat_number"].transform(
            lambda s: pd.factorize(s)[0] + 1
        )

        # 5. Outing-level Checks: Pitch count >= 50 and Missing Data < 5%
        outing_stats = df_sp.groupby(["pitcher", "pitcher_name", "game_pk", "game_date"]).agg(
            total_pitches=("pitch_number_in_outing", "max"),
            total_pas=("pa_number_in_outing", "max"),
            missing_rel_x=("release_pos_x", lambda s: s.isna().mean()),
            missing_spin=("spin_axis", lambda s: s.isna().mean())
        ).reset_index()
        for provenance_col in ("actual_data_source", "requested_data_mode", "dataset_split"):
            if provenance_col in df_sp.columns:
                provenance = (
                    df_sp.groupby(["pitcher", "pitcher_name", "game_pk", "game_date"])[provenance_col]
                    .first().reset_index()
                )
                outing_stats = outing_stats.merge(
                    provenance, on=["pitcher", "pitcher_name", "game_pk", "game_date"], how="left"
                )

        outing_stats["outing_qualification_reason"] = "INCLUDED"
        outing_stats.loc[
            outing_stats["total_pitches"] < self.min_pitches_per_game,
            "outing_qualification_reason",
        ] = "RETROSPECTIVE_OUTING_UNDER_MIN_PITCHES"
        outing_stats.loc[
            outing_stats["missing_rel_x"] > self.max_missing_mechanics_pct,
            "outing_qualification_reason",
        ] = "EXCESS_MISSING_RELEASE_POSITION"
        outing_stats.loc[
            outing_stats["missing_spin"] > self.max_missing_mechanics_pct,
            "outing_qualification_reason",
        ] = "EXCESS_MISSING_SPIN_AXIS"
        qualified_outings_mask = outing_stats["outing_qualification_reason"].eq("INCLUDED")
        qualified_outings_df = outing_stats[qualified_outings_mask].copy()

        # 6. Freeze cohort eligibility using information available before 2025.
        # Future 2025 starts never decide whether a pitcher belongs to the cohort.
        qualified_outings_df["season"] = pd.to_datetime(
            qualified_outings_df["game_date"], errors="coerce"
        ).dt.year
        pretest = qualified_outings_df[
            pd.to_datetime(qualified_outings_df["game_date"]) <= self.cohort_eligibility_end
        ]
        pretest_counts = pretest.groupby("pitcher").size().rename("pretest_qualified_starts")
        pitcher_names = outing_stats[["pitcher", "pitcher_name"]].drop_duplicates("pitcher")
        cohort_audit = pitcher_names.merge(pretest_counts, on="pitcher", how="left").fillna(
            {"pretest_qualified_starts": 0}
        )
        cohort_audit["pretest_qualified_starts"] = cohort_audit["pretest_qualified_starts"].astype(int)
        cohort_audit["cohort_eligible"] = (
            cohort_audit["pretest_qualified_starts"] >= self.min_starts_for_cohort
        )
        cohort_audit["cohort_eligibility_cutoff"] = self.cohort_eligibility_end
        cohort_audit["exclusion_reason"] = np.where(
            cohort_audit["cohort_eligible"],
            "INCLUDED",
            "INSUFFICIENT_PRETEST_QUALIFIED_STARTS",
        )
        qualified_pitchers = set(cohort_audit.loc[cohort_audit["cohort_eligible"], "pitcher"])
        final_outings = qualified_outings_df[
            qualified_outings_df["pitcher"].isin(qualified_pitchers)
        ].copy()
        excluded_outings = outing_stats[~qualified_outings_mask].copy()
        cohort_excluded_outings = qualified_outings_df[
            ~qualified_outings_df["pitcher"].isin(qualified_pitchers)
        ].copy()
        cohort_excluded_outings["outing_qualification_reason"] = "COHORT_INELIGIBLE_PRETEST"
        self.last_excluded_outings = pd.concat(
            [excluded_outings, cohort_excluded_outings], ignore_index=True
        )
        self.last_excluded_pitchers = cohort_audit[
            ~cohort_audit["cohort_eligible"]
        ].reset_index(drop=True)
        final_outing_keys = set(zip(final_outings["game_pk"], final_outings["pitcher"]))

        # Filter pitch-level dataset
        qualified_pitches = df_sp[
            pd.MultiIndex.from_frame(df_sp[["game_pk", "pitcher"]]).isin(final_outing_keys)
        ].copy()

        # 7. Mark Right-Censored Follow-up
        # If pitcher is removed with fewer than horizon_pas remaining, record incomplete follow-up
        max_pa_dict = final_outings.set_index(["game_pk", "pitcher"])["total_pas"].to_dict()
        qualified_pitches["outing_total_pas"] = [
            max_pa_dict.get((game_pk, pitcher), np.nan)
            for game_pk, pitcher in zip(qualified_pitches["game_pk"], qualified_pitches["pitcher"])
        ]
        qualified_pitches["is_censored"] = (qualified_pitches["outing_total_pas"] - qualified_pitches["pa_number_in_outing"]) < self.horizon_pas

        # 8. Identify Primary Pitch Types
        pitch_counts = qualified_pitches.groupby(["pitcher", "pitch_type"]).size().reset_index(name="count")
        pitch_counts = pitch_counts.sort_values(["pitcher", "count"], ascending=[True, False])
        top_pitches = pitch_counts.groupby("pitcher").head(self.top_n_pitch_types)
        
        primary_pitch_map: Dict[int, List[str]] = {}
        for p_id, grp in top_pitches.groupby("pitcher"):
            primary_pitch_map[p_id] = grp["pitch_type"].tolist()

        primary_pairs = {
            (pitcher, pitch_type)
            for pitcher, pitch_types in primary_pitch_map.items()
            for pitch_type in pitch_types
        }
        qualified_pitches["is_primary_pitch_type"] = pd.MultiIndex.from_frame(
            qualified_pitches[["pitcher", "pitch_type"]]
        ).isin(primary_pairs)

        dim_pitchers = qualified_pitches[["pitcher", "pitcher_name", "p_throws"]].drop_duplicates().copy()
        dim_pitchers["primary_pitch_types"] = dim_pitchers["pitcher"].map(lambda pid: ",".join(primary_pitch_map.get(pid, [])))
        total_start_counts = final_outings.groupby("pitcher").size().to_dict()
        dim_pitchers["qualified_starts"] = dim_pitchers["pitcher"].map(total_start_counts)
        dim_pitchers = dim_pitchers.merge(
            cohort_audit[["pitcher", "pretest_qualified_starts", "cohort_eligible",
                          "cohort_eligibility_cutoff"]],
            on="pitcher", how="left",
        )
        if "actual_data_source" in qualified_pitches:
            source_map = qualified_pitches.groupby("pitcher")["actual_data_source"].first().to_dict()
            dim_pitchers["actual_data_source"] = dim_pitchers["pitcher"].map(source_map)
        if "requested_data_mode" in qualified_pitches:
            mode_map = qualified_pitches.groupby("pitcher")["requested_data_mode"].first().to_dict()
            dim_pitchers["requested_data_mode"] = dim_pitchers["pitcher"].map(mode_map)

        dim_games = final_outings.copy()
        logger.info(f"Qualified Dataset: {len(qualified_pitches)} pitches across {len(final_outings)} starting outings ({len(dim_pitchers)} pitchers).")

        return qualified_pitches, dim_pitchers, dim_games
