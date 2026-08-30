"""
Ground Truth Collapse Label Builder (§5 Step 6 & §4.3)
STRICT ISOLATION: This module generates verification ground-truth labels based on
rolling 3-PA windows. MUST NEVER BE FED AS INPUT TO ANOMALY DETECTION.
"""
import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard MLB wOBA event weights
WOBA_WEIGHTS = {
    "walk": 0.69,
    "hit_by_pitch": 0.72,
    "single": 0.88,
    "double": 1.25,
    "triple": 1.58,
    "home_run": 2.03,
    "strikeout": 0.0,
    "field_out": 0.0,
    "force_out": 0.0,
    "grounded_into_double_play": 0.0,
    "sac_fly": 0.0,
    "sac_bunt": 0.0
}

class CollapseLabelBuilder:
    """
    Constructs rolling 3-PA collapse outcome labels.
    """
    def __init__(self, 
                 window_pa_size: int = 3, 
                 blended_xwoba_threshold: float = 0.450,
                 min_barrels_in_window: int = 2,
                 min_bb_hbp_in_window: int = 2):
        self.window_pa_size = window_pa_size
        self.blended_xwoba_threshold = blended_xwoba_threshold
        self.min_barrels_in_window = min_barrels_in_window
        self.min_bb_hbp_in_window = min_bb_hbp_in_window

    def build_labels_for_game(self, game_pitches_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes a single game's pitches and labels collapse occurrences.
        Returns:
            pitch_labels_df: Label per pitch
            collapse_events_summary: Summary of all collapse windows in the game
        """
        df = game_pitches_df.copy()
        if "pitch_number_in_game" not in df.columns:
            if "pitch_number" in df.columns:
                df["pitch_number_in_game"] = df["pitch_number"]
            else:
                df["pitch_number_in_game"] = df.groupby("game_pk").cumcount() + 1

        df = df.sort_values("pitch_number_in_game").reset_index(drop=True)

        # 1. Identify PA Endings and Outcomes
        pa_summary = []
        for ab_num, pa_pitches in df.groupby("at_bat_number"):
            last_pitch = pa_pitches.iloc[-1]
            event = str(last_pitch.get("events", "")).lower()
            
            # xwOBA calculation
            xwoba = last_pitch.get("estimated_woba_using_speedangle", np.nan)
            if np.isnan(xwoba) or xwoba is None:
                xwoba = WOBA_WEIGHTS.get(event, 0.0)

            # Barrel check
            launch_spd = last_pitch.get("launch_speed", np.nan)
            launch_ang = last_pitch.get("launch_angle", np.nan)
            is_barrel = False
            if not np.isnan(launch_spd) and not np.isnan(launch_ang):
                if launch_spd >= 98.0 and (26.0 - (launch_spd - 98.0)) <= launch_ang <= (30.0 + (launch_spd - 98.0)):
                    is_barrel = True

            is_bb_hbp = event in ["walk", "hit_by_pitch"]
            is_hit = event in ["single", "double", "triple", "home_run"]
            runs = 1 if event == "home_run" else (0.5 if is_hit else 0.0)

            pa_summary.append({
                "at_bat_number": ab_num,
                "first_pitch_num": pa_pitches["pitch_number_in_game"].min(),
                "last_pitch_num": pa_pitches["pitch_number_in_game"].max(),
                "inning": last_pitch["inning"],
                "event": event,
                "blended_xwoba": float(xwoba),
                "is_barrel": is_barrel,
                "is_bb_hbp": is_bb_hbp,
                "is_hit": is_hit,
                "runs": runs
            })

        pa_df = pd.DataFrame(pa_summary)
        if pa_df.empty:
            df["is_collapse_event"] = False
            df["collapse_reason"] = "None"
            return df, pd.DataFrame()

        # 2. Rolling 3-PA Window Calculation
        pa_df["rolling_3pa_xwoba"] = pa_df["blended_xwoba"].rolling(self.window_pa_size, min_periods=2).mean().round(3)
        pa_df["rolling_3pa_barrels"] = pa_df["is_barrel"].rolling(self.window_pa_size, min_periods=2).sum()
        pa_df["rolling_3pa_bb_hbp"] = pa_df["is_bb_hbp"].rolling(self.window_pa_size, min_periods=2).sum()
        pa_df["rolling_3pa_hits"] = pa_df["is_hit"].rolling(self.window_pa_size, min_periods=2).sum()

        def check_collapse(row):
            reasons = []
            if row["rolling_3pa_xwoba"] >= self.blended_xwoba_threshold:
                reasons.append(f"High xwOBA ({row['rolling_3pa_xwoba']:.3f})")
            if row["rolling_3pa_barrels"] >= self.min_barrels_in_window:
                reasons.append(f"Multiple Barrels ({int(row['rolling_3pa_barrels'])})")
            if row["rolling_3pa_bb_hbp"] >= self.min_bb_hbp_in_window:
                reasons.append(f"Multiple BB/HBP ({int(row['rolling_3pa_bb_hbp'])})")

            if reasons:
                return True, " & ".join(reasons)
            return False, "Normal"

        pa_df["is_collapse"], pa_df["collapse_reason"] = zip(*pa_df.apply(check_collapse, axis=1))

        # 3. Map back to pitch-level DataFrame
        pitch_is_collapse = []
        pitch_collapse_reason = []
        pitch_rolling_xwoba = []

        pa_collapse_dict = dict(zip(pa_df["at_bat_number"], pa_df["is_collapse"]))
        pa_reason_dict = dict(zip(pa_df["at_bat_number"], pa_df["collapse_reason"]))
        pa_xwoba_dict = dict(zip(pa_df["at_bat_number"], pa_df["rolling_3pa_xwoba"]))

        for _, row in df.iterrows():
            ab = row["at_bat_number"]
            pitch_is_collapse.append(pa_collapse_dict.get(ab, False))
            pitch_collapse_reason.append(pa_reason_dict.get(ab, "Normal"))
            pitch_rolling_xwoba.append(pa_xwoba_dict.get(ab, 0.0))

        df["is_collapse_event"] = pitch_is_collapse
        df["collapse_reason"] = pitch_collapse_reason
        df["window_blended_xwoba"] = pitch_rolling_xwoba

        collapse_events = pa_df[pa_df["is_collapse"]].copy()
        if not collapse_events.empty:
            collapse_events["game_pk"] = df["game_pk"].iloc[0]
            collapse_events["game_date"] = df["game_date"].iloc[0]
            collapse_events["pitcher"] = df["pitcher"].iloc[0]
            collapse_events["pitcher_name"] = df.get("pitcher_name", pd.Series([""]*len(df))).iloc[0]

        return df, collapse_events
