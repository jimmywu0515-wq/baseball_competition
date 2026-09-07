"""
Ground Truth Collapse Label & Episode Builder (§5 Step 6 & Critique Fixes)
STRICT ISOLATION & EPISODE AGGREGATION:
1. Merges overlapping rolling 3-PA collapse windows into distinct Collapse Episodes.
2. Defines a single, unified prediction target:
   "Does a new collapse episode onset occur within the next H (15) pitches / 3 PAs?"
3. Handles right-censoring when pitchers are pulled before the horizon finishes.
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
    Constructs distinct collapse episodes and unified prediction target labels.
    """
    def __init__(self, 
                 window_pa_size: int = 3, 
                 blended_xwoba_threshold: float = 0.450,
                 min_barrels_in_window: int = 2,
                 min_bb_hbp_in_window: int = 2,
                 horizon_pitches: int = 15,
                 horizon_pas: int = 3):
        self.window_pa_size = window_pa_size
        self.blended_xwoba_threshold = blended_xwoba_threshold
        self.min_barrels_in_window = min_barrels_in_window
        self.min_bb_hbp_in_window = min_bb_hbp_in_window
        self.horizon_pitches = horizon_pitches
        self.horizon_pas = horizon_pas

    def build_labels_for_outing(self, outing_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes a single pitcher outing and produces:
        1. Pitch-level DataFrame with unified target `y_true_onset_in_horizon`
        2. Episode-level summary DataFrame of distinct collapse episodes
        """
        df = outing_df.copy().sort_values("pitch_number_in_outing").reset_index(drop=True)

        if "pa_number_in_outing" not in df.columns:
            df["pa_number_in_outing"] = pd.factorize(df["at_bat_number"])[0] + 1

        # 1. PA-level outcomes aggregation
        pa_summary = []
        for pa_num, pa_group in df.groupby("pa_number_in_outing"):
            last_pitch = pa_group.iloc[-1]
            event = str(last_pitch.get("events", "")).lower()

            xwoba = last_pitch.get("estimated_woba_using_speedangle", np.nan)
            if np.isnan(xwoba) or xwoba is None:
                xwoba = WOBA_WEIGHTS.get(event, 0.0)

            launch_spd = last_pitch.get("launch_speed", np.nan)
            launch_ang = last_pitch.get("launch_angle", np.nan)
            is_barrel = False
            if not np.isnan(launch_spd) and not np.isnan(launch_ang):
                if launch_spd >= 98.0 and (26.0 - (launch_spd - 98.0)) <= launch_ang <= (30.0 + (launch_spd - 98.0)):
                    is_barrel = True

            is_bb_hbp = event in ["walk", "hit_by_pitch"]
            is_hit = event in ["single", "double", "triple", "home_run"]

            pa_summary.append({
                "pa_number_in_outing": pa_num,
                "first_pitch_in_outing": pa_group["pitch_number_in_outing"].min(),
                "last_pitch_in_outing": pa_group["pitch_number_in_outing"].max(),
                "inning": last_pitch["inning"],
                "event": event,
                "blended_xwoba": float(xwoba),
                "is_barrel": is_barrel,
                "is_bb_hbp": is_bb_hbp,
                "is_hit": is_hit
            })

        pa_df = pd.DataFrame(pa_summary)
        if pa_df.empty:
            df["is_collapse_event"] = False
            df["y_true_onset_in_horizon"] = False
            return df, pd.DataFrame()

        # 2. Rolling Window Calculation
        pa_df["rolling_xwoba"] = pa_df["blended_xwoba"].rolling(self.window_pa_size, min_periods=2).mean().round(3)
        pa_df["rolling_barrels"] = pa_df["is_barrel"].rolling(self.window_pa_size, min_periods=2).sum()
        pa_df["rolling_bb_hbp"] = pa_df["is_bb_hbp"].rolling(self.window_pa_size, min_periods=2).sum()

        is_bad_window = (
            (pa_df["rolling_xwoba"] >= self.blended_xwoba_threshold) |
            (pa_df["rolling_barrels"] >= self.min_barrels_in_window) |
            (pa_df["rolling_bb_hbp"] >= self.min_bb_hbp_in_window)
        )
        pa_df["is_bad_window"] = is_bad_window

        # 3. Merge contiguous bad windows into distinct Collapse Episodes
        episodes = []
        in_episode = False
        curr_ep = None
        episode_counter = 1

        for _, row in pa_df.iterrows():
            if row["is_bad_window"]:
                if not in_episode:
                    in_episode = True
                    curr_ep = {
                        "episode_id": episode_counter,
                        "onset_pa": row["pa_number_in_outing"],
                        "onset_pitch": row["first_pitch_in_outing"],
                        "end_pa": row["pa_number_in_outing"],
                        "end_pitch": row["last_pitch_in_outing"],
                        "inning": row["inning"]
                    }
                    episode_counter += 1
                else:
                    curr_ep["end_pa"] = row["pa_number_in_outing"]
                    curr_ep["end_pitch"] = row["last_pitch_in_outing"]
            else:
                if in_episode:
                    episodes.append(curr_ep)
                    in_episode = False
                    curr_ep = None

        if in_episode and curr_ep is not None:
            episodes.append(curr_ep)

        episodes_df = pd.DataFrame(episodes)
        if not episodes_df.empty:
            episodes_df["game_pk"] = df["game_pk"].iloc[0]
            episodes_df["pitcher"] = df["pitcher"].iloc[0]
            episodes_df["pitcher_name"] = df.get("pitcher_name", pd.Series([""]*len(df))).iloc[0]
            onset_pitches = episodes_df["onset_pitch"].values
        else:
            onset_pitches = np.array([])

        # 4. Construct Unified Prediction Target on Pitch Level:
        # At pitch t: does a collapse episode onset occur within (t, t + horizon_pitches]?
        max_pitch_in_outing = df["pitch_number_in_outing"].max()
        y_onset_in_horizon = []
        is_collapse_active = []
        is_censored = []

        for _, row in df.iterrows():
            p_num = row["pitch_number_in_outing"]
            
            # Check if any episode onset occurs in (p_num, p_num + self.horizon_pitches]
            has_future_onset = any((onset > p_num) and (onset <= p_num + self.horizon_pitches) for onset in onset_pitches)
            y_onset_in_horizon.append(has_future_onset)

            # Check if pitch is inside an active collapse episode
            is_active = any((ep["onset_pitch"] <= p_num <= ep["end_pitch"]) for ep in episodes) if episodes else False
            is_collapse_active.append(is_active)

            # Check censoring: if pulled before horizon completed without collapse
            remaining_pitches = max_pitch_in_outing - p_num
            censored = (remaining_pitches < self.horizon_pitches) and not has_future_onset and not is_active
            is_censored.append(censored)

        df["y_true_onset_in_horizon"] = y_onset_in_horizon
        df["future_collapse_in_horizon"] = y_onset_in_horizon # Alias
        df["is_collapse_event"] = is_collapse_active
        df["is_censored_followup"] = is_censored
        df["collapse_episodes_count_in_game"] = len(episodes)

        return df, episodes_df

    # Backward compatibility alias
    def build_labels_for_game(self, game_pitches_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.build_labels_for_outing(game_pitches_df)
