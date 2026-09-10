"""
Statcast Data Ingestion Module
Supports:
1. Real MLB Statcast Ingest via pybaseball with Parquet caching.
2. Explicit Simulation Benchmark Generator (for synthetic control testing).
"""
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

try:
    import pybaseball
    from pybaseball import statcast, statcast_pitcher, playerid_lookup
    pybaseball.cache.enable()
except ImportError:
    pybaseball = None

logger = logging.getLogger(__name__)

# Documented MLB Pitcher Cohort (2023-2024 Seasons)
REAL_PITCHER_COHORT = {
    669203: {"name": "Corbin Burnes", "throws": "R", "primary_fb": "FC"},
    554430: {"name": "Zack Wheeler", "throws": "R", "primary_fb": "FF"},
    592332: {"name": "Kevin Gausman", "throws": "R", "primary_fb": "FF"},
    605400: {"name": "Aaron Nola", "throws": "R", "primary_fb": "FF"},
    657277: {"name": "Logan Webb", "throws": "R", "primary_fb": "SI"},
    543037: {"name": "Gerrit Cole", "throws": "R", "primary_fb": "FF"},
    519242: {"name": "Chris Sale", "throws": "L", "primary_fb": "FF"},
    664285: {"name": "Framber Valdez", "throws": "L", "primary_fb": "SI"}
}

class StatcastLoader:
    """
    Handles fetching and caching MLB Statcast pitch-by-pitch data.
    """
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            self.cache_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_real_pitchers_statcast(self, 
                                     pitcher_ids: Optional[List[int]] = None, 
                                     start_dt: str = "2023-03-30",
                                     end_dt: str = "2024-09-30",
                                     force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetches real MLB Statcast data for specified pitchers across date ranges.
        Uses local Parquet caching per pitcher.
        """
        if pitcher_ids is None:
            pitcher_ids = list(REAL_PITCHER_COHORT.keys())

        all_dfs = []
        for pid in pitcher_ids:
            p_name = REAL_PITCHER_COHORT.get(pid, {}).get("name", str(pid))
            cache_file = self.cache_dir / f"real_statcast_{pid}_{start_dt}_{end_dt}.parquet"
            
            if not force_refresh and cache_file.exists():
                logger.info(f"Loading cached real Statcast data for {p_name} ({pid}) from {cache_file.name}")
                df_p = pd.read_parquet(cache_file)
                all_dfs.append(df_p)
                continue

            logger.info(f"Fetching real Statcast data for {p_name} ({pid}) from {start_dt} to {end_dt}...")
            try:
                df_p = pybaseball.statcast_pitcher(start_dt, end_dt, player_id=pid)
                if df_p is not None and not df_p.empty:
                    # Enrich with pitcher name and throws if missing
                    if "pitcher_name" not in df_p.columns:
                        df_p["pitcher_name"] = p_name
                    if "p_throws" not in df_p.columns:
                        df_p["p_throws"] = REAL_PITCHER_COHORT.get(pid, {}).get("throws", "R")
                    df_p["actual_data_source"] = "mlb_statcast"
                    
                    df_p.to_parquet(cache_file, index=False, engine="pyarrow")
                    logger.info(f"Cached {len(df_p)} pitches for {p_name} to {cache_file.name}")
                    all_dfs.append(df_p)
                else:
                    logger.warning(f"No pitches returned for pitcher {p_name} ({pid}).")
            except Exception as e:
                logger.error(f"Error fetching Statcast data for {p_name}: {e}")

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            logger.info(f"Successfully loaded {len(combined)} real Statcast pitches across {len(all_dfs)} pitchers.")
            return combined
        else:
            logger.warning("No real data fetched. Please check network connection.")
            return pd.DataFrame()

    def generate_simulation_benchmark(self, num_pitchers: int = 5, starts_per_pitcher: int = 15) -> pd.DataFrame:
        """
        SYNTHETIC CONTROL BENCHMARK GENERATOR:
        Generates simulated pitches with programmed degradation strictly for
        testing algorithm sensitivity in controlled synthetic environments.
        Labeled explicitly as a simulation benchmark.
        """
        logger.info(f"[Simulation Benchmark] Generating synthetic pitches for {num_pitchers} pitchers across {starts_per_pitcher} starts...")
        np.random.seed(42)
        
        pitchers_meta = [
            {"id": 543037, "name": "Gerrit Cole (Sim)", "throws": "R", "ff_speed": 97.2, "ff_spin": 2450, "sl_speed": 89.1, "sl_spin": 2600, "ch_speed": 89.0, "ch_spin": 1780, "rel_x": -1.8, "rel_z": 5.9, "ext": 6.7},
            {"id": 669203, "name": "Corbin Burnes (Sim)", "throws": "R", "ff_speed": 95.5, "ff_spin": 2650, "sl_speed": 88.0, "sl_spin": 2800, "ch_speed": 89.5, "ch_spin": 1900, "rel_x": -1.9, "rel_z": 5.7, "ext": 6.5},
            {"id": 554430, "name": "Zack Wheeler (Sim)", "throws": "R", "ff_speed": 96.0, "ff_spin": 2420, "sl_speed": 91.0, "sl_spin": 2450, "ch_speed": 88.5, "ch_spin": 1750, "rel_x": -1.7, "rel_z": 5.8, "ext": 7.1},
            {"id": 592332, "name": "Kevin Gausman (Sim)", "throws": "R", "ff_speed": 94.6, "ff_spin": 2250, "sl_speed": 84.5, "sl_spin": 1800, "ch_speed": 85.2, "ch_spin": 1650, "rel_x": -2.3, "rel_z": 5.8, "ext": 6.3},
            {"id": 605400, "name": "Aaron Nola (Sim)", "throws": "R", "ff_speed": 93.2, "ff_spin": 2320, "sl_speed": 85.5, "sl_spin": 2200, "ch_speed": 86.0, "ch_spin": 1700, "rel_x": -2.0, "rel_z": 5.6, "ext": 6.6}
        ]

        all_pitches = []
        game_pk_counter = 716000

        for p_idx in range(min(num_pitchers, len(pitchers_meta))):
            p = pitchers_meta[p_idx]
            for start_idx in range(starts_per_pitcher):
                game_pk = game_pk_counter
                game_pk_counter += 1
                # Distribute every benchmark across all three documented temporal
                # partitions so fallback runs cannot masquerade as held-out MLB results.
                first_cut = max(1, starts_per_pitcher // 3)
                second_cut = max(first_cut + 1, (2 * starts_per_pitcher) // 3)
                if start_idx < first_cut:
                    period_start = datetime(2023, 4, 1)
                    period_index = start_idx
                elif start_idx < second_cut:
                    period_start = datetime(2024, 4, 1)
                    period_index = start_idx - first_cut
                else:
                    period_start = datetime(2024, 7, 5)
                    period_index = start_idx - second_cut
                game_date = (period_start + timedelta(days=period_index * 7 + p_idx)).strftime("%Y-%m-%d")
                total_pitches = np.random.randint(80, 102)
                
                has_collapse = (start_idx % 4 == 0)
                collapse_trigger_pitch = np.random.randint(62, 75) if has_collapse else 999
                
                current_inning = 1
                outs = 0
                ab_number = 1
                pitch_in_ab = 1
                balls = 0
                strikes = 0
                
                day_rel_x = p["rel_x"] + np.random.normal(0, 0.03)
                day_rel_z = p["rel_z"] + np.random.normal(0, 0.03)
                day_ext = p["ext"] + np.random.normal(0, 0.04)

                for pitch_num in range(1, total_pitches + 1):
                    pt_rand = np.random.rand()
                    if pt_rand < 0.60:
                        pt = "FF"
                        base_spd = p["ff_speed"]
                        base_spn = p["ff_spin"]
                        pfx_x = -0.5 if p["throws"] == "R" else 0.5
                        pfx_z = 1.3
                        spin_axis = 210.0
                    elif pt_rand < 0.90:
                        pt = "SL"
                        base_spd = p["sl_speed"]
                        base_spn = p["sl_spin"]
                        pfx_x = 0.6 if p["throws"] == "R" else -0.6
                        pfx_z = 0.2
                        spin_axis = 135.0
                    else:
                        pt = "CH"
                        base_spd = p["ch_speed"]
                        base_spn = p["ch_spin"]
                        pfx_x = -1.1 if p["throws"] == "R" else 1.1
                        pfx_z = 0.6
                        spin_axis = 240.0

                    fatigue_drift_x = 0.0
                    fatigue_drift_z = 0.0
                    fatigue_drift_axis = 0.0
                    spd_penalty = 0.0
                    xwoba_boost = 0.0

                    if pitch_num >= collapse_trigger_pitch:
                        progress = (pitch_num - collapse_trigger_pitch)
                        fatigue_drift_x = (0.015 * progress) * (-1 if p["throws"] == "R" else 1)
                        fatigue_drift_z = -0.012 * progress
                        fatigue_drift_axis = 1.2 * progress
                        if progress > 12:
                            spd_penalty = -0.15 * (progress - 12)
                        if progress > 8:
                            xwoba_boost = min(0.65, 0.04 * (progress - 8))

                    rel_x = day_rel_x + fatigue_drift_x + np.random.normal(0, 0.04)
                    rel_z = day_rel_z + fatigue_drift_z + np.random.normal(0, 0.04)
                    rel_y = 54.5 - (day_ext + np.random.normal(0, 0.05))
                    rel_ext = day_ext + np.random.normal(0, 0.05)
                    rel_spd = base_spd + spd_penalty + np.random.normal(0, 0.6)
                    spin_rt = base_spn + np.random.normal(0, 35)
                    curr_axis = spin_axis + fatigue_drift_axis + np.random.normal(0, 3.5)
                    
                    plate_x = np.random.normal(0, 0.45) + (0.2 if fatigue_drift_x != 0 else 0)
                    plate_z = np.random.normal(2.5, 0.5) - (0.2 if fatigue_drift_z != 0 else 0)
                    
                    vy0 = -rel_spd * 1.467
                    vx0 = (plate_x - rel_x) / (60.5 / abs(vy0))
                    vz0 = (plate_z - rel_z + 0.5 * 32.174 * (60.5 / abs(vy0))**2) / (60.5 / abs(vy0))
                    ax = pfx_x * 12.0
                    ay = 28.0
                    az = -32.174 + pfx_z * 12.0

                    is_in_play = False
                    event = None
                    desc = "ball"
                    pitch_type_result = "B"
                    est_xwoba = np.nan
                    launch_spd = np.nan
                    launch_ang = np.nan
                    woba_val = np.nan
                    woba_den = np.nan

                    rand_outcome = np.random.rand() + xwoba_boost
                    if rand_outcome > 0.82:
                        is_in_play = True
                        pitch_type_result = "X"
                        desc = "hit_into_play"
                        if rand_outcome > 1.25:
                            event = "home_run"
                            est_xwoba = 1.65
                            launch_spd = 106.5
                            launch_ang = 28.0
                            woba_val = 2.0
                            woba_den = 1.0
                        elif rand_outcome > 1.10:
                            event = "double"
                            est_xwoba = 1.15
                            launch_spd = 101.0
                            launch_ang = 19.0
                            woba_val = 1.25
                            woba_den = 1.0
                        elif rand_outcome > 0.95:
                            event = "single"
                            est_xwoba = 0.72
                            launch_spd = 92.0
                            launch_ang = 12.0
                            woba_val = 0.9
                            woba_den = 1.0
                        else:
                            event = "field_out"
                            est_xwoba = 0.18
                            launch_spd = 84.0
                            launch_ang = 35.0
                            woba_val = 0.0
                            woba_den = 1.0
                    elif rand_outcome > 0.50:
                        desc = "called_strike" if np.random.rand() > 0.5 else "swinging_strike"
                        pitch_type_result = "S"
                        strikes += 1
                    else:
                        desc = "ball"
                        pitch_type_result = "B"
                        balls += 1

                    if strikes >= 3:
                        event = "strikeout"
                        woba_val = 0.0
                        woba_den = 1.0
                        outs += 1
                        ab_number += 1
                        pitch_in_ab = 1
                        balls, strikes = 0, 0
                    elif balls >= 4:
                        event = "walk"
                        woba_val = 0.69
                        woba_den = 1.0
                        ab_number += 1
                        pitch_in_ab = 1
                        balls, strikes = 0, 0
                    elif is_in_play:
                        if "out" in (event or ""):
                            outs += 1
                        ab_number += 1
                        pitch_in_ab = 1
                        balls, strikes = 0, 0
                    else:
                        pitch_in_ab += 1

                    if outs >= 3:
                        current_inning += 1
                        outs = 0

                    pitch_row = {
                        "game_pk": game_pk,
                        "game_date": game_date,
                        "pitcher": p["id"],
                        "pitcher_name": p["name"],
                        "batter": 600000 + np.random.randint(1, 200),
                        "p_throws": p["throws"],
                        "stand": "R" if np.random.rand() > 0.4 else "L",
                        "inning": current_inning,
                        "inning_topbot": "Top",
                        "at_bat_number": ab_number,
                        "pitch_number": pitch_in_ab, # Proper PA-relative pitch number
                        "pitch_type": pt,
                        "pitch_name": "4-Seam Fastball" if pt == "FF" else ("Slider" if pt == "SL" else "Changeup"),
                        "release_speed": round(rel_spd, 2),
                        "release_pos_x": round(rel_x, 4),
                        "release_pos_y": round(rel_y, 4),
                        "release_pos_z": round(rel_z, 4),
                        "release_extension": round(rel_ext, 2),
                        "release_spin_rate": round(spin_rt, 1),
                        "spin_axis": round(curr_axis, 1),
                        "pfx_x": round(pfx_x, 3),
                        "pfx_z": round(pfx_z, 3),
                        "plate_x": round(plate_x, 3),
                        "plate_z": round(plate_z, 3),
                        "vx0": round(vx0, 3),
                        "vy0": round(vy0, 3),
                        "vz0": round(vz0, 3),
                        "ax": round(ax, 3),
                        "ay": round(ay, 3),
                        "az": round(az, 3),
                        "events": event,
                        "description": desc,
                        "balls": balls,
                        "strikes": strikes,
                        "outs_when_up": outs,
                        "type": pitch_type_result,
                        "woba_value": woba_val,
                        "woba_denom": woba_den,
                        "launch_speed": round(launch_spd, 1) if not np.isnan(launch_spd) else np.nan,
                        "launch_angle": round(launch_ang, 1) if not np.isnan(launch_ang) else np.nan,
                        "estimated_woba_using_speedangle": round(est_xwoba, 3) if not np.isnan(est_xwoba) else np.nan,
                        "home_team": "NYY",
                        "away_team": "BOS",
                        "actual_data_source": "simulation_benchmark"
                    }
                    all_pitches.append(pitch_row)

        df = pd.DataFrame(all_pitches)
        return df

    # Alias for backward compatibility
    def generate_realistic_sample_data(self, num_pitchers: int = 5, starts_per_pitcher: int = 15) -> pd.DataFrame:
        return self.generate_simulation_benchmark(num_pitchers, starts_per_pitcher)
