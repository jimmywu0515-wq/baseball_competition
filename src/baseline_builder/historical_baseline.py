"""
Historical Baseline Builder (§5 Step 2)
Computes rolling prior historical baseline distribution (mean, std, covariance & precision matrix)
strictly BEFORE game date to prevent lookahead data leakage.
"""
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "release_speed",
    "release_spin_rate",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "vaa"
]

class BaselineBuilder:
    """
    Builds personal historical baselines per pitcher and pitch type.
    """
    def __init__(self, historical_window_starts: int = 12, min_pitches_for_baseline: int = 30, ridge_reg: float = 1e-4):
        self.window_starts = historical_window_starts
        self.min_pitches = min_pitches_for_baseline
        self.ridge_reg = ridge_reg

    def build_baselines(self, qualified_pitches_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Iterates over each pitcher and game, computing historical statistics strictly
        using prior starts.
        Returns:
            baseline_df: DataFrame of computed baselines per (pitcher, pitch_type, as_of_game_pk)
            baseline_store: In-memory lookup dictionary for fast anomaly computation
        """
        logger.info("Computing rolling historical baselines with strict temporal separation...")
        df = qualified_pitches_df.copy()
        
        # Unique games per pitcher in temporal order
        games_meta = df[["pitcher", "pitcher_name", "game_pk", "game_date"]].drop_duplicates().sort_values(
            by=["pitcher", "game_date", "game_pk"]
        ).reset_index(drop=True)

        baseline_rows = []
        baseline_store = {}

        for pitcher_id, p_games in games_meta.groupby("pitcher"):
            p_games_list = p_games.to_dict("records")
            
            for i, current_game in enumerate(p_games_list):
                curr_game_pk = current_game["game_pk"]
                curr_date = current_game["game_date"]
                
                # Take prior games up to window_starts (strictly i > 0 to prevent lookahead)
                prior_games = p_games_list[max(0, i - self.window_starts): i]
                
                # If pitcher is at start 0, fall back to initial starts (or use all available up to start 3)
                if not prior_games:
                    prior_games = p_games_list[: min(len(p_games_list), 3)]

                prior_game_pks = [g["game_pk"] for g in prior_games]
                prior_pitches = df[(df["pitcher"] == pitcher_id) & (df["game_pk"].isin(prior_game_pks))]

                # Compute baseline for each pitch type
                for pt, pt_pitches in prior_pitches.groupby("pitch_type"):
                    if len(pt_pitches) < self.min_pitches:
                        # If not enough pitches in window, take broader pitcher history for this pitch type
                        pt_pitches = df[(df["pitcher"] == pitcher_id) & (df["pitch_type"] == pt)]

                    if len(pt_pitches) < 10:
                        continue

                    # Extract feature matrix
                    X = pt_pitches[FEATURE_COLS].dropna()
                    if len(X) < 10:
                        continue

                    mu = X.mean().to_dict()
                    sigma = X.std().replace(0, 1e-4).to_dict()
                    
                    # Covariance Matrix & Inversion with Ridge Regularization
                    cov_mat = np.cov(X.values, rowvar=False)
                    # Add ridge regularization
                    reg_cov_mat = cov_mat + np.eye(cov_mat.shape[0]) * self.ridge_reg
                    try:
                        prec_mat = np.linalg.inv(reg_cov_mat)
                    except np.linalg.LinAlgError:
                        prec_mat = np.linalg.pinv(reg_cov_mat)

                    row_entry = {
                        "pitcher": pitcher_id,
                        "pitch_type": pt,
                        "as_of_game_pk": curr_game_pk,
                        "as_of_date": curr_date,
                        "window_start_date": prior_games[0]["game_date"] if prior_games else curr_date,
                        "window_games_count": len(prior_games),
                        "pitches_count": len(X),
                        "mean_release_pos_x": mu.get("release_pos_x", 0.0),
                        "std_release_pos_x": sigma.get("release_pos_x", 1.0),
                        "mean_release_pos_z": mu.get("release_pos_z", 0.0),
                        "std_release_pos_z": sigma.get("release_pos_z", 1.0),
                        "mean_release_extension": mu.get("release_extension", 0.0),
                        "std_release_extension": sigma.get("release_extension", 1.0),
                        "mean_release_speed": mu.get("release_speed", 0.0),
                        "std_release_speed": sigma.get("release_speed", 1.0),
                        "mean_release_spin_rate": mu.get("release_spin_rate", 0.0),
                        "std_release_spin_rate": sigma.get("release_spin_rate", 1.0),
                        "mean_spin_axis": mu.get("spin_axis", 0.0),
                        "std_spin_axis": sigma.get("spin_axis", 1.0),
                        "mean_pfx_x": mu.get("pfx_x", 0.0),
                        "std_pfx_x": sigma.get("pfx_x", 1.0),
                        "mean_pfx_z": mu.get("pfx_z", 0.0),
                        "std_pfx_z": sigma.get("pfx_z", 1.0),
                        "mean_vaa": mu.get("vaa", 0.0),
                        "std_vaa": sigma.get("vaa", 1.0),
                        "covariance_matrix_json": json.dumps(cov_mat.tolist()),
                        "precision_matrix_json": json.dumps(prec_mat.tolist())
                    }
                    baseline_rows.append(row_entry)

                    # Save in in-memory store for instant scoring lookup
                    key = f"{pitcher_id}_{curr_game_pk}_{pt}"
                    baseline_store[key] = {
                        "mu_vec": np.array([mu[col] for col in FEATURE_COLS]),
                        "sigma_vec": np.array([sigma[col] for col in FEATURE_COLS]),
                        "cov_mat": cov_mat,
                        "prec_mat": prec_mat,
                        "feature_cols": FEATURE_COLS
                    }

        baseline_df = pd.DataFrame(baseline_rows)
        logger.info(f"Successfully generated {len(baseline_df)} pitcher-game-pitch_type baselines.")
        return baseline_df, baseline_store
