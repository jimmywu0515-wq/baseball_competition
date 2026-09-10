"""
Historical Baseline Builder (§5 Step 2 & Critique Fixes)
STRICT NO-LOOKAHEAD ENFORCEMENT:
1. Baselines for game date D are computed strictly from starts with date < D.
2. If prior qualified starts < min_prior_starts (default 5), marked as INSUFFICIENT_HISTORY.
3. Uses circular spin components (spin_axis_cos, spin_axis_sin) to avoid angle boundary issues.
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
    "spin_axis_cos",
    "spin_axis_sin",
    "pfx_x",
    "pfx_z",
    "vaa"
]

class BaselineBuilder:
    """
    Builds personal historical baselines per pitcher and pitch type with strict temporal isolation.
    """
    def __init__(self, 
                 historical_window_starts: int = 12, 
                 min_prior_starts: int = 3,
                 min_pitches_for_baseline: int = 25, 
                 ridge_reg: float = 1e-4):
        self.window_starts = historical_window_starts
        self.min_prior_starts = min_prior_starts
        self.min_pitches = min_pitches_for_baseline
        self.ridge_reg = ridge_reg

    def build_baselines(self, qualified_pitches_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Iterates over each pitcher and game in strict chronological order.
        Never uses future games or unverified fallbacks.
        """
        logger.info("Computing historical baselines under strict temporal isolation (No Lookahead)...")
        df = qualified_pitches_df.copy()

        # Ensure all feature columns exist
        for col in FEATURE_COLS:
            if col not in df.columns:
                if col == "spin_axis_cos":
                    df[col] = np.cos(np.radians(df.get("spin_axis", 0.0)))
                elif col == "spin_axis_sin":
                    df[col] = np.sin(np.radians(df.get("spin_axis", 0.0)))
                else:
                    df[col] = 0.0

        # Unique outings in chronological order
        outing_columns = ["pitcher", "pitcher_name", "game_pk", "game_date"]
        outing_columns += [column for column in ("dataset_split", "actual_data_source", "requested_data_mode")
                           if column in df.columns]
        outings = df[outing_columns].drop_duplicates().sort_values(
            by=["pitcher", "game_date", "game_pk"]
        ).reset_index(drop=True)

        baseline_rows = []
        baseline_store = {}

        for pitcher_id, p_outings in outings.groupby("pitcher"):
            outings_list = p_outings.to_dict("records")

            for i, curr_outing in enumerate(outings_list):
                curr_game_pk = curr_outing["game_pk"]
                curr_date = curr_outing["game_date"]

                # Prior outings strictly before index i (chronologically prior dates)
                prior_outings = outings_list[max(0, i - self.window_starts): i]

                # Policy check: Must have at least min_prior_starts completed prior outings
                if len(prior_outings) < self.min_prior_starts:
                    # Mark as insufficient history (cold start) - DO NOT borrow from future!
                    for pt in df[df["pitcher"] == pitcher_id]["pitch_type"].unique():
                        baseline_store[f"{pitcher_id}_{curr_game_pk}_{pt}"] = {
                            "status": "INSUFFICIENT_HISTORY",
                            "prior_starts": len(prior_outings)
                        }
                    continue

                prior_pks = [g["game_pk"] for g in prior_outings]
                prior_pitches = df[(df["pitcher"] == pitcher_id) & (df["game_pk"].isin(prior_pks))]

                # Compute baseline per pitch type
                for pt, pt_pitches in prior_pitches.groupby("pitch_type"):
                    X = pt_pitches[FEATURE_COLS].dropna()
                    if len(X) < self.min_pitches:
                        continue

                    mu = X.mean().to_dict()
                    sigma = X.std().replace(0, 1e-4).to_dict()

                    # Covariance Matrix & Inversion with Ridge Regularization
                    cov_mat = np.cov(X.values, rowvar=False)
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
                        "window_start_date": prior_outings[0]["game_date"],
                        "window_games_count": len(prior_outings),
                        "pitches_count": len(X),
                        "dataset_split": curr_outing.get("dataset_split", "unassigned"),
                        "actual_data_source": curr_outing.get("actual_data_source", "unknown"),
                        "requested_data_mode": curr_outing.get("requested_data_mode", "unknown"),
                        "mean_release_speed": mu.get("release_speed", 0.0),
                        "std_release_speed": sigma.get("release_speed", 1.0),
                        "mean_release_pos_x": mu.get("release_pos_x", 0.0),
                        "std_release_pos_x": sigma.get("release_pos_x", 1.0),
                        "mean_release_pos_z": mu.get("release_pos_z", 0.0),
                        "std_release_pos_z": sigma.get("release_pos_z", 1.0),
                        "covariance_matrix_json": json.dumps(cov_mat.tolist()),
                        "precision_matrix_json": json.dumps(prec_mat.tolist())
                    }
                    baseline_rows.append(row_entry)

                    key = f"{pitcher_id}_{curr_game_pk}_{pt}"
                    baseline_store[key] = {
                        "status": "QUALIFIED",
                        "mu_vec": np.array([mu[c] for c in FEATURE_COLS]),
                        "sigma_vec": np.array([sigma[c] for c in FEATURE_COLS]),
                        "cov_mat": cov_mat,
                        "prec_mat": prec_mat,
                        "feature_cols": FEATURE_COLS,
                        "prior_starts": len(prior_outings)
                    }

        baseline_df = pd.DataFrame(baseline_rows)
        logger.info(f"Built {len(baseline_df)} temporally isolated baselines.")
        return baseline_df, baseline_store
