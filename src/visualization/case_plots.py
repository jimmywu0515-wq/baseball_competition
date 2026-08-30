"""
Case Study Visualizations (§5 Step 8)
Generates high-resolution multi-panel plots demonstrating mechanical fatigue preceding velocity drop and collapse.
"""
import logging
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

class CaseStudyVisualizer:
    """
    Generates case study diagnostic charts.
    """
    def __init__(self, output_dir: Optional[str] = None):
        if output_dir is None:
            self.output_dir = Path("/Users/jimmywu/Desktop/baseball_competition/outputs/case_studies")
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_game_case_study(self, game_df: pd.DataFrame, game_pk: int, save_name: Optional[str] = None) -> str:
        """
        Creates a 4-panel diagnostic plot for a specific game:
        1. Pitcher Health Index & CUSUM Alert Point
        2. Release Point Drift (Release X & Z)
        3. Fastball Velocity (Showing velocity did not drop until AFTER alert)
        4. Rolling xwOBA & Collapse Window
        """
        df = game_df[game_df["game_pk"] == game_pk].sort_values("pitch_number_in_game").copy()
        if df.empty:
            logger.warning(f"No data for game_pk {game_pk}")
            return ""

        pitcher_name = df["pitcher_name"].iloc[0]
        game_date = df["game_date"].iloc[0]
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True, gridspec_kw={'hspace': 0.25})
        
        pitch_nums = df["pitch_number_in_game"].values
        alert_pitches = df[df["is_cusum_alert"]]["pitch_number_in_game"].values
        collapse_pitches = df[df["is_collapse_event"]]["pitch_number_in_game"].values
        
        first_alert = alert_pitches[0] if len(alert_pitches) > 0 else None
        first_collapse = collapse_pitches[0] if len(collapse_pitches) > 0 else None

        # -------------------------------------------------------------
        # Panel 1: Pitcher Health Index & Alert Trigger
        # -------------------------------------------------------------
        ax1 = axes[0]
        health = df["health_index"].values if "health_index" in df.columns else 100 * np.exp(-0.45 * df["mahalanobis_calibrated"].values)
        ax1.plot(pitch_nums, health, color="#1e88e5", linewidth=2.5, label="Mechanics Health Index (0-100)", zorder=3)
        ax1.axhline(60, color="#fb8c00", linestyle="--", alpha=0.7, label="Caution Threshold (60)")
        ax1.axhline(35, color="#e53935", linestyle="--", alpha=0.7, label="Critical Anomaly Threshold (35)")
        ax1.set_ylim(0, 105)
        ax1.set_ylabel("Health Index", fontsize=11, fontweight="bold")
        ax1.set_title(f"Case Study: {pitcher_name} | Date: {game_date} (Game PK: {game_pk})", fontsize=14, fontweight="bold", pad=12)

        if first_alert:
            ax1.axvline(first_alert, color="#d81b60", linestyle="-", linewidth=2, label=f"CUSUM Alert Trigger (Pitch #{first_alert})", zorder=4)
            ax1.scatter([first_alert], [health[first_alert-1]], color="#d81b60", s=100, zorder=5)

        if first_collapse:
            ax1.axvspan(first_collapse, max(pitch_nums), color="#ef5350", alpha=0.15, label=f"Actual Collapse Inning (Pitch #{first_collapse}+)")

        ax1.legend(loc="upper right", frameon=True, framealpha=0.9)

        # -------------------------------------------------------------
        # Panel 2: Micro-Mechanics Release Drift
        # -------------------------------------------------------------
        ax2 = axes[1]
        ax2.plot(pitch_nums, df["release_pos_x"], color="#43a047", linewidth=2, label="Release Pos X (ft)", alpha=0.9)
        ax2.plot(pitch_nums, df["release_pos_z"], color="#8e24aa", linewidth=2, label="Release Pos Z (ft)", alpha=0.9)
        ax2.set_ylabel("Release Pos (ft)", fontsize=11, fontweight="bold")
        
        if first_alert:
            ax2.axvline(first_alert, color="#d81b60", linestyle="--", linewidth=1.5)
        if first_collapse:
            ax2.axvspan(first_collapse, max(pitch_nums), color="#ef5350", alpha=0.15)
            
        ax2.legend(loc="upper right", frameon=True, framealpha=0.9)

        # -------------------------------------------------------------
        # Panel 3: Pitch Velocity (Showing Lagging Indicator Effect)
        # -------------------------------------------------------------
        ax3 = axes[2]
        # Fastballs only for clear velocity tracking
        fb_df = df[df["pitch_type"] == "FF"]
        if not fb_df.empty:
            ax3.scatter(fb_df["pitch_number_in_game"], fb_df["release_speed"], color="#3949ab", alpha=0.5, s=30, label="4-Seam FB Pitch Speed")
            # Rolling FB Speed
            ax3.plot(fb_df["pitch_number_in_game"], fb_df["release_speed"].rolling(4, min_periods=1).mean(), color="#3949ab", linewidth=2.5, label="Rolling FB Velocity (mph)")
        else:
            ax3.plot(pitch_nums, df["release_speed"].rolling(5, min_periods=1).mean(), color="#3949ab", linewidth=2.5, label="Rolling Pitch Velocity (mph)")

        ax3.set_ylabel("Velocity (mph)", fontsize=11, fontweight="bold")
        
        if first_alert:
            ax3.axvline(first_alert, color="#d81b60", linestyle="--", linewidth=1.5)
            lead_pitches = (first_collapse - first_alert) if first_collapse and first_collapse >= first_alert else 0
            if lead_pitches > 0:
                ax3.text(first_alert + 1, ax3.get_ylim()[0] + 0.5, f"← Alert Lead Time: {lead_pitches} pitches earlier than collapse", color="#d81b60", fontweight="bold", fontsize=10)

        if first_collapse:
            ax3.axvspan(first_collapse, max(pitch_nums), color="#ef5350", alpha=0.15)

        ax3.legend(loc="upper right", frameon=True, framealpha=0.9)

        # -------------------------------------------------------------
        # Panel 4: Rolling 3-PA Blended xwOBA (Ground Truth Collapse)
        # -------------------------------------------------------------
        ax4 = axes[3]
        xwoba = df["window_blended_xwoba"].values if "window_blended_xwoba" in df.columns else np.zeros(len(df))
        ax4.plot(pitch_nums, xwoba, color="#d81b60", linewidth=2.5, label="Rolling 3-PA Blended xwOBA")
        ax4.axhline(0.450, color="#d32f2f", linestyle="--", label="Collapse Threshold (xwOBA = 0.450)")
        ax4.set_ylabel("Rolling xwOBA", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Pitch Number in Game", fontsize=12, fontweight="bold")
        ax4.set_ylim(0, max(1.0, np.max(xwoba) * 1.15))

        if first_alert:
            ax4.axvline(first_alert, color="#d81b60", linestyle="--", linewidth=1.5)
        if first_collapse:
            ax4.axvspan(first_collapse, max(pitch_nums), color="#ef5350", alpha=0.15)

        ax4.legend(loc="upper left", frameon=True, framealpha=0.9)

        if save_name is None:
            save_name = f"case_study_{pitcher_name.replace(' ', '_')}_{game_pk}.png"
        out_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved case study visualization to {out_path}")
        return str(out_path)
