"""
Case Study Visualizations (§8 & §14 Critique Fixes)
Generates 4 Documented Distinct Case Studies:
1. True Positive (Useful Early Warning): Delivery drifted before velocity loss & collapse.
2. False Positive (False Alarm): Delivery drifted, alert triggered, but pitcher escaped damage.
3. False Negative (Missed Collapse): Mechanics stable, but hard contact occurred.
4. True Negative (Stable Outing): Clean outing, high mechanical consistency throughout.
"""
import logging
from pathlib import Path
from typing import List, Optional, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

PRIMARY_FASTBALLS = ["FF", "SI", "FC"]

class CaseStudyVisualizer:
    """
    Generates diagnostic 4-panel case study plots for the 4 canonical outcomes.
    """
    def __init__(self, output_dir: Optional[str] = None):
        if output_dir is None:
            self.output_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "case_studies"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_case_study(self, 
                        outing_df: pd.DataFrame, 
                        case_type: str, 
                        title_note: str,
                        save_name: str) -> str:
        """
        Plots a 4-panel diagnostic chart for an outing.
        case_type: 'True Positive (Early Warning)', 'False Positive (False Alarm)',
                   'False Negative (Missed Collapse)', or 'True Negative (Stable Outing)'
        """
        df = outing_df.sort_values("pitch_number_in_outing").copy()
        pitcher_name = df["pitcher_name"].iloc[0] if "pitcher_name" in df.columns else "Pitcher"
        game_date = df["game_date"].iloc[0] if "game_date" in df.columns else "2023"
        game_pk = df["game_pk"].iloc[0]
        data_source = df.get("actual_data_source", pd.Series(["unknown"])).iloc[0]

        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True, gridspec_kw={'hspace': 0.25})

        pitch_nums = df["pitch_number_in_outing"].values
        alert_col = "is_proposed_operating_alert" if "is_proposed_operating_alert" in df else "is_cusum_alert"
        alert_mask = df.get(alert_col, pd.Series(False, index=df.index)).fillna(False).astype(bool)
        if "score_available" in df:
            alert_mask &= df["score_available"].fillna(False).astype(bool)
        alert_pitches = df.loc[alert_mask, "pitch_number_in_outing"].values
        collapse_pitches = df[df.get("is_collapse_event", False)]["pitch_number_in_outing"].values

        first_alert = alert_pitches[0] if len(alert_pitches) > 0 else None
        first_collapse = collapse_pitches[0] if len(collapse_pitches) > 0 else None

        # -------------------------------------------------------------
        # Panel 1: Mechanics Stability Index (MSI) & Alert
        # -------------------------------------------------------------
        ax1 = axes[0]
        msi = df.get("health_index", 100.0 * np.exp(-0.40 * df.get("mahalanobis_calibrated", 1.0))).values
        ax1.plot(pitch_nums, msi, color="#1e88e5", linewidth=2.5, label="Mechanics Stability Index (MSI 0-100)", zorder=3)
        ax1.axhline(60, color="#fb8c00", linestyle="--", alpha=0.7, label="Caution Threshold (60)")
        ax1.axhline(35, color="#e53935", linestyle="--", alpha=0.7, label="Danger Threshold (35)")
        ax1.axvspan(0, 20, color="#b0bec5", alpha=0.2, label="Calibration Period (Pitches 1-20)")
        ax1.set_ylim(0, 105)
        ax1.set_ylabel("MSI Score", fontsize=11, fontweight="bold")
        ax1.set_title(
            f"[{case_type}] {pitcher_name} | {game_date} (Game PK: {game_pk}) | source={data_source}\n{title_note}",
            fontsize=13, fontweight="bold", pad=12,
        )

        if first_alert:
            ax1.axvline(first_alert, color="#d81b60", linestyle="-", linewidth=2, label=f"CUSUM Alert (Pitch #{first_alert})", zorder=4)
            alert_position = np.flatnonzero(pitch_nums == first_alert)
            if len(alert_position):
                ax1.scatter([first_alert], [msi[alert_position[0]]], color="#d81b60", s=100, zorder=5)

        if first_collapse:
            ax1.axvspan(first_collapse, max(pitch_nums), color="#ef5350", alpha=0.15, label=f"Collapse Episode (Pitch #{first_collapse}+)")

        ax1.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)

        # -------------------------------------------------------------
        # Panel 2: Release Point 3D Drift (X & Z)
        # -------------------------------------------------------------
        ax2 = axes[1]
        ax2.plot(pitch_nums, df["release_pos_x"], color="#43a047", linewidth=2, label="Release Pos X (ft)")
        ax2.plot(pitch_nums, df["release_pos_z"], color="#8e24aa", linewidth=2, label="Release Pos Z (ft)")
        ax2.axvspan(0, 20, color="#b0bec5", alpha=0.2)
        ax2.set_ylabel("Release Pos (ft)", fontsize=11, fontweight="bold")

        if first_alert:
            ax2.axvline(first_alert, color="#d81b60", linestyle="--", linewidth=1.5)
        if first_collapse:
            ax2.axvspan(first_collapse, max(pitch_nums), color="#ef5350", alpha=0.15)
        ax2.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)

        # -------------------------------------------------------------
        # Panel 3: Fastball Velocity (Within Pitch Type)
        # -------------------------------------------------------------
        ax3 = axes[2]
        fb_df = df[df["pitch_type"].isin(PRIMARY_FASTBALLS)]
        if not fb_df.empty:
            colors = {"FF": "#3949ab", "SI": "#00897b", "FC": "#8e24aa"}
            for pitch_type, pitch_group in fb_df.groupby("pitch_type"):
                ax3.plot(
                    pitch_group["pitch_number_in_outing"],
                    pitch_group["release_speed"].rolling(4, min_periods=1).mean(),
                    color=colors.get(pitch_type, "#3949ab"), linewidth=2,
                    label=f"{pitch_type} rolling velocity",
                )
        else:
            ax3.plot(pitch_nums, df["release_speed"].rolling(5, min_periods=1).mean(), color="#3949ab", linewidth=2.5, label="Rolling Pitch Velocity (mph)")

        ax3.axvspan(0, 20, color="#b0bec5", alpha=0.2)
        ax3.set_ylabel("Fastball Velo (mph)", fontsize=11, fontweight="bold")

        if first_alert:
            ax3.axvline(first_alert, color="#d81b60", linestyle="--", linewidth=1.5)
            if first_collapse and first_collapse >= first_alert:
                lead = first_collapse - first_alert
                ax3.text(first_alert + 1, ax3.get_ylim()[0] + 0.3, f"Lead Time: {lead} pitches earlier", color="#d81b60", fontweight="bold", fontsize=10)

        if first_collapse:
            ax3.axvspan(first_collapse, max(pitch_nums), color="#ef5350", alpha=0.15)
        ax3.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)

        # -------------------------------------------------------------
        # Panel 4: Rolling 3-PA xwOBA (Ground Truth Target)
        # -------------------------------------------------------------
        ax4 = axes[3]
        xwoba_series = df.get("window_blended_xwoba", pd.Series(np.nan, index=df.index))
        if xwoba_series.notna().any():
            ax4.plot(pitch_nums, xwoba_series, color="#c2185b", linewidth=2.5, label="Rolling PA-window blended xwOBA")
        else:
            ax4.text(0.5, 0.5, "xwOBA unavailable", transform=ax4.transAxes,
                     ha="center", va="center", color="#757575")
        ax4.axhline(0.450, color="#d32f2f", linestyle="--", label="Collapse Threshold (0.450)")
        ax4.axvspan(0, 20, color="#b0bec5", alpha=0.2)
        ax4.set_ylabel("Rolling xwOBA", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Pitch Number in Outing", fontsize=12, fontweight="bold")

        if first_alert:
            ax4.axvline(first_alert, color="#d81b60", linestyle="--", linewidth=1.5)
        if first_collapse:
            ax4.axvspan(first_collapse, max(pitch_nums), color="#ef5350", alpha=0.15)
        ax4.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)

        out_path = self.output_dir / save_name
        plt.savefig(out_path, dpi=250, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved [{case_type}] case study to {out_path}")
        return str(out_path)
