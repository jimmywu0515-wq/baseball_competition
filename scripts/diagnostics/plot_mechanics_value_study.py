"""Render the saved development study without rerunning models or loading pitches."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


def plot_summary(directory: Path) -> Path:
    report = json.loads((directory / "mechanics_value_report.json").read_text(encoding="utf-8"))
    episodes = pd.read_csv(directory / "mechanics_value_episode_audit.csv")
    validation = episodes.loc[episodes.dataset_split.eq("validation")]
    comparisons = pd.DataFrame(report["comparison"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
    fig.suptitle("Mechanics warning study — development evidence", fontsize=16, fontweight="bold")
    axes[0].hist(validation.onset_pitch, bins=np.arange(0, validation.onset_pitch.max() + 6, 5),
                 color="#315b79", edgecolor="white")
    calibration = report["plan"]["calibration_pitches"]
    axes[0].axvspan(0, calibration, color="#db754c", alpha=.22, label=f"First {calibration} pitches: calibration")
    axes[0].set(title="When existing episode labels begin (all 2024)", xlabel="Pitch number at episode onset", ylabel="Episodes")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    audit = report["label_audit"]["splits"]["validation"]
    axes[0].text(.98, .88, f"{audit['no_protocol_warning_opportunity']:,} / {audit['episodes']:,} episodes\nhave no eligible prior warning pitch",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=10)
    styles = {"context": ("Context only", "#315b79"), "proposed": ("Context + mechanics", "#d66b37"),
              "pitch_count": ("Pitch count", "#657461")}
    for key, (label, color) in styles.items():
        points = comparisons.loc[comparisons.model_key.eq(key)].sort_values("selection_budget")
        axes[1].plot(points.false_warnings_per_outing, points.episode_recall,
                     marker="o", color=color, label=label)
        primary = points.loc[points.selection_budget.eq(report["plan"]["primary_budget"])].iloc[0]
        axes[1].scatter(primary.false_warnings_per_outing, primary.episode_recall,
                        s=150, marker="*", color=color, edgecolors="black", linewidths=.5, zorder=4)
    axes[1].set(title="Frozen thresholds assessed on July–December 2024",
                 xlabel="False warnings per qualified outing", ylabel="Episode recall")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].text(.98, .02, "Stars: primary 0.5 selection allowance\nOther points: 0.1 and 0.25 selection allowances",
                 transform=axes[1].transAxes, fontsize=9, va="bottom", ha="right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.supxlabel("Thresholds selected on January–June 2024. All of 2024 was previously explored; 2025 is excluded.", fontsize=10)
    path = directory / "mechanics_value_summary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "outputs/diagnostics")
    print(plot_summary(parser.parse_args().directory))
