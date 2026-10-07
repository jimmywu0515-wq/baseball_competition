"""Plot saved directional/dispersion findings without fitting or loading pitches."""
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
import pandas as pd


def plot_summary(directory: Path) -> Path:
    direction = json.loads((directory / "directional_false_alarms.json").read_text())
    dispersion = json.loads((directory / "cross_pitcher_dispersion.json").read_text())
    study = json.loads((directory / "pitcher_scale_study_report.json").read_text())
    table = pd.read_csv(directory / "cross_pitcher_dispersion.csv")
    validation = direction["splits"]["validation"]
    clean = validation["clean_false_warning_starts"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    fig.suptitle("Directional and pitcher-scale hypotheses — development evidence", fontsize=16, fontweight="bold")
    shares = [clean["positive_sign_share"], validation["matched_baseline_positive_share"],
              validation["all_eligible_pitches"]["positive_sign_share"]]
    axes[0].bar(["Clean warning\nstarts", "Matched\nbackground", "All eligible\npitches"], shares,
                color=["#d66b37", "#315b79", "#909fae"])
    for index, value in enumerate(shares):
        axes[0].text(index, value + .025, f"{value:.1%}", ha="center", fontsize=10)
    axes[0].set(title="Hypothesized positive signs (2024)", ylabel="Share among classifiable rows", ylim=(0, 1))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].text(.98, .96, f"Only {clean['classifiable_rows']} of {clean['rows']} clean warning starts\nare classifiable; 8 pitchers",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=9)
    values = [table.loc[table.qualifies & ~table.strict_sensitivity, "std"],
              table.loc[table.qualifies & table.strict_sensitivity, "std"]]
    axes[1].boxplot(values, showmeans=True)
    axes[1].set_xticks([1, 2], ["Non-pre-onset", "Strict sensitivity"])
    axes[1].set(title="Training dispersion across 38 pitchers", ylabel="SD of mechanical distance", ylim=(0, 8))
    axes[1].text(.98, .96, f"Max/min SD: {dispersion['primary_non_pre_onset']['max_to_min_std_ratio']:.2f}×\nStrict sensitivity: {dispersion['strict_sensitivity']['max_to_min_std_ratio']:.2f}×",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=9)
    styles = {"global": ("Global scale", "#315b79", "o"),
              "scale_half": ("Pitcher scale: prior ×0.5", "#a49368", "v"),
              "proposed": ("Pitcher scale: prior ×1 (primary)", "#d66b37", "*"),
              "scale_double": ("Pitcher scale: prior ×2", "#657461", "^")}
    for row in study["comparison"]:
        label, color, marker = styles[row["model_key"]]
        axes[2].scatter(row["false_warnings_per_outing"], row["episode_recall"],
                        color=color, marker=marker, s=130 if marker == "*" else 55,
                        label=label, edgecolors="black", linewidths=.4)
    axes[2].set(title="Later-2024 frozen operating points", xlabel="False warnings per qualified outing",
                ylabel="Episode recall", xlim=(0, .5), ylim=(0, .14))
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].legend(frameon=False, fontsize=8, loc="upper left")
    axes[2].text(.98, .03, "446 outings; 973 episodes\nActual warning burdens differ", transform=axes[2].transAxes,
                 ha="right", va="bottom", fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.supxlabel("Thresholds use early 2024; assessment uses later 2024. Both were previously inspected. 2025 is excluded.", fontsize=10)
    path = directory / "directional_scale_summary.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "outputs/real_data/diagnostics")
    print(plot_summary(parser.parse_args().directory))
