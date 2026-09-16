"""Refresh matching-horizon artifacts from persisted frozen scores and warnings."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.configuration import load_project_config
from src.evaluation.lead_time import fixed_warning_horizon_analysis
from src.label_builder.collapse_labels import CollapseLabelBuilder
from src.reporting import render_validation_report
from src.storage.adapter import StorageManager


MODEL_PREDICTIONS = {
    "proposed": "is_proposed_operating_alert",
    "contextual": "is_contextual_operating_alert",
    "velocity": "is_velocity_operating_alert",
    "pitch_count": "is_pitch_count_operating_alert",
}


def main() -> None:
    config = load_project_config(PROJECT_ROOT / "config" / "config.yaml")
    storage = StorageManager(base_dir=str(PROJECT_ROOT))
    scored = storage.load_table("fact_pitch_anomaly_scores", layer="gold")
    if scored.empty:
        raise RuntimeError("No frozen score table is available; run the full pipeline first.")

    builder = CollapseLabelBuilder(
        window_pa_size=int(config["labels"]["window_pa_size"]),
        blended_xwoba_threshold=float(config["labels"]["blended_xwoba_threshold"]),
        min_barrels_in_window=int(config["labels"]["min_barrels_in_window"]),
        min_bb_hbp_in_window=int(config["labels"]["min_bb_hbp_in_window"]),
        horizon_pitches=int(config["evaluation"]["prediction_horizon_pitches"]),
    )
    labeled_frames = []
    episode_frames = []
    for _, outing in scored.groupby(["game_pk", "pitcher"], sort=False):
        labeled, episodes = builder.build_labels_for_outing(outing)
        labeled_frames.append(labeled)
        if not episodes.empty:
            episode_frames.append(episodes)
    labeled = pd.concat(labeled_frames, ignore_index=True)
    episodes = pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame()
    sensitivity, matches, distribution = fixed_warning_horizon_analysis(
        labeled, episodes, MODEL_PREDICTIONS,
        horizons=config["evaluation"]["sensitivity_horizons"], split="test",
    )
    actual_source = str(scored["actual_data_source"].dropna().iloc[0])
    for frame in (matches, distribution):
        if not frame.empty and "actual_data_source" not in frame:
            frame["actual_data_source"] = actual_source

    storage.save_tables_atomically([
        (sensitivity, "mart_lead_time_sensitivity", "gold"),
        (matches, "fact_matched_warning_episodes", "gold"),
        (distribution, "mart_lead_time_distribution", "gold"),
    ])
    output_dir = PROJECT_ROOT / "outputs" / "real_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(output_dir / "lead_time_sensitivity.csv", index=False)
    matches.to_csv(output_dir / "matched_warning_episodes.csv", index=False)
    distribution.to_csv(output_dir / "lead_time_distribution.csv", index=False)
    render_validation_report(output_dir)
    storage.close()


if __name__ == "__main__":
    main()
