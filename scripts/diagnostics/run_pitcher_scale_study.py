"""Gated, frozen-training pitcher-scale comparison on development data only.

Fit 2023 scales; select thresholds on early 2024; assess on late 2024.
Both 2024 periods were previously inspected. No 2025 data or warehouse writes.
The directional gate failed, so no unsupported directional mode is fitted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics.check_cusum_pitch_count_correlation import (
    EVALUATION_COLUMNS, _load_parquet, _validate_provenance, episodes_from_labels, scoring_mask,
)
from scripts.diagnostics.directional_scale_diagnostics import write_json
from src.baseline_builder.dispersion_shrinkage import fit_pitcher_dispersion_scales
from src.configuration import load_project_config
from src.evaluation.ablation_runner import AblationRunner
from src.evaluation.bootstrap import paired_bootstrap_confidence_intervals
from src.evaluation.mechanics_value import development_roles
from src.evaluation.protocol import evaluate_warning_predictions, select_operating_threshold


def run_study(data_root: Path, samples: int = 1000) -> dict:
    out = ROOT / "outputs/real_data/diagnostics"
    config = load_project_config(ROOT / "config/config.yaml")
    manifest = json.loads((ROOT / "outputs/real_data/protocol_manifest.json").read_text())
    gate = json.loads((out / "cross_pitcher_dispersion.json").read_text())
    if (not gate["proceed_to_scale_development_comparison"] or gate["run_id"] != manifest["run_id"]
            or gate["protocol_sha256"] != manifest["protocol_sha256"]):
        raise ValueError("A passing Phase 0 dispersion gate for the current parent run is required")
    if config != manifest["resolved_runtime"]["resolved_config"]:
        raise ValueError("Configuration must match the published warehouse")
    if samples < 20:
        raise ValueError("At least 20 bootstrap replicates are required")
    sources = [Path(__file__), ROOT / "src/baseline_builder/dispersion_shrinkage.py",
               ROOT / "src/changepoint_detector/cusum_detector.py", ROOT / "src/evaluation/ablation_runner.py",
               ROOT / "src/evaluation/protocol.py", ROOT / "src/evaluation/bootstrap.py",
               ROOT / "src/evaluation/mechanics_value.py", ROOT / "scripts/diagnostics/directional_scale_diagnostics.py",
               ROOT / "scripts/diagnostics/check_cusum_pitch_count_correlation.py"]
    plan = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_id": manifest["run_id"],
        "protocol_sha256": manifest["protocol_sha256"], "test_rows_loaded": False,
        "gate_report_sha256": hashlib.sha256((out / "cross_pitcher_dispersion.json").read_bytes()).hexdigest(),
        "train": "2023 only", "selection": "January-June 2024", "assessment": "July-December 2024",
        "prior_pitches": "Median qualifying training pitch count; fixed sensitivity multipliers 0.5, 1, 2",
        "reference_mean": float(config["changepoint"]["cusum_reference_mean"]),
        "reference_std": float(config["changepoint"]["cusum_reference_std"]),
        "primary_candidate": "proposed (prior multiplier 1)", "selection_budget": .5,
        "threshold_policy": "Production 31-quantile selector plus no-alert candidate; every threshold saved before assessment",
        "availability_policy": "All models use the identical production mechanics availability; all qualified outings and episodes retained",
        "application_policy": "Frozen train parameters applied only to 2024, never retrospectively to earlier train outings",
        "fallback_policy": "Explicit scale 1 for pitchers without sufficient scored training outings",
        "bootstrap": {"samples": samples, "seed": 20261007, "units": ["outing", "pitcher"], "thresholds_reselected": False},
        "subgroups": "Low/high dispersion uses training scale median among qualifying pitchers; fallback pitchers are separate",
        "exposure_disclosure": "Both 2024 periods have previously been inspected; this is exploratory development evidence",
        "prior_2025_exposure_disclosure": manifest["prior_2025_exposure_disclosure"],
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    write_json(out / "pitcher_scale_study_plan.json", plan)
    print("Saved scale-study plan before fitting", flush=True)
    predicate = ds.field("dataset_split").isin(["train", "validation"])
    frame = _load_parquet(data_root, "data/gold/fact_pitch_anomaly_scores.parquet", predicate)
    labels = _load_parquet(data_root, "data/gold/fact_collapse_labels.parquet", predicate)
    _validate_provenance(frame, manifest, "scores")
    _validate_provenance(labels, manifest, "labels")
    identity = ["game_pk", "pitcher", "pitch_number_in_outing"]
    truth = ["is_collapse_event", "is_censored_followup", "y_true_onset_in_horizon", "collapse_episodes_count_in_game"]
    pd.testing.assert_frame_equal(frame.set_index(identity)[truth].sort_index(), labels.set_index(identity)[truth].sort_index())
    episodes = development_roles(episodes_from_labels(labels))
    calibration = int(config["baseline"]["intra_game_calibration_pitches"])
    horizon = int(config["evaluation"]["warning_matching_horizon_pitches"])
    train = frame.loc[frame.dataset_split.eq("train")]
    scales, fitted = fit_pitcher_dispersion_scales(train, min_outings=int(config["baseline"]["min_prior_starts"]),
                                                 calibration_pitches=calibration)
    variants = {"global": None, "scale_half": .5, "proposed": 1.0, "scale_double": 2.0}
    fitted_variants, scale_tables = {}, []
    scored = development_roles(frame.loc[frame.dataset_split.eq("validation")].copy())
    scored = scored[list(dict.fromkeys(EVALUATION_COLUMNS + ["mahalanobis_calibrated", "cusum_stat", "y_true_onset_in_horizon"]))].copy()
    if pd.to_datetime(scored.game_date).le(fitted["training_end"]).any():
        raise ValueError("Frozen training scales cannot score dates at/before their training cutoff")
    available = scoring_mask(scored, "mahalanobis_calibrated", calibration)
    scored["_ablation_avail_mahalanobis_calibrated"] = available
    for key, multiplier in variants.items():
        reference = plan["reference_std"]
        if multiplier is not None:
            table, params = fit_pitcher_dispersion_scales(
                train, prior_pitches=fitted["prior_pitches"]*multiplier,
                min_outings=int(config["baseline"]["min_prior_starts"]), calibration_pitches=calibration,
            )
            mapping = table.set_index("pitcher").scale_used.to_dict()
            reference = {int(p): plan["reference_std"] * mapping.get(int(p), 1.0) for p in scored.pitcher.unique()}
            params["fallback_pitchers"] = sorted(int(p) for p in scored.pitcher.unique() if p not in mapping)
            fitted_variants[key] = params
            scale_tables.append(table.assign(model_key=key, **{k:plan[k] for k in ["run_id","protocol_sha256"]}))
        scored = AblationRunner._run_cusum_on_scores(
            scored, "mahalanobis_calibrated", f"scale_study_{key}",
            slack_k=float(config["changepoint"]["cusum_slack_k"]),
            threshold_h=float(config["changepoint"]["cusum_internal_alert_h"]),
            reference_mean=plan["reference_mean"], reference_std=reference, calibration_pitches=calibration,
        )
    if not np.allclose(scored.scale_study_global, scored.cusum_stat, equal_nan=True, rtol=0, atol=1e-12):
        raise ValueError("Scalar baseline changed: recomputed CUSUM does not match the committed scores")
    pd.concat(scale_tables,ignore_index=True).to_csv(out / "mart_pitcher_dispersion_scale.csv",index=False)
    write_json(out / "pitcher_scale_fitted_parameters.json", fitted_variants)
    selected, results, raw_warnings, matches, subgroups = {}, [], [], [], []
    selection = scored.loc[scored.dataset_split.eq("selection")].copy()
    assessment = scored.loc[scored.dataset_split.eq("assessment")].copy()
    for key in variants:
        threshold, metrics = select_operating_threshold(selection,episodes,f"scale_study_{key}",.5,horizon_pitches=horizon,split="selection")
        selected[key] = {"threshold": threshold, "threshold_status": "selected" if np.isfinite(threshold) else "no_alert", "selection_metrics": metrics}
    write_json(out / "pitcher_scale_frozen_thresholds.json",selected)
    print("All scale-study thresholds frozen; assessing later 2024",flush=True)
    training_scale = scales.set_index("pitcher").scale_used
    median = training_scale.median()
    assessment["dispersion_group"] = assessment.pitcher.map(training_scale).map(
        lambda value: "fallback" if pd.isna(value) else "low" if value<=median else "high",
    )
    for key in variants:
        prediction = assessment[f"scale_study_{key}"].ge(selected[key]["threshold"]) & available.reindex(assessment.index)
        assessment[f"prediction_{key}"] = prediction
        metrics,warnings,matched = evaluate_warning_predictions(
            assessment,episodes,prediction,horizon_pitches=horizon,split="assessment",availability=available,
        )
        results.append({"model_key":key,"prior_multiplier":variants[key],**selected[key],**metrics})
        raw_warnings.append(warnings.assign(model_key=key))
        matches.append(matched.assign(model_key=key))
        for group,pitches in assessment.groupby("dispersion_group"):
            subgroup_metrics,_,_=evaluate_warning_predictions(
                pitches,episodes,prediction,horizon_pitches=horizon,split="assessment",availability=available,
            )
            subgroups.append({"model_key":key,"dispersion_group":group,"pitchers":pitches.pitcher.nunique(),**subgroup_metrics})
        print(key,{m:metrics[m] for m in ["episode_recall","warning_precision","false_warnings_per_outing"]},flush=True)
    assessment["study_available"] = available.reindex(assessment.index)
    intervals=[]
    for clustered in [False,True]:
        print("Bootstrap unit:","pitcher" if clustered else "outing",flush=True)
        intervals.append(paired_bootstrap_confidence_intervals(
            assessment,episodes,{"original":"prediction_global","proposed":"prediction_proposed"},
            horizon_pitches=horizon,split="assessment",n_bootstrap=samples,seed=20261007,
            cluster_by_pitcher=clustered,model_availability={"original":"study_available","proposed":"study_available"},
        ))
    intervals=pd.concat(intervals,ignore_index=True)
    for name,data in [("comparison",pd.DataFrame(results)),("subgroups",pd.DataFrame(subgroups)),
                      ("warnings",pd.concat(raw_warnings,ignore_index=True)),("matches",pd.concat(matches,ignore_index=True)),
                      ("bootstrap",intervals)]:
        data.assign(**{k:plan[k] for k in ["run_id","protocol_sha256"]}).to_csv(out/f"pitcher_scale_{name}.csv",index=False)
    report={"plan":plan,"default_cusum_parity":"Every 2024 pitch equals the committed baseline",
            "fitted_parameters":fitted_variants,"comparison":results,"subgroups":subgroups,
            "bootstrap":intervals.to_dict(orient="records"),
            "limits":["The directional gate did not pass; unsupported directional grid cells are omitted.",
                      "Different selected thresholds can imply different realized warning burdens.",
                      "This scale-only intervention retains the original mismatched global reference location.",
                      "The retrospective minimum-50-pitch filter and original episode definition are retained.",
                      "Development results from previously explored 2024; no 2025 evaluation or production promotion."]}
    write_json(out/"pitcher_scale_study_report.json",report)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=ROOT)
    parser.add_argument("--bootstrap-samples",type=int,default=1000)
    args=parser.parse_args()
    run_study(args.data_root,args.bootstrap_samples)
