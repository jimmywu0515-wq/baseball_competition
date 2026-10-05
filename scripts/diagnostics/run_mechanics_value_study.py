"""Audit existing labels and test incremental mechanics value on development data.

Usage: python scripts/diagnostics/run_mechanics_value_study.py

Fits on 2023, chooses operating thresholds on January-June 2024, and assesses
frozen thresholds on July-December 2024. Both parts of 2024 have previously been
explored, so this is a development experiment, never a pristine holdout claim.
The plan is saved before model fitting. No 2025 rows are loaded or evaluated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnostics.check_cusum_pitch_count_correlation import (
    EVALUATION_COLUMNS, _load_parquet, _validate_provenance, episodes_from_labels, json_safe,
)
from src.configuration import load_project_config
from src.evaluation.bootstrap import paired_bootstrap_confidence_intervals
from src.evaluation.mechanics_value import (
    CONTEXT_FEATURES, MECHANICS_FEATURES, audit_labels, development_roles,
    fit_matched_models, select_from_curve,
)
from src.evaluation.protocol import (
    eligible_pitch_mask, evaluate_warning_predictions, threshold_tradeoff_curve,
)

logger = logging.getLogger(__name__)
PREFIX = "mechanics_value"
MODELS = {"context": "Context only", "proposed": "Context plus mechanics", "pitch_count": "Pitch count"}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_study(out_dir: Path, bootstrap_samples: int = 1000, root: Path = ROOT) -> dict:
    if bootstrap_samples < 20:
        raise ValueError("At least 20 bootstrap replicates are required")
    config = load_project_config(root / "config/config.yaml")
    manifest = json.loads((root / "outputs/real_data/protocol_manifest.json").read_text(encoding="utf-8"))
    if config != manifest["resolved_runtime"]["resolved_config"] or manifest["actual_data_source"] != "mlb_statcast":
        raise ValueError("The real-data manifest and current configuration must agree")
    predicate = ds.field("dataset_split").isin(["train", "validation"])
    frame = _load_parquet(root, "data/gold/fact_pitch_anomaly_scores.parquet", predicate)
    labels = _load_parquet(root, "data/gold/fact_collapse_labels.parquet", predicate)
    _validate_provenance(frame, manifest, "scores")
    _validate_provenance(labels, manifest, "labels")
    identity = ["game_pk", "pitcher", "pitch_number_in_outing"]
    truth = ["is_collapse_event", "is_censored_followup", "y_true_onset_in_horizon", "collapse_episodes_count_in_game"]
    pd.testing.assert_frame_equal(
        frame.set_index(identity)[truth].sort_index(), labels.set_index(identity)[truth].sort_index(),
    )
    episodes = episodes_from_labels(labels)
    horizon = int(config["evaluation"]["warning_matching_horizon_pitches"])
    calibration = int(config["baseline"]["intra_game_calibration_pitches"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT / "src/evaluation/mechanics_value.py",
               ROOT / "scripts/diagnostics/check_cusum_pitch_count_correlation.py",
               ROOT / "src/evaluation/protocol.py", ROOT / "src/evaluation/bootstrap.py",
               ROOT / "src/evaluation/baseline_comparator.py"]
    plan = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_run_id": manifest["run_id"], "parent_protocol_sha256": manifest["protocol_sha256"],
        "prior_2025_exposure_disclosure": manifest["prior_2025_exposure_disclosure"],
        "development_exposure_disclosure": "Both halves of 2024 were used in earlier exploratory work. This chronological development comparison is not a new independent holdout.",
        "training": "2023", "threshold_selection": "2024-01-01 through 2024-06-30",
        "assessment": "2024-07-01 through 2024-12-31", "selection_end": "2024-06-30",
        "test_rows_loaded": False,
        "context_features": CONTEXT_FEATURES, "additional_mechanics_features": MECHANICS_FEATURES,
        "classifier": {"type": "logistic_regression", "C": 1.0, "class_weight": "balanced", "random_state": 42},
        "score_interpretation": "Ranking scores, not calibrated probabilities",
        "preprocessing": "Mechanics clipped at 2023 common-training-row 0.5/99.5 percentiles; StandardScaler fitted on those same train rows",
        "availability_policy": "Both fitted models share training rows and finite mechanics scoring opportunities. The pitch-count comparator uses the same availability. All qualified outings and episodes remain in denominators.",
        "false_warning_budgets": [0.1, 0.25, 0.5], "primary_budget": 0.5,
        "threshold_candidates": "Union of 31 and 101 selection-score quantiles, plus no-alert infinity; recall/precision/false-warning tie-breaks match production",
        "horizon_pitches": horizon, "calibration_pitches": calibration,
        "bootstrap": {"samples": bootstrap_samples, "seed": 20261002, "unit": "pitcher", "thresholds_reselected": False},
        "label_policy": "Audit existing labels without revising them or removing early/unavailable episodes",
        "label_review_sample": "Two episodes per original split, onset-reason combination, and calibration flag; seed 20261002; independent of model predictions",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    write_json(out_dir / f"{PREFIX}_plan.json", plan)
    logger.info("Saved experiment plan; auditing existing labels")
    label_report, episode_audit, cases, pa_context = audit_labels(
        frame, episodes, calibration, horizon, int(config["labels"]["window_pa_size"]),
    )
    write_json(out_dir / f"{PREFIX}_label_audit.json", label_report)
    for suffix, data in [("episode_audit", episode_audit), ("review_cases", cases), ("review_pa_context", pa_context)]:
        data.to_csv(out_dir / f"{PREFIX}_{suffix}.csv", index=False)
    logger.info("Label audit saved: %s", label_report["splits"])

    needed = list(dict.fromkeys(EVALUATION_COLUMNS + [
        "pitch_type", "inning", "mahalanobis_calibrated", "z_release_speed",
        "z_release_pos_x", "z_release_pos_z", "z_release_extension", "y_true_onset_in_horizon",
    ]))
    working = development_roles(frame[needed], plan["selection_end"])
    development_episodes = development_roles(episodes, plan["selection_end"])
    for split in ["train", "selection", "assessment"]:
        if not working.dataset_split.eq(split).any():
            raise ValueError(f"No {split} data for the development experiment")
    logger.info("Fitting matched context and context-plus-mechanics models on 2023 only")
    scored, fitted = fit_matched_models(working)
    write_json(out_dir / f"{PREFIX}_fitted_models.json", fitted)
    evaluation_cols = EVALUATION_COLUMNS + ["study_available", "y_true_onset_in_horizon"]
    evaluation_cols += [f"study_score_{key}" for key in MODELS]
    scored = scored[list(dict.fromkeys(evaluation_cols))].copy()
    selection = scored.loc[scored.dataset_split.eq("selection")].copy()
    assessment = scored.loc[scored.dataset_split.eq("assessment")].copy()
    curves, results, warning_tables, match_tables, primary = [], [], [], [], {}
    frozen_thresholds = {}
    # Complete every selection sweep and save all chosen thresholds before any
    # assessment metric is calculated.
    for key, display in MODELS.items():
        logger.info("Selecting %s thresholds on January-June 2024", display)
        col = f"study_score_{key}"
        mask = eligible_pitch_mask(selection, "selection", availability=selection.study_available)
        finite = selection.loc[mask, col].dropna()
        if finite.empty:
            raise ValueError("No common scoring opportunities in the selection period")
        quantiles = np.unique(np.r_[np.linspace(0, 1, 31), np.linspace(0, 1, 101)])
        candidates = np.r_[np.inf, np.unique(np.quantile(finite, quantiles))[::-1]]
        curve = threshold_tradeoff_curve(
            selection, development_episodes, col, horizon_pitches=horizon,
            split="selection", candidates=candidates,
        )
        curve["model_key"] = key
        curve["threshold_status"] = np.where(np.isfinite(curve.threshold), "selected", "no_alert")
        curves.append(curve)
        frozen_thresholds[key] = {}
        for budget in plan["false_warning_budgets"]:
            chosen = select_from_curve(curve, budget)
            frozen_thresholds[key][str(budget)] = {
                "threshold": float(chosen.threshold), "threshold_status": chosen.threshold_status,
                "selection_episode_recall": float(chosen.episode_recall),
                "selection_warning_precision": float(chosen.warning_precision),
                "selection_false_warnings_per_outing": float(chosen.false_warnings_per_outing),
            }
    write_json(out_dir / f"{PREFIX}_frozen_thresholds.json", frozen_thresholds)
    pd.concat(curves, ignore_index=True).replace([np.inf, -np.inf], np.nan).to_csv(
        out_dir / f"{PREFIX}_threshold_curves.csv", index=False,
    )
    logger.info("Thresholds frozen; evaluating July-December 2024")
    for key, display in MODELS.items():
        col = f"study_score_{key}"
        for budget in plan["false_warning_budgets"]:
            choice = frozen_thresholds[key][str(budget)]
            prediction = assessment[col].ge(choice["threshold"]) & assessment.study_available
            metrics, warnings, matches = evaluate_warning_predictions(
                assessment, development_episodes, prediction, horizon_pitches=horizon,
                split="assessment", availability=assessment.study_available,
            )
            results.append({"model_key": key, "model": display, "selection_budget": budget, **choice, **metrics})
            if budget == plan["primary_budget"]:
                primary[key] = metrics
                assessment[f"prediction_{key}"] = prediction
                warning_tables.append(warnings.assign(model_key=key))
                match_tables.append(matches.assign(model_key=key))
                logger.info("%s: recall %.4f; precision %.4f; false warnings/outing %.4f", display,
                            metrics["episode_recall"], metrics["warning_precision"], metrics["false_warnings_per_outing"])
    comparison = pd.DataFrame(results)
    comparison.replace([np.inf, -np.inf], np.nan).to_csv(out_dir / f"{PREFIX}_comparison.csv", index=False)
    pd.concat(warning_tables, ignore_index=True).to_csv(out_dir / f"{PREFIX}_warnings.csv", index=False)
    pd.concat(match_tables, ignore_index=True).to_csv(out_dir / f"{PREFIX}_matches.csv", index=False)
    logger.info("Computing paired pitcher bootstrap intervals at frozen primary thresholds")
    intervals = paired_bootstrap_confidence_intervals(
        assessment, development_episodes,
        {key: f"prediction_{key}" for key in MODELS}, horizon_pitches=horizon,
        split="assessment", n_bootstrap=bootstrap_samples, seed=20261002,
        cluster_by_pitcher=True, model_availability={key: "study_available" for key in MODELS},
    )
    intervals.to_csv(out_dir / f"{PREFIX}_bootstrap.csv", index=False)
    report = {
        "plan": plan, "fitted_models": fitted, "label_audit": label_report,
        "comparison": results, "bootstrap": intervals.to_dict(orient="records"),
        "primary_context_plus_mechanics_minus_context": {
            metric: primary["proposed"][metric] - primary["context"][metric]
            for metric in ["episode_recall", "warning_precision", "false_warnings_per_outing"]
        },
        "interpretation_limits": [
            "Exploratory chronological development assessment; 2024 was previously inspected.",
            "Shared selection caps do not guarantee equal realized warning burdens or an assessment cap.",
            "Only eligible opportunities with finite mechanics are scored; every qualified outing and episode remains in the denominator.",
            "The retrospective minimum-50-pitch outing rule remains; early removals are outside this population.",
            "An alert is a monitoring/review signal; this experiment estimates neither fatigue nor the causal benefit of removing a pitcher.",
        ],
    }
    write_json(out_dir / f"{PREFIX}_report.json", report)
    logger.info("Study completed: %s", out_dir / f"{PREFIX}_report.json")
    return json_safe(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/diagnostics")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_study(args.out_dir, args.bootstrap_samples)


if __name__ == "__main__":
    main()
