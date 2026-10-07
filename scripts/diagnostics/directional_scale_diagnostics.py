"""Read-only, development-only gates for proposed directional/CUSUM changes.

Reuse published signed calibrated deltas and the production warning matcher.
No warehouse connection, test rows, scorer/configuration changes, or retuning.
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
    _load_parquet, _validate_provenance, episodes_from_labels, json_safe, scoring_mask,
)
from src.configuration import load_project_config
from src.evaluation.protocol import eligible_pitch_mask, evaluate_warning_predictions

KEYS = ["game_pk", "pitcher"]
IDENTITY = KEYS + ["pitch_number_in_outing"]
POSITIVE_HYPOTHESES = ["release_speed", "release_pos_z", "release_extension", "release_spin_rate"]
MATCH_KEYS = ["pitcher", "pitch_type", "dominant_drift_feature", "workload_band"]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def hypothesis_signs(frame: pd.DataFrame) -> pd.DataFrame:
    """Positive/negative are hypothesized directions, never performance diagnoses.

    Only four named raw calibrated features are classified. Other dominant
    features and rounded zero deltas remain explicit unknown/neutral cases.
    """
    result = frame.copy()
    result["hypothesized_positive"] = np.nan
    result["direction_status"] = "unclassified_dominant_feature"
    for feature in POSITIVE_HYPOTHESES:
        selected = result.dominant_drift_feature.eq(feature)
        delta = pd.to_numeric(result[f"calib_delta_{feature}"], errors="coerce")
        finite = selected & np.isfinite(delta)
        result.loc[selected, "direction_status"] = "missing_delta"
        result.loc[finite & delta.eq(0), "direction_status"] = "neutral_or_rounded_zero"
        signed = finite & delta.ne(0)
        result.loc[signed, "hypothesized_positive"] = delta.loc[signed].gt(0).astype(float)
        result.loc[signed, "direction_status"] = np.where(delta.loc[signed].gt(0), "positive", "negative")
    result["workload_band"] = ((result.pitch_number_in_outing - 1) // 10).astype(int)
    return result


def sign_summary(rows: pd.DataFrame) -> dict:
    values = rows.hypothesized_positive.dropna()
    return {
        "rows": len(rows), "classifiable_rows": len(values),
        "positive": int(values.eq(1).sum()), "negative": int(values.eq(0).sum()),
        "positive_sign_share": float(values.mean()) if len(values) else np.nan,
        "direction_status_counts": rows.direction_status.value_counts().to_dict(),
    }


def matched_background(warnings: pd.DataFrame, eligible: pd.DataFrame) -> pd.DataFrame:
    """Reweight all eligible pitches to the exact warning-start strata."""
    background = eligible.groupby(MATCH_KEYS, dropna=False).hypothesized_positive.agg(
        baseline_positive_share="mean", baseline_classifiable_pitches="count",
    ).reset_index()
    return warnings.merge(background, on=MATCH_KEYS, how="left", validate="many_to_one")


def excess_interval(rows: pd.DataFrame, samples: int, seed: int) -> dict:
    classified = rows.dropna(subset=["hypothesized_positive", "baseline_positive_share"]).copy()
    classified["excess"] = classified.hypothesized_positive - classified.baseline_positive_share
    clusters = classified.groupby("pitcher").excess.agg(["sum", "count"])
    if len(clusters) < 2:
        return {"estimate": float(classified.excess.mean()), "ci_lower_95": np.nan,
                "ci_upper_95": np.nan, "pitchers": len(clusters), "warnings": len(classified),
                "valid_replicates": 0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), size=(samples, len(clusters)))
    sums, counts = clusters["sum"].to_numpy(), clusters["count"].to_numpy()
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"estimate": float(classified.excess.mean()),
            "ci_lower_95": float(np.quantile(estimates, .025)),
            "ci_upper_95": float(np.quantile(estimates, .975)),
            "pitchers": len(clusters), "warnings": len(classified), "valid_replicates": samples}


def directional_diagnostic(frame: pd.DataFrame, episodes: pd.DataFrame, alerts: pd.DataFrame,
                           threshold: float, horizon: int, plan: dict) -> tuple[dict, pd.DataFrame]:
    dates = pd.to_datetime(frame.game_date, errors="coerce")
    if (not frame.dataset_split.isin(["train", "validation"]).all()
            or dates.isna().any() or dates.dt.year.gt(2024).any()):
        raise ValueError("Direction decisions cannot load test rows")
    report, exported = {"splits": {}}, []
    for split, pitches in frame.groupby("dataset_split", sort=True):
        available = pitches.score_available.fillna(False).astype(bool) & np.isfinite(pitches.cusum_stat)
        predictions = pitches.cusum_stat.ge(threshold) & available
        metrics, warnings, _ = evaluate_warning_predictions(
            pitches, episodes, predictions, horizon_pitches=horizon, split=split, availability=available,
        )
        stored = alerts.loc[alerts.dataset_split.eq(split)]
        observed = set(map(tuple, warnings[KEYS + ["warning_pitch"]].to_numpy()))
        expected = set(map(tuple, stored[KEYS + ["alert_pitch_number"]].to_numpy()))
        if observed != expected or len(stored) != len(expected):
            raise ValueError(f"{split} warning starts differ from published fact_alert_events")
        if not np.allclose(stored.operating_threshold, threshold, rtol=0, atol=1e-10):
            raise ValueError("Persisted alert threshold differs from the frozen manifest")
        signed = hypothesis_signs(pitches)
        eligible = signed.loc[eligible_pitch_mask(pitches, split, availability=available)].copy()
        warning_rows = warnings.merge(signed, left_on=KEYS + ["warning_pitch"], right_on=IDENTITY,
                                     how="left", validate="one_to_one", suffixes=("", "_pitch"))
        if "is_matched_warning" not in warning_rows:
            warning_rows["is_matched_warning"] = False
        episode_outings = set(map(tuple, episodes.loc[episodes.dataset_split.eq(split), KEYS].drop_duplicates().to_numpy()))
        warning_rows["clean_outing"] = pd.Series(
            [tuple(key) not in episode_outings for key in warning_rows[KEYS].to_numpy()], index=warning_rows.index, dtype=bool,
        )
        warning_rows = matched_background(warning_rows, eligible)
        warning_rows["warning_category"] = np.select(
            [~warning_rows.is_evaluable_warning, warning_rows.is_matched_warning, warning_rows.clean_outing],
            ["censored", "matched", "false_warning_clean_outing"], default="false_warning_episode_outing",
        )
        clean = warning_rows.loc[warning_rows.warning_category.eq("false_warning_clean_outing")]
        interval = excess_interval(clean, plan["uncertainty"]["bootstrap_samples"], plan["uncertainty"]["seed"])
        summary = sign_summary(clean)
        # Secondary pitch summary uses authoritative starts propagated through
        # the original sequence; only flagged, evaluable pitches within H enter.
        starts = warning_rows.set_index(KEYS + ["warning_pitch"], drop=False).warning_pitch
        ordered = signed.sort_values(IDENTITY).copy()
        ordered["warning_start"] = [starts.get(tuple(key), np.nan) for key in ordered[IDENTITY].to_numpy()]
        ordered["warning_start"] = ordered.groupby(KEYS).warning_start.ffill()
        clean_starts = set(map(tuple, clean[KEYS + ["warning_pitch"]].to_numpy()))
        in_clean_run = pd.Series([tuple(key) in clean_starts for key in ordered[KEYS + ["warning_start"]].to_numpy()], index=ordered.index)
        run_mask = (in_clean_run & predictions.reindex(ordered.index) &
                    eligible_pitch_mask(ordered, split, availability=available) &
                    ordered.pitch_number_in_outing.le(ordered.warning_start + horizon))
        report["splits"][split] = {
            "production_metrics": metrics, "persisted_warning_start_parity": True,
            "all_eligible_pitches": sign_summary(eligible), "clean_false_warning_starts": summary,
            "matched_baseline_positive_share": float(clean.loc[clean.hypothesized_positive.notna(), "baseline_positive_share"].mean()),
            "paired_pitcher_bootstrap_excess": interval,
            "clean_warning_run_pitches_within_horizon": sign_summary(ordered.loc[run_mask]),
            "warning_categories": warning_rows.warning_category.value_counts().to_dict(),
            "other_unmatched_warning_starts": sign_summary(warning_rows.loc[warning_rows.warning_category.eq("false_warning_episode_outing")]),
        }
        exported.append(warning_rows[KEYS + ["warning_pitch", "game_date", "dataset_split", "pitch_type",
                                            "dominant_drift_feature", "direction_status", "hypothesized_positive",
                                            "workload_band", "baseline_positive_share", "baseline_classifiable_pitches",
                                            "warning_category", "clean_outing"]])
    validation = report["splits"]["validation"]
    summary, interval, gate = validation["clean_false_warning_starts"], validation["paired_pitcher_bootstrap_excess"], plan["directional_gate"]
    checks = {
        "positive_share_above_half": bool(summary["positive_sign_share"] > gate["positive_sign_share_above"]),
        "excess_above_matched_background": bool(interval["estimate"] >= gate["minimum_matched_baseline_excess"]),
        "enough_warning_starts": interval["warnings"] >= gate["minimum_classifiable_warning_starts"],
        "enough_pitchers": interval["pitchers"] >= gate["minimum_pitchers"],
        "bootstrap_excess_above_zero": bool(interval["ci_lower_95"] > gate["paired_pitcher_bootstrap_lower_excess_above"]),
    }
    report.update({"gate_checks": checks, "proceed_to_directional_production": all(checks.values()),
                   "interpretation": "Feature signs are hypotheses. Gates address positive-sign overrepresentation among classifiable clean-warning starts, not fatigue or pitch quality."})
    return report, pd.concat(exported, ignore_index=True)


def training_dispersion(frame: pd.DataFrame, plan: dict, strict: bool = False,
                        calibration_pitches: int = 20) -> tuple[dict, pd.DataFrame]:
    if not frame.dataset_split.eq("train").all() or pd.to_datetime(frame.game_date).dt.year.ne(2023).any():
        raise ValueError("Dispersion fitting accepts 2023 train rows only")
    mask = scoring_mask(frame, "mahalanobis_calibrated", calibration_pitches) & ~frame.y_true_onset_in_horizon.astype(bool)
    if strict:
        mask &= ~frame.is_collapse_event.astype(bool) & ~frame.is_censored_followup.astype(bool)
    scores = frame.loc[mask].copy()
    gate = plan["dispersion_gate"]
    rows = []
    for pitcher, pitches in scores.groupby("pitcher", sort=True):
        values = pitches.mahalanobis_calibrated
        outings = pitches.game_pk.nunique()
        std = float(values.std(ddof=1))
        qualifies = outings >= gate["minimum_eligible_outings_per_pitcher"] and len(values) >= gate["minimum_pitches_per_pitcher"] and np.isfinite(std) and std > 1e-6
        rows.append({"pitcher": int(pitcher), "pitcher_name": pitches.pitcher_name.iloc[0],
                     "n_eligible_pitches": len(values), "n_eligible_outings": outings,
                     "mean": float(values.mean()), "std": std, "median": float(values.median()),
                     "p25": float(values.quantile(.25)), "p75": float(values.quantile(.75)),
                     "qualifies": bool(qualifies), "strict_sensitivity": strict})
    table = pd.DataFrame(rows)
    qualified = table.loc[table.qualifies] if len(table) else table
    if len(qualified) < 2:
        raise ValueError("At least two qualifying pitchers are required")
    report = {"pitchers": len(qualified), "excluded_pitchers": int((~table.qualifies).sum()),
              "pitches": int(qualified.n_eligible_pitches.sum()), "distribution": {}}
    for col in ["mean", "std"]:
        report["distribution"][col] = {"min": float(qualified[col].min()), "max": float(qualified[col].max()),
                                        "p25": float(qualified[col].quantile(.25)), "median": float(qualified[col].median()),
                                        "p75": float(qualified[col].quantile(.75)),
                                        "iqr": float(qualified[col].quantile(.75)-qualified[col].quantile(.25))}
    report["max_to_min_std_ratio"] = float(qualified["std"].max()/qualified["std"].min())
    pooled = scores.loc[scores.pitcher.isin(qualified.pitcher), "mahalanobis_calibrated"]
    report["pooled_mean"] = float(pooled.mean())
    report["pooled_std"] = float(pooled.std(ddof=1))
    # Per-outing sufficient statistics allow a cluster bootstrap without
    # pretending that individual pitches are independent observations.
    scores["squared"] = scores.mahalanobis_calibrated.pow(2)
    outing_stats = scores.loc[scores.pitcher.isin(qualified.pitcher)].groupby(KEYS).agg(
        n=("mahalanobis_calibrated", "size"), total=("mahalanobis_calibrated", "sum"), squared=("squared", "sum"),
    )
    rng = np.random.default_rng(plan["uncertainty"]["seed"])
    sampled_stds = []
    samples = plan["uncertainty"]["bootstrap_samples"]
    for pitcher in qualified.pitcher:
        stats = outing_stats.xs(pitcher, level="pitcher").to_numpy()
        draws = rng.integers(0, len(stats), size=(samples, len(stats)))
        totals = stats[draws].sum(axis=1)
        variance = (totals[:,2] - totals[:,1]**2/totals[:,0])/(totals[:,0]-1)
        sampled_stds.append(np.sqrt(np.maximum(variance,0)))
    stds = np.array(sampled_stds)
    ratios = stds.max(axis=0)/stds.min(axis=0)
    finite = ratios[np.isfinite(ratios)]
    report["outing_bootstrap_ratio"] = {"ci_lower_95": float(np.quantile(finite,.025)),
                                         "ci_upper_95": float(np.quantile(finite,.975)), "valid_replicates": len(finite)}
    report["passes_ratio_gate"] = report["max_to_min_std_ratio"] >= gate["minimum_max_to_min_std_ratio"]
    return report, table


def run_diagnostic(kind: str, data_root: Path, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or ROOT / "outputs/real_data/diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT / "outputs/real_data/diagnostics/directional_scale_plan.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "outputs/real_data/protocol_manifest.json").read_text(encoding="utf-8"))
    config = load_project_config(ROOT / "config/config.yaml")
    if config != manifest["resolved_runtime"]["resolved_config"]:
        raise ValueError("Configuration must agree with published scores")
    if plan["run_id"] != manifest["run_id"] or plan["protocol_sha256"] != manifest["protocol_sha256"]:
        raise ValueError("The frozen diagnostic plan belongs to another parent run")
    predicate = ds.field("dataset_split").isin(["train", "validation"])
    frame = _load_parquet(data_root, "data/gold/fact_pitch_anomaly_scores.parquet", predicate)
    labels = _load_parquet(data_root, "data/gold/fact_collapse_labels.parquet", predicate)
    _validate_provenance(frame, manifest, "scores")
    _validate_provenance(labels, manifest, "labels")
    truth = ["is_collapse_event", "is_censored_followup", "y_true_onset_in_horizon", "collapse_episodes_count_in_game"]
    pd.testing.assert_frame_equal(frame.set_index(IDENTITY)[truth].sort_index(), labels.set_index(IDENTITY)[truth].sort_index())
    common = {"run_id": manifest["run_id"], "protocol_sha256": manifest["protocol_sha256"],
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "test_rows_loaded": False,
              "parent_threshold": manifest["resolved_runtime"]["selected_models"]["proposed"]["validation_selected_operating_threshold"],
              "plan_sha256": hashlib.sha256((ROOT / "outputs/real_data/diagnostics/directional_scale_plan.json").read_bytes()).hexdigest(),
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), ROOT / "src/evaluation/protocol.py", ROOT / "scripts/diagnostics/check_cusum_pitch_count_correlation.py"]}}
    horizon = int(config["evaluation"]["warning_matching_horizon_pitches"])
    if kind == "directional":
        alerts = _load_parquet(data_root, "data/gold/fact_alert_events.parquet", predicate)
        _validate_provenance(alerts, manifest, "alerts")
        if not alerts.detector_type.eq("CUSUM_VALIDATION_SELECTED").all():
            raise ValueError("Unexpected detector in persisted warning starts")
        result, table = directional_diagnostic(frame, episodes_from_labels(labels), alerts, common["parent_threshold"], horizon, plan)
        name = "directional_false_alarms"
    elif kind == "dispersion":
        train = frame.loc[frame.dataset_split.eq("train")].copy()
        calibration = int(config["baseline"]["intra_game_calibration_pitches"])
        primary, table = training_dispersion(train, plan, calibration_pitches=calibration)
        strict, strict_table = training_dispersion(train, plan, strict=True, calibration_pitches=calibration)
        result = {"primary_non_pre_onset": primary, "strict_sensitivity": strict,
                  "proceed_to_scale_development_comparison": primary["passes_ratio_gate"] and strict["passes_ratio_gate"],
                  "interpretation": "Training dispersion heterogeneity alone does not establish that scaling improves warnings. The original global reference mean remains a separate problem."}
        table = pd.concat([table,strict_table],ignore_index=True)
        name = "cross_pitcher_dispersion"
    else:
        raise ValueError(kind)
    result = {**common, **result}
    table["run_id"] = common["run_id"]
    table["protocol_sha256"] = common["protocol_sha256"]
    table.to_csv(out_dir / f"{name}.csv", index=False)
    write_json(out_dir / f"{name}.json", result)
    print(json.dumps(json_safe(result), indent=2, allow_nan=False))
    return json_safe(result)


def main(kind: str) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT, help="Repository containing the published Parquet warehouse")
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    run_diagnostic(kind, args.data_root, args.out_dir)
