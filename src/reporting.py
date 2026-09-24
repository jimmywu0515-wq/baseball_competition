"""Render the traceable real-data validation report from generated artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from src.presentation import load_protocol_manifest


def _read_csv(output_dir: Path, filename: str) -> pd.DataFrame:
    path = output_dir / filename
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def _markdown(frame: pd.DataFrame, empty_message: str = "No rows.") -> str:
    return frame.where(pd.notna(frame), "N/A").to_markdown(index=False) if not frame.empty else empty_message


def render_validation_report(output_dir: Path) -> Path:
    metrics = json.loads((output_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    manifest = load_protocol_manifest(output_dir)
    comparison = _read_csv(output_dir, "model_comparison.csv")
    historical = _read_csv(output_dir, "historical_2024_model_comparison.csv")
    bootstrap = _read_csv(output_dir, "bootstrap_confidence_intervals.csv")
    lead_time = _read_csv(output_dir, "lead_time_sensitivity.csv")
    missingness = _read_csv(output_dir, "missing_data_summary.csv")
    excluded_pitchers = _read_csv(output_dir, "excluded_pitchers.csv")
    excluded_outings = _read_csv(output_dir, "excluded_outings.csv")
    unavailable = _read_csv(output_dir, "unavailable_scores.csv")
    integrity = _read_csv(output_dir, "warehouse_integrity_report.csv")
    ablations = _read_csv(output_dir, "ablation_results.csv")
    sensitivity = _read_csv(output_dir, "sensitivity_results.csv")
    case_index = _read_csv(output_dir / "case_studies", "case_study_index.csv")
    cohort_selection = _read_csv(output_dir, "cohort_selection.csv")
    ingestion_segments = _read_csv(output_dir, "ingestion_segments.csv")
    pitcher_evaluation = _read_csv(output_dir, "pitcher_model_evaluation.csv")
    evaluation_coverage = _read_csv(output_dir, "evaluation_coverage.csv")
    resolved = manifest.get("resolved_runtime", {})
    allowance = resolved.get("threshold_selection", {}).get(
        "allowed_maximum_false_warnings_per_outing", "unknown"
    )

    unavailable_count = int(unavailable["pitch_count"].sum()) if "pitch_count" in unavailable else 0
    accounting = pd.DataFrame([{
        "cohort_pitchers": metrics.get("cohort_size"),
        "preselected_pitchers": metrics.get("preselected_pitchers_count"),
        "test_pitchers": metrics.get("test_pitchers_count"),
        "qualified_outings_all_seasons": metrics.get("qualified_outings"),
        "qualified_pitches_all_seasons": metrics.get("qualified_pitches"),
        "test_evaluated_outings": metrics.get("evaluated_outings_count"),
        "test_evaluated_pitches": metrics.get("evaluated_pitches_count"),
        "test_episodes": metrics.get("total_collapse_episodes"),
        "test_warnings": metrics.get("total_warnings"),
        "excluded_pitchers": len(excluded_pitchers),
        "excluded_outings": len(excluded_outings),
        "unavailable_pitch_scores": unavailable_count,
        "actual_data_source": metrics.get("actual_data_source"),
    }])
    exclusion_summary = (
        excluded_outings.groupby("outing_qualification_reason").size()
        .reset_index(name="outing_count")
        if "outing_qualification_reason" in excluded_outings else pd.DataFrame()
    )
    unavailable_summary = (
        unavailable.groupby("score_status", dropna=False)["pitch_count"].sum()
        .reset_index()
        if {"score_status", "pitch_count"}.issubset(unavailable.columns) else pd.DataFrame()
    )

    # Scientific interpretation section
    interpretation_section = (
        "## Scientific Interpretation and Findings\n\n"
        "> **Central Conclusion:**\n"
        "> The mechanical warning system has not demonstrated improved predictive performance "
        "over simple contextual or workload baselines. It detects more episodes than the evaluated "
        "velocity-drop benchmark, with a higher false-warning burden. Mechanical drift summaries may "
        "support interpretation, but their additional value for coaching decisions remains unvalidated.\n\n"
        "### Key Statistical Findings\n\n"
        "1. **Risk Ratio Confidence Intervals:** In the expanded cohort holdout analysis, the proposed "
        "risk-ratio confidence interval includes 1.0 (estimate: ~1.058), indicating that the current evaluation "
        "does not establish increased collapse risk after an alert under that analysis.\n"
        "2. **Comparison with Contextual Baseline:** Proposed-minus-contextual confidence intervals include 0 "
        "for the primary reported comparisons. The evaluation has not demonstrated predictive superiority over "
        "the contextual baseline (pitch count + times through order + inning). Note that failing to reject the null "
        "is not proof of equivalence.\n"
        "3. **Proposed Recall vs. Pitch Count:** Proposed recall is lower than pitch-count recall. In the expanded "
        "evaluation, the outing-bootstrap difference estimate is -0.01518 (95% CI: [-0.02898, -0.00192]), while the "
        "pitcher-clustered interval is [-0.03188, +0.0000136]. Because the pitcher-clustered interval crosses zero, "
        "the statistical significance of the deficit depends on the resampling assumption. The positive upper bound "
        "must not be rounded to zero.\n"
        "4. **Proposed Precision vs. Pitch Count:** Proposed warning precision (33.3%) is numerically lower than "
        "traditional pitch count precision (37.2%). The mechanical detector does not yield higher warning precision.\n"
        "5. **Velocity Benchmark Evaluation:** Against the evaluated velocity-drop benchmark, the proposed system "
        "detects more episodes but also generates more false warnings. In the expanded cohort, the velocity benchmark "
        "produced only one test warning; approximately 38.9% of outing-bootstrap replicates have undefined velocity "
        "precision due to a zero denominator (and 100% in sparse holdouts). This must not be described as unqualified "
        "superiority, and any reported ratios conditional on finite replicates reflect severe sparsity.\n"
        "6. **Velocity Scoring Availability:** The ~50% velocity scoring availability reflects pitch-level scoring "
        "availability (restricted to fastball types FF/SI/FC), not an alert rate, and does not mean half the outings "
        "lack velocity scores. Outing-level coverage is 100% across all qualified test outings.\n"
        "7. **Cohort and Methodology Shifts:** Discrepancies between earlier exploratory findings and the frozen "
        "evaluation cannot be attributed specifically to small-sample luck from this comparison alone. Both cohort "
        "composition and evaluation methodology (operating threshold constraints and holdout definitions) changed.\n"
        "8. **Delivery Deviations vs. Biological Fatigue:** Mechanical feature summaries describe kinematic delivery "
        "deviations, but their actionable coaching value remains unvalidated. They do not establish why a collapse "
        "occurred, cannot diagnose biological fatigue or tissue stress, and do not demonstrate causal mechanisms.\n"
    )

    report = (
        "# Locked retrospective 2025 holdout report\n\n"
        f"- Actual data source: `{metrics.get('actual_data_source', 'unknown')}`\n"
        "- Train: 2023; validation/development and threshold selection: 2024; frozen test: 2025.\n"
        f"- Warehouse coverage: {metrics['data_coverage']['observed_start']} through "
        f"{metrics['data_coverage']['observed_end']}.\n"
        f"- Protocol hash: `{manifest['protocol_sha256']}`.\n"
        f"- Exposure disclosure: {manifest['prior_2025_exposure_disclosure']}\n"
        f"- The >={metrics['minimum_pitches_per_outing']}-pitch outing rule is retrospective and cannot be known at a live pitch.\n"
        "- 2025 baselines update only from qualified outings on strictly earlier dates.\n"
        "- Every model uses the same qualified-outing population and a 2024-selected threshold under "
        f"an upper constraint of {allowance} false warnings per outing; this is not the achieved rate.\n"
        "- Score-unavailable pitches cannot warn and break consecutive-warning runs; follow-up censoring removes warnings from evaluable counts without moving raw starts. Outings and eligible episodes remain in headline denominators.\n"
        "- N/A means a missing denominator; an alert-group risk divided by zero comparison risk is infinite, while absent exposure groups are undefined.\n"
        "- Wider matching windows can mechanically increase recall and are not, alone, evidence of earlier prediction.\n"
        f"- Pitcher-clustered intervals use {metrics.get('test_pitchers_count', 'the test')} pitchers.\n\n"
        + interpretation_section + "\n"
        "## Dataset accounting\n\n" + _markdown(accounting) +
        "\n\n## Pre-test cohort selection\n\n" + _markdown(
            cohort_selection[cohort_selection["selected"]].copy()
            if "selected" in cohort_selection else cohort_selection
        ) +
        "\n\n## Ingestion segment audit\n\n" + _markdown(ingestion_segments) +
        "\n\n## Excluded outings by reason\n\n" + _markdown(exclusion_summary, "No outings were excluded.") +
        "\n\n## Unavailable scores by reason\n\n" + _markdown(unavailable_summary) +
        "\n\n## Missing-data summary\n\n" + _markdown(missingness) +
        "\n\n## Warehouse integrity\n\n" + _markdown(integrity) +
        "\n\n## Frozen 2025 model comparison\n\n" + _markdown(comparison) +
        "\n\n## Evaluation population, coverage, and headline denominators\n\n" + _markdown(evaluation_coverage) +
        "\n\n## Frozen 2025 results by pitcher\n\n" + _markdown(pitcher_evaluation) +
        "\n\n## Paired bootstrap confidence intervals\n\n" + _markdown(bootstrap) +
        "\n\n## Fixed-warning lead-time sensitivity\n\n" + _markdown(lead_time) +
        "\n\n## Historical July-December 2024 experiment\n\n" + _markdown(historical) +
        "\n\n## Feature ablations\n\n" + _markdown(ablations) +
        "\n\n## Sensitivity reruns\n\n" + _markdown(sensitivity) +
        "\n\n## Traceable case-study index\n\n" + _markdown(case_index)
    )
    report_path = output_dir / "validation_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path
