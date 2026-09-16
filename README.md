# MLB Pitcher Mechanics Stability Index

This project tests whether pitch-level mechanical drift is associated with a collapse episode in the next 15 pitches. The Mechanics Stability Index (MSI) is an inverse transform of Mahalanobis distance from a pitcher- and pitch-type-specific historical baseline. It is not a direct measure of fatigue, and this observational analysis does not establish causality.

## Frozen evaluation protocol

Real-data runs retrieve regular-season (`game_type = R`) Statcast pitches for the configured cohort and seasons. The primary experiment is:

| Partition | Dates | Use |
|---|---|---|
| Train | 2023 | Fit the contextual model; construct prior-only mechanics baselines |
| Validation/development | 2024 | Select feature rules, hyperparameters, and one operating threshold per model |
| Frozen test | 2025 | Report final comparisons using the frozen 2024 thresholds |

The prior experiment—2023 training, January–June 2024 validation, and July–December 2024 testing—is retained separately as `historical_2024_model_comparison.csv`. It is never combined with the primary 2025 result.

Important disclosure: exploratory 2025 metrics were generated in this workspace before the revised protocol was adopted. The current protocol is locked and leakage-safe in code, but the 2025 result should be described as a locked retrospective holdout evaluation, not a pristine first look. The generated `protocol_manifest.json` records this limitation, the frozen settings, thresholds, and a SHA-256 protocol hash.

## Eligibility and live-use limitation

The 2025 pitcher cohort is fixed using qualified outings completed by December 31, 2024. A pitcher needs 10 pre-test qualified starts. Starts completed during 2025 cannot determine cohort inclusion.

The centralized settings in `config/config.yaml` are:

- at least 50 pitches in a qualified outing;
- 10 pre-test qualified starts for cohort membership;
- at least five completed prior starts before mechanics scoring;
- a rolling historical window of at most 12 prior starts; and
- at least 40 complete observations for a pitcher × pitch-type baseline.

The minimum prior starts is validated not to exceed the historical window. Cohort eligibility is separate from outing eligibility and pitch-type baseline sufficiency. Excluded pitchers, excluded outings, and unavailable score reasons are exported.

The ≥50-pitch outing filter is retrospective: final outing length is unknown during live prediction. Therefore, reported results apply to retrospectively qualified starter outings and do not directly establish live performance for short starts, openers, or relief appearances.

## Mechanics baselines and alert isolation

Means and covariance matrices are estimated separately for each pitcher × pitch type before each outing. Any pitch type with enough completed historical observations can be scored; the `is_primary_pitch_type` descriptor does not gate scoring. During 2025, a baseline may update only with qualified outings on strictly earlier dates. Same-day and future outings are excluded from the baseline cutoff.

Pitches 1–20 are calibration-only. They receive no Mahalanobis score, MSI, CUSUM update, or warning. Missing features, insufficient prior starts, insufficient pitch-type observations, and insufficient calibration are explicit unavailable states. The same `score_available` policy is used by detection, evaluation, ablation, and display.

Behavioral tests verify that:

- every baseline source date precedes its scoring cutoff;
- changing future games cannot change earlier baselines, scores, or warnings;
- changing outcome-label columns cannot change mechanics scores or warning times;
- pitches 1–20 cannot score or update CUSUM;
- unavailable scores remain excluded;
- 2025 data cannot affect 2023 training or 2024 threshold selection; and
- changing pitch-selection proportions while holding within-type mechanics stable does not unexpectedly change the combined CUSUM warning rate.

## Fair model comparison

MSI+CUSUM, the contextual model, pitch count, and pitch-type-specific velocity drop use the same eligible observations and one-to-one warning–episode matcher. Consecutive flagged pitches form one warning. A warning matches at most one episode when it occurs 1–15 pitches before onset.

Every model chooses its threshold on 2024 under an upper bound of 0.5 false warnings per outing—approximately one unmatched warning every two starts. The selected operating point is distinct from the allowance itself. Every validation candidate is exported and plotted, and the frozen operating point is shown on 2025 without retuning.

The velocity benchmark calculates four-seamer (`FF`), sinker (`SI`), and cutter (`FC`) drops independently and assigns scores by row index. A velocity risk ratio below one is not evidence that velocity is a lagging indicator.

## Uncertainty and lead time

Paired bootstrap intervals resample complete 2025 outings and use the identical sampled outings for every model. Frozen thresholds are never reselected within bootstrap samples. The output reports 95% intervals for episode recall, warning precision, false warnings per outing, risk ratio, and direct proposed-minus-comparator differences. Zero-denominator replicate frequency is reported explicitly.

A pitcher-clustered sensitivity analysis resamples pitchers and includes all their outings. Because the cohort contains few pitchers, those intervals can be unstable and should be interpreted cautiously.

The fixed-warning lead-time experiment holds scores, thresholds, and warning times constant while changing only the match horizon across 10, 15, 20, and 25 pitches. It reports follow-up coverage, censoring, a common-follow-up comparison, matched warning–episode records, and lead-time distributions. Improved recall under a wider window alone is not proof of earlier predictive value.

## Provenance and warehouse integrity

Every persisted result records its actual source (`mlb_statcast` or `simulation_benchmark`). Real-data retrieval is strict: any failed or empty pitcher-season segment aborts publication. Simulation is available only through explicit `use_real_data=False` mode.

Before publication, the pipeline checks requested seasons, source consistency, regular-season scope, pitch-key uniqueness, raw-to-qualified referential integrity, gold fact alignment, and 2025 test assignment. DuckDB tables and Parquet files are replaced atomically and reloaded for a second audit.

Real and simulation evidence are written separately:

- `outputs/real_data/`
- `outputs/simulation/`

The real-data directory contains the protocol manifest, metrics, model comparisons, paired confidence intervals, threshold candidates and plot, lead-time analyses, exclusions, missing-data summary, case evidence, ablations, sensitivity reruns, and warehouse audit.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -v
python scripts/run_full_pipeline.py
python scripts/audit_warehouse.py
streamlit run dashboard/app.py
```

For an explicitly synthetic run:

```python
from scripts.run_full_pipeline import run_pipeline
run_pipeline(use_real_data=False)
```

## Main modules

- `src/configuration.py`: centralized configuration validation
- `src/data_ingest/`: strict Statcast retrieval, caching, and pre-test cohort qualification
- `src/baseline_builder/historical_baseline.py`: pitcher × pitch-type, pre-outing baselines
- `src/evaluation/protocol.py`: splits, eligibility, event matching, and threshold curves
- `src/evaluation/baseline_comparator.py`: frozen-threshold model comparison
- `src/evaluation/bootstrap.py`: paired outing and pitcher-clustered confidence intervals
- `src/evaluation/lead_time.py`: fixed-warning horizon sensitivity
- `scripts/run_full_pipeline.py`: canonical local pipeline
- `scripts/run_cloud_elt.py`: canonical pipeline plus cloud publication
