# MLB Pitcher Mechanics Stability Index

This project tests whether pitch-level mechanical drift is associated with a collapse episode in the next 15 pitches. The Mechanics Stability Index (MSI) is an inverse transform of a pitcher's Mahalanobis distance from a personal historical baseline. It is not a direct measure of fatigue and the analysis does not establish causality.

## Reproducible evaluation design

The pipeline downloads the selected pitchers from March 30, 2023 through September 30, 2024 and assigns each pitch to exactly one temporal partition:

| Partition | Dates | Use |
|---|---|---|
| Train | through 2023-12-31 | Fit the contextual logistic regression; build prior-only rolling baselines |
| Validation | 2024-01-01 through 2024-06-30 | Select an operating threshold for every model |
| Test | from 2024-07-01 | Report held-out comparison metrics |

Every model uses the same eligible test pitches, the same one-to-one episode matcher, and a threshold selected on validation under a maximum of 0.5 distinct false warnings per outing. A warning matches an episode only if it occurs 1–15 pitches before that episode's onset. Consecutive flagged pitches are one warning, not multiple false alarms.

The contextual model is fit on 2023 only. Its early-2024 predictions select its threshold; its late-2024 predictions are held out. The velocity benchmark computes its calibration mean and rolling drop separately for four-seamers (`FF`), sinkers (`SI`), and cutters (`FC`) and assigns results by DataFrame index, making it invariant to input row order.

## Scoring availability

Pitches 1–20 are calibration-only. They receive no Mahalanobis score, MSI, CUSUM update, or alert. Pitches without sufficient prior history or complete mechanics features also receive no numeric score. `score_status` explains why and `score_available` is the common gate used by detection, evaluation, ablation, and display.

Outings under 50 pitches are excluded by design. Results therefore describe qualified starter outings and must not be generalized to openers, relief appearances, or short starts without another study. Right-censored forecast rows are excluded consistently.

## Outcomes and metrics

A collapse episode is a contiguous sequence of adverse PA windows. A window is adverse when at least one configured rule is met: blended xwOBA at or above 0.450, at least two barrels, or at least two walks/HBP. The label table includes `window_blended_xwoba` and `collapse_reason` so plots and the dashboard use the actual label schema.

The comparison table reports:

- episode recall: matched episodes divided by evaluable episodes;
- warning precision: matched distinct warnings divided by all distinct warnings;
- false warnings per outing and clean-outing false-alarm rate;
- pitch PR-AUC, explicitly identified as a pitch-level metric; and
- risk ratio comparing alert observations with no-alert observations.

A risk ratio below 1 for velocity is not proof that velocity is a lagging indicator. Timing or causal claims require a dedicated lead-lag analysis.

## Provenance

Every persisted pitch, warning, comparison row, sensitivity result, report, and case-study index records the actual source (`mlb_statcast` or `simulation_benchmark`). If Statcast returns no data, the pipeline falls back to simulation but continues to label the run as simulation; it never reports that fallback as MLB evidence. The cloud entry point calls the same canonical pipeline before publishing its tables.

## Ablations and sensitivity

Feature ablations use the same eligible late-2024 population as the main evaluation. “Spin & Movement” includes spin rate, circular spin-axis coordinates, horizontal/vertical movement, and VAA. Sensitivity analysis rebuilds labels and reruns validation threshold selection plus held-out event matching for every xwOBA, PA-window, and forecast-horizon setting.

## Traceable case studies

Cases are selected from held-out test outings using the same timely matcher as the metrics. `outputs/case_studies/case_study_index.csv` records game ID, date, pitcher, actual data source, warning pitch, episode onset, and measured lead time. Plot captions are generated from those fields. No grip-change, mound-visit, or pitch-causation explanation is inferred from Statcast alone.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -v
python scripts/run_full_pipeline.py
streamlit run dashboard/app.py
```

For an explicitly synthetic run:

```python
from scripts.run_full_pipeline import run_pipeline
run_pipeline(use_real_data=False)
```

Generated, source-specific results are written to `outputs/metrics_summary.json`, `outputs/validation_report.md`, `outputs/ablation_results.csv`, `outputs/sensitivity_results.csv`, and `outputs/case_studies/`. Committed or previously generated artifacts should not be treated as current evidence until the pipeline has been rerun.

## Main modules

- `scripts/run_full_pipeline.py`: canonical local pipeline
- `scripts/run_cloud_elt.py`: canonical pipeline plus GCS/BigQuery publishing
- `src/evaluation/protocol.py`: splits, eligibility, warning construction, matching, threshold selection
- `src/evaluation/baseline_comparator.py`: held-out model comparison
- `src/evaluation/ablation_runner.py`: executable ablation and sensitivity runs
- `src/anomaly_scorer/mahalanobis_scorer.py`: explicit score availability
- `src/changepoint_detector/`: calibration-isolated detectors
