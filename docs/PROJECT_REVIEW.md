# Project review — 2026-09-24

## Implementation update — 2026-09-25

The findings below describe commit `231fe24` before repair. Current source changes restore the offline pipeline, model-specific evaluation masks, release coverage and identity, complete config/source provenance, strict ablation parity, safe refresh entry points, missing-measurement handling, separate simulation storage, and generated report/dashboard wording. Case plots now use evaluated warning and episode records. Regression tests and a GitHub Actions test workflow were added.

The prior published real-data directory contains a stale ablation manifest from another protocol. The stronger validator intentionally rejects that directory. A complete real-data rerun is needed before the current source can publish a verified real-data release. This update did not fetch Statcast data or run a cloud job.

Remaining work includes coordinating warehouse and output publication as one release pointer, auditing cloud publication, pinning a reproducible dependency set, and redesigning CUSUM reference calibration as a separate research experiment. The earlier 2025 exposure disclosure must remain attached to any rerun.

Reviewed commit: `231fe24` (`main`). Compared the merge against both parents, `f72ecd4` and `684f76f`.

## Assessment

The project has useful foundations: deterministic selection using pre-test records, temporal separation, pitcher/pitch-type baselines using earlier outings, an explicit common evaluation population, paired bootstrap calculations, and warehouse integrity checks. The saved primary results and their caveats are useful research evidence.

At reviewed commit `231fe24`, the source could not complete its production pipeline. Several integration repairs from `f72ecd4` were lost during the merge. The previous claim that the merge preserved all functionality and solved the project was incorrect. Including a commit in Git history does not establish that its behavior or tests survived conflict resolution.

The immediate priority is a coherent, reproducible release. Further modeling work should follow that repair and retain the disclosure that 2025 has already been examined.

## Verification and scope

Reviewed ingestion, cohort selection, qualification, feature engineering, historical baselines, calibration, scoring, detectors, labels, evaluation, bootstrap, ablations, manifests, storage, refresh scripts, reporting, dashboard, deployment scripts, and tests. Inspected local artifacts and sampled/calculated statistics from the saved score table.

| Check | Result |
| --- | --- |
| `python -m pytest tests -q -p no:cacheprovider` | **38 passed, 1 failed**, 18.05 seconds. The production smoke test fails with `NameError: subprocess is not defined`. |
| Smoke test with only the missing import supplied in memory | **Failed again**, at the historical evaluation's obsolete configuration key. Source files were not changed for this diagnostic. |
| `python scripts/validate_artifacts.py outputs/real_data` | Passes for run `08a3a0b1ccde4ff4a59b9a36690cd1a3`, but the validator omits checks described below. |
| `python scripts/audit_warehouse.py` | Passes; zero duplicate raw pitch keys, zero qualified orphans, expected seasons/splits and source columns. |
| Small model-availability counterexample | Headline contextual recall **1.0**; bootstrap with the current pipeline's calling convention **0.0**; supplying the proper mask restores **1.0**. |
| Ablation parity with all numeric results missing | Incorrectly accepted as valid against finite expected metrics. |
| Model comparison with no velocity-eligible pitch types | Reproduced `IndexError` at `baseline_comparator.py:205`. |
| Feature probes | Missing spin/kinematics become finite features; identical input produces different fabricated pitch intervals. |

No production source was edited, no release was regenerated, and no cloud jobs were executed during this review. Passing checks on saved artifacts do not establish that current source can reproduce those artifacts.

## 1. High priority: restore an executable production pipeline

**Evidence:** `scripts/run_full_pipeline.py:227`, `:497`, `:556`, `:563`, `:617`; `src/configuration.py:16`.

`_git_state()` calls `subprocess.run()` without importing `subprocess`. Its exception handler also references the missing name. The full smoke test reaches this function and fails.

After supplying that import in memory, execution reaches another failure: historical evaluation requests `evaluation.prediction_horizon_pitches`. The configuration now uses `evaluation.warning_matching_horizon_pitches`, and the loader explicitly rejects the old key. Bootstrap and per-pitcher evaluation contain the same obsolete lookup. The ablation refresh script also requests the old CUSUM key.

**Improve:** reconcile every caller with one canonical configuration interface. Run the actual production smoke test through publication before declaring the merge complete. Do not add permissive defaults that hide stale callers.

## 2. High priority: restore the complete release producer/validator contract

**Evidence:** `scripts/run_full_pipeline.py:511`, `:681`, `:700`; `src/presentation.py:64`; `src/evaluation/ablation_runner.py:375`.

The pipeline no longer emits `evaluation_coverage.csv`, which the validator unconditionally reads. It also omits `run_id` from the metrics summary and does not stamp derived CSVs with run/protocol IDs. The validator requires all of those fields.

The integrated ablation runner emits a different table schema: it lacks the required `status="verified"` and `Test Qualified Outings` fields and rounds metrics/thresholds before exporting them. The release validator expects the original schema and unrounded parity at an absolute tolerance of `1e-12`. Fixing the first runtime exception will therefore not make publication succeed.

**Improve:** define a single release schema shared by producers, validators, report, and dashboard. Retain unrounded scientific values and round only for display. Generate coverage and all identity columns through the canonical pipeline. Require the offline smoke test to produce a complete report and pass the real validator.

## 3. High priority: use each model's own availability throughout evaluation

**Evidence:** `scripts/run_full_pipeline.py:154`, `:554`, `:571`; `src/evaluation/baseline_comparator.py:184`; `src/evaluation/bootstrap.py:53`.

The headline comparator supplies each model's finite-score mask. The pipeline's bootstrap, lead-time, and per-pitcher calls omit that information. Those paths fall back to the MSI `score_available` column, which can suppress valid contextual, pitch-count, or velocity warnings.

A deterministic two-outing example demonstrated the consequence: a contextual warning in an MSI-unavailable outing detects the only episode in the headline evaluation, but disappears from bootstrap evaluation. Recall changes from 1.0 to 0.0. Bootstrap's internal agreement check does not catch this because its ordinary evaluation is invoked with the same incorrect default mask.

**Improve:** define the model-to-score mapping once and pass it to every consumer. Assert agreement with the actual headline metrics for every model, including different availability patterns, rather than only comparing two calculations using the same default.

## 4. High priority: make provenance complete and reject stale companion artifacts

**Evidence:** `scripts/run_full_pipeline.py:468`; `src/presentation.py:64`; `outputs/real_data/ablation_manifest.json`.

The saved primary manifest has protocol hash `43822a20a6ac13b4360f1a9c82cb8dad93de281e7cef8eb4dfef3a8fd1195b68` and threshold approximately **211.9586**. The ablation manifest has parent hash `a02d63cfc72e614b9a93a959b9b2eb51ce48a4689937c0792b297bb5c8dfb62d` and threshold **214.651**. Both are in the same published directory. The validator passes because it does not read the ablation JSON manifest.

The current manifest producer also dropped split dates, label/matching horizons, MSI alpha, several qualification/baseline settings, random seeds, bootstrap settings, and the dirty-source patch digest. Consequently the protocol hash no longer describes the full experiment. The local gold score Parquet has neither `run_id` nor `protocol_sha256`.

**Improve:** restore the complete resolved runtime record, source snapshot identity, and warehouse identity. Recompute the protocol hash during validation. Verify every companion JSON and case-study identity, along with derived tables, against the same release. Clearly archive stale artifacts instead of leaving them beside current evidence.

## 5. High priority: repair ablation verification and experimental comparability

**Evidence:** `src/evaluation/ablation_runner.py:92`, `:202`, `:313`, `:422`; `scripts/refresh_ablation_outputs.py:90`.

The full-feature parity helper checks only a few rounded rates and a threshold. Missing values pass because comparisons with NaN never trigger its rejection condition. A probe with all-NaN ablation results and finite expected metrics returned `True`. It does not verify counts, warning identities, coverage, or run identity.

The full-feature runner reselects a threshold instead of accepting the frozen primary threshold and predictions. Subsets use historical means while the primary model uses calibrated means; test evaluation still defaults to full-model availability; and coverage is measured after filtering to already-eligible full-model rows. These differences prevent a clean interpretation of what removing a feature group changed.

When baseline data are absent, it substitutes Euclidean z-score distances for the intended covariance-based distances. The refresh script can also copy primary metrics into a reference row and then describe that comparison as verification. Its live path passes the pitch-level `fact_collapse_labels` table where the evaluator requires episode-level records with `onset_pitch`.

**Improve:** persist or reconstruct the required baseline vectors/covariances and episode table. Use the frozen primary predictions for full-model parity, validate unrounded counts and warning IDs, handle missing values explicitly, and use each variant's availability. Mark missing experiments unavailable. Preserve the intended scoring/calibration policy or identify a changed method as a separate experiment.

## 6. High priority: strengthen merge verification and automated tests

**Evidence:** parent comparison of `tests/test_pipeline.py`; `tests/test_protocol_contract.py:123`; `tests/test_offline_smoke.py`.

Nine ablation/CUSUM test functions from `684f76f` are absent from the merge. Some behaviors overlap with replacement tests, but the remote ablation-specific tests were not retained as a set. The earlier strict ablation-denominator mismatch test was replaced with a simpler rounded-rate helper test.

The configuration unit test proves that a factory constructs configured objects; it does not prove the production call uses every returned value. That is why it misses the MSI problem below. There is no tracked GitHub Actions workflow, and dependencies have lower bounds without a lock or constrained reproducible environment.

**Improve:** restore/adapt the omitted regression cases; require the full suite and real smoke test on pushes and pull requests; test supported Python environments; lock dependencies; and add a small static undefined-name check. The smoke test should include detected and missed test episodes, because the current synthetic fixture has no test episodes in the observed diagnostic run.

## 7. Medium priority: honor configuration in outputs and generated explanations

**Evidence:** `scripts/run_full_pipeline.py:396`, `:442`, `:492`; `src/anomaly_scorer/health_index.py:17`; `src/reporting.py:72`; `dashboard/app.py:420`.

Configuration says MSI alpha is **0.45**, but the pipeline calls the MSI function without the configured argument, so its default **0.40** is used. Primary/historical split helpers are also called without the configured split parameters. Case generation hardcodes a 15-pitch horizon and does not pass configured visualization parameters.

The report generator embeds results such as 33.3%, 37.2%, -0.01518 and particular confidence limits directly in source. The same claims would appear after a materially different run or a simulation. The dashboard embeds some of these claims too.

**Improve:** pass resolved values explicitly at the actual call sites. Generate narrative claims from the selected run's tables and confidence intervals. Label research limitations separately from measured results, and mark undefined or unsupported comparisons accordingly.

## 8. Medium priority: preserve missing measurements and remove fabricated real-data features

**Evidence:** `src/feature_engineering/mechanics_features.py:26`, `:34`; `src/feature_engineering/rolling_stats.py:78`, `:92`.

Missing spin is converted to zero degrees before circular decomposition; missing kinematics receive typical numeric values before VAA calculation. A probe with all those measurements missing produced finite spin components and VAA of -6.612 degrees. Those transformed values can evade the scorer's missing-feature gate.

When pitch timing is absent, the rolling feature code manufactures intervals using an unseeded random draw. Repeating the same input produced different values. These fields do not currently drive the main Mahalanobis score, but they pollute persisted real-data features and future consumers. The spin “shift” features use ordinary angular standard deviation even though the function describes circular handling.

**Improve:** preserve missingness, explicitly flag any justified imputation, restrict synthetic timing to simulation, and use circular statistics for circular variables. In the saved score table, I found zero scored rows with missing raw spin or the four checked VAA inputs; this audit demonstrates a code-path defect, not measured contamination of those saved scores.

## 9. Medium priority: make storage, refresh, and dashboard consume one release

**Evidence:** `scripts/run_full_pipeline.py:673`; `scripts/refresh_lead_time_outputs.py:60`; `scripts/run_cloud_elt.py:91`; `dashboard/app.py:68`.

Warehouse replacement occurs before final output validation. A later failure can leave the dashboard's warehouse data newer than its report files. Real and simulation runs share the same warehouse location even though output directories are separate. The dashboard always reads the real-data manifest and validates comparison thresholds, without verifying warehouse run IDs or sources. Its argument-free data cache also has no release-based invalidation.

Refresh scripts overwrite published files directly, use live YAML rather than frozen runtime settings, and omit required artifact identities. Cloud publication replaces latest objects/tables individually and skips empty tables, which can retain stale rows from a previous run. The older BigQuery sync script logs a success message even after individual uploads fail.

**Improve:** use immutable run directories/tables with a final validated release pointer, separate real and simulation warehouses, validate identity at every consumer, invalidate caches by release ID, and publish empty tables deliberately. Route refreshes through staging and the same publication gate. Make cloud failure status explicit.

## 10. Medium priority: handle absent models and truthful visual evidence

**Evidence:** `src/evaluation/baseline_comparator.py:205`; `scripts/run_full_pipeline.py:279`; `src/visualization/case_plots.py:75`; `dashboard/app.py:256`.

If a model has no validation scores, its threshold curve is empty, but the comparator indexes its first row. A fixture without FF/SI/FC pitches reproduces an `IndexError`. The no-score/no-alert state should be representable without aborting the other models.

Case classification uses raw warning outings, including censored warnings, for false-positive selection. Plotting then uses the first alert and first collapse independently, even when the evaluator matched a later pair, and shades the remainder of the outing after the first collapse. This can visually disagree with the matched episode record.

The dashboard's “Caution 60” and “Critical 35” MSI lines are fixed display heuristics, and its alert banner recommends warming the bullpen despite the project's stated lack of validated coaching utility.

**Improve:** represent absent models explicitly; select cases from evaluable warning/match records; plot actual episode spans and matched warnings; label heuristic reference lines; and present the interface as retrospective research/replay until intervention value is established.

## Modeling improvements after correctness repairs

The most concrete issue to investigate is CUSUM calibration. With mean 1.0, standard deviation 0.5, and slack 0.5, the update is positive whenever distance exceeds **1.25**. Saved available-score medians are about 3.008 in training and 2.961 in validation. Positive-update fractions are **99.68%** and **99.58%**, respectively. Test has a similar fraction, **99.67%**, and pooled CUSUM/pitch-count correlation is approximately **0.879**.

These are descriptive diagnostics. They suggest the accumulated score may strongly track workload, but do not prove the mechanism behind predictive performance. Estimate reference behavior using development data and evaluate added value beyond context/workload. Treat a revised detector as a new experiment; avoid repeated optimization against the already-inspected 2025 period.

The saved primary evaluation reports 153 detected episodes out of 1,581 (**9.68% recall**), 460 warnings, and 307 unmatched warnings (**33.26% precision**). Its recall difference from the contextual baseline has a confidence interval spanning zero. Prioritize evidence of incremental utility, workload burden, heterogeneity across pitchers, and independently evaluated operating choices before adding a more complex model or live coaching claims.

For runtime, measure stages first. The comparator evaluates the same validation candidate grid in threshold selection and again for the tradeoff table. Ablation and sensitivity paths perform repeated large DataFrame copies, per-row iteration, and relabeling. Reuse candidate results and prepared outing data while preserving frozen predictions and deterministic matching. Add resumable stages identified by input/config/source hashes.

## Suggested implementation order

1. Restore the complete configuration, availability, manifest, and output contracts lost in the merge; retain both parents' relevant regression coverage.
2. Require a successful isolated production smoke test and strengthen artifact checks for stale/mismatched manifests and model-specific denominators.
3. Repair ablation/refresh paths and generated report/dashboard claims; isolate simulation and coordinate publication.
4. Regenerate a complete release from approved cached data after these gates pass, documenting any changed metrics and remaining unavailable experiments.
5. Profile runtime and design the next detector-calibration/incremental-value experiment using development data and a separately specified evaluation.
