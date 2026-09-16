# Frozen 2025 temporal holdout report

- Actual data source: `mlb_statcast`
- Train: 2023; validation/development and threshold selection: 2024; frozen test: 2025.
- Warehouse coverage: 2023-03-30 through 2025-09-28.
- Protocol hash: `a02d63cfc72e614b9a93a959b9b2eb51ce48a4689937c0792b297bb5c8dfb62d`.
- Exposure disclosure: Exploratory 2025 metrics were generated in this workspace before this revised protocol. The current result is therefore a locked retrospective holdout evaluation, not a pristine first look.
- The >=50-pitch outing rule is retrospective and cannot be known at a live pitch.
- 2025 baselines update only from qualified outings on strictly earlier dates.
- Every model uses the same eligible observations and a 2024-selected threshold under the 0.5 false-warnings-per-outing ceiling.
- Wider matching windows can mechanically increase recall and are not, alone, evidence of earlier prediction.
- Pitcher-clustered intervals use six pitchers and should be interpreted cautiously.

## Dataset accounting

|   cohort_pitchers |   qualified_outings_all_seasons |   qualified_pitches_all_seasons |   test_evaluated_outings |   test_evaluated_pitches |   test_episodes |   test_warnings |   excluded_pitchers |   excluded_outings |   unavailable_pitch_scores | actual_data_source   |
|------------------:|--------------------------------:|--------------------------------:|-------------------------:|-------------------------:|----------------:|----------------:|--------------------:|-------------------:|---------------------------:|:---------------------|
|                 6 |                             501 |                           48135 |                      138 |                     9172 |             219 |              97 |                   0 |                  7 |                      12660 | mlb_statcast         |

## Excluded outings by reason

| outing_qualification_reason   |   outing_count |
|:------------------------------|---------------:|
| EXCESS_MISSING_SPIN_AXIS      |              7 |

## Unavailable scores by reason

| score_status                         |   pitch_count |
|:-------------------------------------|--------------:|
| CALIBRATION_PHASE                    |         10020 |
| INSUFFICIENT_HISTORY                 |          2217 |
| INSUFFICIENT_PITCH_TYPE_OBSERVATIONS |           387 |
| MISSING_FEATURES                     |            36 |

## Missing-data summary

|   season | column            |   row_count |   missing_count |   missing_fraction | actual_data_source   |
|---------:|:------------------|------------:|----------------:|-------------------:|:---------------------|
|     2023 | release_speed     |       17260 |              18 |        0.00104287  | mlb_statcast         |
|     2023 | release_pos_x     |       17260 |              18 |        0.00104287  | mlb_statcast         |
|     2023 | release_pos_z     |       17260 |              18 |        0.00104287  | mlb_statcast         |
|     2023 | release_extension |       17260 |              25 |        0.00144844  | mlb_statcast         |
|     2023 | release_spin_rate |       17260 |             107 |        0.0061993   | mlb_statcast         |
|     2023 | spin_axis         |       17260 |             107 |        0.0061993   | mlb_statcast         |
|     2023 | pfx_x             |       17260 |              18 |        0.00104287  | mlb_statcast         |
|     2023 | pfx_z             |       17260 |              18 |        0.00104287  | mlb_statcast         |
|     2024 | release_speed     |       18312 |             109 |        0.00595238  | mlb_statcast         |
|     2024 | release_pos_x     |       18312 |             109 |        0.00595238  | mlb_statcast         |
|     2024 | release_pos_z     |       18312 |             109 |        0.00595238  | mlb_statcast         |
|     2024 | release_extension |       18312 |             114 |        0.00622543  | mlb_statcast         |
|     2024 | release_spin_rate |       18312 |             118 |        0.00644386  | mlb_statcast         |
|     2024 | spin_axis         |       18312 |             118 |        0.00644386  | mlb_statcast         |
|     2024 | pfx_x             |       18312 |             109 |        0.00595238  | mlb_statcast         |
|     2024 | pfx_z             |       18312 |             109 |        0.00595238  | mlb_statcast         |
|     2025 | release_speed     |       13323 |               9 |        0.000675524 | mlb_statcast         |
|     2025 | release_pos_x     |       13323 |               9 |        0.000675524 | mlb_statcast         |
|     2025 | release_pos_z     |       13323 |               9 |        0.000675524 | mlb_statcast         |
|     2025 | release_extension |       13323 |              10 |        0.000750582 | mlb_statcast         |
|     2025 | release_spin_rate |       13323 |              12 |        0.000900698 | mlb_statcast         |
|     2025 | spin_axis         |       13323 |              12 |        0.000900698 | mlb_statcast         |
|     2025 | pfx_x             |       13323 |               9 |        0.000675524 | mlb_statcast         |
|     2025 | pfx_z             |       13323 |               9 |        0.000675524 | mlb_statcast         |

## Warehouse integrity

| check_name                          | status   |   row_count | details                                         | actual_data_source   | checked_at_utc                   |
|:------------------------------------|:---------|------------:|:------------------------------------------------|:---------------------|:---------------------------------|
| raw_season_2023                     | PASS     |       17260 | 2023-03-30 00:00:00 through 2023-09-28 00:00:00 | mlb_statcast         | 2026-09-16T20:38:49.918133+00:00 |
| raw_season_2024                     | PASS     |       18312 | 2024-03-28 00:00:00 through 2024-09-29 00:00:00 | mlb_statcast         | 2026-09-16T20:38:49.918133+00:00 |
| raw_season_2025                     | PASS     |       13323 | 2025-03-27 00:00:00 through 2025-09-28 00:00:00 | mlb_statcast         | 2026-09-16T20:38:49.918133+00:00 |
| raw_pitch_key_uniqueness            | PASS     |       48895 | zero duplicate pitch keys                       | mlb_statcast         | 2026-09-16T20:38:49.918133+00:00 |
| gold_score_label_alignment          | PASS     |       48135 | identical ordered pitch keys                    | mlb_statcast         | 2026-09-16T20:38:49.918133+00:00 |
| qualified_raw_referential_integrity | PASS     |       48135 | all qualified keys exist in raw                 | mlb_statcast         | 2026-09-16T20:38:49.918133+00:00 |
| game_scope                          | PASS     |       48895 | MLB regular season only                         | mlb_statcast         | 2026-09-16T20:38:49.918133+00:00 |
| duckdb_reload                       | PASS     |       48895 | all production tables reloaded and revalidated  | mlb_statcast         | 2026-09-16T20:38:49.933886+00:00 |

## Frozen 2025 model comparison

| Model / System                                 |   Test Episode Recall |   Test Warning Precision |   Test False Warnings / Outing |   Test Clean Outing FAR |   Test Pitch PR-AUC |   Test Risk Ratio (Alert vs No Alert) |   Validation-Selected Threshold |   Validation False Warnings / Outing | Actual Data Source   | Interpretation                                                                       | Experiment               |
|:-----------------------------------------------|----------------------:|-------------------------:|-------------------------------:|------------------------:|--------------------:|--------------------------------------:|--------------------------------:|-------------------------------------:|:---------------------|:-------------------------------------------------------------------------------------|:-------------------------|
| Proposed Micro-Mechanics (CUSUM + MSI)         |                 0.192 |                    0.433 |                          0.399 |                   0.5   |               0.355 |                                  1.45 |                        214.651  |                                0.365 | mlb_statcast         | Mechanical-drift score; association is not evidence of fatigue causality.            | Primary frozen 2025 test |
| Contextual Model (Pitch Count + TTO + Inning)  |                 0.146 |                    0.305 |                          0.529 |                   0.273 |               0.316 |                                  1.09 |                          0.5385 |                                0.455 | mlb_statcast         | Fit on 2023 only; threshold selected on 2024 validation data.                        | Primary frozen 2025 test |
| Pitch-Type Velocity Drop (FF/SI/FC separately) |                 0     |                    0     |                          0     |                   0     |               0.315 |                                  0    |                        nan      |                                0     | mlb_statcast         | A contemporaneous velocity benchmark; relative risk alone does not establish timing. | Primary frozen 2025 test |
| Traditional Pitch Count                        |                 0.137 |                    0.4   |                          0.326 |                   0.273 |               0.328 |                                  1.13 |                         86      |                                0.349 | mlb_statcast         | A workload heuristic with the same validation selection rule.                        | Primary frozen 2025 test |

## Paired bootstrap confidence intervals

| resampling_unit   | comparison                 | metric                       |    estimate |   ci_lower_95 |   ci_upper_95 |   zero_denominator_frequency |   bootstrap_samples |   cluster_count | actual_data_source   |
|:------------------|:---------------------------|:-----------------------------|------------:|--------------:|--------------:|-----------------------------:|--------------------:|----------------:|:---------------------|
| outing            | proposed                   | episode_recall               |   0.191781  |    0.150746   |     0.229683  |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | proposed                   | warning_precision            |   0.43299   |    0.333333   |     0.521277  |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | proposed                   | false_warnings_per_outing    |   0.398551  |    0.318841   |     0.478261  |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | proposed                   | risk_ratio_alert_vs_no_alert |   1.45374   |    1.21569    |     1.70684   |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | contextual                 | episode_recall               |   0.146119  |    0.0940141  |     0.196174  |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | contextual                 | warning_precision            |   0.304762  |    0.217822   |     0.388363  |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | contextual                 | false_warnings_per_outing    |   0.528986  |    0.391304   |     0.673913  |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | contextual                 | risk_ratio_alert_vs_no_alert |   1.09446   |    0.793626   |     1.36502   |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | velocity                   | episode_recall               |   0         |    0          |     0         |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | velocity                   | warning_precision            | nan         |  nan          |   nan         |                            1 |                1000 |             138 | mlb_statcast         |
| outing            | velocity                   | false_warnings_per_outing    |   0         |    0          |     0         |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | velocity                   | risk_ratio_alert_vs_no_alert | nan         |  nan          |   nan         |                            1 |                1000 |             138 | mlb_statcast         |
| outing            | pitch_count                | episode_recall               |   0.136986  |    0.0976519  |     0.179726  |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | pitch_count                | warning_precision            |   0.4       |    0.295752   |     0.513167  |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | pitch_count                | false_warnings_per_outing    |   0.326087  |    0.253623   |     0.405797  |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | pitch_count                | risk_ratio_alert_vs_no_alert |   1.13227   |    0.86463    |     1.40633   |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | proposed_minus_contextual  | episode_recall               |   0.0456621 |   -0.00909195 |     0.102442  |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | proposed_minus_contextual  | warning_precision            |   0.128228  |    0.0213544  |     0.234893  |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | proposed_minus_contextual  | false_warnings_per_outing    |  -0.130435  |   -0.28279    |     0.0289855 |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | proposed_minus_contextual  | risk_ratio_alert_vs_no_alert |   0.359284  |    0.035373   |     0.693061  |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | proposed_minus_velocity    | episode_recall               |   0.191781  |    0.150746   |     0.229683  |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | proposed_minus_velocity    | warning_precision            | nan         |  nan          |   nan         |                            1 |                1000 |             138 | mlb_statcast         |
| outing            | proposed_minus_velocity    | false_warnings_per_outing    |   0.398551  |    0.318841   |     0.478261  |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | proposed_minus_velocity    | risk_ratio_alert_vs_no_alert | nan         |  nan          |   nan         |                            1 |                1000 |             138 | mlb_statcast         |
| outing            | proposed_minus_pitch_count | episode_recall               |   0.0547945 |    0.00430421 |     0.105775  |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | proposed_minus_pitch_count | warning_precision            |   0.0329897 |   -0.107001   |     0.148857  |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | proposed_minus_pitch_count | false_warnings_per_outing    |   0.0724638 |   -0.0217391  |     0.166667  |                            0 |                1000 |             138 | mlb_statcast         |
| outing            | proposed_minus_pitch_count | risk_ratio_alert_vs_no_alert |   0.321471  |    0.00238442 |     0.63056   |                            0 |                1000 |             138 | mlb_statcast         |
| pitcher           | proposed                   | episode_recall               |   0.191781  |    0.121457   |     0.25774   |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | proposed                   | warning_precision            |   0.43299   |    0.271318   |     0.592176  |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | proposed                   | false_warnings_per_outing    |   0.398551  |    0.274336   |     0.506757  |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | proposed                   | risk_ratio_alert_vs_no_alert |   1.45374   |    1.15451    |     1.67931   |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | contextual                 | episode_recall               |   0.146119  |    0.118939   |     0.17284   |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | contextual                 | warning_precision            |   0.304762  |    0.255319   |     0.354211  |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | contextual                 | false_warnings_per_outing    |   0.528986  |    0.48715    |     0.557385  |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | contextual                 | risk_ratio_alert_vs_no_alert |   1.09446   |    0.719454   |     1.36174   |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | velocity                   | episode_recall               |   0         |    0          |     0         |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | velocity                   | warning_precision            | nan         |  nan          |   nan         |                            1 |                1000 |               6 | mlb_statcast         |
| pitcher           | velocity                   | false_warnings_per_outing    |   0         |    0          |     0         |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | velocity                   | risk_ratio_alert_vs_no_alert | nan         |  nan          |   nan         |                            1 |                1000 |               6 | mlb_statcast         |
| pitcher           | pitch_count                | episode_recall               |   0.136986  |    0.0847105  |     0.1813    |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | pitch_count                | warning_precision            |   0.4       |    0.272727   |     0.488372  |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | pitch_count                | false_warnings_per_outing    |   0.326087  |    0.274483   |     0.370979  |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | pitch_count                | risk_ratio_alert_vs_no_alert |   1.13227   |    0.764616   |     1.41935   |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | proposed_minus_contextual  | episode_recall               |   0.0456621 |   -0.00746269 |     0.103699  |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | proposed_minus_contextual  | warning_precision            |   0.128228  |    0.00117312 |     0.268866  |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | proposed_minus_contextual  | false_warnings_per_outing    |  -0.130435  |   -0.246004   |    -0.0264231 |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | proposed_minus_contextual  | risk_ratio_alert_vs_no_alert |   0.359284  |    0.160551   |     0.609208  |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | proposed_minus_velocity    | episode_recall               |   0.191781  |    0.121457   |     0.25774   |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | proposed_minus_velocity    | warning_precision            | nan         |  nan          |   nan         |                            1 |                1000 |               6 | mlb_statcast         |
| pitcher           | proposed_minus_velocity    | false_warnings_per_outing    |   0.398551  |    0.274336   |     0.506757  |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | proposed_minus_velocity    | risk_ratio_alert_vs_no_alert | nan         |  nan          |   nan         |                            1 |                1000 |               6 | mlb_statcast         |
| pitcher           | proposed_minus_pitch_count | episode_recall               |   0.0547945 |    0.024046   |     0.105974  |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | proposed_minus_pitch_count | warning_precision            |   0.0329897 |   -0.0521442  |     0.176901  |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | proposed_minus_pitch_count | false_warnings_per_outing    |   0.0724638 |   -0.0666667  |     0.184821  |                            0 |                1000 |               6 | mlb_statcast         |
| pitcher           | proposed_minus_pitch_count | risk_ratio_alert_vs_no_alert |   0.321471  |   -0.0818454  |     0.704742  |                            0 |                1000 |               6 | mlb_statcast         |

## Fixed-warning lead-time sensitivity

| model_key   |   matching_horizon |   frozen_warning_count |   complete_followup_warning_count |   censored_warning_count |   followup_coverage |   episode_count |   matched_episode_count |   episode_recall_complete_followup |   warning_precision_complete_followup |   common_followup_warning_count |   common_followup_episode_count |   common_followup_matched_episode_count |   common_followup_episode_recall | actual_data_source   |
|:------------|-------------------:|-----------------------:|----------------------------------:|-------------------------:|--------------------:|----------------:|------------------------:|-----------------------------------:|--------------------------------------:|--------------------------------:|--------------------------------:|----------------------------------------:|---------------------------------:|:---------------------|
| proposed    |                 10 |                    125 |                                97 |                       28 |           0.776     |             219 |                      23 |                         0.105023   |                              0.237113 |                              34 |                             219 |                                      11 |                       0.0502283  | mlb_statcast         |
| proposed    |                 15 |                    125 |                                77 |                       48 |           0.616     |             219 |                      34 |                         0.155251   |                              0.441558 |                              34 |                             219 |                                      20 |                       0.0913242  | mlb_statcast         |
| proposed    |                 20 |                    125 |                                54 |                       71 |           0.432     |             219 |                      27 |                         0.123288   |                              0.5      |                              34 |                             219 |                                      20 |                       0.0913242  | mlb_statcast         |
| proposed    |                 25 |                    125 |                                34 |                       91 |           0.272     |             219 |                      23 |                         0.105023   |                              0.676471 |                              34 |                             219 |                                      23 |                       0.105023   | mlb_statcast         |
| contextual  |                 10 |                    134 |                               103 |                       31 |           0.768657  |             219 |                      17 |                         0.0776256  |                              0.165049 |                              51 |                             219 |                                       5 |                       0.0228311  | mlb_statcast         |
| contextual  |                 15 |                    134 |                                84 |                       50 |           0.626866  |             219 |                      25 |                         0.114155   |                              0.297619 |                              51 |                             219 |                                      11 |                       0.0502283  | mlb_statcast         |
| contextual  |                 20 |                    134 |                                74 |                       60 |           0.552239  |             219 |                      27 |                         0.123288   |                              0.364865 |                              51 |                             219 |                                      14 |                       0.0639269  | mlb_statcast         |
| contextual  |                 25 |                    134 |                                51 |                       83 |           0.380597  |             219 |                      17 |                         0.0776256  |                              0.333333 |                              51 |                             219 |                                      17 |                       0.0776256  | mlb_statcast         |
| velocity    |                 10 |                      0 |                                 0 |                        0 |         nan         |             219 |                       0 |                         0          |                            nan        |                               0 |                             219 |                                       0 |                       0          | mlb_statcast         |
| velocity    |                 15 |                      0 |                                 0 |                        0 |         nan         |             219 |                       0 |                         0          |                            nan        |                               0 |                             219 |                                       0 |                       0          | mlb_statcast         |
| velocity    |                 20 |                      0 |                                 0 |                        0 |         nan         |             219 |                       0 |                         0          |                            nan        |                               0 |                             219 |                                       0 |                       0          | mlb_statcast         |
| velocity    |                 25 |                      0 |                                 0 |                        0 |         nan         |             219 |                       0 |                         0          |                            nan        |                               0 |                             219 |                                       0 |                       0          | mlb_statcast         |
| pitch_count |                 10 |                    119 |                                83 |                       36 |           0.697479  |             219 |                      22 |                         0.100457   |                              0.26506  |                               2 |                             219 |                                       2 |                       0.00913242 | mlb_statcast         |
| pitch_count |                 15 |                    119 |                                41 |                       78 |           0.344538  |             219 |                      16 |                         0.0730594  |                              0.390244 |                               2 |                             219 |                                       2 |                       0.00913242 | mlb_statcast         |
| pitch_count |                 20 |                    119 |                                17 |                      102 |           0.142857  |             219 |                       8 |                         0.0365297  |                              0.470588 |                               2 |                             219 |                                       2 |                       0.00913242 | mlb_statcast         |
| pitch_count |                 25 |                    119 |                                 2 |                      117 |           0.0168067 |             219 |                       2 |                         0.00913242 |                              1        |                               2 |                             219 |                                       2 |                       0.00913242 | mlb_statcast         |

## Historical July-December 2024 experiment

| Model / System                                 |   Test Episode Recall |   Test Warning Precision |   Test False Warnings / Outing |   Test Clean Outing FAR |   Test Pitch PR-AUC |   Test Risk Ratio (Alert vs No Alert) |   Validation-Selected Threshold |   Validation False Warnings / Outing | Actual Data Source   | Interpretation                                                                       | Experiment                         |
|:-----------------------------------------------|----------------------:|-------------------------:|-------------------------------:|------------------------:|--------------------:|--------------------------------------:|--------------------------------:|-------------------------------------:|:---------------------|:-------------------------------------------------------------------------------------|:-----------------------------------|
| Proposed Micro-Mechanics (CUSUM + MSI)         |                 0.094 |                    0.295 |                          0.348 |                   0.273 |               0.296 |                                  0.64 |                        227.396  |                                 0.25 | mlb_statcast         | Mechanical-drift score; association is not evidence of fatigue causality.            | Historical July-December 2024 test |
| Contextual Model (Pitch Count + TTO + Inning)  |                 0.109 |                    0.273 |                          0.449 |                   0.182 |               0.333 |                                  1.28 |                          0.5405 |                                 0.39 | mlb_statcast         | Fit on 2023 only; threshold selected on 2024 validation data.                        | Historical July-December 2024 test |
| Pitch-Type Velocity Drop (FF/SI/FC separately) |                 0     |                    0     |                          0.011 |                   0     |               0.331 |                                  0    |                          2.9873 |                                 0    | mlb_statcast         | A contemporaneous velocity benchmark; relative risk alone does not establish timing. | Historical July-December 2024 test |
| Traditional Pitch Count                        |                 0.123 |                    0.23  |                          0.64  |                   0.636 |               0.303 |                                  0.81 |                         80      |                                 0.5  | mlb_statcast         | A workload heuristic with the same validation selection rule.                        | Historical July-December 2024 test |

## Feature ablations

| Feature Subset                         |   Test Pitch PR-AUC |   Test Episode Recall |   Test Warning Precision |   Test False Warnings / Outing |   Validation-Selected Threshold |   Validation False Warnings / Outing | Actual Data Source   |
|:---------------------------------------|--------------------:|----------------------:|-------------------------:|-------------------------------:|--------------------------------:|-------------------------------------:|:---------------------|
| Velocity Alone                         |               0.319 |                     0 |                        0 |                              0 |                         nan     |                                    0 | mlb_statcast         |
| Release Point Alone (X, Z, Extension)  |               0.332 |                     0 |                        0 |                              0 |                         nan     |                                    0 | mlb_statcast         |
| Spin & Movement (rate, axis, PFX, VAA) |               0.309 |                     0 |                        0 |                              0 |                         163.239 |                                    0 | mlb_statcast         |
| Full Micro-Mechanics Suite             |               0.338 |                     0 |                        0 |                              0 |                         nan     |                                    0 | mlb_statcast         |

## Sensitivity reruns

|   xwOBA Threshold |   Window (PA) |   Horizon (Pitches) |   Test Episodes |   Test Episode Recall |   Test Warning Precision |   Test False Warnings / Outing |   Validation-Selected Threshold |   Validation False Warnings / Outing | Actual Data Source   |
|------------------:|--------------:|--------------------:|----------------:|----------------------:|-------------------------:|-------------------------------:|--------------------------------:|-------------------------------------:|:---------------------|
|              0.4  |             3 |                  15 |             270 |                 0.2   |                    0.45  |                          0.478 |                         193.664 |                                0.487 | mlb_statcast         |
|              0.45 |             3 |                  15 |             219 |                 0.192 |                    0.433 |                          0.399 |                         214.651 |                                0.365 | mlb_statcast         |
|              0.5  |             3 |                  15 |             190 |                 0.205 |                    0.371 |                          0.478 |                         199.383 |                                0.46  | mlb_statcast         |
|              0.45 |             2 |                  15 |             296 |                 0.186 |                    0.43  |                          0.529 |                         174.14  |                                0.497 | mlb_statcast         |
|              0.45 |             3 |                  10 |             219 |                 0.132 |                    0.296 |                          0.5   |                         222.777 |                                0.444 | mlb_statcast         |
|              0.45 |             3 |                  20 |             219 |                 0.233 |                    0.418 |                          0.514 |                         164.639 |                                0.476 | mlb_statcast         |
|              0.45 |             3 |                  25 |             219 |                 0.233 |                    0.386 |                          0.587 |                         119.578 |                                0.429 | mlb_statcast         |

## Traceable case-study index

| case_type   |   game_pk | game_date   |   pitcher | pitcher_name   | actual_data_source   |   warning_pitch |   episode_onset_pitch |   lead_time_pitches |
|:------------|----------:|:------------|----------:|:---------------|:---------------------|----------------:|----------------------:|--------------------:|
| TP          |    776217 | 2025-09-23  |    592332 | Kevin Gausman  | mlb_statcast         |              80 |                    82 |                   2 |
| FP          |    776211 | 2025-09-23  |    657277 | Logan Webb     | mlb_statcast         |              82 |                   nan |                 nan |
| FN          |    776145 | 2025-09-28  |    592332 | Kevin Gausman  | mlb_statcast         |             nan |                     4 |                 nan |
| TN          |    776137 | 2025-09-28  |    657277 | Logan Webb     | mlb_statcast         |             nan |                   nan |                 nan |