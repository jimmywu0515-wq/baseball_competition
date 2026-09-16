# Temporal holdout validation report

- Requested mode: `mlb_statcast`
- Actual data source: `mlb_statcast`
- Train: 2023; threshold validation: January-June 2024; held-out test: July 2024 through September 2025.
- Warehouse coverage: 2023-03-30 through 2025-09-28.
- Every model uses the same eligible pitches, one-to-one 15-pitch event matching, and a threshold selected under the same validation allowance of 0.5 distinct false warnings per outing.

## Warehouse integrity

| check_name                          | status   |   row_count | details                                         | actual_data_source   | checked_at_utc                   |
|:------------------------------------|:---------|------------:|:------------------------------------------------|:---------------------|:---------------------------------|
| raw_season_2023                     | PASS     |       17260 | 2023-03-30 00:00:00 through 2023-09-28 00:00:00 | mlb_statcast         | 2026-09-16T16:21:03.790242+00:00 |
| raw_season_2024                     | PASS     |       18312 | 2024-03-28 00:00:00 through 2024-09-29 00:00:00 | mlb_statcast         | 2026-09-16T16:21:03.790242+00:00 |
| raw_season_2025                     | PASS     |       13323 | 2025-03-27 00:00:00 through 2025-09-28 00:00:00 | mlb_statcast         | 2026-09-16T16:21:03.790242+00:00 |
| raw_pitch_key_uniqueness            | PASS     |       48895 | zero duplicate pitch keys                       | mlb_statcast         | 2026-09-16T16:21:03.790242+00:00 |
| gold_score_label_alignment          | PASS     |       48135 | identical ordered pitch keys                    | mlb_statcast         | 2026-09-16T16:21:03.790242+00:00 |
| qualified_raw_referential_integrity | PASS     |       48135 | all qualified keys exist in raw                 | mlb_statcast         | 2026-09-16T16:21:03.790242+00:00 |
| game_scope                          | PASS     |       48895 | MLB regular season only                         | mlb_statcast         | 2026-09-16T16:21:03.790242+00:00 |
| duckdb_reload                       | PASS     |       48895 | all production tables reloaded and revalidated  | mlb_statcast         | 2026-09-16T16:21:03.802895+00:00 |

## Held-out model comparison

| Model / System                                 |   Test Episode Recall |   Test Warning Precision |   Test False Warnings / Outing |   Test Clean Outing FAR |   Test Pitch PR-AUC |   Test Risk Ratio (Alert vs No Alert) |   Validation-Selected Threshold |   Validation False Warnings / Outing | Actual Data Source   | Interpretation                                                                       |
|:-----------------------------------------------|----------------------:|-------------------------:|-------------------------------:|------------------------:|--------------------:|--------------------------------------:|--------------------------------:|-------------------------------------:|:---------------------|:-------------------------------------------------------------------------------------|
| Proposed Micro-Mechanics (CUSUM + MSI)         |                 0.148 |                    0.414 |                          0.33  |                   0.394 |               0.329 |                                  1.15 |                        227.396  |                                 0.25 | mlb_statcast         | Mechanical-drift score; association is not evidence of fatigue causality.            |
| Contextual Model (Pitch Count + TTO + Inning)  |                 0.137 |                    0.312 |                          0.476 |                   0.242 |               0.322 |                                  1.14 |                          0.5405 |                                 0.39 | mlb_statcast         | Fit on 2023 only; threshold selected on early 2024.                                  |
| Pitch-Type Velocity Drop (FF/SI/FC separately) |                 0     |                    0     |                          0.031 |                   0     |               0.321 |                                  0    |                          2.9873 |                                 0    | mlb_statcast         | A contemporaneous velocity benchmark; relative risk alone does not establish timing. |
| Traditional Pitch Count                        |                 0.16  |                    0.317 |                          0.542 |                   0.606 |               0.318 |                                  1.05 |                         80      |                                 0.5  | mlb_statcast         | A workload heuristic with the same validation selection rule.                        |

## Feature ablations

| Feature Subset                         |   Test Pitch PR-AUC |   Test Episode Recall |   Test Warning Precision |   Test False Warnings / Outing |   Validation-Selected Threshold |   Validation False Warnings / Outing | Actual Data Source   |
|:---------------------------------------|--------------------:|----------------------:|-------------------------:|-------------------------------:|--------------------------------:|-------------------------------------:|:---------------------|
| Velocity Alone                         |               0.318 |                     0 |                        0 |                              0 |                             nan |                                    0 | mlb_statcast         |
| Release Point Alone (X, Z, Extension)  |               0.321 |                     0 |                        0 |                              0 |                             nan |                                    0 | mlb_statcast         |
| Spin & Movement (rate, axis, PFX, VAA) |               0.307 |                     0 |                        0 |                              0 |                             nan |                                    0 | mlb_statcast         |
| Full Micro-Mechanics Suite             |               0.322 |                     0 |                        0 |                              0 |                             nan |                                    0 | mlb_statcast         |

## Sensitivity reruns

|   xwOBA Threshold |   Window (PA) |   Horizon (Pitches) |   Test Episodes |   Test Episode Recall |   Test Warning Precision |   Test False Warnings / Outing |   Validation-Selected Threshold |   Validation False Warnings / Outing | Actual Data Source   |
|------------------:|--------------:|--------------------:|----------------:|----------------------:|-------------------------:|-------------------------------:|--------------------------------:|-------------------------------------:|:---------------------|
|              0.4  |             3 |                  15 |             447 |                 0.148 |                    0.41  |                          0.419 |                         213.848 |                                 0.34 | mlb_statcast         |
|              0.45 |             3 |                  15 |             357 |                 0.148 |                    0.414 |                          0.33  |                         227.396 |                                 0.25 | mlb_statcast         |
|              0.5  |             3 |                  15 |             313 |                 0.182 |                    0.37  |                          0.427 |                         208.034 |                                 0.46 | mlb_statcast         |
|              0.45 |             2 |                  15 |             492 |                 0.187 |                    0.44  |                          0.515 |                         169.578 |                                 0.49 | mlb_statcast         |
|              0.45 |             3 |                  10 |             357 |                 0.118 |                    0.261 |                          0.524 |                         219.271 |                                 0.48 | mlb_statcast         |
|              0.45 |             3 |                  20 |             357 |                 0.238 |                    0.411 |                          0.537 |                         152.422 |                                 0.46 | mlb_statcast         |

## Traceable case-study index

| case_type   |   game_pk | game_date           |   pitcher | pitcher_name   | actual_data_source   |   warning_pitch |   episode_onset_pitch |   lead_time_pitches |
|:------------|----------:|:--------------------|----------:|:---------------|:---------------------|----------------:|----------------------:|--------------------:|
| TP          |    744798 | 2024-09-29 00:00:00 |    605400 | Aaron Nola     | mlb_statcast         |              85 |                    87 |                   2 |
| FP          |    744799 | 2024-09-28 00:00:00 |    554430 | Zack Wheeler   | mlb_statcast         |              77 |                   nan |                 nan |
| FN          |    744821 | 2024-08-05 00:00:00 |    657277 | Logan Webb     | mlb_statcast         |              89 |                   nan |                 nan |
| TN          |    745536 | 2024-09-01 00:00:00 |    605400 | Aaron Nola     | mlb_statcast         |             nan |                   nan |                 nan |