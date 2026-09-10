# Temporal holdout validation report

- Requested mode: `simulation_benchmark`
- Actual data source: `simulation_benchmark`
- Train: 2023; threshold validation: January-June 2024; held-out test: July 2024 onward.
- Every model uses the same eligible pitches, one-to-one 15-pitch event matching, and a threshold selected under the same validation allowance of 0.5 distinct false warnings per outing.

## Held-out model comparison

| Model / System                                 |   Test Episode Recall |   Test Warning Precision |   Test False Warnings / Outing |   Test Clean Outing FAR |   Test Pitch PR-AUC |   Test Risk Ratio (Alert vs No Alert) |   Validation-Selected Threshold |   Validation False Warnings / Outing | Actual Data Source   | Interpretation                                                                       |
|:-----------------------------------------------|----------------------:|-------------------------:|-------------------------------:|------------------------:|--------------------:|--------------------------------------:|--------------------------------:|-------------------------------------:|:---------------------|:-------------------------------------------------------------------------------------|
| Proposed Micro-Mechanics (CUSUM + MSI)         |                 0.364 |                    0.8   |                           0.04 |                   0     |               0.407 |                                  6.97 |                        173.45   |                                 0.06 | simulation_benchmark | Mechanical-drift score; association is not evidence of fatigue causality.            |
| Contextual Model (Pitch Count + TTO + Inning)  |                 0.5   |                    0.688 |                           0.1  |                   0.118 |               0.415 |                                  6.16 |                          0.8511 |                                 0.08 | simulation_benchmark | Fit on 2023 only; threshold selected on early 2024.                                  |
| Pitch-Type Velocity Drop (FF/SI/FC separately) |                 0     |                    0     |                           0    |                   0     |               0.169 |                                  0    |                        nan      |                                 0    | simulation_benchmark | A contemporaneous velocity benchmark; relative risk alone does not establish timing. |
| Traditional Pitch Count                        |                 0.5   |                    0.647 |                           0.12 |                   0.118 |               0.411 |                                  5.72 |                         85      |                                 0.08 | simulation_benchmark | A workload heuristic with the same validation selection rule.                        |

## Feature ablations

| Feature Subset                         |   Test Pitch PR-AUC |   Test Episode Recall |   Test Warning Precision |   Test False Warnings / Outing |   Validation-Selected Threshold |   Validation False Warnings / Outing | Actual Data Source   |
|:---------------------------------------|--------------------:|----------------------:|-------------------------:|-------------------------------:|--------------------------------:|-------------------------------------:|:---------------------|
| Velocity Alone                         |               0.151 |                 0     |                    0     |                           0    |                        nan      |                                 0    | simulation_benchmark |
| Release Point Alone (X, Z, Extension)  |               0.32  |                 0.5   |                    0.688 |                           0.1  |                          4.2547 |                                 0.24 | simulation_benchmark |
| Spin & Movement (rate, axis, PFX, VAA) |               0.329 |                 0.545 |                    0.429 |                           0.32 |                          4.7156 |                                 0.18 | simulation_benchmark |
| Full Micro-Mechanics Suite             |               0.301 |                 0.545 |                    0.414 |                           0.34 |                          4.677  |                                 0.48 | simulation_benchmark |

## Sensitivity reruns

|   xwOBA Threshold |   Window (PA) |   Horizon (Pitches) |   Test Episodes |   Test Episode Recall |   Test Warning Precision |   Test False Warnings / Outing |   Validation-Selected Threshold |   Validation False Warnings / Outing | Actual Data Source   |
|------------------:|--------------:|--------------------:|----------------:|----------------------:|-------------------------:|-------------------------------:|--------------------------------:|-------------------------------------:|:---------------------|
|              0.4  |             3 |                  15 |              22 |                 0.364 |                    0.8   |                           0.04 |                         175.853 |                                 0.06 | simulation_benchmark |
|              0.45 |             3 |                  15 |              22 |                 0.364 |                    0.8   |                           0.04 |                         173.45  |                                 0.06 | simulation_benchmark |
|              0.5  |             3 |                  15 |              17 |                 0.471 |                    0.444 |                           0.2  |                         139.942 |                                 0.22 | simulation_benchmark |
|              0.45 |             2 |                  15 |              31 |                 0.323 |                    0.556 |                           0.16 |                         144.295 |                                 0.12 | simulation_benchmark |
|              0.45 |             3 |                  10 |              22 |                 0.318 |                    0.636 |                           0.08 |                         172.931 |                                 0.06 | simulation_benchmark |
|              0.45 |             3 |                  20 |              22 |                 0.5   |                    0.786 |                           0.06 |                         139.564 |                                 0.06 | simulation_benchmark |

## Traceable case-study index

| case_type   |   game_pk | game_date   |   pitcher | pitcher_name      | actual_data_source   |   warning_pitch |   episode_onset_pitch |   lead_time_pitches |
|:------------|----------:|:------------|----------:|:------------------|:---------------------|----------------:|----------------------:|--------------------:|
| TP          |    716024 | 2024-08-02  |    543037 | Gerrit Cole (Sim) | simulation_benchmark |              85 |                    88 |                   3 |
| FP          |    716028 | 2024-08-30  |    543037 | Gerrit Cole (Sim) | simulation_benchmark |              73 |                   nan |                 nan |
| FN          |    716025 | 2024-08-09  |    543037 | Gerrit Cole (Sim) | simulation_benchmark |             nan |                    93 |                 nan |
| TN          |    716020 | 2024-07-05  |    543037 | Gerrit Cole (Sim) | simulation_benchmark |             nan |                   nan |                 nan |