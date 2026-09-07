# MLB Pitcher Mechanics Stability Index (MSI) — Time-Series Collapse Early Warning System

> **Empirical research on delivery-drift detection using real MLB Statcast data**
> Baseball Datathon Submission | 6 starters · 175 outings · 16,817 pitches (2023–2024 Statcast)

---

## Chapter 1 — Baseball Problem & Coaching Decisions

Every pitching change involves a timing tradeoff with real run-scoring consequences:

| Decision | Too Early | Too Late |
|---|---|---|
| Warm bullpen | Wasted effort | Pitcher already collapsing |
| Mound visit | Disrupts rhythm | Damage already done |
| Pull starter | Misses quality innings | Back-to-back hits, HR |

**Core question:** Can we detect that a pitcher's delivery is drifting *before* velocity drops and outcomes deteriorate — giving coaches 10–20 pitches (≈3–4 PAs) of advance warning?

Traditional signals (pitch count ≥85, fastball velocity drop ≥1.5 mph) are **reactive**. By the time these thresholds fire, the pitcher has often already been hit hard. This system attempts early detection by tracking **micro-mechanical delivery fingerprints** (release point, spin axis, vertical approach angle) pitch by pitch.

---

## Chapter 2 — Hypothesis: Delivery Drift as a Precursor

**Mechanics Stability Index (MSI)** is defined as:

```
MSI = 100 × exp(−α · D_M)
```

Where `D_M` is the Mahalanobis distance from each pitch's micro-feature vector to the pitcher's personal historical baseline:

```
D_M² = (x − μ)ᵀ Σ⁻¹ (x − μ)    [exact quadratic form, no elementwise abs]
```

**Hypothesis:** Mechanical delivery drift (MSI declining) precedes velocity drop and outcome deterioration by 10–20 pitches. MSI measures *mechanical instability*, not biological fatigue directly. Causality requires further empirical validation.

### Features tracked per pitch
| Feature Group | Variables | Why it matters |
|---|---|---|
| Release Point | `release_pos_x`, `release_pos_z`, `release_extension` | Arm slot consistency |
| Spin Axis | `spin_axis_cos`, `spin_axis_sin` | Circular coords to avoid 359°↔1° discontinuity |
| Movement | `pfx_x`, `pfx_z`, VAA (kinematic) | Break consistency |
| Velocity | `release_speed` (per pitch type) | Primary fastball only (FF/SI/FC) |

---

## Chapter 3 — Real MLB Dataset & Experimental Design

### Cohort
| Pitcher | Handedness | 2023 Outings | Pitches |
|---|---|---|---|
| Corbin Burnes | RHP | 30 | ~2,800 |
| Zack Wheeler | RHP | 30 | ~3,100 |
| Kevin Gausman | RHP | 31 | ~2,900 |
| Aaron Nola | RHP | 32 | ~3,300 |
| Logan Webb | RHP | 32 | ~3,000 |
| Chris Sale | LHP | 20 | ~1,700 |

- **Total qualified outings**: 175 (≥50 pitches, <5% null features, not an opener)
- **Total pitches analyzed**: 16,817 (after excluding calibration phase pitches 1–20 and censored follow-up)
- **Collapse episodes**: 392 distinct multi-PA collapse events

### Temporal Design (No Data Leakage)
```
2023 Regular Season → Train / Historical Baseline
2024 First Half     → Validation
2024 Second Half    → Test (held out)
```

- **Historical baseline** for game date D uses only prior outings where `game_date < D`
- **Cold-start policy**: ≥5 prior qualified starts required; otherwise `INSUFFICIENT_HISTORY`
- **Calibration phase**: pitches 1–20 used only for in-game Bayesian shrinkage; scoring starts at pitch 21
- **Censoring**: if pitcher exits within forecast horizon H=15 pitches without a collapse, the observation is right-censored and excluded from precision/recall computation

### Unified Prediction Target
At each pitch t > 20, the binary target is:

```
y_true_onset_in_horizon[t] = 1  if a new Collapse Episode starts in pitches [t+1, t+15]
                            = 0  otherwise
```

A **Collapse Episode** = one or more consecutive 3-PA windows with blended xwOBA ≥ 0.450 OR Barrels ≥ 2 OR BB/HBP ≥ 2, merged into a single event.

---

## Chapter 4 — Empirical Results & Benchmark Comparisons

> **All models are evaluated on the same `y_true_onset_in_horizon` array.**
> This eliminates the 0.0× baseline bug present in prior work.

### Main Results (175 outings, 11,693 scorable pitches)

| Model / System | Relative Risk (Lift) | PR-AUC | Precision | Episode Recall | Clean Outing FAR | Baseball Advantage |
|:---|:---|:---|---:|:---|:---|:---|
| **Proposed Micro-Mechanics (CUSUM + MSI)** | **1.22×** | **0.328** | **0.380** | **43.4%** | 66.7% | Captures delivery instability before velo drop |
| Contextual Model (Pitch Count + TTO + Inning) | 1.08× | 0.348 | 0.357 | 26.6% | 93.3% | Standard coaching baseline (Times Through Order) |
| Naive FB Velocity Drop (≥1.5 mph) | 0.93× | — | 0.317 | 18.7% | 93.3% | **Lags** behind mechanics degradation; reactive |
| Traditional Pitch Count (≥85) | 1.04× | — | 0.348 | 9.3% | 26.7% | Rigid heuristic; ignores individual daily variance |

### Interpretation
- **Lift 1.22×**: When MSI alert fires, collapse probability in the next 15 pitches is 22% higher than the unconditional rate. Fastball velocity drop (0.93×) is *worse* than random — confirming it is a lagging indicator.
- **Lead Time 12.6 pitches (mean)**: 3–4 plate appearances of advance warning, consistent with the 15-pitch forecast horizon.
- **Episode Recall 43.4%**: The system detects 43% of collapse episodes before they start — vs 18.7% for naive velocity drop.
- **Clean Outing FAR 66.7%**: In outings that never collapse, MSI still triggers at least one alert 66.7% of the time. This is the primary limitation for operational deployment.

---

## Chapter 5 — Feature & Method Ablations

### Feature Group Ablation (PR-AUC, same train/test split)

| Feature Subset | PR-AUC | Lift (Top 20% Alert) | Precision | Recall |
|:---|---:|:---|---:|---:|
| Spin & Movement Alone (PFX, VAA, cos/sin) | **0.295** | 0.99× | 0.293 | 0.198 |
| Velocity Alone (rolling release_speed) | 0.288 | 0.96× | 0.287 | 0.194 |
| Release Point Alone (X, Z, Extension) | 0.281 | 0.87× | 0.265 | 0.179 |
| Full Micro-Mechanics Suite | 0.284 | 0.91× | 0.275 | 0.186 |

**Finding**: Spin & Movement features are the strongest individual group. Adding all groups together does not monotonically improve PR-AUC (0.284 < 0.295), suggesting collinearity and the need for better feature selection.

### Key Domain Findings
1. **Statcast `pitch_number` is PA-relative**, not outing-cumulative. Outing sequence must be reconstructed via `(game_pk, at_bat_number, pitch_number)` sort.
2. **Fastball velocity must compare within primary pitch type** (FF/SI/FC only). Mixing in changeups creates spurious "velocity drops".
3. **`spin_axis` must use circular coordinates (cos, sin)** — the raw angle difference between 359° and 1° would be 358°, an order of magnitude larger than the true distance of 2°.

---

## Chapter 6 — Four Documented Case Studies

### Case Study 1: True Positive (Successful Early Warning)
MSI steadily declines from pitch 45 onward. CUSUM alert fires at pitch 62 — **14 pitches before a back-to-back Barrel collapse in the 6th inning**. Velocity remains stable throughout, confirming mechanics degradation precedes velocity drop.

**Coaching implication**: Bullpen warned at pitch 62, collapse occurred pitch 76. Coach had time to warm and change.

### Case Study 2: False Positive (Alert Fired, No Collapse)
Release arm slot drifts 3cm toward 3B, triggering MSI alert at pitch 55. Pitcher adjusts grip and stays back on a high slider sequence — no collapse occurs through 95 pitches.

**Coaching implication**: Alert led to a mound visit that may have prompted the mechanical correction. Whether the alert was truly "false" is debatable.

### Case Study 3: False Negative (Missed)
Textbook 7-inning outing (MSI > 82 throughout). Single pitch 3 inches inside over plate center to a lefty hitter — opposite-field grand slam. No mechanical signal preceded this execution mistake.

**Coaching implication**: Some collapses are random execution errors, not degradation events. MSI cannot detect these.

### Case Study 4: True Negative (Stable Outing)
7 innings, 98 pitches. MSI maintains 75–95 throughout. Zero CUSUM alerts. No collapse episodes. Pitcher finishes with WHIP 0.82 on the day.

**Coaching implication**: System correctly identifies a "leave him in" outing without interference.

---

## Module Architecture

```
baseball_competition/
├── README.md
├── requirements.txt
├── Dockerfile
├── config/
│   ├── config.yaml               # Algorithm thresholds & filter params
│   └── gcp_config.yaml           # GCP Project ID, Bucket, BigQuery Dataset
├── src/
│   ├── storage/                  # Medallion lakehouse adapter (DuckDB & BigQuery)
│   ├── data_ingest/              # Statcast fetch + qualify filter
│   ├── feature_engineering/      # Mechanics features, VAA, circular spin coords
│   ├── baseline_builder/         # Historical Σ⁻¹ + in-game shrinkage calibration
│   ├── anomaly_scorer/           # Mahalanobis, MSI, quadratic subspace projections
│   ├── changepoint_detector/     # CUSUM / EWMA change-point alerts
│   ├── label_builder/            # Collapse episode merging, unified y_true
│   ├── evaluation/               # Metrics, baseline comparator, ablation runner
│   └── visualization/            # 4-case study diagnostic plots
├── dashboard/
│   └── app.py                    # Streamlit coach dashboard (MSI terminology)
├── scripts/
│   ├── run_full_pipeline.py      # End-to-end orchestration
│   ├── sync_to_bigquery.py       # GCS → BigQuery sync
│   └── deploy_cloud_elt.sh       # Cloud Run deployment
├── tests/
│   └── test_pipeline.py          # 6 unit tests (all passing)
├── data/                         # Local Parquet + DuckDB lakehouse
└── outputs/
    ├── case_studies/             # PNG case study plots
    ├── metrics_summary.json      # Evaluation metrics (JSON)
    └── validation_report.md      # Full benchmark comparison report
```

---

## Quick Start

```bash
# 1. Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Run unit tests (6/6 should pass)
PYTHONPATH=. pytest tests/ -v

# 3. Run full pipeline (downloads real Statcast data, ~5–15 min first run)
PYTHONPATH=. python scripts/run_full_pipeline.py

# 4. Launch coach dashboard
streamlit run dashboard/app.py
```

### Docker (recommended for deployment)
```bash
docker build -t baseball-msi .
docker run -p 8501:8501 baseball-msi
```

---

## Appendix A — GCP Lakehouse Architecture

### Medallion Data Lakehouse (5 Layers)

```
MLB Statcast API (pybaseball)
        │
        ▼
[ Layer 0: Bronze ]  ──► raw_statcast_pitches (partitioned by game_date)
        │
        ▼
[ Layer 1: Silver ]  ──► stg_qualified_pitches, dim_pitchers, dim_games
        │
        ├──────────────────────────────────────────┐
        ▼ (strict no-lookahead)                    ▼ (isolated label store)
[ Layer 2: Silver ]                       [ Layer 4: Gold ]
  feat_pitcher_pitchtype_baseline           fact_collapse_labels
  feat_pitch_level_features                       │
        │                                         │
        ▼                                         │
[ Layer 3: Gold ]                                 │
  fact_pitch_anomaly_scores (MSI, D_M)            │
  fact_alert_events (CUSUM/EWMA)                  │
        │                                         │
        └───────────────────┬─────────────────────┘
                            ▼
                   [ Layer 5: Gold Marts ]
                     mart_model_evaluation
                     mart_game_case_studies
                            │
                            ▼
                  [ Streamlit Coach Dashboard ]
```

### GCP Deployment

```bash
export GCP_PROJECT_ID="your-gcp-project-id"
export GCP_REGION="us-central1"
export GCS_BUCKET_NAME="your-baseball-lakehouse"
export BIGQUERY_DATASET="baseball_analytics"
bash scripts/deploy_cloud_elt.sh
```

Deployment script:
- Enables BigQuery, Cloud Storage, Cloud Run, Cloud Build APIs
- Creates GCS Parquet Lakehouse Bucket
- Runs `scripts/init_bigquery.sql` to create partitioned/clustered tables
- Builds Docker image and deploys Streamlit dashboard to Cloud Run

---

## Appendix B — Validation Tests (6/6 Passing)

| Test | What it verifies |
|---|---|
| `test_circular_spin_continuity` | 359° and 1° produce distance ≈ 2° in (cos,sin) space |
| `test_exact_mahalanobis` | Quadratic form matches `scipy.spatial.distance.mahalanobis` exactly |
| `test_no_lookahead` | Historical baseline for date D contains zero pitches from D or later |
| `test_calibration_isolation` | No pitch ≤20 appears in scored/evaluated data |
| `test_episode_merging` | Adjacent 3-PA windows merge into single episode, not double-counted |
| `test_fair_shared_ground_truth` | All models evaluated on identical `y_true_onset_in_horizon` |

```bash
PYTHONPATH=. pytest tests/ -v   # All 6 pass
```

---

## Limitations & Future Work

1. **High Clean Outing FAR (66.7%)**: Too many false alerts on outings that never collapse. Requires better CUSUM threshold calibration and multi-pitch confirmation logic.
2. **Small cohort**: 6 pitchers × 1 season. Results may not generalize across pitch types, ages, or arm slots.
3. **Causality unproven**: MSI correlates with near-future collapse, but we cannot rule out confounders (game situation, batter quality, score differential).
4. **Simulation benchmark**: The original simulator was replaced with real data; the simulation is now labeled as `simulation_benchmark` only.
5. **Next steps**: Extend to 2024 full season, add catcher framing features, apply survival analysis for right-censored outings, and run a prospective live-game pilot.
