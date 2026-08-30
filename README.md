# 基於微觀物理特徵衰退的時間序列疲勞與崩盤預測系統
### Baseball Fatigue & Collapse Early Warning System (GCP-Ready Data Lakehouse)

本專案旨在解決棒球投手調度上的核心痛點：**能否在「球速還沒掉、失分還沒發生」之前，用逐球的釋放機制微特徵（Release Point 3D、Spin Axis、VAA 等）偵測投手機制正在崩壞？**

系統採用 **「無監督異常偵測（即時警報）」 + 「結果導向規則（事後驗證標籤，嚴格隔離）」** 的科學研究架構，並以 **GCP 雲端架構（BigQuery + Cloud Storage + Cloud Run）與 Medallion Data Lakehouse 五層式資料工程** 進行模組化實作。

---

## 系統架構與資料分層 (Medallion Data Lakehouse)

```
MLB Statcast API (pybaseball)
            │
            ▼
   [ Layer 0: Bronze ]  ───► raw_statcast_pitches (逐球原始數據，按日期分區)
            │
            ▼
   [ Layer 1: Silver ]  ───► dim_pitchers, dim_games, stg_qualified_pitches (嚴格 Qualify 篩選)
            │
            ├─────────────────────────────────────────┐
            ▼ (無未來資訊洩漏)                         ▼ (嚴格隔離防線)
   [ Layer 2: Silver ]                       [ Layer 4: Gold ]
     - feat_pitcher_pitchtype_baseline         - fact_collapse_labels (3-PA 滾動 xwOBA/Barrels/BB)
     - feat_pitch_level_features                      │
            │                                         │
            ▼                                         │
   [ Layer 3: Gold ]                                  │
     - fact_pitch_anomaly_scores (馬氏距離/AE)        │
     - fact_alert_events (CUSUM/EWMA 變點警報)        │
            │                                         │
            └────────────────────┬────────────────────┘
                                 ▼
                        [ Layer 5: Gold Marts ]
                          - mart_model_evaluation (Lift / Lead Time / Naive 對比)
                          - mart_game_case_studies
                                 │
                                 ▼
                     [ 教練即時戰情儀表板 (Streamlit) ]
```

### 資料表綱要 (Table Schemas)

| 分層 | 資料表名稱 | 鍵值 (Keys) | 核心用途與說明 |
|---|---|---|---|
| **Layer 0 (Bronze)** | `raw_statcast_pitches` | `(game_pk, pitch_number)` | Statcast 90+ 原始欄位，按 `game_date` 分區 |
| **Layer 1 (Silver)** | `stg_qualified_pitches` | `(game_pk, pitch_number)` | 符合 §3 Qualify 之先發投手逐球資料（$\ge 50$ 球、缺值 $<5\%$、排除 opener） |
| **Layer 1 (Silver)** | `dim_pitchers` | `pitcher` | 投手維度表，含主要球種分類（Top 2-3 pitch types） |
| **Layer 1 (Silver)** | `dim_games` | `game_pk` | 賽事維度表 |
| **Layer 2 (Silver)** | `feat_pitcher_pitchtype_baseline` | `(pitcher, pitch_type, as_of_date)` | **個人歷史滾動基準**（均值、標準差、共變異數矩陣 $\Sigma$ 與精度矩陣 $\Sigma^{-1}$，嚴格取賽前歷史，杜絕未來資訊洩漏） |
| **Layer 2 (Silver)** | `feat_pitch_level_features` | `(game_pk, pitch_number)` | 物理運動學特徵（垂直進壘角 VAA、3D 出手點距離）、個人歷史 z-score、開局 20 球 Shrinkage 經驗貝氏校正殘差、5/10 球滾動變異度 |
| **Layer 3 (Gold)** | `fact_pitch_anomaly_scores` | `(game_pk, pitch_number)` | 馬氏距離 $D_M$、Autoencoder 重建誤差、**機制健康指數 (Health Index, 0-100)** 與四維漂移貢獻佔比 |
| **Layer 3 (Gold)** | `fact_alert_events` | `(game_pk, alert_pitch_number)` | CUSUM / EWMA 變點警報事件記錄 |
| **Layer 4 (Gold)** | `fact_collapse_labels` | `(game_pk, at_bat_number)` | **隔離標籤表**：滾動 3-PA 窗口 Blended xwOBA $\ge 0.450$、Barrels $\ge 2$ 或 BB/HBP $\ge 2$ |
| **Layer 5 (Gold)** | `mart_model_evaluation` | `evaluation_id` | 系統評估與 Naive Baseline（球速下降 $\ge 1.5$ mph）效益對比總表 |

---

## 模組化目錄結構 (Project Directory)

```
baseball_competition/
├── README.md                      # 完整研究架構與技術文件
├── requirements.txt               # Python 依賴套件
├── Dockerfile                     # GCP Cloud Run / 容器化設定
├── config/
│   ├── config.yaml                # 演算法、閾值與篩選參數
│   └── gcp_config.yaml            # GCP Project ID, Bucket, BigQuery Dataset
├── src/
│   ├── storage/                   # Medallion 資料庫配接器 (DuckDB & BigQuery)
│   ├── data_ingest/               # Statcast 抓取與 §3 Qualify 篩選
│   ├── feature_engineering/       # Level 0/1 特徵、VAA 計算與滾動趨勢
│   ├── baseline_builder/          # 歷史基準 $\Sigma^{-1}$ 與開局 Shrinkage 校正
│   ├── anomaly_scorer/            # 馬氏距離、Autoencoder 與 Health Index (0-100)
│   ├── changepoint_detector/      # CUSUM / EWMA 變點警報演算法
│   ├── label_builder/             # 3-PA 窗口崩盤標籤（嚴格隔離）
│   ├── evaluation/                # Lift, Lead Time, PR-AUC, Naive Baseline 對比
│   └── visualization/             # 4 視圖個案研究圖表產出器
├── dashboard/
│   └── app.py                     # 教練即時戰情儀表板 (Streamlit)
├── scripts/
│   ├── run_full_pipeline.py       # 端到端自動化執行管線
│   ├── init_bigquery.sql          # GCP BigQuery DDL 建表指令檔
│   └── deploy_gcp.sh              # GCP 部署腳本
├── data/                          # 本地 Parquet 與 DuckDB Lakehouse
└── outputs/
    ├── case_studies/              # 個案視覺化高解析圖表 (PNG)
    ├── metrics_summary.json       # 評估指標 JSON
    └── validation_report.md       # 詳細驗證報告
```

---

## 快速上手與執行指南

### 1. 安裝環境與依賴套件
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 執行單元測試
```bash
PYTHONPATH=. pytest tests/ -v
```

### 3. 執行端到端完整管線 (Data Ingest -> Marts -> Case Studies)
```bash
PYTHONPATH=. python scripts/run_full_pipeline.py
```

### 4. 啟動教練即時戰情儀表板
```bash
streamlit run dashboard/app.py
```

---

## GCP 雲端部署指南 (GCP Deployment)

本系統提供一鍵式 GCP 部署支援：

1. **設定 GCP 環境變數**：
   ```bash
   export GCP_PROJECT_ID="your-gcp-project-id"
   export GCP_REGION="us-central1"
   export GCS_BUCKET_NAME="your-baseball-lakehouse"
   export BIGQUERY_DATASET="baseball_analytics"
   ```

2. **執行自動化部署腳本**：
   ```bash
   bash scripts/deploy_gcp.sh
   ```
   腳本會自動完成：
   - 啟用 BigQuery, Cloud Storage, Cloud Run, Cloud Build APIs
   - 建立 GCS Parquet Lakehouse Bucket
   - 執行 `scripts/init_bigquery.sql` 建立分區與叢集優化資料表
   - 建置 Docker 映像檔並部署 Streamlit 教練儀表板至 Cloud Run。

---

## 系統驗證與實證成果 (§6 & §7)

依據 80 場先發、7,000+ 逐球資料之實證驗證結果：

| 評估指標 | 本微特徵預警系統 (Proposed) | 傳統球速下降模型 (Naive Velocity) | 傳統球數限制 (Pitch Count $\ge 85$) |
|---|---|---|---|
| **Lift (Odds Ratio)** | **8.31x** | 0.0x (滯後) | 0.0x |
| **平均提前量 (Mean Lead Time)** | **31.2 球** | 68.6 球 (過度延遲) | 4.9 球 |
| **中位數提前量 (Median Lead Time)** | **24.0 球 (~1.5 PA)** | 77.0 球 | 4.0 球 |
| **精準度曲線面積 (PR-AUC)** | **0.357** | - | - |
| **即時決策優勢** | 在球速尚未下降前提前 1.5–2 打席示警 | 需等到被打爆/球速失速才亮燈 | 無法適應投手當日狀況與天氣 |
