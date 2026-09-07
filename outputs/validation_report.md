# 實證驅動之 MLB Statcast 投手機制漂移與近程崩盤預警系統
## 深度驗證與基準對比實證報告 (Empirical Verification Report)

### 1. 核心評估指標總表 (§6 & §7)
- **資料來源模式**：MLB Statcast 實證數據
- **評估出賽總數**：175 場
- **評估投球總數**：11693 球 (嚴格排除開局 20 球校正期與審查投球)
- **崩盤事件 (Collapse Episodes) 總數**：392 次
- **崩盤事件召回率 (Episode Recall)**：**43.4%**
- **相對風險比 (Relative Risk / Lift)**：**1.22x** (警報後 15 球內發生崩盤起點之相對倍率)
- **真陽性提前量 (Lead Time - Pitches)**：**平均 12.6 球** (中位數 15.0 球)
- **真陽性提前量 (Lead Time - PAs)**：**平均 3.8 打席** (中位數 4.0 打席)
- **乾淨出賽誤警率 (Clean Outing FAR)**：**66.7%**
- **精準度曲線面積 (PR-AUC)**：**0.328**

---

### 2. 與真實棒球情境基準之公平對比 (§2 & §10)
所有模型均在相同的 `y_true_onset_in_horizon` 陣列上評估：

| Model / System                                | Relative Risk (Lift)   | PR-AUC   |   Precision | Episode Recall   | Clean Outing FAR   | Baseball Advantage                                 |
|:----------------------------------------------|:-----------------------|:---------|------------:|:-----------------|:-------------------|:---------------------------------------------------|
| Proposed Micro-Mechanics (CUSUM + MSI)        | 1.22x                  | 0.328    |       0.38  | 43.4%            | 66.7%              | Captures delivery instability before velo drop     |
| Contextual Model (Pitch Count + TTO + Inning) | 1.08x                  | 0.348    |       0.357 | 26.6%            | 93.3%              | Standard coaching baseline (Times Through Order)   |
| Naive FB Velocity Drop (>=1.5 mph)            | 0.93x                  | -        |       0.317 | 18.7%            | 93.3%              | Lags behind mechanics degradation; reactive        |
| Traditional Pitch Count (>=85)                | 1.04x                  | -        |       0.348 | 9.3%             | 26.7%              | Rigid heuristic; ignores daily individual variance |

---

### 3. 特徵群消融實驗 (§12)
| Feature Subset                   |   PR-AUC | Lift (Top 20% Alert)   |   Precision |   Recall |
|:---------------------------------|---------:|:-----------------------|------------:|---------:|
| Velocity Alone                   |    0.288 | 0.96x                  |       0.287 |    0.194 |
| Release Point Alone (X, Z, Ext)  |    0.281 | 0.87x                  |       0.265 |    0.179 |
| Spin & Movement Alone (PFX, VAA) |    0.295 | 0.99x                  |       0.293 |    0.198 |
| Full Micro-Mechanics Suite       |    0.284 | 0.91x                  |       0.275 |    0.186 |

---

### 4. 四大真實個案診斷 (§14)
1. **真陽性（True Positive，成功預警）**：`case_study_1_true_positive.png`
2. **偽陽性（False Positive，虛驚一場）**：`case_study_2_false_positive.png`
3. **偽陰性（False Negative，漏報）**：`case_study_3_false_negative.png`
4. **真陰性（True Negative，穩定好投）**：`case_study_4_true_negative.png`
