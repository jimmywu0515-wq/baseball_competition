"""
Baseball Pitcher Mechanics Stability Index (MSI) Coach Dashboard
Interactive Streamlit Application for Real-Time Pitch Monitoring & Case Review

Terminology note: "Health Index" replaced with "Mechanics Stability Index (MSI)".
MSI measures delivery drift relative to a pitcher's personal historical baseline --
it is a hypothesis about mechanical instability, not a direct biological fatigue measurement.
"""
import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Ensure src is importable
app_root = Path(__file__).resolve().parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

from src.storage.adapter import StorageManager

st.set_page_config(
    page_title="MLB Pitcher Mechanics Stability Index (MSI) Dashboard",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #2d3139;
        margin-bottom: 15px;
    }
    .alert-banner-red {
        background-color: rgba(239, 83, 80, 0.2);
        border: 2px solid #ef5350;
        border-radius: 8px;
        padding: 14px;
        color: #ffcdd2;
        margin-bottom: 20px;
        font-weight: bold;
    }
    .alert-banner-green {
        background-color: rgba(67, 160, 71, 0.2);
        border: 2px solid #43a047;
        border-radius: 8px;
        padding: 14px;
        color: #c8e6c9;
        margin-bottom: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    sm = StorageManager(base_dir=str(app_root))
    scored_df = sm.load_table("fact_pitch_anomaly_scores", layer="gold")
    if scored_df.empty:
        from src.data_ingest.statcast_loader import StatcastLoader
        from scripts.run_full_pipeline import run_pipeline
        run_pipeline(base_dir=str(app_root))
        scored_df = sm.load_table("fact_pitch_anomaly_scores", layer="gold")

    labels_df = sm.load_table("fact_collapse_labels", layer="gold")
    alerts_df = sm.load_table("fact_alert_events", layer="gold")
    comp_df = sm.load_table("mart_model_evaluation", layer="gold")
    return scored_df, labels_df, alerts_df, comp_df


st.sidebar.title("⚾ 投手機制穩定度監控系統")
st.sidebar.caption("Mechanics Stability Index (MSI) — 基於微觀物理特徵漂移之預警系統")

scored_df, labels_df, alerts_df, comp_df = load_data()

if scored_df.empty:
    st.error("No data found in lakehouse. Please run `scripts/run_full_pipeline.py` first.")
    st.stop()

pitchers = sorted(scored_df["pitcher_name"].unique())
selected_pitcher = st.sidebar.selectbox("選擇先發投手 (Pitcher)", pitchers)

pitcher_games = scored_df[scored_df["pitcher_name"] == selected_pitcher]["game_pk"].unique()
selected_game = st.sidebar.selectbox("選擇出賽場次 (Game PK)", pitcher_games)

game_data = (
    scored_df[
        (scored_df["pitcher_name"] == selected_pitcher) &
        (scored_df["game_pk"] == selected_game)
    ]
    .sort_values("pitch_number_in_game")
    .copy()
)

if not labels_df.empty:
    g_labels = labels_df[labels_df["game_pk"] == selected_game]
    if not g_labels.empty and "is_collapse_event" in g_labels.columns:
        if "is_collapse_event" not in game_data.columns:
            merge_key = "pitch_number_in_outing"
            optional = ["is_collapse_event", "collapse_reason", "window_blended_xwoba"]
            label_columns = [merge_key] + [column for column in optional if column in g_labels.columns]
            game_data = game_data.merge(
                g_labels[label_columns].drop_duplicates(subset=[merge_key]),
                on=merge_key,
                how="left"
            )

# MSI column normalisation: support old health_index name
MSI_COL = None
for candidate in ("mechanics_stability_index", "msi", "health_index"):
    if candidate in game_data.columns:
        MSI_COL = candidate
        break
if MSI_COL is None:
    game_data["mechanics_stability_index"] = np.nan
    MSI_COL = "mechanics_stability_index"

max_pitches = len(game_data)
current_pitch_idx = st.sidebar.slider("逐球重播 (Pitch Number in Game)", 1, max_pitches, max_pitches)

visible_df = game_data.iloc[:current_pitch_idx]
curr_pitch = visible_df.iloc[-1]

st.title(f"📊 賽事實況：{selected_pitcher} (Game #{selected_game})")
actual_sources = sorted(map(str, game_data.get("actual_data_source", pd.Series(["unknown"])).dropna().unique()))
st.caption(f"Actual data source: {', '.join(actual_sources) if actual_sources else 'unknown'}")
st.markdown(
    f"**出賽日期**：`{curr_pitch.get('game_date', '2024-05-01')}` | "
    f"**目前局數**：第 `{curr_pitch.get('inning', 1)}` 局 | "
    f"**當前投球數**：`{current_pitch_idx}` 球"
)

alert_col = next((column for column in ("is_proposed_operating_alert", "is_cusum_alert")
                  if column in visible_df.columns), None)
alert_in_current = (
    visible_df[alert_col].any() if alert_col else (visible_df[MSI_COL] < 40).any()
)
first_alert_pitch = None
if alert_col and alert_in_current and "pitch_number_in_game" in visible_df.columns:
    alerts_only = visible_df[visible_df[alert_col]]
    if not alerts_only.empty:
        first_alert_pitch = int(alerts_only["pitch_number_in_game"].min())

score_available = bool(curr_pitch.get("score_available", pd.notna(curr_pitch.get(MSI_COL))))
score_status = str(curr_pitch.get("score_status", "AVAILABLE" if score_available else "UNAVAILABLE"))

if not score_available:
    st.info(f"Score unavailable for this pitch: `{score_status}`. No alert can be displayed or evaluated.")
elif alert_in_current:
    dominant_feature = curr_pitch.get("dominant_drift_feature", "Release Arm Slot")
    st.markdown(f"""
    <div class="alert-banner-red">
        🚨 <b>機制漂移警報觸發 (MECHANICS DRIFT ALERT)</b><br>
        於第 <b>{first_alert_pitch}</b> 球偵測到投球機制顯著偏離個人歷史常態！主導漂移特徵：<code>{dominant_feature}</code>。<br>
        ⚠️ 建議牛棚熱身並密切關注，預防後續可能之近程崩盤失分。
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="alert-banner-green">
        ✅ <b>機制穩定 (MECHANICS STABLE)</b><br>
        出手機制、自轉軸座標與進壘角均維持在個人歷史基準常態區間內。MSI 保持高位，未觸發 CUSUM 變點警報。
    </div>
    """, unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    msi_val = float(curr_pitch.get(MSI_COL, np.nan))
    delta_msi = round(msi_val - 100.0, 1) if np.isfinite(msi_val) else None
    st.metric(
        "機制穩定度指數 MSI (0–100)",
        f"{msi_val:.1f}" if np.isfinite(msi_val) else "Unavailable",
        f"{delta_msi:+.1f} vs baseline" if delta_msi is not None else score_status,
        help="Mechanics Stability Index = 100 × exp(−α·D_M). Higher = more stable. Scoring starts at pitch 21."
    )

with col2:
    spd = float(curr_pitch.get("release_speed", 95.0))
    ptype = curr_pitch.get("pitch_type", "FF")
    st.metric("當前球速 (Release Speed)", f"{spd:.1f} mph", ptype)

with col3:
    maha = float(curr_pitch.get("mahalanobis_calibrated", np.nan))
    st.metric(
        "馬氏異常距離 (Mahalanobis)",
        f"{maha:.2f} σ" if np.isfinite(maha) else "Unavailable",
        "Normal < 2.0 σ",
        help="Exact quadratic form (x−μ)ᵀΣ⁻¹(x−μ). Spin axis as (cos θ, sin θ) to avoid circular discontinuity."
    )

with col4:
    cusum_val = float(curr_pitch.get("cusum_stat", 0.0))
    st.metric("CUSUM 累積漂移量", f"{cusum_val:.2f}", "Threshold: 4.0")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 即時 MSI 時間序列監控",
    "🎯 微觀機制漂移分析",
    "🔬 基準對比與消融實驗",
    "📋 四類個案診斷"
])

with tab1:
    col_t1, col_t2 = st.columns([2, 1])

    with col_t1:
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=visible_df["pitch_number_in_game"],
            y=visible_df[MSI_COL],
            mode="lines+markers",
            name="Mechanics Stability Index (MSI)",
            line=dict(color="#29b6f6", width=3),
            marker=dict(size=6)
        ))
        fig_ts.add_vrect(
            x0=1, x1=20,
            fillcolor="rgba(255,235,59,0.08)",
            line_width=0,
            annotation_text="Calibration Zone (Pitches 1–20)",
            annotation_position="top left",
            annotation_font_size=10
        )
        fig_ts.add_hline(y=60, line_dash="dash", line_color="#ffa726", annotation_text="Caution (MSI 60)")
        fig_ts.add_hline(y=35, line_dash="dash", line_color="#ef5350", annotation_text="Critical (MSI 35)")
        if first_alert_pitch:
            fig_ts.add_vline(x=first_alert_pitch, line_color="#d81b60", line_width=2,
                             annotation_text=f"CUSUM Alert (#{first_alert_pitch})")
        if "is_collapse_event" in visible_df.columns:
            active = visible_df["is_collapse_event"].fillna(False).astype(bool)
            collapse_rows = visible_df[active & ~active.shift(fill_value=False)]
            if not collapse_rows.empty:
                fig_ts.add_trace(go.Scatter(
                    x=collapse_rows["pitch_number_in_game"],
                    y=collapse_rows[MSI_COL],
                    mode="markers",
                    marker=dict(color="#ff1744", size=12, symbol="x"),
                    name="Collapse Episode Onset"
                ))
        fig_ts.update_layout(
            title="逐球機制穩定度指數趨勢 (Pitch-by-Pitch MSI Time-Series)",
            xaxis_title="投球數 (Pitch Number in Game)",
            yaxis_title="Mechanics Stability Index (MSI, 0–100)",
            yaxis=dict(range=[0, 105]),
            template="plotly_dark",
            height=420,
            margin=dict(l=20, r=20, t=45, b=20)
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    with col_t2:
        fb_types = {"FF", "SI", "FC"}
        fb_df = visible_df[visible_df["pitch_type"].isin(fb_types)]
        if fb_df.empty:
            fb_df = visible_df
        fig_vel = go.Figure()
        for pitch_type, pitch_group in fb_df.groupby("pitch_type"):
            fig_vel.add_trace(go.Scatter(
                x=pitch_group["pitch_number_in_game"],
                y=pitch_group["release_speed"],
                mode="lines+markers",
                name=f"{pitch_type} velocity (mph)",
            ))
        if first_alert_pitch:
            fig_vel.add_vline(x=first_alert_pitch, line_color="#d81b60", line_dash="dash",
                              annotation_text="Alert")
        fig_vel.update_layout(
            title="球速走勢 — 主力速球僅限 FF/SI/FC",
            xaxis_title="Pitch #",
            yaxis_title="Speed (mph)",
            template="plotly_dark",
            height=420,
            margin=dict(l=20, r=20, t=45, b=20)
        )
        st.plotly_chart(fig_vel, use_container_width=True)

    if "cusum_stat" in visible_df.columns:
        fig_cusum = go.Figure()
        fig_cusum.add_trace(go.Scatter(
            x=visible_df["pitch_number_in_game"],
            y=visible_df["cusum_stat"],
            mode="lines",
            name="CUSUM Statistic",
            line=dict(color="#66bb6a", width=2)
        ))
        fig_cusum.add_hline(y=4.0, line_dash="dash", line_color="#ef5350", annotation_text="Alert Threshold (4.0)")
        fig_cusum.update_layout(
            title="CUSUM 變點累積漂移量 (Change-Point Detection)",
            xaxis_title="Pitch #", yaxis_title="CUSUM",
            template="plotly_dark", height=250, margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_cusum, use_container_width=True)

with tab2:
    c_r1, c_r2 = st.columns(2)

    with c_r1:
        categories = ['出手機制 (Release)', '自轉軸 (Spin cos/sin)', '位移/VAA (Movement)', '球速變異 (Speed)']
        r_vals = [
            float(curr_pitch.get("contrib_release_pct", 25.0)),
            float(curr_pitch.get("contrib_spin_pct", 25.0)),
            float(curr_pitch.get("contrib_movement_pct", 25.0)),
            float(curr_pitch.get("contrib_speed_pct", 25.0))
        ]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=r_vals, theta=categories, fill='toself',
            name='當前異常貢獻佔比 (%)', line_color='#26a69a'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="微觀機制異常貢獻分解 (Sub-Group Quadratic Projection)",
            template="plotly_dark", height=420
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.info("自轉軸以圓形座標 (cos θ, sin θ) 計算，避免 359°↔1° 邊界斷裂問題。")

    with c_r2:
        fig_rel = go.Figure()
        fig_rel.add_trace(go.Scatter(
            x=visible_df["release_pos_x"],
            y=visible_df["release_pos_z"],
            mode="markers",
            marker=dict(
                color=visible_df["pitch_number_in_game"],
                colorscale="Viridis", size=9,
                showscale=True, colorbar=dict(title="Pitch #")
            ),
            text=[f"Pitch #{p}" for p in visible_df["pitch_number_in_game"]],
            name="All Pitches"
        ))
        fig_rel.add_trace(go.Scatter(
            x=[curr_pitch["release_pos_x"]],
            y=[curr_pitch["release_pos_z"]],
            mode="markers",
            marker=dict(color="#ff1744", size=14, symbol="star"),
            name="Current Pitch"
        ))
        fig_rel.update_layout(
            title="出手點空間分布 (Release Point 2D — X vs Z)",
            xaxis_title="Release X (ft)", yaxis_title="Release Z (ft)",
            template="plotly_dark", height=420
        )
        st.plotly_chart(fig_rel, use_container_width=True)

    if "spin_axis_cos" in visible_df.columns and "spin_axis_sin" in visible_df.columns:
        fig_spin = go.Figure()
        fig_spin.add_trace(go.Scatter(
            x=visible_df["spin_axis_cos"],
            y=visible_df["spin_axis_sin"],
            mode="markers",
            marker=dict(
                color=visible_df["pitch_number_in_game"],
                colorscale="Plasma", size=8,
                showscale=True, colorbar=dict(title="Pitch #")
            ),
            name="Spin Axis (cos θ, sin θ)"
        ))
        theta = np.linspace(0, 2 * np.pi, 200)
        fig_spin.add_trace(go.Scatter(
            x=np.cos(theta), y=np.sin(theta),
            mode="lines", line=dict(color="rgba(255,255,255,0.2)", dash="dot"),
            name="Unit Circle"
        ))
        fig_spin.update_layout(
            title="自轉軸圓形座標分布 (Spin Axis on Unit Circle)",
            xaxis_title="cos(spin_axis)", yaxis_title="sin(spin_axis)",
            xaxis=dict(range=[-1.2, 1.2]),
            yaxis=dict(range=[-1.2, 1.2], scaleanchor="x", scaleratio=1),
            template="plotly_dark", height=380
        )
        st.plotly_chart(fig_spin, use_container_width=True)

with tab3:
    st.subheader("📊 Held-out model comparison")
    if comp_df.empty:
        st.info("No evaluated comparison is available. Run the full pipeline first.")
    else:
        st.caption("Held-out late-2024 results; thresholds were selected on early 2024 at one false-warning allowance.")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.subheader("🔬 Held-out feature ablations")
    ablation_path = app_root / "outputs" / "ablation_results.csv"
    if ablation_path.exists():
        st.dataframe(pd.read_csv(ablation_path), use_container_width=True, hide_index=True)
    else:
        st.info("No computed ablation results are available. Run the full pipeline first.")

    st.caption("Ablation and sensitivity values are calculated by the current pipeline, never embedded constants.")

with tab4:
    st.subheader("📋 Held-out case diagnostics (TP / FP / FN / TN)")
    case_dir = app_root / "outputs" / "case_studies"
    case_index_path = case_dir / "case_study_index.csv"
    case_index = pd.read_csv(case_index_path) if case_index_path.exists() else pd.DataFrame()

    case_info = [
        {
            "label": "✅ 真陽性 (True Positive — 成功預警)",
            "files": [case_dir / "case_study_1_true_positive.png"]
        },
        {
            "label": "⚠️ 偽陽性 (False Positive — 虛驚一場)",
            "files": [case_dir / "case_study_2_false_positive.png"]
        },
        {
            "label": "❌ 偽陰性 (False Negative — 漏報)",
            "files": [case_dir / "case_study_3_false_negative.png"]
        },
        {
            "label": "🟢 真陰性 (True Negative — 穩定好投)",
            "files": [case_dir / "case_study_4_true_negative.png"]
        }
    ]

    for case in case_info:
        st.markdown(f"#### {case['label']}")
        image_name = Path(case["files"][0]).name
        case_code = {
            "case_study_1_true_positive.png": "TP",
            "case_study_2_false_positive.png": "FP",
            "case_study_3_false_negative.png": "FN",
            "case_study_4_true_negative.png": "TN",
        }.get(image_name)
        evidence = case_index[case_index["case_type"].eq(case_code)] if not case_index.empty else pd.DataFrame()
        if evidence.empty:
            st.caption("No traceable evidence row is available for this category.")
        else:
            item = evidence.iloc[0]
            st.caption(
                f"Game {item['game_pk']} on {item['game_date']} | source={item['actual_data_source']} | "
                f"warning={item.get('warning_pitch')} | onset={item.get('episode_onset_pitch')} | "
                f"lead={item.get('lead_time_pitches')}"
            )
        img_path = None
        for fp in case["files"][:1]:
            if Path(fp).exists():
                img_path = fp
                break
        if img_path:
            st.image(str(img_path), use_column_width=True)
        else:
            st.info("Case study image not found. Run `scripts/run_full_pipeline.py` to generate case studies.")
        st.markdown("---")

    st.info(
        "**MSI 定義**：Mechanics Stability Index = 100 × exp(−α · D_M)，"
        "其中 D_M 為嚴格二次型馬氏距離 (x−μ)ᵀΣ⁻¹(x−μ)。\n\n"
        "MSI 衡量的是投球機制相對於投手個人歷史常態的偏離程度，"
        "**並非直接生理疲勞量測**。機制漂移是崩盤發生的假設性前兆。"
    )
