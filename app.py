import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math
import json
import pandas as pd
import google.generativeai as genai
from PIL import Image
from scipy.signal import savgol_filter  # ★追加：高精度のG算出（SGフィルタ）用

# --- 1. ページ設定（※絶対に一番最初に書く） ---
st.set_page_config(page_title="タミケンシム - フロントサスシミュレーター/AI分析ツール", layout="wide")

# ===============================
# Session State（状態保存）の初期化
# ===============================
# ①シミュレーター用パラメータ
default_params = {
    "setting_name": "260315_基本セット",
    "m_bike": 90.0, "m_rider": 68.0, "caster_angle": 25.0, 
    "decel_g": 1.00, "g_conversion": 80,  # ★初期値を1.50に変更し、G残存率(80%)を追加
    "fork_id": 36.0, "x_max": 97.0, "oil_lock_len": 10.0,
    "k_init": 0.37, "k_late": 0.98, "s_change": 82.0, "preload": 27.0,
    "n_index": 2.40, "oil_base": 60, "oil_comp": 70, "target_stroke": 50.0
}
for key, val in default_params.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ②AI連携用パラメータ
default_ai_params = {
    "track_name": "", 
    "tire_info": "リア　TT93PRO ミディアムソフト　100LAP", 
    "front_protrusion": 0.0,
    "front_stroke_free": 221.0, 
    "front_stroke_bottom": 125.0,
    "comp_level": 5, "reb_level": 5,
    "rear_k_init": 14.07, "rear_k_late": 0.0,
    "rear_preload": 0.5, 
    "rear_ride_height": 0.0,
    "rear_rate_change": 0.0, 
    "rear_stroke_free": 195.0, 
    "rear_stroke_bottom": 117.0,
    "rear_comp_level": 5, "rear_reb_level": 5
}
for key, val in default_ai_params.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ===============================
# サイドバー：セッティング管理
# ===============================
with st.sidebar:
    st.title("💾 セッティング管理")
    
    st.header("1. フロントセッティング")
    st.caption("シミュレーター用の基本設定（ばねレートツールと共通読込可）")
    
    current_settings = {k: st.session_state[k] for k in default_params.keys()}
    json_str = json.dumps(current_settings, indent=4)
    export_file_name = f"{st.session_state['setting_name']}.json"
    
    st.download_button(label="フロント設定を保存 (.json)", data=json_str, file_name=export_file_name, mime="application/json")
    
    uploaded_file = st.file_uploader("フロント設定を読込", type="json")
    if uploaded_file is not None:
        try:
            loaded_data = json.load(uploaded_file)
            for k, v in loaded_data.items():
                if k in st.session_state: st.session_state[k] = v
            st.success("フロント設定を読み込みました！")
        except Exception:
            st.error("読込失敗。ファイル形式を確認してください。")

    st.divider()

    st.header("2. AI解析用・セッティング")
    st.caption("AI連携用の車高や減衰目安（専用読込ファイル）")
    
    ai_save_keys = list(default_ai_params.keys())
    ai_settings = {k: st.session_state[k] for k in ai_save_keys if k in st.session_state}
    ai_json_str = json.dumps(ai_settings, indent=4)
    export_ai_name = f"AI_Prompt_{st.session_state['setting_name']}.json"
    
    st.download_button(label="AI設定を保存 (.json)", data=ai_json_str, file_name=export_ai_name, mime="application/json", key="ai_save_btn")
    
    uploaded_ai_file = st.file_uploader("AI設定を読込（専用）", type="json", key="ai_load_btn")
    if uploaded_ai_file is not None:
        try:
            loaded_ai_data = json.load(uploaded_ai_file)
            for k, v in loaded_ai_data.items():
                if k in ai_save_keys and k in st.session_state: st.session_state[k] = v
            st.success("AI設定を読み込みました！")
        except Exception:
            st.error("読込失敗。ファイル形式を確認してください。")

# ===============================
# メイン画面：タイトル
# ===============================
title_design = """
<style>
    .tamiken-title {
        font-family: "Hiragino Mincho ProN", "MS Mincho", serif;
        font-size: 80px !important; font-weight: 900 !important; color: #E60000 !important;
        text-shadow: -3px -3px 0 #FFD700, 3px -3px 0 #FFD700, -3px 3px 0 #FFD700, 3px 3px 0 #FFD700, -3px 0px 0 #FFD700, 3px 0px 0 #FFD700, 0px -3px 0 #FFD700, 0px 3px 0 #FFD700, 4px 4px 6px rgba(0, 0, 0, 0.5);
        letter-spacing: -2px; margin-bottom: 0px; padding-bottom: 0px; line-height: 1.2;
    }
    .tamiken-subtitle { font-family: sans-serif; font-size: 30px !important; font-weight: bold !important; color: var(--text-color) !important; margin-top: 5px; margin-bottom: 25px; }
</style>
"""
st.markdown(title_design, unsafe_allow_html=True)
st.markdown('<p class="tamiken-title">タミケンシム</p>', unsafe_allow_html=True)
st.markdown('<p class="tamiken-subtitle">フロントサスシミュレーター / AI分析ツール（『こぼれ小話 タミケンバーン』連動）</p>', unsafe_allow_html=True)

st.info("YouTubeチャンネル『こぼれ小話 タミケンバーン』連動ツール、素人構築にて精度向上検証中です。\n異常値報告等ご指摘に数値共有などは、下記チャンネルのフロントサスシミュレーター関連の動画コメント欄へお願いいたします。")
st.markdown("▶ [ばねレート簡易判定ツール v2.5 はこちら](https://spring-rate-tool.streamlit.app/)  \n▶ [YouTube：こぼれ小話タミケンバーン チャンネルTOP](https://www.youtube.com/@dogtamy-Lean-burn)")

setting_name_input = st.text_input("📝 セッティング名（保存ファイル名に反映されます）", value=st.session_state["setting_name"])
st.session_state["setting_name"] = setting_name_input
st.divider()

# ===============================
# 1. 車両・ライディング条件
# ===============================
st.header("① 車両・ライディング条件")
col_v1, col_v2, col_v3 = st.columns(3)
with col_v1:
    m_bike = st.number_input("車体重量 [kg]", 0.0, 300.0, value=float(st.session_state["m_bike"]), step=1.0)
    st.session_state["m_bike"] = m_bike
    m_rider = st.number_input("装備体重 [kg]", 0.0, 200.0, value=float(st.session_state["m_rider"]), step=1.0)
    st.session_state["m_rider"] = m_rider
    total_m = m_bike + m_rider
with col_v2:
    caster_angle = st.number_input("キャスター角 (静止時) [deg]", 0.0, 45.0, value=float(st.session_state["caster_angle"]), step=0.1)
    st.session_state["caster_angle"] = caster_angle
    
    # ★名称をロガーのピーク値に変更し、範囲も拡大
    decel_g = st.number_input("ロガー最大減速G (ピーク値)", 0.5, 2.5, value=float(st.session_state["decel_g"]), step=0.01)
    st.session_state["decel_g"] = decel_g
with col_v3:
    # ★水面下の変数「G残存率」をUIに追加
    g_conversion = st.slider("フルボトム時のG残存率 [%]", 50, 100, value=int(st.session_state["g_conversion"]), step=1, help="ロガーの最大ピークGから、フォークが最奥に到達し姿勢が安定した瞬間のGの落ち込み割合。通常75〜85%程度が目安です。")
    st.session_state["g_conversion"] = g_conversion
    
    # ★実効G（ピークG × 残存率）を計算して荷重に反映
    rad = math.radians(caster_angle)
    effective_g = decel_g * (g_conversion / 100.0)
    f_target_total_kg = (total_m * math.cos(rad) + total_m * effective_g * math.sin(rad)) * 1.18
    
    st.metric("フォーク全体への想定最大荷重", f"{f_target_total_kg:.1f} kg")
    st.write(f"**（車重＋実効G換算({effective_g:.2f}G)＋重心移動補正：1.18）**")

# ===============================
# 2. フォーク・バネ内部仕様
# ===============================
st.header("② フォーク・バネ内部仕様")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    fork_id = st.number_input("フォーク内径 [mm]", 10.0, 60.0, value=float(st.session_state["fork_id"]), step=0.1)
    st.session_state["fork_id"] = fork_id
    x_max = st.number_input("最大ストローク [mm]", 1.0, 250.0, value=float(st.session_state["x_max"]), step=1.0)
    st.session_state["x_max"] = x_max
    oil_lock_len = st.number_input("オイルロック長（フルストロークからの残り） [mm]", 0.0, 50.0, value=float(st.session_state["oil_lock_len"]), step=1.0)
    st.session_state["oil_lock_len"] = oil_lock_len

with col_f2:
    k_init = st.number_input("初期レート (1本分) [kg/mm]", 0.0, 20.0, value=float(st.session_state["k_init"]), step=0.01)
    st.session_state["k_init"] = k_init
    k_late = st.number_input("後半レート (1本分) [kg/mm]", 0.0, 20.0, value=float(st.session_state["k_late"]), step=0.01)
    st.session_state["k_late"] = k_late
    s_change = st.number_input("レート変化点 [mm]", 0.0, 250.0, value=float(st.session_state["s_change"]), step=1.0)
    st.session_state["s_change"] = s_change

with col_f3:
    preload = st.number_input("プリロード [mm]", 0.0, 50.0, value=float(st.session_state["preload"]), step=1.0)
    st.session_state["preload"] = preload
    n_index = st.slider("空気断熱指数 n", 1.0, 3.0, value=float(st.session_state["n_index"]), step=0.01)
    st.session_state["n_index"] = n_index

st.markdown("""
**【空気断熱指数 $n$ の目安】**
* **0% (理論値) 1.60**：非常に緩やか。奥での踏ん張りが足りない。
* **25% 占拠 1.80 〜 1.95**：正立フォークなどでバネが細い場合。
* **50% 占拠 2.10 〜 2.50**：倒立フォークや、太いカラー・インナーが入っている状態。
* **60% 占拠 2.50 〜 2.70**：ほぼオイルロックに近い、極めて急激な立ち上がり。

※本数値は、オイルによる動的減衰抵抗を空気の反力として擬似的に合算したセッティング指標です。
""")

# ===============================
# 3. 油面比較 ＆ ストローク荷重調査
# ===============================
st.header("③ 油面比較 ＆ ストローク位置の荷重調査")
col_o1, col_o2 = st.columns(2)
with col_o1:
    oil_base = st.slider("基準油面 [mm]", 10, 200, value=int(st.session_state["oil_base"]))
    st.session_state["oil_base"] = oil_base
    oil_comp = st.slider("比較油面 [mm]", 10, 200, value=int(st.session_state["oil_comp"]))
    st.session_state["oil_comp"] = oil_comp
with col_o2:
    target_stroke = st.number_input("調査したいストローク位置 [mm]（補足機能）", 0.0, float(x_max), value=float(st.session_state["target_stroke"]), step=1.0)
    st.session_state["target_stroke"] = target_stroke

# ===============================
# 計算ロジック & グラフ描画
# ===============================
area_air = math.pi * (fork_id / 2)**2
def get_spring_f(x):
    x_total = x + preload
    total_change = s_change + preload
    if k_late == 0 or s_change <= 0: return k_init * x_total
    if x_total <= total_change: return k_init * x_total
    else: return k_init * total_change + k_late * (x_total - total_change)

def get_air_f(x, air_space_at_full):
    p0 = 1.033 
    L0 = (air_space_at_full + x_max) * 0.95 
    L_current = L0 - x
    if L_current <= 0.1: return 3000.0
    p1 = p0 * ((L0 / L_current) ** n_index)
    return (p1 - p0) * (area_air / 100)

def total_f_2pcs(x, oil): return (get_spring_f(x) + get_air_f(x, oil)) * 2

def find_res_stroke(target_f, oil):
    search = np.linspace(0, x_max, 1000)
    for sx in search:
        if total_f_2pcs(sx, oil) >= target_f: return max(0.0, x_max - sx)
    return 0.0

st.header("④ シミュレーション結果比較")
c1, c2 = st.columns(2)
res_base = find_res_stroke(f_target_total_kg, oil_base)
res_comp = find_res_stroke(f_target_total_kg, oil_comp)
with c1:
    st.metric(f"基準油面 ({oil_base}mm) 最大荷重時残スト", f"{res_base:.1f} mm")
    st.write(f"（補足）ストローク {target_stroke}mm 時の荷重: **{total_f_2pcs(target_stroke, oil_base):.1f} kg**")
with c2:
    st.metric(f"比較油面 ({oil_comp}mm) 最大荷重時残スト", f"{res_comp:.1f} mm", delta=f"{res_comp - res_base:.1f} mm")
    st.write(f"（補足）ストローク {target_stroke}mm 時の荷重: **{total_f_2pcs(target_stroke, oil_comp):.1f} kg**")

st.write("---")
show_spring = st.checkbox("金属バネ反力を表示", value=True)
show_air = st.checkbox("エアバネ反力を表示", value=False)
show_total = st.checkbox("合成反力を表示", value=True)

x_plot = np.linspace(0, x_max, 500)
fig = go.Figure()
if show_spring:
    fig.add_trace(go.Scatter(x=x_plot, y=[get_spring_f(x) * 2 for x in x_plot], name="金属バネ(両側)", line=dict(dash='dash', color='silver')))
if show_air:
    fig.add_trace(go.Scatter(x=x_plot, y=[get_air_f(x, oil_base) * 2 for x in x_plot], name=f"エア単体({oil_base}mm)", line=dict(color='lightblue', width=2)))
    fig.add_trace(go.Scatter(x=x_plot, y=[get_air_f(x, oil_comp) * 2 for x in x_plot], name=f"エア単体({oil_comp}mm)", line=dict(color='pink', width=2)))
if show_total:
    fig.add_trace(go.Scatter(x=x_plot, y=[total_f_2pcs(x, oil_base) for x in x_plot], name="基準合成", line=dict(color="blue", width=4)))
    fig.add_trace(go.Scatter(x=x_plot, y=[total_f_2pcs(x, oil_comp) for x in x_plot], name="比較合成", line=dict(color="red", width=4)))

fig.add_vrect(x0=x_max - oil_lock_len, x1=x_max, fillcolor="gray", opacity=0.1, layer="below", annotation_text="オイルロック域")
fig.add_hline(y=f_target_total_kg, line_dash="dot", line_color="green", annotation_text="想定最大荷重")

# ★修正箇所：font=dict(...) を追加して、どんな環境でも日本語が文字化けしないように強制指定しました
fig.update_layout(
    xaxis_title="ストローク量 [mm]", 
    yaxis_title="荷重 (2本合計) [kg]", 
    template="simple_white", 
    height=600,
    font=dict(family="'Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Meiryo', 'Noto Sans JP', sans-serif") 
)
st.plotly_chart(fig, use_container_width=True)

# ===============================
# ★ AI連携用データ入力UI
# ===============================
st.divider()
st.header("⑤ AIデータ解析設定")

analysis_mode = st.radio("📊 解析モードを選択してください", ["単一データ解析（1つの走行ログを深く分析）", "2つのデータ比較解析（A/Bテスト・仕様違いの比較）"], horizontal=True)

col_ai1, col_ai2 = st.columns(2)

with col_ai1:
    st.subheader("A. セッティングの状態入力")

    if "単一" in analysis_mode:
        track_name = st.text_input("サーキット名（任意）", value=st.session_state.get("track_name", ""), placeholder="例：近畿スポーツランド、鈴鹿サーキット")
        
        st.markdown("##### ⚙️ 車体・電子制御・ブレーキ設定")
        c_etc1, c_etc2 = st.columns(2)
        with c_etc1:
            brake_pad = st.selectbox("ブレーキパッド初期特性", ["2: リニア (握力に比例)", "1: 初期大 (ガツンと効く)", "3: 奥大 (奥で強く効く)"])
            steering_damper = st.selectbox("ステアリングダンパー減衰", ["無し", "1 (弱い)", "2", "3", "4", "5 (強い)"])
        with c_etc2:
            tc_base = st.selectbox("トラクションコントロール(TC)基本設定", ["無し", "1 (弱い)", "2", "3", "4", "5 (強い)"])
            eb_base = st.selectbox("エンジンブレーキ(EBC)・アイドル設定", ["無し/固定", "1 (エンブレ弱)", "2", "3", "4", "5 (エンブレ強)"])
        tc_memo = st.text_input("TC・電子制御 コーナー別指定 (自由記入)", placeholder="例: 最終コーナーのみTCを強めに設定(レベル4)")

        st.markdown("##### 🛞 タイヤ・空気圧・環境設定")
        tire_info = st.text_input("タイヤ銘柄・状態（任意）", value=st.session_state.get("tire_info", ""), placeholder="例：リア TT93PRO ミディアムソフト 100LAP")
        c_tp1, c_tp2, c_tp3 = st.columns(3)
        with c_tp1:
            tire_p_unit = st.selectbox("空気圧 単位", ["kgf/cm2 (kg)", "PSI", "Bar"])
            tire_temp = st.number_input("測定時タイヤ温度 [℃]", value=15, step=1)
        with c_tp2:
            tire_p_front = st.number_input("フロント空気圧", value=1.50, step=0.05)
            track_temp = st.number_input("路面温度 [℃] (任意)", value=15, step=1)
        with c_tp3:
            tire_p_rear = st.number_input("リア空気圧", value=1.90, step=0.05)
            track_cond = st.selectbox("路面状況", ["ドライ", "ハーフウェット", "ウェット"])
        
        st.markdown("##### 🏍️ フロント設定")
        front_spring_dir = st.selectbox("フロント バネの向き (密・荒のセット方向)", ["指定なし/等ピッチ", "密巻きが下 (地面側)", "密巻きが上 (車体側)"])
        c_f1, c_f2 = st.columns(2)
        with c_f1:
            front_protrusion = st.number_input("突き出し量 [±mm]", value=float(st.session_state.get("front_protrusion", 0.0)), step=1.0)
        with c_f2:
            st.write("ロガー計測値設定")
            front_stroke_free = st.number_input("フロントストロークフリー(最大全伸び) [mm]", value=float(st.session_state.get("front_stroke_free", 221.0)), step=1.0)
            front_stroke_bottom = st.number_input("フロントストロークボトム(最小フルストローク) [mm]", value=float(st.session_state.get("front_stroke_bottom", 125.0)), step=1.0)
            
        st.write("フロント減衰目安 ※1:弱い 〜 10:強い")
        c_f3, c_f4 = st.columns(2)
        with c_f3:
            comp_level = st.slider("フロント圧側（コンプ）", 1, 10, int(st.session_state.get("comp_level", 5)))
        with c_f4:
            reb_level = st.slider("フロント伸び側（リバウンド）", 1, 10, int(st.session_state.get("reb_level", 5)))

        st.markdown("##### 🏍️ リア設定")
        st.caption("※リアツインサスの場合は、左右のバネレートを合算した数値を入力してください。")
        rear_spring_dir = st.selectbox("リア バネの向き (密・荒のセット方向)", ["指定なし/等ピッチ", "密巻きが下 (地面側)", "密巻きが上 (車体側)"])
        c_r1, c_r2 = st.columns(2)
        with c_r1:
            rear_k_init = st.number_input("リア初期レート [kg/mm]", value=float(st.session_state.get("rear_k_init", 14.07)), step=0.1)
            rear_k_late = st.number_input("リア後半レート [kg/mm] ※0でシングル", value=float(st.session_state.get("rear_k_late", 0.0)), step=0.1)
            rear_rate_change = st.number_input("レート変化点 [mm] ※シングルは0", value=float(st.session_state.get("rear_rate_change", 0.0)), step=1.0)
        with c_r2:
            rear_preload = st.number_input("リアプリロード量 [mm]", value=float(st.session_state.get("rear_preload", 0.5)), step=0.1)
            rear_ride_height = st.number_input("車高調整(サス単体セット長) [±mm]", value=float(st.session_state.get("rear_ride_height", 0.0)), step=1.0)
            
            st.write("ロガー計測値設定")
            rear_stroke_free = st.number_input("リアストロークフリー(最大全伸び) [mm]", value=float(st.session_state.get("rear_stroke_free", 195.0)), step=1.0)
            rear_stroke_bottom = st.number_input("リアストロークボトム(最小フルストローク) [mm]", value=float(st.session_state.get("rear_stroke_bottom", 117.0)), step=1.0)
            
        st.write("リア減衰目安 ※1:弱い 〜 10:強い")
        c_r3, c_r4 = st.columns(2)
        with c_r3:
            rear_comp_level = st.slider("リア圧側（コンプ）", 1, 10, int(st.session_state.get("rear_comp_level", 5)))
        with c_r4:
            rear_reb_level = st.slider("リア伸び側（リバウンド）", 1, 10, int(st.session_state.get("rear_reb_level", 5)))
            
    else:
        st.write("💡 **比較する2つのデータ（CSV）の違いをメモしてください。**")
        data_a_memo = st.text_area("Data A (基準) の条件・ファイル名", value="例：Data_A.csv (フロントバネ5.5Nm, リア車高±0mm)")
        data_b_memo = st.text_area("Data B (比較) の条件・ファイル名", value="例：Data_B.csv (フロントバネ5.0Nm, リア車高+2mm)")
        # ★ 重複エラー回避のため、keyを設定しています
        track_name = st.text_input("サーキット名（任意）", value="", placeholder="例：近畿スポーツランド", key="track_name_compare")
        tire_info = st.text_area("タイヤ・その他共通設定（前後サス・ディメンション・電子制御等）", value="", placeholder="例：前後タイヤ新品。TCはレベル2固定。", key="tire_info_compare")

# --- (ここから下は既存の with col_ai2: のコードが続きます) ---

with col_ai2:
    st.subheader("B. 解析したい課題と状況の入力")
    run_condition = st.selectbox("走行状況（タイムや走り方への影響）", ["単独走行（マイペースでのアタック・クリアラップ）", "追い走行（前走者をターゲットにしたアタック）", "混走・トラフィックあり（ペースの乱れあり）"])
    
    # ★追加：対象ラップの絞り込み
    target_lap_mode = st.selectbox("解析対象ラップの絞り込み", [
        "指定なし（全体から課題フェーズを検索）", 
        "ベストラップ付近を中心に解析", 
        "平均値に一番近いラップ付近を中心に解析", 
        "ベストラップと平均値に近いラップを比較して解析"
    ])
    
    phase_selection = st.selectbox("課題が発生しているフェーズ（場所）", ["進入・フルブレーキング", "旋回中・コーナリング", "切り返し・S字", "立ち上がり・アクセルオン", "ストレート・全開加速"])
    st.caption("⚠️ **注意:** ロガーの画面とCSVデータで「Lap数」がズレている場合があります。")
    user_comment = st.text_area("具体的な悩み・知りたいこと（★良いと感じている部分もあれば記載）", value="例：Lap 9の1コーナー進入でフロントが戻ってこない感覚がある。逆にS字の切り返しは軽快で良い感じなので、そこは犠牲にしたくない。", height=150)

# ===============================
# ★ AI連携用ファイルアップロードと実行
# ===============================
st.write("---")
st.subheader("C. AIデータ解析実行")

col_out1, col_out2 = st.columns([1, 1.5])

with col_out1:
    st.write("**STEP 1: 反力テーブルの確認（任意）**")
    stroke_range = np.arange(0, x_max + 1, 1.0)
    df_export = pd.DataFrame({
        "Stroke_mm": stroke_range,
        "Total_Force_Base_2pcs_N": [total_f_2pcs(x, oil_base) * 9.80665 for x in stroke_range],
        "Total_Force_Comp_2pcs_N": [total_f_2pcs(x, oil_comp) * 9.80665 for x in stroke_range]
    })
    csv_data = df_export.to_csv(index=False).encode('utf-8')
    st.download_button("反力テーブルをダウンロード", data=csv_data, file_name=f"ForceTable_{st.session_state.get('setting_name', 'data')}.csv", mime="text/csv")

with col_out2:
    st.write("**STEP 2: 走行ログ(CSV)と設定画像(任意)のアップロード**")
    if "単一" in analysis_mode:
        uploaded_logs = st.file_uploader("走行ログ(CSV)をアップロードしてください", type="csv", accept_multiple_files=False)
    else:
        uploaded_logs = st.file_uploader("比較する2つの走行ログ(CSV)をアップロードしてください", type="csv", accept_multiple_files=True)

    # ★復活：燃調マップ等の画像アップロード
    uploaded_images = st.file_uploader("燃調マップ・設定画面のスクショ等があればアップロード（任意）", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    # カラム自動検出UI
    selected_cols = []
    custom_sensor_memo = ""
    if uploaded_logs:
        preview_file = uploaded_logs[0] if isinstance(uploaded_logs, list) else uploaded_logs
        try:
            preview_df = pd.read_csv(preview_file, nrows=0)
            all_cols = preview_df.columns.tolist()
            preview_file.seek(0)
            
            typical_cols = ['Lap', 'RunTime', 'LapTime', 'ThrottoleP', 'Speed', 'Front', 'Rear', 'Rpm', 'Afr', 'T1', 'T2', 'T3']
            default_cols = [c for c in typical_cols if c in all_cols]
            
            st.markdown("##### ⚙️ ロガーデータ抽出設定（HRC・カスタムセンサー対応）")
            selected_cols = st.multiselect("📊 AIに解析させるデータ列を選択", options=all_cols, default=default_cols)
            custom_sensor_memo = st.text_input("📝 カスタムセンサー等の意味をAIに教える（任意）", placeholder="例：T1はオイル温度、T2はシリンダー温度")
        except Exception:
            st.error("CSVファイルの列名読み込みに失敗しました。")

st.write("---")
st.write("**STEP 3: 解析設定と実行**")

analysis_focus = st.radio(
    "🧠 AIの解析アプローチを選択してください",
    [
        "【実走・フィーリング重視】実際のサスの動きの流れ（フロー）と、ライダーの悩みの解決を最優先する ※推奨",
        "【バランス型】ロガー波形の数値とシミュレーターの反力テーブルの理論値を総合して解析する"
    ]
)
if st.button("AIに事前処理（ADA）をかけて解析させる", type="primary"):
    if not uploaded_logs:
        st.warning("⚠️ 走行ログ(CSV)をアップロードしてください。")
    elif not selected_cols:
        st.warning("⚠️ 解析するデータ列（カラム）を1つ以上選択してください。")
    elif "GEMINI_API_KEY" not in st.secrets:
        st.error("⚠️ APIキーが設定されていません。StreamlitのSecretsを確認してください。")
    else:
        with st.spinner("データエンジニア(Python)が異常値フィルターと高精度G算出を実行中..."):
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                model = genai.GenerativeModel('gemini-2.5-flash')

                stroke_range_text = df_export.to_csv(index=False)

                log_contents = ""
                logs_to_process = uploaded_logs if isinstance(uploaded_logs, list) else [uploaded_logs]

                def get_stable_stroke(df, col_name, time_col='RunTime'):
                    if df.empty or col_name not in df.columns: return None
                    idx_min = df[col_name].idxmin() 
                    val_min = df.loc[idx_min, col_name]
                    time_min = df.loc[idx_min, time_col]
                    lap_min = df.loc[idx_min, 'Lap'] if 'Lap' in df.columns else '不明'
                    
                    window = df[(df[time_col] >= time_min - 0.5) & (df[time_col] <= time_min + 0.5)]
                    q10 = window[col_name].quantile(0.10)
                    stable_mean = window[window[col_name] > q10][col_name].mean()
                    if pd.isna(stable_mean): stable_mean = val_min
                    
                    return {'val': val_min, 'stable_mean': stable_mean, 'time': time_min, 'lap': lap_min}

                for uploaded_log in logs_to_process:
                    uploaded_log.seek(0)
                    log_df = pd.read_csv(uploaded_log)
                    ada_summary = []
                    ada_summary.append(f"--- File: {uploaded_log.name} の事前処理レポート ---")
                    
                    exist_cols = [c for c in selected_cols if c in log_df.columns]
                    if exist_cols:
                        ada_summary.append("\n【0. パフォーマンス・姿勢指標（SGフィルタ＆異常ラップ自動除外済）】")
                        
                        if 'RunTime' in log_df.columns:
                            log_df['dt'] = log_df['RunTime'].diff().fillna(0.1)
                            log_df['dt'] = log_df['dt'].apply(lambda x: x if x > 0 else 0.1)
                            
                            # ==========================================
                            # ★ 異常ラップ（アウトラップ・ショートカット）の自動判定フィルター
                            valid_laps_mask = pd.Series(True, index=log_df.index)
                            valid_laps = []
                            if 'Lap' in log_df.columns:
                                if 'LapTime' in log_df.columns and not log_df['LapTime'].isna().all():
                                    lap_durations = log_df.groupby('Lap')['LapTime'].first()
                                else:
                                    lap_durations = log_df.groupby('Lap')['RunTime'].agg(lambda x: x.max() - x.min())
                                
                                if not lap_durations.empty:
                                    median_lap = lap_durations.median()
                                    valid_durations = lap_durations[lap_durations >= median_lap * 0.7]
                                    if not valid_durations.empty:
                                        base_lap_time = valid_durations.min()
                                        valid_laps = lap_durations[(lap_durations >= base_lap_time * 0.90) & (lap_durations <= base_lap_time * 1.20)].index.tolist()
                                        excluded_laps = [l for l in lap_durations.index if l not in valid_laps]
                                        if excluded_laps:
                                            valid_laps_mask = log_df['Lap'].isin(valid_laps)
                                            ada_summary.append(f"・[異常値自動フィルター] タイム基準で Lap {excluded_laps} をショートカットやアウトラップ等と判定し除外しました。")
                            # ==========================================
                            
                            dt_median = log_df['dt'].median()
                            window_len = int(0.4 / dt_median) if dt_median > 0 else 7
                            if window_len % 2 == 0: window_len += 1
                            if window_len < 5: window_len = 5
                            
                            # ★ [追加] Gセンサー（縦G・前後G）の列があるか確認し優先処理
                            lon_g_cols = [c for c in log_df.columns if c.lower() in ['g_lon', 'long', 'lon_g', '縦g', 'acc_x', 'accel_x']]
                            has_lon_g_sensor = len(lon_g_cols) > 0
                            
                            if has_lon_g_sensor:
                                lon_g_col = lon_g_cols[0]
                                raw_lon_g = log_df[lon_g_col].clip(lower=-3.0, upper=3.0)
                                try:
                                    log_df['Acc_G_Sensor'] = savgol_filter(raw_lon_g, window_length=window_len, polyorder=2)
                                except Exception:
                                    log_df['Acc_G_Sensor'] = raw_lon_g.rolling(window=window_len, center=True).mean()
                                    
                                valid_g_sens = log_df.loc[valid_laps_mask, 'Acc_G_Sensor']
                                valid_g_sens = valid_g_sens[(valid_g_sens >= -1.6) & (valid_g_sens <= 1.6)]
                                if not valid_g_sens.empty:
                                    min_g_idx = valid_g_sens.idxmin()
                                    max_g_idx = valid_g_sens.idxmax()
                                    ada_summary.append(f"・[★Gセンサー実測優先] 最大減速G: {valid_g_sens.min():.3f} G (Lap {log_df.loc[min_g_idx, 'Lap'] if 'Lap' in log_df.columns else '不明'}, {log_df.loc[min_g_idx, 'RunTime']:.1f}s)")
                                    ada_summary.append(f"・[★Gセンサー実測優先] 最大加速G: {valid_g_sens.max():.3f} G (Lap {log_df.loc[max_g_idx, 'Lap'] if 'Lap' in log_df.columns else '不明'}, {log_df.loc[max_g_idx, 'RunTime']:.1f}s)")

                            # 2. 車速センサからの加減速G
                            if 'Speed' in log_df.columns:
                                speed_ms = log_df['Speed'] / 3.6
                                raw_g_speed = speed_ms.diff().fillna(0) / log_df['dt'] / 9.80665
                                raw_g_speed = raw_g_speed.clip(lower=-3.0, upper=3.0)
                                
                                try:
                                    log_df['Acc_G_Speed'] = savgol_filter(raw_g_speed, window_length=window_len, polyorder=2)
                                except Exception:
                                    log_df['Acc_G_Speed'] = raw_g_speed.rolling(window=window_len, center=True).mean()
                                
                                # ★ 低速ノイズ除外：車速が「60km/h以上」出ている時の減速のみを評価
                                valid_g_speed = log_df.loc[valid_laps_mask, 'Acc_G_Speed']
                                valid_g_speed = valid_g_speed[(valid_g_speed >= -1.6) & (valid_g_speed <= 1.6) & (log_df['Speed'] >= 60.0)]
                                
                                if not valid_g_speed.empty:
                                    min_g_idx = valid_g_speed.idxmin()
                                    max_g_idx = valid_g_speed.idxmax()
                                    label_prefix = "[車速センサ推計]" if not has_lon_g_sensor else "[参考:車速推計]"
                                    ada_summary.append(f"・{label_prefix} 最大減速G: {valid_g_speed.min():.3f} G (Lap {log_df.loc[min_g_idx, 'Lap'] if 'Lap' in log_df.columns else '不明'}, {log_df.loc[min_g_idx, 'RunTime']:.1f}s)")
                                    if not has_lon_g_sensor:
                                        ada_summary.append(f"・{label_prefix} 最大加速G: {valid_g_speed.max():.3f} G (Lap {log_df.loc[max_g_idx, 'Lap'] if 'Lap' in log_df.columns else '不明'}, {log_df.loc[max_g_idx, 'RunTime']:.1f}s)")

                            # 3. GPS速度からの加減速G
                            gps_cols = [c for c in log_df.columns if c.lower() in ['gps_speed', 'gpsspeed']]
                            if gps_cols:
                                gps_ms = log_df[gps_cols[0]] / 3.6
                                raw_g_gps = gps_ms.diff().fillna(0) / log_df['dt'] / 9.80665
                                raw_g_gps = raw_g_gps.clip(lower=-3.0, upper=3.0)
                                
                                try:
                                    log_df['Acc_G_GPS'] = savgol_filter(raw_g_gps, window_length=window_len, polyorder=2)
                                except Exception:
                                    log_df['Acc_G_GPS'] = raw_g_gps.rolling(window=window_len, center=True).mean()
                                    
                                # ★ 低速ノイズ除外：GPS速度が「60km/h以上」出ている時の減速のみを評価
                                valid_g_gps = log_df.loc[valid_laps_mask, 'Acc_G_GPS']
                                valid_g_gps = valid_g_gps[(valid_g_gps >= -1.6) & (valid_g_gps <= 1.6) & (log_df[gps_cols[0]] >= 60.0)]
                                if not valid_g_gps.empty:
                                    min_gps_idx = valid_g_gps.idxmin()
                                    max_gps_idx = valid_g_gps.idxmax()
                                    label_prefix = "[GPS推計]" if not has_lon_g_sensor else "[参考:GPS推計]"
                                    ada_summary.append(f"・{label_prefix} 最大減速G: {valid_g_gps.min():.3f} G (Lap {log_df.loc[min_gps_idx, 'Lap'] if 'Lap' in log_df.columns else '不明'}, {log_df.loc[min_gps_idx, 'RunTime']:.1f}s)")
                                    if not has_lon_g_sensor:
                                        ada_summary.append(f"・{label_prefix} 最大加速G: {valid_g_gps.max():.3f} G (Lap {log_df.loc[max_gps_idx, 'Lap'] if 'Lap' in log_df.columns else '不明'}, {log_df.loc[max_gps_idx, 'RunTime']:.1f}s)")

                            # 4. 横Gデータの取得
                            lat_g_cols = [c for c in log_df.columns if c.lower() in ['g_lat', 'latg', 'lat_g', '横g']]
                            if lat_g_cols:
                                lat_g_col = lat_g_cols[0]
                                raw_lat_g_sens = log_df[lat_g_col].clip(lower=-3.0, upper=3.0)
                                try:
                                    log_df[lat_g_col] = savgol_filter(raw_lat_g_sens, window_length=window_len, polyorder=2)
                                except Exception:
                                    log_df[lat_g_col] = raw_lat_g_sens.rolling(window=window_len, center=True).mean()
                                    
                                valid_lat = log_df.loc[valid_laps_mask, lat_g_col]
                                if not valid_lat.empty:
                                    max_lat_idx = valid_lat.abs().idxmax()
                                    ada_summary.append(f"・[センサー実測] 最大横G: {abs(log_df.loc[max_lat_idx, lat_g_col]):.3f} G (Lap {log_df.loc[max_lat_idx, 'Lap'] if 'Lap' in log_df.columns else '不明'}, {log_df.loc[max_lat_idx, 'RunTime']:.1f}s)")
                            else:
                                lat_cols = [c for c in log_df.columns if c.lower() in ['latitude', 'lat']]
                                lon_cols = [c for c in log_df.columns if c.lower() in ['longitude', 'lon']]
                                if lat_cols and lon_cols:
                                    lat_rad = np.radians(log_df[lat_cols[0]])
                                    lon_rad = np.radians(log_df[lon_cols[0]])
                                    dlon = lon_rad.diff()
                                    dlat = lat_rad.diff()
                                    x = dlon * np.cos((lat_rad + lat_rad.shift(1))/2)
                                    y = dlat
                                    heading = np.arctan2(x, y)
                                    d_heading = heading.diff()
                                    d_heading = (d_heading + np.pi) % (2 * np.pi) - np.pi
                                    yaw_rate = d_heading / log_df['dt']
                                    
                                    v_ms = log_df[gps_cols[0]] / 3.6 if gps_cols else (log_df['Speed'] / 3.6 if 'Speed' in log_df.columns else 0)
                                    raw_lat_g = (v_ms * yaw_rate) / 9.80665
                                    raw_lat_g = raw_lat_g.clip(lower=-3.0, upper=3.0)
                                    
                                    try:
                                        log_df['Lat_G_GPS'] = savgol_filter(raw_lat_g.fillna(0), window_length=window_len, polyorder=2)
                                    except Exception:
                                        log_df['Lat_G_GPS'] = raw_lat_g.rolling(window=window_len, center=True).mean()
                                        
                                    valid_lat_gps = log_df.loc[valid_laps_mask, 'Lat_G_GPS']
                                    valid_lat_gps = valid_lat_gps[(valid_lat_gps >= -1.6) & (valid_lat_gps <= 1.6)]
                                    if not valid_lat_gps.empty:
                                        max_lat_gps_idx = valid_lat_gps.abs().idxmax()
                                        ada_summary.append(f"・[GPS座標推計] 最大横G: {abs(log_df.loc[max_lat_gps_idx, 'Lat_G_GPS']):.3f} G (Lap {log_df.loc[max_lat_gps_idx, 'Lap'] if 'Lap' in log_df.columns else '不明'}, {log_df.loc[max_lat_gps_idx, 'RunTime']:.1f}s)")

                            # 5. 最大ストローク (有効ラップのみ)
                            f_stroke = get_stable_stroke(log_df[valid_laps_mask], 'Front')
                            if f_stroke:
                                ada_summary.append(f"・【フロント】瞬間最大ストローク: {f_stroke['val']:.1f}mm | ギャップ除外の安定平均: {f_stroke['stable_mean']:.1f}mm (Lap {f_stroke['lap']}, {f_stroke['time']:.1f}s付近)")
                            r_stroke = get_stable_stroke(log_df[valid_laps_mask], 'Rear')
                            if r_stroke:
                                ada_summary.append(f"・【リア】瞬間最大ストローク: {r_stroke['val']:.1f}mm | ギャップ除外の安定平均: {r_stroke['stable_mean']:.1f}mm (Lap {r_stroke['lap']}, {r_stroke['time']:.1f}s付近)")

                        ada_summary.append("\n【1. 全体ピーク値（間引き前の生データより抽出）】")
                        num_cols = log_df[exist_cols].select_dtypes(include=np.number).columns
                        for col in num_cols:
                            max_val, min_val = log_df[col].max(), log_df[col].min()
                            if pd.notna(max_val) and pd.notna(min_val):
                                ada_summary.append(f"{col} 最大値: {max_val:.2f}, 最小値: {min_val:.2f}")

                        # ==========================================
                        # ★ 各ラップのピークG・最大ストローク（カンペ作成処理・Gセンサー対応）
                        if 'Lap' in log_df.columns:
                            ada_summary.append("\n【1.5. 各ラップのピークG・最大ストローク（全データ抽出）】")
                            for current_lap in sorted(log_df['Lap'].dropna().unique()):
                                lap_data = log_df[log_df['Lap'] == current_lap]
                                if lap_data.empty: continue
                                
                                ex_mark = " (異常値として除外)" if valid_laps and current_lap not in valid_laps else ""
                                lap_str = f"・Lap {int(current_lap)}{ex_mark}: "
                                
                                # Gセンサーがあればそれを、無ければ車速由来（60km/h以上）を使う
                                g_col_to_use = 'Acc_G_Sensor' if has_lon_g_sensor and 'Acc_G_Sensor' in lap_data.columns else 'Acc_G_Speed'
                                if g_col_to_use in lap_data.columns:
                                    if g_col_to_use == 'Acc_G_Speed' and 'Speed' in lap_data.columns:
                                        valid_g = lap_data[g_col_to_use][(lap_data[g_col_to_use] >= -1.6) & (lap_data[g_col_to_use] <= 1.6) & (lap_data['Speed'] >= 60.0)]
                                    else:
                                        valid_g = lap_data[g_col_to_use][(lap_data[g_col_to_use] >= -1.6) & (lap_data[g_col_to_use] <= 1.6)]
                                        
                                    if not valid_g.empty:
                                        lap_str += f"最大減速 {valid_g.min():.3f}G / 最大加速 {valid_g.max():.3f}G | "
                                        
                                if 'Front' in lap_data.columns:
                                    lap_str += f"Fストローク最奥 {lap_data['Front'].min():.1f}mm | "
                                    
                                ada_summary.append(lap_str)
                        # ==========================================

                        df_trend = log_df[exist_cols].copy()
                        df_trend[num_cols] = df_trend[num_cols].rolling(window=5, min_periods=1).mean()
                        
                        step = max(1, len(df_trend) // 300)
                        df_trend_sampled = df_trend.iloc[::step, :]
                        
                        if 'Acc_G_Sensor' in log_df.columns:
                             df_trend_sampled['Est_G_Lon'] = log_df['Acc_G_Sensor'].iloc[::step].values
                        elif 'Acc_G_Speed' in log_df.columns:
                             df_trend_sampled['Est_G_Speed'] = log_df['Acc_G_Speed'].iloc[::step].values

                        ada_summary.append("\n【2. トレンド波形データ（分析用サンプリング済）】")
                        ada_summary.append(df_trend_sampled.to_csv(index=False))
                    
                    log_contents += "\n".join(ada_summary) + "\n"

                if "単一" in analysis_mode:
                    settings_info = f"""
[車体・電子制御・ブレーキ]
・ブレーキパッド特性: {brake_pad} / ステアリングダンパー: {steering_damper}
・トラクションコントロール: 基本 {tc_base} (個別指定: {tc_memo}) / エンジンブレーキ(EBC): {eb_base}
[タイヤ・空気圧設定]
・銘柄/状態: {tire_info}
・空気圧 (測定温度 {tire_temp}℃): F {tire_p_front} / R {tire_p_rear} [{tire_p_unit}]
・路面環境: 路面温度 {track_temp}℃ / コンディション: {track_cond}
[フロント仕様・設定]
・バネレート: {k_init} kg/mm (後半: {k_late} kg/mm) / プリロード: {preload} mm / 油面: {oil_base} mm
・突き出し量: {front_protrusion} mm / バネセット方向: {front_spring_dir}
・ロガー計測値: フリー(全伸び) {front_stroke_free} mm / ボトム(最小) {front_stroke_bottom} mm
・減衰目安(1-10): 圧側 {comp_level} / 伸び側 {reb_level}
[リア仕様・設定]
・バネレート: 初期 {rear_k_init} kg/mm, 後半 {rear_k_late} kg/mm (変化点: {rear_rate_change} mm) ※ツインは合算
・プリロード: {rear_preload} mm / 車高調整(サス単体セット長): {rear_ride_height} mm / バネセット方向: {rear_spring_dir}
・ロガー計測値: フリー(全伸び) {rear_stroke_free} mm / ボトム(最小) {rear_stroke_bottom} mm
・減衰目安(1-10): 圧側 {rear_comp_level} / 伸び側 {rear_reb_level}
"""
                else:
                    settings_info = f"""
【比較するデータ条件（差分メモ）】
・Data A (基準): {data_a_memo}
・Data B (比較): {data_b_memo}
・タイヤ・電子制御等共通情報: {tire_info}
"""

                focus_instruction = f"""
                【最重要指示（絶対厳守）】
                ・【対象ラップの絶対厳守】ユーザーは「{target_lap_mode}」を指定しています。特に「ベストラップ付近を中心に」と指定されている場合、最大減速Gや最大ストロークが別の遅いラップ（異常値）で記録されていたとしても、**必ず最もタイムが速い「ベストラップ」の挙動を分析の主軸（主語）**としてください。異常値に引っ張られて遅いラップを中心に解説することは厳禁です。
                """
                if "フィーリング重視" in analysis_focus:
                    focus_instruction += "・シミュレーターの反力テーブルの数値（理論値）に固執せず、添付した「トレンド波形データ」から読み取れる【実際のサスの動きの流れ（ピッチングのフロー）】と、ユーザーの具体的な悩み（フィーリング）を最優先にすり合わせ、論理的な解決策を提示してください。\n"

                sensor_instruction = f"\n【カスタムセンサー・列名の補足情報（重要）】\nユーザーからの補足: {custom_sensor_memo}\n" if custom_sensor_memo else ""

                full_prompt = f"""
                あなたはワークスチームのチーフ・サスペンションエンジニア 兼 データエンジニアです。
                添付ファイル【反力テーブル(CSV)】【走行ログ要約データ】【設定画像(任意)】を掛け合わせ、論理的な解析とアドバイスを行ってください。

                【車両・環境・サスセッティング情報】
                ・サーキット名: {track_name}
                ・ライダー込重量: {total_m} kg / 想定最大減速G: {decel_g} G
                ・キャスター角(静止時ベース): {caster_angle} deg
                {settings_info}
                
                【ターゲット課題】
                ・走行状況: {run_condition}
                ・対象ラップの絞り込み: {target_lap_mode}
                ・発生フェーズ: {phase_selection}
                ・具体的な悩み: {user_comment}

                {focus_instruction}
                {sensor_instruction}

                =========================================
                【評価基準（裏設定：出力見出しには直接書かず、内部の分析プロセスとして使用すること）】
                AIは以下の基準を基に波形を分析し、結論を導き出してください。

                ■ [旋回を引き出す前後動作] の評価基準
                サスペンションの挙動が「鋭い旋回性を引き出すための動的な姿勢変化」を作り出せているかを以下の4点で評価する。
                ・ブレーキング: フロントの高い初期ストロークスピードと適切なリアのリフト状態を作り出せているか。タイヤグリップの限界に合わせたフロントブレーキリリースができているか。また逆に倒しこみでの遠心力の増加と共にリアブレーキ操作増の操作を実現できているか。減速の最大Gを表記してブレーキング技術の基準として扱うが、全体にスムーズな操作で実現するR半径の大きい走行も高いレベルの場合は否定しないこと、エアバネ反力のグラフスロープの適切な部位を使えているか。
                ・倒しこみ: ブレーキ終盤からのフロント側操舵(前後ブレーキ、バンク角、舵角補助)の状態を作り出せているか。
                ・初期旋回: 減速と加速の間の旋回瞬間のフルバンク状態で、適切な車高から実舵角をより引き出せる操舵状態を作り出せているか。またリアブレーキを常に操作しリアへの遠心力荷重の付加状態を作り出せているか。
                ・後期旋回: 瞬間旋回後半にて前後ブレーキ操作(特にリアブレーキ)と荷重コントロールでのリアタイヤの変形で、リアが外側軌道へと向かわせるリア操舵状態を作り出せているか。
                ※注意：切り返し（S字）などの連続コーナーを評価する場合、手前のコーナーの「脱出動作」なのか、本題のコーナーへの「進入動作」なのかを、時系列データから明確に区別して記載すること。

                ■ [トラクションと車体の乱れ] の評価基準
                アクセルON時のリアの沈み込みとフロントの伸びを分析し、「効率の良い加速状態を作り出せているか」、ライディングフォームやバンク角のバランスによるトラクション抜けや、前後サスの反発による車体の乱れ（跳ねやチャタリング）が発生していないか確認と評価すること。（※空気圧の影響やTC・ステダンの介入状況も推測すること）

                ■ [燃調・その他の補足基準]
                ・燃調: AfrやRpmからトルク変動の悪影響がないか確認し、必要に応じた燃調修正案を提示すること。
                ・実舵角: バンク角との相関関係が密接であるとして表現し、バンク角は重心位置(ライディングフォーム)を加味した実効バンク角として表現し指摘する場合もあること。
                ・リアサスペンションと荷重のリアブレーキコントロール :
                  1. リアブレーキの操作により減速時だけでなく加速時でも沈ませられる。
                  2. 予備的な荷重負荷のコントロールによりストロークの奥の部分を積極的に活用することによりストロークの抑制も可能。
                  3. 加速時のリアブレーキの操作によるストローク量の増加によりリアの安定感向上と荷重の増加によるグリップ力の向上。
                  4. 後期旋回操舵から加速旋回操舵にて、リアの軌道を外側へずらして旋回力を高められる。
                ・フロント最大ストローク付近の部位を使用している解析の場面ではエアバネ反力は適切な部位を使えているかグラフを見返せるようなストローク数値を提示し、必要なら検証することを促す事。
                ・バネのセット方向: 密巻きと荒巻の上下セット方向での変化について、密巻き向きが地面よりの入力側(下側)の場合は同じプリロードでも若干動き出しがやわらかい特性を考慮すること。
                ・サスセッティングの主目的: 適切な動的姿勢変化を見つけることである。
                ・ライディングポジション: ハンドル、シート、ステップ位置の設定によるライディングポジションやフォームによる動的姿勢変化も表現すること。
                =========================================

                【出力フォーマット】
                ※指示文や裏設定の文章をそのまま見出しとしてオウム返しすることは厳禁です。必ず以下の1〜5のシンプルな見出しのみを使用して出力してください。

                1. [データ抽出とコース把握]
                   ・【コーナー名称の直感的な明示（必須）】秒数（RunTime〇秒）だけの無機質な表記は避け、AIの持つサーキットの知識（{track_name}）、および速度推移や横Gから推測し、「1コーナー」「第1ヘアピン」「左2コーナー」「右3コーナー」「複合1,2コーナー」「第2ヘアピン手前高速左」「最終手前3連続ヘアピンの2個目右」など、ライダーが直感的に場所を特定できる名称（左右の向きや順番を含む）を必ず割り当てて記述すること。
                   ・該当コーナーの特定と、ショートカット等の異常ラップ除外の結果（カンペ情報に基づく）。
                   ・【対象ラップの絞り込み結果】指定された「{target_lap_mode}」に基づき、最も速いベストラップを主軸にしたことを明記。
                   ・【採用したGデータの根拠】ログデータから、「優先度1: Gセンサー実測値 ＞ 優先度2: 車速センサー計算値 ＞ 優先度3: GPS速度計算値」の優先順位で最も精度の高いものを1つ選択し、「今回は〇〇データが存在するため、最も精度が高いと判断して採用します」と明記すること。以降のG数値はすべてその選択したデータソースを使うこと。
                   ・【★ブレーキング減速G トップ3箇所（各3ラップ比較）】※ユーザーの「発生フェーズ」の指定に関わらず、全体把握のために必ず記載すること。コース上で減速Gが大きい上位3箇所のブレーキングポイント（推測したコーナー名）を特定し、各ポイントについて「ベストラップの数値」と「他の2ラップの数値」の計3ラップ分（3箇所×3ラップ＝計9データ程度）を並べてリストアップすること。
                   ・【★ストレート加速G トップ3ラップ】※ユーザーの「発生フェーズ」の指定に関わらず必ず記載すること。コントロールライン前後、またはコース上で最も長いストレートにおける「最大加速G」を特定し、数値が高かった上位3ラップの数値（コーナー立ち上がりの比較材料）をリストアップすること。
                   ・【最大ストロークと安定平均位置】コース全周において前後が最大ストロークした場所を明記。ギャップ除外の安定平均ストロークの数値を引き合いに出し、エアバネの特性評価に使用すること。
                2. [旋回を引き出す前後動作]
                   ※【厳守事項】ロギングの数値を語る際、必ず「▼ Lap O - 第1ヘアピン進入」のように直感的なコーナー名を明示すること。
                   ・ブレーキング分析: (評価結果を記載。ここでAIへ渡されたカンペを参照し、必ずベストラップのデータにフォーカスして解説すること)
                   ・倒しこみ動作: (評価結果を記載)
                   ・初期旋回操舵: (評価結果を記載)
                   ・後期旋回操舵: (評価結果を記載)
                3. [加速状態での旋回操舵]
                   (トラクションと車体の乱れに関する評価結果を記載。ここでも必要に応じて直感的なコーナー名を明記)
                4. [燃調の影響と総合解決策]
                   (燃調の確認結果と修正案を記載)
                5. [課題解決のための操舵技術とサスセッティング案]
                   (悩み「{user_comment}」を解決し最高の旋回性を引き出すための、意識するべき操舵技術とセッティング案を提示)

                以下は計算された【反力テーブル】です：
                {stroke_range_text}

                以下はPythonデータエンジニアによって事前処理された【走行ログ要約データ】です：
                {log_contents}
                """

                api_request_data = [full_prompt]
                if uploaded_images:
                    for img_file in uploaded_images:
                        img = Image.open(img_file)
                        api_request_data.append(img)

                response = model.generate_content(api_request_data)

                st.success("✅ 解析が完了しました！")
                st.markdown("### 🏁 AIエンジニアの診断結果")
                st.write(response.text)

            except Exception as e:
                st.error(f"❌ 解析中にエラーが発生しました: {e}")

st.caption("※解析にはGoogle Gemini APIを使用しています。入力されたデータは今回限りの解析のみに使用されます。")
