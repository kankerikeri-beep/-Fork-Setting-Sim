import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math
import json
import pandas as pd
import google.generativeai as genai  # ★追加：Gemini連携用ライブラリ

# --- 1. ページ設定（※絶対に一番最初に書く） ---
st.set_page_config(page_title="タミケンシム - フロントサスシミュレーター/AI分析ツール", layout="wide")

# ===============================
# Session State（状態保存）の初期化
# ===============================
# ①シミュレーター用パラメータ
default_params = {
    "setting_name": "260315_基本セット",
    "m_bike": 90.0, "m_rider": 68.0, "caster_angle": 25.0, "decel_g": 1.7,
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
# ★サイドバー：セッティング管理の集約
# ===============================
with st.sidebar:
    st.title("💾 セッティング管理")
    
    # -------------------------------------
    # 1. フロントセッティング管理 (ばねレート共通可)
    # -------------------------------------
    st.header("1. フロントセッティング")
    st.caption("シミュレーター用の基本設定（ばねレートツールと共通読込可）")
    
    current_settings = {k: st.session_state[k] for k in default_params.keys()}
    json_str = json.dumps(current_settings, indent=4)
    export_file_name = f"{st.session_state['setting_name']}.json"
    
    st.download_button(
        label="フロント設定を保存 (.json)",
        data=json_str,
        file_name=export_file_name,
        mime="application/json"
    )
    
    uploaded_file = st.file_uploader("フロント設定を読込", type="json")
    if uploaded_file is not None:
        try:
            loaded_data = json.load(uploaded_file)
            for k, v in loaded_data.items():
                if k in st.session_state:
                    st.session_state[k] = v
            st.success("フロント設定を読み込みました！")
        except Exception:
            st.error("読込失敗。ファイル形式を確認してください。")

    st.divider()

    # -------------------------------------
    # 2. AI解析用・セッティング (専用)
    # -------------------------------------
    st.header("2. AI解析用・セッティング")
    st.caption("AI連携用の車高や減衰目安（専用読込ファイル）")
    
    ai_save_keys = list(default_ai_params.keys())
    ai_settings = {k: st.session_state[k] for k in ai_save_keys if k in st.session_state}
    ai_json_str = json.dumps(ai_settings, indent=4)
    export_ai_name = f"AI_Prompt_{st.session_state['setting_name']}.json"
    
    st.download_button(
        label="AI設定を保存 (.json)",
        data=ai_json_str,
        file_name=export_ai_name,
        mime="application/json",
        key="ai_save_btn"
    )
    
    uploaded_ai_file = st.file_uploader("AI設定を読込（専用）", type="json", key="ai_load_btn")
    if uploaded_ai_file is not None:
        try:
            loaded_ai_data = json.load(uploaded_ai_file)
            for k, v in loaded_ai_data.items():
                if k in ai_save_keys and k in st.session_state:
                    st.session_state[k] = v
            st.success("AI設定を読み込みました！")
        except Exception:
            st.error("読込失敗。ファイル形式を確認してください。")


# ===============================
# メイン画面：タイトル（CSSで画像を再現）
# ===============================
title_design = """
<style>
    /* タイトル（タミケンシム）のデザイン */
    .tamiken-title {
        font-family: "Hiragino Mincho ProN", "MS Mincho", serif;
        font-size: 80px !important;
        font-weight: 900 !important;
        color: #E60000 !important; /* ベースは力強い赤 */
        text-shadow:
            /* 1層目：極太の黄色フチ（Gold系で見やすく） */
            -3px -3px 0 #FFD700,  3px -3px 0 #FFD700,
            -3px  3px 0 #FFD700,  3px  3px 0 #FFD700,
            -3px  0px 0 #FFD700,  3px  0px 0 #FFD700,
             0px -3px 0 #FFD700,  0px  3px 0 #FFD700,
            /* 2層目：背景が白でも目立つように、薄い黒の影を落とす */
             4px  4px 6px rgba(0, 0, 0, 0.5);
        letter-spacing: -2px;
        margin-bottom: 0px;
        padding-bottom: 0px;
        line-height: 1.2;
    }

    /* サブタイトルのデザイン */
    .tamiken-subtitle {
        font-family: sans-serif;
        font-size: 30px !important;
        font-weight: bold !important;
        /* ★修正：背景に合わせて白/黒が自動で切り替わります！ */
        color: var(--text-color) !important;
        margin-top: 5px;
        margin-bottom: 25px;
    }
</style>
"""

# HTML/CSSをアプリに反映
st.markdown(title_design, unsafe_allow_html=True)

# 実際のタイトル・サブタイトルを表示
st.markdown('<p class="tamiken-title">タミケンシム</p>', unsafe_allow_html=True)
st.markdown('<p class="tamiken-subtitle">フロントサスシミュレーター / AI分析ツール（『こぼれ小話 タミケンバーン』連動）</p>', unsafe_allow_html=True)


# --- 案内文 ---
st.info("""
YouTubeチャンネル『こぼれ小話 タミケンバーン』連動ツール、素人構築にて精度向上検証中です。
異常値報告等ご指摘に数値共有などは、下記チャンネルのフロントサスシミュレーター関連の動画コメント欄へお願いいたします。
""")
st.markdown("""
▶ [ばねレート簡易判定ツール v2.5 はこちら](https://spring-rate-tool.streamlit.app/)  
▶ [YouTube：こぼれ小話タミケンバーン チャンネルTOP](https://www.youtube.com/@dogtamy-Lean-burn)
""")


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
    st.caption("※フル制動時のキャスター変化を含めて荷重移動を計算します。")
    decel_g = st.number_input("最大減速G (1.0〜1.9G推奨)", 0.5, 2.0, value=float(st.session_state["decel_g"]), step=0.01)
    st.session_state["decel_g"] = decel_g

with col_v3:
    rad = math.radians(caster_angle)
    f_target_total_kg = (total_m * math.cos(rad) + total_m * decel_g * math.sin(rad)) * 1.18
    st.metric("フォーク全体への想定最大荷重", f"{f_target_total_kg:.1f} kg")
    st.write("**（車重＋G換算＋重心移動補正：1.18）**")

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
    if k_late == 0 or s_change <= 0:
        return k_init * x_total
    if x_total <= total_change:
        return k_init * x_total
    else:
        return k_init * total_change + k_late * (x_total - total_change)

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
        if total_f_2pcs(sx, oil) >= target_f:
            return max(0.0, x_max - sx)
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
    y_s = [get_spring_f(x) * 2 for x in x_plot]
    fig.add_trace(go.Scatter(x=x_plot, y=y_s, name="金属バネ(両側)", line=dict(dash='dash', color='silver')))

if show_air:
    fig.add_trace(go.Scatter(x=x_plot, y=[get_air_f(x, oil_base) * 2 for x in x_plot], name=f"エア単体({oil_base}mm)", line=dict(color='lightblue', width=2)))
    fig.add_trace(go.Scatter(x=x_plot, y=[get_air_f(x, oil_comp) * 2 for x in x_plot], name=f"エア単体({oil_comp}mm)", line=dict(color='pink', width=2)))

if show_total:
    fig.add_trace(go.Scatter(x=x_plot, y=[total_f_2pcs(x, oil_base) for x in x_plot], name="基準合成", line=dict(color="blue", width=4)))
    fig.add_trace(go.Scatter(x=x_plot, y=[total_f_2pcs(x, oil_comp) for x in x_plot], name="比較合成", line=dict(color="red", width=4)))

fig.add_vrect(x0=x_max - oil_lock_len, x1=x_max, fillcolor="gray", opacity=0.1, layer="below", annotation_text="オイルロック域")
fig.add_hline(y=f_target_total_kg, line_dash="dot", line_color="green", annotation_text="想定最大荷重")
fig.update_layout(xaxis_title="ストローク量 [mm]", yaxis_title="荷重 (2本合計) [kg]", template="simple_white", height=600)
st.plotly_chart(fig, use_container_width=True)

# ===============================
# ★新機能：AI連携用データ作成 ＆ プロンプト生成
# ===============================
st.divider()
st.header("⑤ AI解析用プロンプト自動生成（AI連携用）")
st.info("このセクションで入力した情報とシミュレーターの数値を合体させ、AIへ完璧な指示を自動で送信します。")

# 解析モードの選択
analysis_mode = st.radio(
    "📊 解析モードを選択してください",
    ["単一データ解析（1つの走行ログを深く分析）", "2つのデータ比較解析（A/Bテスト・仕様違いの比較）"],
    horizontal=True,
    key="analysis_mode_radio"
)

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
            tire_temp = st.number_input("測定時タイヤ温度 [℃]", value=25, step=1)
        with c_tp2:
            tire_p_front = st.number_input("フロント空気圧", value=1.80, step=0.05)
            track_temp = st.number_input("路面温度 [℃] (任意)", value=30, step=1)
        with c_tp3:
            tire_p_rear = st.number_input("リア空気圧", value=1.90, step=0.05)
            track_cond = st.selectbox("路面状況", ["ドライ", "ハーフウェット", "ウェット"])
        
        st.markdown("##### 🏍️ フロント設定")
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
        track_name = st.text_input("サーキット名（任意）", value="", placeholder="例：近畿スポーツランド")
        tire_info = st.text_area("タイヤ・その他共通設定（前後サス・ディメンション・電子制御等）", value="", placeholder="例：前後タイヤ新品。TCはレベル2固定。")

with col_ai2:
    st.subheader("B. 解析したい課題と状況の入力")
    
    run_condition = st.selectbox(
        "走行状況（タイムや走り方への影響）",
        ["単独走行（マイペースでのアタック・クリアラップ）", "追い走行（前走者をターゲットにしたアタック）", "混走・トラフィックあり（ペースの乱れあり）"]
    )
    phase_selection = st.selectbox(
        "課題が発生しているフェーズ（場所）",
        ["進入・フルブレーキング", "旋回中・コーナリング", "切り返し・S字", "立ち上がり・アクセルオン", "ストレート・全開加速"]
    )
    st.caption("⚠️ **注意:** ロガーの画面とCSVデータで「Lap数」がズレている場合があります。特定の周回を指定する場合はご注意ください。")
    user_comment = st.text_area(
        "具体的な悩み・知りたいこと（★良いと感じている部分もあれば記載）", 
        value="例：Lap 9の1コーナー進入でフロントが戻ってこない感覚がある。逆にS字の切り返しは軽快で良い感じなので、そこは犠牲にしたくない。",
        height=150
    )

# --- プロンプトの構築（裏側用） ---
if "単一" in analysis_mode:
    prompt_text = f"""あなたはワークスチームのチーフ・サスペンションエンジニア 兼 データエンジニアです。
以下の【セッティング情報】と添付ファイル【反力テーブル(CSV)】【走行ログ(CSV)】【燃調マップ等画像(任意)】を掛け合わせ、論理的な解析とアドバイスを行ってください。

【車両・環境・サスセッティング情報】
・サーキット名: {track_name}
・ライダー込重量: {total_m} kg / 想定最大減速G: {decel_g} G
・キャスター角(静止時ベース): {caster_angle} deg

[車体・電子制御・ブレーキ]
・ブレーキパッド特性: {brake_pad}
・ステアリングダンパー: {steering_damper}
・トラクションコントロール: 基本 {tc_base} (個別指定: {tc_memo})
・エンジンブレーキ(EBC): {eb_base}

[タイヤ・空気圧設定]
・銘柄/状態: {tire_info}
・空気圧 (測定温度 {tire_temp}℃): F {tire_p_front} / R {tire_p_rear} [{tire_p_unit}]
・路面環境: 路面温度 {track_temp}℃ / コンディション: {track_cond}

[フロント仕様・設定]
・バネレート: {k_init} kg/mm (後半: {k_late} kg/mm) / プリロード: {preload} mm / 油面: {oil_base} mm
・突き出し量: {front_protrusion} mm
・ロガー計測値: フリー(全伸び) {front_stroke_free} mm / ボトム(最小) {front_stroke_bottom} mm
・減衰目安(1-10): 圧側 {comp_level} / 伸び側 {reb_level}

[リア仕様・設定]
・バネレート: 初期 {rear_k_init} kg/mm, 後半 {rear_k_late} kg/mm (変化点: {rear_rate_change} mm)
・プリロード: {rear_preload} mm / 車高調整(サス単体セット長): {rear_ride_height} mm
・ロガー計測値: フリー(全伸び) {rear_stroke_free} mm / ボトム(最小) {rear_stroke_bottom} mm
・減衰目安(1-10): 圧側 {rear_comp_level} / 伸び側 {rear_reb_level}

【ターゲット課題】
・走行状況: {run_condition}
・発生フェーズ: {phase_selection}
・具体的な悩み: {user_comment}

【解析・回答のステップ】
AIは以下の1〜5の順序で必ず思考し、結果を出力してください。
1. [データ抽出とコース把握]: 走行ログにGPSデータがある場合は該当コーナーを特定し（固有名詞を使用）、異常ラップは除外すること。
2. [旋回を引き出す前後動作]: ブレーキングから倒し込みにかけてや切り返しにて、沈み込みの挙動が「鋭い旋回性を引き出すためのピッチングモーション（動的な姿勢変化）」や「バンク時に適切な車高から生まれる操舵状態」を作り出せているか「ブレーキ終盤からのフロント操舵」及び「ブレーキ操作と荷重コントロールによるリア操舵」の瞬間的な挙動の状態を各種波形から評価すること。
3. [トラクションと車体の乱れ]: アクセルON時のリアの沈み込みとフロントの伸びを分析し、「効率の良い加速状態を作り出せているか」、トラクション抜けや、サスの反発による車体の乱れ（跳ねやチャタリング）が発生していないか確認と評価すること。（※空気圧の影響やTC・ステダンの介入状況も推測すること）
4. [燃調の影響]: AfrやRpmからトルク変動の悪影響がないか確認し、【走行状況】も加味した上で必要に応じた燃調修正案を提示すること。
5. [総合解決策の提示]: 具体的な悩み・知りたいこと「{user_comment}」を解決し最高の旋回性を引き出すための「意識するべき操舵技術」と「サスセッティング案（空気圧・電子制御含む）」、必要であればフロントやリアのストローク位置で予測される荷重数値から分析提案できる数値等を提示すること。"""

else:
    prompt_text = f"""あなたはワークスチームのチーフ・サスペンションエンジニア 兼 データエンジニアです。
添付された【2つの走行ログ(CSV)】と【反力テーブル(CSV)】【燃調マップ等画像(任意)】を比較解析し、論理的なアドバイスを行ってください。

【比較するデータ条件（差分メモ）】
・サーキット名: {track_name}
・キャスター角(静止時ベース): {caster_angle} deg
・Data A (基準): {data_a_memo}
・Data B (比較): {data_b_memo}
・タイヤ・電子制御等共通情報: {tire_info}

【ターゲット課題】
・走行状況: {run_condition}
・発生フェーズ: {phase_selection}
・具体的な悩み: {user_comment}

【解析・回答のステップ】
AIは以下の1〜5の順序で必ず思考し、結果を出力してください。
1. [データ比較とコース把握]: 添付された2つの走行ログにGPSデータがある場合は該当コーナーを特定し（固有名詞を使用）、Data AとBのそれぞれの波形を抽出・比較すること。異常ラップは除外すること。
2. [A/B 旋回を引き出す前後動作の差]: ブレーキングから倒し込みにかけてや切り返しにて、どちらのデータがより沈み込みの挙動から「鋭い旋回性を引き出すためのピッチングモーション（動的な姿勢変化）」や「バンク時に適切な車高から生まれる操舵状態」を作り出せているか、「ブレーキ終盤からのフロント操舵」及び「ブレーキ操作と荷重コントロールによるリア操舵」の瞬間的な挙動の状態を各種波形から比較評価すること。
3. [A/B トラクションと車体の乱れの差]: アクセルON時のリアの沈み込みとフロントの伸びを分析し、どちらが「効率の良い加速状態を作り出せているか」、トラクション抜けや、サスの反発による車体の乱れ（跳ねやチャタリング）の有無について比較確認と評価すること。（※空気圧の影響やTC・ステダンの介入状況も推測すること）
4. [燃調の影響]: AfrやRpmからトルク変動の悪影響がないか確認し、【走行状況】も加味した上で必要に応じた燃調修正案を提示すること。
5. [総合結論とセッティング案]: 具体的な悩み・知りたいこと「{user_comment}」に対する結論として、最高の旋回性を引き出すための「意識するべき操舵技術」と次に試すべき「トータルセッティング案（AとBの良い所取り、サス・空気圧・電子制御含む）」、必要であればフロントやリアのストローク位置で予測される荷重数値から分析提案できる数値等を提示すること。"""


# ===============================
# ★新機能：AIへのデータ送信と解析実行
# ===============================
st.write("---")
st.subheader("C. AIデータ解析実行")

col_out1, col_out2 = st.columns([1, 1.5])

with col_out1:
    st.write("**STEP 1: 反力テーブルの確認（任意）**")
    st.caption("シミュレーターで計算した反力データです。裏側でAIに自動送信されますが、手元に残したい場合はダウンロードしてください。")
    
    stroke_range = np.arange(0, x_max + 1, 1.0)
    df_export = pd.DataFrame({
        "Stroke_mm": stroke_range,
        "Total_Force_Base_2pcs_N": [total_f_2pcs(x, oil_base) * 9.80665 for x in stroke_range],
        "Total_Force_Comp_2pcs_N": [total_f_2pcs(x, oil_comp) * 9.80665 for x in stroke_range]
    })
    
    csv_data = df_export.to_csv(index=False).encode('utf-8')
    export_csv_name = f"ForceTable_{st.session_state.get('setting_name', 'data')}.csv"
    st.download_button("反力テーブルをダウンロード", data=csv_data, file_name=export_csv_name, mime="text/csv")

with col_out2:
    st.write("**STEP 2: 走行ログ(CSV)のアップロード**")
    if "単一" in analysis_mode:
        uploaded_logs = st.file_uploader("走行ログ(CSV)をアップロードしてください", type="csv", accept_multiple_files=False)
    else:
        uploaded_logs = st.file_uploader("比較する2つの走行ログ(CSV)をアップロードしてください", type="csv", accept_multiple_files=True)

st.write("---")
st.write("**STEP 3: 解析実行**")

# AI解析実行ボタン
if st.button("AIに解析させる（数十秒かかります）", type="primary"):
    if not uploaded_logs:
        st.warning("⚠️ 走行ログ(CSV)をアップロードしてください。")
    elif "GEMINI_API_KEY" not in st.secrets:
        st.error("⚠️ APIキーが設定されていません。StreamlitのSecretsを確認してください。")
    else:
        with st.spinner("ワークスエンジニアがデータを解析中..."):
            try:
                # APIの初期化
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                model = genai.GenerativeModel('gemini-2.0-flash')

                # 反力テーブルのテキスト化
                stroke_range_text = df_export.to_csv(index=False)

                # 走行ログの読み込みとテキスト化
                log_contents = ""
                if isinstance(uploaded_logs, list):
                    for uploaded_log in uploaded_logs:
                        log_df = pd.read_csv(uploaded_log)
                        log_contents += f"\n--- File: {uploaded_log.name} ---\n"
                        log_contents += log_df.to_csv(index=False)
                else:
                    log_df = pd.read_csv(uploaded_logs)
                    log_contents += f"\n--- File: {uploaded_logs.name} ---\n"
                    log_contents += log_df.to_csv(index=False)

                # すべてのデータを合体させて裏側で送信
                full_prompt = f"""
                {prompt_text}

                以下は計算された【反力テーブル】です：
                {stroke_range_text}

                以下はアップロードされた【走行ログデータ】です：
                {log_contents}
                """

                # AIへリクエスト
                response = model.generate_content(full_prompt)

                # 結果表示
                st.success("✅ 解析が完了しました！")
                st.markdown("### 🏁 AIエンジニアの診断結果")
                st.write(response.text)

            except Exception as e:
                st.error(f"❌ 解析中にエラーが発生しました: {e}")

st.caption("※解析にはGoogle Gemini APIを使用しています。入力されたデータは今回限りの解析のみに使用されます。")
