import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math
import json
import pandas as pd # AI連携のCSV出力用

# --- 1. ページ設定（※絶対に一番最初に書く） ---
st.set_page_config(page_title="フロントサスシミュレーター/AI連携ツール", layout="wide")

# ===============================
# Session State（状態保存）の初期化
# ===============================
default_params = {
    "setting_name": "260315_基本セット", # ★ファイル名用の変数を追加
    "m_bike": 90.0, "m_rider": 64.0, "caster_angle": 25.0, "decel_g": 1.25,
    "fork_id": 36.0, "x_max": 97.0, "oil_lock_len": 10.0,
    "k_init": 0.37, "k_late": 0.98, "s_change": 82.0, "preload": 27.0,
    "n_index": 2.40, "oil_base": 60, "oil_comp": 70, "target_stroke": 50.0
}
# アプリ起動時に1回だけ初期値をセットする
for key, val in default_params.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ===============================
# 2. タイトルと案内文
# ===============================
st.title("フロントサスシミュレーター/AI連携ツール")
st.caption("YouTubeチャンネル『こぼれ小話 タミケンバーン』連動ツール")

st.info("""
YouTubeチャンネル『こぼれ小話 タミケンバーン』連動ツール、素人構築にて精度向上検証中です。
異常値報告等ご指摘に数値共有などは、下記チャンネルのフロントサスシミュレーター関連の動画コメント欄へお願いいたします。
""")

st.markdown("""
▶ [ばねレート簡易判定ツール v2.5 はこちら](https://spring-rate-tool.streamlit.app/)  
▶ [YouTube：こぼれ小話タミケンバーン チャンネルTOP](https://www.youtube.com/@dogtamy-Lean-burn)
""")

# ★追加：セッティング名の入力（①の上）
setting_name_input = st.text_input("📝 セッティング名（保存ファイル名に反映されます）", value=st.session_state["setting_name"])
st.session_state["setting_name"] = setting_name_input

st.divider()

# ===============================
# ★サイドバー（フロントセッティング管理）
# ===============================
with st.sidebar:
    st.header("💾 フロントセッティング管理")
    
    # 1. 保存機能
    current_settings = {k: st.session_state[k] for k in default_params.keys()}
    json_str = json.dumps(current_settings, indent=4)
    # 入力されたセッティング名をファイル名に適用
    export_file_name = f"{st.session_state['setting_name']}.json"
    
    st.download_button(
        label="現在の設定を保存 (.json)",
        data=json_str,
        file_name=export_file_name,
        mime="application/json"
    )
    st.caption("※現在の各数値をファイルとして保存します。")

    st.divider()

    # 2. 読込機能
    uploaded_file = st.file_uploader("設定ファイルを読み込む", type="json")
    if uploaded_file is not None:
        try:
            loaded_data = json.load(uploaded_file)
            for k, v in loaded_data.items():
                if k in st.session_state:
                    st.session_state[k] = v
            st.success(f"「{st.session_state.get('setting_name', '設定')}」を読み込みました！")
        except Exception as e:
            st.error("ファイルの読み込みに失敗しました。形式を確認してください。")

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
    decel_g = st.number_input("最大減速G (1.0〜1.5G推奨)", 0.5, 2.0, value=float(st.session_state["decel_g"]), step=0.01)
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
st.header("⑤ AI解析用プロンプト自動生成（Gemini連携用）")
st.info("このセクションで入力した情報とシミュレーターの数値を合体させ、Gemini（AI）へ渡す完璧な指示書を自動生成します。")

# ★追加：解析モードの選択
analysis_mode = st.radio(
    "📊 解析モードを選択してください",
    ["単一データ解析（1つのDroggerログを深く分析）", "2つのデータ比較解析（A/Bテスト・仕様違いの比較）"],
    horizontal=True
)

col_ai1, col_ai2 = st.columns(2)

with col_ai1:
    st.subheader("A. セッティングの体感・状態入力")
    
    if "単一" in analysis_mode:
        track_name = st.text_input("サーキット名（任意）", value="", placeholder="例：近畿スポーツランド、鈴鹿サーキット")
        tire_info = st.text_input("タイヤ銘柄・状態（任意）", value="", placeholder="例：BT-601SS フロント逆履き 5時間使用")
        st.write("ダンピング（減衰）の体感 ※1:最弱/速い 〜 10:最強/遅い")
        comp_level = st.slider("圧側（コンプ）感覚", 1, 10, 5)
        reb_level = st.slider("伸び側（リバウンド）感覚", 1, 10, 5)
    else:
        # 比較モード時の入力欄
        st.write("💡 **比較する2つのデータ（CSV）の違いをメモしてください。**")
        data_a_memo = st.text_area("Data A (基準) の条件・ファイル名", value="例：2026-01-18_dlog.csv (バネ5.5Nm, 油面54mm)")
        data_b_memo = st.text_area("Data B (比較) の条件・ファイル名", value="例：2025-12-20_dlog.csv (JC92純正, 油面65mm)")
        track_name = st.text_input("サーキット名（任意）", value="", placeholder="例：近畿スポーツランド")
        tire_info = st.text_input("タイヤ情報など共通事項", value="", placeholder="例：タイヤは両データとも新品BT-601SS")

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
    st.caption("⚠️ **注意:** ロガーアプリとCSVデータで「Lap数」がズレている場合があります。特定の周回を指定する場合はご注意ください。")
    user_comment = st.text_area(
        "具体的な悩み・知りたいこと", 
        value="例：Lap 9の1コーナー進入でフロントが突っ張り、戻ってこない感覚がある。"
    )

st.write("---")
st.subheader("C. 出力とAIへの指示生成")

col_out1, col_out2, col_out3 = st.columns([1, 1, 1.5])

# 【CSV出力機能】
with col_out1:
    st.write("**STEP 1: 反力テーブルの出力**")
    st.caption("シミュレーターで計算したストロークごとの反力データをCSVでダウンロードします。")
    
    stroke_range = np.arange(0, x_max + 1, 1.0)
    df_export = pd.DataFrame({
        "Stroke_mm": stroke_range,
        "Total_Force_Base_2pcs_N": [total_f_2pcs(x, oil_base) * 9.80665 for x in stroke_range],
        "Total_Force_Comp_2pcs_N": [total_f_2pcs(x, oil_comp) * 9.80665 for x in stroke_range]
    })
    
    csv_data = df_export.to_csv(index=False).encode('utf-8')
    export_csv_name = f"ForceTable_{st.session_state.get('setting_name', 'data')}.csv"
    st.download_button("反力テーブルをダウンロード", data=csv_data, file_name=export_csv_name, mime="text/csv")

# 【燃調マップ添付の案内】
with col_out2:
    st.write("**STEP 2: 燃調マップの準備（任意）**")
    st.caption("セッティングツール等の「燃調マップ画面のスクショ」があれば用意してください。AIから燃調マップの具体的な変更数値の提案が可能になります。")

# 【プロンプト生成機能】
with col_out3:
    st.write("**STEP 3: プロンプトのコピー**")
    st.caption("以下のテキストをコピーしてください。")
    
    if "単一" in analysis_mode:
        prompt_text = f"""あなたはワークスチームのチーフ・サスペンションエンジニア 兼 エンジンチューナーです。
以下の【セッティング情報】と添付ファイル【反力テーブル(CSV)】【走行ログ(CSV)】【燃調マップ画像(任意)】を掛け合わせ、論理的な解析とアドバイスを行ってください。

【車両・サスセッティング情報】
・サーキット名: {track_name}
・ライダー込重量: {total_m} kg / 想定最大減速G: {decel_g} G
・バネレート: {k_init} kg/mm (後半: {k_late} kg/mm)
・プリロード: {preload} mm / 油面: {oil_base} mm
・減衰感覚(1-10): 圧側 {comp_level} / 伸び側 {reb_level}
・タイヤ情報: {tire_info}

【ターゲット課題】
・走行状況: {run_condition}
・発生フェーズ: {phase_selection}
・具体的な悩み: {user_comment}

【解析・回答のステップ】
AIは以下の1〜4の順序で必ず思考し、結果を出力してください。
1. [データ抽出とコース把握]: 走行ログにGPSデータがある場合は該当コーナーを特定し（固有名詞を使用）、異常ラップは除外すること。
2. [旋回を引き出す前後動作]: ブレーキングから倒し込みにかけてや切り返しにて、フロントの沈み込みスピードとリアの挙動が連動し、「鋭い旋回性を引き出すためのピッチングモーション（動的な姿勢変化）」や「旋回の適切な前後車高バランス」が作れているか波形から評価すること。
3. [トラクションと車体の乱れ]: アクセルON時のリアの沈み込みとフロントの伸びを分析し、トラクション抜けや、サスの反発による車体の乱れ（跳ねやチャタリング）が発生していないか確認すること。
4. [燃調の影響と総合解決策]: AfrやRpmからトルク変動の悪影響がないか確認し、【走行状況】も加味した上で、「{user_comment}」を解決し最高の旋回性を引き出すためのサスセッティング案と、必要に応じた燃調修正案を提示すること。"""

    else:
        prompt_text = f"""あなたはワークスチームのチーフ・サスペンションエンジニア 兼 エンジンチューナーです。
添付された【2つの走行ログ(CSV)】と【反力テーブル(CSV)】【燃調マップ画像(任意)】を比較解析し、論理的なアドバイスを行ってください。

【比較するデータ条件（差分メモ）】
・サーキット名: {track_name}
・Data A (基準): {data_a_memo}
・Data B (比較): {data_b_memo}
・タイヤ等共通情報: {tire_info}

【ターゲット課題】
・走行状況: {run_condition}
・発生フェーズ: {phase_selection}
・具体的な悩み: {user_comment}

【解析・回答のステップ】
1. [データ比較とコース把握]: 添付された2つの走行ログにGPSデータがある場合は該当コーナーを特定し（固有名詞を使用）、Data AとBのそれぞれの波形を抽出・比較すること。異常ラップは除外すること。
2. [A/B 旋回を引き出す前後動作の差]: ブレーキングから倒し込みにかけてや切り返しにて、どちらのデータがより「鋭い旋回性を引き出すピッチングモーション」と「旋回の適切な前後車高バランス」を作れているか比較評価すること。
3. [A/B トラクションと車体の乱れの差]: アクセルON時のトラクションの掛かり方や、車体の乱れ（跳ね・チャタリング）の有無について、Data AとBでどちらが優れているか比較分析すること。
4. [燃調の影響と総合結論]: 燃調データの悪影響の有無を確認し、【走行状況】も加味した上で、「{user_comment}」に対する結論として次に試すべきセッティング案（AとBの良い所取りなど）と燃調修正案を提示すること。"""

    st.code(prompt_text, language="")

# 【Gemini送信前の最終チェックリスト】
st.write("---")
st.info("💡 **Geminiへ投げる準備はできましたか？ 以下のデータをまとめてGeminiに貼り付けて送信してください。**\n"
        "1. 📋 **コピーしたプロンプト**（STEP 3でコピーしたもの）\n"
        "2. 📊 **反力テーブル.csv**（STEP 1でダウンロードしたもの）\n"
        "3. 📈 **走行ログ.csv**（比較モードの場合は Data A と Data B の **2つ** を添付）\n"
        "4. 🖼️ **燃調マップ画面のスクショ**（任意：用意した場合のみ）")
