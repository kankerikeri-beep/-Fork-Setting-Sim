import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math

st.set_page_config(page_title="フロントフォークエアバネシミュレーター v2.7", layout="wide")

st.title("フロントフォークエアバネシミュレーター")
st.caption("YouTubeチャンネル『こぼれ小話 タミケンバーン』連動ツール")
st.markdown("""
▶ [ばねレート判定ツールはこちら](https://share.streamlit.io/...)  
▶ [使用方法解説動画（YouTube）](https://www.youtube.com/watch?v=...)
""")
st.divider()

# ===============================
# 1. 車両・ライディング条件
# ===============================
st.header("① 車両・ライディング条件")
col_v1, col_v2, col_v3 = st.columns(3)

with col_v1:
    m_bike = st.number_input("車体重量 [kg]", 0.0, 300.0, 90.0, step=1.0)
    m_rider = st.number_input("装備体重 [kg]", 0.0, 200.0, 64.0, step=1.0)
    total_m = m_bike + m_rider
with col_v2:
    caster_angle = st.number_input("キャスター角 (静止時) [deg]", 0.0, 45.0, 25.0, step=0.1)
    decel_g = st.number_input("最大減速G (1.0〜1.5G推奨)", 0.5, 2.0, 1.25, step=0.01)
with col_v3:
    rad = math.radians(caster_angle)
    # 荷重補正係数 1.18 (重心移動補正)
    f_target_total_kg = (total_m * math.cos(rad) + total_m * decel_g * math.sin(rad)) * 1.18
    st.metric("フォーク全体への想定荷重", f"{f_target_total_kg:.1f} kg")

# ===============================
# 2. フォーク・バネ内部仕様
# ===============================
st.header("② フォーク・バネ内部仕様")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    fork_id = st.number_input("フォーク内径 [mm]", 10.0, 60.0, 36.0, step=0.1)
    x_max = st.number_input("最大ストローク [mm]", 1.0, 250.0, 97.0, step=1.0)
    oil_lock_len = st.number_input("オイルロック長（フルストロークからの残り） [mm]", 0.0, 50.0, 10.0, step=1.0)
with col_f2:
    k_init = st.number_input("初期レート (1本分) [kg/mm]", 0.0, 20.0, 0.37, step=0.01)
    k_late = st.number_input("後半レート (1本分) [kg/mm]", 0.0, 20.0, 0.98, step=0.01)
    s_change = st.number_input("レート変化点 [mm]", 0.0, 250.0, 82.0, step=1.0)
with col_f3:
    preload = st.number_input("プリロード [mm]", 0.0, 50.0, 27.0, step=1.0)
    n_index = st.slider("空気断熱指数 n", 1.0, 3.0, 2.40, step=0.01)

# ===============================
# 3. 油面・任意ストローク荷重調査
# ===============================
st.header("③ 油面比較 ＆ ストローク位置の荷重調査")
col_o1, col_o2 = st.columns(2)
with col_o1:
    oil_base = st.slider("基準油面 [mm]", 10, 200, 60)
    oil_comp = st.slider("比較油面 [mm]", 10, 200, 70)
with col_o2:
    target_stroke = st.number_input("調査したいストローク位置 [mm]", 0.0, float(x_max), 50.0, step=1.0)

# ===============================
# 計算ロジック
# ===============================
area_air = math.pi * (fork_id / 2)**2

def get_spring_f(x):
    """金属バネ1本分の反力"""
    x_total = x + preload
    total_change = s_change + preload
    if k_late == 0 or s_change <= 0:
        return k_init * x_total
    if x_total <= total_change:
        return k_init * x_total
    else:
        f_at_change = k_init * total_change
        return f_at_change + k_late * (x_total - total_change)

def get_air_f(x, air_space_at_full):
    """エアバネ1本分の反力"""
    p0 = 1.033 
    L0 = (air_space_at_full + x_max) * 0.95 
    L_current = L0 - x
    if L_current <= 0.1: return 3000.0
    p1 = p0 * ((L0 / L_current) ** n_index)
    force = (p1 - p0) * (area_air / 100)
    return force

# ===============================
# 4. シミュレーション結果
# ===============================
st.header("④ シミュレーション結果比較")

def get_total_f_2pcs(x, oil): return (get_spring_f(x) + get_air_f(x, oil)) * 2

c1, c2 = st.columns(2)
with c1:
    f_at_stroke_base = get_total_f_2pcs(target_stroke, oil_base)
    st.metric(f"基準油面 ({oil_base}mm)", f"{f_at_stroke_base:.1f} kg")
    st.write(f"ストローク {target_stroke}mm 時の**合成荷重**")

with c2:
    f_at_stroke_comp = get_total_f_2pcs(target_stroke, oil_comp)
    st.metric(f"比較油面 ({oil_comp}mm)", f"{f_at_stroke_comp:.1f} kg", delta=f"{f_at_stroke_comp - f_at_stroke_base:.1f} kg")
    st.write(f"ストローク {target_stroke}mm 時の**合成荷重**")

# 表示スイッチ
st.write("---")
st.subheader("グラフ表示設定")
show_spring = st.checkbox("金属バネ反力を表示", value=True)
show_air = st.checkbox("エアバネ反力を表示", value=False)
show_total = st.checkbox("合成反力を表示", value=True)

# グラフ描画
x_plot = np.linspace(0, x_max, 500)
fig = go.Figure()

if show_spring:
    y_spring = [get_spring_f(x) * 2 for x in x_plot]
    fig.add_trace(go.Scatter(x=x_plot, y=y_spring, name="金属バネ反力", line=dict(dash='dash', color='silver')))

if show_air:
    y_air_base = [get_air_f(x, oil_base) * 2 for x in x_plot]
    y_air_comp = [get_air_f(x, oil_comp) * 2 for x in x_plot]
    fig.add_trace(go.Scatter(x=x_plot, y=y_air_base, name=f"エアバネ単体 ({oil_base}mm)", line=dict(color='lightblue', width=2)))
    fig.add_trace(go.Scatter(x=x_plot, y=y_air_comp, name=f"エアバネ単体 ({oil_comp}mm)", line=dict(color='pink', width=2)))

if show_total:
    y_total_base = [get_total_f_2pcs(x, oil_base) for x in x_plot]
    y_total_comp = [get_total_f_2pcs(x, oil_comp) for x in x_plot]
    fig.add_trace(go.Scatter(x=x_plot, y=y_total_base, name=f"合成反力 ({oil_base}mm)", line=dict(color='blue', width=4)))
    fig.add_trace(go.Scatter(x=x_plot, y=y_total_comp, name=f"合成反力 ({oil_comp}mm)", line=dict(color='red', width=4)))

# オイルロック域・ターゲット荷重表示
fig.add_vrect(x0=x_max - oil_lock_len, x1=x_max, fillcolor="gray", opacity=0.1, layer="below", annotation_text="オイルロック域")
fig.add_hline(y=f_target_total_kg, line_dash="dot", line_color="green", annotation_text="想定最大荷重")
fig.add_vline(x=target_stroke, line_dash="dash", line_color="orange", annotation_text=f"調査位置:{target_stroke}mm")

fig.update_layout(xaxis_title="ストローク量 [mm]", yaxis_title="荷重 (2本合計) [kg]", template="simple_white", height=600)
st.plotly_chart(fig, use_container_width=True, key="fork_sim_chart")
