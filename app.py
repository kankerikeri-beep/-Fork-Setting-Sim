import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math

st.set_page_config(page_title="フロントフォークエアバネシミュレーター v2.6", layout="wide")

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
# 3. 油面比較(フルストローク油面) / 任意荷重設定
# ===============================
st.header("③ 油面比較 ＆ 任意荷重調査")
col_o1, col_o2 = st.columns(2)
with col_o1:
    oil_base = st.slider("基準油面 [mm]", 10, 200, 60)
    oil_comp = st.slider("比較油面 [mm]", 10, 200, 70)
with col_o2:
    custom_force = st.number_input("調査したい任意荷重 (2本合計) [kg]", 0.0, 500.0, 150.0, step=1.0)

# ===============================
# 計算ロジック
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
        f_at_change = k_init * total_change
        return f_at_change + k_late * (x_total - total_change)

def get_air_f(x, air_space_at_full):
    p0 = 1.033 
    L0 = (air_space_at_full + x_max) * 0.95 
    L_current = L0 - x
    if L_current <= 0.1: return 3000.0
    p1 = p0 * ((L0 / L_current) ** n_index)
    force = (p1 - p0) * (area_air / 100)
    return force

def total_f_2pcs(x, oil):
    return (get_spring_f(x) + get_air_f(x, oil)) * 2

def find_stroke_by_force(target_f, oil):
    search = np.linspace(0, x_max, 1000)
    for sx in search:
        if total_f_2pcs(sx, oil) >= target_f:
            return sx
    return x_max

# ===============================
# 4. シミュレーション結果
# ===============================
st.header("④ シミュレーション結果比較")

c1, c2 = st.columns(2)
res_base = x_max - find_stroke_by_force(f_target_total_kg, oil_base)
res_comp = x_max - find_stroke_by_force(f_target_total_kg, oil_comp)

with c1:
    st.metric(f"基準油面 ({oil_base}mm) 最大荷重時残スト", f"{res_base:.1f} mm")
    stroke_custom_base = find_stroke_by_force(custom_force, oil_base)
    st.write(f"任意荷重({custom_force}kg)時のストローク: **{stroke_custom_base:.1f} mm**")

with c2:
    st.metric(f"比較油面 ({oil_comp}mm) 最大荷重時残スト", f"{res_comp:.1f} mm", delta=f"{res_comp - res_base:.1f} mm")
    stroke_custom_comp = find_stroke_by_force(custom_force, oil_comp)
    st.write(f"任意荷重({custom_force}kg)時のストローク: **{stroke_custom_comp:.1f} mm**")

# グラフ描画
x_plot = np.linspace(0, x_max, 500)
fig = go.Figure()

# 金属バネのみ（2本分）
f_s_total = [get_spring_f(x) * 2 for x in x_plot]
fig.add_trace(go.Scatter(x=x_plot, y=f_s_total, name="金属ばね反力(合成)", line=dict(dash='dash', color='silver')))

def add_comp_trace(oil, name, base_color):
    x_low = x_plot[x_plot <= s_change]
    x_high = x_plot[x_plot > s_change]
    if len(x_high) > 0: x_high = np.insert(x_high, 0, x_low[-1])
    
    y_low = [total_f_2pcs(x, oil) for x in x_low]
    y_high = [total_f_2pcs(x, oil) for x in x_high]
    
    # ID重複回避のためnameにユニークな値を付与
    fig.add_trace(go.Scatter(x=x_low, y=y_low, name=f"{name}(初期)", line=dict(color=base_color, width=4)))
    if len(x_high) > 0:
        fig.add_trace(go.Scatter(x=x_high, y=y_high, name=f"{name}(後半)", line=dict(color="orange", width=4)))

add_comp_trace(oil_base, "基準油面", "blue")
add_comp_trace(oil_comp, "比較油面", "red")

# オイルロックエリア表示
fig.add_vrect(x0=x_max - oil_lock_len, x1=x_max, fillcolor="gray", opacity=0.2, layer="below", annotation_text="オイルロック域")

# ターゲット荷重と任意荷重
fig.add_hline(y=f_target_total_kg, line_dash="dot", line_color="green", annotation_text="想定最大荷重")
fig.add_hline(y=custom_force, line_dash="dash", line_color="purple", annotation_text="調査任意荷重")

fig.update_layout(xaxis_title="ストローク量 [mm]", yaxis_title="荷重 (2本合計) [kg]", template="simple_white", height=600)
st.plotly_chart(fig, use_container_width=True, key="main_chart") # keyを追加してエラー回避
