import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math

st.set_page_config(page_title="フロントフォークエアバネシミュレーター v2.4", layout="wide")

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
    st.caption("※フル制動時のキャスター変化を含めて荷重移動を計算します。")
    decel_g = st.number_input("最大減速G (1.0〜1.5G推奨)", 0.5, 2.0, 1.25, step=0.01)
with col_v3:
    rad = math.radians(caster_angle)
    # 荷重補正係数 1.18 (重心移動補正)
    f_target_total_kg = (total_m * math.cos(rad) + total_m * decel_g * math.sin(rad)) * 1.18
    st.metric("フォーク全体への想定荷重", f"{f_target_total_kg:.1f} kg")
    st.write("**（車重＋G換算＋重心移動補正：1.18）**")

# ===============================
# 2. フォーク・バネ内部仕様
# ===============================
st.header("② フォーク・バネ内部仕様")
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    fork_id = st.number_input("フォーク内径 [mm]", 10.0, 60.0, 36.0, step=0.1)
    x_max = st.number_input("最大ストローク [mm]", 1.0, 250.0, 97.0, step=1.0)
with col_f2:
    k_init = st.number_input("初期レート (1本分) [kg/mm]", 0.0, 20.0, 0.37, step=0.01)
    k_late = st.number_input("後半レート (1本分) [kg/mm]", 0.0, 20.0, 0.98, step=0.01)
    s_change = st.number_input("レート変化点 [mm]", 0.0, 250.0, 82.0, step=1.0)
with col_f3:
    preload = st.number_input("プリロード [mm]", 0.0, 50.0, 27.0, step=1.0)
    n_index = st.slider("空気断熱指数 n", 1.0, 3.0, 2.40, step=0.01)

st.markdown("""
**【空気断熱指数 $n$ の目安】**
* **0% (理論値) 1.60**：非常に緩やか。奥での踏ん張りが足りない。
* **25% 占拠 1.80 〜 1.95**：正立フォークなどでバネが細い場合。
* **50% 占拠 2.10 〜 2.50**：倒立フォークや、太いカラー・インナーが入っている状態。
* **60% 占拠 2.50 〜 2.70**：ほぼオイルロックに近い、極めて急激な立ち上がり。

※本数値は、オイルによる動的減衰抵抗を空気の反力として擬似的に合算したセッティング指標です。
""")

# ===============================
# 3. 油面シミュレーション
# ===============================
st.header("③ 油面比較（ストローク奥の空気室長）")
st.info("油面数値(mm)はフルストローク時の空気室の長さ。数値が大きいほど空気が多く、柔らかくなります。")
oil_base = st.slider("基準油面 [mm]", 10, 200, 60)
oil_comp = st.slider("比較油面 [mm]", 10, 200, 70)

# ===============================
# 計算ロジック
# ===============================
area_air = math.pi * (fork_id / 2)**2

def get_spring_f(x):
    x_total = x + preload
    total_change = s_change + preload
    if k_late == 0 or s_change <= 0:
        return k_init * x_total
    else:
        f_at_change = k_init * total_change
        return f_at_change + k_late * (x_total - total_change)

def get_air_f(x, air_space_at_full):
    p0 = 1.033 
    L0 = (air_space_at_full + x_max) * 0.95 
    L_current = L0 - x
    if L_current <= 0.1: return 3000.0 # オイルロック
    p1 = p0 * ((L0 / L_current) ** n_index)
    force = (p1 - p0) * (area_air / 100)
    return force

def total_f_2pcs(x, oil):
    return (get_spring_f(x) + get_air_f(x, oil)) * 2

def find_res(oil):
    search = np.linspace(0, x_max, 1000)
    for sx in search:
        if total_f_2pcs(sx, oil) >= f_target_total_kg:
            return max(0.0, x_max - sx)
    return 0.0

res_base = find_res(oil_base)
res_comp = find_res(oil_comp)

# ===============================
# 4. グラフ表示
# ===============================
st.header("④ シミュレーション結果比較")

c1, c2 = st.columns(2)
with c1:
    st.metric(f"基準油面 ({oil_base}mm) 残スト予測", f"{res_base:.1f} mm")
with c2:
    st.metric(f"比較油面 ({oil_comp}mm) 残スト予測", f"{res_comp:.1f} mm", 
              delta=f"{res_comp - res_base:.1f} mm", delta_color="normal")

x_plot = np.linspace(0, x_max, 500)
f_s_total = [get_spring_f(x) * 2 for x in x_plot]
f_t_base_total = [total_f_2pcs(x, oil_base) for x in x_plot]
f_t_comp_total = [total_f_2pcs(x, oil_comp) for x in x_plot]

fig = go.Figure()
fig.add_trace(go.Scatter(x=x_plot, y=f_s_total, name="金属ばね反力 (2本合計)", line=dict(dash='dash', color='silver')))
fig.add_trace(go.Scatter(x=x_plot, y=f_t_base_total, name=f"基準 {oil_base}mm 合成", line=dict(color='blue', width=4)))
fig.add_trace(go.Scatter(x=x_plot, y=f_t_comp_total, name=f"比較 {oil_comp}mm 合成", line=dict(color='red', width=4)))

fig.add_hline(y=f_target_total_kg, line_dash="dot", line_color="green", 
              annotation_text=f"荷重ターゲット: {f_target_total_kg:.1f}kg")

fig.update_layout(xaxis_title="ストローク量 [mm]", yaxis_title="荷重 (2本合計) [kg]", template="simple_white", height=600)
st.plotly_chart(fig, use_container_width=True)
