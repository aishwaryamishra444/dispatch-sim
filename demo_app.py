"""
demo_app.py — Wednesday presentation app for dispatch_sim.

Runs the CERTIFIED engine (dispatch_sim.runners) live — the numbers on screen
are identical to the xlsx outputs, because it's the same code path.

Run:   streamlit run demo_app.py     (from the repo root)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dispatch_sim.core.battery import Battery
from dispatch_sim.io.loaders import load_dsm_config, load_yaml
from dispatch_sim.runners.rules import run_s1, run_s2, run_s3

CFG = Path(__file__).parent / "dispatch_sim" / "config"
BLOCKS, DT = 96, 0.25
INR = lambda v: f"₹{v:,.0f}"

st.set_page_config(page_title="Agentic Grid Simulator",
                   page_icon="⚡", layout="wide")


# ---------------------------------------------------------------- day maker
def make_day(seed: int, err_pct: float, plant_mw: float):
    rng = np.random.default_rng(seed)
    h = np.arange(BLOCKS) * DT
    day = (h > 6.25) & (h < 18.25)
    fc = np.zeros(BLOCKS)
    fc[day] = plant_mw * np.sin(np.pi * (h[day] - 6.25) / 12.0) ** 1.35
    p1, p2 = rng.uniform(0, 2 * np.pi, 2)
    err = (0.6 * np.sin(rng.uniform(2, 5) * h + p1)
           + 0.4 * np.sin(rng.uniform(5, 10) * h + p2)) * (err_pct / 100) * 1.6
    act = fc * (1 + err)
    for _ in range(rng.integers(1, 4)):
        c, w, d = rng.uniform(9, 16), rng.uniform(0.4, 1.6), rng.uniform(0.25, 0.7)
        act *= 1 - d * np.exp(-((h - c) ** 2) / (2 * w * w))
    return fc.tolist(), np.clip(act, 0, plant_mw).tolist(), h


def fresh_battery(spec_overrides=None):
    spec = {"batteryUsableCapacity": 40.0, "cRateMW": 20.0,
            "roundTripEfficiency": 0.88, "socMinPct": 10, "socMaxPct": 90}
    if spec_overrides:
        spec.update(spec_overrides)
    return Battery(usable_capacity_mwh=spec["batteryUsableCapacity"],
                   c_rate_mw=spec["cRateMW"], rte=spec["roundTripEfficiency"],
                   soc_min_pct=spec["socMinPct"], soc_max_pct=spec["socMaxPct"])


# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚡ Agentic Grid Simulator")
st.sidebar.caption("Baseline Scenarios 1–3 · CERC DSM 2024")

err = st.sidebar.slider("Forecast error σ (%)", 3, 30, 12,
                        help="Gap between day-ahead forecast and actual generation")
ppa = st.sidebar.slider("PPA tariff (₹/kWh)", 2.0, 4.5, 2.60, 0.05)
deg = st.sidebar.slider("Battery degradation (₹/kWh cycled)", 0.5, 3.0, 2.5, 0.1,
                        help="The single most sensitive input — flips S2/S3 economics")
cap = st.sidebar.slider("BESS capacity (MWh)", 10, 80, 40, 5)

if "seed" not in st.session_state:
    st.session_state.seed = 20260614
if st.sidebar.button("🎲 New weather day"):
    st.session_state.seed = int(np.random.default_rng().integers(1e9))

st.sidebar.divider()
st.sidebar.caption("Engine: dispatch_sim (Reg 6(2)/8(4), 20 unit tests). "
                   "Same code path as the xlsx outputs — this UI adds nothing "
                   "to the math.")

# ---------------------------------------------------------------- run engine
plant = load_yaml(CFG / "plant.yaml")
plant["ppa_rate_inr_per_kwh"] = ppa
dsm_cfg = load_dsm_config(CFG / "dsm_bands.yaml")
s2_cfg = load_yaml(CFG / "scenario_s2.yaml"); s2_cfg["degradation_inr_per_kwh"] = deg
s3_cfg = load_yaml(CFG / "scenario_s3.yaml"); s3_cfg["degradation_inr_per_kwh"] = deg
s1_cfg = load_yaml(CFG / "scenario_s1.yaml")

forecast, actual, hours = make_day(st.session_state.seed, err, plant["plant_mw"])

r1 = run_s1(forecast, actual, plant, dsm_cfg, s1_cfg)
r2 = run_s2(forecast, actual, plant, dsm_cfg, s2_cfg,
            fresh_battery({"batteryUsableCapacity": float(cap)}))
r3 = run_s3(forecast, actual, plant, dsm_cfg, s3_cfg,
            fresh_battery({"batteryUsableCapacity": float(cap)}))
results = {"S1 · PPA only": r1, "S2 · Battery buffer": r2, "S3 · Time windows": r3}

# ---------------------------------------------------------------- header
st.title("Agentic Grid Simulator")
st.markdown("**Solar-BESS Dispatch — Baseline Scenarios**")
st.caption("10 MW plant · flat-rate PPA · CERC DSM 2024 settlement per 15-min "
           "block · synthetic weather day (historical replay: Phase 1)")

cols = st.columns(3)
p1, p2_, p3 = (r.total("profit") for r in (r1, r2, r3))
cols[0].metric("S1 · PPA only", INR(p1), f"DSM −{INR(r1.total('dsm_penalty'))}",
               delta_color="inverse")
cols[1].metric("S2 · Battery buffer", INR(p2_), f"{INR(p2_-p1)} vs S1",
               delta_color="normal" if p2_ >= p1 else "inverse")
cols[2].metric("S3 · Time windows", INR(p3), f"{INR(p3-p1)} vs S1",
               delta_color="normal" if p3 >= p1 else "inverse")

best = max(results, key=lambda k: results[k].total("profit"))
worst_msg = (" — the battery cannot pay for itself on DSM avoidance alone "
             "under a flat PPA. That gap is the case for the Scenario 5 optimizer."
             if best == "S1 · PPA only" else "")
st.info(f"**Best today: {best}** at "
        f"{INR(results[best].total('profit'))}/day{worst_msg}")

# ---------------------------------------------------------------- day chart
tabs = st.tabs(list(results))
for tab, (name, r) in zip(tabs, results.items()):
    with tab:
        fig = go.Figure()
        fig.add_scatter(x=hours, y=[m * 4 for m in
                        [row.actual_gen_mwh for row in r.rows]],
                        name="Actual gen (MW)", line=dict(color="#F59E0B", width=2),
                        fill="tozeroy", fillcolor="rgba(245,158,11,.10)")
        fig.add_scatter(x=hours, y=[row.scheduled_mwh * 4 for row in r.rows],
                        name="Schedule (MW)",
                        line=dict(color="#16202E", width=1.6, dash="dash"))
        fig.add_scatter(x=hours, y=[row.delivered_mwh * 4 for row in r.rows],
                        name="Delivered (MW)", line=dict(color="#1D4ED8", width=1.6))
        soc = [row.soc_mwh for row in r.rows]
        if any(s is not None for s in soc):
            fig.add_scatter(x=hours, y=[s / 4 if s else 0 for s in soc],
                            name="SoC (MWh÷4)",
                            line=dict(color="#059669", width=2),
                            fill="tozeroy", fillcolor="rgba(16,185,129,.10)")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=1.1),
                          xaxis_title="Hour of day", yaxis_title="MW")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- P&L
st.subheader("Daily P&L decomposition")
names = list(results)
fig = go.Figure()
fig.add_bar(name="PPA revenue", x=names,
            y=[r.total("ppa_revenue") for r in results.values()],
            marker_color="#16202E")
fig.add_bar(name="DSM receivable", x=names,
            y=[r.total("dsm_receivable") for r in results.values()],
            marker_color="#1D4ED8")
fig.add_bar(name="DSM payable", x=names,
            y=[-r.total("dsm_payable") for r in results.values()],
            marker_color="#DC2626")
fig.add_bar(name="Degradation", x=names,
            y=[-r.total("degradation") for r in results.values()],
            marker_color="#F59E0B")
fig.add_bar(name="O&M", x=names,
            y=[-r.total("om") for r in results.values()], marker_color="#8A97A8")
fig.update_layout(barmode="relative", height=320,
                  margin=dict(l=10, r=10, t=10, b=10),
                  legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig, use_container_width=True)

df = pd.DataFrame({
    "Scenario": names,
    "PPA revenue": [r.total("ppa_revenue") for r in results.values()],
    "DSM net": [r.total("dsm_receivable") - r.total("dsm_payable")
                for r in results.values()],
    "Degradation": [r.total("degradation") for r in results.values()],
    "O&M": [r.total("om") for r in results.values()],
    "Profit/day": [r.total("profit") for r in results.values()],
    "Profit/year (330d)": [r.total("profit") * 330 for r in results.values()],
})
st.dataframe(df.style.format({c: "₹{:,.0f}" for c in df.columns[1:]}),
             use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- researcher
with st.expander("🔬 Block-level settlement (researcher view)"):
    pick = st.selectbox("Scenario", names)
    r = results[pick]
    bl = pd.DataFrame([{
        "time": row.time, "sched_mwh": row.scheduled_mwh,
        "actual_mwh": row.actual_gen_mwh, "delivered_mwh": row.delivered_mwh,
        "dev_%AvC": row.deviation_pct_of_avc,
        "dsm_recv": row.dsm_receivable, "dsm_pay": row.dsm_payable,
        "penalty": row.dsm_penalty, "profit": row.profit,
    } for row in r.rows])
    st.dataframe(bl, use_container_width=True, height=300, hide_index=True)
    st.download_button("⬇ Export CSV", bl.to_csv(index=False),
                       file_name=f"{pick[:2]}_blocks.csv")

st.caption("dispatch_sim · github.com/aishwaryamishra444/dispatch-sim · "
           "values marked NEEDS-CONFIRMATION in config pending team sign-off")
