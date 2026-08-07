"""
demo_app.py — Atria University · Agentic Grid Simulator (Wednesday demo app)

Runs the CERTIFIED engine (dispatch_sim.runners) live -- the numbers on
screen are identical to the xlsx outputs, because it's the same code path.

Run:   streamlit run demo_app.py     (from the repo root)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from dispatch_sim.core.battery import Battery
from dispatch_sim.io.loaders import load_dsm_config, load_yaml, load_series_csv_buffer
from dispatch_sim.io.iex_loader import IEXFormatError, parse_iex_file
from dispatch_sim.runners.rules import run_s1, run_s2, run_s3
from dispatch_sim.optimizer.lp_dispatch import OptimizerBatterySpec, solve_optimal_dispatch

CFG = Path(__file__).parent / "dispatch_sim" / "config"
BLOCKS, DT = 96, 0.25
INR = lambda v: f"Rs {v:,.0f}"

# Atria University brand palette (from the wordmark: indigo triangle, green U)
BLUE = "#3D34E0"
GREEN = "#3FAE49"
INK = "#231F20"
GREY = "#5B6B7F"
LINE = "#E5EAF1"

st.set_page_config(page_title="Agentic Grid Simulator | Atria University",
                   page_icon=None, layout="wide")

# ---------------------------------------------------------------- global style
st.markdown(f"""
<style>
:root {{ --au-blue: {BLUE}; --au-green: {GREEN}; --au-ink: {INK}; }}

/* underlined tab navigation */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 2px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{
    height: 42px; padding: 0 18px; font-weight: 600; font-size: 0.92rem;
    color: {GREY}; border-bottom: 3px solid transparent; margin-bottom: -2px;
}}
.stTabs [aria-selected="true"] {{
    color: {BLUE} !important; border-bottom: 3px solid {BLUE} !important;
}}

/* live status badge */
.au-badge {{display:inline-flex;align-items:center;gap:7px;
    background:#EEF0FF;border:1px solid #D2D6FB;color:{BLUE};
    padding:4px 12px;border-radius:999px;font-size:0.76rem;font-weight:700;
    letter-spacing:.05em;text-transform:uppercase;margin-bottom:10px;}}
.au-dot {{width:7px;height:7px;border-radius:50%;background:{BLUE};
    animation:aupulse 1.6s ease-in-out infinite;}}
@keyframes aupulse{{0%,100%{{opacity:1;transform:scale(1);}}
                    50%{{opacity:.25;transform:scale(.7);}}}}

/* institution header strip */
.au-mark-row {{display:flex;align-items:center;gap:10px;margin-bottom:2px;}}
.au-mark-text {{font-size:0.82rem;font-weight:700;letter-spacing:.14em;
    color:{INK};text-transform:uppercase;}}
.au-mark-sub {{font-size:0.82rem;color:{GREY};letter-spacing:.06em;}}

/* section labels */
.au-section {{font-size:0.95rem;font-weight:700;color:{INK};
    letter-spacing:.02em;margin:4px 0 10px;
    border-left:4px solid {BLUE};padding-left:10px;}}

/* grid intelligence header */
.gi-header {{display:flex;align-items:center;gap:10px;margin:6px 0 10px;}}
.gi-title  {{font-size:0.9rem;font-weight:700;letter-spacing:.16em;
            text-transform:uppercase;color:{BLUE};}}
.gi-dot    {{width:9px;height:9px;border-radius:50%;background:{BLUE};
            animation:aupulse 1.6s ease-in-out infinite;}}
</style>
""", unsafe_allow_html=True)


def au_mark(size: int = 34) -> str:
    """Small vector mark echoing the Atria University wordmark: an indigo
    triangle (A) sitting over a green U. Built as inline SVG so the app has
    no external image dependency."""
    svg = f"""
    <svg width="{size}" height="{size}" viewBox="0 0 100 100"
         xmlns="http://www.w3.org/2000/svg">
      <path d="M50 12 L88 62 L66 62 L50 40 L34 62 L12 62 Z" fill="{BLUE}"/>
      <path d="M22 55 L40 55 L40 74 Q40 84 50 84 Q60 84 60 74 L60 55 L78 55
               L78 74 Q78 96 50 96 Q22 96 22 74 Z" fill="{GREEN}"/>
    </svg>"""
    return "\n".join(line.lstrip() for line in svg.split("\n"))


def energy_flow_diagram() -> str:
    """Product energy/data-flow diagram: physical power flow on top
    (PV -> PCC -> Grid, with BESS buffering), commercial/data flow below
    (Forecast -> Schedule -> DSM Settlement -> Profit). Pure SVG, brand
    colors only, no external assets."""
    def node(x, y, w, h, title, sub, fill, text_color=INK, border=LINE):
        return f'''
        <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12"
              fill="{fill}" stroke="{border}" stroke-width="1.4"/>
        <text x="{x + w/2}" y="{y + h/2 - 4}" text-anchor="middle"
              font-size="13.5" font-weight="700" fill="{text_color}"
              font-family="sans-serif">{title}</text>
        <text x="{x + w/2}" y="{y + h/2 + 15}" text-anchor="middle"
              font-size="10.5" fill="{GREY}" font-family="sans-serif">{sub}</text>'''

    def arrow(x1, y1, x2, y2, dash=False, color=GREY):
        d = f'stroke-dasharray="6 5"' if dash else ""
        return f'''
        <defs><marker id="ah{x1}{y1}{x2}{y2}" markerWidth="8" markerHeight="8"
            refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="{color}"/>
        </marker></defs>
        <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}"
              stroke-width="1.8" {d} marker-end="url(#ah{x1}{y1}{x2}{y2})"/>'''

    solar_icon = '''
    <g transform="translate(60,50)">
      <line x1="0" y1="-24" x2="0" y2="-32" stroke="#F59E0B" stroke-width="2.5"/>
      <line x1="17" y1="-17" x2="23" y2="-23" stroke="#F59E0B" stroke-width="2.5"/>
      <line x1="-17" y1="-17" x2="-23" y2="-23" stroke="#F59E0B" stroke-width="2.5"/>
      <circle cx="0" cy="0" r="13" fill="#F59E0B"/>
      <rect x="-20" y="18" width="40" height="20" rx="2" fill="none"
            stroke="{ink}" stroke-width="2"/>
      <line x1="-20" y1="28" x2="20" y2="28" stroke="{ink}" stroke-width="1.4"/>
      <line x1="-7" y1="18" x2="-7" y2="38" stroke="{ink}" stroke-width="1.4"/>
      <line x1="7" y1="18" x2="7" y2="38" stroke="{ink}" stroke-width="1.4"/>
    </g>'''.replace("{ink}", INK)

    battery_icon = f'''
    <g transform="translate(60,44)">
      <rect x="-22" y="-14" width="44" height="28" rx="4" fill="none"
            stroke="{GREEN}" stroke-width="2.5"/>
      <rect x="22" y="-6" width="5" height="12" rx="1.5" fill="{GREEN}"/>
      <rect x="-15" y="-7" width="10" height="14" fill="{GREEN}"/>
      <rect x="0" y="-9" width="10" height="18" fill="{GREEN}" opacity="0.55"/>
    </g>'''

    tower_icon = f'''
    <g transform="translate(60,52)" stroke="{BLUE}" stroke-width="2.2" fill="none">
      <line x1="0" y1="-30" x2="-16" y2="30"/>
      <line x1="0" y1="-30" x2="16" y2="30"/>
      <line x1="-11" y1="-6" x2="11" y2="-6"/>
      <line x1="-8" y1="10" x2="8" y2="10"/>
      <line x1="-20" y1="-22" x2="20" y2="-22"/>
      <circle cx="-20" cy="-22" r="2.4" fill="{BLUE}"/>
      <circle cx="20" cy="-22" r="2.4" fill="{BLUE}"/>
    </g>'''

    svg = f'''
    <svg viewBox="0 0 1180 430" width="100%" xmlns="http://www.w3.org/2000/svg"
         font-family="sans-serif">
      <text x="10" y="24" font-size="11" font-weight="700" letter-spacing="1.5"
            fill="{GREY}">PHYSICAL POWER FLOW</text>
      {node(20, 40, 140, 110, "Solar PV", "10 MW plant", "#FFF7E6")}
      {solar_icon}
      {node(230, 40, 150, 110, "PCC / Meter", "point of common coupling", "#F4F6F8")}
      {node(500, 40, 160, 110, "Grid / DISCOM", "scheduled injection", "#EEF0FF")}
      {node(230, 190, 150, 110, "BESS", "20 MW / 40 MWh", "#EAF7EC")}

      {arrow(160, 95, 230, 95)}
      {arrow(380, 95, 500, 95)}
      {arrow(305, 150, 305, 190, dash=True)}
      {arrow(305, 190, 305, 150, dash=True)}

      <g transform="translate(305,244)">{battery_icon}</g>
      <g transform="translate(580,95) scale(0.9)">{tower_icon}</g>

      <text x="10" y="330" font-size="11" font-weight="700" letter-spacing="1.5"
            fill="{GREY}">COMMERCIAL / SETTLEMENT FLOW</text>
      {node(20, 346, 160, 74, "Forecast", "day-ahead generation", "#F4F6F8")}
      {node(240, 346, 170, 74, "Schedule vs Actual", "15-min deviation", "#F4F6F8")}
      {node(470, 346, 220, 74, "DSM Settlement Engine", "CERC Reg 6(2) / 8(4)", BLUE, "#FFFFFF")}
      {node(750, 346, 180, 74, "Profit / Loss", "Rs per 15-min block", "#EAF7EC")}

      {arrow(180, 383, 240, 383)}
      {arrow(410, 383, 470, 383)}
      {arrow(690, 383, 750, 383)}
      {arrow(305, 300, 320, 346, dash=True, color=BLUE)}
    </svg>'''
    return "\n".join(line.lstrip() for line in svg.split("\n"))


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
st.sidebar.markdown(
    f'<div class="au-mark-row">{au_mark(30)}'
    f'<div><div class="au-mark-text">Atria University</div>'
    f'<div class="au-mark-sub">Centre of Excellence</div></div></div>',
    unsafe_allow_html=True)
st.sidebar.title("Agentic Grid Simulator")
st.sidebar.caption("Baseline Scenarios 1-3 - CERC DSM 2024")

st.sidebar.divider()
st.sidebar.subheader("Data source")
data_mode = st.sidebar.radio(
    "Generation data", ["Synthetic (demo)", "Upload real day"],
    label_visibility="collapsed",
    help="Real data replaces the synthetic weather day used for the demo - "
         "the settlement math is unchanged either way.")

real_forecast, real_actual = None, None
if data_mode == "Upload real day":
    st.sidebar.caption("CSV format: columns `time,mw` - 96 rows (15-min blocks). "
                       "Same format the CLI (`--forecast`/`--actual`) expects.")
    fc_file = st.sidebar.file_uploader("Forecast CSV (day-ahead schedule basis)",
                                       type=["csv"], key="fc_upload")
    act_file = st.sidebar.file_uploader("Actual generation CSV (telemetry)",
                                        type=["csv"], key="act_upload")
    if fc_file is not None:
        try:
            real_forecast = load_series_csv_buffer(fc_file, fc_file.name)
            st.sidebar.success(f"Forecast: {len(real_forecast)} blocks loaded.")
        except Exception as e:
            st.sidebar.error(str(e))
    if act_file is not None:
        try:
            real_actual = load_series_csv_buffer(act_file, act_file.name)
            st.sidebar.success(f"Actual: {len(real_actual)} blocks loaded.")
        except Exception as e:
            st.sidebar.error(str(e))
    if fc_file is None or act_file is None:
        st.sidebar.caption("Upload both files to switch off synthetic data - "
                           "showing the synthetic day below until then.")

with st.sidebar.expander("Real IEX price upload (for future P2P scenarios)"):
    st.caption("Baseline Scenarios S1-S3 are PPA-only per the team plan and do "
              "not consume market price - this loader is here for Scenario 5 / "
              "P2P work, not wired into today's profit numbers.")
    iex_file = st.file_uploader(
        "IEX Area Price / Market Snapshot", type=["csv", "xls", "xlsx"],
        key="iex_upload",
        help="Download from iexindia.com -> Market Data -> Day Ahead Market "
             "-> Market Snapshot / Area Price.")
    if iex_file is not None:
        try:
            iex_result = parse_iex_file(iex_file)
            st.success(f"Parsed {iex_result.detected_rows} rows from "
                      f"'{iex_file.name}' (column: {iex_result.source_columns['price']}). "
                      f"Not yet applied to S1-S3 profit - reserved for Scenario 5.")
        except IEXFormatError as e:
            st.warning(f"Couldn't auto-detect columns: {e}")
            if e.columns:
                tcol = st.selectbox("Time column", e.columns, key="iex_tcol")
                pcol = st.selectbox("Price column", e.columns, key="iex_pcol")
                if st.button("Parse with these columns", key="iex_parse_btn"):
                    try:
                        iex_file.seek(0)
                        r2 = parse_iex_file(iex_file, time_col=tcol, price_col=pcol)
                        st.success(f"Parsed using '{tcol}' / '{pcol}'.")
                    except Exception as e2:
                        st.error(f"Still couldn't parse: {e2}")

st.sidebar.divider()
err = st.sidebar.slider("Forecast error (%)", 3, 30, 12,
                        help="Gap between day-ahead forecast and actual generation")
ppa = st.sidebar.slider("PPA tariff (Rs/kWh)", 2.0, 4.5, 2.60, 0.05)
deg = st.sidebar.slider("Battery degradation (Rs/kWh cycled)", 0.5, 3.0, 2.5, 0.1,
                        help="The single most sensitive input - flips S2/S3 economics")
cap = st.sidebar.slider("BESS capacity (MWh)", 10, 80, 40, 5)

if "seed" not in st.session_state:
    st.session_state.seed = 20260614
if st.sidebar.button("New weather day"):
    st.session_state.seed = int(np.random.default_rng().integers(1e9))

st.sidebar.divider()
st.sidebar.caption("Engine: dispatch_sim (Reg 6(2)/8(4), 20 unit tests). "
                   "Same code path as the xlsx outputs - this UI adds nothing "
                   "to the math.")

# ---------------------------------------------------------------- run engine
plant = load_yaml(CFG / "plant.yaml")
plant["ppa_rate_inr_per_kwh"] = ppa
dsm_cfg = load_dsm_config(CFG / "dsm_bands.yaml")
s2_cfg = load_yaml(CFG / "scenario_s2.yaml"); s2_cfg["degradation_inr_per_kwh"] = deg
s3_cfg = load_yaml(CFG / "scenario_s3.yaml"); s3_cfg["degradation_inr_per_kwh"] = deg
s1_cfg = load_yaml(CFG / "scenario_s1.yaml")

using_real = real_forecast is not None and real_actual is not None
if using_real:
    forecast, actual = real_forecast, real_actual
    hours = np.arange(BLOCKS) * DT
else:
    forecast, actual, hours = make_day(st.session_state.seed, err, plant["plant_mw"])

r1 = run_s1(forecast, actual, plant, dsm_cfg, s1_cfg)
r2 = run_s2(forecast, actual, plant, dsm_cfg, s2_cfg,
            fresh_battery({"batteryUsableCapacity": float(cap)}))
r3 = run_s3(forecast, actual, plant, dsm_cfg, s3_cfg,
            fresh_battery({"batteryUsableCapacity": float(cap)}))
results = {"S1 - PPA only": r1, "S2 - Battery buffer": r2, "S3 - Time windows": r3}

optimizer_error = None
try:
    r5 = solve_optimal_dispatch(
        forecast, actual, plant, dsm_cfg,
        OptimizerBatterySpec(float(cap), 20.0, 0.88, 10, 90), deg)
    results["S5 - Optimizer"] = r5
except Exception as e:  # noqa: BLE001 -- never let a solver hiccup crash the demo
    optimizer_error = str(e)
    r5 = None

# ---------------------------------------------------------------- header
st.markdown(
    f'<div class="au-mark-row">{au_mark(26)}'
    f'<span class="au-mark-sub">Atria University -- Centre of Excellence</span></div>',
    unsafe_allow_html=True)
st.markdown(
    '<div class="au-badge"><span class="au-dot"></span>Live simulation</div>',
    unsafe_allow_html=True)

hcol1, hcol2 = st.columns([4, 2])
with hcol1:
    st.title("Agentic Grid Simulator")
    st.markdown("**Solar-BESS Dispatch -- Baseline Scenarios**")
with hcol2:
    st.write("")
    bcol1, bcol2 = st.columns(2)
    comparison_export = pd.DataFrame({
        "Scenario": list(results),
        "Profit/day (INR)": [r.total("profit") for r in results.values()],
        "DSM penalty (INR)": [r.total("dsm_penalty") for r in results.values()],
        "Degradation (INR)": [r.total("degradation") for r in results.values()],
    })
    with bcol1:
        st.download_button("Export CSV", comparison_export.to_csv(index=False),
                           file_name="agentic_grid_results.csv")
    with bcol2:
        if st.button("Refresh"):
            st.rerun()

prov = "real uploaded generation data" if using_real else "synthetic weather day (upload a real one from the left panel)"
st.caption(f"10 MW plant - flat-rate PPA - CERC DSM 2024 settlement per 15-min "
          f"block - {prov}")


def _cuboid(cx, cy, cz, color, dx=0.26, dy=0.26, dz=0.22, name="", hover=""):
    """One solid 3D box (12-triangle mesh) representing a system node."""
    xs = [cx - dx, cx - dx, cx + dx, cx + dx, cx - dx, cx - dx, cx + dx, cx + dx]
    ys = [cy - dy, cy + dy, cy + dy, cy - dy, cy - dy, cy + dy, cy + dy, cy - dy]
    zs = [cz - dz, cz - dz, cz - dz, cz - dz, cz + dz, cz + dz, cz + dz, cz + dz]
    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    return go.Mesh3d(
        x=xs, y=ys, z=zs, i=i, j=j, k=k, color=color, opacity=1.0,
        flatshading=True, name=name, hovertext=hover or name, hoverinfo="text",
        lighting=dict(ambient=0.55, diffuse=0.85, specular=0.35, roughness=0.4,
                      fresnel=0.15),
        lightposition=dict(x=150, y=200, z=250), showlegend=False)


def _floor(x0, x1, y0, y1, z, color="#F0F3F8", opacity=0.65):
    """A flat translucent plane that grounds the diagram visually."""
    xs, ys, zs = [x0, x1, x1, x0], [y0, y0, y1, y1], [z, z, z, z]
    return go.Mesh3d(x=xs, y=ys, z=zs, i=[0, 0], j=[1, 2], k=[2, 3],
                     color=color, opacity=opacity, hoverinfo="skip",
                     showlegend=False, flatshading=True)


def build_3d_diagram():
    """Interactive 3D system overview: solid geometry, not flat points.
    Upper plane -- physical power flow (Solar -> PCC -> Grid, with BESS
    buffering). Lower plane -- commercial settlement flow (Forecast ->
    Schedule vs Actual -> DSM Settlement Engine -> Profit/Loss). Built with
    Plotly (already proven reliable elsewhere in this app), rendered
    natively -- no custom HTML/SVG embedding involved."""
    physical = [
        ("Solar PV", "10 MW plant", 0, 2, 1.4, "#F59E0B"),
        ("PCC / Meter", "point of common coupling", 1, 2, 1.4, INK),
        ("Grid / DISCOM", "scheduled injection", 2.1, 2.6, 1.4, BLUE),
        ("BESS", "20 MW / 40 MWh", 1, 1.1, 1.4, GREEN),
    ]
    commercial = [
        ("Forecast", "day-ahead generation", 0, 0, 0, INK),
        ("Schedule vs Actual", "15-min deviation", 1, 0, 0, INK),
        ("DSM Settlement Engine", "CERC Reg 6(2) / 8(4)", 2.1, 0, 0, BLUE),
        ("Profit / Loss", "Rs per 15-min block", 3.1, 0, 0, GREEN),
    ]
    physical_edges = [(0, 1), (1, 2), (1, 3)]
    commercial_edges = [(0, 1), (1, 2), (2, 3)]

    fig = go.Figure()
    fig.add_trace(_floor(-0.6, 3.7, -0.6, 3.2, z=-0.35))

    for edges, nodes, ecolor in [(physical_edges, physical, GREY),
                                 (commercial_edges, commercial, GREY)]:
        xs, ys, zs = [], [], []
        for a, b in edges:
            xs += [nodes[a][2], nodes[b][2], None]
            ys += [nodes[a][3], nodes[b][3], None]
            zs += [nodes[a][4], nodes[b][4], None]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                   line=dict(color=ecolor, width=7),
                                   hoverinfo="skip", showlegend=False))

    fig.add_trace(go.Scatter3d(
        x=[1, 1], y=[2, 0], z=[1.15, 0.25], mode="lines",
        line=dict(color=BLUE, width=5, dash="dash"),
        hoverinfo="skip", showlegend=False))

    for title, sub, x, y, z, color in physical + commercial:
        fig.add_trace(_cuboid(x, y, z, color, name=title,
                              hover=f"<b>{title}</b><br>{sub}"))
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z + 0.42], mode="text",
            text=[f"<b>{title}</b>"], textposition="top center",
            textfont=dict(size=12, color=INK), hoverinfo="skip",
            showlegend=False))

    fig.add_trace(go.Scatter3d(
        x=[-0.4], y=[2.9], z=[1.9], mode="text",
        text=["PHYSICAL POWER FLOW"], textposition="middle right",
        textfont=dict(size=11, color=GREY), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(
        x=[-0.4], y=[0.5], z=[0.5], mode="text",
        text=["COMMERCIAL SETTLEMENT FLOW"], textposition="middle right",
        textfont=dict(size=11, color=GREY), hoverinfo="skip", showlegend=False))

    fig.update_layout(
        height=600, margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(eye=dict(x=1.9, y=-1.9, z=1.3)),
            aspectmode="manual", aspectratio=dict(x=1.5, y=1.2, z=0.85),
            bgcolor="#FFFFFF",
        ),
        paper_bgcolor="#FFFFFF",
        hoverlabel=dict(bgcolor="#FFFFFF", font_size=12, font_color=INK),
    )
    return fig

tab_sim, tab_tech, tab_docs, tab_contact = st.tabs(
    ["Simulation", "Technology", "Documentation", "Contact"])

with tab_sim:
    st.write("")
    with st.container(border=True):
        cols = st.columns(3)
        p1, p2_, p3 = (r.total("profit") for r in (r1, r2, r3))
        cols[0].metric("S1 - PPA only", INR(p1), f"DSM -{INR(r1.total('dsm_penalty'))}",
                       delta_color="inverse")
        cols[1].metric("S2 - Battery buffer", INR(p2_), f"{INR(p2_-p1)} vs S1",
                       delta_color="normal" if p2_ >= p1 else "inverse")
        cols[2].metric("S3 - Time windows", INR(p3), f"{INR(p3-p1)} vs S1",
                       delta_color="normal" if p3 >= p1 else "inverse")

        baseline_keys = ["S1 - PPA only", "S2 - Battery buffer", "S3 - Time windows"]
        best = max(baseline_keys, key=lambda k: results[k].total("profit"))
        worst_msg = (" -- the battery cannot pay for itself on DSM avoidance alone "
                     "under a flat PPA. That gap is the case for the Scenario 5 optimizer."
                     if best == "S1 - PPA only" else "")
        st.info(f"**Best baseline today: {best}** at "
                f"{INR(results[best].total('profit'))}/day{worst_msg}")

    st.write("")
    if r5 is not None:
        uplift = r5.total("profit") - results[best].total("profit")
        with st.container(border=True):
            st.markdown('<div class="au-section">Optimizer opportunity -- Scenario 5</div>',
                       unsafe_allow_html=True)
            ocol1, ocol2, ocol3 = st.columns(3)
            ocol1.metric("Optimizer profit/day", INR(r5.total("profit")))
            ocol2.metric("Uplift vs best baseline",
                        f"+{INR(uplift)}" if uplift >= 0 else INR(uplift),
                        delta_color="normal" if uplift >= 0 else "inverse")
            ocol3.metric("Annualized uplift (330d)", INR(uplift * 330))
            st.caption(
                "LP-based dispatch optimizer: chooses the schedule and battery "
                "trajectory that jointly maximize profit under the exact same "
                "CERC DSM 2024 settlement engine as the baselines above. Reported "
                "as a perfect-foresight upper bound -- the ceiling this quantifies "
                "how much value a real-time optimizer could recover.")
    elif optimizer_error:
        st.caption(f"Optimizer unavailable this run: {optimizer_error}")

    st.write("")
    st.markdown('<div class="au-section">Day profile -- schedule vs delivered</div>',
               unsafe_allow_html=True)
    tabs = st.tabs(list(results))
    for tab, (name, r) in zip(tabs, results.items()):
        with tab, st.container(border=True):
            fig = go.Figure()
            fig.add_scatter(x=hours, y=[m * 4 for m in
                            [row.actual_gen_mwh for row in r.rows]],
                            name="Actual gen (MW)", line=dict(color="#F59E0B", width=2),
                            fill="tozeroy", fillcolor="rgba(245,158,11,.10)")
            fig.add_scatter(x=hours, y=[row.scheduled_mwh * 4 for row in r.rows],
                            name="Schedule (MW)",
                            line=dict(color=INK, width=1.6, dash="dash"))
            fig.add_scatter(x=hours, y=[row.delivered_mwh * 4 for row in r.rows],
                            name="Delivered (MW)", line=dict(color=BLUE, width=1.6))
            soc = [row.soc_mwh for row in r.rows]
            if any(s is not None for s in soc):
                fig.add_scatter(x=hours, y=[s / 4 if s else 0 for s in soc],
                                name="SoC (MWh/4)",
                                line=dict(color=GREEN, width=2),
                                fill="tozeroy", fillcolor="rgba(63,174,73,.10)")
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                              legend=dict(orientation="h", y=1.1),
                              xaxis_title="Hour of day", yaxis_title="MW")
            st.plotly_chart(fig, use_container_width=True)

    st.write("")
    st.markdown('<div class="au-section">Daily P&L decomposition</div>', unsafe_allow_html=True)
    names = list(results)
    with st.container(border=True):
        fig = go.Figure()
        fig.add_bar(name="PPA revenue", x=names,
                    y=[r.total("ppa_revenue") for r in results.values()],
                    marker_color=INK)
        fig.add_bar(name="DSM receivable", x=names,
                    y=[r.total("dsm_receivable") for r in results.values()],
                    marker_color=BLUE)
        fig.add_bar(name="DSM payable", x=names,
                    y=[-r.total("dsm_payable") for r in results.values()],
                    marker_color="#DC2626")
        fig.add_bar(name="Degradation", x=names,
                    y=[-r.total("degradation") for r in results.values()],
                    marker_color="#F59E0B")
        fig.add_bar(name="O&M", x=names,
                    y=[-r.total("om") for r in results.values()], marker_color=GREY)
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
        st.dataframe(df.style.format({c: "Rs {:,.0f}" for c in df.columns[1:]}),
                    use_container_width=True, hide_index=True)

    st.write("")
    st.markdown('<div class="au-section">Researcher tools</div>', unsafe_allow_html=True)
    with st.container(border=True), st.expander("Block-level settlement (96 rows)"):
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
        st.download_button("Export CSV", bl.to_csv(index=False),
                           file_name=f"{pick[:2]}_blocks.csv")

    st.write("")
    st.markdown(f"""
    <div class="gi-header"><span class="gi-dot"></span>
    <span class="gi-title">Grid Intelligence</span></div>
    """, unsafe_allow_html=True)

    def analyst(r1, r2, r3, deg, err, r5=None):
        p1, p2, p3 = r1.total("profit"), r2.total("profit"), r3.total("profit")
        pen1 = r1.total("dsm_penalty")
        gross1 = max(r1.total("ppa_revenue"), 1.0)
        lines = [
            f"On this weather day, {err}% forecast error cost the plant "
            f"Rs {pen1:,.0f} in true DSM penalty under S1 -- "
            f"{100 * pen1 / gross1:.1f}% of gross PPA revenue."
        ]
        if p2 < p1:
            lines.append(
                f"S2 buffered most deviation but destroyed Rs {p1 - p2:,.0f}/day of value: "
                f"degradation (Rs {r2.total('degradation'):,.0f}) plus round-trip losses "
                f"exceeded the penalty avoided -- at Rs {deg:.2f}/kWh, cycling everything "
                f"is a cure costlier than the disease."
            )
        else:
            lines.append(
                f"Unusually, S2 beats S1 by Rs {p2 - p1:,.0f} today -- cheap degradation "
                f"(Rs {deg:.2f}/kWh) and heavy forecast error make pure buffering pay."
            )
        if p3 >= p1:
            lines.append(
                f"S3 overtakes S1 by Rs {p3 - p1:,.0f}: at this degradation cost, timed "
                f"shifting plus deviation buffering finally covers its own tolls."
            )
        else:
            vs2 = f"recovers Rs {p3 - p2:,.0f} versus S2 but " if p3 > p2 else ""
            lines.append(
                f"S3 {vs2}still trails S1 by Rs {p1 - p3:,.0f} -- under a flat PPA there "
                f"is no price spread to capture, so timed shifting earns nothing it can bill."
            )
        if max(p2, p3) < p1:
            lines.append(
                "Implication: with a single fixed price, no rule-based strategy beats "
                "doing nothing. That gap is the quantified case for the Scenario 5 "
                "optimizer and P2P merchant revenue."
            )
        else:
            lines.append(
                "Implication: the battery verdict is driven almost entirely by the "
                "degradation price -- confirming it with the battery team is the "
                "single highest-value open item."
            )
        if r5 is not None:
            best_baseline = max(p1, p2, p3)
            uplift = r5.total("profit") - best_baseline
            if uplift > 1:
                lines.append(
                    f"The Scenario 5 optimizer earns Rs {r5.total('profit'):,.0f}/day -- "
                    f"Rs {uplift:,.0f} more than the best baseline strategy today. That "
                    f"gap, roughly Rs {uplift*330/1e5:,.1f} lakh a year, is value left on "
                    f"the table by any fixed rule, however well-chosen."
                )
        return " ".join(lines)

    def _typed(text, delay=0.018):
        for word in text.split(" "):
            yield word + " "
            time.sleep(delay)

    with st.container(border=True):
        st.write_stream(_typed(analyst(r1, r2, r3, deg, err, r5)))


with tab_tech:
    st.write("")
    st.markdown('<div class="au-section">System diagram -- interactive 3D</div>',
               unsafe_allow_html=True)
    with st.container(border=True):
        st.plotly_chart(build_3d_diagram(), use_container_width=True)
        st.caption("Drag to rotate, scroll to zoom. Upper plane: physical "
                  "power flow (Solar to Grid, BESS buffering). Lower plane: "
                  "commercial settlement flow (Forecast to Profit/Loss), "
                  "linked by the meter-to-settlement telemetry feed.")

    st.write("")
    st.markdown('<div class="au-section">Engine internals</div>',
               unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            "- **Settlement math:** CERC DSM 2024, Regulation 6(2) (deviation "
            "as a percentage of Available Capacity) and Regulation 8(4) "
            "(marginal volume-limit penalty bands).\n"
            "- **Verification:** an automated test suite runs on every push "
            "via GitHub Actions -- the same suite this app's numbers are "
            "checked against.\n"
            "- **Same code path:** this interface calls the identical Python "
            "functions used by the command-line tool that produces the xlsx "
            "settlement reports. The UI adds nothing to the math.\n"
            "- **Configuration, not hard-coding:** CERC bands, the PPA rate, "
            "and battery parameters are read from version-controlled config "
            "files, so a regulatory amendment is a data change, not a "
            "code change."
        )
        st.link_button("View source on GitHub",
                      "https://github.com/aishwaryamishra444/dispatch-sim")

with tab_docs:

    st.markdown('<div class="au-section">Documentation</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            "**Repository:** "
            "[github.com/aishwaryamishra444/dispatch-sim]"
            "(https://github.com/aishwaryamishra444/dispatch-sim)\n\n"
            "- `dispatch_sim/core/` -- settlement engine, battery model, ledger\n"
            "- `dispatch_sim/config/` -- CERC bands, plant terms, scenario "
            "definitions (every unconfirmed value flagged in the file)\n"
            "- `dispatch_sim/runners/` -- Scenario 1-3 dispatch logic\n"
            "- `dispatch_sim/rl/` -- Scenario 5 reinforcement-learning "
            "environment\n"
            "- `dispatch_sim/tests/` -- the automated verification suite\n"
            "- `docs/` -- system design and build-guide documents"
        )

with tab_contact:

    st.markdown('<div class="au-section">Contact</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            "**Atria University -- Centre of Excellence**\n\n"
            "For questions about this simulator or the underlying research, "
            "reach out via the project repository above, or through Atria "
            "University's Centre of Excellence."
        )

st.write("")
st.markdown(
    f'<div style="display:flex;align-items:center;gap:8px;color:{GREY};'
    f'font-size:0.82rem;">{au_mark(18)}<span>Atria University Centre of '
    f'Excellence -- dispatch_sim -- github.com/aishwaryamishra444/dispatch-sim'
    f' -- values marked NEEDS-CONFIRMATION in config pending team sign-off</span></div>',
    unsafe_allow_html=True)
