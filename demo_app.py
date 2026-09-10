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

plant_capacity_mw = st.sidebar.number_input(
    "Contracted / nameplate capacity (MW)", min_value=0.05, max_value=500.0,
    value=10.0, step=0.05,
    help="The capacity in your PPA paperwork. This is the denominator CERC "
         "uses for deviation percentage (Regulation 6(2)) -- Available "
         "Capacity = this x 0.25h per block. O&M below scales proportionally "
         "from this too."
)
real_capacity_mw = st.sidebar.number_input(
    "Real achievable capacity (MW)", min_value=0.05, max_value=500.0,
    value=plant_capacity_mw, step=0.05,
    help="What the plant can genuinely produce -- may be lower than the "
         "contracted figure above (inverter clipping, site losses, panel "
         "derating). This caps actual generation; the contracted figure "
         "above still sets the DSM penalty denominator -- these are two "
         "genuinely different numbers when they differ, per PPA regulation."
)
if real_capacity_mw > plant_capacity_mw:
    st.sidebar.caption(":orange[Real capacity exceeds contracted capacity -- "
                       "generation will be capped at the contracted figure, "
                       "since you can't legally exceed what's contracted.]")

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
err = st.sidebar.slider("Generation Deviation: Actual vs Scheduled (%)", 3, 30, 12,
                        help="How far actual generation lands from the day-ahead "
                             "schedule. This single input is what drives every DSM "
                             "penalty below -- it IS the 'Actual minus Scheduled' gap, "
                             "expressed as a percentage. Watch S1/S2/S3 change as you "
                             "move it.")
ppa = st.sidebar.slider("PPA tariff (Rs/kWh)", 2.0, 4.5, 2.60, 0.05)

with st.sidebar.expander("Battery degradation -- research & calculator"):
    st.caption(
        "**Internationally sourced range.** IRENA (International Renewable "
        "Energy Agency) reports 2024 battery storage installed cost at "
        "$197/kWh, down 93% from $2,634/kWh in 2010. Peer-reviewed power "
        "systems literature commonly assumes a 3,000-cycle life for "
        "grid-scale lithium-ion; published LFP chemistry specs (the "
        "dominant chemistry for stationary storage) range 2,500-9,000 "
        "cycles. Amortizing IRENA's cost over that cycle range gives:"
    )
    st.markdown(
        "- **3,000 cycles:** ~Rs 5.58/kWh cycled\n"
        "- **6,000 cycles:** ~Rs 2.79/kWh cycled\n"
        "- **9,000 cycles:** ~Rs 1.86/kWh cycled"
    )
    st.caption(
        "(Using an approximate Rs 85/$ rate -- re-verify against a live "
        "rate for a final figure. Our Rs 2.50/kWh default sits inside this "
        "sourced range, close to the 6,000-cycle midpoint.)"
    )
    st.divider()
    st.caption(
        "**Or calculate from your own battery quote** -- same linear "
        "amortization method used in the academic literature above "
        "(cost per cycle / usable capacity)."
    )
    calc_mode = st.radio("I know the warranty in:", ["Cycles", "Years"],
                        horizontal=True, key="deg_calc_mode")
    capex = st.number_input("Battery purchase price (Rs)", min_value=0.0,
                            value=669_800_000.0, step=1_000_000.0, format="%.0f",
                            help="Default reflects IRENA's 2024 $197/kWh for a "
                                 "40 MWh system at ~Rs 85/$.")
    usable_kwh = st.number_input("Usable capacity (kWh)", min_value=1.0,
                                 value=float(st.session_state.get("cap_val", 40)) * 1000,
                                 step=1000.0, format="%.0f",
                                 help="Defaults to the BESS capacity slider below, in kWh.")
    if calc_mode == "Cycles":
        warranted_cycles = st.number_input("Warranted cycles", min_value=1.0,
                                           value=6000.0, step=100.0)
    else:
        warranted_years = st.number_input("Warranted years", min_value=0.1,
                                          value=15.0, step=0.5)
        cycles_per_day = st.number_input("Assumed full cycles per day", min_value=0.01,
                                         value=1.0, step=0.1)
        warranted_cycles = warranted_years * cycles_per_day * 330

    cost_per_cycle = capex / warranted_cycles
    computed_deg = cost_per_cycle / usable_kwh
    computed_deg_clamped = min(max(computed_deg, 0.5), 6.0)

    st.markdown(f"**-> Degradation cost: Rs {computed_deg:.2f}/kWh cycled**")
    if st.button("Apply to slider below"):
        st.session_state["deg_val"] = computed_deg_clamped
        st.rerun()

deg = st.sidebar.slider("Battery degradation (Rs/kWh cycled)", 0.5, 6.0,
                        st.session_state.get("deg_val", 2.5), 0.1, key="deg_val",
                        help="The single most sensitive input - flips S2/S3 economics. "
                             "Sourced range: Rs 1.86-5.58/kWh per IRENA 2024 + peer-"
                             "reviewed cycle-life literature (see expander above).")
cap = st.sidebar.slider("BESS capacity (MWh)", 10, 80, 40, 5, key="cap_val")

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
_REFERENCE_MW = plant["plant_mw"]  # 10.0, the config's baseline reference plant
plant["plant_mw"] = plant_capacity_mw
# O&M was calibrated as a flat Rs/day figure for the 10 MW reference plant.
# Scaling it proportionally keeps the 10 MW default byte-for-byte identical
# (scale factor = 1.0) while giving honest, non-crushing O&M for smaller
# plants -- a flat Rs 20,000/day would otherwise swamp a 300 kW system's
# entire economics, which isn't realistic.
plant["om_inr_per_day"] = plant["om_inr_per_day"] * (plant_capacity_mw / _REFERENCE_MW)
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
    # Generation is capped at what the plant can genuinely produce, not the
    # contracted figure -- AvC (plant["plant_mw"], used throughout the DSM
    # settlement engine) stays at the CONTRACTED capacity regardless, since
    # that's the PPA-paperwork figure CERC's regulation measures against.
    effective_gen_cap_mw = min(real_capacity_mw, plant_capacity_mw)
    forecast, actual, hours = make_day(st.session_state.seed, err, effective_gen_cap_mw)

r1 = run_s1(forecast, actual, plant, dsm_cfg, s1_cfg)
r2 = run_s2(forecast, actual, plant, dsm_cfg, s2_cfg,
            fresh_battery({"batteryUsableCapacity": float(cap)}))

# S3 is now ADAPTIVE: the fixed noon-charge/evening-discharge rule is only
# followed if it actually beats not using the battery at all. A rational
# operator wouldn't blindly run a battery that loses money -- so neither
# does this strategy. r3_raw is what the fixed rule alone would have
# produced; r3 is what actually gets reported, and battery_deployed tells
# the UI which one happened.
r3_raw = run_s3(forecast, actual, plant, dsm_cfg, s3_cfg,
                fresh_battery({"batteryUsableCapacity": float(cap)}))
battery_deployed = r3_raw.total("profit") > r1.total("profit")
r3 = r3_raw if battery_deployed else r1
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

# ---------------------------------------------------------------- hero
import base64

HERO_IMG_PATH = Path(__file__).parent / "hero_solar.jpg"

def solar_hero_html():
    """Cinematic hero banner: a real solar farm photo (AI-generated by the
    user, local file, no external hosting) with a slow CSS "Ken Burns"
    zoom/pan and drifting bokeh-light particles for a video-like feel --
    no actual video file needed or available, so motion is done in pure
    CSS animation instead. Rendered via components.html, the reliable path
    for anything beyond trivial inline HTML."""
    with open(HERO_IMG_PATH, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    particles = "".join(
        f'<div class="agx-bokeh" style="left:{x}%;width:{w}px;height:{w}px;'
        f'animation-delay:{d}s;animation-duration:{dur}s;"></div>'
        for x, w, d, dur in [
            (8, 10, 0, 14), (18, 6, 3, 18), (30, 14, 1.5, 16),
            (46, 8, 5, 20), (60, 12, 2, 15), (74, 7, 6, 19),
            (85, 16, 0.5, 13), (93, 9, 4, 17),
        ]
    )

    return f"""
    <style>
      @keyframes agx-kenburns {{
        0%   {{ transform: scale(1.0) translate(0,0); }}
        50%  {{ transform: scale(1.10) translate(-1.2%,-0.8%); }}
        100% {{ transform: scale(1.0) translate(0,0); }}
      }}
      @keyframes agx-float {{
        0%   {{ transform: translateY(0); opacity: 0; }}
        10%  {{ opacity: .55; }}
        90%  {{ opacity: .35; }}
        100% {{ transform: translateY(-340px); opacity: 0; }}
      }}
      .agx-bg {{
        position: absolute; inset: -4%; background-image:
          url(data:image/jpeg;base64,{b64});
        background-size: cover; background-position: center;
        animation: agx-kenburns 22s ease-in-out infinite;
      }}
      .agx-bokeh {{
        position: absolute; bottom: -20px; border-radius: 50%;
        background: radial-gradient(circle, rgba(255,238,200,.9) 0%,
                    rgba(255,238,200,0) 70%);
        filter: blur(1px);
        animation-name: agx-float; animation-timing-function: ease-in;
        animation-iteration-count: infinite;
      }}
    </style>
    <div style="position:relative;width:100%;height:400px;overflow:hidden;
                border-radius:18px;font-family:sans-serif;">
      <div class="agx-bg"></div>
      <div style="position:absolute;inset:0;background:linear-gradient(
                  180deg,rgba(8,14,26,.55) 0%,rgba(8,14,26,.30) 45%,
                  rgba(8,14,26,.62) 100%);"></div>
      {particles}
      <div style="position:absolute;inset:0;display:flex;flex-direction:column;
                  align-items:center;justify-content:center;text-align:center;
                  padding:0 24px;">
        <div style="font-size:13px;font-weight:700;letter-spacing:.22em;
                    color:#EAF1FF;text-transform:uppercase;opacity:.85;">
          Atria University -- Centre of Excellence</div>
        <div style="font-size:52px;font-weight:800;color:#FFFFFF;
                    margin-top:14px;letter-spacing:.02em;
                    text-shadow:0 4px 18px rgba(0,0,0,.45);">
          Agentic Grid Simulator</div>
        <div style="font-size:17px;color:#EAF1FF;margin-top:14px;
                    max-width:640px;opacity:.92;">
          A live digital twin of a solar-BESS plant's daily economics,
          under India's real CERC deviation settlement regulation.</div>
        <div style="margin-top:26px;color:#FFFFFF;font-size:14px;
                    font-weight:600;letter-spacing:.08em;
                    border:1.5px solid rgba(255,255,255,.55);
                    border-radius:999px;padding:10px 26px;opacity:.9;">
          SCROLL TO SIMULATE &#8595;</div>
      </div>
    </div>"""

components.html(solar_hero_html(), height=420, scrolling=False)
st.write("")

# ---------------------------------------------------------------- header
hcol1, hcol2 = st.columns([4, 2])
with hcol1:
    st.markdown(
        '<div class="au-badge"><span class="au-dot"></span>Live simulation</div>',
        unsafe_allow_html=True)
with hcol2:
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
st.caption(f"{plant_capacity_mw:g} MW plant - flat-rate PPA - CERC DSM 2024 "
          f"settlement per 15-min block - {prov}")


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
    st.markdown(
        "### :orange[Solar] + :green[Battery] Dispatch Simulator\n"
        "A live digital twin of a solar-BESS plant's daily economics -- "
        "simulating generation, storage, and grid settlement under India's "
        "real CERC deviation penalty regulation."
    )
    st.write("")

    def scenario_card(name, r, compare_to=None, badge=None, badge_kind="info",
                      live_attempt=None, breakdown_r=None):
        with st.container(border=True):
            st.markdown(f"**{name}**")
            if badge:
                (st.success if badge_kind == "good" else
                 st.warning if badge_kind == "warn" else st.caption)(badge)
            # breakdown_r lets a scenario show its OWN line items even when
            # its adopted headline profit equals another scenario's (e.g. S3
            # held-back == S1) -- so the table never looks like a duplicate.
            b = breakdown_r if breakdown_r is not None else r
            dsm_net = b.total("dsm_receivable") - b.total("dsm_payable")
            df = pd.DataFrame({
                "Line item": ["PPA revenue", "DSM net", "Degradation", "O&M"],
                "Amount (Rs)": [b.total("ppa_revenue"), dsm_net,
                               -b.total("degradation"), -b.total("om")],
            })
            st.dataframe(df.style.format({"Amount (Rs)": "{:,.0f}"}),
                        hide_index=True, use_container_width=True, height=175)

            total_sched_mwh = sum(row.scheduled_mwh for row in b.rows)
            with st.expander("How is PPA revenue calculated?"):
                st.caption(
                    f"**PPA Revenue = Total Scheduled Energy x 1,000 (MWh->kWh) x Tariff**\n\n"
                    f"= {total_sched_mwh:.4f} MWh x 1,000 x Rs {ppa:.2f}/kWh\n\n"
                    f"= **Rs {b.total('ppa_revenue'):,.2f}**\n\n"
                    f"Not driven directly by plant capacity -- driven by *scheduled "
                    f"energy*, which is the sum of 96 individual 15-minute schedules. "
                    f"Plant capacity ({plant_capacity_mw:g} MW) shapes that total "
                    f"indirectly: it caps how much any single block can schedule "
                    f"(Available Capacity = capacity x 0.25h), and the day's overall "
                    f"shape scales with it."
                )

            delta = f"{INR(r.total('profit') - compare_to)} vs S1" if compare_to is not None else None
            dcolor = ("normal" if compare_to is None or r.total("profit") >= compare_to
                     else "inverse")
            mcol1, mcol2 = st.columns(2) if live_attempt is not None else (st, None)
            mcol1.metric("Net profit / day (adopted)" if live_attempt is not None
                        else "Net profit / day",
                        INR(r.total("profit")), delta, delta_color=dcolor)
            if live_attempt is not None:
                mcol2.metric("If battery used (live)", INR(live_attempt),
                            help="This number moves instantly with the degradation "
                                 "and capacity sliders -- it's the fixed rule's raw "
                                 "result, shown even while not adopted.")

    p1 = r1.total("profit")
    baseline_keys = ["S1 - PPA only", "S2 - Battery buffer", "S3 - Time windows"]
    best = max(baseline_keys, key=lambda k: results[k].total("profit"))
    worst_msg = (" -- the battery cannot pay for itself on DSM avoidance alone "
                 "under a flat PPA. That gap is the case for the Scenario 5 optimizer."
                 if best == "S1 - PPA only" else "")
    st.info(f"**Best baseline today: {best}** at "
            f"{INR(results[best].total('profit'))}/day{worst_msg}")

    st.write("")
    st.markdown('<div class="au-section">Click a scenario to see its full '
               'breakdown and day profile</div>', unsafe_allow_html=True)
    scenario_tabs = st.tabs(list(results))
    for tab, (name, r) in zip(scenario_tabs, results.items()):
        with tab:
            if name == "S3 - Time windows":
                if battery_deployed:
                    s3_badge = (f"Battery DEPLOYED: this strategy earns Rs "
                              f"{r3_raw.total('profit'):,.0f}/day -- DSM savings cover "
                              f"the wear cost.")
                    s3_kind = "good"
                else:
                    s3_badge = (f"Battery HELD BACK: this strategy's own numbers earn "
                              f"only Rs {r3_raw.total('profit'):,.0f}/day, not enough to "
                              f"justify running the battery today, so it isn't used.")
                    s3_kind = "warn"
                scenario_card(name, r, compare_to=None, badge=s3_badge,
                             badge_kind=s3_kind, live_attempt=r3_raw.total("profit"),
                             breakdown_r=r3_raw)
            elif name == "S1 - PPA only":
                scenario_card(name, r)
            else:
                scenario_card(name, r, compare_to=p1)

            st.write("")
            # chart_r: S3's chart must always show its OWN battery physics --
            # even when held back and r==r1 for the adopted numbers, the
            # strategy still genuinely charges/discharges when run, and that
            # real SoC curve is what the chart should show, not a flat line.
            chart_r = r3_raw if name == "S3 - Time windows" else r
            sched_mw = [row.scheduled_mwh * 4 for row in chart_r.rows]
            deliv_mw = [row.delivered_mwh * 4 for row in chart_r.rows]
            actual_mw_list = [row.actual_gen_mwh * 4 for row in chart_r.rows]

            fig = go.Figure()

            # Real tiered DSM tolerance corridor around the schedule line --
            # using the actual CERC band edges from dsm_bands.yaml (5/10/20%
            # of Available Capacity), not an invented "dead-band". Band 1
            # (0-5%) is NOT zero-penalty -- it's charged at 100% of tariff,
            # just the gentlest of the four tiers. Colors deepen with
            # severity to show which tier a delivery falls into, visually.
            avc_mw_equiv = plant_capacity_mw  # AvC(MWh)=cap*DT; in MW-equiv terms (x4) = cap
            band_edges_mw = [0.05 * avc_mw_equiv, 0.10 * avc_mw_equiv, 0.20 * avc_mw_equiv]
            band_colors = ["rgba(250,204,21,.16)", "rgba(249,115,22,.14)", "rgba(220,38,38,.12)"]
            band_labels = ["Band 1 (0-5% AvC)", "Band 2 (5-10% AvC)", "Band 3 (10-20% AvC)"]
            prev_upper = sched_mw
            prev_lower = sched_mw
            for edge, color, blabel in zip(band_edges_mw, band_colors, band_labels):
                upper = [s + edge for s in sched_mw]
                lower = [max(0, s - edge) for s in sched_mw]
                fig.add_scatter(x=hours, y=upper, mode="lines", line=dict(width=0),
                               showlegend=False, hoverinfo="skip")
                fig.add_scatter(x=hours, y=prev_upper, mode="lines", line=dict(width=0),
                               fill="tonexty", fillcolor=color, showlegend=False,
                               hoverinfo="skip")
                fig.add_scatter(x=hours, y=lower, mode="lines", line=dict(width=0),
                               showlegend=False, hoverinfo="skip")
                fig.add_scatter(x=hours, y=prev_lower, mode="lines", line=dict(width=0),
                               fill="tonexty", fillcolor=color, showlegend=False,
                               hoverinfo="skip", name=blabel)
                prev_upper, prev_lower = upper, lower

            fig.add_scatter(x=hours, y=actual_mw_list,
                            name="Actual gen (MW)", line=dict(color="#F59E0B", width=2))
            fig.add_scatter(x=hours, y=sched_mw, name="Schedule (MW)",
                            line=dict(color=INK, width=2.0, dash="dash"))
            fig.add_scatter(x=hours, y=deliv_mw, name="Delivered (MW)",
                            line=dict(color=BLUE, width=2.2))

            # "Optimal vs Base" overlay -- superimpose S5's own optimal
            # schedule directly on top of this baseline's chart, so the
            # uplift is visible, not just a number to read separately.
            if r5 is not None and name != "S5 - Optimizer":
                r5_sched_mw = [row.scheduled_mwh * 4 for row in r5.rows]
                fig.add_scatter(x=hours, y=r5_sched_mw, name="S5 Optimal Schedule",
                               line=dict(color="#9333EA", width=2.2, dash="dot"))

            soc = [row.soc_mwh for row in chart_r.rows]
            has_battery = any(s is not None for s in soc)
            if has_battery:
                soc_pct = [100 * s / cap if s is not None else None for s in soc]
                fig.add_scatter(x=hours, y=soc_pct, name="Battery charge (%)",
                                line=dict(color=GREEN, width=2.6),
                                fill="tozeroy", fillcolor="rgba(63,174,73,.08)",
                                yaxis="y2")

                # Detect charging vs discharging from the SoC trajectory's
                # DOMINANT trend, not raw per-block deltas -- deviation
                # buffering causes small block-to-block noise even within a
                # genuine charging period, which would otherwise flicker
                # between "charging"/"discharging" labels. A 1-hour rolling
                # average filters that noise while still catching real cycles.
                soc_clean = [s if s is not None else soc[0] for s in soc]
                raw_deltas = [0.0] + [soc_clean[i] - soc_clean[i-1] for i in range(1, len(soc_clean))]
                W = 4  # 1 hour = 4 blocks of 15 min
                smoothed = [sum(raw_deltas[max(0,i-W+1):i+1]) / min(i+1, W)
                           for i in range(len(raw_deltas))]
                THRESH = 0.02
                charge_hrs = [hours[i] for i, d in enumerate(smoothed) if d > THRESH]
                discharge_hrs = [hours[i] for i, d in enumerate(smoothed) if d < -THRESH]

                def _shade_regions(times, color, label, label_y):
                    """Group contiguous hours into shaded background bands
                    with one clear text label per contiguous region."""
                    if not times:
                        return
                    times = sorted(times)
                    start = times[0]
                    prev = times[0]
                    regions = []
                    for t in times[1:]:
                        if t - prev > DT * 1.5:
                            regions.append((start, prev))
                            start = t
                        prev = t
                    regions.append((start, prev))
                    for s, e in regions:
                        if e - s < DT * 2:  # skip tiny slivers, keep it clean
                            continue
                        fig.add_vrect(x0=s, x1=e + DT, fillcolor=color, opacity=0.10,
                                     line_width=0, layer="below")
                        fig.add_annotation(x=(s + e + DT) / 2, y=label_y,
                                          text=f"<b>{label}</b>", showarrow=False,
                                          font=dict(size=10, color=color),
                                          opacity=0.75, yref="y2 domain",
                                          yanchor="top")

                _shade_regions(charge_hrs, "#2563EB", "CHARGING", 0.998)
                _shade_regions(discharge_hrs, "#F97316", "DISCHARGING", 0.998)

            # find and mark the single worst deviation block, with its real
            # rupee penalty, so the chart states the story directly instead
            # of leaving it to be inferred
            worst_idx = max(range(len(chart_r.rows)),
                           key=lambda i: abs(chart_r.rows[i].deviation_mwh))
            worst_row = chart_r.rows[worst_idx]
            if abs(worst_row.deviation_mwh) > 0.001:
                fig.add_scatter(
                    x=[hours[worst_idx]], y=[deliv_mw[worst_idx]],
                    mode="markers", marker=dict(size=13, color="#DC2626",
                    symbol="circle", line=dict(color="white", width=2)),
                    name="Largest deviation", showlegend=False,
                    hovertext=[f"Gap: {worst_row.deviation_mwh:+.2f} MWh this block "
                              f"-> Rs {worst_row.dsm_penalty:,.0f} penalty"],
                    hoverinfo="text")
                fig.add_annotation(
                    x=hours[worst_idx], y=deliv_mw[worst_idx],
                    text=f"<b>Largest gap:</b> {worst_row.deviation_mwh:+.2f} MWh<br>"
                         f"-> Rs {worst_row.dsm_penalty:,.0f} penalty",
                    showarrow=True, arrowhead=2, arrowsize=0.8, arrowwidth=1.3,
                    arrowcolor="#9CA3AF",
                    ax=0, ay=-42, bgcolor="#F9FAFB", bordercolor="#E5E7EB",
                    borderwidth=1, borderpad=6, font=dict(size=11, color="#374151"))
            else:
                # Zero deviation everywhere is a real, common result for the
                # optimizer (perfect foresight -> schedule exactly matches
                # generation -> zero DSM exposure, zero reason to touch the
                # battery under a flat PPA). Without this note, an empty
                # chart can look broken instead of looking like what it
                # actually is: proof of compliance.
                fig.add_annotation(
                    x=hours[len(hours)//2], y=max(actual_mw_list) * 0.5 if actual_mw_list else 1,
                    text="<b>Zero deviation achieved</b><br>Schedule exactly matches "
                         "generation -- no DSM exposure, no reason to use the battery "
                         "under a flat PPA. This is the optimum, not an empty chart.<br>"
                         "<i>True at any Generation Deviation slider setting -- the "
                         "optimizer sees the real outcome before scheduling, so its "
                         "own deviation is always zero by design. The slider still "
                         "raises the BASELINE scenarios' exposure (and therefore the "
                         "uplift number) -- just never this chart.</i>",
                    showarrow=False, bgcolor="#F0FDF4", bordercolor="#BBF7D0",
                    borderwidth=1, borderpad=8, font=dict(size=10.5, color="#166534"))

            fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                              legend=dict(orientation="h", y=1.12),
                              xaxis_title="Hour of day", yaxis_title="MW",
                              yaxis2=dict(title="Battery charge (%)", overlaying="y",
                                        side="right", range=[0, 105],
                                        showgrid=False),
                              hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True,
                           key=f"dayprofile_{name}")
            if name == "S3 - Time windows":
                st.caption("Yellow/orange/red bands around Schedule = the real "
                          "CERC tiers (5/10/20% of Available Capacity) -- Band 1 "
                          "is charged at 100% of tariff, not zero-penalty. Purple "
                          "dotted line = the optimizer's own schedule, for direct "
                          "comparison. Blue/orange bands mark this strategy's real "
                          "CHARGING/DISCHARGING physics, whether or not it's "
                          "actually adopted today (see the badge above).")
            else:
                st.caption("Yellow/orange/red bands around Schedule = the real "
                          "CERC tiers (5/10/20% of Available Capacity) -- Band 1 "
                          "is charged at 100% of tariff, not zero-penalty. Purple "
                          "dotted line = the optimizer's own schedule, for direct "
                          "comparison. Blue/orange bands mark CHARGING and "
                          "DISCHARGING. The marked point is the single worst "
                          "block of the day, with its real penalty.")

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
    st.markdown('<div class="au-section">Where the money goes -- % of gross revenue</div>',
               unsafe_allow_html=True)
    names = list(results)
    with st.container(border=True):
        st.caption("Each bar is one scenario's own gross PPA revenue, normalized "
                  "to 100% -- showing what SHARE is kept as profit versus lost to "
                  "DSM, degradation, and O&M. For exact rupee amounts, see the "
                  "table below; this answers a different question: which "
                  "scenario keeps the biggest slice of its own revenue.")
        profit_pct, dsm_pct, deg_pct, om_pct = [], [], [], []
        for r in results.values():
            gross = max(r.total("ppa_revenue"), 1.0)
            dsm_net = r.total("dsm_receivable") - r.total("dsm_payable")
            profit_pct.append(r.total("profit") / gross * 100)
            dsm_pct.append(-dsm_net / gross * 100)
            deg_pct.append(r.total("degradation") / gross * 100)
            om_pct.append(r.total("om") / gross * 100)

        fig = go.Figure()
        fig.add_bar(name="Net profit kept", x=names, y=profit_pct, marker_color=GREEN)
        fig.add_bar(name="DSM cost", x=names, y=dsm_pct, marker_color="#DC2626")
        fig.add_bar(name="Degradation cost", x=names, y=deg_pct, marker_color="#F59E0B")
        fig.add_bar(name="O&M cost", x=names, y=om_pct, marker_color=GREY)
        fig.update_layout(barmode="stack", height=320,
                          margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=1.12),
                          yaxis_title="% of gross PPA revenue")
        st.plotly_chart(fig, use_container_width=True, key="pl_composition")

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
    st.markdown('<div class="au-section">Multi-day backtest -- beyond a single day</div>',
               unsafe_allow_html=True)
    with st.container(border=True):
        st.caption(
            "A single weather day is a snapshot, not a risk profile. This runs "
            "every scenario -- including the optimizer -- across many different "
            "weather days at your current slider settings, and shows the spread "
            "of outcomes, not just one day's number. Directly addresses the "
            "'single-day focus' limitation: a real investment decision should "
            "look at this distribution, not one day."
        )
        n_days = st.slider("Number of days to backtest", 5, 60, 20, 5)
        run_backtest = st.button("Run backtest")

        if run_backtest:
            rng_master = np.random.default_rng(2026)
            seeds = rng_master.integers(1, int(1e9), size=n_days)
            rows = []
            progress = st.progress(0.0, text="Running backtest...")
            for i, seed in enumerate(seeds):
                bt_fc, bt_act, _ = make_day(int(seed), err, plant["plant_mw"])
                bt_r1 = run_s1(bt_fc, bt_act, plant, dsm_cfg, s1_cfg)
                bt_r2 = run_s2(bt_fc, bt_act, plant, dsm_cfg, s2_cfg,
                              fresh_battery({"batteryUsableCapacity": float(cap)}))
                bt_r3_raw = run_s3(bt_fc, bt_act, plant, dsm_cfg, s3_cfg,
                                  fresh_battery({"batteryUsableCapacity": float(cap)}))
                bt_deployed = bt_r3_raw.total("profit") > bt_r1.total("profit")
                bt_r3 = bt_r3_raw if bt_deployed else bt_r1
                row = {"day": i + 1, "seed": int(seed),
                      "S1": bt_r1.total("profit"), "S2": bt_r2.total("profit"),
                      "S3": bt_r3.total("profit"),
                      "S3_battery_used": bt_deployed}
                try:
                    bt_r5 = solve_optimal_dispatch(
                        bt_fc, bt_act, plant, dsm_cfg,
                        OptimizerBatterySpec(float(cap), 20.0, 0.88, 10, 90), deg)
                    row["S5"] = bt_r5.total("profit")
                except Exception:
                    row["S5"] = None
                rows.append(row)
                progress.progress((i + 1) / n_days, text=f"Day {i+1}/{n_days}")
            progress.empty()

            bt_df = pd.DataFrame(rows)
            st.session_state["backtest_df"] = bt_df

        if "backtest_df" in st.session_state:
            bt_df = st.session_state["backtest_df"]
            scen_cols = [c for c in ["S1", "S2", "S3", "S5"] if c in bt_df.columns]

            summary = pd.DataFrame({
                "Scenario": scen_cols,
                "Mean profit/day": [bt_df[c].mean() for c in scen_cols],
                "Min": [bt_df[c].min() for c in scen_cols],
                "Max": [bt_df[c].max() for c in scen_cols],
                "Std dev": [bt_df[c].std() for c in scen_cols],
                "Win rate (best day)": [
                    f"{(bt_df[scen_cols].idxmax(axis=1) == c).mean()*100:.0f}%"
                    for c in scen_cols],
            })
            st.dataframe(
                summary.style.format({c: "Rs {:,.0f}" for c in
                                     ["Mean profit/day", "Min", "Max", "Std dev"]}),
                hide_index=True, use_container_width=True)

            fig_bt = go.Figure()
            colors_bt = {"S1": GREY, "S2": "#DC2626", "S3": "#F59E0B", "S5": BLUE}
            for c in scen_cols:
                fig_bt.add_scatter(x=bt_df["day"], y=bt_df[c], name=c, mode="lines+markers",
                                  line=dict(color=colors_bt.get(c, INK), width=1.6))
            fig_bt.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                                legend=dict(orientation="h", y=1.12),
                                xaxis_title="Backtest day", yaxis_title="Profit (Rs)")
            st.plotly_chart(fig_bt, use_container_width=True, key="backtest_chart")

            if "S3" in bt_df.columns:
                s3_rate = bt_df["S3_battery_used"].mean() * 100
                st.caption(f"S3 deployed its battery on {s3_rate:.0f}% of backtested days "
                          f"at current settings -- confirming the adaptive logic responds "
                          f"to real day-to-day variation, not just one fixed day.")

    st.write("")
    st.markdown(f"""
    <div class="gi-header"><span class="gi-dot"></span>
    <span class="gi-title">Grid Intelligence</span></div>
    """, unsafe_allow_html=True)

    def analyst(r1, r2, r3_raw, deg, err, r5=None):
        p1, p2, p3 = r1.total("profit"), r2.total("profit"), r3_raw.total("profit")
        pen1 = r1.total("dsm_penalty")
        gross1 = max(r1.total("ppa_revenue"), 1.0)
        lines = [
            f"On this weather day, a {err}% Actual-vs-Scheduled deviation cost the "
            f"plant Rs {pen1:,.0f} in true DSM penalty under S1 -- "
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
                f"(Rs {deg:.2f}/kWh) and a large Actual-vs-Scheduled gap make pure "
                f"buffering pay."
            )
        if battery_deployed:
            lines.append(
                f"S3's fixed rule (charge at peak sun, discharge in the evening) earns "
                f"Rs {p3:,.0f}/day on its own economics: at this degradation cost, timed "
                f"shifting plus deviation buffering covers its own wear cost -- so the "
                f"battery is deployed today."
            )
        else:
            lines.append(
                f"S3's fixed rule (charge at peak sun, discharge in the evening) earns "
                f"only Rs {p3:,.0f}/day on its own economics -- not enough to justify "
                f"running the battery today. Even at low degradation cost, the battery's "
                f"own round-trip efficiency loss (~12% per cycle) is often enough on its "
                f"own to erase what little DSM saving timed shifting buys."
            )
        if max(p2, p3) < p1:
            lines.append(
                "Implication: with a single fixed price, no rule-based strategy reliably "
                "beats doing nothing -- driven jointly by degradation cost and the "
                "battery's unavoidable round-trip efficiency loss, not degradation alone. "
                "That gap is the quantified case for the Scenario 5 optimizer and P2P "
                "merchant revenue."
            )
        else:
            lines.append(
                "Implication: the battery verdict is close today -- confirming the real "
                "degradation price with the battery team is the single highest-value "
                "open item."
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
        st.write_stream(_typed(analyst(r1, r2, r3_raw, deg, err, r5)))


with tab_tech:
    st.write("")
    st.markdown('<div class="au-section">System diagram -- interactive 3D</div>',
               unsafe_allow_html=True)
    with st.container(border=True):
        st.plotly_chart(build_3d_diagram(), use_container_width=True, key="diagram_3d")
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
