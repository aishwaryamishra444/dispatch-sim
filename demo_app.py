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
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

# Streamlit Cloud updates the files on a push but keeps the running Python
# process, so an already-imported dispatch_sim package would stay at the OLD
# version while this script is the NEW one -- a new import then fails
# (seen live: ImportError on charge_trigger_mw). Dropping the package from
# the module cache makes every run import the files actually on disk.
import sys as _sys
for _m in [m for m in _sys.modules if m == "dispatch_sim" or m.startswith("dispatch_sim.")]:
    del _sys.modules[_m]

from dispatch_sim.core.battery import Battery
from dispatch_sim.io.loaders import load_dsm_config, load_yaml, load_series_csv_buffer
from dispatch_sim.io.engine_file import detect_columns, detect_unit, parse_engine_file, read_table
from dispatch_sim.io.iex_loader import IEXFormatError, parse_iex_file
from dispatch_sim.runners.rules import run_s1, run_s2, run_s3, run_s4, discharge_trigger_mw, charge_trigger_mw
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


# Power rating scales with energy capacity: the 40 MWh / 20 MW reference
# system is 0.5C. A fixed 20 MW would make a 10 MWh battery a physically
# unrealistic 2C unit and an 80 MWh one an under-powered 0.25C unit.
C_RATE_PER_MWH = 0.5


def fresh_battery(spec_overrides=None):
    spec = {"batteryUsableCapacity": 40.0, "cRateMW": None,
            "roundTripEfficiency": 0.88, "socMinPct": 10, "socMaxPct": 90}
    if spec_overrides:
        spec.update(spec_overrides)
    if spec["cRateMW"] is None:
        spec["cRateMW"] = C_RATE_PER_MWH * float(spec["batteryUsableCapacity"])
    return Battery(usable_capacity_mwh=spec["batteryUsableCapacity"],
                   c_rate_mw=spec["cRateMW"], rte=spec["roundTripEfficiency"],
                   soc_min_pct=spec["socMinPct"], soc_max_pct=spec["socMaxPct"])


# ---------------------------------------------------------------- settings
# All inputs live on the "Set up" page, but every page needs them. Widget
# values normally vanish when their page isn't shown, so each input is kept
# in session_state under its widget key and re-asserted on every run.
ss = st.session_state

MODE_SYNTH = "Synthetic (demo)"
MODE_ENGINE = "Forecast-engine file (multi-day)"
MODE_SINGLE = "Upload real day"
SCALE_SCALED = "Scaled to the contracted plant"
SCALE_OWN = "Its own size (contracted capacity set to match)"

DEFAULTS = {
    "plant_cap": 10.0, "real_cap": 10.0, "data_mode": MODE_SYNTH, "err": 12,
    "ppa": 2.60, "bess_hours": 4.0, "deg_calc_mode": "Cycles",
    "deg_price": 16_320.0, "deg_cycles": 6000.0, "deg_years": 15.0, "deg_cpd": 1.0,
    "engine_unit": "kW", "engine_tz": "UTC", "engine_src_cap": 5.0,
    "engine_scale": SCALE_SCALED, "seed": 20260614,
}
for _k, _v in DEFAULTS.items():
    ss.setdefault(_k, _v)
for _k in list(DEFAULTS) + ["engine_day"]:
    if _k in ss:
        ss[_k] = ss[_k]   # keep the value alive while its widget isn't on screen


@st.cache_data(show_spinner=False)
def _parse_engine_cached(raw: bytes, unit: str, naive_tz: str):
    return parse_engine_file(raw, unit=unit, naive_timestamps=naive_tz)


def _nice_ceiling(x: float) -> float:
    """Round up to a tidy plant size: 4.2 kW -> 5 kW, 7.97 MW -> 8 MW."""
    if x <= 0:
        return 1.0
    step = 10 ** np.floor(np.log10(x))
    return float(np.ceil(x / step - 1e-9) * step)


def _engine_has_offset(raw: bytes) -> bool:
    try:
        head = read_table(raw)
        t_c, _, _ = detect_columns(head.columns)
        return bool(t_c is not None and head[t_c].astype(str)
                    .str.contains(r"(?:[+-]\d{2}:?\d{2}|Z)$", regex=True).all())
    except Exception:  # noqa: BLE001
        return False


def _reset_engine_capacity():
    """Default 'this plant's capacity' to the file's peak, rounded up, in the
    chosen unit -- re-run whenever the file or its unit changes."""
    try:
        eng_ = _parse_engine_cached(ss["engine_raw"], ss["engine_unit"], ss["engine_tz"])
        peak = eng_.peak_mw * (1000 if ss["engine_unit"] == "kW" else 1)
        ss["engine_src_cap"] = _nice_ceiling(peak)
    except Exception:  # noqa: BLE001 -- the Setup page shows the parse error
        pass


def _on_engine_upload():
    f = ss.get("engine_upload")
    if f is None:
        for k in ("engine_raw", "engine_name", "engine_day"):
            ss.pop(k, None)
        return
    _ingest_engine_bytes(f.getvalue(), f.name)


def _ingest_engine_bytes(raw: bytes, name: str):
    ss["engine_raw"], ss["engine_name"] = raw, name
    ss.pop("engine_day", None)
    try:
        head = read_table(raw)
        _, a_c, f_c = detect_columns(head.columns)
        ss["engine_unit"] = (detect_unit(a_c, f_c) if a_c and f_c else None) or "kW"
    except Exception:  # noqa: BLE001
        pass
    _reset_engine_capacity()


def _on_single_upload(which: str):
    f = ss.get(f"{which}_upload")
    if f is None:
        ss.pop(f"single_{which}", None)
        ss.pop(f"single_{which}_name", None)
        ss.pop(f"single_{which}_error", None)
        return
    try:
        ss[f"single_{which}"] = load_series_csv_buffer(f, f.name)
        ss[f"single_{which}_name"] = f.name
        ss.pop(f"single_{which}_error", None)
    except Exception as e:  # noqa: BLE001
        ss.pop(f"single_{which}", None)
        ss[f"single_{which}_error"] = str(e)


# ---- derive this run's inputs from the saved settings (no widgets here) ----
plant_capacity_mw = float(ss["plant_cap"])
real_capacity_mw = float(ss["real_cap"])
data_mode = ss["data_mode"]
err = int(ss["err"])
ppa = float(ss["ppa"])
bess_hours = float(ss["bess_hours"])

real_forecast, real_actual = None, None
engine_days_mw = None        # {day: (forecast_mw, actual_mw)} at simulator scale
selected_engine_day = None
real_source_note = ""
engine_error = None
eng = None
if data_mode == MODE_ENGINE and ss.get("engine_raw"):
    try:
        eng = _parse_engine_cached(ss["engine_raw"], ss["engine_unit"], ss["engine_tz"])
    except Exception as e:  # noqa: BLE001
        engine_error = str(e)
    if eng is not None:
        unit_k = 1000 if ss["engine_unit"] == "kW" else 1
        src_cap_mw = max(float(ss["engine_src_cap"]), 1e-6) / unit_k
        if ss["engine_scale"] != SCALE_OWN:
            k_scale = plant_capacity_mw / src_cap_mw
        else:
            k_scale = 1.0
            plant_capacity_mw = src_cap_mw
            real_capacity_mw = src_cap_mw
        engine_days_mw = {d: ([x * k_scale for x in fc], [x * k_scale for x in ac])
                          for d, (fc, ac) in eng.days.items()}
        if ss.get("engine_day") not in engine_days_mw:
            ss["engine_day"] = next(iter(engine_days_mw))
        selected_engine_day = ss["engine_day"]
        real_forecast, real_actual = engine_days_mw[selected_engine_day]
        src_txt = (f"{src_cap_mw * 1000:g} kW" if src_cap_mw < 1 else f"{src_cap_mw:g} MW")
        real_source_note = (f"forecast-engine data, {selected_engine_day} (IST)"
                            + (f", scaled from {src_txt}" if k_scale != 1.0 else ""))
elif data_mode == MODE_SINGLE and ss.get("single_fc") and ss.get("single_act"):
    real_forecast, real_actual = ss["single_fc"], ss["single_act"]
    real_source_note = (f"uploaded day ({ss.get('single_fc_name')} / "
                        f"{ss.get('single_act_name')})")

# Unit / capacity sanity check for any real data: catches a kW file read as
# MW, or a profile from a different-sized plant, before it silently produces
# meaningless settlement numbers.
data_warnings = []
if real_forecast is not None and real_actual is not None:
    up_peak = max(max(real_forecast), max(real_actual))
    if up_peak > plant_capacity_mw * 1.02:
        data_warnings.append(
            f"This day peaks at {up_peak:.3f} MW -- above the {plant_capacity_mw:g} MW "
            f"contracted capacity. Check the unit (kW vs MW) or raise the capacity; "
            f"deviation percentages will be wrong otherwise.")
    elif 0 < up_peak < plant_capacity_mw * 0.10:
        data_warnings.append(
            f"This day peaks at only {up_peak:.4f} MW -- under 10% of the "
            f"{plant_capacity_mw:g} MW contracted capacity. If the values are kW or "
            f"come from a smaller plant, use the 'Forecast-engine file' option to set "
            f"the unit and scale, or lower the capacity.")

# Battery wear cost: price per kWh (scaled by capacity) over warranted cycles.
usable_kwh = bess_hours * plant_capacity_mw * 1000
capex = float(ss["deg_price"]) * usable_kwh
if ss["deg_calc_mode"] == "Cycles":
    warranted_cycles = float(ss["deg_cycles"])
else:
    warranted_cycles = float(ss["deg_years"]) * float(ss["deg_cpd"]) * 330
computed_deg = capex / max(warranted_cycles, 1.0) / max(usable_kwh, 1e-9)
DEG_MIN, DEG_MAX = 0.5, 6.0
deg = min(max(computed_deg, DEG_MIN), DEG_MAX)

cap = bess_hours * plant_capacity_mw   # usable MWh


def _size_text(mw: float) -> str:
    return f"{mw:g} MW" if mw >= 1 else f"{mw * 1000:g} kW"


def _energy_text(mwh: float) -> str:
    return f"{mwh:,.3g} MWh" if mwh >= 1 else f"{mwh * 1000:,.3g} kWh"


# ---- sidebar: brand + a short summary of what is being simulated ----------
st.sidebar.markdown(
    f'<div class="au-mark-row">{au_mark(30)}'
    f'<div><div class="au-mark-text">Atria University</div>'
    f'<div class="au-mark-sub">Centre of Excellence</div></div></div>',
    unsafe_allow_html=True)
st.sidebar.markdown("**Agentic Grid Simulator**")


def sidebar_summary():
    data_txt = {MODE_SYNTH: f"Synthetic day, {err}% forecast error",
                MODE_ENGINE: (f"Forecast-engine file, {selected_engine_day}"
                              if engine_days_mw else "Forecast-engine file (none loaded)"),
                MODE_SINGLE: ("Uploaded day" if real_forecast is not None
                              else "Single day (not uploaded yet)")}[data_mode]
    st.sidebar.caption("**Simulating**")
    st.sidebar.caption(
        f"{_size_text(plant_capacity_mw)} plant · Rs {ppa:.2f}/kWh  \n"
        f"{data_txt}  \n"
        f"Battery {_energy_text(cap)} ({bess_hours:g} h) · wear Rs {deg:.2f}/kWh")
    st.sidebar.page_link(PAGE_SETUP, label="Change settings", icon=":material/tune:")
    st.sidebar.divider()
    st.sidebar.caption("Engine: dispatch_sim · CERC DSM 2024 (Reg 6(2)/8(4)) · "
                       "60+ automated tests. The app uses the same code as the "
                       "xlsx reports; the UI adds nothing to the math.")



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
s4_cfg = load_yaml(CFG / "scenario_s4.yaml"); s4_cfg["degradation_inr_per_kwh"] = deg
s1_cfg = load_yaml(CFG / "scenario_s1.yaml")

using_real = real_forecast is not None and real_actual is not None
if using_real:
    forecast, actual = real_forecast, real_actual
    hours = np.arange(BLOCKS) * DT
    # The real gap between the two series, as % of contracted capacity --
    # what Grid Intelligence reports instead of the hidden slider's default.
    _gap_pct = [abs(a_ - f_) / plant_capacity_mw * 100 for a_, f_ in zip(actual, forecast)]
    _day = [g for g, f_ in zip(_gap_pct, forecast) if f_ > 0.01 * plant_capacity_mw]
    real_gap_mean_pct = float(np.mean(_day)) if _day else 0.0
    real_gap_max_pct = max(_gap_pct)
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

# S4 - Real-time reactive controller: genuinely deployable (no perfect
# foresight, unlike S5), reacting block-by-block to each block's own
# actual-vs-schedule gap. Its discharge trigger is a real CERC band edge --
# by default the first band whose penalty exceeds the cost of stored energy
# (see run_s4 / scenario_s4.yaml).
r4 = run_s4(forecast, actual, plant, dsm_cfg, s4_cfg,
           fresh_battery({"batteryUsableCapacity": float(cap)}))
s4_trigger_mw = discharge_trigger_mw(plant, dsm_cfg, s4_cfg,
                                     fresh_battery({"batteryUsableCapacity": float(cap)}))
s4_trigger_pct = (100 * s4_trigger_mw / plant["plant_mw"]
                  if s4_trigger_mw != float("inf") else None)
s4_charge_mw = charge_trigger_mw(plant, dsm_cfg, s4_cfg,
                                 fresh_battery({"batteryUsableCapacity": float(cap)}))


def _mw_text(mw):
    return f"{mw:.2f} MW" if mw >= 1 else f"{mw * 1000:.2f} kW"


s4_trigger_text = ("never discharges at these costs -- no CERC band's penalty "
                   "exceeds the cost of stored energy"
                   if s4_trigger_pct is None else
                   f"discharges only for shortfalls beyond {s4_trigger_pct:.0f}% of "
                   f"Available Capacity ({_mw_text(s4_trigger_mw)})")
s4_charge_text = ("never stores excess -- the grid always pays more for it than "
                  "storing keeps" if s4_charge_mw == float("inf") else
                  f"stores only excess beyond {100 * s4_charge_mw / plant['plant_mw']:.0f}% "
                  f"of Available Capacity ({_mw_text(s4_charge_mw)}), where the grid "
                  f"pays less for it than storing keeps"
                  if s4_cfg.get("discharge_trigger") == "economic" else
                  f"absorbs excess beyond the Band 1 edge ({_mw_text(s4_charge_mw)})")

results = {"S1 - PPA only": r1, "S2 - Battery buffer": r2, "S3 - Time windows": r3,
          "S4 - Real-time Controller": r4}

optimizer_error = None
try:
    r5 = solve_optimal_dispatch(
        forecast, actual, plant, dsm_cfg,
        OptimizerBatterySpec(float(cap), C_RATE_PER_MWH * float(cap), 0.88, 10, 90), deg)
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
          GET STARTED BELOW &#8595;</div>
      </div>
    </div>"""

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

HOME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --hx-bg-1: #0f0b30; --hx-bg-2: #211a5c; --hx-bg-3: #2f2577;
  --hx-ink: #f4f2ff; --hx-mute: #b9b3e6; --hx-dim: #8c85c4;
  --hx-line: rgba(196, 186, 255, 0.16); --hx-card: rgba(76, 62, 170, 0.28);
  --hx-violet: #a996ff; --hx-teal: #5fe3c0; --hx-coral: #ff9c8a;
}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: radial-gradient(1200px 600px at 50% -10%, var(--hx-bg-3) 0%, transparent 60%),
              linear-gradient(180deg, var(--hx-bg-1) 0%, var(--hx-bg-2) 55%, var(--hx-bg-1) 100%) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stHeader"] * { color: var(--hx-mute) !important; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"] { display: none !important; }
[data-testid="stMainBlockContainer"], .block-container {
  max-width: 1240px !important; padding-top: 1.2rem !important;
}
[data-testid="stMain"] p, [data-testid="stMain"] span { color: var(--hx-mute); }

/* nav */
.st-key-hx_nav { border-bottom: 1px solid var(--hx-line); padding-bottom: 10px; }
.st-key-hx_nav [data-testid="stPageLink-NavLink"] {
  background: transparent !important; justify-content: center;
}
.st-key-hx_nav [data-testid="stPageLink-NavLink"] [data-testid="stIconMaterial"],
.st-key-hx_nav [data-testid="stPageLink-NavLink"] [data-testid="stIconEmoji"] { display: none !important; }
.st-key-hx_nav [data-testid="stPageLink-NavLink"] p {
  font-family: 'JetBrains Mono', monospace !important; font-size: 12.5px !important;
  letter-spacing: .16em; text-transform: uppercase; color: var(--hx-ink) !important;
}
.st-key-hx_nav [data-testid="stPageLink-NavLink"]:hover p { color: var(--hx-teal) !important; }
.st-key-hx_navcta [data-testid="stPageLink-NavLink"] {
  border: 1px solid rgba(244, 242, 255, .55); border-radius: 999px; padding: 4px 18px;
}

/* buttons */
.st-key-hx_start button, .st-key-hx_start2 button, .st-key-hx_demo button {
  border-radius: 999px !important; min-height: 48px;
  font-family: 'JetBrains Mono', monospace !important;
}
.st-key-hx_start button p, .st-key-hx_start2 button p, .st-key-hx_demo button p {
  font-family: 'JetBrains Mono', monospace !important; font-size: 13.5px !important;
  letter-spacing: .1em; text-transform: uppercase; white-space: nowrap;
}
.st-key-hx_start button, .st-key-hx_start2 button {
  background: linear-gradient(90deg, var(--hx-violet) 0%, var(--hx-teal) 100%) !important;
  border: 0 !important; box-shadow: 0 8px 30px rgba(95, 227, 192, .18);
}
.st-key-hx_start button p, .st-key-hx_start2 button p { color: #120d36 !important; font-weight: 500; }
.st-key-hx_start button:hover, .st-key-hx_start2 button:hover { filter: brightness(1.08); }
.st-key-hx_demo button {
  background: transparent !important; border: 1px solid rgba(244, 242, 255, .5) !important;
}
.st-key-hx_demo button p { color: var(--hx-ink) !important; }
.st-key-hx_demo button:hover { border-color: var(--hx-teal) !important; }
.st-key-hx_demo button:hover p { color: var(--hx-teal) !important; }

/* page content */
.hx-brand { display: flex; align-items: center; gap: 10px; padding-top: 4px; }
.hx-brand b { font-family: 'Inter Tight', sans-serif; font-weight: 600; font-size: 21px;
  letter-spacing: .14em; color: var(--hx-ink); }
.hx-hero { text-align: center; padding: 72px 12px 34px; }
.hx-kicker { font-family: 'JetBrains Mono', monospace; font-size: 12.5px; letter-spacing: .22em;
  text-transform: uppercase; color: var(--hx-teal); margin-bottom: 22px; }
.hx-h1 { font-family: 'Inter Tight', sans-serif; font-weight: 500; color: var(--hx-ink);
  font-size: clamp(38px, 6.2vw, 82px); line-height: 1.04; letter-spacing: -0.025em;
  max-width: 1050px; margin: 0 auto; }
.hx-sub { font-family: 'Inter Tight', sans-serif; font-size: clamp(16px, 1.6vw, 20px);
  line-height: 1.55; color: var(--hx-mute); max-width: 760px; margin: 26px auto 0; }
.hx-cards { display: grid; grid-template-columns: 1fr 1.55fr 1fr; gap: 26px; margin: 56px 0 10px; }
.hx-card { background: var(--hx-card); border: 1px solid var(--hx-line); border-radius: 14px;
  padding: 18px 18px 14px; position: relative; overflow: hidden; }
.hx-card-t { font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: .16em;
  text-transform: uppercase; color: var(--hx-ink); text-align: center; margin-bottom: 8px; }
.hx-card-c { font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: var(--hx-dim);
  letter-spacing: .06em; text-align: center; margin-top: 4px; }
.hx-photo { border-radius: 14px; min-height: 300px; background-size: cover;
  background-position: center; border: 1px solid var(--hx-line); position: relative; }
.hx-photo span { position: absolute; left: 16px; bottom: 14px; font-family: 'JetBrains Mono', monospace;
  font-size: 11px; letter-spacing: .16em; text-transform: uppercase; color: #fff !important;
  background: rgba(15, 11, 48, .55); padding: 5px 10px; border-radius: 999px; }
.hx-sec { margin-top: 96px; }
.hx-label { font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: .22em;
  text-transform: uppercase; color: var(--hx-teal); }
.hx-h2 { font-family: 'Inter Tight', sans-serif; font-weight: 500; color: var(--hx-ink);
  font-size: clamp(28px, 3.6vw, 46px); line-height: 1.1; letter-spacing: -0.02em;
  margin: 12px 0 34px; max-width: 820px; }
.hx-grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 22px; }
.hx-grid5 { display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; }
.hx-grid4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 22px; }
.hx-step { border-top: 1px solid var(--hx-line); padding-top: 20px; }
.hx-step .n { font-family: 'JetBrains Mono', monospace; color: var(--hx-violet); font-size: 13px;
  letter-spacing: .14em; }
.hx-step h3 { font-family: 'Inter Tight', sans-serif; font-weight: 500; color: var(--hx-ink);
  font-size: 24px; margin: 10px 0 10px; padding: 0; }
.hx-step p, .hx-strat p, .hx-stat p { font-family: 'Inter Tight', sans-serif; color: var(--hx-mute);
  font-size: 16px; line-height: 1.55; margin: 0; }
.hx-strat { background: var(--hx-card); border: 1px solid var(--hx-line); border-radius: 12px;
  padding: 18px 16px; }
.hx-strat .tag { font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: .14em;
  color: var(--hx-teal); }
.hx-strat h4 { font-family: 'Inter Tight', sans-serif; font-weight: 500; color: var(--hx-ink);
  font-size: 18px; margin: 8px 0 6px; padding: 0; }
.hx-strat p { font-size: 14.5px; }
.hx-stat { border-left: 1px solid var(--hx-line); padding-left: 18px; }
.hx-stat .v { font-family: 'Inter Tight', sans-serif; font-weight: 500; color: var(--hx-ink);
  font-size: clamp(34px, 4vw, 52px); letter-spacing: -0.02em; line-height: 1; }
.hx-stat p { font-size: 15px; margin-top: 10px; }
.hx-final { text-align: center; margin: 110px 0 24px; }
@media (max-width: 860px) {
  .hx-cards, .hx-grid3, .hx-grid4 { grid-template-columns: 1fr; }
  .hx-grid5 { grid-template-columns: 1fr 1fr; }
  .hx-hero { padding-top: 36px; }
  .st-key-hx_nav [data-testid="stColumn"]:has([class*="st-key-hx_l"]),
  .st-key-hx_nav [data-testid="stColumn"]:has(.st-key-hx_navcta) { display: none !important; }
}
</style>
"""


def _svg_img(svg: str, alt: str, style: str = "width:100%;display:block") -> str:
    """st.html strips inline <svg>, so embed it as an image instead."""
    b64_ = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img alt="{alt}" style="{style}" src="data:image/svg+xml;base64,{b64_}"/>'


def _hx_profile_svg(fc, ac, w=320, h=210):
    """Forecast vs actual for the day being simulated, as a small SVG chart."""
    n = len(fc)
    top = max(max(fc), max(ac), 1e-9) * 1.1
    px = lambda i: 30 + (w - 40) * i / (n - 1)
    py = lambda v: 10 + (h - 40) * (1 - v / top)
    grid = "".join(
        f'<line x1="{30 + (w - 40) * k / 8:.1f}" y1="10" x2="{30 + (w - 40) * k / 8:.1f}" '
        f'y2="{h - 30}" stroke="rgba(196,186,255,.13)"/>' for k in range(9))
    grid += "".join(
        f'<line x1="30" y1="{10 + (h - 40) * k / 5:.1f}" x2="{w - 10}" '
        f'y2="{10 + (h - 40) * k / 5:.1f}" stroke="rgba(196,186,255,.13)"/>' for k in range(6))
    line = lambda vals, col: (
        '<polyline fill="none" stroke="{}" stroke-width="2" stroke-linejoin="round" '
        'points="{}"/>'.format(col, " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))))
    ticks = "".join(
        f'<text x="{px(i):.1f}" y="{h - 12}" fill="#8c85c4" font-size="10" '
        f'font-family="JetBrains Mono, monospace" text-anchor="middle">{hh:02d}</text>'
        for i, hh in ((0, 0), (n // 4, 6), (n // 2, 12), (3 * n // 4, 18), (n - 1, 24)))
    unit = "MW" if top >= 1 else "kW"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'>{grid}'
            f'<text x="12" y="{h / 2:.0f}" fill="#8c85c4" font-size="10" '
            f'font-family="JetBrains Mono, monospace" transform="rotate(-90 12 {h / 2:.0f})" '
            f'text-anchor="middle">{unit}</text>'
            f'{line(fc, "#a996ff")}{line(ac, "#5fe3c0")}{ticks}</svg>')


def _hx_gain_svg(gains, w=320, h=210):
    """Diverging bars: each strategy's profit minus S1's, for this day."""
    # Scale to the gains, so one large loss (S2 often loses heavily) doesn't
    # flatten every other bar; a clipped bar gets a break mark and keeps its
    # exact label.
    pos = [g for _, g in gains if g > 0]
    span = max(max(pos) * 1.25 if pos else max(abs(g) for _, g in gains), 1.0)
    zero_y = 20 + (h - 60) / 2
    bw = (w - 60) / len(gains)
    out = [f'<line x1="30" y1="{zero_y:.1f}" x2="{w - 10}" y2="{zero_y:.1f}" '
           f'stroke="rgba(244,242,255,.45)"/>']
    for k in range(5):
        y = 20 + (h - 60) * k / 4
        out.append(f'<line x1="30" y1="{y:.1f}" x2="{w - 10}" y2="{y:.1f}" '
                   f'stroke="rgba(196,186,255,.13)"/>')
    for i, (tag, g) in enumerate(gains):
        x = 34 + i * bw
        clipped = abs(g) > span
        bh = min(abs(g), span) / span * (h - 60) / 2
        y = zero_y - bh if g >= 0 else zero_y
        col = "#5fe3c0" if g > 0 else ("#ff9c8a" if g < 0 else "#a996ff")
        out.append(f'<rect x="{x + bw * .18:.1f}" y="{y:.1f}" width="{bw * .64:.1f}" '
                   f'height="{max(bh, 1.5):.1f}" rx="3" fill="{col}" opacity=".9"/>')
        if clipped:
            by = y + bh - 9 if g < 0 else y + 9
            out.append(f'<path d="M{x + bw * .12:.1f} {by:.1f} l{bw * .25:.1f} -4 '
                       f'l{bw * .25:.1f} 8 l{bw * .26:.1f} -4" stroke="#211a5c" '
                       f'stroke-width="3" fill="none"/>')
        lab = "0" if abs(g) < 0.5 else f"{g:+,.0f}"
        ly = (y - 6) if g >= 0 else (y + bh + 13)
        out.append(f'<text x="{x + bw / 2:.1f}" y="{ly:.1f}" fill="#f4f2ff" font-size="10" '
                   f'font-family="JetBrains Mono, monospace" text-anchor="middle">{lab}</text>')
        out.append(f'<text x="{x + bw / 2:.1f}" y="{h - 12}" fill="#8c85c4" font-size="10.5" '
                   f'font-family="JetBrains Mono, monospace" text-anchor="middle">{tag}</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'>{"".join(out)}</svg>')


def page_home():
    st.html(HOME_CSS)

    # ---- nav bar -----------------------------------------------------------
    with st.container(key="hx_nav"):
        n0, n1, n2, n3, n4, n5 = st.columns([2.4, 1, 1, 1.25, 1.35, 1.2],
                                            vertical_alignment="center")
        n0.html(f'<div class="hx-brand">{_svg_img(au_mark(28), "Atria University", "width:28px;height:28px")}'
                f'<b>AGENTIC GRID</b></div>')
        for i_, (col_, page_, lab_) in enumerate((
                (n1, PAGE_SETUP, "Set up"), (n2, PAGE_RESULTS, "Results"),
                (n3, PAGE_ALLDAYS, "All days"), (n4, PAGE_HOW, "How it works"))):
            with col_, st.container(key=f"hx_l{i_}"):
                st.page_link(page_, label=lab_)
        with n5, st.container(key="hx_navcta"):
            st.page_link(PAGE_SETUP, label="Start")

    # ---- hero -------------------------------------------------------------
    st.html(
        '<div class="hx-hero">'
        '<div class="hx-kicker">Atria University · Centre of Excellence</div>'
        '<div class="hx-h1">Know What Storage Is Really Worth to a Solar Plant</div>'
        '<div class="hx-sub">A settlement-accurate simulator for solar and battery plants '
        "under India's CERC DSM 2024 rules. Five dispatch strategies, every 15-minute "
        'block, on demo days or your own forecast data.</div></div>')
    _, b1, b2, _ = st.columns([1.5, 1.35, 1.35, 1.5])
    with b1:
        if st.button("Start a simulation", key="hx_start", use_container_width=True):
            st.switch_page(PAGE_SETUP)
    with b2:
        if st.button("See a demo result", key="hx_demo", use_container_width=True):
            st.switch_page(PAGE_RESULTS)

    # ---- card row (live numbers from the day being simulated) --------------
    with open(HERO_IMG_PATH, "rb") as f:
        img = base64.b64encode(f.read()).decode("ascii")
    base = r1.total("profit")
    gains = [(name.split(" ")[0], res.total("profit") - base) for name, res in results.items()]
    day_txt = (selected_engine_day if data_mode == MODE_ENGINE and engine_days_mw
               else "uploaded day" if real_forecast is not None else "demo day")
    st.html(
        '<div class="hx-cards">'
        '<div class="hx-card"><div class="hx-card-t">Forecast vs actual</div>'
        f'{_svg_img(_hx_profile_svg(list(forecast), list(actual)), "Forecast and actual generation through the day")}'
        '<div class="hx-card-c"><span style="color:#a996ff">━ forecast</span>&nbsp;&nbsp;'
        f'<span style="color:#5fe3c0">━ actual</span>&nbsp;&nbsp;· {day_txt}</div></div>'
        f'<div class="hx-photo" style="background-image:url(data:image/jpeg;base64,{img})">'
        f'<span>{_size_text(plant_capacity_mw)} solar + {_energy_text(cap)} battery</span></div>'
        '<div class="hx-card"><div class="hx-card-t">Gain over S1</div>'
        f'{_svg_img(_hx_gain_svg(gains), "Profit gain over S1 for each strategy")}'
        f'<div class="hx-card-c">Rs per day vs no battery · {day_txt}</div></div>'
        '</div>')

    # ---- how it works ------------------------------------------------------
    st.html(
        '<div class="hx-sec"><div class="hx-label">How it works</div>'
        '<div class="hx-h2">From a forecast file to a rupee answer in three steps.</div>'
        '<div class="hx-grid3">'
        '<div class="hx-step"><div class="n">01</div><h3>Set up</h3><p>Plant size, tariff and '
        "battery. Use a demo day or upload the forecasting engine's export, in kW or MW, "
        'UTC or IST.</p></div>'
        '<div class="hx-step"><div class="n">02</div><h3>Compare</h3><p>Five strategies settled '
        'side by side: profit, CERC penalties, battery wear, and a 15-minute chart of what '
        'the plant and battery did.</p></div>'
        '<div class="hx-step"><div class="n">03</div><h3>Backtest</h3><p>Every uploaded day at '
        'once, so you see the typical result and the spread, not one lucky or unlucky day.'
        '</p></div></div></div>')

    # ---- strategies ----------------------------------------------------------
    st.html(
        '<div class="hx-sec"><div class="hx-label">Five strategies</div>'
        '<div class="hx-h2">From no battery to perfect hindsight.</div>'
        '<div class="hx-grid5">'
        + "".join(
            f'<div class="hx-strat"><div class="tag">{t}</div><h4>{n}</h4><p>{d}</p></div>'
            for t, n, d in (
                ("S1", "PPA only", "No battery. The benchmark every other strategy is "
                                   "measured against."),
                ("S2", "Battery buffer", "All generation flows through the battery before "
                                         "the grid."),
                ("S3", "Time windows", "Charge at midday, discharge in the evening."),
                ("S4", "Real-time controller", "Acts block by block, only when the CERC "
                                               "penalty is worth more than the wear."),
                ("S5", "Optimizer", "Perfect hindsight. The ceiling no real controller "
                                    "can beat."),
            ))
        + '</div></div>')

    # ---- trust ----------------------------------------------------------------
    st.html(
        '<div class="hx-sec"><div class="hx-label">Why trust the numbers</div>'
        '<div class="hx-h2">The same engine, checked against real data.</div>'
        '<div class="hx-grid4">'
        '<div class="hx-stat"><div class="v">15 min</div><p>Settlement resolution, using '
        'the CERC DSM 2024 deviation bands.</p></div>'
        '<div class="hx-stat"><div class="v">68</div><p>Real days of our forecasting '
        "team's output, validated end to end.</p></div>"
        '<div class="hx-stat"><div class="v">60+</div><p>Automated tests, run on every '
        'change to the engine.</p></div>'
        '<div class="hx-stat"><div class="v">5</div><p>Dispatch strategies compared on '
        'every block, from no battery to perfect hindsight.</p></div>'
        '</div></div>'
        '<div class="hx-final"><div class="hx-h2" style="margin:0 auto 26px">'
        'Ready to run your plant?</div></div>')
    _, c1, _ = st.columns([2.3, 1.4, 2.3])
    with c1:
        if st.button("Start a simulation", key="hx_start2", use_container_width=True):
            st.switch_page(PAGE_SETUP)
    st.write("")


def _step(n: int, title: str, blurb: str):
    st.markdown(f'<div class="au-step"><span class="au-step-n">{n}</span>'
                f'<span class="au-step-t">{title}</span></div>', unsafe_allow_html=True)
    st.caption(blurb)


def page_setup():
    st.markdown("### Set up your simulation")
    st.markdown(
        "Four quick steps. Every input has a **?** next to it that explains what "
        "it means. The defaults already work, so you can go straight to the "
        "results and come back to change anything.")
    if st.button("Skip to results with the defaults", icon=":material/arrow_forward:",
                 key="skip_to_results"):
        st.switch_page(PAGE_RESULTS)
    st.write("")

    # ---- 1. Plant ---------------------------------------------------------
    with st.container(border=True):
        _step(1, "Your plant", "How big the solar plant is on paper.")
        st.number_input(
            "Contracted capacity (MW)", min_value=0.001, max_value=500.0,
            step=0.5, format="%.3f", key="plant_cap",
            help="The plant size written in the power purchase agreement (PPA). "
                 "CERC measures every deviation as a % of this, so it decides "
                 "which penalty band a shortfall falls into. O&M cost scales "
                 "with it too.")
        with st.expander("Advanced: real achievable capacity"):
            st.number_input(
                "Real achievable capacity (MW)", min_value=0.001, max_value=500.0,
                step=0.5, format="%.3f", key="real_cap",
                help="What the plant can physically produce (inverter limits, "
                     "panel losses). Caps generation on synthetic days only; "
                     "penalties are still measured against the contracted figure.")
            if ss["real_cap"] > ss["plant_cap"]:
                st.caption(":orange[Above the contracted capacity -- generation "
                           "will be capped at the contracted figure.]")
        if data_mode == MODE_ENGINE and ss["engine_scale"] == SCALE_OWN and engine_days_mw:
            st.caption(f":blue[Using the uploaded plant's own size "
                       f"({_size_text(plant_capacity_mw)}) -- set in step 2.]")

    # ---- 2. Generation data -----------------------------------------------
    with st.container(border=True):
        _step(2, "Generation data", "Where the forecast and the actual generation "
                                    "come from.")
        st.radio(
            "Generation data", [MODE_SYNTH, MODE_ENGINE, MODE_SINGLE],
            key="data_mode", label_visibility="collapsed",
            captions=["A made-up sunny day with random clouds. Good for "
                      "exploring how the tool behaves.",
                      "Our forecasting team's export, uploaded as it is: many "
                      "days, kW or MW, UTC or IST.",
                      "One day as two small files (time, mw), for anyone "
                      "without the engine's export."])

        if data_mode == MODE_SYNTH:
            st.slider(
                "Forecast error (% of capacity)", 3, 30, key="err",
                help="How far actual generation lands from the day-ahead "
                     "schedule on the synthetic day. This gap drives every "
                     "DSM penalty: bigger error, bigger penalties.")
            st.caption("Pick a different random day with **New weather day** on "
                       "the Results page.")

        elif data_mode == MODE_ENGINE:
            st.file_uploader(
                "Forecast-engine export (CSV)", type=["csv"], key="engine_upload",
                on_change=_on_engine_upload,
                help="One CSV with a timestamp column plus actual and forecast "
                     "generation columns at 15-minute resolution, any number of "
                     "days, e.g. date, power_generation_actual_kw, "
                     "forecast_generation_kw.")
            if ss.get("engine_raw") and ss.get("engine_upload") is None:
                st.caption(f"Using **{ss.get('engine_name')}** (uploaded earlier "
                           f"this session).")
            if ss.get("engine_raw"):
                c1, c2 = st.columns(2)
                with c1:
                    st.radio("Values are in", ["kW", "MW"], horizontal=True,
                             key="engine_unit", on_change=_reset_engine_capacity,
                             help="Read from the column names when they say "
                                  "(e.g. '_kw'). Check it if they don't.")
                with c2:
                    if _engine_has_offset(ss["engine_raw"]):
                        st.caption("Timestamps carry a UTC offset -- converted to "
                                   "India time (IST) automatically.")
                    else:
                        st.radio("Timestamps are in", ["UTC", "IST"], horizontal=True,
                                 key="engine_tz",
                                 help="How to read timestamps that carry no "
                                      "timezone. Solar should peak around 12:00 IST.")
                if engine_error:
                    st.error(f"Couldn't read that file: {engine_error}")
                elif eng is not None:
                    msg = f"{len(eng.days)} complete days loaded"
                    if eng.skipped:
                        msg += f"; {len(eng.skipped)} incomplete days skipped"
                    st.success(msg + ".")
                    if eng.skipped or eng.filled_blocks:
                        with st.expander("Data gaps"):
                            if eng.filled_blocks:
                                st.caption(
                                    f"{sum(eng.filled_blocks.values())} isolated missing "
                                    f"readings on {len(eng.filled_blocks)} days were "
                                    f"filled from their neighbours.")
                            for d, why in eng.skipped.items():
                                st.caption(f"{d}: skipped -- {why}")
                    unit = ss["engine_unit"]
                    peak_u = eng.peak_mw * (1000 if unit == "kW" else 1)
                    st.number_input(
                        f"This plant's capacity ({unit})", min_value=0.001,
                        step=0.5 if unit == "kW" else 0.1, format="%.3f",
                        key="engine_src_cap",
                        help=f"The real nameplate of the plant this data came "
                             f"from. The file peaks at {peak_u:.3f} {unit}; the "
                             f"default rounds that up. It decides how the profile "
                             f"is scaled, so use the real figure if you know it.")
                    st.radio(
                        "Run it as", [SCALE_SCALED, SCALE_OWN], key="engine_scale",
                        captions=[f"Expressed on the {ss['plant_cap']:g} MW plant from step 1.",
                                  "Runs the plant at the size entered above."],
                        help="Scaling keeps each day's real shape and forecast "
                             "error but expresses it on the contracted plant, so "
                             "rupee figures and battery sizes are comparable.")
                    st.caption("Pick which day to look at on the Results page.")

        else:  # MODE_SINGLE
            st.caption("CSV format: columns `time,mw` -- 96 rows (15-min blocks), "
                       "values in MW, times in IST.")
            c1, c2 = st.columns(2)
            with c1:
                st.file_uploader("Forecast CSV (day-ahead schedule)", type=["csv"],
                                 key="fc_upload",
                                 on_change=_on_single_upload, args=("fc",),
                                 help="What the plant promised the grid for each "
                                      "15-minute block.")
            with c2:
                st.file_uploader("Actual generation CSV (meter)", type=["csv"],
                                 key="act_upload",
                                 on_change=_on_single_upload, args=("act",),
                                 help="What the plant actually generated in each "
                                      "15-minute block.")
            for which, label in (("fc", "Forecast"), ("act", "Actual")):
                if ss.get(f"single_{which}_error"):
                    st.error(f"{label}: {ss[f'single_{which}_error']}")
                elif ss.get(f"single_{which}"):
                    st.caption(f"{label}: {ss.get(f'single_{which}_name')} "
                               f"({len(ss[f'single_{which}'])} blocks)")
            if real_forecast is None:
                st.caption("Upload both files -- the synthetic day is used until then.")

        for w in data_warnings:
            st.warning(w)

    # ---- 3. Money -----------------------------------------------------------
    with st.container(border=True):
        _step(3, "Price", "What the plant is paid.")
        st.slider(
            "PPA tariff (Rs/kWh)", 2.0, 4.5, step=0.05, key="ppa",
            help="The flat price paid for every scheduled kWh. CERC penalties are "
                 "multiples of this tariff, so it scales them too.")

    # ---- 4. Battery ---------------------------------------------------------
    with st.container(border=True):
        _step(4, "Battery", "The storage scenarios S2-S5 can use.")
        st.slider(
            "Battery size (hours of plant capacity)", 0.5, 8.0, step=0.5,
            key="bess_hours",
            help="Usable energy = hours x contracted capacity. Power is half the "
                 "energy per hour (0.5C). 4 h on a 10 MW plant = 40 MWh, 20 MW.")
        st.caption(f"= {_energy_text(cap)} usable, "
                   f"{_size_text(C_RATE_PER_MWH * cap)} power, 88% round-trip "
                   f"efficiency.")
        with st.expander(f"Advanced: battery wear cost (now Rs {deg:.2f} per kWh cycled)"):
            st.caption("Every time the battery cycles it uses up part of its life. "
                       "This turns the purchase price into a cost per kWh cycled: "
                       "price ÷ warranted cycles.")
            st.number_input("Battery price (Rs per kWh usable)", min_value=1.0,
                            step=500.0, format="%.0f", key="deg_price",
                            help="Default: IRENA 2024, $192/kWh at about Rs 85/$. "
                                 "Use a live quote for a final figure.")
            st.radio("I know the warranty in", ["Cycles", "Years"], horizontal=True,
                     key="deg_calc_mode")
            if ss["deg_calc_mode"] == "Cycles":
                st.number_input("Warranted cycles", min_value=1.0, step=100.0,
                                key="deg_cycles")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    st.number_input("Warranted years", min_value=0.1, step=0.5,
                                    key="deg_years")
                with c2:
                    st.number_input("Full cycles per day", min_value=0.01, step=0.1,
                                    key="deg_cpd",
                                    help="Warranted cycles = years x cycles/day x "
                                         "330 operating days.")
            st.caption(f"Total price Rs {capex / 1e7:,.2f} crore for "
                       f"{usable_kwh:,.0f} kWh -> **Rs {deg:.2f} per kWh cycled**")
            if deg != computed_deg:
                st.caption(f":orange[Your inputs give Rs {computed_deg:.2f}/kWh; the "
                           f"simulator limits this to Rs {DEG_MIN:.2f}-{DEG_MAX:.2f}.]")

    st.write("")
    if st.button("See the results", type="primary", icon=":material/arrow_forward:",
                 use_container_width=True, key="see_results"):
        st.switch_page(PAGE_RESULTS)



def page_results():
    st.markdown("### Results")
    prov = ((real_source_note or "real uploaded generation data") if using_real
            else "synthetic weather day")
    st.caption(f"{_size_text(plant_capacity_mw)} plant · flat-rate PPA Rs {ppa:.2f}/kWh · "
               f"CERC DSM 2024 settlement per 15-minute block · {prov}")
    top_l, top_r = st.columns([3, 2], vertical_alignment="bottom")
    with top_l:
        if engine_days_mw:
            st.selectbox(
                "Day (IST)", list(engine_days_mw), key="engine_day",
                help="Each day has its own weather and forecast error, so every "
                     "result changes with the day. The next page shows all days "
                     "at once.")
        elif not using_real:
            if st.button("New weather day", icon=":material/refresh:",
                         help="Draw a different random synthetic day with the same "
                              "settings."):
                ss["seed"] = int(np.random.default_rng().integers(1e9))
                st.rerun()
    with top_r:
        comparison_export = pd.DataFrame({
            "Scenario": list(results),
            "Profit/day (INR)": [r.total("profit") for r in results.values()],
            "DSM penalty (INR)": [r.total("dsm_penalty") for r in results.values()],
            "Degradation (INR)": [r.total("degradation") for r in results.values()],
        })
        st.download_button("Export this day (CSV)", comparison_export.to_csv(index=False),
                           file_name="agentic_grid_results.csv",
                           icon=":material/download:")
    for w in data_warnings:
        st.warning(w)
    if data_mode != MODE_SYNTH and not using_real:
        st.info("No real data loaded yet, so this shows the synthetic day. "
                "Upload your file in step 2 of Set up.")
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
            items = ["PPA revenue", "DSM net", "Degradation", "O&M"]
            # "+ 0.0" turns -0.0 into 0.0 so a zero cost never prints as "-0".
            amounts = [b.total("ppa_revenue"), dsm_net + 0.0,
                       -b.total("degradation") + 0.0, -b.total("om") + 0.0]
            inv = b.total("inventory_inr")
            if abs(inv) >= 0.5:
                items.append("Battery stock change")
                amounts.append(inv + 0.0)
            df = pd.DataFrame({"Line item": items, "Amount (Rs)": amounts})
            st.dataframe(df.style.format({"Amount (Rs)": "{:,.0f}"}),
                        hide_index=True, use_container_width=True,
                        height=35 * (len(items) + 1) + 3)
            if abs(inv) >= 0.5:
                st.caption("S4 starts the day with its battery half full. "
                           "'Battery stock change' charges for stored energy it "
                           "used up (at the PPA revenue needed to refill it) or "
                           "credits energy it ended with -- so it isn't rewarded "
                           "for spending yesterday's energy.")

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
                if name.startswith("S2") and s2_cfg.get("revise_schedule_for_rte", True):
                    st.caption(
                        "**Why S2's PPA revenue is lower:** every MWh in S2 passes "
                        "through the battery and loses its round-trip efficiency "
                        "(~12%), so S2 schedules forecast x RTE instead of the full "
                        "forecast (team decision D1). Less scheduled energy means "
                        "less PPA revenue, even on the same weather day."
                    )

            if live_attempt is not None:
                # S3-specific: the ONLY big bold number shown is always S3's
                # own genuine result (live_attempt). The real adopted outcome
                # is still stated honestly in the badge text above (prose),
                # just never repeated as a separate metric widget that could
                # visually read as a copy of another scenario's own number.
                st.metric("S3's own result (live)", INR(live_attempt),
                         help="This is this strategy's own number, always -- "
                              "moves instantly with any slider. Whether it's "
                              "the one actually used today is stated in the "
                              "badge above.")
            else:
                delta = f"{INR(r.total('profit') - compare_to)} vs S1" if compare_to is not None else None
                dcolor = ("normal" if compare_to is None or r.total("profit") >= compare_to
                         else "inverse")
                st.metric("Net profit / day", INR(r.total("profit")), delta,
                         delta_color=dcolor)

    p1 = r1.total("profit")
    baseline_keys = ["S1 - PPA only", "S2 - Battery buffer", "S3 - Time windows",
                     "S4 - Real-time Controller"]
    best = max(baseline_keys, key=lambda k: results[k].total("profit"))
    worst_msg = (" -- the battery cannot pay for itself on DSM avoidance alone "
                 "under a flat PPA. That gap is the case for the Scenario 5 optimizer."
                 if best == "S1 - PPA only" else "")
    st.info(f"**Best baseline today: {best}** at "
            f"{INR(results[best].total('profit'))}/day{worst_msg}")

    # At a glance: every scenario's profit for this day in one row, so the
    # comparison doesn't need five tab clicks.
    st.markdown('<div class="au-section">This day at a glance</div>',
                unsafe_allow_html=True)
    glance = st.columns(len(results))
    for col_, (nm_, r_) in zip(glance, results.items()):
        v_ = (r3_raw.total("profit") if nm_.startswith("S3") else r_.total("profit"))
        d_ = v_ - p1
        tie_ = abs(d_) < 0.5
        col_.metric(
            nm_.split(" - ")[0] + " · " + nm_.split(" - ")[1].replace("Real-time ", ""),
            INR(v_),
            None if nm_.startswith("S1") else
            ("same as S1" if tie_ else f"{'+' if d_ > 0 else '-'}{INR(abs(d_))} vs S1"),
            delta_color="off" if tie_ else "normal", delta_arrow="off" if tie_ else "auto",
            help={"S1": "No battery: sell what is generated. The benchmark.",
                  "S2": "All generation passes through the battery.",
                  "S3": "Fixed timetable: charge midday, discharge evening. Shown "
                        "as its own result even on days it's held back.",
                  "S4": "Reactive: uses the battery only where the CERC penalty "
                        "costs more than stored energy and wear.",
                  "S5": "Perfect hindsight: an upper bound, not deployable."}[nm_[:2]])
    st.write("")
    st.markdown(f"""
    <div class="gi-header"><span class="gi-dot"></span>
    <span class="gi-title">Grid Intelligence</span></div>
    """, unsafe_allow_html=True)

    def analyst(r1, r2, r3_raw, r4, deg, err, r5=None):
        # Every sentence below is derived from this run's numbers -- including
        # S4, so the narrative can never contradict the "Best baseline" banner.
        p1, p2, p3, p4 = (r1.total("profit"), r2.total("profit"),
                          r3_raw.total("profit"), r4.total("profit"))
        pen1, pen2 = r1.total("dsm_penalty"), r2.total("dsm_penalty")
        gross1 = max(r1.total("ppa_revenue"), 1.0)
        rte_loss_pct = (1 - fresh_battery({"batteryUsableCapacity": float(cap)}).rte) * 100
        gap_desc = (f"the real Actual-vs-Scheduled gap (averaging {real_gap_mean_pct:.1f}% "
                    f"of capacity in daylight, {real_gap_max_pct:.0f}% at worst)"
                    if using_real else f"a {err}% Actual-vs-Scheduled deviation")
        lines = [
            f"On this weather day, {gap_desc} cost the "
            f"plant Rs {pen1:,.0f} in DSM penalty under S1 -- "
            f"{100 * pen1 / gross1:.1f}% of gross PPA revenue. (This is the loss versus "
            f"delivering exactly to schedule; the larger 'DSM net' line in the cards "
            f"is the settlement cash flow, which also includes paying back the tariff "
            f"on energy that was scheduled but never generated.)"
        ]
        cut = pen1 - pen2
        cut_pct = 100 * cut / pen1 if pen1 > 1 else 0.0
        if p2 < p1:
            lines.append(
                f"S2 routes everything through the battery: it cut the DSM penalty by "
                f"{cut_pct:.0f}% (Rs {cut:,.0f}) but lost Rs {p1 - p2:,.0f}/day overall, "
                f"because battery wear (Rs {r2.total('degradation'):,.0f} at a "
                f"degradation cost of Rs {deg:.2f} per kWh cycled) plus round-trip "
                f"losses outweigh the penalty it avoids."
            )
        else:
            lines.append(
                f"Unusually, S2 beats S1 by Rs {p2 - p1:,.0f} today -- cheap degradation "
                f"(Rs {deg:.2f} per kWh cycled) and a large Actual-vs-Scheduled gap make "
                f"pure buffering pay."
            )
        if battery_deployed:
            lines.append(
                f"S3's fixed rule (charge at peak sun, discharge in the evening) earns "
                f"Rs {p3:,.0f}/day, beating S1 -- so S3 runs its battery today."
            )
        else:
            lines.append(
                f"S3's fixed rule (charge at peak sun, discharge in the evening) would "
                f"earn only Rs {p3:,.0f}/day, less than S1, so S3 leaves its battery "
                f"idle today. Each cycle also loses ~{rte_loss_pct:.0f}% of the energy "
                f"to round-trip efficiency, which often erases what timed shifting saves."
            )
        if p4 > p1:
            lines.append(
                f"S4's real-time controller earns Rs {p4:,.0f}/day -- Rs {p4 - p1:,.0f} "
                f"more than S1. It {s4_trigger_text}, where CERC's penalty costs more "
                f"than the stored energy and wear it spends -- targeted, reactive use "
                f"of the battery pays where blanket cycling does not."
            )
        else:
            lines.append(
                f"Even S4's real-time controller (Rs {p4:,.0f}/day) does not beat "
                f"S1 today."
            )
        if max(p2, p3, p4) < p1:
            lines.append(
                "Implication: with a single fixed price, no rule-based strategy beats "
                "doing nothing today -- the case for the Scenario 5 optimizer and P2P "
                "merchant revenue."
            )
        elif max(p2, p3, p4) == p4 and p4 > p1:
            lines.append(
                "Implication: a battery earns its keep under a flat PPA only when it is "
                "dispatched selectively against the CERC bands, not cycled by a fixed "
                "rule."
            )
        if r5 is not None:
            best_baseline = max(p1, p2, p3, p4)
            uplift = r5.total("profit") - best_baseline
            if uplift > 1:
                lines.append(
                    f"The Scenario 5 optimizer earns Rs {r5.total('profit'):,.0f}/day -- "
                    f"Rs {uplift:,.0f} more than the best baseline strategy today. That "
                    f"gap, roughly Rs {uplift*330/1e5:,.1f} lakh a year (330 operating "
                    f"days), is the most perfect foresight could add on top."
                )
        return lines

    def _typed(text, delay=0.018):
        for word in text.split(" "):
            yield word + " "
            time.sleep(delay)

    # Short and scannable: the first point always shows, the rest are one
    # click away.
    gi_lines = analyst(r1, r2, r3_raw, r4, deg, err, r5)
    with st.container(border=True):
        st.markdown(gi_lines[0])
        if len(gi_lines) > 1:
            st.markdown("\n".join(f"- {x}" for x in gi_lines[1:]))



    st.write("")
    with st.expander("What do S1-S5 mean?", icon=":material/help:"):
        st.markdown(SCENARIO_GUIDE)
    st.write("")
    st.markdown('<div class="au-section">Click a scenario to see its full '
               'breakdown and day profile</div>', unsafe_allow_html=True)
    scenario_tabs = st.tabs(list(results))
    for tab, (name, r) in zip(scenario_tabs, results.items()):
        with tab:
            if name == "S1 - PPA only":
                scenario_card(name, r)
            elif name == "S4 - Real-time Controller":
                scenario_card(name, r, compare_to=p1,
                              badge=f"Reactive, no foresight: S4 {s4_trigger_text}, "
                                    f"and {s4_charge_text}.")
            elif name == "S3 - Time windows":
                # Show S3's OWN line items and result, with a badge saying
                # whether it actually ran today -- otherwise a held-back S3
                # renders as an exact copy of S1's card.
                if battery_deployed:
                    s3_badge, s3_kind = ("Battery deployed today -- the fixed rule "
                                         "beat leaving it idle."), "good"
                else:
                    s3_badge, s3_kind = (
                        f"Held back today: the fixed rule alone would make "
                        f"{INR(r3_raw.total('profit'))}/day vs S1's {INR(p1)}, so "
                        f"S3 leaves the battery idle and its adopted result equals "
                        f"S1. The table and number below are what the rule itself "
                        f"would have done."), "warn"
                scenario_card(name, r, compare_to=p1, badge=s3_badge,
                              badge_kind=s3_kind, breakdown_r=r3_raw,
                              live_attempt=r3_raw.total("profit"))
            else:
                scenario_card(name, r, compare_to=p1)

            if name == "S5 - Optimizer":
                with st.expander("View Optimization Engine Architecture & Formulation"):
                    st.markdown(
                        "This engine is a deterministic Linear Program (LP), solved "
                        "with `PuLP`/`CBC` across all 96 fifteen-minute blocks jointly. "
                        "Every value below is read live from this run's actual inputs, "
                        "not a fixed example."
                    )
                    st.markdown("**Objective function** (matches `lp_dispatch.py` exactly):")
                    st.latex(r"\text{Maximize} \sum_{t=1}^{96} \Big["
                            r"\text{Revenue}_t + \text{Receivable}_t - \text{Payable}_t "
                            r"- C_{\text{deg}} \cdot \text{Throughput}_t \Big]")
                    st.caption(f"Live values this run: PPA tariff = Rs {ppa:.2f}/kWh, "
                              f"degradation cost = Rs {deg:.2f}/kWh cycled.")

                    st.markdown("**CERC DSM 2024 piecewise penalty** -- the real "
                              "tiered structure from `dsm_bands.yaml`, not a "
                              "zero-penalty dead-band (there is no free zone; "
                              "Band 1 is charged at 100% of tariff):")
                    edges = dsm_cfg.bands.edges_pct
                    over_rates = dsm_cfg.bands.over_rates
                    under_rates = dsm_cfg.bands.under_rates
                    band_rows = []
                    prev_edge = 0.0
                    for i, edge in enumerate(edges):
                        band_rows.append(f"{prev_edge:.0f}\\%\\text{{--}}{edge:.0f}\\%: "
                                        f"{under_rates[i]:.2f}\\times\\lambda_{{PPA}} "
                                        f"\\text{{ (under)}}, "
                                        f"{over_rates[i]:.2f}\\times\\lambda_{{PPA}} "
                                        f"\\text{{ (over)}}")
                        prev_edge = edge
                    band_rows.append(f">{edges[-1]:.0f}\\%: "
                                    f"{under_rates[-1]:.2f}\\times\\lambda_{{PPA}} "
                                    f"\\text{{ (under)}}, "
                                    f"{over_rates[-1]:.2f}\\times\\lambda_{{PPA}} "
                                    f"\\text{{ (over)}}")
                    st.latex(r"\text{Penalty}(D_t) = \begin{cases}" +
                            r" \\ ".join(band_rows) + r"\end{cases}")

                    st.markdown("**Battery SoC transition** (matches "
                              "`core/battery.py` exactly):")
                    st.latex(r"SoC_t = SoC_{t-1} + \left(\eta \cdot P_{ch,t} - "
                            r"\frac{P_{dis,t}}{\eta}\right) \cdot \Delta t "
                            r"\quad,\quad \eta = \sqrt{RTE}")
                    st.caption(f"Live values this run: RTE = 88% (fixed constant "
                              f"used throughout the app, not a slider), "
                              f"so eta (each leg) = {0.88**0.5:.3f}.")

                    st.markdown("**Physical bounds:**")
                    st.latex(r"SoC_{\min} \le SoC_t \le SoC_{\max} \quad,\quad "
                            r"0 \le P_{ch,t}, P_{dis,t} \le P_{\text{c-rate}}")

                    st.info(
                        "**Why the lines converge under a flat PPA:** since "
                        "$\\lambda_{PPA}$ doesn't vary by hour, there's no revenue "
                        "gain from shifting energy in time -- only downside from "
                        "$C_{deg}$. The mathematically optimal move is to set the "
                        "schedule to match generation exactly, driving the penalty "
                        "term to zero, and leave the battery idle since cycling it "
                        "would only add cost with no offsetting gain. This is a "
                        "genuine property of the optimal solution under these "
                        "conditions, not a rule written into the code."
                    )

            st.write("")
            # chart_r: S3's chart must always show its OWN battery physics --
            # even when held back and r==r1 for the adopted numbers, the
            # strategy still genuinely charges/discharges when run, and that
            # real SoC curve is what the chart should show, not a flat line.
            chart_r = r3_raw if name == "S3 - Time windows" else r
            sched_mw = [row.scheduled_mwh * 4 for row in chart_r.rows]
            deliv_mw = [row.delivered_mwh * 4 for row in chart_r.rows]
            actual_mw_list = [row.actual_gen_mwh * 4 for row in chart_r.rows]

            # S1 has no battery, so it gets the power panel only -- an empty
            # battery panel under it just looked broken.
            _batt_panel = any(row.soc_mwh is not None for row in chart_r.rows)
            fig = make_subplots(
                rows=2 if _batt_panel else 1, cols=1, shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.65, 0.35] if _batt_panel else None
            )

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
                               showlegend=False, hoverinfo="skip", row=1, col=1)
                fig.add_scatter(x=hours, y=prev_upper, mode="lines", line=dict(width=0),
                               fill="tonexty", fillcolor=color, showlegend=False,
                               hoverinfo="skip", row=1, col=1)
                fig.add_scatter(x=hours, y=lower, mode="lines", line=dict(width=0),
                               showlegend=False, hoverinfo="skip", row=1, col=1)
                fig.add_scatter(x=hours, y=prev_lower, mode="lines", line=dict(width=0),
                               fill="tonexty", fillcolor=color, showlegend=False,
                               hoverinfo="skip", name=blabel, row=1, col=1)
                prev_upper, prev_lower = upper, lower

            fig.add_scatter(x=hours, y=actual_mw_list,
                            name="Actual gen (MW)",
                            line=dict(color="rgba(245,158,11,.75)", width=4),
                            row=1, col=1)
            fig.add_scatter(x=hours, y=sched_mw, name="Schedule (MW)",
                            line=dict(color=INK, width=2.0, dash="dash"),
                            row=1, col=1)
            fig.add_scatter(x=hours, y=deliv_mw, name="Delivered (MW)",
                            line=dict(color=BLUE, width=1.75),
                            fill="tonexty", fillcolor="rgba(61,52,224,.08)",
                            row=1, col=1)

            # For S5 specifically: show the ORIGINAL, pre-optimization
            # forecast as a muted phantom reference line. S5 rewrites its
            # own schedule to match actual generation, so without this,
            # the baseline promise a naive forecast would have made is
            # invisible -- this makes the "how much did it correct for"
            # story visible, not just a number to read separately.
            if name == "S5 - Optimizer":
                fig.add_scatter(x=hours, y=list(forecast),
                               name="Original Forecast Schedule (MW)",
                               line=dict(color="rgba(150,150,150,.65)",
                                       dash="dot", width=1.5), row=1, col=1)

            # "Optimal vs Base" overlay -- superimpose S5's own optimal
            # schedule directly on top of this baseline's chart, so the
            # uplift is visible, not just a number to read separately.
            if r5 is not None and name != "S5 - Optimizer":
                r5_sched_mw = [row.scheduled_mwh * 4 for row in r5.rows]
                fig.add_scatter(x=hours, y=r5_sched_mw, name="S5 Optimal Schedule",
                               line=dict(color="#9333EA", width=2.2, dash="dot"),
                               row=1, col=1)

            # find and mark the single worst deviation block, with its real
            # rupee penalty, so the chart states the story directly instead
            # of leaving it to be inferred -- this belongs on the power
            # panel (Row 1), since it's about delivery vs schedule, not
            # battery state.
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
                    hoverinfo="text", row=1, col=1)
                fig.add_annotation(
                    x=hours[worst_idx], y=deliv_mw[worst_idx],
                    text=f"<b>Largest gap:</b> {worst_row.deviation_mwh:+.2f} MWh<br>"
                         f"-> Rs {worst_row.dsm_penalty:,.0f} penalty",
                    showarrow=True, arrowhead=2, arrowsize=0.8, arrowwidth=1.3,
                    arrowcolor="#9CA3AF",
                    ax=0, ay=-42, bgcolor="#F9FAFB", bordercolor="#E5E7EB",
                    borderwidth=1, borderpad=6, font=dict(size=11, color="#374151"),
                    row=1, col=1)
            zero_deviation_today = abs(worst_row.deviation_mwh) <= 0.001

            # ----------------- ROW 2: battery physics, own lane -----------
            soc = [row.soc_mwh for row in chart_r.rows]
            has_battery = any(s is not None for s in soc)
            if has_battery:
                # Each stored SoC is the END of its 15-min block, so plot it at
                # block end and prepend the opening level at 00:00 (S4 opens
                # at 50%, the others at the 10% floor). Plotting it at block
                # start drew every change 15 minutes early.
                opening_pct = 50.0 if name.startswith("S4") else 10.0
                soc_x = [0.0] + [h + DT for h in hours]
                soc_pct = [opening_pct] + [100 * s / cap if s is not None else None
                                           for s in soc]
                fig.add_scatter(x=soc_x, y=soc_pct, name="Battery charge (%)",
                                line=dict(color=GREEN, width=2.6),
                                fill="tozeroy", fillcolor="rgba(63,174,73,.08)",
                                row=2, col=1)

                # Shade what the battery is actually doing in each block, from
                # that block's own change in stored energy (start -> end):
                # rising = charging, falling = discharging. (The old version
                # shaded a trailing 1-hour average, which lagged reality by up
                # to 45 minutes -- e.g. "DISCHARGING" shown after the battery
                # was already empty. Grid-side flow isn't used because
                # round-trip losses make it negative even when the stored
                # level is flat, as in S2's pass-through.)
                MIN_STEP_MWH = 0.005  # ignore numerically-zero changes
                opening_mwh = cap * opening_pct / 100
                soc_mwh = [opening_mwh] + [x if x is not None else opening_mwh for x in soc]
                state = []
                for i in range(len(soc)):
                    step = soc_mwh[i + 1] - soc_mwh[i]
                    state.append(-1 if step > MIN_STEP_MWH else
                                 1 if step < -MIN_STEP_MWH else 0)

                def _regions(kind):
                    """Contiguous runs of `kind`, bridging gaps of <= 2 idle
                    blocks so one buffering episode reads as one band."""
                    out, start, last = [], None, None
                    for i, st_ in enumerate(state):
                        if st_ == kind:
                            if start is None:
                                start = i
                            elif i - last > 3 or any(state[last + 1:i]):
                                # too long a gap, or the battery did the
                                # opposite in between -- start a new band
                                out.append((start, last)); start = i
                            last = i
                    if start is not None:
                        out.append((start, last))
                    return out

                LABEL_MIN_H = 2.0  # label only bands >= 2 h wide (~75 px) -> no overlaps
                for kind, color, label in [(-1, "#2563EB", "CHARGING"),
                                           (1, "#F97316", "DISCHARGING")]:
                    for a_, b_ in _regions(kind):
                        x0, x1 = hours[a_], hours[b_] + DT
                        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.12,
                                     line_width=0, layer="below", row=2, col=1)
                        if x1 - x0 >= LABEL_MIN_H:
                            fig.add_annotation(x=(x0 + x1) / 2, y=103,
                                              text=f"<b>{label}</b>", showarrow=False,
                                              font=dict(size=10, color=color),
                                              opacity=0.8, yanchor="top",
                                              row=2, col=1)
                    # legend key, so narrow unlabeled bands are still readable
                    if any(sv == kind for sv in state):
                        fig.add_scatter(x=[None], y=[None], mode="markers",
                                        marker=dict(symbol="square", size=11,
                                                    color=color, opacity=0.35),
                                        name=label.capitalize(), row=2, col=1)

            fig.update_layout(height=600 if _batt_panel else 420,
                              margin=dict(l=10, r=10, t=10, b=10),
                              legend=dict(orientation="h", y=1.08),
                              hovermode="x unified", template="plotly_white")
            fig.update_xaxes(range=[0, 24], row=1, col=1,
                             title_text=None if _batt_panel else "Hour of day")
            fig.update_yaxes(title_text="MW", row=1, col=1)
            if _batt_panel:
                fig.update_xaxes(range=[0, 24], row=2, col=1, title_text="Hour of day")
                fig.update_yaxes(title_text="Battery charge (%)", range=[0, 110],
                                 row=2, col=1)

            if zero_deviation_today and name == "S5 - Optimizer":
                weather_noise_mwh = sum(abs(a - f) for a, f in zip(actual, forecast))
                soc_now_pct = 100 * chart_r.rows[0].soc_mwh / cap if chart_r.rows[0].soc_mwh is not None else 0
                penalty_saved = r1.total("dsm_penalty") - r.total("dsm_penalty")
                wear_cost = r.total("degradation")
                uplift_here = r.total("profit") - results[best].total("profit")

                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("DSM Penalty Exposure", INR(r.total("dsm_penalty")),
                            f"Saved {INR(penalty_saved)} vs S1" if penalty_saved > 0 else None)
                mcol2.metric("Battery Wear Cost", INR(wear_cost),
                            f"SoC held at {soc_now_pct:.0f}%")
                mcol3.metric("Annualized Value vs Best Baseline",
                            INR(uplift_here * 330) if uplift_here > 0 else INR(0))

                st.info(
                    f"**OPTIMIZER DISPATCH STRATEGY: PERFECT FORESIGHT ALIGNMENT**\n\n"
                    f"**Status:** Optimal convergence achieved (0.00% deviation exposure)\n\n"
                    f"**Dynamic schedule morphing:** the engine saw {weather_noise_mwh:.1f} "
                    f"MWh of intraday weather noise between forecast and actual, and "
                    f"re-declared the schedule to match physical generation exactly, "
                    f"eliminating DSM exposure entirely.\n\n"
                    f"**Asset protection:** because the PPA rate is flat (Rs {ppa:.2f}/kWh) "
                    f"and deviation penalties are already eliminated by scheduling "
                    f"accuracy alone, the battery is held at {soc_now_pct:.0f}% SoC -- "
                    f"real wear cost incurred: {INR(wear_cost)}. This holds at any "
                    f"Generation Deviation slider setting, since the optimizer always "
                    f"sees the real outcome before scheduling."
                )
            elif zero_deviation_today:
                st.success(
                    "**Zero deviation achieved.** Schedule exactly matches "
                    "generation: no DSM exposure, and no reason to use the "
                    "battery under a flat PPA. This holds at any Generation "
                    "Deviation slider setting, since the optimizer sees the "
                    "real outcome before scheduling and its own deviation is "
                    "zero by design. The slider still raises the baseline "
                    "scenarios' exposure, and therefore the uplift number, "
                    "just never this chart."
                )

            st.plotly_chart(fig, use_container_width=True,
                           key=f"dayprofile_{name}")
            # caption built from what this chart actually contains
            cap_parts = ["Yellow, orange, and red bands around Schedule show the "
                         "real CERC tiers (5%, 10%, and 20% of Available Capacity). "
                         "Band 1 is charged at 100% of tariff, not zero-penalty."]
            if r5 is not None and name != "S5 - Optimizer":
                cap_parts.append("The purple dotted line is the optimizer's own "
                                 "schedule, shown for direct comparison.")
            if has_battery:
                cap_parts.append(
                    "Blue and orange bands in the lower panel mark the blocks where "
                    "the battery is actually charging or discharging"
                    + (" -- this strategy's own physics, whether or not it's "
                       "adopted today (see the badge above)." if name == "S3 - Time windows"
                       else "."))
            if not zero_deviation_today:
                cap_parts.append("The marked point is the single worst block of the "
                                 "day, with its real penalty.")
            st.caption(" ".join(cap_parts))

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
    with st.expander("Show the comparison chart and table", expanded=False):
        st.caption("Each bar is one scenario's own gross PPA revenue, normalized "
                  "to 100% -- showing what SHARE is kept as profit versus lost to "
                  "DSM, degradation, and O&M. For exact rupee amounts, see the "
                  "table below; this answers a different question: which "
                  "scenario keeps the biggest slice of its own revenue.")
        profit_pct, dsm_pct, deg_pct, om_pct, inv_pct = [], [], [], [], []
        for r in results.values():
            gross = max(r.total("ppa_revenue"), 1.0)
            dsm_net = r.total("dsm_receivable") - r.total("dsm_payable")
            profit_pct.append(r.total("profit") / gross * 100)
            dsm_pct.append(-dsm_net / gross * 100)
            deg_pct.append(r.total("degradation") / gross * 100)
            om_pct.append(r.total("om") / gross * 100)
            inv_pct.append(-r.total("inventory_inr") / gross * 100)

        fig = go.Figure()
        fig.add_bar(name="Net profit kept", x=names, y=profit_pct, marker_color=GREEN)
        fig.add_bar(name="DSM cost", x=names, y=dsm_pct, marker_color="#DC2626")
        fig.add_bar(name="Degradation cost", x=names, y=deg_pct, marker_color="#F59E0B")
        fig.add_bar(name="O&M cost", x=names, y=om_pct, marker_color=GREY)
        if any(abs(v) > 1e-9 for v in inv_pct):
            fig.add_bar(name="Battery stock used", x=names, y=inv_pct,
                        marker_color="#7C3AED")
        # "relative" stacks positives up and negatives down from zero, so a
        # loss-making scenario reads as costs above the axis and the loss
        # below it, instead of segments overlapping across zero.
        fig.update_layout(barmode="relative", height=320,
                          margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h", y=1.12),
                          yaxis_title="% of gross PPA revenue")
        fig.add_hline(y=0, line_color=INK, line_width=1)
        st.plotly_chart(fig, use_container_width=True, key="pl_composition")
        if any(p < 0 for p in profit_pct):
            st.caption("A green segment below zero is a loss: that scenario's costs "
                       "exceeded its own PPA revenue, so its cost segments add up "
                       "to more than 100%.")

        df = pd.DataFrame({
            "Scenario": names,
            "PPA revenue": [r.total("ppa_revenue") for r in results.values()],
            "DSM net": [r.total("dsm_receivable") - r.total("dsm_payable")
                        for r in results.values()],
            # Costs shown negative, matching the scenario cards above.
            "Degradation": [-r.total("degradation") + 0.0 for r in results.values()],
            "O&M": [-r.total("om") + 0.0 for r in results.values()],
            "Battery stock": [r.total("inventory_inr") + 0.0 for r in results.values()],
            "Profit/day": [r.total("profit") for r in results.values()],
            "Profit/year (330d)": [r.total("profit") * 330 for r in results.values()],
        })
        st.dataframe(df.style.format({c: "Rs {:,.0f}" for c in df.columns[1:]}),
                    use_container_width=True, hide_index=True)

    st.write("")
    if st.button("Next: all days and the multi-day backtest", type="primary",
                 icon=":material/arrow_forward:", use_container_width=True,
                 key="to_alldays"):
        st.switch_page(PAGE_ALLDAYS)

def page_alldays():
    st.markdown("### All days and backtest")
    st.caption("One day is a snapshot. These views run the scenarios over many days "
               "at your current settings, so you can see the typical result and the "
               "spread, not just one number.")
    if not engine_days_mw:
        st.info("Upload a forecast-engine file in step 2 of Set up to see every real "
                "day at once here. The backtest below also works with synthetic days.")

    # ---- All uploaded days at a glance (forecast-engine file only) --------
    # Every number in the scenario cards is for ONE day; picking another day
    # changes them because that day's weather and forecast error differ.
    # This panel runs S1-S4 over every complete day at the current settings
    # (~0.1 s for 68 days), so the whole period is visible without clicking
    # through days one by one.
    if engine_days_mw:
        ov_rows = []
        for d_, (fc_, ac_) in engine_days_mw.items():
            o1 = run_s1(fc_, ac_, plant, dsm_cfg, s1_cfg)
            o2 = run_s2(fc_, ac_, plant, dsm_cfg, s2_cfg,
                        fresh_battery({"batteryUsableCapacity": float(cap)}))
            o3 = run_s3(fc_, ac_, plant, dsm_cfg, s3_cfg,
                        fresh_battery({"batteryUsableCapacity": float(cap)}))
            o4 = run_s4(fc_, ac_, plant, dsm_cfg, s4_cfg,
                        fresh_battery({"batteryUsableCapacity": float(cap)}))
            p1_ = o1.total("profit")
            gap_ = [abs(a_ - f_) / plant["plant_mw"] * 100 for a_, f_ in zip(ac_, fc_)]
            lit_ = [g for g, f_ in zip(gap_, fc_) if f_ > 0.01 * plant["plant_mw"]]
            ov_rows.append({
                "Day": d_,
                "Generation (MWh)": sum(ac_) * DT,
                "Forecast error (%)": float(np.mean(lit_)) if lit_ else 0.0,
                "Worst block (%)": max(gap_),
                "S1 profit (Rs)": p1_,
                "S1 DSM penalty (Rs)": o1.total("dsm_penalty"),
                "S2 vs S1 (Rs)": o2.total("profit") - p1_,
                "S3 vs S1 (Rs)": max(o3.total("profit"), p1_) - p1_,
                "S4 vs S1 (Rs)": o4.total("profit") - p1_,
                "_ppa": o1.total("ppa_revenue"),
            })
        ov = pd.DataFrame(ov_rows)
        n_ov = len(ov)
        st.write("")
        st.markdown(f'<div class="au-section">All {n_ov} uploaded days at a glance</div>',
                    unsafe_allow_html=True)
        with st.container(border=True):
            st.caption(
                "The Results page shows one day. Each day has different weather and a "
                "different forecast error, so its penalties -- and every scenario's "
                "result -- differ. This runs S1-S4 on every day at your current "
                "settings. (S5, the perfect-foresight optimizer, is in the backtest "
                "further down.)")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Avg forecast error", f"{ov['Forecast error (%)'].mean():.1f}%",
                      help="Mean |actual - forecast| in daylight blocks, as % of "
                           "contracted capacity, averaged over the days.")
            k2.metric("S1 penalty, % of revenue",
                      f"{100 * ov['S1 DSM penalty (Rs)'].sum() / max(ov['_ppa'].sum(), 1):.1f}%",
                      INR(ov['S1 DSM penalty (Rs)'].mean()) + "/day",
                      delta_color="off", delta_arrow="off",
                      help="CERC deviation penalties under S1 (no battery), as a "
                           "share of PPA revenue, over all uploaded days.")
            g4 = ov["S4 vs S1 (Rs)"]
            k3.metric("S4 vs S1", f"{'+' if g4.mean() >= 0 else '-'}{INR(abs(g4.mean()))}/day",
                      f"ahead {int((g4 > 1).sum())} · level {int((g4.abs() <= 1).sum())} · "
                      f"behind {int((g4 < -1).sum())} days", delta_color="off")
            k4.metric("Avg S1 profit", f"{INR(ov['S1 profit (Rs)'].mean())}/day",
                      f"{INR(ov['S1 profit (Rs)'].min())} - {INR(ov['S1 profit (Rs)'].max())}",
                      delta_color="off")

            fig_ov = make_subplots(specs=[[{"secondary_y": True}]])
            fig_ov.add_bar(x=ov["Day"], y=ov["S1 DSM penalty (Rs)"],
                           name="S1 DSM penalty (Rs)", marker_color="#FCA5A5",
                           customdata=ov[["Forecast error (%)", "Worst block (%)"]],
                           hovertemplate="%{x}<br>Penalty Rs %{y:,.0f}<br>Forecast error "
                                         "%{customdata[0]:.1f}% (worst block "
                                         "%{customdata[1]:.0f}%)<extra></extra>")
            fig_ov.add_scatter(x=ov["Day"], y=g4, name="S4 gain vs S1 (Rs)",
                               mode="lines+markers", line=dict(color="#059669", width=2),
                               hovertemplate="%{x}<br>S4 vs S1: Rs %{y:,.0f}<extra></extra>")
            fig_ov.add_scatter(x=ov["Day"], y=ov["Forecast error (%)"],
                               name="Forecast error (%)", mode="lines",
                               line=dict(color=GREY, width=1.2, dash="dot"),
                               hovertemplate="%{x}<br>Error %{y:.1f}%<extra></extra>",
                               secondary_y=True)
            if selected_engine_day in set(ov["Day"]):
                fig_ov.add_vline(x=selected_engine_day, line_color=BLUE, line_width=2,
                                 annotation_text="selected day",
                                 annotation_font=dict(size=10, color=BLUE))
            fig_ov.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                                 legend=dict(orientation="h", y=1.15),
                                 hovermode="x unified", template="plotly_white",
                                 bargap=0.25)
            # Days as evenly spaced categories: the file has multi-week gaps,
            # and a date axis would draw lines straight across them.
            fig_ov.update_xaxes(type="category", nticks=12, tickangle=-45)
            fig_ov.update_yaxes(title_text="Rs / day", secondary_y=False)
            fig_ov.update_yaxes(title_text="Forecast error", secondary_y=True,
                                showgrid=False, ticksuffix="%", tickformat=".0f",
                                tickmode="linear", tick0=0, dtick=2,
                                range=[0, max(1.0, float(ov["Forecast error (%)"].max())) * 1.15])
            st.plotly_chart(fig_ov, use_container_width=True, key="all_days_chart")

            with st.expander(f"Day-by-day table ({n_ov} days) -- pick a day on the "
                             f"Results page to see its full breakdown"):
                show = ov.drop(columns=["_ppa"])

                def _hl(row):
                    on = row["Day"] == selected_engine_day
                    return ["background-color: rgba(61,52,224,.10)" if on else ""] * len(row)

                st.dataframe(
                    show.style.apply(_hl, axis=1).format({
                        "Generation (MWh)": "{:.1f}", "Forecast error (%)": "{:.1f}",
                        "Worst block (%)": "{:.0f}", "S1 profit (Rs)": "{:,.0f}",
                        "S1 DSM penalty (Rs)": "{:,.0f}", "S2 vs S1 (Rs)": "{:+,.0f}",
                        "S3 vs S1 (Rs)": "{:+,.0f}", "S4 vs S1 (Rs)": "{:+,.0f}"}),
                    hide_index=True, use_container_width=True, height=320)
                st.download_button("Download this table (CSV)",
                                   show.to_csv(index=False).encode(),
                                   file_name="all_days_results.csv",
                                   key="all_days_csv")

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
        # With a forecast-engine file loaded, backtest its REAL days; otherwise
        # synthetic weather days (said plainly, so a single uploaded day isn't
        # mistaken for the source of these results).
        if engine_days_mw:
            real_days = list(engine_days_mw)
            max_days = len(real_days)
            n_days = (st.slider("Number of real days to backtest", 1, max_days,
                                max_days, 1) if max_days > 1 else 1)
            bt_source = f"real:{len(real_days)}:{plant_capacity_mw}"
            st.caption(f"Runs on the uploaded forecast-engine file: the first "
                       f"{n_days} of its {max_days} complete days.")
        else:
            n_days = st.slider("Number of days to backtest", 5, 60, 20, 5)
            bt_source = "synthetic"
            if using_real:
                st.caption(":orange[A single uploaded day can't be backtested, so this "
                           "runs synthetic weather days at a 12% deviation. Use the "
                           "'Forecast-engine file' option to backtest real days.]")
        run_backtest = st.button("Run backtest")

        if run_backtest:
            if bt_source == "synthetic":
                rng_master = np.random.default_rng(2026)
                seeds = rng_master.integers(1, int(1e9), size=n_days)
                bt_days = [(i + 1, int(sd), None) for i, sd in enumerate(seeds)]
            else:
                bt_days = [(d, None, engine_days_mw[d]) for d in real_days[:n_days]]
            rows = []
            progress = st.progress(0.0, text="Running backtest...")
            for i, (bt_label, seed, real_pair) in enumerate(bt_days):
                if real_pair is None:
                    bt_fc, bt_act, _ = make_day(seed, err, plant["plant_mw"])
                else:
                    bt_fc, bt_act = real_pair
                bt_r1 = run_s1(bt_fc, bt_act, plant, dsm_cfg, s1_cfg)
                bt_r2 = run_s2(bt_fc, bt_act, plant, dsm_cfg, s2_cfg,
                              fresh_battery({"batteryUsableCapacity": float(cap)}))
                bt_r3_raw = run_s3(bt_fc, bt_act, plant, dsm_cfg, s3_cfg,
                                  fresh_battery({"batteryUsableCapacity": float(cap)}))
                bt_deployed = bt_r3_raw.total("profit") > bt_r1.total("profit")
                bt_r3 = bt_r3_raw if bt_deployed else bt_r1
                bt_r4 = run_s4(bt_fc, bt_act, plant, dsm_cfg, s4_cfg,
                               fresh_battery({"batteryUsableCapacity": float(cap)}))
                row = {"day": bt_label, "seed": seed,
                      "S1": bt_r1.total("profit"), "S2": bt_r2.total("profit"),
                      "S3": bt_r3.total("profit"), "S4": bt_r4.total("profit"),
                      "S3_battery_used": bt_deployed}
                try:
                    bt_r5 = solve_optimal_dispatch(
                        bt_fc, bt_act, plant, dsm_cfg,
                        OptimizerBatterySpec(float(cap), C_RATE_PER_MWH * float(cap), 0.88, 10, 90), deg)
                    row["S5"] = bt_r5.total("profit")
                except Exception as e:  # noqa: BLE001
                    row["S5"] = float("nan")
                    st.session_state["backtest_s5_error"] = str(e)
                rows.append(row)
                progress.progress((i + 1) / n_days, text=f"Day {i+1}/{n_days}")
            progress.empty()

            bt_df = pd.DataFrame(rows)
            st.session_state["backtest_df"] = bt_df
            st.session_state["backtest_source"] = bt_source

        # Only show results that came from the data currently selected.
        if ("backtest_df" in st.session_state
                and st.session_state.get("backtest_source") == bt_source):
            bt_df = st.session_state["backtest_df"]
            scen_cols = [c for c in ["S1", "S2", "S3", "S4", "S5"]
                        if c in bt_df.columns and bt_df[c].notna().any()]
            if "S5" in bt_df.columns and "S5" not in scen_cols:
                st.warning("S5 optimizer did not solve during this backtest, so it is "
                           "left out of the table: "
                           f"{st.session_state.get('backtest_s5_error', 'unknown error')}")

            # Win rate: a day's winners are every scenario within Rs 1 of that
            # day's best profit. When S3 is held back it is a copy of S1, so it
            # can't "win" a day it didn't actually play -- otherwise S1 and S3
            # tie on every such day and idxmax() hands every tie to S1.
            best_per_day = bt_df[scen_cols].max(axis=1)
            def _wins(c):
                won = bt_df[c] >= best_per_day - 1.0
                if c == "S3":
                    won &= bt_df["S3_battery_used"]
                return won.mean() * 100

            summary = pd.DataFrame({
                "Scenario": scen_cols,
                "Mean profit/day": [bt_df[c].mean() for c in scen_cols],
                "Min": [bt_df[c].min() for c in scen_cols],
                "Max": [bt_df[c].max() for c in scen_cols],
                "Std dev": [bt_df[c].std() for c in scen_cols],
                "Win rate (best day)": [f"{_wins(c):.0f}%" for c in scen_cols],
            })
            st.dataframe(
                summary.style.format({c: "Rs {:,.0f}" for c in
                                     ["Mean profit/day", "Min", "Max", "Std dev"]}),
                hide_index=True, use_container_width=True)
            st.caption("Win rate counts the days a scenario matched the best profit "
                      "(ties are credited to each tied scenario, so rates can sum "
                      "past 100%). S3 only counts days it actually ran its battery.")

            fig_bt = go.Figure()
            colors_bt = {"S1": GREY, "S2": "#DC2626", "S3": "#F59E0B",
                         "S4": "#059669", "S5": BLUE}
            for c in scen_cols:
                # S3 held back == S1, so its line would hide S1's; dash it.
                dash = "dash" if c == "S3" else None
                fig_bt.add_scatter(x=bt_df["day"], y=bt_df[c], name=c, mode="lines+markers",
                                  line=dict(color=colors_bt.get(c, INK), width=1.6,
                                            dash=dash))
            fig_bt.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                                legend=dict(orientation="h", y=1.12),
                                xaxis_title=("Date (IST)" if bt_source != "synthetic"
                                             else "Backtest day"),
                                yaxis_title="Profit (Rs)")
            st.plotly_chart(fig_bt, use_container_width=True, key="backtest_chart")

            if "S3" in bt_df.columns:
                s3_rate = bt_df["S3_battery_used"].mean() * 100
                if s3_rate == 0:
                    s3_note = ("its fixed rule never beat leaving the battery idle, so "
                               "S3 reports the same numbers as S1 on every day.")
                elif s3_rate == 100:
                    s3_note = ("its fixed rule beat leaving the battery idle on every "
                               "day at these settings.")
                else:
                    s3_note = ("so the deploy-or-hold decision does respond to "
                               "day-to-day weather, not just one fixed day.")
                st.caption(f"S3 deployed its battery on {s3_rate:.0f}% of backtested "
                          f"days at current settings -- {s3_note}")

    names = list(results)
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
            "penalty": row.dsm_penalty, "battery_stock": row.inventory_inr,
            "profit": row.profit,
        } for row in r.rows])
        st.dataframe(bl, use_container_width=True, height=300, hide_index=True)
        st.download_button("Export CSV", bl.to_csv(index=False),
                           file_name=f"{pick[:2]}_blocks.csv")


def page_how():
    st.markdown("### How it works")
    st.caption("What the simulator models, the terms it uses, and where the code lives.")
    st.markdown('<div class="au-section">Key terms</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(GLOSSARY)
    st.markdown('<div class="au-section">The five scenarios</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(SCENARIO_GUIDE)
    st.write("")

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


    st.markdown('<div class="au-section">Documentation</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            "**Repository:** "
            "[github.com/aishwaryamishra444/dispatch-sim]"
            "(https://github.com/aishwaryamishra444/dispatch-sim)\n\n"
            "- `dispatch_sim/core/` -- settlement engine, battery model, ledger\n"
            "- `dispatch_sim/config/` -- CERC bands, plant terms, scenario "
            "definitions (every unconfirmed value flagged in the file)\n"
            "- `dispatch_sim/runners/` -- Scenario 1-4 dispatch logic\n"
            "- `dispatch_sim/rl/` -- Scenario 5 reinforcement-learning "
            "environment\n"
            "- `dispatch_sim/tests/` -- the automated verification suite\n"
            "- `docs/` -- system design and build-guide documents"
        )

    st.write("")
    st.markdown('<div class="au-section">Coming next</div>', unsafe_allow_html=True)
    with st.expander("Coming next: market price upload (IEX) for Scenario 5 / P2P"):
        st.caption("Baseline Scenarios S1-S4 are PPA-only per the team plan and do "
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
                          f"Not yet applied to S1-S4 profit - reserved for Scenario 5.")
            except IEXFormatError as e:
                st.warning(f"Couldn't auto-detect columns: {e}")
                if e.columns:
                    tcol = st.selectbox("Time column", e.columns, key="iex_tcol")
                    pcol = st.selectbox("Price column", e.columns, key="iex_pcol")
                    if st.button("Parse with these columns", key="iex_parse_btn"):
                        try:
                            iex_file.seek(0)
                            iex_r2 = parse_iex_file(iex_file, time_col=tcol, price_col=pcol)
                            st.success(f"Parsed using '{tcol}' / '{pcol}'.")
                        except Exception as e2:
                            st.error(f"Still couldn't parse: {e2}")


    st.markdown('<div class="au-section">Contact</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            "**Atria University -- Centre of Excellence**\n\n"
            "For questions about this simulator or the underlying research, "
            "reach out via the project repository above, or through Atria "
            "University's Centre of Excellence."
        )


SCENARIO_GUIDE = """
| Scenario | What it does |
| --- | --- |
| **S1 · PPA only** | No battery. Sell whatever is generated. The benchmark every other scenario is compared with. |
| **S2 · Battery buffer** | All generation passes through the battery before reaching the grid. |
| **S3 · Time windows** | Charge at midday, discharge in the evening on a fixed timetable -- only on days that beats S1. |
| **S4 · Real-time controller** | Watches each 15-minute block and uses the battery only when the CERC penalty costs more than the stored energy and wear. No foresight, so it could run on a real plant. |
| **S5 · Optimizer** | Perfect hindsight: picks the schedule and battery moves together. A ceiling on what is possible, not something a real plant could do. |
"""

GLOSSARY = """
- **Contracted capacity** -- the plant size in the PPA. CERC judges every deviation as a % of it.
- **Schedule** -- what the plant promised the grid for each 15-minute block, from the day-ahead forecast.
- **Deviation** -- actual delivery minus the schedule in a block.
- **DSM bands** -- CERC's penalty tiers for deviations of 0-5%, 5-10%, 10-20% and over 20% of capacity; the further off, the steeper the rate.
- **PPA revenue** -- scheduled energy x tariff.
- **DSM net** -- the deviation settlement cash flow (negative = the plant paid). Inside the first band it just cancels revenue for energy not delivered.
- **DSM penalty** -- what deviations cost compared with delivering exactly to schedule.
- **Degradation** -- battery wear, charged per kWh cycled.
- **Battery stock change** -- S4 starts the day half full; energy it uses up (or adds) is charged (or credited) so it is compared fairly.
- **Backtest** -- the same comparison repeated over many days.
"""

st.markdown("""
<style>
.au-step{display:flex;align-items:center;gap:10px;margin:2px 0 2px 0;}
.au-step-n{display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;
  border-radius:50%;background:#3D34E0;color:#fff;font-weight:700;font-size:0.85rem;}
.au-step-t{font-weight:700;font-size:1.05rem;}
</style>""", unsafe_allow_html=True)

# Headless tests can't navigate function pages, so AGS_TEST_PAGE (unset in
# production) picks which page opens first.
import os as _os
_start = _os.environ.get("AGS_TEST_PAGE", "home")
PAGE_HOME = st.Page(page_home, title="Home", icon=":material/home:",
                    url_path="home", default=_start == "home")
PAGE_SETUP = st.Page(page_setup, title="Set up", icon=":material/tune:",
                     url_path="setup", default=_start == "setup")
PAGE_RESULTS = st.Page(page_results, title="Results", icon=":material/insights:",
                       url_path="results", default=_start == "results")
PAGE_ALLDAYS = st.Page(page_alldays, title="All days & backtest",
                       icon=":material/calendar_month:", url_path="all-days",
                       default=_start == "all-days")
PAGE_HOW = st.Page(page_how, title="How it works", icon=":material/menu_book:",
                   url_path="how-it-works", default=_start == "how-it-works")
pg = st.navigation([PAGE_HOME, PAGE_SETUP, PAGE_RESULTS, PAGE_ALLDAYS, PAGE_HOW])
sidebar_summary()
pg.run()


st.write("")
st.markdown(
    f'<div style="display:flex;align-items:center;gap:8px;color:{GREY};'
    f'font-size:0.82rem;">{au_mark(18)}<span>Atria University Centre of '
    f'Excellence &nbsp;·&nbsp; <a href="https://github.com/aishwaryamishra444/dispatch-sim" '
    f'target="_blank" style="color:inherit;">Source on GitHub</a> &nbsp;·&nbsp; '
    f'Some tariff and cost inputs are provisional, pending team review.</span></div>',
    unsafe_allow_html=True)
