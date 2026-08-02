"""Loaders — CSV time series, flexitwin battery spec, YAML configs.

Input contracts (all 96 x 15-min blocks for one day):
  forecast CSV: columns time,mw   (time = HH:MM)
  actual   CSV: columns time,mw
  flexitwin JSON: batteryUsableCapacity (MWh), cRateMW, roundTripEfficiency,
                  socMinPct, socMaxPct
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import yaml

from dispatch_sim.core.battery import Battery
from dispatch_sim.core.dsm_settlement import BandTable, DenominatorPolicy, DsmConfig

BLOCKS = 96


def load_series_csv(path: str | Path) -> List[float]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != BLOCKS:
        raise ValueError(f"{path}: expected {BLOCKS} rows (15-min blocks), got {len(rows)}")
    return [float(r["mw"]) for r in rows]


def load_series_csv_buffer(buffer, label: str = "file") -> List[float]:
    """Same contract as load_series_csv (columns time,mw, 96 rows) but reads
    from an in-memory buffer -- e.g. a Streamlit UploadedFile -- instead of a
    path. Column name is matched case-insensitively so 'MW'/'Mw'/'mw' all work."""
    if hasattr(buffer, "seek"):
        buffer.seek(0)
    raw = buffer.read()
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError(f"{label}: file is empty or has no header row.")
    mw_key = next((k for k in rows[0] if k.strip().lower() == "mw"), None)
    if mw_key is None:
        raise ValueError(
            f"{label}: no 'mw' column found - columns present: {list(rows[0].keys())}")
    if len(rows) != BLOCKS:
        raise ValueError(f"{label}: expected {BLOCKS} rows (15-min blocks), got {len(rows)}")
    try:
        return [float(r[mw_key]) for r in rows]
    except ValueError as e:
        raise ValueError(f"{label}: non-numeric value in '{mw_key}' column - {e}") from e


def load_battery(path: str | Path) -> Battery:
    spec = json.loads(Path(path).read_text())
    return Battery(
        usable_capacity_mwh=float(spec["batteryUsableCapacity"]),
        c_rate_mw=float(spec["cRateMW"]),
        rte=float(spec["roundTripEfficiency"]),
        soc_min_pct=float(spec.get("socMinPct", 10)),
        soc_max_pct=float(spec.get("socMaxPct", 90)),
    )


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_dsm_config(path: str | Path) -> DsmConfig:
    c = load_yaml(path)
    bands = BandTable(
        name=f"{c['seller_type']} ({Path(path).name})",
        edges_pct=tuple(float(x) for x in c["volume_limits_pct"]),
        over_rates=tuple(float(x) for x in c["over_rates"]),
        under_rates=tuple(float(x) for x in c["under_rates"]),
    )
    return DsmConfig(
        bands=bands,
        denominator=DenominatorPolicy(c.get("denominator", "available_capacity")),
        blend_avc_pct=float(c.get("blend_avc_pct", 50)),
    )


def window_mask(win: dict) -> List[bool]:
    """Boolean mask over 96 blocks for a {start: 'HH:MM', end: 'HH:MM'} window."""
    def to_h(s):
        hh, mm = s.split(":")
        return int(hh) + int(mm) / 60.0
    a, b = to_h(win["start"]), to_h(win["end"])
    return [(a <= t * 0.25 < b) for t in range(BLOCKS)]


def x_factor_for_date(dsm_yaml: dict, seller: str, date) -> float:
    """Resolve the CERC X-factor (%% of AvC in the deviation denominator)
    from the x_trajectory table for a given date. Indian fiscal year:
    Apr 1 - Mar 31. Falls back to 100 if no trajectory configured."""
    traj = dsm_yaml.get("x_trajectory", {}).get(seller)
    if not traj:
        return 100.0
    fy_start = date.year if date.month >= 4 else date.year - 1
    key = f"FY{fy_start % 100}-{(fy_start + 1) % 100}"
    if key in traj:
        return float(traj[key])
    last = sorted(k for k in traj if k.endswith("+"))
    return float(traj[last[-1]]) if last else 100.0
