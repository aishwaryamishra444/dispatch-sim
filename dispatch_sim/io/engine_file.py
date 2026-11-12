"""Loader for forecasting-engine output: one CSV holding many days of
15-minute actual and forecast generation, e.g.

    date,power_generation_actual_kw,forecast_generation_kw
    2026-02-14 08:00:00+00:00,3.686,3.701
    ...

The simulator runs one IST day at a time (96 blocks, values in MW), so this
module converts the file into that shape:

* timestamps -> Asia/Kolkata. Timezone-aware stamps (like "+00:00") are
  converted; naive ones are taken as UTC or IST, whichever the caller says.
* units -> MW. Detected from the column names ("_kw" / "_mw") when they
  say so; otherwise the caller must choose.
* each calendar day -> exactly 96 blocks. Short gaps (up to max_gap_blocks
  missing readings in a row) are filled by linear interpolation; days with
  longer gaps, or that don't cover the whole day, are skipped and reported
  with the reason, never silently padded.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

BLOCKS = 96
IST = "Asia/Kolkata"


@dataclass
class EngineData:
    days: Dict[str, Tuple[List[float], List[float]]]   # day -> (forecast_mw, actual_mw)
    skipped: Dict[str, str]                             # day -> reason
    time_col: str
    actual_col: str
    forecast_col: str
    unit: str                                           # "kW" or "MW" (as read)
    filled_blocks: Dict[str, int] = field(default_factory=dict)

    @property
    def peak_mw(self) -> float:
        return max((max(a) for _, a in self.days.values()), default=0.0)


def detect_columns(columns) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Best guess at (time, actual, forecast) columns from their names."""
    low = {c: c.strip().lower() for c in columns}
    time_col = next((c for c, l in low.items()
                     if l in ("date", "datetime", "time", "timestamp", "ts")), None)
    if time_col is None:
        time_col = next((c for c, l in low.items() if "time" in l or "date" in l), None)
    actual_col = next((c for c, l in low.items() if "actual" in l), None)
    forecast_col = next((c for c, l in low.items()
                         if "forecast" in l or "fcst" in l or "predict" in l), None)
    return time_col, actual_col, forecast_col


def detect_unit(*column_names: str) -> Optional[str]:
    """'kW' or 'MW' if every named column says so, else None."""
    units = set()
    for name in column_names:
        l = name.lower()
        if "kw" in l and "mw" not in l:
            units.add("kW")
        elif "mw" in l and "kw" not in l:
            units.add("MW")
        else:
            return None
    return units.pop() if len(units) == 1 else None


def read_table(source) -> pd.DataFrame:
    """Read a CSV from a path, bytes, or file-like (e.g. Streamlit UploadedFile)."""
    if hasattr(source, "seek"):
        source.seek(0)
    if hasattr(source, "read"):
        raw = source.read()
        source = io.BytesIO(raw if isinstance(raw, bytes) else raw.encode("utf-8"))
    elif isinstance(source, bytes):
        source = io.BytesIO(source)
    return pd.read_csv(source)


def parse_engine_file(source, unit: Optional[str] = None,
                      naive_timestamps: str = "UTC",
                      time_col: Optional[str] = None,
                      actual_col: Optional[str] = None,
                      forecast_col: Optional[str] = None,
                      max_gap_blocks: int = 2) -> EngineData:
    """Split a multi-day engine file into complete IST days in MW.

    unit: "kW" or "MW"; None means use the column names, and raise if they
          don't say.
    naive_timestamps: "UTC" or "IST" -- how to read stamps with no offset.
    """
    df = read_table(source)
    if df.empty:
        raise ValueError("The file has no rows.")
    t_guess, a_guess, f_guess = detect_columns(df.columns)
    time_col = time_col or t_guess
    actual_col = actual_col or a_guess
    forecast_col = forecast_col or f_guess
    missing = [n for n, c in (("time", time_col), ("actual", actual_col),
                              ("forecast", forecast_col)) if c is None]
    if missing:
        raise ValueError(f"Couldn't find the {', '.join(missing)} column(s) -- columns "
                         f"present: {list(df.columns)}")

    unit = unit or detect_unit(actual_col, forecast_col)
    if unit not in ("kW", "MW"):
        raise ValueError("Couldn't tell whether the values are kW or MW from the column "
                         "names -- please choose the unit.")
    to_mw = 0.001 if unit == "kW" else 1.0

    raw_ts = df[time_col].astype(str)
    has_offset = raw_ts.str.contains(r"(?:[+-]\d{2}:?\d{2}|Z)$", regex=True).all()
    if has_offset:
        ts = pd.to_datetime(raw_ts, utc=True).dt.tz_convert(IST)
    else:
        ts = pd.to_datetime(raw_ts)
        src_tz = "UTC" if naive_timestamps == "UTC" else IST
        ts = ts.dt.tz_localize(src_tz).dt.tz_convert(IST)

    data = pd.DataFrame({
        "ts": ts,
        "actual": pd.to_numeric(df[actual_col], errors="coerce") * to_mw,
        "forecast": pd.to_numeric(df[forecast_col], errors="coerce") * to_mw,
    }).dropna(subset=["ts"])
    if not data["ts"].dt.minute.isin([0, 15, 30, 45]).all():
        raise ValueError("Timestamps aren't on 15-minute boundaries (:00/:15/:30/:45).")
    data = data.groupby("ts").mean().sort_index()   # duplicate stamps -> mean

    days, skipped, filled = {}, {}, {}
    for day in sorted(set(data.index.date)):
        start = pd.Timestamp(day).tz_localize(IST)
        grid = pd.date_range(start, periods=BLOCKS, freq="15min")
        block = data.reindex(grid)
        n_missing = int(block.isna().any(axis=1).sum())
        if n_missing == BLOCKS:
            continue
        block = block.interpolate(limit=max_gap_blocks, limit_area="inside")
        still = int(block.isna().any(axis=1).sum())
        if still:
            skipped[str(day)] = (f"{still} of 96 blocks missing"
                                 + (" (day not fully covered)" if n_missing >= still else ""))
            continue
        if n_missing:
            filled[str(day)] = n_missing
        block = block.clip(lower=0.0)
        days[str(day)] = (block["forecast"].tolist(), block["actual"].tolist())

    if not days:
        raise ValueError("No complete days found -- every day had gaps longer than "
                         f"{max_gap_blocks} blocks or didn't cover all 96 blocks.")
    return EngineData(days=days, skipped=skipped, time_col=time_col,
                      actual_col=actual_col, forecast_col=forecast_col,
                      unit=unit, filled_blocks=filled)
