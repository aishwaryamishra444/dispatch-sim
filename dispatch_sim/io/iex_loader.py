"""iex_loader.py — parse a real IEX Area Price / Market Snapshot download
into the 96-block price series the engine and app expect.

IEX (iexindia.com -> Market Data -> Day Ahead Market -> Market Snapshot /
Area Price) lets you download a day's report as CSV or Excel. The exact
column names have varied across report versions and regions, so this
loader is deliberately tolerant:

  - detects a time/block column from common header spellings
  - detects a price column from common header spellings
  - detects whether price is in Rs/MWh or Rs/kWh (IEX reports are usually
    Rs/MWh; this engine works in Rs/kWh) and converts
  - resamples hourly (24 rows) or 96-block (15-min) data onto the 96-block
    grid the engine uses, via forward-fill for hourly data
  - if columns can't be confidently detected, raises IEXFormatError with
    the list of columns found, so the caller (CLI or app) can ask the
    user to pick manually rather than silently guessing wrong

This keeps a real download from a slightly different report layout from
silently producing wrong prices — a wrong guess here is worse than an
explicit "please pick the column" prompt.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd

BLOCKS = 96

TIME_HEADER_HINTS = [
    "time block", "timeblock", "time_block", "block", "time", "hour",
    "period", "interval",
]
PRICE_HEADER_HINTS = [
    "mcp", "final scheduled volume", "price", "rs/mwh", "rs/kwh",
    "purchase bid", "area price", "clearing price",
]


class IEXFormatError(ValueError):
    """Raised when the loader cannot confidently identify the columns it
    needs. Carries the detected column list so a UI can offer a picker."""

    def __init__(self, message: str, columns: Sequence[str]):
        super().__init__(message)
        self.columns = list(columns)


def _read_any(source: Union[str, Path, "io.BytesIO", "io.StringIO"]) -> pd.DataFrame:
    """Read a CSV or Excel file/buffer, trying a couple of header offsets
    since IEX reports sometimes have a title/blank row before the header."""
    name = getattr(source, "name", str(source))
    is_excel = str(name).lower().endswith((".xls", ".xlsx"))
    reader = pd.read_excel if is_excel else pd.read_csv
    last_err = None
    for skiprows in (0, 1, 2, 3):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            df = reader(source, skiprows=skiprows)
            df.columns = [str(c).strip() for c in df.columns]
            # heuristic: a real header row has >1 non-"Unnamed" column
            named = [c for c in df.columns if not c.lower().startswith("unnamed")]
            if len(named) >= 2:
                return df
        except Exception as e:  # noqa: BLE001 - trying multiple offsets on purpose
            last_err = e
    if last_err:
        raise last_err
    raise IEXFormatError("Could not parse a header row from this file.", [])


def _find_column(columns: Sequence[str], hints: Sequence[str]) -> Optional[str]:
    low = {c: c.lower() for c in columns}
    for hint in hints:
        for col, col_low in low.items():
            if hint in col_low:
                return col
    return None


def _detect_price_unit(series: pd.Series, header: str) -> float:
    """Return a divisor to convert the price column to Rs/kWh.
    IEX MCP is almost always Rs/MWh; typical values 2000-12000.
    If header says kWh, or values look like 2-15, treat as already Rs/kWh."""
    h = header.lower()
    if "kwh" in h:
        return 1.0
    if "mwh" in h:
        return 1000.0
    med = float(series.median())
    return 1.0 if med < 50 else 1000.0  # values >50 are almost certainly Rs/MWh


@dataclass
class IEXParseResult:
    price_inr_per_kwh: List[float]      # length 96
    source_columns: dict                # which columns were used
    detected_rows: int                  # rows before resampling


def parse_iex_file(source: Union[str, Path, "io.BytesIO", "io.StringIO"],
                   time_col: Optional[str] = None,
                   price_col: Optional[str] = None) -> IEXParseResult:
    """Parse an IEX Area Price / Market Snapshot download into a 96-block
    Rs/kWh price series. Pass time_col/price_col explicitly to skip
    auto-detection (used by the manual-picker fallback in the app)."""
    df = _read_any(source)
    cols = list(df.columns)

    tcol = time_col or _find_column(cols, TIME_HEADER_HINTS)
    pcol = price_col or _find_column(cols, PRICE_HEADER_HINTS)

    if tcol is None or pcol is None:
        raise IEXFormatError(
            f"Could not auto-detect time/price columns. "
            f"time_col={tcol!r} price_col={pcol!r}", cols)

    prices = pd.to_numeric(df[pcol], errors="coerce")
    if prices.isna().all():
        raise IEXFormatError(f"Price column '{pcol}' has no numeric values.", cols)

    divisor = _detect_price_unit(prices.dropna(), pcol)
    prices_kwh = (prices / divisor).ffill().bfill()

    n = len(prices_kwh)
    if n == BLOCKS:
        series = prices_kwh.tolist()
    elif n == 24:
        # hourly -> 15-min blocks: repeat each hour 4x
        series = [v for v in prices_kwh for _ in range(4)]
    elif n == 48:
        # 30-min blocks -> 15-min: repeat each 2x
        series = [v for v in prices_kwh for _ in range(2)]
    elif n > 0:
        # generic resample: nearest-neighbour onto 96 evenly spaced blocks
        idx = [int(i * n / BLOCKS) for i in range(BLOCKS)]
        vals = prices_kwh.tolist()
        series = [vals[min(i, n - 1)] for i in idx]
    else:
        raise IEXFormatError("Price column parsed to zero usable rows.", cols)

    return IEXParseResult(
        price_inr_per_kwh=series,
        source_columns={"time": tcol, "price": pcol, "divisor": divisor},
        detected_rows=n,
    )
