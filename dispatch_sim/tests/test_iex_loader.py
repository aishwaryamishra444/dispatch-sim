"""Tests for iex_loader — checked against several plausible real-world
layouts, since the exact IEX export format can vary by report and date."""

import io

import pytest

from dispatch_sim.io.iex_loader import IEXFormatError, parse_iex_file


def _csv(text: str) -> io.StringIO:
    return io.StringIO(text)


def test_96_block_mwh_clean_header():
    rows = ["Time Block,MCP (Rs/MWh)"]
    rows += [f"{i},{3000 + i}" for i in range(96)]
    r = parse_iex_file(_csv("\n".join(rows)))
    assert len(r.price_inr_per_kwh) == 96
    assert r.price_inr_per_kwh[0] == pytest.approx(3.000)   # Rs/MWh -> Rs/kWh
    assert r.price_inr_per_kwh[50] == pytest.approx(3.050)
    assert r.source_columns["divisor"] == 1000.0


def test_hourly_24_rows_expands_to_96_blocks():
    rows = ["Hour,Purchase Bid Price (Rs/MWh)"]
    rows += [f"{h},{4000 + h * 10}" for h in range(24)]
    r = parse_iex_file(_csv("\n".join(rows)))
    assert len(r.price_inr_per_kwh) == 96
    # each hour repeated 4x
    assert r.price_inr_per_kwh[0] == r.price_inr_per_kwh[3] == pytest.approx(4.0)
    assert r.price_inr_per_kwh[4] == pytest.approx(4.01)


def test_already_rs_per_kwh_not_double_converted():
    rows = ["Time Block,Area Price (Rs/kWh)"]
    rows += [f"{i},{2.5 + i * 0.01}" for i in range(96)]
    r = parse_iex_file(_csv("\n".join(rows)))
    assert r.price_inr_per_kwh[0] == pytest.approx(2.5)
    assert r.source_columns["divisor"] == 1.0


def test_header_after_blank_title_rows():
    text = (
        "IEX Day Ahead Market Snapshot\n"
        "Region: Karnataka\n"
        "Time Block,MCP (Rs/MWh)\n"
        + "\n".join(f"{i},{3500 + i}" for i in range(96))
    )
    r = parse_iex_file(_csv(text))
    assert len(r.price_inr_per_kwh) == 96
    assert r.price_inr_per_kwh[0] == pytest.approx(3.5)


def test_unrecognized_columns_raise_with_column_list():
    text = "Foo,Bar\n1,2\n3,4"
    with pytest.raises(IEXFormatError) as ei:
        parse_iex_file(_csv(text))
    assert "Foo" in ei.value.columns and "Bar" in ei.value.columns


def test_manual_column_override_bypasses_detection():
    text = "X,Y,Z\n" + "\n".join(f"{i},{3000+i},skip" for i in range(96))
    r = parse_iex_file(_csv(text), time_col="X", price_col="Y")
    assert len(r.price_inr_per_kwh) == 96
    assert r.price_inr_per_kwh[0] == pytest.approx(3.0)


def test_missing_values_are_filled():
    rows = ["Time Block,MCP (Rs/MWh)"]
    for i in range(96):
        val = "" if i in (10, 11, 50) else str(3000 + i)
        rows.append(f"{i},{val}")
    r = parse_iex_file(_csv("\n".join(rows)))
    assert len(r.price_inr_per_kwh) == 96
    assert all(v > 0 for v in r.price_inr_per_kwh)  # no NaNs leaked through
