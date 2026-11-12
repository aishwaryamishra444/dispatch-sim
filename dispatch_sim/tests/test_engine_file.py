"""Tests for the forecasting-engine file loader (multi-day, UTC/IST, kW/MW)."""
import pandas as pd
import pytest

from dispatch_sim.io.engine_file import detect_unit, parse_engine_file


def make_csv(start="2026-07-12 18:30:00+00:00", periods=96 * 2, kw=True,
             drop=(), actual_nan=(), offset=True):
    ts = pd.date_range(pd.Timestamp(start), periods=periods, freq="15min")
    # recognisable values: block index within the IST day, in kW
    ist = ts.tz_convert("Asia/Kolkata") if ts.tz is not None else ts
    vals = [(t.hour * 4 + t.minute // 15) / 10 for t in ist]
    stamps = [str(t) if offset else t.strftime("%Y-%m-%d %H:%M:%S") for t in ts]
    unit = "kw" if kw else "mw"
    df = pd.DataFrame({"date": stamps, f"power_generation_actual_{unit}": vals,
                       f"forecast_generation_{unit}": [v + 0.05 for v in vals]})
    for i in actual_nan:
        df.loc[i, f"power_generation_actual_{unit}"] = None
    df = df.drop(index=list(drop))
    return df.to_csv(index=False).encode()


def test_utc_offsets_become_ist_days_in_mw():
    # 2026-07-12 18:30 UTC == 2026-07-13 00:00 IST: two full IST days
    d = parse_engine_file(make_csv())
    assert list(d.days) == ["2026-07-13", "2026-07-14"]
    fc, act = d.days["2026-07-13"]
    assert len(fc) == len(act) == 96
    assert act[48] == pytest.approx(4.8 / 1000)          # 12:00 IST -> block 48, kW -> MW
    assert fc[48] == pytest.approx(4.85 / 1000)
    assert d.unit == "kW" and d.skipped == {}


def test_partial_first_day_is_skipped_with_reason():
    d = parse_engine_file(make_csv(start="2026-07-13 02:30:00+00:00", periods=96 * 2))
    assert "2026-07-13" in d.skipped and "not fully covered" in d.skipped["2026-07-13"]
    assert "2026-07-14" in d.days


def test_short_gaps_are_filled_long_gaps_skip_the_day():
    d = parse_engine_file(make_csv(drop=[40, 41], actual_nan=[100]))
    assert d.filled_blocks == {"2026-07-13": 2, "2026-07-14": 1}
    # interpolation lands on the straight line between neighbours
    assert d.days["2026-07-13"][1][40] == pytest.approx(4.0 / 1000)
    d = parse_engine_file(make_csv(drop=[40, 41, 42]))
    assert "2026-07-13" in d.skipped and "2026-07-14" in d.days


def test_naive_timestamps_follow_the_callers_choice():
    raw = make_csv(start="2026-07-13 00:00:00", offset=False)
    as_ist = parse_engine_file(raw, naive_timestamps="IST")
    assert list(as_ist.days) == ["2026-07-13", "2026-07-14"]
    as_utc = parse_engine_file(raw, naive_timestamps="UTC")
    assert "2026-07-13" in as_utc.skipped    # shifted +5:30, so IST day 1 is partial


def test_mw_columns_are_not_rescaled_and_unknown_unit_must_be_chosen():
    d = parse_engine_file(make_csv(kw=False))
    assert d.unit == "MW" and d.days["2026-07-13"][1][48] == pytest.approx(4.8)
    raw = make_csv().replace(b"_kw", b"")
    with pytest.raises(ValueError, match="kW or MW"):
        parse_engine_file(raw)
    assert parse_engine_file(raw, unit="MW").days["2026-07-13"][1][48] == pytest.approx(4.8)


def test_detect_unit():
    assert detect_unit("actual_kw", "forecast_kw") == "kW"
    assert detect_unit("actual_MW", "forecast_mw") == "MW"
    assert detect_unit("actual", "forecast_kw") is None
