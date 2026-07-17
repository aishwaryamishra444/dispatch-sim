"""Hand-computed settlement tests for the CERC DSM 2024 WS-seller engine.

Reference block: 10 MW solar plant, 15-min block => Available Capacity
= 2.5 MWh/block. Contract rate CR = Rs 3.00/kWh = Rs 3,000/MWh (chosen
for clean arithmetic).
"""

import math

import pytest

from dispatch_sim.core.dsm_settlement import (
    SOLAR_2024, WIND_2024, BandTable, BlockInput, DenominatorPolicy,
    DsmConfig, settle_block, settle_day,
)

AVC = 2.5          # MWh per block for 10 MW
CR = 3.00          # Rs/kWh
CFG = DsmConfig(bands=SOLAR_2024)


def blk(sched, actual, avc=AVC, cr=CR, i=0):
    return BlockInput(sched, actual, avc, cr, i)


# ---------------------------------------------------------------- basics

def test_zero_deviation_settles_to_zero():
    s = settle_block(blk(2.0, 2.0), CFG)
    assert s.deviation_mwh == 0
    assert s.receivable_inr == s.payable_inr == s.penalty_inr == 0
    assert s.slices == ()


def test_deviation_pct_uses_available_capacity_not_schedule():
    # dev = +0.25 MWh on a 0.5 MWh schedule = 50% of schedule,
    # but only 10% of AvC — Reg 6(2) says AvC is the denominator.
    s = settle_block(blk(0.5, 0.75), CFG)
    assert s.deviation_pct == pytest.approx(10.0)
    # Entirely within VL1+VL2 => paid, not zeroed out.
    assert s.receivable_inr > 0


# ------------------------------------------------- over-injection bands

def test_over_injection_within_vl1_paid_full_contract_rate():
    # dev = +0.1 MWh => 4% of AvC => VL1 @ 100% CR
    s = settle_block(blk(2.0, 2.1), CFG)
    assert s.deviation_pct == pytest.approx(4.0)
    assert s.receivable_inr == pytest.approx(0.1 * 3000)   # Rs 300
    assert s.penalty_inr == pytest.approx(0.0)


def test_over_injection_marginal_slices_across_three_bands():
    # dev = +0.5 MWh => 20% of AvC. Marginal slices:
    #   VL1 0-5%   : 0.125 MWh @ 1.00 -> Rs 375.0
    #   VL2 5-10%  : 0.125 MWh @ 0.90 -> Rs 337.5
    #   VL3 10-20% : 0.250 MWh @ 0.50 -> Rs 375.0
    s = settle_block(blk(2.0, 2.5), CFG)
    assert s.deviation_pct == pytest.approx(20.0)
    assert [round(x.energy_mwh, 6) for x in s.slices] == [0.125, 0.125, 0.25]
    assert s.receivable_inr == pytest.approx(1087.5)
    # Penalty = value lost vs CR on the discounted slices:
    # 0.125*0.10*3000 + 0.25*0.50*3000 = 37.5 + 375 = 412.5
    assert s.penalty_inr == pytest.approx(412.5)
    # Identity: receivable + penalty == full CR value of the deviation
    assert s.receivable_inr + s.penalty_inr == pytest.approx(0.5 * 3000)


def test_over_injection_beyond_vl3_paid_zero():
    # dev = +0.75 MWh => 30%: last 10% (0.25 MWh) beyond VL3 @ 0
    s = settle_block(blk(1.75, 2.5), CFG)
    beyond = [x for x in s.slices if x.band == 3]
    assert len(beyond) == 1
    assert beyond[0].energy_mwh == pytest.approx(0.25)
    assert beyond[0].amount_inr == pytest.approx(0.0)


# ------------------------------------------------ under-injection bands

def test_under_injection_within_vl1_is_pure_refund_no_penalty():
    # dev = -0.1 MWh => 4% => pay back 0.1 MWh @ 100% CR = plain refund
    s = settle_block(blk(2.5, 2.4), CFG)
    assert s.payable_inr == pytest.approx(300.0)
    assert s.penalty_inr == pytest.approx(0.0)


def test_under_injection_through_all_four_bands():
    # dev = -0.75 MWh => 30% of AvC. Slices:
    #   VL1 0.125 @ 1.00 -> 375.0   (penalty 0)
    #   VL2 0.125 @ 1.10 -> 412.5   (penalty 37.5)
    #   VL3 0.250 @ 1.50 -> 1125.0  (penalty 375.0)
    #   >VL3 0.250 @ 2.00 -> 1500.0 (penalty 750.0)
    s = settle_block(blk(2.5, 1.75), CFG)
    assert s.deviation_pct == pytest.approx(-30.0)
    assert s.payable_inr == pytest.approx(3412.5)
    assert s.penalty_inr == pytest.approx(1162.5)


# --------------------------------------------------- revenue identities

@pytest.mark.parametrize("sched,actual", [
    (2.0, 2.0), (2.0, 2.1), (2.0, 2.5), (1.75, 2.5),
    (2.5, 2.4), (2.5, 1.75), (2.5, 0.0),
])
def test_block_revenue_equals_actual_value_minus_penalty(sched, actual):
    s = settle_block(blk(sched, actual), CFG)
    revenue = sched * CR * 1000 + s.receivable_inr - s.payable_inr
    assert revenue == pytest.approx(actual * CR * 1000 - s.penalty_inr)


def test_day_aggregation_matches_block_sums():
    blocks = [blk(2.0, 2.1, i=0), blk(2.5, 1.75, i=1), blk(2.0, 2.0, i=2)]
    day = settle_day(blocks, CFG)
    parts = [settle_block(b, CFG) for b in blocks]
    assert day.receivable_inr == pytest.approx(sum(p.receivable_inr for p in parts))
    assert day.payable_inr == pytest.approx(sum(p.payable_inr for p in parts))
    assert day.total_revenue_inr == pytest.approx(
        day.actual_mwh * CR * 1000 - day.penalty_inr)


# ------------------------------------------------- policies and tables

def test_wind_table_has_wider_tolerance():
    # 12% deviation: beyond solar VL2 but still within wind VL2.
    solar = settle_block(blk(2.5, 2.2), DsmConfig(bands=SOLAR_2024))   # -12%
    wind = settle_block(blk(2.5, 2.2), DsmConfig(bands=WIND_2024))
    assert solar.penalty_inr > wind.penalty_inr


def test_blended_denominator_2026_style():
    # x = 50%: denom = 0.5*2.5 + 0.5*2.0 = 2.25 MWh; dev 0.225 => 10%
    cfg = DsmConfig(bands=SOLAR_2024,
                    denominator=DenominatorPolicy.BLENDED, blend_avc_pct=50)
    s = settle_block(blk(2.0, 2.225), cfg)
    assert s.deviation_pct == pytest.approx(10.0)


def test_schedule_denominator_policy():
    cfg = DsmConfig(bands=SOLAR_2024, denominator=DenominatorPolicy.SCHEDULE)
    s = settle_block(blk(2.0, 2.2), cfg)   # +0.2 on 2.0 = 10%
    assert s.deviation_pct == pytest.approx(10.0)


def test_zero_denominator_settles_in_worst_band():
    cfg = DsmConfig(bands=SOLAR_2024)
    s = settle_block(BlockInput(0.0, 0.4, 0.0, CR, 0), cfg)
    assert math.isinf(s.deviation_pct)
    assert all(x.band == 3 for x in s.slices)
    assert s.receivable_inr == pytest.approx(0.0)   # over beyond VL3 pays zero


def test_band_table_validation():
    with pytest.raises(ValueError):
        BandTable("bad", (10.0, 5.0), (1, 1, 1), (1, 1, 1))
    with pytest.raises(ValueError):
        BandTable("bad", (5.0,), (1.0,), (1.0, 1.0))
