"""Hand-verified tests for run_s3 (Scenario 3 -- Time windows). Every
expected value below was derived independently by hand, tracing the real
battery physics (charge()/discharge() semantics in core/battery.py) block
by block -- not copied from the engine's own output. Written because S3
had zero dedicated test coverage despite being a real, industry-standard
charge-at-peak / discharge-in-evening strategy that stakeholders will
compare against their own reference numbers.
"""
import math
import pytest
from pathlib import Path

from dispatch_sim.core.battery import Battery
from dispatch_sim.io.loaders import load_dsm_config
from dispatch_sim.runners.rules import run_s3

BLOCKS = 96
PLANT_MW = 10.0
RATE = 2.60
CFG = Path(__file__).parent.parent / "config"


def make_plant():
    return {"plant_mw": PLANT_MW, "ppa_rate_inr_per_kwh": RATE, "om_inr_per_day": 20000.0}


def make_dsm():
    return load_dsm_config(CFG / "dsm_bands.yaml")


def make_s3_cfg(deg_rate=2.5):
    return {
        "name": "S3 - Time windows",
        "charge_window": {"start": "09:00", "end": "14:00"},
        "discharge_window": {"start": "18:00", "end": "22:00"},
        "charge_fraction": 0.35,
        "schedule_integrated": True,
        "buffer_deviations_outside_windows": True,
        "degradation_inr_per_kwh": deg_rate,
    }


def make_battery():
    # capacity=40, c_rate=20 (never binding here), rte=0.81 -> eta=0.9 exactly
    return Battery(usable_capacity_mwh=40.0, c_rate_mw=20.0, rte=0.81,
                  soc_min_pct=10.0, soc_max_pct=90.0)


class TestS3WindowBoundaries:
    """Confirm the charge/discharge windows cover exactly the blocks the
    config says they should -- 09:00-14:00 = 20 blocks, 18:00-22:00 = 16
    blocks, independently counted by hand against window_mask's half-open
    interval semantics (a <= t*0.25 < b)."""

    def test_constant_day_full_hand_trace(self):
        """Forecast = actual = 5.0 MW constant, all 96 blocks. This makes
        every quantity hand-computable exactly -- see the docstring math
        above. Every asserted number here was derived independently,
        before running the engine."""
        forecast = [5.0] * BLOCKS
        actual = [5.0] * BLOCKS
        plant, dsm, s3_cfg, battery = make_plant(), make_dsm(), make_s3_cfg(), make_battery()

        r = run_s3(forecast, actual, plant, dsm, s3_cfg, battery)

        eta = math.sqrt(0.81)
        assert abs(eta - 0.9) < 1e-9  # sanity on the test's own chosen RTE

        # --- Hand-computed plan ---
        # plan_chg = min(5.0*0.35, c_rate=20) = 1.75 MW for the 20 charge blocks
        plan_chg_mw = min(5.0 * 0.35, 20.0)
        assert abs(plan_chg_mw - 1.75) < 1e-9
        # stored_mwh = 20 blocks * 1.75 MW * eta * DT
        stored_mwh = 20 * plan_chg_mw * eta * 0.25
        assert abs(stored_mwh - 7.875) < 1e-9
        # plan_dis_mw = min(stored_mwh*eta/(16*DT), c_rate=20)
        plan_dis_mw = min(stored_mwh * eta / (16 * 0.25), 20.0)
        assert abs(plan_dis_mw - 1.771875) < 1e-9

        rows_by_time = {row.time: row for row in r.rows}

        # --- Scheduled ENERGY (MWh per block = MW-rate x DT) ---
        # charge window: sched_MW = 5.0 - 1.75 = 3.25 MW -> scheduled_mwh = 3.25*0.25 = 0.8125
        for t in ["09:00", "10:00", "13:45"]:
            assert abs(rows_by_time[t].scheduled_mwh - 3.25 * 0.25) < 1e-6, t
        # discharge window: sched_MW = 5.0 + 1.771875 -> scheduled_mwh = that * 0.25
        for t in ["18:00", "19:00", "21:45"]:
            assert abs(rows_by_time[t].scheduled_mwh - 6.771875 * 0.25) < 1e-6, t
        # outside both windows: sched_MW = forecast = 5.0 -> scheduled_mwh = 5.0*0.25 = 1.25
        for t in ["00:00", "08:45", "14:00", "17:45", "22:00", "23:45"]:
            assert abs(rows_by_time[t].scheduled_mwh - 5.0 * 0.25) < 1e-6, t

        # --- Zero deviation everywhere (delivered exactly matches schedule
        # by construction in this controlled scenario) => zero DSM anywhere ---
        for row in r.rows:
            assert abs(row.deviation_mwh) < 1e-6, row.time
            assert row.dsm_receivable == 0.0, row.time
            assert row.dsm_payable == 0.0, row.time

        # --- SoC trajectory: hand-traced checkpoints ---
        # starts at floor = 40*0.10 = 4.0
        assert abs(rows_by_time["00:00"].soc_mwh - 4.0) < 1e-6
        # end of charge window (13:45, last charge block): soc = 4.0 + 20*(1.75*0.9*0.25)
        #                                                       = 4.0 + 7.875 = 11.875
        assert abs(rows_by_time["13:45"].soc_mwh - 11.875) < 1e-4
        # end of discharge window (21:45, last discharge block): back to floor,
        # since stored_mwh's planning formula is internally consistent with the
        # real charge()/discharge() physics -- hand-traced to land exactly at 4.0
        assert abs(rows_by_time["21:45"].soc_mwh - 4.0) < 1e-3

        # --- Degradation: total throughput x rate x 0.5 ---
        # charge throughput = 20 blocks * 1.75 MW * DT = 8.75 MWh
        # discharge throughput (drawn side) = 16 blocks * (1.771875/eta) * DT = 7.875 MWh
        charge_thr = 20 * 1.75 * 0.25
        discharge_thr = 16 * (1.771875 / eta) * 0.25
        total_thr_kwh = (charge_thr + discharge_thr) * 1000
        expected_degradation = total_thr_kwh * 2.5 * 0.5
        assert abs(r.total("degradation") - expected_degradation) < 1.0, \
            f"expected ~{expected_degradation:.2f}, got {r.total('degradation'):.2f}"

    def test_schedule_integrated_false_bypasses_windowing_in_schedule(self):
        """When schedule_integrated=False, the submitted schedule must
        equal the raw forecast, un-touched by any battery plan -- even
        though the battery still physically charges/discharges in
        real time."""
        forecast = [5.0] * BLOCKS
        actual = [5.0] * BLOCKS
        plant, dsm, battery = make_plant(), make_dsm(), make_battery()
        s3_cfg = make_s3_cfg()
        s3_cfg["schedule_integrated"] = False

        r = run_s3(forecast, actual, plant, dsm, s3_cfg, battery)
        for row in r.rows:
            assert abs(row.scheduled_mwh - 5.0 * 0.25) < 1e-6, row.time

    def test_battery_never_exceeds_configured_soc_bounds(self):
        """Regardless of the day's shape, SoC must never breach the
        floor (10%) or ceiling (90%) of usable capacity -- 4.0 to 36.0
        MWh for this test's battery."""
        import numpy as np
        rng = np.random.default_rng(7)
        forecast = list(rng.uniform(0, 10, BLOCKS))
        actual = list(rng.uniform(0, 10, BLOCKS))
        plant, dsm, s3_cfg, battery = make_plant(), make_dsm(), make_s3_cfg(), make_battery()

        r = run_s3(forecast, actual, plant, dsm, s3_cfg, battery)
        for row in r.rows:
            assert 4.0 - 1e-6 <= row.soc_mwh <= 36.0 + 1e-6, \
                f"SoC breach at {row.time}: {row.soc_mwh}"

    def test_daily_total_equals_sum_of_blocks(self):
        """The scenario's total profit must be exactly the sum of its
        96 individual block profits -- no separate daily calculation
        exists anywhere else that could silently diverge."""
        import numpy as np
        rng = np.random.default_rng(3)
        forecast = list(rng.uniform(0, 10, BLOCKS))
        actual = list(rng.uniform(0, 10, BLOCKS))
        plant, dsm, s3_cfg, battery = make_plant(), make_dsm(), make_s3_cfg(), make_battery()

        r = run_s3(forecast, actual, plant, dsm, s3_cfg, battery)
        assert abs(sum(row.profit for row in r.rows) - r.total("profit")) < 1e-6
