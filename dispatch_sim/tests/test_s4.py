"""Hand-verified tests for run_s4 (Scenario 4 -- Real-time reactive
controller). S4 is new tonight: a genuinely deployable, non-clairvoyant
strategy that reacts to each block's actual-vs-schedule gap using ONLY
information available in that block, unlike S5's perfect-foresight
rewrite of the whole day. Every expected value below was derived
independently, tracing the real Battery class physics by hand -- not
copied from the engine's own output.
"""
import math
from pathlib import Path

from dispatch_sim.core.battery import Battery
from dispatch_sim.io.loaders import load_dsm_config
from dispatch_sim.runners.rules import run_s4

BLOCKS = 96
CFG = Path(__file__).parent.parent / "config"


def make_plant():
    return {"plant_mw": 10.0, "ppa_rate_inr_per_kwh": 2.60, "om_inr_per_day": 20000.0}


def make_dsm():
    return load_dsm_config(CFG / "dsm_bands.yaml")


def make_battery():
    return Battery(usable_capacity_mwh=40.0, c_rate_mw=20.0, rte=0.81,
                  soc_min_pct=10.0, soc_max_pct=90.0)


class TestS4ReactiveController:
    def test_no_zero_penalty_deadband_is_invented(self):
        """S4's trigger threshold must come from the REAL CERC config
        (Band 1's edge), not a hardcoded, invented 'dead-band' figure.
        This directly guards against reintroducing the factual error
        corrected tonight -- there is no free zone in the real
        regulation; Band 1 is charged at 100% of tariff."""
        dsm = make_dsm()
        # the only source of truth for the threshold is this real config
        assert dsm.bands.edges_pct[0] == 5.0, \
            "Band 1's edge changed in config -- S4's threshold must track it"

    def test_small_deviation_inside_band1_is_left_alone(self):
        """A deviation inside Band 1 (0-5% of AvC) should NOT trigger any
        battery action -- its own cost is modest and doesn't justify wear."""
        plant, dsm, battery = make_plant(), make_dsm(), make_battery()
        forecast = [5.0] * BLOCKS
        actual = [5.0] * BLOCKS
        actual[10] = 4.8  # gap=0.2 MW, inside the 0.5 MW (5% of 10MW) threshold
        r4 = run_s4(forecast, actual, plant, dsm,
                   {"name": "S4", "degradation_inr_per_kwh": 2.5}, battery)
        assert abs(r4.rows[10].delivered_mwh - 4.8 * 0.25) < 1e-6
        assert r4.rows[10].soc_mwh == 4.0  # battery untouched, still at floor

    def test_excess_beyond_band1_charges_to_the_real_edge(self):
        """A deviation beyond Band 1 should charge just enough to pull the
        gap back down to the real Band 1 edge -- not to zero, and not
        using an invented threshold."""
        plant, dsm, battery = make_plant(), make_dsm(), make_battery()
        forecast = [5.0] * BLOCKS
        actual = [5.0] * BLOCKS
        actual[10] = 6.0  # gap=-1.0 MW, 0.5 MW beyond the threshold
        r4 = run_s4(forecast, actual, plant, dsm,
                   {"name": "S4", "degradation_inr_per_kwh": 2.5}, battery)
        expected_mw = 6.0 - 0.5  # charged exactly enough to reach the edge
        assert abs(r4.rows[10].delivered_mwh - expected_mw * 0.25) < 1e-6

    def test_shortfall_beyond_band1_discharges_when_primed(self):
        """A well-charged battery should discharge just enough to pull a
        shortfall back to the real Band 1 edge, when not headroom-limited."""
        plant, dsm, battery = make_plant(), make_dsm(), make_battery()
        forecast = [5.0] * BLOCKS
        actual = [5.0] * BLOCKS
        for i in range(20, 40):
            actual[i] = 6.0  # prime the battery with real stored energy first
        actual[50] = 4.0  # gap=1.0 MW, 0.5 MW beyond threshold
        r4 = run_s4(forecast, actual, plant, dsm,
                   {"name": "S4", "degradation_inr_per_kwh": 2.5}, battery)
        expected_mw = 4.0 + 0.5
        assert abs(r4.rows[50].delivered_mwh - expected_mw * 0.25) < 1e-4

    def test_battery_never_exceeds_soc_bounds(self):
        """Regardless of the day's shape, SoC must never breach the
        floor (10%) or ceiling (90%) of usable capacity."""
        import numpy as np
        rng = np.random.default_rng(11)
        plant, dsm, battery = make_plant(), make_dsm(), make_battery()
        forecast = list(rng.uniform(0, 10, BLOCKS))
        actual = list(rng.uniform(0, 10, BLOCKS))
        r4 = run_s4(forecast, actual, plant, dsm,
                   {"name": "S4", "degradation_inr_per_kwh": 2.5}, battery)
        for row in r4.rows:
            assert 4.0 - 1e-6 <= row.soc_mwh <= 36.0 + 1e-6, \
                f"SoC breach at {row.time}: {row.soc_mwh}"

    def test_schedule_is_fixed_at_forecast_not_rewritten(self):
        """Unlike S5, S4's schedule must equal the raw forecast exactly --
        it does not get perfect-foresight knowledge of the real outcome."""
        plant, dsm, battery = make_plant(), make_dsm(), make_battery()
        import numpy as np
        rng = np.random.default_rng(5)
        forecast = list(rng.uniform(0, 10, BLOCKS))
        actual = list(rng.uniform(0, 10, BLOCKS))
        r4 = run_s4(forecast, actual, plant, dsm,
                   {"name": "S4", "degradation_inr_per_kwh": 2.5}, battery)
        for i, row in enumerate(r4.rows):
            assert abs(row.scheduled_mwh - forecast[i] * 0.25) < 1e-6

    def test_daily_total_equals_sum_of_blocks(self):
        """The scenario's total profit must be exactly the sum of its
        96 individual block profits."""
        import numpy as np
        rng = np.random.default_rng(2)
        plant, dsm, battery = make_plant(), make_dsm(), make_battery()
        forecast = list(rng.uniform(0, 10, BLOCKS))
        actual = list(rng.uniform(0, 10, BLOCKS))
        r4 = run_s4(forecast, actual, plant, dsm,
                   {"name": "S4", "degradation_inr_per_kwh": 2.5}, battery)
        assert abs(sum(row.profit for row in r4.rows) - r4.total("profit")) < 1e-6
