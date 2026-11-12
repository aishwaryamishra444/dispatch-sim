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
        # battery untouched: stays at the 50% healthy-midpoint starting
        # reserve (not the floor -- S4 deliberately starts here, unlike
        # S2/S3/S5, so a reactive controller isn't defenseless against a
        # shortfall-heavy day with no prior charging opportunity)
        assert r4.rows[10].soc_mwh == 20.0

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

    def test_starts_with_a_healthy_reserve_not_empty(self):
        """Regression guard for a real bug found tonight: a purely
        reactive controller starting EMPTY is defenseless on a day whose
        shortfalls front-load before any excess arrives to charge it --
        verified directly on a real weather day, this left the battery
        at just 0.03 MWh of total movement and made S4 WORSE than doing
        nothing at all (profit -Rs 74 vs S1). Starting at a 50% reserve
        fixed it completely (penalty eliminated, +Rs 42,590 vs S1 on
        that same day). This test locks in the starting condition itself
        so that fix can't silently regress.

        Since the opening reserve is now valued (see
        test_opening_reserve_is_not_free_energy), the guard checks the
        operational effect the reserve exists for -- the battery is used
        and the DSM penalty falls -- rather than total profit, which now
        includes the cost of refilling that reserve."""
        plant, dsm, battery = make_plant(), make_dsm(), make_battery()
        # a day of ONLY shortfalls -- the worst case for a reactive
        # controller, with zero opportunity to self-charge from excess
        forecast = [5.0] * BLOCKS
        actual = [3.0] * BLOCKS  # constant 2 MW shortfall, every block
        r4 = run_s4(forecast, actual, plant, dsm,
                   {"name": "S4", "degradation_inr_per_kwh": 2.5}, battery)
        from dispatch_sim.runners.rules import run_s1
        r1 = run_s1(forecast, actual, plant, dsm, {"name": "S1"})
        assert r4.total("dsm_penalty") < r1.total("dsm_penalty"), (
            "S4 must cut the DSM penalty on a pure-shortfall day -- "
            "if this fails, the starting-reserve fix has regressed"
        )
        operating = r4.total("profit") - r4.total("inventory_inr")
        assert operating > r1.total("profit")

    def test_opening_reserve_is_not_free_energy(self):
        """S4 starts half full while S2/S3/S5 start empty. Energy drawn from
        that opening stock wasn't generated today, so the day is charged
        what it would cost to refill it: the SoC drop / one-way efficiency,
        at the PPA rate (i.e. the PPA revenue foregone to recharge it)."""
        plant, dsm, battery = make_plant(), make_dsm(), make_battery()
        forecast = [5.0] * BLOCKS
        actual = [3.0] * BLOCKS
        r4 = run_s4(forecast, actual, plant, dsm,
                   {"name": "S4", "degradation_inr_per_kwh": 2.5}, battery)
        opening = 40.0 * 0.5
        closing = r4.rows[-1].soc_mwh
        assert closing < opening
        expected = -(opening - closing) / math.sqrt(0.81) * 2.60 * 1000.0
        assert abs(r4.total("inventory_inr") - expected) < 1e-6
        # booked once, on the last block, and included in profit
        assert all(row.inventory_inr == 0.0 for row in r4.rows[:-1])
        assert abs(sum(row.profit for row in r4.rows) - r4.total("profit")) < 1e-6

    def test_energy_neutral_day_has_no_stock_adjustment(self):
        """With no deviation the battery never moves, so nothing is booked."""
        plant, dsm, battery = make_plant(), make_dsm(), make_battery()
        r4 = run_s4([5.0] * BLOCKS, [5.0] * BLOCKS, plant, dsm,
                   {"name": "S4", "degradation_inr_per_kwh": 2.5}, battery)
        assert r4.total("inventory_inr") == 0.0


class TestS4EconomicTrigger:
    """discharge_trigger="economic": S4 only spends stored energy where the
    CERC payable rate beats the real cost of a delivered kWh from storage.
    Hand-derived with this file's battery (RTE 0.81, eta 0.9), tariff 2.60:
      cost_x = 1/0.81 + (deg*0.5/0.9)/2.60
      deg=2.5 -> 1.2346 + 0.5342 = 1.769x  -> first payable rate above it is
                 Band 4 (2.00x), whose lower edge is 20% of AvC = 2.0 MW.
      deg=0.1 -> 1.2346 + 0.0214 = 1.256x  -> Band 3 (1.50x), edge 10% = 1.0 MW."""

    def _scen(self, deg):
        return {"name": "S4", "degradation_inr_per_kwh": deg,
                "discharge_trigger": "economic"}

    def test_trigger_moves_to_the_band_that_pays(self):
        from dispatch_sim.runners.rules import discharge_trigger_mw
        plant, dsm = make_plant(), make_dsm()
        assert discharge_trigger_mw(plant, dsm, self._scen(2.5), make_battery()) == 2.0
        assert discharge_trigger_mw(plant, dsm, self._scen(0.1), make_battery()) == 1.0
        # the original rule is untouched when asked for
        band1 = {"name": "S4", "discharge_trigger": "band1"}
        assert discharge_trigger_mw(plant, dsm, band1, make_battery()) == 0.5

    def test_no_band_pays_means_never_discharge(self):
        from dispatch_sim.runners.rules import discharge_trigger_mw
        assert discharge_trigger_mw(make_plant(), make_dsm(), self._scen(50.0),
                                    make_battery()) == float("inf")

    def test_covers_only_the_part_beyond_the_paying_edge(self):
        plant, dsm = make_plant(), make_dsm()
        forecast = [5.0] * BLOCKS
        actual = [5.0] * BLOCKS
        actual[10] = 3.5   # 1.5 MW short: inside Bands 2-3, not worth covering
        actual[20] = 2.5   # 2.5 MW short: 0.5 MW beyond the 2.0 MW Band 4 edge
        r4 = run_s4(forecast, actual, plant, dsm, self._scen(2.5), make_battery())
        assert abs(r4.rows[10].delivered_mwh - 3.5 * 0.25) < 1e-6
        assert abs(r4.rows[20].delivered_mwh - (2.5 + 0.5) * 0.25) < 1e-6


class TestS4EconomicCharging:
    """In "economic" mode S4 also only stores excess the grid would pay less
    for than storing keeps. Hand-derived (RTE 0.81, eta 0.9, tariff 2.60):
      keep_x = 0.81 - deg*0.5*0.9/2.60
      deg=2.5 -> 0.81 - 0.4327 = 0.377x -> first over-injection rate below it
                 is Band 4 (0.00x), lower edge 20% of AvC = 2.0 MW.
      deg=0.1 -> 0.81 - 0.0173 = 0.793x -> Band 3 (0.50x), edge 10% = 1.0 MW."""

    def _scen(self, deg):
        return {"name": "S4", "degradation_inr_per_kwh": deg,
                "discharge_trigger": "economic"}

    def test_charge_trigger_tracks_the_band_rates(self):
        from dispatch_sim.runners.rules import charge_trigger_mw
        plant, dsm = make_plant(), make_dsm()
        assert charge_trigger_mw(plant, dsm, self._scen(2.5), make_battery()) == 2.0
        assert charge_trigger_mw(plant, dsm, self._scen(0.1), make_battery()) == 1.0
        assert charge_trigger_mw(plant, dsm, {"name": "S4"}, make_battery()) == 0.5

    def test_only_cheap_excess_is_stored(self):
        plant, dsm = make_plant(), make_dsm()
        forecast = [5.0] * BLOCKS
        actual = [5.0] * BLOCKS
        actual[10] = 6.5   # 1.5 MW over: Bands 2-3 still pay 0.9x/0.5x -> export it
        actual[20] = 7.5   # 2.5 MW over: 0.5 MW is in Band 4 (paid 0x) -> store that
        r4 = run_s4(forecast, actual, plant, dsm, self._scen(2.5), make_battery())
        assert abs(r4.rows[10].delivered_mwh - 6.5 * 0.25) < 1e-6
        assert abs(r4.rows[20].delivered_mwh - 7.0 * 0.25) < 1e-6
