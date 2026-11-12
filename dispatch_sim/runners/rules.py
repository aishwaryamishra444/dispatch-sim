"""Scenario runners — thin layers over the core engine (per the design doc).

Each runner takes (forecast_mw, actual_mw, plant cfg, dsm cfg, scenario cfg,
battery or None) and returns a ScenarioResult. No optimization solver anywhere:
Scenarios 1-3 are rule-based, per the team plan. Scenario 5 (Pyomo) later
plugs in as another function with the same signature.
"""

from __future__ import annotations

from typing import List, Optional

from dispatch_sim.core.battery import Battery, DT
from dispatch_sim.core.dsm_settlement import DsmConfig
from dispatch_sim.core.ledger import BLOCKS, ScenarioResult, build_ledger
from dispatch_sim.io.loaders import window_mask


def run_s1(forecast, actual, plant, dsm_cfg: DsmConfig, scen: dict,
           battery: Optional[Battery] = None) -> ScenarioResult:
    """S1 - PPA only: deliver as generated against the day-ahead schedule."""
    zeros = [0.0] * BLOCKS
    return build_ledger(scen["name"], forecast, actual, list(actual), None,
                        zeros, plant, dsm_cfg, 0.0, scen)


def discharge_trigger_mw(plant, dsm_cfg: DsmConfig, scen: dict,
                         battery: Battery) -> float:
    """Shortfall (MW) beyond which S4 discharges. See run_s4's docstring."""
    avc_mw = plant["plant_mw"]
    edges_pct = list(dsm_cfg.bands.edges_pct)
    if scen.get("discharge_trigger", "band1") != "economic":
        return avc_mw * edges_pct[0] / 100.0
    rate = plant["ppa_rate_inr_per_kwh"]
    deg = scen.get("degradation_inr_per_kwh", 2.5)
    # cost of one delivered kWh from storage, as a multiple of the tariff:
    # refill (1/RTE kWh at the PPA rate) + wear on the 1/eta kWh drawn (x0.5, D3)
    cost_x = 1.0 / battery.rte + (deg * 0.5 / battery.eta) / rate
    lower_edges = [0.0] + edges_pct  # band k spans lower_edges[k] .. next edge
    for k, under_rate in enumerate(dsm_cfg.bands.under_rates):
        if under_rate > cost_x:
            return avc_mw * lower_edges[k] / 100.0
    return float("inf")


def run_s4(forecast, actual, plant, dsm_cfg: DsmConfig, scen: dict,
           battery: Battery) -> ScenarioResult:
    """S4 - Real-time reactive controller. Unlike S5 (perfect foresight,
    rewrites the whole day's schedule), S4 keeps the schedule fixed at the
    day-ahead forecast and reacts block-by-block using ONLY that block's
    actual-vs-schedule gap -- a genuinely deployable, non-clairvoyant
    strategy. It intervenes only once a deviation would push past CERC's
    gentlest band (Band 1, the first entry in dsm_cfg.bands.edges_pct) into
    the steeper tiers -- there is NO zero-penalty dead-band in the real
    regulation (Band 1 is charged at 100% of tariff), so the trigger and
    target are both set at the real Band 1 edge, not an invented free zone.
    Uses the real Battery class throughout, so its physics/efficiency are
    identical to every other scenario -- a fair comparison, not a separate
    approximation.

    Starts at a 50% "healthy midpoint" reserve, not the empty floor S2/S3/S5
    use. This is a deliberate, S4-specific choice: a purely reactive
    controller with no foresight is defenseless against a day whose
    shortfalls front-load before any excess has arrived to charge it --
    verified directly: on one such real weather day, starting empty left
    the battery almost entirely unused (0.03 MWh of movement, worse than
    doing nothing at all); starting at 50% let it eliminate the day's
    entire DSM penalty instead. A real operator would keep a working
    reserve for exactly this reason, rather than starting every day at
    zero.

    Discharge trigger (scen["discharge_trigger"]):
      "band1"    -- the original rule: cover any shortfall beyond the Band 1
                    edge, back to that edge.
      "economic" -- cover a shortfall only beyond the band edge where
                    CERC's payable rate first exceeds what a delivered kWh
                    from storage really costs (refilling it at the PPA rate
                    after round-trip loss, plus cell wear). Still purely
                    reactive and still set by the real CERC bands -- it just
                    stops spending stored energy where the penalty it avoids
                    is cheaper than the energy itself. Recomputed from the
                    live tariff, degradation cost and RTE, so it moves with
                    the sliders. If no band's rate exceeds that cost, S4
                    never discharges.
    Excess beyond the Band 1 edge is absorbed in both modes. (Absorbing
    smaller excess to refill the reserve was tested and loses money: Band 1
    over-injection is already paid at the full contract rate.)"""
    battery.soc_mwh = battery.usable_capacity_mwh * 0.5
    opening_soc_mwh = battery.soc_mwh

    sched = list(forecast)  # fixed at forecast -- no foresight, unlike S5
    avc_mw = plant["plant_mw"]  # AvC(MWh) = plant_mw * DT; in MW terms this IS plant_mw
    edges_pct = list(dsm_cfg.bands.edges_pct)
    band1_edge_pct = edges_pct[0] / 100.0  # real config value, e.g. 0.05
    threshold_mw = avc_mw * band1_edge_pct
    discharge_threshold_mw = discharge_trigger_mw(plant, dsm_cfg, scen, battery)

    delivered, soc_series, thr = [], [], []
    for t in range(BLOCKS):
        thr0 = battery.throughput_mwh
        gap_mw = sched[t] - actual[t]  # positive = shortfall, negative = excess

        if gap_mw > discharge_threshold_mw:
            # Shortfall past the trigger -- discharge just enough to pull
            # the gap back to the trigger edge, not all the way to zero.
            need = gap_mw - discharge_threshold_mw
            dis = battery.discharge(need)
            net_delivered = actual[t] + dis
        elif -gap_mw > threshold_mw:
            # Excess beyond Band 1 -- charge just enough to pull the gap
            # back down to the Band 1 edge.
            need = (-gap_mw) - threshold_mw
            accepted = battery.charge(need)
            net_delivered = actual[t] - accepted
        else:
            # Inside the trigger -- not worth spending battery wear on.
            net_delivered = actual[t]

        delivered.append(net_delivered)
        soc_series.append(battery.soc_mwh)
        thr.append(battery.throughput_mwh - thr0)

    # The 50% opening reserve is not free energy. Value the day's net change
    # in stored energy so S4 is compared fairly with strategies that start
    # empty: a drawdown is charged at the PPA revenue foregone to refill it
    # (delta / one-way efficiency); a net gain is credited at what it could
    # deliver (delta x one-way efficiency).
    rate_inr_per_mwh = plant["ppa_rate_inr_per_kwh"] * 1000.0
    delta_mwh = battery.soc_mwh - opening_soc_mwh
    if delta_mwh < 0:
        inventory_inr = delta_mwh / battery.eta * rate_inr_per_mwh
    else:
        inventory_inr = delta_mwh * battery.eta * rate_inr_per_mwh

    return build_ledger(scen["name"], sched, actual, delivered, soc_series,
                        thr, plant, dsm_cfg,
                        scen.get("degradation_inr_per_kwh", 2.5), scen,
                        end_inventory_inr=inventory_inr)


def run_s2(forecast, actual, plant, dsm_cfg: DsmConfig, scen: dict,
           battery: Battery) -> ScenarioResult:
    """S2 - Buffer: all generation passes through the BESS; discharge targets
    the schedule. D1 (signed-off default): schedule revised to forecast x RTE.
    If the battery cannot accept generation (C-rate/SoC), the excess spills
    straight to grid — power has nowhere else to go."""
    rte = battery.rte
    sched = [f * (rte if scen.get("revise_schedule_for_rte", True) else 1.0)
             for f in forecast]
    delivered, soc_series, thr = [], [], []
    for t in range(BLOCKS):
        thr0 = battery.throughput_mwh
        accepted = battery.charge(actual[t])
        spill = actual[t] - accepted
        dis = battery.discharge(max(0.0, sched[t] - spill))
        delivered.append(spill + dis)
        soc_series.append(battery.soc_mwh)
        thr.append(battery.throughput_mwh - thr0)
    return build_ledger(scen["name"], sched, actual, delivered, soc_series,
                        thr, plant, dsm_cfg,
                        scen.get("degradation_inr_per_kwh", 2.5), scen)


def run_s3(forecast, actual, plant, dsm_cfg: DsmConfig, scen: dict,
           battery: Battery) -> ScenarioResult:
    """S3 - Time windows: charge a fixed fraction of generation in the charge
    window, discharge in the evening window; one cycle per day. Still PPA-only
    (flat rate) — the only upside is DSM reduction, so outside the windows the
    battery buffers deviations against the schedule.

    D2 (signed-off default): schedule-integrated — planned charge and planned
    discharge are baked into the submitted schedule, so the timed discharge is
    scheduled energy at full PPA rate, not over-injection."""
    frac = float(scen.get("charge_fraction", 0.35))
    chg_win = window_mask(scen["charge_window"])
    dis_win = window_mask(scen["discharge_window"])
    eta = battery.eta

    # ---- day-ahead plan (from forecast only; actuals unknown at planning) ----
    plan_chg = [min(forecast[t] * frac, battery.c_rate_mw) if chg_win[t] else 0.0
                for t in range(BLOCKS)]
    stored_mwh = sum(p * eta * DT for p in plan_chg)
    dis_blocks = sum(dis_win)
    plan_dis_mw = min(stored_mwh * eta / (dis_blocks * DT), battery.c_rate_mw) \
        if dis_blocks else 0.0
    plan_dis = [plan_dis_mw if dis_win[t] else 0.0 for t in range(BLOCKS)]

    if scen.get("schedule_integrated", True):
        sched = [max(0.0, forecast[t] - plan_chg[t]) + plan_dis[t]
                 for t in range(BLOCKS)]
    else:
        sched = list(forecast)

    # ---- real-time dispatch against actuals ----
    # Each block decides ONE net battery flow. Previously the window step
    # and the deviation-buffer step called the battery separately, so a
    # charge-window block with a shortfall would store its planned share
    # and then immediately discharge it again -- ~17 blocks/day of pointless
    # cycling, paying wear and round-trip loss twice for nothing.
    buffer_dev = scen.get("buffer_deviations_outside_windows", True)
    delivered, soc_series, thr = [], [], []
    for t in range(BLOCKS):
        thr0 = battery.throughput_mwh
        if buffer_dev:
            # aim delivery at the submitted schedule (which already carries
            # the planned window charge/discharge, per D2)
            target = sched[t]
        else:
            planned_chg = min(actual[t] * frac, plan_chg[t]) if chg_win[t] else 0.0
            planned_dis = plan_dis[t] if dis_win[t] else 0.0
            target = actual[t] - planned_chg + planned_dis
        net = target - actual[t]  # >0: battery must export, <0: absorb
        if net > 0:
            to_grid = actual[t] + battery.discharge(net)
        elif net < 0:
            to_grid = actual[t] - battery.charge(-net)
        else:
            to_grid = actual[t]
        delivered.append(to_grid)
        soc_series.append(battery.soc_mwh)
        thr.append(battery.throughput_mwh - thr0)
    return build_ledger(scen["name"], sched, actual, delivered, soc_series,
                        thr, plant, dsm_cfg,
                        scen.get("degradation_inr_per_kwh", 2.5), scen)


RUNNERS = {"s1": run_s1, "s2": run_s2, "s3": run_s3}
