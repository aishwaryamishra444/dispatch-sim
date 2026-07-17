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
    buffer_dev = scen.get("buffer_deviations_outside_windows", True)
    delivered, soc_series, thr = [], [], []
    for t in range(BLOCKS):
        thr0 = battery.throughput_mwh
        to_grid = actual[t]
        if chg_win[t]:
            accepted = battery.charge(min(actual[t] * frac, plan_chg[t]))
            to_grid = actual[t] - accepted
        if dis_win[t]:
            to_grid += battery.discharge(plan_dis[t])
        if buffer_dev:
            dev = to_grid - sched[t]
            if dev > 0:
                to_grid -= battery.charge(dev)
            elif dev < 0:
                to_grid += battery.discharge(-dev)
        delivered.append(to_grid)
        soc_series.append(battery.soc_mwh)
        thr.append(battery.throughput_mwh - thr0)
    return build_ledger(scen["name"], sched, actual, delivered, soc_series,
                        thr, plant, dsm_cfg,
                        scen.get("degradation_inr_per_kwh", 2.5), scen)


RUNNERS = {"s1": run_s1, "s2": run_s2, "s3": run_s3}
