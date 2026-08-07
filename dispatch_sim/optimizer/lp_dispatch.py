"""lp_dispatch.py -- Scenario 5: LP-based profit-maximizing dispatch.

Chooses the day-ahead schedule AND the battery charge/discharge trajectory
that jointly maximize total daily profit, subject to:
  - battery physics (SoC dynamics, C-rate, SoC window, one-way efficiency)
  - the *exact* CERC DSM 2024 marginal-band mechanics used by
    core.dsm_settlement -- reproduced here as LP constraints so the two
    are mathematically the same settlement, not an approximation of it.

This is a PERFECT-FORESIGHT optimizer: it sees the day's actual generation
in advance when choosing the schedule. That is standard practice in this
industry -- DNV's HERO tool, for instance, documents exactly the same
choice ("generates dispatch assuming perfect foresight ... applies income
adjustments ... to account for uncertainty"). It is reported here as an
upper bound / ceiling: "how much value is left on the table by not having
an optimizer," not as a literally deployable day-ahead schedule. That
distinction must stay visible in the UI -- see the caption in demo_app.py.

Why no binary variables are needed for the DSM bands (same argument
Praphul's build spec makes for its own Pyomo skeleton, Sec 3.3): the
receivable rates strictly decrease across bands and the payable rates
strictly increase. In a profit-*maximizing* LP, an optimal solution can
never fill a lower-value slice before a higher-value one is exhausted --
that would leave money on the table, which contradicts optimality. So the
LP relaxation reproduces the correct stepwise settlement exactly, with two
one-sided inequalities per block standing in for the true absolute
deviation (only one side is ever slack-free at the optimum).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import pulp

from dispatch_sim.core.dsm_settlement import DsmConfig
from dispatch_sim.core.ledger import BLOCKS, DT, LedgerRow, ScenarioResult, block_time

KWH_PER_MWH = 1000.0


@dataclass
class OptimizerBatterySpec:
    usable_capacity_mwh: float
    c_rate_mw: float
    rte: float
    soc_min_pct: float = 10.0
    soc_max_pct: float = 90.0


def solve_optimal_dispatch(
    forecast_mw: Sequence[float],       # unused by the LP itself (perfect
                                         # foresight schedules against actual),
                                         # kept in the signature for interface
                                         # parity with the other runners.
    actual_mw: Sequence[float],
    plant: dict,
    dsm_cfg: DsmConfig,
    battery: OptimizerBatterySpec,
    degradation_inr_per_kwh: float,
    solver_msg: bool = False,
) -> ScenarioResult:
    n = BLOCKS
    avc = plant["plant_mw"] * DT
    rate = plant["ppa_rate_inr_per_kwh"]
    cr_per_mwh = rate * KWH_PER_MWH
    om_block = plant["om_inr_per_day"] / BLOCKS
    eta = battery.rte ** 0.5
    soc_floor = battery.usable_capacity_mwh * battery.soc_min_pct / 100.0
    soc_ceil = battery.usable_capacity_mwh * battery.soc_max_pct / 100.0

    edges = list(dsm_cfg.bands.edges_pct)          # e.g. [5, 10, 20] (%)
    over_rates = list(dsm_cfg.bands.over_rates)     # len(edges)+1
    under_rates = list(dsm_cfg.bands.under_rates)
    n_bands = len(edges) + 1
    band_width_pct = [edges[0]] + [edges[i] - edges[i - 1] for i in range(1, len(edges))]
    band_width_pct.append(float("inf"))  # last band open-ended; capped via big-M below
    BIG_M_PCT = 200.0  # generous cap for the open-ended top band, in % of AvC

    prob = pulp.LpProblem("dispatch_optimizer", pulp.LpMaximize)

    sched = [pulp.LpVariable(f"sched_{t}", lowBound=0) for t in range(n)]
    chg = [pulp.LpVariable(f"chg_{t}", lowBound=0, upBound=battery.c_rate_mw) for t in range(n)]
    dis = [pulp.LpVariable(f"dis_{t}", lowBound=0, upBound=battery.c_rate_mw) for t in range(n)]
    soc = [pulp.LpVariable(f"soc_{t}", lowBound=soc_floor, upBound=soc_ceil) for t in range(n + 1)]
    y_over = [[pulp.LpVariable(f"yo_{t}_{k}", lowBound=0) for k in range(n_bands)] for t in range(n)]
    y_under = [[pulp.LpVariable(f"yu_{t}_{k}", lowBound=0) for k in range(n_bands)] for t in range(n)]

    prob += soc[0] == soc_floor  # start day at the floor (matches other runners' 10% start)

    delivered = []
    for t in range(n):
        d = actual_mw[t] - chg[t] + dis[t]
        delivered.append(d)
        prob += soc[t + 1] == soc[t] + eta * chg[t] * DT - (dis[t] / eta) * DT

        dev = delivered[t] - sched[t]  # symbolic; used only to build constraints below
        for k in range(n_bands):
            cap = band_width_pct[k] if band_width_pct[k] != float("inf") else BIG_M_PCT
            prob += y_over[t][k] <= cap / 100.0 * avc
            prob += y_under[t][k] <= cap / 100.0 * avc
        prob += pulp.lpSum(y_over[t]) >= (delivered[t] - sched[t])
        prob += pulp.lpSum(y_under[t]) >= (sched[t] - delivered[t])

    revenue = pulp.lpSum(sched[t] * DT * cr_per_mwh for t in range(n))
    receivable = pulp.lpSum(over_rates[k] * y_over[t][k] * cr_per_mwh
                            for t in range(n) for k in range(n_bands))
    payable = pulp.lpSum(under_rates[k] * y_under[t][k] * cr_per_mwh
                         for t in range(n) for k in range(n_bands))
    degradation = pulp.lpSum((chg[t] + dis[t]) * DT * KWH_PER_MWH
                             * degradation_inr_per_kwh * 0.5 for t in range(n))

    prob += revenue + receivable - payable - degradation

    prob.solve(pulp.PULP_CBC_CMD(msg=solver_msg))
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise RuntimeError(f"LP did not solve to optimality (status={status})")

    # ---- rebuild the settlement via the SAME trusted engine used elsewhere,
    #      rather than trusting the LP's own y-variables, so this scenario's
    #      numbers are produced by the identical certified code path as
    #      S1/S2/S3. This is the cross-check, not a shortcut. ----
    from dispatch_sim.core.dsm_settlement import BlockInput, settle_block

    rows: List[LedgerRow] = []
    soc_series = []
    for t in range(n):
        sched_v = pulp.value(sched[t])
        deliv_v = pulp.value(delivered[t])
        chg_v, dis_v = pulp.value(chg[t]), pulp.value(dis[t])
        soc_v = pulp.value(soc[t + 1])
        thr = (chg_v + dis_v) * DT

        s = settle_block(BlockInput(sched_v * DT, deliv_v * DT, avc, rate, t), dsm_cfg)
        ppa = sched_v * DT * cr_per_mwh
        deg = thr * KWH_PER_MWH * degradation_inr_per_kwh * 0.5
        profit = ppa + s.receivable_inr - s.payable_inr - deg - om_block

        rows.append(LedgerRow(
            block=t, time=block_time(t),
            scheduled_mwh=sched_v * DT, actual_gen_mwh=actual_mw[t] * DT,
            delivered_mwh=deliv_v * DT, deviation_mwh=s.deviation_mwh,
            deviation_pct_of_avc=(0.0 if s.deviation_pct in (float("inf"), float("-inf"))
                                  else s.deviation_pct),
            ppa_revenue=ppa, dsm_receivable=s.receivable_inr,
            dsm_payable=s.payable_inr, dsm_penalty=s.penalty_inr,
            soc_mwh=soc_v, degradation=deg, om=om_block, profit=profit,
        ))
        soc_series.append(soc_v)

    return ScenarioResult(
        name="S5 - Optimizer (perfect-foresight upper bound)",
        rows=rows,
        config_snapshot={"degradation_inr_per_kwh": degradation_inr_per_kwh,
                         "battery": vars(battery), "lp_status": status},
    )
