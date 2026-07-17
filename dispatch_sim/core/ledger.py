"""Ledger — turns (schedule, delivered) series into per-block P&L rows.

Profit per the team plan:
  S1: generation x PPA - DSM penalty - O&M
  S2: grid supply x PPA - DSM - O&M - degradation
  S3: direct sales + timed-discharge sales - DSM - O&M - degradation

All monetary lines are INR; energy MWh. DSM settlement is delegated to
core.dsm_settlement (CERC DSM 2024 WS-seller, Reg 6(2) + Reg 8(4))."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from dispatch_sim.core.dsm_settlement import BlockInput, DsmConfig, settle_day

DT = 0.25
BLOCKS = 96


@dataclass
class LedgerRow:
    block: int
    time: str
    scheduled_mwh: float
    actual_gen_mwh: float
    delivered_mwh: float
    deviation_mwh: float
    deviation_pct_of_avc: float
    ppa_revenue: float
    dsm_receivable: float
    dsm_payable: float
    dsm_penalty: float
    soc_mwh: Optional[float]
    degradation: float
    om: float
    profit: float


@dataclass
class ScenarioResult:
    name: str
    rows: List[LedgerRow]
    config_snapshot: dict

    def total(self, attr: str) -> float:
        return sum(getattr(r, attr) for r in self.rows)


def block_time(i: int) -> str:
    h = i * DT
    return f"{int(h):02d}:{int(h % 1 * 60):02d}"


def build_ledger(name: str, schedule_mw: Sequence[float],
                 actual_gen_mw: Sequence[float], delivered_mw: Sequence[float],
                 soc_mwh: Optional[Sequence[float]],
                 block_throughput_mwh: Sequence[float],
                 plant: dict, dsm_cfg: DsmConfig,
                 degradation_inr_per_kwh: float,
                 config_snapshot: dict) -> ScenarioResult:
    avc = plant["plant_mw"] * DT
    rate = plant["ppa_rate_inr_per_kwh"]
    om_block = plant["om_inr_per_day"] / BLOCKS

    inputs = [BlockInput(schedule_mw[t] * DT, delivered_mw[t] * DT, avc, rate, t)
              for t in range(BLOCKS)]
    day = settle_day(inputs, dsm_cfg)

    rows: List[LedgerRow] = []
    for t, s in enumerate(day.blocks):
        ppa = schedule_mw[t] * DT * rate * 1000.0
        deg = block_throughput_mwh[t] * 1000.0 * degradation_inr_per_kwh * 0.5
        profit = ppa + s.receivable_inr - s.payable_inr - deg - om_block
        rows.append(LedgerRow(
            block=t, time=block_time(t),
            scheduled_mwh=schedule_mw[t] * DT,
            actual_gen_mwh=actual_gen_mw[t] * DT,
            delivered_mwh=delivered_mw[t] * DT,
            deviation_mwh=s.deviation_mwh,
            deviation_pct_of_avc=(0.0 if s.deviation_pct in (float("inf"), float("-inf"))
                                  else s.deviation_pct),
            ppa_revenue=ppa,
            dsm_receivable=s.receivable_inr,
            dsm_payable=s.payable_inr,
            dsm_penalty=s.penalty_inr,
            soc_mwh=None if soc_mwh is None else soc_mwh[t],
            degradation=deg, om=om_block, profit=profit,
        ))
    return ScenarioResult(name=name, rows=rows, config_snapshot=config_snapshot)
