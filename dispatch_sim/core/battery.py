"""Battery model — SoC bookkeeping with C-rate, SoC limits, and one-way
efficiency sqrt(RTE). Tracks total throughput (MWh moved in either direction)
for degradation costing (design decision D3: cost per kWh throughput, x0.5)."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

DT = 0.25  # hours per 15-min block


@dataclass
class Battery:
    usable_capacity_mwh: float
    c_rate_mw: float
    rte: float
    soc_min_pct: float = 10.0
    soc_max_pct: float = 90.0
    soc_mwh: float = field(default=None)
    throughput_mwh: float = 0.0

    def __post_init__(self):
        self.eta = math.sqrt(self.rte)  # one-way efficiency
        self.soc_floor = self.usable_capacity_mwh * self.soc_min_pct / 100.0
        self.soc_ceil = self.usable_capacity_mwh * self.soc_max_pct / 100.0
        if self.soc_mwh is None:
            self.soc_mwh = self.soc_floor

    # ---- capability queries (MW for one block) ----
    def max_charge_mw(self) -> float:
        headroom = max(0.0, self.soc_ceil - self.soc_mwh)
        return min(self.c_rate_mw, headroom / DT / self.eta)

    def max_discharge_mw(self) -> float:
        """Max deliverable MW (after efficiency) for one block."""
        available = max(0.0, self.soc_mwh - self.soc_floor)
        return min(self.c_rate_mw, available / DT * self.eta)

    # ---- actions; return what was actually accepted / delivered ----
    def charge(self, mw: float) -> float:
        mw = max(0.0, min(mw, self.max_charge_mw()))
        self.soc_mwh += mw * self.eta * DT
        self.throughput_mwh += mw * DT
        return mw

    def discharge(self, mw_delivered: float) -> float:
        mw_delivered = max(0.0, min(mw_delivered, self.max_discharge_mw()))
        drawn = mw_delivered / self.eta
        self.soc_mwh -= drawn * DT
        self.throughput_mwh += drawn * DT
        return mw_delivered
