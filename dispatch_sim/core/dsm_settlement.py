"""
CERC DSM 2024 settlement engine — WS seller (wind/solar/hybrid).

Implements Regulation 6(2) and Regulation 8(4) of the CERC (Deviation
Settlement Mechanism and Related Matters) Regulations, 2024:

  Deviation (DWS, MWh)  = Actual injection - Scheduled generation
  Deviation (DWS, %)    = 100 * DWS_MWh / Available Capacity          [Reg 6(2)]

  Charges are NOT frequency-linked for WS sellers and are applied as
  MARGINAL slices across volume limits (VL), rates linked to the
  contract rate (CR):                                                  [Reg 8(4)]

                     over-injection        under-injection
                     (receivable)          (payable)
      VL1            100% of CR            100% of CR
      VL2             90% of CR            110% of CR
      VL3             50% of CR            150% of CR
      beyond VL3        0                  200% of CR

  Volume limits — solar / hybrid / pooling-station aggregation:
      VL1 <= 5% DWS, VL2 5-10%, VL3 10-20%, beyond > 20%
  Volume limits — wind:
      VL1 <= 10%, VL2 10-15%, VL3 15-25%, beyond > 25%

Design notes
------------
* Band tables are data, not code: the 2026 amendment (blended
  AvC/schedule denominator, tighter solar bands) drops in as a new
  ``BandTable`` / ``DenominatorPolicy`` without touching settlement logic.
* All money in INR; energy in MWh; contract rate in INR/kWh
  (converted internally: 1 MWh = 1000 kWh).
* ``penalty_inr`` is the economic loss versus a perfect-foresight
  benchmark (schedule == actual, all energy at CR). It satisfies the
  identity:  block_revenue == actual_mwh * CR - penalty.

Verify the default tables against the notified text (5 Aug 2024) and any
subsequent amendment orders before certifying settlement outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Sequence, Tuple

KWH_PER_MWH = 1000.0


class DenominatorPolicy(Enum):
    """Basis for expressing deviation as a percentage."""

    AVAILABLE_CAPACITY = "available_capacity"  # CERC DSM 2024, Reg 6(2)
    SCHEDULE = "schedule"                      # general-seller style
    BLENDED = "blended"                        # 2026 amendment: x% AvC + (100-x)% schedule


@dataclass(frozen=True)
class BandTable:
    """Marginal deviation bands and contract-rate multipliers.

    ``edges_pct`` are the ascending upper edges of the volume limits in
    percent deviation; the final band is open-ended.  ``over_rates`` /
    ``under_rates`` have exactly ``len(edges_pct) + 1`` entries and are
    expressed as fractions of the contract rate.
    """

    name: str
    edges_pct: Tuple[float, ...]
    over_rates: Tuple[float, ...]
    under_rates: Tuple[float, ...]

    def __post_init__(self) -> None:
        if list(self.edges_pct) != sorted(self.edges_pct) or any(e <= 0 for e in self.edges_pct):
            raise ValueError("edges_pct must be positive and ascending")
        if len(self.over_rates) != len(self.edges_pct) + 1:
            raise ValueError("over_rates must have len(edges_pct)+1 entries")
        if len(self.under_rates) != len(self.edges_pct) + 1:
            raise ValueError("under_rates must have len(edges_pct)+1 entries")


# Default tables per CERC DSM 2024, Reg 8(4).
SOLAR_2024 = BandTable(
    name="CERC-2024 WS seller (solar / hybrid / pooled)",
    edges_pct=(5.0, 10.0, 20.0),
    over_rates=(1.00, 0.90, 0.50, 0.00),
    under_rates=(1.00, 1.10, 1.50, 2.00),
)

WIND_2024 = BandTable(
    name="CERC-2024 WS seller (wind)",
    edges_pct=(10.0, 15.0, 25.0),
    over_rates=(1.00, 0.90, 0.50, 0.00),
    under_rates=(1.00, 1.10, 1.50, 2.00),
)


@dataclass(frozen=True)
class DsmConfig:
    bands: BandTable = SOLAR_2024
    denominator: DenominatorPolicy = DenominatorPolicy.AVAILABLE_CAPACITY
    blend_avc_pct: float = 50.0  # 'x' in the 2026 blended formula; only used for BLENDED


@dataclass(frozen=True)
class BlockInput:
    """One 15-minute time block. Energies in MWh (MW * 0.25 for a block)."""

    scheduled_mwh: float
    actual_mwh: float
    available_capacity_mwh: float
    contract_rate_inr_per_kwh: float
    block_index: int = 0


@dataclass(frozen=True)
class BandSlice:
    band: int            # 0 = VL1, 1 = VL2, 2 = VL3, 3 = beyond
    energy_mwh: float
    rate_fraction: float  # of contract rate
    amount_inr: float     # receivable (over) or payable (under) for this slice


@dataclass(frozen=True)
class BlockSettlement:
    block_index: int
    deviation_mwh: float          # signed: + over-injection, - under-injection
    deviation_pct: float          # signed, per denominator policy
    receivable_inr: float         # paid to seller for over-injection
    payable_inr: float            # paid by seller for under-injection
    penalty_inr: float            # economic loss vs perfect-foresight benchmark
    slices: Tuple[BandSlice, ...]

    @property
    def net_dsm_inr(self) -> float:
        """Net DSM cash flow to the seller (positive = receivable)."""
        return self.receivable_inr - self.payable_inr


@dataclass(frozen=True)
class DaySettlement:
    blocks: Tuple[BlockSettlement, ...]
    scheduled_mwh: float
    actual_mwh: float
    ppa_revenue_inr: float        # scheduled energy at contract rate
    receivable_inr: float
    payable_inr: float
    penalty_inr: float

    @property
    def total_revenue_inr(self) -> float:
        return self.ppa_revenue_inr + self.receivable_inr - self.payable_inr


def _denominator_mwh(block: BlockInput, config: DsmConfig) -> float:
    if config.denominator is DenominatorPolicy.AVAILABLE_CAPACITY:
        return block.available_capacity_mwh
    if config.denominator is DenominatorPolicy.SCHEDULE:
        return block.scheduled_mwh
    x = config.blend_avc_pct / 100.0
    return x * block.available_capacity_mwh + (1.0 - x) * block.scheduled_mwh


def _split_bands(dev_pct_abs: float, edges: Sequence[float]) -> List[Tuple[int, float]]:
    """Split an absolute %-deviation into marginal (band, pct_width) slices."""
    slices: List[Tuple[int, float]] = []
    lower = 0.0
    for i, edge in enumerate(edges):
        if dev_pct_abs <= lower:
            break
        width = min(dev_pct_abs, edge) - lower
        if width > 0:
            slices.append((i, width))
        lower = edge
    if dev_pct_abs > lower:
        slices.append((len(edges), dev_pct_abs - lower))
    return slices


def settle_block(block: BlockInput, config: DsmConfig = DsmConfig()) -> BlockSettlement:
    """Settle a single time block per CERC DSM 2024 WS-seller rules."""
    if block.contract_rate_inr_per_kwh < 0:
        raise ValueError("contract rate must be non-negative")

    dev_mwh = block.actual_mwh - block.scheduled_mwh
    cr_per_mwh = block.contract_rate_inr_per_kwh * KWH_PER_MWH
    denom = _denominator_mwh(block, config)

    if abs(dev_mwh) < 1e-12:
        return BlockSettlement(block.block_index, 0.0, 0.0, 0.0, 0.0, 0.0, ())

    over = dev_mwh > 0
    rates = config.bands.over_rates if over else config.bands.under_rates

    if denom <= 1e-12:
        # Degenerate block (e.g. AvC reported zero while deviation exists):
        # conservatively settle the full deviation in the worst band.
        pct_slices = [(len(config.bands.edges_pct), float("inf"))]
        dev_pct = float("inf") if over else float("-inf")
        slice_energies = [(len(config.bands.edges_pct), abs(dev_mwh))]
    else:
        dev_pct_abs = 100.0 * abs(dev_mwh) / denom
        dev_pct = dev_pct_abs if over else -dev_pct_abs
        pct_slices = _split_bands(dev_pct_abs, config.bands.edges_pct)
        slice_energies = [(band, width / 100.0 * denom) for band, width in pct_slices]

    slices: List[BandSlice] = []
    amount = 0.0
    penalty = 0.0
    for band, energy in slice_energies:
        rate = rates[band]
        inr = energy * rate * cr_per_mwh
        amount += inr
        # Loss vs perfect foresight: over-injection is paid below CR,
        # under-injection is charged above the simple CR refund.
        penalty += energy * ((1.0 - rate) if over else (rate - 1.0)) * cr_per_mwh
        slices.append(BandSlice(band, energy, rate, inr))

    return BlockSettlement(
        block_index=block.block_index,
        deviation_mwh=dev_mwh,
        deviation_pct=dev_pct,
        receivable_inr=amount if over else 0.0,
        payable_inr=0.0 if over else amount,
        penalty_inr=penalty,
        slices=tuple(slices),
    )


def settle_day(blocks: Iterable[BlockInput], config: DsmConfig = DsmConfig()) -> DaySettlement:
    """Settle a sequence of time blocks and aggregate."""
    settled: List[BlockSettlement] = []
    sched = actual = ppa = recv = pay = pen = 0.0
    for b in blocks:
        s = settle_block(b, config)
        settled.append(s)
        sched += b.scheduled_mwh
        actual += b.actual_mwh
        ppa += b.scheduled_mwh * b.contract_rate_inr_per_kwh * KWH_PER_MWH
        recv += s.receivable_inr
        pay += s.payable_inr
        pen += s.penalty_inr
    return DaySettlement(tuple(settled), sched, actual, ppa, recv, pay, pen)


if __name__ == "__main__":
    # Sample settlement statement: 10 MW plant, CR = Rs 2.60/kWh, one block each way.
    cfg = DsmConfig(bands=SOLAR_2024)
    demo = [
        BlockInput(2.00, 2.00, 2.5, 2.60, 0),   # on schedule
        BlockInput(2.00, 2.10, 2.5, 2.60, 1),   # +4% over  (VL1)
        BlockInput(2.00, 2.50, 2.5, 2.60, 2),   # +20% over (VL1-VL3)
        BlockInput(2.50, 1.75, 2.5, 2.60, 3),   # -30% under (through all bands)
    ]
    day = settle_day(demo, cfg)
    print(f"{'blk':>3} {'dev MWh':>8} {'dev %':>7} {'recv Rs':>9} {'pay Rs':>9} {'penalty':>9}")
    for s in day.blocks:
        print(f"{s.block_index:>3} {s.deviation_mwh:>8.3f} {s.deviation_pct:>7.1f}"
              f" {s.receivable_inr:>9.1f} {s.payable_inr:>9.1f} {s.penalty_inr:>9.1f}")
    print(f"\nPPA revenue  Rs {day.ppa_revenue_inr:,.1f}")
    print(f"DSM net      Rs {day.receivable_inr - day.payable_inr:,.1f}")
    print(f"Penalty      Rs {day.penalty_inr:,.1f}")
    print(f"Total        Rs {day.total_revenue_inr:,.1f}")
