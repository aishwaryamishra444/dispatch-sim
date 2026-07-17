"""Scenario 5 (research track) — RL dispatch environment.

A gymnasium.Env where the agent controls the battery each 15-min block and
the reward is the block profit computed by the SAME certified engine as
Scenarios 1-3 (core.dsm_settlement + core.battery). Nothing about the
economics is reimplemented for RL — the environment IS the product's ledger,
so a trained policy is automatically evaluated under real CERC DSM 2024
rules and is directly comparable to S1/S2/S3 in the comparison workbook.

Design notes
------------
* Episode = one day = 96 steps. reset() draws a new synthetic day
  (or replays a fixed CSV pair for evaluation).
* The day-ahead schedule is fixed at reset from the forecast (as in S1);
  scheduling itself can become part of the action space in a later phase.
* Action: Box(-1, 1) — battery power command as a fraction of C-rate.
  a > 0 discharge (adds to grid injection), a < 0 charge (diverts generation).
  Physics (SoC limits, C-rate, sqrt(RTE)) clip infeasible commands — the agent
  cannot cheat the battery.
* Observation (all normalized): [t/96, actual_gen/plant, soc fraction,
  schedule[t]/plant, forecast lookahead of H blocks /plant].
* Reward: block profit in thousand-INR (ppa + dsm_recv - dsm_pay
  - degradation - om/96), computed via settle_block.
* pymgrid note: this env is framework-free on purpose (economics stay
  auditable). If the team wants pymgrid's module ecosystem later, its
  modules can replace _sample_day / Battery while the reward path —
  settle_block + ledger math — stays exactly this.

Compatible with stable-baselines3 / CleanRL out of the box.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError("Scenario 5 env needs gymnasium: pip install gymnasium") from e

from dispatch_sim.core.battery import Battery, DT
from dispatch_sim.core.dsm_settlement import BlockInput, DsmConfig, settle_block

BLOCKS = 96


def _sample_day(rng: np.random.Generator, plant_mw: float, err_pct: float):
    """Synthetic (forecast, actual) day — same shape family as sample_data."""
    h = np.arange(BLOCKS) * DT
    day = (h > 6.25) & (h < 18.25)
    fc = np.zeros(BLOCKS)
    fc[day] = plant_mw * np.sin(np.pi * (h[day] - 6.25) / 12.0) ** 1.35
    p1, p2 = rng.uniform(0, 2 * np.pi, 2)
    err = (0.6 * np.sin(rng.uniform(2, 5) * h + p1)
           + 0.4 * np.sin(rng.uniform(5, 10) * h + p2)) * (err_pct / 100) * 1.6
    act = fc * (1 + err)
    for _ in range(rng.integers(1, 4)):
        c, w, d = rng.uniform(9, 16), rng.uniform(0.4, 1.6), rng.uniform(0.25, 0.7)
        act *= 1 - d * np.exp(-((h - c) ** 2) / (2 * w * w))
    return fc, np.clip(act, 0, plant_mw)


class DsmDispatchEnv(gym.Env):
    """RL battery dispatch under CERC DSM 2024 for a solar IPP."""

    metadata = {"render_modes": []}

    def __init__(self, plant: dict, dsm_cfg: DsmConfig, battery_spec: dict,
                 degradation_inr_per_kwh: float = 2.5, err_pct: float = 12.0,
                 lookahead: int = 4, fixed_day=None, seed: Optional[int] = None):
        super().__init__()
        self.plant = plant
        self.dsm_cfg = dsm_cfg
        self.battery_spec = battery_spec
        self.deg = degradation_inr_per_kwh
        self.err_pct = err_pct
        self.H = lookahead
        self.fixed_day = fixed_day        # (forecast, actual) arrays for eval
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(0.0, 1.5, shape=(4 + self.H,),
                                            dtype=np.float32)

    # ------------------------------------------------------------------
    def _obs(self):
        P = self.plant["plant_mw"]
        look = [self.forecast[min(self.t + k, BLOCKS - 1)] / P
                for k in range(1, self.H + 1)]
        soc_frac = ((self.batt.soc_mwh - self.batt.soc_floor)
                    / max(1e-9, self.batt.soc_ceil - self.batt.soc_floor))
        return np.array([self.t / BLOCKS, self.actual[self.t] / P, soc_frac,
                         self.sched[self.t] / P, *look], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.fixed_day is not None:
            self.forecast, self.actual = map(np.asarray, self.fixed_day)
        else:
            self.forecast, self.actual = _sample_day(
                self.rng, self.plant["plant_mw"], self.err_pct)
        self.sched = self.forecast.copy()      # day-ahead schedule (as S1)
        self.batt = Battery(
            usable_capacity_mwh=self.battery_spec["batteryUsableCapacity"],
            c_rate_mw=self.battery_spec["cRateMW"],
            rte=self.battery_spec["roundTripEfficiency"],
            soc_min_pct=self.battery_spec.get("socMinPct", 10),
            soc_max_pct=self.battery_spec.get("socMaxPct", 90),
        )
        self.t = 0
        self.day_profit = 0.0
        return self._obs(), {}

    def step(self, action):
        a = float(np.clip(action[0], -1.0, 1.0))
        gen = float(self.actual[self.t])
        thr0 = self.batt.throughput_mwh

        if a < 0:   # charge from generation (cannot charge from grid in v1)
            charged = self.batt.charge(min(-a * self.batt.c_rate_mw, gen))
            delivered = gen - charged
        else:       # discharge adds to injection
            delivered = gen + self.batt.discharge(a * self.batt.c_rate_mw)

        # ---- settle this block with the certified engine ----
        rate = self.plant["ppa_rate_inr_per_kwh"]
        avc = self.plant["plant_mw"] * DT
        s = settle_block(BlockInput(self.sched[self.t] * DT, delivered * DT,
                                    avc, rate, self.t), self.dsm_cfg)
        ppa = self.sched[self.t] * DT * rate * 1000.0
        deg = (self.batt.throughput_mwh - thr0) * 1000.0 * self.deg * 0.5
        om = self.plant["om_inr_per_day"] / BLOCKS
        profit = ppa + s.receivable_inr - s.payable_inr - deg - om
        self.day_profit += profit

        self.t += 1
        terminated = self.t >= BLOCKS
        info = {"profit_inr": profit, "dsm_penalty_inr": s.penalty_inr,
                "delivered_mw": delivered, "soc_mwh": self.batt.soc_mwh,
                "day_profit_inr": self.day_profit}
        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._obs()
        return obs, profit / 1000.0, terminated, False, info
