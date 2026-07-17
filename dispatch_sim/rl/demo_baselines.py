"""Sanity demo: run the S5 env with three hand policies over 20 random days.
Any RL agent must beat these to justify itself. (No training here — this
verifies the env API and produces the baseline table.)"""
from pathlib import Path

import numpy as np

from dispatch_sim.io.loaders import load_dsm_config, load_yaml
from dispatch_sim.rl.env import DsmDispatchEnv

CFG = Path(__file__).resolve().parents[1] / "config"
plant = load_yaml(CFG / "plant.yaml")
dsm = load_dsm_config(CFG / "dsm_bands.yaml")
spec = {"batteryUsableCapacity": 40, "cRateMW": 20,
        "roundTripEfficiency": 0.88, "socMinPct": 10, "socMaxPct": 90}


def run(policy, episodes=20, seed0=100):
    env = DsmDispatchEnv(plant, dsm, spec)
    totals = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done, info = False, {}
        while not done:
            obs, r, done, _, info = env.step(policy(obs, env))
        totals.append(info["day_profit_inr"])
    return float(np.mean(totals)), float(np.std(totals))


def no_battery(obs, env):
    """Do nothing — equivalent to Scenario 1."""
    return np.array([0.0], dtype=np.float32)


def random_policy(obs, env):
    return env.action_space.sample()


def buffer_policy(obs, env):
    """Charge on surplus vs schedule, discharge on shortfall (rule baseline)."""
    gen = obs[1] * plant["plant_mw"]
    sched = obs[3] * plant["plant_mw"]
    dev = gen - sched                                # + surplus, - shortfall
    a = np.clip(-dev / spec["cRateMW"], -1.0, 1.0)   # surplus -> charge (a<0)
    return np.array([a], dtype=np.float32)


if __name__ == "__main__":
    print(f"{'policy':<22}{'mean profit/day':>18}{'std':>12}")
    for name, pol in [("no-battery (~S1)", no_battery),
                      ("random", random_policy),
                      ("buffer heuristic", buffer_policy)]:
        m, s = run(pol)
        print(f"{name:<22}{'₹{:,.0f}'.format(m):>18}{'±{:,.0f}'.format(s):>12}")
