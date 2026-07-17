"""CLI — input files in, simulation xlsx out.

  python -m dispatch_sim.cli run s2 \
      --forecast sample_data/forecast.csv --actual sample_data/actual.csv \
      --battery sample_data/flexitwin_battery.json --out output/

  python -m dispatch_sim.cli run-all --forecast ... --actual ... --battery ... --out output/
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from dispatch_sim.io.loaders import (load_battery, load_dsm_config,
                                     load_series_csv, load_yaml)
from dispatch_sim.io.report import write_comparison_xlsx, write_scenario_xlsx
from dispatch_sim.runners.rules import RUNNERS

CFG_DIR = Path(__file__).parent / "config"


def _run_one(key: str, forecast, actual, plant, dsm_cfg, battery_path):
    scen = load_yaml(CFG_DIR / f"scenario_{key}.yaml")
    battery = None
    if scen.get("battery", {}).get("enabled"):
        if not battery_path:
            raise SystemExit(f"{scen['name']} needs --battery <flexitwin json>")
        battery = load_battery(battery_path)   # fresh battery per scenario
    return RUNNERS[key](forecast, actual, plant, dsm_cfg, scen, battery)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dispatch_sim")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "run-all"):
        s = sub.add_parser(name)
        if name == "run":
            s.add_argument("scenario", choices=list(RUNNERS))
        s.add_argument("--forecast", required=True)
        s.add_argument("--actual", required=True)
        s.add_argument("--battery")
        s.add_argument("--out", default="output")
    args = ap.parse_args(argv)

    forecast = load_series_csv(args.forecast)
    actual = load_series_csv(args.actual)
    plant = load_yaml(CFG_DIR / "plant.yaml")
    dsm_cfg = load_dsm_config(CFG_DIR / "dsm_bands.yaml")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    keys = list(RUNNERS) if args.cmd == "run-all" else [args.scenario]
    results = []
    for k in keys:
        res = _run_one(k, forecast, actual, plant, dsm_cfg, args.battery)
        path = out / f"{k}_result.xlsx"
        write_scenario_xlsx(res, path)
        results.append(res)
        print(f"{res.name:<22} profit ₹{res.total('profit'):>12,.0f}   "
              f"DSM penalty ₹{res.total('dsm_penalty'):>10,.0f}   -> {path}")

    if len(results) > 1:
        write_comparison_xlsx(results, out / "comparison.xlsx")
        if results[1].total("profit") >= results[0].total("profit"):
            print("CHECK: S2 did NOT come out worse than S1 on this day "
                  "(possible on low-error days) — review per plan.")
        print(f"comparison -> {out / 'comparison.xlsx'}")


if __name__ == "__main__":
    main()
