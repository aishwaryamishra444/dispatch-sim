# dispatch_sim - Solar-BESS Baseline Scenarios 1-3

Input files in → simulation → xlsx out. One engine, scenarios as YAML configs,
no solver (per the approved baseline plan; Scenario 5/Pyomo plugs in later).

## Run it
```bash
pip install pyyaml numpy openpyxl pytest
pytest dispatch_sim/tests/ -q                     # 20 settlement tests

python -m dispatch_sim.cli run-all \
    --forecast sample_data/forecast_2026-06-14.csv \
    --actual   sample_data/actual_2026-06-14.csv \
    --battery  sample_data/flexitwin_battery.json \
    --out output/
```

## Inputs (swap sample_data for real feeds)
- forecast CSV / actual CSV - 96 rows, columns `time,mw` (from the forecasting
  module and plant telemetry)
- flexitwin JSON — batteryUsableCapacity, cRateMW, roundTripEfficiency, SoC limits
- dispatch_sim/config/*.yaml — PPA rate, O&M, CERC DSM bands, scenario knobs
  (every placeholder is marked NEEDS-CONFIRMATION)

## Outputs (per the team plan)
- `s1|s2|s3_result.xlsx` — sheet "blocks": 96 rows x [time, scheduled_mwh,
  actual_gen_mwh, delivered_mwh, deviation, dsm receivable/payable/penalty,
  soc, degradation, om, profit]; sheet "summary": live SUM formulas + config
  audit snapshot
- `comparison.xlsx` — three scenarios side by side, profit as live formulas,
  with the S2-worse-than-S1 check flag

## Design decisions implemented (configurable)
- D1: S2 schedule revised to forecast x RTE (`revise_schedule_for_rte`)
- D2: S3 schedule-integrated windows (`schedule_integrated`)
- D3: degradation per kWh throughput x0.5 (`degradation_inr_per_kwh`)
- D4: O&M fixed per day (`om_inr_per_day`)
