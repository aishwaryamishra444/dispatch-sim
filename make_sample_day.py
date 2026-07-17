"""Generate sample input files for one day (the 'input' side of the product).
Swap these for real forecast-module output and plant telemetry later."""
import csv, json
import numpy as np

B, DT, PLANT = 96, 0.25, 10.0
rng = np.random.default_rng(20260614)
h = np.arange(B) * DT
day = (h > 6.25) & (h < 18.25)
fc = np.zeros(B); fc[day] = PLANT * np.sin(np.pi * (h[day] - 6.25) / 12.0) ** 1.35
p1, p2 = rng.uniform(0, 2*np.pi, 2)
err = (0.6*np.sin(3.1*h+p1) + 0.4*np.sin(7.3*h+p2)) * 0.12 * 1.6
act = fc * (1 + err)
for _ in range(2):
    c, w, d = rng.uniform(9, 16), rng.uniform(0.5, 1.4), rng.uniform(0.3, 0.6)
    act *= 1 - d*np.exp(-((h-c)**2)/(2*w*w))
act = np.clip(act, 0, PLANT)

def write(path, series):
    with open(path, "w", newline="") as f:
        wtr = csv.writer(f); wtr.writerow(["time", "mw"])
        for t in range(B):
            wtr.writerow([f"{int(h[t]):02d}:{int(h[t]%1*60):02d}", f"{series[t]:.4f}"])

write("sample_data/forecast_2026-06-14.csv", fc)
write("sample_data/actual_2026-06-14.csv", act)
json.dump({"batteryUsableCapacity": 40, "cRateMW": 20,
           "roundTripEfficiency": 0.88, "socMinPct": 10, "socMaxPct": 90},
          open("sample_data/flexitwin_battery.json", "w"), indent=2)
print("sample inputs written")
