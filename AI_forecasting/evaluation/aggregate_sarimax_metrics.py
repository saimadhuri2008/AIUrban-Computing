#!/usr/bin/env python3
"""
aggregate_target_metrics.py

Aggregates ward-level SARIMAX metrics into target-level summaries.

Input :
AI_forecasting/results/classical/metrics/{target}_metrics.csv

Output :
AI_forecasting/results/classical/metrics/target_level_metrics.csv
"""

from pathlib import Path
import pandas as pd

# ---------------- CONFIG ----------------
METRICS_DIR = Path("AI_forecasting/evaluation/classical")
OUT_FILE = METRICS_DIR / "target_level_metrics.csv"

TARGETS = [
    "congestion_index",
    "electricity_demand",
    "pm25",
    "water_demand"
]
# ----------------------------------------

rows = []

for target in TARGETS:
    path = METRICS_DIR / f"{target}_metrics.csv"
    if not path.exists():
        continue

    df = pd.read_csv(path)

    # skip empty files safely
    if df.empty:
        continue

    rows.append({
        "target": target,
        "mean_mae": df["mae"].mean(),
        "median_mae": df["mae"].median(),
        "std_mae": df["mae"].std(),
        "mean_rmse": df["rmse"].mean(),
        "mean_mape": df["mape_pct"].mean()
    })

# write final table
pd.DataFrame(rows).to_csv(OUT_FILE, index=False)

print("Target-level metrics written to:", OUT_FILE)
