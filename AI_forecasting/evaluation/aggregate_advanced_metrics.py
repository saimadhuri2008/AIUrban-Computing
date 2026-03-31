#!/usr/bin/env python3
"""
aggregate_tft_nbeats_metrics.py

Aggregates rolling backtest metrics for TFT + N-BEATS
into a single, paper-ready CSV.

Expected input:
results/backtest_tft_nbeats/
 ├─ ward_<id>_<target>_tft_nbeats_backtest_metrics.json
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

BASE_DIR = Path("AI_forecasting")
METRICS_DIR = BASE_DIR / "evaluation/advanced"

METRICS_DIR.mkdir(parents=True,exist_ok=True)


# ---------------- CONFIG ----------------
BACKTEST_DIR = BASE_DIR / "evaluation/advanced/metrics"
OUT_FILE = METRICS_DIR / "tft_nbeats_aggregated_metrics.csv"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
# --------------------------------------


rows = []

for f in BACKTEST_DIR.glob("*_tft_nbeats_backtest_metrics.json"):
    with open(f, "r") as fh:
        data = json.load(fh)

    overall = data.get("overall", {})
    if not overall:
        continue

    # infer target name from filename
    # pattern: ward_<id>_<target>_tft_nbeats_backtest_metrics.json
    parts = f.stem.split("_")
    target = parts[2] if len(parts) >= 3 else "unknown"

    rows.append({
        "target": target,
        "MAE": overall.get("MAE", np.nan),
        "RMSE": overall.get("RMSE", np.nan),
        "MAPE": overall.get("MAPE", np.nan),
        "n_obs": overall.get("n_obs", 0),
    })

df = pd.DataFrame(rows)

# -------- Aggregate per target --------
summary = (
    df.groupby("target", as_index=False)
      .agg(
          mean_MAE=("MAE", "mean"),
          std_MAE=("MAE", "std"),
          mean_RMSE=("RMSE", "mean"),
          std_RMSE=("RMSE", "std"),
          mean_MAPE=("MAPE", "mean"),
          n_obs=("n_obs", "sum"),
      )
      .sort_values("mean_RMSE")
)

summary.to_csv(OUT_FILE, index=False)

print("✅ Aggregated TFT + N-BEATS metrics saved to:")
print(OUT_FILE)
print("\nPreview:")
print(summary)
