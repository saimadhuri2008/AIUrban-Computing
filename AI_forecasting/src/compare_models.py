#!/usr/bin/env python3
"""
compute_classical_metrics.py

Compute evaluation metrics for the statistical baseline (SARIMAX)
by comparing ward-level forecasts against ground truth.

Produces:
- Paper-ready city-level metrics table (1-year horizon)

NOTE:
- Evaluates SARIMAX only
- ETS/STL used only for diagnostics (not evaluated)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================
# PATHS
# ============================

BASE_DIR = Path("AI_forecasting")

FORECAST_DIR = BASE_DIR / "results/classical/aggregated"
GROUND_TRUTH = BASE_DIR / "data/input/timeseries/all_wards_monthly.parquet"

OUT_DIR = BASE_DIR / "results/paper_ready"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUT_DIR / "statistical_baseline_sarimax_summary.csv"

# ============================
# CONFIG
# ============================

TARGETS = [
    "electricity_demand",
    "water_demand",
    "congestion_index",
    "pm25"
]

MODEL_NAME = "Statistical Baseline (SARIMAX)"
TEST_HORIZON_MONTHS = 12

# ============================
# HELPERS
# ============================

def load_ground_truth():
    df = pd.read_parquet(GROUND_TRUTH)
    df["date"] = pd.to_datetime(
        dict(year=df.year, month=df.month, day=1)
    )
    return df


def load_forecasts(target):
    path = FORECAST_DIR / f"{target}_forecasts_all_wards.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing forecast file: {path}")
    return pd.read_csv(path, parse_dates=["date"])


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(
        np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-6, y_true))
    ) * 100
    return mae, rmse, mape


# ============================
# MAIN
# ============================

def main():
    print("\n🔹 Computing SARIMAX statistical baseline metrics...")

    gt = load_ground_truth()

    target_level_rows = []

    for target in TARGETS:
        fc = load_forecasts(target)

        merged = fc.merge(
            gt[["ward_id", "date", target]],
            on=["ward_id", "date"],
            how="inner"
        )

        merged = (
            merged
            .sort_values("date")
            .groupby("ward_id", as_index=False)
            .tail(TEST_HORIZON_MONTHS)
        )

        ward_metrics = []

        for ward, wdf in merged.groupby("ward_id"):
            mae, rmse, mape = compute_metrics(
                wdf[target].values,
                wdf["forecast"].values
            )
            ward_metrics.append({
                "mae": mae,
                "rmse": rmse,
                "mape_pct": mape
            })

        # aggregate across wards for this target
        ward_df = pd.DataFrame(ward_metrics)
        target_level_rows.append({
            "target": target,
            "mae": ward_df["mae"].mean(),
            "rmse": ward_df["rmse"].mean(),
            "mape_pct": ward_df["mape_pct"].mean()
        })

    # aggregate across targets → city-level
    target_df = pd.DataFrame(target_level_rows)

    city_summary = pd.DataFrame([{
        "model": MODEL_NAME,
        "mae": target_df["mae"].mean(),
        "rmse": target_df["rmse"].mean(),
        "mape_pct": target_df["mape_pct"].mean()
    }])

    city_summary.to_csv(OUTPUT_CSV, index=False)

    print("\n📊 Table 1 — Statistical Baseline Performance (1-Year Horizon)")
    print(city_summary.round(4).to_string(index=False))
    print("\n📁 Saved to:")
    print(f"  {OUTPUT_CSV.resolve()}")

    print("\n✅ SARIMAX baseline metrics ready for paper.")


if __name__ == "__main__":
    main()
