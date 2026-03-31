#!/usr/bin/env python3
"""
ets_stl_forecast.py

City-level ETS (Holt-Winters) forecasting with STL diagnostics
for rainfall and temperature.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL

import logging
from datetime import datetime
import json

# ============================
# LOGGING
# ============================
LOG_DIR = Path("AI_forecasting/logs/forecasting/classical")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "ets_stl_run.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ============================
# PATHS
# ============================
BASE_DIR = Path("AI_forecasting")

INPUT_PARQUET = BASE_DIR / "data/input/timeseries/all_wards_monthly.parquet"

RESULTS_DIR = BASE_DIR / "results/classical"
FORECASTS_DIR = RESULTS_DIR / "forecasts"
AGG_DIR = RESULTS_DIR / "aggregated"

REPORTS_DIR = BASE_DIR / "reports/classical"
FIGURES_DIR = REPORTS_DIR / "figures"
SUMMARY_DIR = REPORTS_DIR / "summary"

for p in [FORECASTS_DIR, AGG_DIR, FIGURES_DIR, SUMMARY_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ============================
# CONFIG
# ============================
TARGETS = ["rainfall", "temperature"]
FORECAST_HORIZON = 12
MIN_POINTS = 36
SEASONAL_PERIOD = 12

# ============================
# CORE
# ============================
def fit_ets(series, horizon):
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=SEASONAL_PERIOD
    )
    res = model.fit(optimized=True)
    fc = res.forecast(horizon)
    return res, fc

# ============================
# MAIN
# ============================
def main():
    start_time = datetime.utcnow()
    logger.info("START ETS/STL CITY-LEVEL RUN")

    df = pd.read_parquet(INPUT_PARQUET)
    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    df = df.sort_values("date")

    all_forecasts = []

    for t in TARGETS:
        if t not in df.columns:
            logger.warning(f"{t} not found — skipping")
            continue

        # ✅ THIS IS WHERE YOUR BLOCK GOES
        ts = (
            df.groupby("date")[t]
              .mean()
              .dropna()
        )
        ts = ts.asfreq("MS")
        if len(ts) < MIN_POINTS:
            logger.warning(f"Not enough data for {t}")
            continue

        # STL
        stl = STL(ts, period=SEASONAL_PERIOD, robust=False).fit()
        fig = stl.plot()
        fig.set_size_inches(10, 6)
        fig.suptitle(f"STL Decomposition – City-level {t}", fontsize=10)
        fig.savefig(FIGURES_DIR / f"{t}_stl_components.png", dpi=150)
        plt.close(fig)

        # ETS
        _, fc = fit_ets(ts, FORECAST_HORIZON)

        future_idx = pd.date_range(
            start=ts.index[-1] + pd.offsets.MonthBegin(1),
            periods=FORECAST_HORIZON,
            freq="MS"
        )

        out = pd.DataFrame({
            "date": future_idx,
            "forecast": fc.values,
            "target": t
        })

        out.to_csv(FORECASTS_DIR / f"{t}_ets_forecast.csv", index=False)
        all_forecasts.append(out)

        logger.info(f"Completed ETS/STL for {t}")

    # Aggregated artifact
    if all_forecasts:
        pd.concat(all_forecasts, ignore_index=True).to_csv(
            AGG_DIR / "ets_city_level_forecasts.csv",
            index=False
        )

    # Metadata
    run_meta = {
        "method": "City-level ETS (Holt-Winters) with STL diagnostics",
        "targets": TARGETS,
        "forecast_horizon_months": FORECAST_HORIZON,
        "seasonal_period": SEASONAL_PERIOD,
        "aggregation": "mean over wards",
        "min_training_points": MIN_POINTS,
        "uncertainty": "point forecasts only",
        "input_data": str(INPUT_PARQUET),
        "generated_at_utc": datetime.utcnow().isoformat()
    }

    with open(SUMMARY_DIR / "ets_stl_run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    duration = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"END RUN | duration_sec={duration:.1f}")

# ============================
if __name__ == "__main__":
    main()
