#!/usr/bin/env python3
"""
sarimax_eval_2025.py

ONE-SHOT HOLDOUT EVALUATION
Train: 2014-01 → 2024-12
Test : 2025-01 → 2025-12

Outputs:
- MAE / RMSE / MAPE per ward × target
"""

from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
INPUT = Path("AI_forecasting/data/input/timeseries/all_wards_monthly.parquet")
OUT_DIR = Path("AI_forecasting/evaluation/classical")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["electricity_demand", "water_demand", "congestion_index", "pm25"]
EXOG = ["rainfall", "temperature", "job_density"]

TRAIN_END = pd.Timestamp("2024-12-01")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-12-01")

ORDER = (1, 1, 1)
SEASONAL = (0, 1, 1, 12)
# ----------------------------------------

df = pd.read_parquet(INPUT)
df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))

for target in TARGETS:
    rows = []

    for ward, wdf in df.groupby("ward_id"):
        wdf = wdf.sort_values("date").set_index("date")

        train = wdf.loc[:TRAIN_END]
        test = wdf.loc[TEST_START:TEST_END]

        if len(train) < 36 or len(test) < 12:
            continue

        y_train = train[target].astype(float)
        y_test = test[target].astype(float)

        # skip numerically degenerate series
        if y_train.var() < 1e-6:
            continue

        X_train = train[EXOG] if all(c in train.columns for c in EXOG) else None
        X_test = test[EXOG] if X_train is not None else None

        try:
            model = SARIMAX(
                y_train,
                exog=X_train,
                order=ORDER,
                seasonal_order=SEASONAL,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            # 🔑 CRITICAL: disable covariance / Hessian completely
            res = model.fit(
                method="powell",
                maxiter=80,
                disp=False,
                cov_type="none",     # ← THIS FIXES YOUR CRASH
                low_memory=True
            )

            forecast = res.forecast(steps=len(y_test), exog=X_test)

            rows.append({
                "ward_id": ward,
                "target": target,
                "mae": mean_absolute_error(y_test, forecast),
                "rmse": np.sqrt(mean_squared_error(y_test, forecast)),
                "mape_pct": np.mean(
                    np.abs((y_test.values - forecast.values) /
                           np.where(y_test.values == 0, 1e-6, y_test.values))
                ) * 100
            })

        except Exception:
            continue

    pd.DataFrame(rows).to_csv(
        OUT_DIR / f"{target}_metrics.csv",
        index=False
    )
