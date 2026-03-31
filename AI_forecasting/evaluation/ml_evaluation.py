#!/usr/bin/env python3
"""
ml_forecast_eval_real_units.py

RESEARCH-GRADE ML EVALUATION
- Targets are in ORIGINAL UNITS
- No target scaling
- Metrics are directly comparable to SARIMAX / LSTM / TFT / N-BEATS
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

# ---------------- CONFIG ----------------
DATA_PATH = Path("AI_forecasting/data/input/timeseries/all_wards_monthly.parquet")
OUT_DIR = Path("AI_forecasting/evaluation/ml/evaluation_real_units")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "electricity_demand",
    "water_demand",
    "congestion_index",
    "pm25",
]

TRAIN_END = pd.Timestamp("2024-12-01")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-12-01")
# --------------------------------------


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred):
    y_true = np.array(y_true)
    denom = np.where(y_true == 0, 1e-6, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def make_features(df, target):
    X = pd.DataFrame(index=df.index)

    # lags
    for lag in [1, 2, 3, 6, 12]:
        X[f"{target}_lag{lag}"] = df[target].shift(lag)

    # rolling means
    for w in [3, 6, 12]:
        X[f"{target}_rollmean_{w}"] = df[target].shift(1).rolling(w).mean()

    # seasonality
    X["month"] = df.index.month
    X["month_sin"] = np.sin(2 * np.pi * (X["month"] - 1) / 12)
    X["month_cos"] = np.cos(2 * np.pi * (X["month"] - 1) / 12)

    return X


# ---------------- LOAD DATA ----------------
df = pd.read_parquet(DATA_PATH)
# ---------------------------
# SAFE DATE CONSTRUCTION
# ---------------------------
if {"year", "month"}.issubset(df.columns):
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" +
        df["month"].astype(str).str.zfill(2) + "-01"
    )
else:
    raise ValueError("Expected 'year' and 'month' columns for date construction.")

# ensure sorting
df = df.sort_values(["ward_id", "date"]).reset_index(drop=True)
# ---------------------------

df["ward_id"] = df["ward_id"].astype(str)

rows = []

# ---------------- EVALUATION ----------------
for target in TARGETS:
    print(f"Evaluating ML (real units) for target: {target}")

    for ward, g in df.groupby("ward_id"):
        g = g.sort_values("date").set_index("date")

        train = g[g.index <= TRAIN_END]
        test = g[(g.index >= TEST_START) & (g.index <= TEST_END)]

        if len(train) < 24 or len(test) < 12:
            continue

        X_train = make_features(train, target)
        y_train = train[target]

        data_train = X_train.join(y_train).dropna()
        if data_train.empty:
            continue

        X_tr = data_train.drop(columns=[target])
        y_tr = data_train[target]

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=1
        )
        model.fit(X_tr, y_tr)

        # -------- TEST --------
        full = pd.concat([train, test])
        X_all = make_features(full, target)

        X_test = X_all.loc[test.index].dropna()
        if X_test.empty:
            continue

        y_pred = model.predict(X_test)
        y_true = test.loc[X_test.index, target]

        rows.append({
            "ward_id": ward,
            "target": target,
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "mape": mape(y_true, y_pred),
        })


# ---------------- AGGREGATE ----------------
df_metrics = pd.DataFrame(rows)

summary = df_metrics.groupby("target").agg(
    mean_mae=("mae", "mean"),
    median_mae=("mae", "median"),
    std_mae=("mae", "std"),
    mean_rmse=("rmse", "mean"),
    mean_mape=("mape", "mean"),
).reset_index()

summary.to_csv(OUT_DIR / "ml_2025_metrics_real_units.csv", index=False)
df_metrics.to_csv(OUT_DIR / "ml_2025_metrics_per_ward.csv", index=False)

print("✅ ML evaluation in real units complete")
