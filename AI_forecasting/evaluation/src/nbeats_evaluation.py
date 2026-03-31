#!/usr/bin/env python3
"""
N-BEATS Direct Multi-Horizon Evaluation (Last-Origin)
Leakage-safe, horizon-correct, reviewer-proof
Comparable with SARIMAX / ML / LSTM (direct mode)
"""

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from neuralforecast.losses.pytorch import MAE
import pandas as pd
import numpy as np
from common import *

df = pd.read_parquet(DATA_PATH)
df["ds"] = pd.to_datetime(df["date"], unit="ms")
df["unique_id"] = df["ward_id"]

rows = []

for target in TARGETS:
    print(f"\nEvaluating N-BEATS (last-origin): {target}")

    data = (
        df[["unique_id", "ds", target]]
        .rename(columns={target: "y"})
        .dropna()
        .sort_values(["unique_id", "ds"])
    )

    model = NBEATS(
        h=MAX_H,
        input_size=LOOKBACK,
        max_steps=2000,
        loss=MAE(),
        random_seed=SEED,
    )

    nf = NeuralForecast(models=[model], freq="M")
    nf.fit(data)

    forecasts = nf.predict()
    forecasts = forecasts.rename(columns={"NBEATS": "yhat"})

    # horizon index = forecast step
    forecasts["horizon"] = forecasts.groupby("unique_id").cumcount() + 1

    for H in HORIZONS:
        f_h = forecasts[forecasts["horizon"] == H]

        # true future values
        y_true = (
            data.groupby("unique_id")
            .tail(MAX_H)
            .groupby("unique_id")
            .nth(H - 1)
            .reset_index()
        )

        eval_df = f_h.merge(
            y_true[["unique_id", "y"]],
            on="unique_id",
            how="inner",
        )

        ae = np.abs(eval_df["y"] - eval_df["yhat"])
        se = (eval_df["y"] - eval_df["yhat"]) ** 2
        ape = ae / np.maximum(np.abs(eval_df["y"]), EPS) * 100

        rows.append({
            "model": "NBEATS",
            "target": target,
            "horizon": H,
            "mae": ae.mean(),
            "rmse": np.sqrt(se.mean()),
            "mape_pct": ape.mean(),
            "n_splits": len(eval_df),
        })

out = pd.DataFrame(rows)
out.to_csv(
    OUT_BASE / "nbeats_direct_last_origin_metrics.csv",
    index=False,
)

print("✅ N-BEATS evaluation complete (last-origin, horizon-correct)")
