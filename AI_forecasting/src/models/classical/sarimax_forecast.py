#!/usr/bin/env python3
"""
sarimax_forecast.py

Improved SARIMAX forecasting with robust auto_arima order selection,
fallbacks for failed fits, and stable optimizer settings.
"""

import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import json


from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm



LOG_DIR = Path("logs/forecasting/statistical")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "sarimax_run.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ============================
# CONFIGURATION (STATIC)
# ============================



INPUT_PARQUET = Path("AI_forecasting/data/input/timeseries/all_wards_monthly.parquet")


# ============================
# OUTPUT ROUTING (RESEARCH-GRADE)
# ============================
BASE_DIR = Path("AI_forecasting")
RESULTS_DIR = BASE_DIR / "results"
RESULTS_FORECASTS_DIR = RESULTS_DIR / "classical/forecasts"
RESULTS_AGG_DIR = RESULTS_DIR / "classical/aggregated"

REPORTS_DIR = Path("AI_forecasting/reports/classical")
FIGURES_DIR = REPORTS_DIR / "figures"
SUMMARY_DIR = REPORTS_DIR / "summary"


for p in [
    RESULTS_FORECASTS_DIR,
    RESULTS_AGG_DIR,
    FIGURES_DIR,
    SUMMARY_DIR
]:
    p.mkdir(parents=True, exist_ok=True)



TARGETS = [
    "electricity_demand",
    "water_demand",
    "congestion_index",
    "pm25"
]

EXOG_VARS = [
    "rainfall",
    "temperature",
    "job_density"
]

TRAIN_END = pd.Timestamp("2024-12-01")
FORECAST_HORIZON = 12
TEST_LENGTH = 12



def date_index(df):
    return pd.to_datetime(dict(year=df.year, month=df.month, day=1))


# ---------------------------------------------------------------------
# NEW: Replace grid search with fast + stable pmdarima.auto_arima
# ---------------------------------------------------------------------
def select_sarima_order(endog, exog=None, seasonal_period=12):
    try:
        model = pm.auto_arima(
            endog,
            exogenous=exog,
            seasonal=True,
            m=seasonal_period,
            stepwise=True,
            max_p=3, max_q=3,
            max_P=1, max_Q=1,
            d=None, D=None,
            error_action="ignore",
            suppress_warnings=True,
            trace=False,
            n_jobs=1
        )
        return model.order, model.seasonal_order
    except Exception as e:
        warnings.warn(f"auto_arima failed; falling back to (1,1,1)x(0,1,1,12): {e}")
        return (1,1,1), (0,1,1,seasonal_period)


# ---------------------------------------------------------------------
# Fit + forecast with strong robustness
# ---------------------------------------------------------------------
def fit_and_forecast(series, exog=None, exog_future=None, target_name="target", horizon=12, outdir=Path(".")):

    # ensure index monthly
    series = series.copy()
    series.index = pd.DatetimeIndex(series.index).to_period("M").to_timestamp()

    endog = series[target_name].astype(float)

    # optional STL (kept but does not affect SARIMAX order)
    try:
        STL(endog, period=12, robust=True).fit()
    except Exception:
        pass

    # -----------------------------------------------------------
    # 1. Determine ARIMA order via auto_arima
    # -----------------------------------------------------------
    order, seasonal = select_sarima_order(
        endog,
        exog=(exog.values if exog is not None else None),
        seasonal_period=12
    )

    # -----------------------------------------------------------
    # 2. Fit SARIMAX with robust optimizer + fallback
    # -----------------------------------------------------------
    model = SARIMAX(
        endog,
        exog=(exog.values if exog is not None else None),
        order=order,
        seasonal_order=seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    try:
        # safer optimizer than L-BFGS
        res = model.fit(method="powell", maxiter=300, disp=False)
    except Exception as e1:
        warnings.warn(f"SARIMAX Powell failed → trying Nelder-Mead: {e1}")
        try:
            res = model.fit(method="nm", maxiter=200, disp=False)
        except Exception as e2:
            warnings.warn(f"SARIMAX failed → falling back to ETS: {e2}")
            ets = ExponentialSmoothing(
                endog,
                seasonal="add",
                seasonal_periods=12
            ).fit()
            future_idx = pd.date_range(
                start=series.index[-1] + pd.offsets.MonthBegin(1),
                periods=horizon,
                freq="MS"
            )
            fc = ets.forecast(horizon)
            out = pd.DataFrame({
                "date": future_idx,
                "forecast": fc.values,
                "lower_95": fc.values * 0.95,
                "upper_95": fc.values * 1.05
            })
            plt.figure(figsize=(10,5))
            series.plot(label="obs")
            plt.plot(out["date"], out["forecast"], label="ETS forecast")
            plt.legend()
            plt.savefig(outdir / f"{target_name}_diagnostics.png", dpi=150)
            plt.close()
            return ets, out

    # -----------------------------------------------------------
    # 3. Forecast SARIMAX
    # -----------------------------------------------------------
    pred = res.get_forecast(
        steps=horizon,
        exog=(exog_future.values if exog_future is not None else None)
    )
    mean = pred.predicted_mean
    conf = pred.conf_int(alpha=0.05)

    last_period = series.index[-1]
    future_idx = pd.date_range(start=last_period + pd.offsets.MonthBegin(1),
                               periods=horizon,
                               freq="MS")

    out = pd.DataFrame({
        "date": future_idx,
        "forecast": mean.values,
        "lower_95": conf.iloc[:, 0].values,
        "upper_95": conf.iloc[:, 1].values
    })

    # -----------------------------------------------------------
    # 4. Diagnostics plot
    # -----------------------------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(10, 6),
                           gridspec_kw={"height_ratios": [2, 1]})

    series.plot(ax=ax[0], label="obs")
    ax[0].plot(out["date"], out["forecast"], label="forecast", color="C1")
    ax[0].fill_between(out["date"], out["lower_95"], out["upper_95"],
                       color="C1", alpha=0.2)
    ax[0].legend()
    ax[0].set_title(f"{target_name} forecast")

    res.resid.plot(ax=ax[1], title="Residuals")
    plt.tight_layout()
    plt.savefig(outdir / f"{target_name}_diagnostics.png", dpi=150)
    plt.close(fig)

    return res, out


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate_forecast(train_series, test_series, forecast_df, target_name):
    y_true = test_series[target_name].values
    y_pred = forecast_df["forecast"].values[:len(y_true)]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred) /
                np.where(y_true == 0, 1e-6, y_true))) * 100
    return {"mae": mae, "rmse": rmse, "mape_pct": mape}




# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    start_time = datetime.utcnow()
    logger.info("START SARIMAX RUN")


    
    aggdir = RESULTS_AGG_DIR

    df = pd.read_parquet(INPUT_PARQUET)

    df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    df = df.sort_values(["ward_id", "date"])

    TOP_K = 10

    top_wards = (
        df.groupby("ward_id")["electricity_demand"]
        .max()
        .sort_values(ascending=False)
        .head(TOP_K)
        .index
        .tolist()
    )

    df = df[df["ward_id"].isin(top_wards)]


    targets = TARGETS
    exog_vars = EXOG_VARS


    wards = df["ward_id"].unique().tolist()

    aggregated_forecasts = {t: [] for t in targets}
    aggregated_metrics = {t: [] for t in targets}

    for ward in wards:
        wdf = df[df["ward_id"] == ward].set_index("date").copy()
        ward_results = RESULTS_FORECASTS_DIR / f"{ward}"
        ward_figures = FIGURES_DIR / f"{ward}"
        ward_summary = SUMMARY_DIR / f"{ward}"

        for p in [ward_results, ward_figures, ward_summary]:
            p.mkdir(parents=True, exist_ok=True)


        train_end = TRAIN_END 
        if train_end > wdf.index.max():
            train_end = wdf.index.max()

        train = wdf.loc[:train_end]
        test_start = train_end + pd.offsets.MonthBegin(1)
        test_end = test_start + pd.offsets.MonthBegin(TEST_LENGTH - 1)

        test = wdf.loc[test_start:test_end]


        for target in targets:
            try:
                series = train[[target]].dropna()
                if len(series) < 36:
                    continue

                if all(col in train.columns for col in exog_vars):
                    exog_train = train[exog_vars]
                else:
                    exog_train = None

                # naive repeating seasonal exog forecast
                if exog_train is not None:
                    last_year = exog_train.tail(12)
                    repeats = int(np.ceil(FORECAST_HORIZON / 12))
                    exog_future = pd.concat([last_year] * repeats).iloc[:FORECAST_HORIZON]
                    exog_future.index = pd.date_range(
                        start=series.index[-1] + pd.offsets.MonthBegin(1),
                        periods=FORECAST_HORIZON,
                        freq="MS"
                    )
                else:
                    exog_future = None

                res, forecast_df = fit_and_forecast(
                    series,
                    exog=exog_train,
                    exog_future=exog_future,
                    target_name=target,
                    horizon=FORECAST_HORIZON,
                    outdir=ward_figures
                )

                forecast_df.to_csv(ward_results / f"{target}_forecast.csv", index=False)
                with open(ward_summary / f"{target}_model_summary.txt", "w") as f:
                    f.write(res.summary().as_text())

                # evaluation
                if target in test.columns and len(test) >= 3:
                    metrics = evaluate_forecast(train, test, forecast_df, target)
                    metrics["ward_id"] = ward
                    metrics["target"] = target
                    aggregated_metrics[target].append(metrics)
                else:
                    logger.warning(
                        f"Skipping evaluation | ward={ward} target={target} test_len={len(test)}"
                    )


                adf = forecast_df.copy()
                adf["ward_id"] = ward
                adf["target"] = target
                aggregated_forecasts[target].append(adf)

            except Exception as e:
                logger.warning(f"Failed {ward} {target}: {e}")

    for target, lst in aggregated_metrics.items():
        logger.info(
            f"Evaluation summary | target={target} | n_evaluated_wards={len(lst)}"
        )


    # save aggregated outputs
    for target, lst in aggregated_forecasts.items():
        if lst:
            pd.concat(lst, ignore_index=True).to_csv(
                aggdir / f"{target}_forecasts_all_wards.csv", index=False)

    for target, lst in aggregated_metrics.items():
        out_path = aggdir / f"{target}_metrics.csv"
        if lst:
            pd.DataFrame(lst).to_csv(out_path, index=False)
        else:
            pd.DataFrame(
                columns=["ward_id", "target", "mae", "rmse", "mape_pct"]
            ).to_csv(out_path, index=False)
            logger.warning(f"No metrics written for {target} — empty evaluation set")

    logger.info(f"SARIMAX forecasting finished. Results in:{RESULTS_DIR}")

    duration = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"END RUN | duration_sec={duration:.1f}")


    run_meta = {
        "method": "SARIMAX (auto_arima) with ETS fallback",
        "targets": TARGETS,
        "exogenous_variables": EXOG_VARS,
        "train_end": str(TRAIN_END.date()),
        "forecast_horizon_months": FORECAST_HORIZON,
        "test_length_months": TEST_LENGTH,
        "input_data": str(INPUT_PARQUET),
        "generated_at_utc": datetime.utcnow().isoformat()
    }

    with open(RESULTS_DIR / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)



if __name__ == "__main__":
    main()
