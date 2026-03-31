#!/usr/bin/env python3
"""
ml_forecast_final.py

Robust, Windows-safe ML forecasting per-ward.
- Uses RandomForest as default, LightGBM if available (with safe checks).
- Uses ThreadPoolExecutor (no problematic joblib pickling on Windows).
- Reads enriched parquet (generated above), produces per-ward output CSVs and summary.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")

# optional
try:
    import lightgbm as lgb
except Exception:
    lgb = None

import logging
import joblib
from datetime import datetime

INPUT_PATH = Path("AI_forecasting/data/input/timeseries/all_wards_monthly.parquet")

BASE_DIR = Path("AI_forecasting")

ARTIFACT_DIR = BASE_DIR / "artifacts/ml"
MODEL_DIR = ARTIFACT_DIR / "models"
FI_DIR = ARTIFACT_DIR / "feature_importance"
METRIC_DIR = ARTIFACT_DIR / "metrics"
META_DIR = ARTIFACT_DIR / "metadata"

FORECAST_DIR = BASE_DIR / "results/ml/forecasts"
SUMMARY_DIR = BASE_DIR / "results/ml/summaries"
LOG_DIR = BASE_DIR / "logs/forecasting"

for d in [MODEL_DIR, FI_DIR, METRIC_DIR, META_DIR, FORECAST_DIR, SUMMARY_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "ml_forecasting.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("ML_FORECAST")


np.random.seed(42)

def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def generate_lags_and_feats(df, target, lags=(1,2,3,6,12), roll_windows=(3,6,12)):
    X = pd.DataFrame(index=df.index)
    for lag in lags:
        X[f"{target}_lag{lag}"] = df[target].shift(lag)
    for w in roll_windows:
        X[f"{target}_rmean_{w}"] = df[target].shift(1).rolling(window=w, min_periods=1).mean()
    # month seasonality
    if "month" in df.columns:
        month = df["month"]
    else:
        month = df.index.month
    X["month"] = month
    X["month_sin"] = np.sin(2*np.pi*(X["month"]-1)/12)
    X["month_cos"] = np.cos(2*np.pi*(X["month"]-1)/12)
    X["t"] = np.arange(len(df))
    # add exogs
    for col in df.columns:
        if col == target:
            continue
        X[col] = df[col]
        if pd.api.types.is_numeric_dtype(df[col]):
            X[f"{col}_lag1"] = df[col].shift(1)
    return X

def train_test_split_time(df, train_end):
    train_end_ts = pd.to_datetime(train_end)
    df = df.sort_index()
    return df[df.index <= train_end_ts], df[df.index > train_end_ts]

def backtest_split_time(df, backtest_train_end):
    """
    Backtesting split:
    - Train until backtest_train_end
    - Test = next 12 months
    """
    df = df.sort_index()
    train_end = pd.to_datetime(backtest_train_end)

    train_df = df[df.index <= train_end]
    test_df = df[(df.index > train_end) & 
                 (df.index <= train_end + pd.DateOffset(months=12))]

    return train_df, test_df


def data_sanity_check(X, y, min_periods=12):
    if y.dropna().shape[0] < min_periods:
        return {"error": "insufficient_rows"}
    if y.dropna().nunique() <= 1:
        return {"error": "target_constant"}
    zero_var = [c for c in X.columns if X[c].dropna().nunique() <= 1]
    if X.shape[1] == 0:
        return {"error": "no_features"}
    if zero_var:
        return {"warning": "zero_var", "cols": zero_var}
    return None

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(y_true == 0, 1e-9, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

def run_backtest_evaluation(df_ts, target, model, feature_columns, backtest_train_end):
    """
    One-step-ahead rolling backtest for 12 months
    """
    train_df, test_df = backtest_split_time(df_ts, backtest_train_end)

    if test_df.empty or len(test_df) < 3:
        return {}

    full = pd.concat([train_df, test_df], axis=0)

    X_all = generate_lags_and_feats(full, target)
    X_test = X_all.loc[test_df.index].drop(columns=[target], errors="ignore").fillna(0)

    # Align features
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    try:
        y_pred = model.predict(X_test.values)
    except Exception:
        return {}

    y_true = test_df[target].values

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mape(y_true, y_pred))
    }


def fit_models_for_ward(ward_id, df_ts, target, models_to_run, train_end, horizon):

    df_ts = df_ts.sort_index().copy()
    df_ts = df_ts.ffill().bfill()

    if target not in df_ts.columns:
        return None

    df_ts[target] = pd.to_numeric(df_ts[target], errors="coerce")
    train_df, test_df = train_test_split_time(df_ts, train_end)

    X_all = generate_lags_and_feats(train_df, target)
    y_all = train_df[target]

    df_feat = X_all.join(y_all).dropna()
    if df_feat.shape[0] < 12:
        return None

    X = df_feat.drop(columns=[target])
    y = df_feat[target]

    san = data_sanity_check(X, y)
    if san is not None and san.get("error"):
        return None
    if san is not None and san.get("warning"):
        drop_cols = san.get("cols", [])
        X = X.drop(columns=drop_cols, errors="ignore")

    # params
    rf_params = {"n_estimators": 200, "max_depth": 12, "n_jobs": 1, "random_state": 42}
    lgb_params = {
        "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 64,
        "min_data_in_leaf": 1, "min_gain_to_split": 0.0, "force_row_wise": True,
        "verbosity": -1, "n_jobs": 1, "random_state": 42
    }

    models = {}

    # RF
    try:
        rf = RandomForestRegressor(**rf_params)
        rf.fit(X, y)
        models["rf"] = rf
    except Exception as e:
        pass

    # LGB (only if available and variable features present)
    if "lgb" in models_to_run and lgb is not None:
        X_lgb = X.loc[:, X.nunique(dropna=True) > 1]
        if X_lgb.shape[1] >= 1:
            try:
                mdl = lgb.LGBMRegressor(**lgb_params)
                mdl.fit(X_lgb, y)
                models["lgb"] = mdl
            except Exception:
                pass

    if not models:
        return None

    # choose model preference
    for name in ("lgb", "rf"):
        if name in models:
            best_name = name
            best = models[name]
            break
    # -----------------------------
# BACKTEST EVALUATION (RESEARCH)
# -----------------------------
    backtest_metrics = run_backtest_evaluation(
        df_ts=df_ts,
        target=target,
        model=best,
        feature_columns=X.columns.tolist(),
        backtest_train_end="2024-12"
    )


    model_path = MODEL_DIR / f"{ward_id}_{target}_{best_name}.joblib"
    joblib.dump(best, model_path)
    log.info(f"Saved model at {model_path}")


    def recursive_forecast(df_full, model, steps, target):
        history = df_full.copy()
        preds = []
        for _ in range(steps):
            feats = generate_lags_and_feats(history, target).iloc[[-1]].drop(columns=[target], errors="ignore")
            feats = feats.fillna(0)
            # if model expects features
            if hasattr(model, "feature_names_in_"):
                feats = feats.reindex(columns=model.feature_names_in_, fill_value=0)
                arr = feats.values
            else:
                # align with X columns used for training (approx)
                train_cols = X.columns if 'X' in locals() else feats.columns
                feats = feats.reindex(columns=train_cols, fill_value=0)
                arr = feats.values
            try:
                pred = float(model.predict(arr)[0])
            except Exception:
                pred = float(history[target].iloc[-1])
            next_date = history.index[-1] + pd.offsets.MonthBegin(1)
            new_row = {c: history[c].iloc[-1] for c in history.columns}
            new_row[target] = pred
            history = pd.concat([history, pd.DataFrame([new_row], index=[next_date])])
            preds.append((next_date, pred))
        return preds

    preds = recursive_forecast(df_ts, best, horizon, target)
    forecast_df = pd.DataFrame(preds, columns=["date", "forecast"]).set_index("date")
    forecast_df.index = pd.to_datetime(forecast_df.index)
    forecast_df["lower_95"] = forecast_df["forecast"] * 0.9
    forecast_df["upper_95"] = forecast_df["forecast"] * 1.1

    # save
    forecast_path = FORECAST_DIR / f"{ward_id}_{target}_forecast.csv"
    forecast_df.reset_index().to_csv(forecast_path, index=False)
    log.info(f"Saved forecast at {forecast_path}")



    
    # feature importance
    fi = {}
    try:
        if hasattr(best, "feature_importances_"):
            fi = dict(zip(X.columns, getattr(best, "feature_importances_").astype(float)))
        elif hasattr(best, "booster_") or hasattr(best, "get_booster"):
            # lightgbm
            try:
                booster = best.booster_ if hasattr(best, "booster_") else best.get_booster()
                fi = booster.feature_importance(importance_type="gain")
                # mapping
                names = booster.feature_name()
                fi = dict(zip(names, fi.astype(float)))
            except Exception:
                fi = {}
    except Exception:
        fi = {}

    with open(FI_DIR / f"{ward_id}_{target}_fi.json", "w") as f:
        json.dump(fi, f, indent=2)


    # Evaluate on test set (one-step predictions)
    metrics = {}
    if not test_df.empty:
        try:
            full = pd.concat([train_df, test_df], axis=0)
            X_test_all = generate_lags_and_feats(full, target)
            X_test = X_test_all.loc[test_df.index].drop(columns=[target], errors="ignore").fillna(0)
            if hasattr(best, "feature_names_in_"):
                X_test = X_test.reindex(columns=best.feature_names_in_, fill_value=0)
                arr = X_test.values
            else:
                arr = X_test.values
            y_pred = best.predict(arr)
            y_true = test_df[target].values
            metrics["mae_test"] = float(mean_absolute_error(y_true, y_pred))
            metrics["mse_test"] = float(mean_squared_error(y_true, y_pred))
            metrics["mape_test"] = float(mape(y_true, y_pred))
        except Exception:
            pass

    metric_payload = {
        "ward_id": ward_id,
        "target": target,
        "model": best_name,
        "train_rows": int(len(train_df)),
        "forecast_horizon_months": horizon,
        "backtest": {
            "train_end": "2024-12",
            "test_period": "2025-01 to 2025-12",
            "metrics": backtest_metrics
        },
        "timestamp": datetime.utcnow().isoformat()
    }

    with open(METRIC_DIR / f"{ward_id}_{target}_backtest_metrics.json", "w") as f:
        json.dump(metric_payload, f, indent=2)

    return {
        "ward_id": str(ward_id),
        "target": target,
        "model": best_name,
        "forecast_path": str(forecast_path),
        "model_path": str(model_path),
        "metrics": metrics
    }


def load_input_timeseries(path):
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix.lower()==".parquet" else pd.read_csv(path)
    # parse date
    if "date" not in df.columns:
        if {"year","month"}.issubset(df.columns):
            df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01")
        else:
            candidates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            if candidates:
                df["date"] = pd.to_datetime(df[candidates[0]])
            else:
                raise ValueError("No date column found in input.")
    else:
        df["date"] = pd.to_datetime(df["date"])
    # ward id detection
    if "ward_id" not in df.columns:
        candidates = [c for c in df.columns if "ward" in c.lower()]
        if len(candidates) == 1:
            df = df.rename(columns={candidates[0]: "ward_id"})
        else:
            raise ValueError("Could not find ward identifier column.")
    df["ward_id"] = df["ward_id"].astype(str)
    out = {}
    for w, g in df.groupby("ward_id"):
        g = g.sort_values("date")
        g = g.set_index("date")
        g = g.drop(columns=["ward_id"], errors="ignore")
        out[str(w)] = g
    return out

def _worker_wrapper(ward_id, df_ts, target, models, train_end, horizon):
    try:
        return fit_models_for_ward(ward_id, df_ts, target, models, train_end, horizon)
    except Exception as e:
        return {"ward_id": str(ward_id), "target": target, "error": str(e)}

def run_all(input_path, targets, train_end, horizon, models_list, n_jobs):
    ward_dict = load_input_timeseries(input_path)
    tasks = []
    for ward_id, df_ts in ward_dict.items():
        for t in targets:
            if t in df_ts.columns:
                tasks.append((ward_id, df_ts.copy(), t))
    print("Total tasks:", len(tasks))
    results = []
    max_workers = max(1, n_jobs)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {
            ex.submit(_worker_wrapper, wid, df, t, models_list, train_end, horizon): (wid, t)
            for (wid, df, t) in tasks
        }
        for fut in as_completed(future_map):
            wid, t = future_map[fut]
            try:
                r = fut.result()
                if r is None:
                    continue
                if isinstance(r, dict) and r.get("error"):
                    print(f"[ERROR] ward {wid} {t}: {r['error']}")
                else:
                    results.append(r)
            except Exception as e:
                print(f"[ERROR] ward {wid} {t} raised exception: {e}")

def aggregate_city_level_metrics(metric_dir: Path, out_dir: Path):
    rows = []
    for f in metric_dir.glob("*_backtest_metrics.json"):
        with open(f) as fh:
            data = json.load(fh)

        bt = data.get("backtest", {})
        m = bt.get("metrics", {})

        if not m:
            continue

        rows.append({
            "ward_id": data["ward_id"],
            "target": data["target"],
            "model": data["model"],
            "mae": m.get("mae"),
            "rmse": m.get("rmse"),
            "mape": m.get("mape")
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return None

    summary = df.groupby("target").agg(
        mean_mae=("mae", "mean"),
        median_mae=("mae", "median"),
        std_mae=("mae", "std"),
        mean_rmse=("rmse", "mean"),
        mean_mape=("mape", "mean")
    ).reset_index()

    out_path = out_dir / "city_level_backtest_summary.csv"
    summary.to_csv(out_path, index=False)
    log.info(f"City-level backtest summary saved at {out_path}")

    return df

def rank_wards_by_error(df_metrics: pd.DataFrame, out_dir: Path, top_k=10):
    for target in df_metrics["target"].unique():
        sub = df_metrics[df_metrics["target"] == target].sort_values("mae")

        best = sub.head(top_k)
        worst = sub.tail(top_k)

        best.to_csv(out_dir / f"top_{top_k}_best_{target}.csv", index=False)
        worst.to_csv(out_dir / f"top_{top_k}_worst_{target}.csv", index=False)

        log.info(f"Saved top/bottom {top_k} wards for {target}")



def plot_top_wards_forecasts(
    forecast_dir: Path,
    top_wards_csv: Path,
    out_dir: Path
):
    wards = pd.read_csv(top_wards_csv)["ward_id"].tolist()
    out_dir.mkdir(parents=True, exist_ok=True)

    for ward in wards:
        files = list(forecast_dir.glob(f"{ward}_*_forecast.csv"))
        for f in files:
            df = pd.read_csv(f, parse_dates=["date"])
            plt.figure(figsize=(8,3))
            plt.plot(df["date"], df["forecast"], label="Forecast")
            plt.fill_between(
                df["date"],
                df["lower_95"],
                df["upper_95"],
                alpha=0.3
            )
            plt.title(f"{ward} – {f.stem}")
            plt.tight_layout()
            plt.savefig(out_dir / f"{f.stem}.png", dpi=150)
            plt.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="AI_forecasting/data/input/timeseries/all_wards_monthly.parquet")
    parser.add_argument("--targets", default="electricity_demand,water_demand,congestion_index,pm25")
    parser.add_argument("--train-end", default="2025-12")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--models", default="rf,lgb")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    targets = [t.strip() for t in args.targets.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    run_all(
        input_path=args.input,   # internally used, not CLI-facing
        targets=targets,
        train_end=args.train_end,
        horizon=args.horizon,
        models_list=models,
        n_jobs=args.n_jobs
    )

    df_metrics = aggregate_city_level_metrics(
        metric_dir=METRIC_DIR,
        out_dir=SUMMARY_DIR
    )

    if df_metrics is not None:
        rank_wards_by_error(df_metrics, SUMMARY_DIR, top_k=10)

    for target in targets:
        plot_top_wards_forecasts(
            forecast_dir=FORECAST_DIR,
            top_wards_csv=SUMMARY_DIR / "top_10_worst_pm25.csv",
            out_dir=BASE_DIR / "reports/ml/top_10_forecasts"
        )

    run_meta = {
        "run_time": datetime.utcnow().isoformat(),
        "train_end": args.train_end,
        "horizon": args.horizon,
        "targets": targets,
        "models": models,
        "n_jobs": args.n_jobs,
        "input_path": args.input
    }

    with open(META_DIR / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)


if __name__ == "__main__":
    main()
