# backtest_tft_nbeats.py
"""
Backtesting for TFT and N-BEATS using your orchestrate script components.

Requirements:
 - Place this file next to `orchestrate_forecast_cascaded_FINAL_bengaluru.py`
 - That orchestrate file must expose:
     - ensure_time_idx, add_cyclical_features, create_lag_and_rolling_features, fill_engineered_nans
     - MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH, MIN_ENCODER_LENGTH, BATCH_SIZE, USE_GPU, DEVICE
     - DeviceSafeWrapper, predict_with_model
     - TemporalFusionTransformer, NBeats (from pytorch_forecasting)
 - Or modify import path below.

What it does:
 - Runs rolling-origin backtest per-ward.
 - For each fold:
     - builds train_df (single ward) up to train_cutoff
     - builds TFT and N-BEATS TimeSeriesDataSet for that train
     - trains models (or loads existing checkpoints if found)
     - produces horizon forecasts using the last encoder window
 - Collects per-fold metrics and saves predictions + summary metrics.

Warning: training TFT repeatedly is expensive. Reduce folds or epochs for quick testing.
"""
import os
import math
from pathlib import Path
from typing import Dict, Any, Tuple, List
import json
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("C:\\AIurban-planning\\AI_forecasting\\src")

# change this import to match your orchestrate filename if necessary
from models.dl.orchestrate_forecast import (
    ensure_time_idx, add_cyclical_features, create_lag_and_rolling_features,
    fill_engineered_nans, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH, MIN_ENCODER_LENGTH,
    BATCH_SIZE, USE_GPU, DEVICE, DeviceSafeWrapper, predict_with_model
)

# pytorch / pytorch_forecasting imports used to instantiate models
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, NBeats
from pytorch_forecasting.metrics import MAPE
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor


BASE_DIR = Path("AI_forecasting")
PLOTS_DIR = BASE_DIR / "evaluation/advanced/plots"
METRICS_DIR = BASE_DIR / "evaluation//advanced/metrics"

for d in [PLOTS_DIR,METRICS_DIR]:
    d.mkdir(parents=True,exist_ok=True)


# ---------------------------
# small utilities
# ---------------------------
def mape(y_true, y_pred, eps=1e-9):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = np.where(y_true == 0, eps, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def ensure_month_start_index(idx):
    return pd.DatetimeIndex(pd.to_datetime(idx).to_period("M").to_timestamp())

# ---------------------------
# Build per-fold datasets (single-ward)
# ---------------------------
def build_fold_datasets_for_ward(
    ward_df: pd.DataFrame,
    target: str,
    max_encoder_length: int = MAX_ENCODER_LENGTH,
    max_prediction_length: int = MAX_PREDICTION_LENGTH,
    min_encoder_length: int = MIN_ENCODER_LENGTH,
    batch_size: int = BATCH_SIZE
):
    """
    Build TFT and N-BEATS TimeSeriesDataSet objects for a single ward training dataframe.
    ward_df must contain: date, ward_id, target, and any exog variables included in DEPENDENCY_ORDER
    """
    df = ward_df.copy().sort_values("date").reset_index(drop=True)
    df = ensure_time_idx(df, date_col="date")
    if "month" not in df.columns:
        df["month"] = pd.to_datetime(df["date"]).dt.month
    df = add_cyclical_features(df)
    # Engineer lags/rolling for target only (you can add exogs similarly)
    df = create_lag_and_rolling_features(df, [target])
    engineered_cols = [c for c in df.columns if any(s in c for s in ("_lag","_roll_","_ema_"))]
    df = fill_engineered_nans(df, engineered_cols)

    # basic required features (time_idx, month, ward_id, t)
    if "t" not in df.columns:
        df["t"] = df.groupby("ward_id").cumcount()

    # minimal set of known/unknown features for TFT
    tft_known = ["time_idx", "month", "month_sin", "month_cos", "quarter_sin", "quarter_cos", "t"]
    lag_features = [c for c in df.columns if any(c.endswith(f"_lag{i}") for i in [1,2,3,6,12])]
    roll_features = [c for c in df.columns if "_roll_" in c or "_ema_" in c]
    tft_unknown = [target] + lag_features + roll_features

    tft_kwargs = dict(
        time_idx="time_idx",
        target=target,
        group_ids=["ward_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        min_encoder_length=min_encoder_length,
        min_prediction_length=1,
        time_varying_unknown_reals=tft_unknown,
        time_varying_known_reals=tft_known,
        static_categoricals=["ward_id"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # For NBeats use simplified dataset (only target as unknown)
    nbeats_kwargs = dict(
        time_idx="time_idx",
        target=target,
        group_ids=["ward_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        min_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        time_varying_unknown_reals=[target],
        time_varying_known_reals=[],
        static_categoricals=[],
        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,
    )

    # build datasets (train dataset expects the whole series, we will later use to subset)
    tft_dataset = TimeSeriesDataSet(df, **tft_kwargs)
    nbeats_dataset = TimeSeriesDataSet(df, **nbeats_kwargs)

    # dataloaders for training
    pin_mem = USE_GPU and torch.cuda.is_available()
    tft_loader = tft_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0, pin_memory=pin_mem)
    nbeats_loader = nbeats_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0, pin_memory=pin_mem)

    return (tft_dataset, tft_loader), (nbeats_dataset, nbeats_loader), df

# ---------------------------
# Train model helper (TFT & NBeats)
# ---------------------------
def train_tft_nbeats_on_fold(
    tft_ds,
    tft_loader,
    nbeats_ds,
    nbeats_loader,
    out_dir: Path,
    target: str,
    max_epochs: int = 8,
    use_gpu: bool = USE_GPU
) -> Tuple[DeviceSafeWrapper, DeviceSafeWrapper]:
    """
    Train or load TFT and N-Beats models for a fold and return wrapped models.
    Saves checkpoints to out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_tft = out_dir / f"tft_{target}.ckpt"
    ckpt_nbeats = out_dir / f"nbeats_{target}.ckpt"

    # instantiate models (mirrors your orchestrate defaults)
    tft_model = TemporalFusionTransformer.from_dataset(
        tft_ds,
        learning_rate=3e-4,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        output_size=1,
        loss=MAPE(),
        log_interval=10,
        reduce_on_plateau_patience=3,
    )
    nbeats_model = NBeats.from_dataset(
        nbeats_ds,
        learning_rate=3e-4,
        log_interval=10,
        weight_decay=1e-5,
        widths=[128, 64],
        backcast_loss_ratio=0.1,
        loss=MAPE()
    )

    wrapped_tft = DeviceSafeWrapper(tft_model)
    wrapped_nbeats = DeviceSafeWrapper(nbeats_model)

    # Build callbacks list but only add LearningRateMonitor if we have a logger
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        ModelCheckpoint(dirpath=str(out_dir), save_top_k=1, monitor="val_loss", mode="min"),
    ]
    try:
        # only add LearningRateMonitor when Trainer logger will be available
        from pytorch_lightning.callbacks import LearningRateMonitor
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    except Exception:
        # if import fails or no logger, skip it (avoids the warning you saw).
        pass

    trainer = pl.Trainer(
        default_root_dir=str(out_dir),
        max_epochs=max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1 if use_gpu else None,
        callbacks=callbacks,
        enable_checkpointing=True,
        logger=False,  # keep logger False to avoid extra logging in backtests
    )

    # train nbeats (fast)
    try:
        if ckpt_nbeats.exists():
            print(f"[LOAD] NBEATS checkpoint found at {ckpt_nbeats}, loading skipped training.")
        else:
            trainer.fit(wrapped_nbeats, train_dataloaders=nbeats_loader)
            try:
                trainer.save_checkpoint(str(ckpt_nbeats))
            except Exception:
                pass
    except Exception as e:
        print("[WARN] NBeats training exception:", e)

    # train tft (slower, optional)
    try:
        if ckpt_tft.exists():
            print(f"[LOAD] TFT checkpoint found at {ckpt_tft}, loading skipped training.")
        else:
            trainer.fit(wrapped_tft, train_dataloaders=tft_loader)
            try:
                trainer.save_checkpoint(str(ckpt_tft))
            except Exception:
                pass
    except Exception as e:
        print("[WARN] TFT training exception:", e)

    # if checkpoints exist, attempt to load to the wrappers (robust)
    try:
        if ckpt_nbeats.exists():
            loaded_n = NBeats.load_from_checkpoint(str(ckpt_nbeats))
            wrapped_nbeats = DeviceSafeWrapper(loaded_n)
    except Exception:
        pass
    try:
        if ckpt_tft.exists():
            loaded_t = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_tft))
            wrapped_tft = DeviceSafeWrapper(loaded_t)
    except Exception:
        pass

    # attach dataset object to wrapped models where possible (useful later)
    try:
        wrapped_tft.inner.hparams.dataset = tft_ds
    except Exception:
        pass
    try:
        wrapped_nbeats.inner.hparams.dataset = nbeats_ds
    except Exception:
        pass

    return wrapped_tft, wrapped_nbeats

# ---------------------------
# Create prediction windows and forecast horizon for a ward using wrapped models
# ---------------------------
def forecast_horizon_for_ward(
    wrapped_tft,
    wrapped_nbeats,
    full_df: pd.DataFrame,
    ward: str,
    target: str,
    base_last_date: pd.Timestamp,
    base_enc_df: pd.DataFrame,
    n_months: int = 12
) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    """
    Given wrapped models and the last encoder window (base_enc_df), produce n_months forecast.
    Returns (point_series, { 'tft':series, 'nbeats':series })
    """
    # build future rows like in your orchestrate code
    enc = base_enc_df.copy().reset_index(drop=True)
    base_time_idx = int(enc["time_idx"].iloc[-1])
    base_t = int(enc["t"].iloc[-1]) if "t" in enc.columns else len(enc)-1
    last_date = pd.to_datetime(enc["date"].iloc[-1])

    future_rows = []
    for i in range(n_months):
        ti = base_time_idx + 1 + i
        dt = last_date + pd.DateOffset(months=(i+1))
        row = {c: None for c in enc.columns}
        row["ward_id"] = ward
        row["time_idx"] = ti
        row["date"] = dt
        row["month"] = dt.month
        row["t"] = base_t + (ti - base_time_idx)
        row["month_sin"] = np.sin(2 * np.pi * (row["month"] - 1) / 12)
        row["month_cos"] = np.cos(2 * np.pi * (row["month"] - 1) / 12)
        row["quarter"] = dt.quarter
        row["quarter_sin"] = np.sin(2 * np.pi * (row["quarter"] - 1) / 4)
        row["quarter_cos"] = np.cos(2 * np.pi * (row["quarter"] - 1) / 4)

        # copy last known values for exogs if present in enc
        for ex in enc.columns:
            if ex in ("ward_id","date","time_idx","month","month_sin","month_cos","quarter","quarter_sin","quarter_cos","t"):
                continue
            try:
                row[ex] = float(enc[ex].iloc[-1])
            except Exception:
                row[ex] = None
        future_rows.append(row)

    predict_df = pd.concat([enc, pd.DataFrame(future_rows)], ignore_index=True, sort=False)

    # fill engineered features by simple ffill/bfill (mirrors orchestrate)
    engineered_cols = [c for c in enc.columns if any(s in c for s in ("_lag","_roll_","_ema_"))]
    for c in engineered_cols:
        if c not in predict_df.columns:
            predict_df[c] = None
        predict_df[c] = predict_df[c].fillna(method='ffill').fillna(method='bfill')
        if predict_df[c].isna().any() and c in enc.columns:
            predict_df[c] = predict_df[c].fillna(enc[c].iloc[-1])
        predict_df[c] = predict_df[c].fillna(0.0)

    # Build TimeSeriesDataSet objects from existing training dataset is tricky here;
    # we will create small TimeSeriesDataSet objects reusing the schema of the training dataset.
    # For simplicity, we will create temporary datasets using the same definitions as training single-ward datasets
    try:
        tft_pred_ds = TimeSeriesDataSet.from_dataset(wrapped_tft.inner.hparams.dataset, predict_df, predict=True, stop_randomization=True)
        tft_dl = tft_pred_ds.to_dataloader(train=False, batch_size=min(32, n_months), num_workers=0, pin_memory=USE_GPU)
    except Exception:
        tft_dl = None

    try:
        nbeats_pred_ds = TimeSeriesDataSet.from_dataset(wrapped_nbeats.inner.hparams.dataset, predict_df, predict=True, stop_randomization=True)
        nbeats_dl = nbeats_pred_ds.to_dataloader(train=False, batch_size=min(32, n_months), num_workers=0, pin_memory=USE_GPU)
    except Exception:
        nbeats_dl = None

    # predictions
    if tft_dl is not None:
        tft_pred = predict_with_model(tft_dl, wrapped_tft, DEVICE)
    else:
        tft_pred = None

    if nbeats_dl is not None:
        nbeats_pred = predict_with_model(nbeats_dl, wrapped_nbeats, DEVICE)
    else:
        nbeats_pred = None

    # Convert predictions to pandas series indexed by dates
    idx = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_months, freq="MS")
    series_tft = pd.Series(tft_pred if tft_pred is not None else [np.nan]*n_months, index=idx)
    series_nbeats = pd.Series(nbeats_pred if nbeats_pred is not None else [np.nan]*n_months, index=idx)

    # ensemble simple average where both exist
    combined_vals = []
    for i in range(n_months):
        a = series_tft.iloc[i] if not np.isnan(series_tft.iloc[i]) else None
        b = series_nbeats.iloc[i] if not np.isnan(series_nbeats.iloc[i]) else None
        if a is None and b is None:
            combined_vals.append(np.nan)
        elif a is None:
            combined_vals.append(float(b))
        elif b is None:
            combined_vals.append(float(a))
        else:
            combined_vals.append(float(0.5*a + 0.5*b))
    combined_series = pd.Series(combined_vals, index=idx)

    return combined_series, {"tft": series_tft, "nbeats": series_nbeats}

# ---------------------------
# Rolling-origin backtest engine (per-ward)
# ---------------------------
def rolling_backtest_ward(
    full_df: pd.DataFrame,
    ward: str,
    target: str,
    initial_train_end: pd.Timestamp,
    horizon: int = 12,
    step_months: int = 12,
    out_dir: str = "evaluation/advanced/artifacts",
    max_epochs: int = 6,
    max_folds: int = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    full_df : DataFrame with ward-level monthly series (must include ward_id, date, target)
    Returns predictions DataFrame and metrics summary
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # filter ward history
    ward_df_all = full_df[full_df["ward_id"]==ward].sort_values("date").reset_index(drop=True)
    if ward_df_all.empty:
        raise ValueError(f"No data for ward {ward}")

    # train_cutoffs: from initial_train_end, expanding by step_months until last_possible_start
    last_possible_start = ward_df_all["date"].max() - pd.DateOffset(months=horizon)
    train_cutoffs = []
    t = pd.to_datetime(initial_train_end)
    while t <= last_possible_start:
        train_cutoffs.append(t)
        t = t + pd.DateOffset(months=step_months)
        if max_folds and len(train_cutoffs) >= max_folds:
            break
    if len(train_cutoffs) == 0:
        raise RuntimeError("No folds possible - adjust initial_train_end or horizon or ensure data span is sufficient.")

    all_fold_records = []
    fold_metrics = []

    for fold_idx, train_cutoff in enumerate(train_cutoffs, start=1):
        print(f"[FOLD {fold_idx}] train_end: {train_cutoff.date()}")
        # build training data for this fold (single ward)
        train_df = ward_df_all[ward_df_all["date"] <= train_cutoff].copy()
        if len(train_df) < 12:
            print("[SKIP] not enough rows for training")
            continue

        # build datasets and loaders for this fold (single ward)
        (tft_ds, tft_loader), (nbeats_ds, nbeats_loader), processed = build_fold_datasets_for_ward(
            train_df, target,
            max_encoder_length=MAX_ENCODER_LENGTH,
            max_prediction_length=MAX_PREDICTION_LENGTH,
            min_encoder_length=MIN_ENCODER_LENGTH,
            batch_size=BATCH_SIZE
        )

        # Train models (store checkpoints per-fold to avoid re-training later)
        fold_out = out_dir / f"{ward}_fold_{fold_idx}"
        wrapped_tft, wrapped_nbeats = train_tft_nbeats_on_fold(
            tft_ds, tft_loader, nbeats_ds, nbeats_loader,
            out_dir=fold_out, target=target, max_epochs=max_epochs, use_gpu=USE_GPU
        )

        # prepare last encoder window for forecasting (use most recent MAX_ENCODER_LENGTH rows from training window)
        enc = train_df.sort_values("date").copy()
        if len(enc) >= MAX_ENCODER_LENGTH:
            base_enc = enc.iloc[-MAX_ENCODER_LENGTH:].copy().reset_index(drop=True)
        else:
            pad_needed = MAX_ENCODER_LENGTH - len(enc)
            pad = pd.concat([enc.head(1)] * pad_needed, ignore_index=True)
            base_enc = pd.concat([pad, enc], ignore_index=True).reset_index(drop=True)

        base_last_date = pd.to_datetime(base_enc["date"].iloc[-1])

        # Forecast horizon
        preds_combined, preds_components = forecast_horizon_for_ward(
            wrapped_tft, wrapped_nbeats, full_df, ward, target, base_last_date, base_enc, n_months=horizon
        )

        # truth slice
        predict_start = base_last_date + pd.DateOffset(months=1)
        predict_end = base_last_date + pd.DateOffset(months=horizon)
        truth_mask = (ward_df_all["date"] >= predict_start) & (ward_df_all["date"] <= predict_end)
        truth_slice = ward_df_all[truth_mask].set_index("date")[target]

        # collect records per forecast month
        for dt in preds_combined.index:
            y_hat = float(preds_combined.loc[dt]) if not pd.isna(preds_combined.loc[dt]) else np.nan
            y_true = float(truth_slice.loc[dt]) if dt in truth_slice.index and not pd.isna(truth_slice.loc[dt]) else np.nan
            all_fold_records.append({
                "ward": ward,
                "fold": fold_idx,
                "train_end": train_cutoff,
                "date": dt,
                "y_true": y_true,
                "y_pred": y_hat,
                # optional components
                "pred_tft": float(preds_components["tft"].loc[dt]) if dt in preds_components["tft"].index and not pd.isna(preds_components["tft"].loc[dt]) else np.nan,
                "pred_nbeats": float(preds_components["nbeats"].loc[dt]) if dt in preds_components["nbeats"].index and not pd.isna(preds_components["nbeats"].loc[dt]) else np.nan
            })

        # compute fold metrics on available truth & predictions (non-NaN pairs only)
        df_fold_pred = pd.DataFrame([r for r in all_fold_records if r["fold"] == fold_idx])
        # keep only rows where both y_true and y_pred are finite numbers
        df_this_fold = df_fold_pred.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_true", "y_pred"])
        if len(df_this_fold) > 0:
            y_true = df_this_fold["y_true"].astype(float).values
            y_pred = df_this_fold["y_pred"].astype(float).values
            m = {
                "fold": fold_idx,
                "train_end": str(train_cutoff.date()),
                "n": int(len(y_true)),
                "MAE": float(mean_absolute_error(y_true, y_pred)),
                "RMSE": float(rmse(y_true, y_pred)),
                "MAPE": float(mape(y_true, y_pred))
            }
        else:
            # no valid pairs for this fold — set metrics to None/NaN and keep predictions for inspection
            m = {"fold": fold_idx, "train_end": str(train_cutoff.date()), "n": 0, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
        fold_metrics.append(m)

        # optional: flush intermediate results to disk per fold to avoid data loss
        pd.DataFrame(all_fold_records).to_csv(out_dir / f"{ward}_backtest_progress.csv", index=False)
        with open(METRICS_DIR / f"{ward}_fold_metrics.json", "w") as fh:
            json.dump(fold_metrics, fh, indent=2)

    results_df = pd.DataFrame(all_fold_records)
    # aggregate metrics for all folds using only valid rows
    valid = results_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_true", "y_pred"])
    if len(valid) > 0:
        overall = {
            "MAE": float(mean_absolute_error(valid["y_true"].astype(float), valid["y_pred"].astype(float))),
            "RMSE": float(rmse(valid["y_true"].astype(float), valid["y_pred"].astype(float))),
            "MAPE": float(mape(valid["y_true"].astype(float), valid["y_pred"].astype(float))),
            "n_obs": int(len(valid))
        }
    else:
        overall = {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "n_obs": 0}

    # save outputs
    results_df.to_csv(out_dir / f"{ward}_{target}_tft_nbeats_backtest_predictions.csv", index=False)
    with open(METRICS_DIR / f"{ward}_{target}_tft_nbeats_backtest_metrics.json", "w") as fh:
        json.dump({"folds": fold_metrics, "overall": overall}, fh, indent=2)

    return results_df, {"folds": fold_metrics, "overall": overall}

# ---------------------------
# Example usage (script)
# ---------------------------
if __name__ == "__main__":
    # Path to your ward-level CSV or the master parquet that contains ward_id and monthly series
    # IMPORTANT: For population you said you have Year,Population CSV; convert that to monthly
    data_path = "C:/AIurban-planning/data/processed/masterdata_for_modeling.parquet"
    if not Path(data_path).exists():
        raise FileNotFoundError(f"{data_path} not found - update path in script")

    # load master dataset (must contain ward_id, date, and the target field)
    df_all = pd.read_parquet(data_path) if Path(data_path).suffix.lower() == ".parquet" else pd.read_csv(data_path, parse_dates=["date"])
    df_all["ward_id"] = df_all["ward_id"].astype(str)
    df_all = ensure_time_idx(df_all, date_col="date")
    if "month" not in df_all.columns:
        df_all["month"] = pd.to_datetime(df_all["date"]).dt.month
    if "t" not in df_all.columns:
        df_all = df_all.sort_values(["ward_id","time_idx"])
        df_all["t"] = df_all.groupby("ward_id").cumcount()

    # example parameters - shorten for quick tests
    ward_to_test = list(df_all["ward_id"].unique())[0]  # pick first ward or set explicitly
    TARGETS = [
        "population",
        "electricity_demand",
        "water_demand",
        "congestion_index",
        "pm25",
    ]

    WARD_SAMPLE = list(df_all["ward_id"].unique())[:5]  # keep small to save time
  # replace with desired target
    initial_train_end = pd.Timestamp("2019-12-01")   # first training cutoff
    horizon = 12
    step_months = 12   # expand training window yearly to reduce folds
    outdir = BASE_DIR/"evaluation/advanced/artifacts"

    
    for target_col in TARGETS:
        for ward_to_test in WARD_SAMPLE:
            print(f"\nRunning backtest | Target: {target_col} | Ward: {ward_to_test}")

            preds_df, metrics = rolling_backtest_ward(
                df_all,
                ward_to_test,
                target_col,
                initial_train_end=pd.Timestamp("2019-12-01"),
                horizon=12,
                step_months=12,
                out_dir=BASE_DIR/"evaluation/advanced/artifacts",
                max_epochs=4,   # keep small
                max_folds=6
            )

    print("Backtest complete. Summary:", metrics["overall"])
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error
# helper small rmse
def rmse(a,b): return float(np.sqrt(np.mean((np.array(a)-np.array(b))**2)))


# results_df must contain columns: ['ward','fold','train_end','date','y_true','y_pred']
# if your backtest returned `results_df`, use it. otherwise read the CSV saved by the backtest:
results_df = pd.read_csv("C:\\AIurban-planning\\results\\backtest_tft_nbeats\\ward_1_population_tft_nbeats_backtest_predictions.csv", parse_dates=["date"])

# per-fold plots
for fold, g in results_df.groupby("fold"):
    g = g.sort_values("date")
    plt.figure(figsize=(10,4))
    if g["y_true"].notna().any():
        plt.plot(g["date"], g["y_true"], label="truth", marker="o")
    plt.plot(g["date"], g["y_pred"], label="pred", marker="x", linestyle="--")
    plt.title(f"Backtest Fold {fold} (train_end={g['train_end'].iloc[0] if 'train_end' in g.columns else 'NA'})")
    plt.xlabel("date")
    plt.ylabel("population")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"fold_{fold}_pred_vs_true.png")
    plt.close()

# residuals and error histogram overall
results_df = results_df.replace([np.inf, -np.inf], np.nan)
valid = results_df.dropna(subset=["y_true","y_pred"]).copy()
valid["resid"] = valid["y_true"] - valid["y_pred"]
plt.figure(figsize=(8,4))
sns.histplot(valid["resid"], kde=True)
plt.title("Residual distribution (y_true - y_pred)")
plt.savefig(PLOTS_DIR / "residual_hist.png")
plt.close()

# print per-fold metrics summary
fold_stats = []
for fold, g in valid.groupby("fold"):
    mae = mean_absolute_error(g["y_true"], g["y_pred"])
    fold_stats.append({"fold": int(fold), "n": len(g), "MAE": float(mae), "RMSE": float(rmse(g["y_true"],g["y_pred"]))})
pd.DataFrame(fold_stats).to_csv(outdir / "per_fold_metrics.csv", index=False)
print("Saved diagnostics to", outdir)
