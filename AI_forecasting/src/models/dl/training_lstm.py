# lstm_global_training_fixed.py
"""
Robust LSTM training + autoregressive rollforward for ward-level forecasts (2014-2025 history -> forecast 2026-2035)

Usage:
    python training_lstm.py

Expectations:
 - Input parquet should be a cleaned canonical dataset (2014-2025)
 - No explicit lag/rolling features are required
 - Temporal learning is handled internally by the LSTM


Outputs:
 - OUTDIR/checkpoints (best ckpt)
 - OUTDIR/lstm_predictions_2026_2035.csv
 - OUTDIR/lstm_example_plots/ (plots for sample wards)
 - OUTDIR/train_metrics.json
"""
import os
from pathlib import Path
import pickle
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# PyTorch / Lightning
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging


BASE_DIR = Path("AI_forecasting")
# Input (preprocessed canonical dataset for modeling 2014-2025)
DATA_PATH = BASE_DIR / "data/input/masterdata_for_modeling_lstm.parquet"

# scaler/meta saved by preprocessing step (adjust if you used different paths)
SCALER_PATH = BASE_DIR/ "artifacts/dl/metadata/scalers_lstm.joblib"
META_PATH = BASE_DIR/ "artifacts/dl/metadata/meta_lstm.json"

ARTIFACT_DIR = BASE_DIR / "artifacts/dl/lstm"
RESULTS_DIR = BASE_DIR / "results/lstm"
REPORTS_DIR = BASE_DIR / "reports/dl"
LOG_DIR = BASE_DIR / "logs/dl/lstm"

for d in [ARTIFACT_DIR, RESULTS_DIR, REPORTS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)




PLOTS_DIR = REPORTS_DIR / "lstm_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "run.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# modeling hyperparams
INPUT_WINDOW = 36       # months used as input
TRAIN_HORIZON = 12      # model trained to predict next 12 months
BATCH_SIZE = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 2
EPOCHS = 40
LR = 1e-3
SEED = 42

# ----------------------------
# Experiment metadata
# ----------------------------
EXPERIMENT_NAME = "lstm_global_forecasting_v1"
EXPERIMENT_DATE = datetime.now().strftime("%Y-%m-%d_%H-%M")
GIT_COMMIT = os.environ.get("GIT_COMMIT", "unknown")  # optional


# default target ordering (will prefer meta target_cols if present)
DEFAULT_TARGET_COLS = ["electricity_demand", "water_demand", "congestion_index", "pm25"]

# columns that are static or identifying
ID_COL = "ward_id"
DATE_COL = "date"

# ----------------------------
# ScalerWrapper - robust utility
# ----------------------------
class ScalerWrapper:
    """
    Wrapper around a fitted StandardScaler & column metadata that provides:
     - transform / inverse_transform via underlying scaler
     - inverse_y(y_scaled) to convert predicted scaled targets back to original scale
    """
    def __init__(self, x_scaler: StandardScaler, numeric_cols: list, target_cols: list):
        self.x_scaler = x_scaler
        self.numeric_cols = list(numeric_cols)
        self.col_to_idx = {c: i for i, c in enumerate(self.numeric_cols)}
        self.target_cols = list(target_cols)
        # ensure target_idxs are valid
        self.target_idxs = [self.col_to_idx[t] for t in self.target_cols]

    def transform(self, arr):
        # arr: (..., n_features) matching numeric_cols
        return self.x_scaler.transform(arr)

    def inverse_transform(self, arr):
        return self.x_scaler.inverse_transform(arr)

    def inverse_y(self, y_scaled):
        """
        y_scaled: numpy array shaped (B, H, T) or (H, T) in scaled space for the T targets in self.target_cols.
        Returns: numpy array of shape (B, H, T) in original scale aligned to self.target_cols order.
        Implementation:
         - create zero array in full numeric_cols space,
         - insert scaled predictions at the target indices,
         - inverse_transform with scaler,
         - extract target columns.
        """
        arr = np.array(y_scaled)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]  # (1, H, T)
        if arr.ndim != 3:
            raise ValueError("inverse_y expects array of shape (B,H,T) or (H,T)")

        B, H, T = arr.shape
        n_features = len(self.numeric_cols)
        full = np.zeros((B, H, n_features), dtype=float)
        # Place scaled target predictions into the appropriate indices
        for ti, idx in enumerate(self.target_idxs):
            full[:, :, idx] = arr[:, :, ti]

        # Reshape and inverse transform
        flat = full.reshape(-1, n_features)
        inv_flat = self.x_scaler.inverse_transform(flat)
        inv = inv_flat.reshape(B, H, n_features)
        # extract only target columns in order
        targets = inv[:, :, self.target_idxs]  # (B, H, T)
        return targets

# ----------------------------
# SlidingWindowDataset Class
# ----------------------------
class SlidingWindowDataset(Dataset):
    def __init__(self, df, numeric_cols, targets, id_col="ward_id",
                 date_col="date", input_window=36, horizon=12, scaler=None):
        self.input_window = input_window
        self.horizon = horizon
        self.numeric_cols = list(numeric_cols)
        self.targets = list(targets)
        self.scaler = scaler

        # validation
        missing = [c for c in self.numeric_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Numeric columns missing from dataframe: {missing}")

        # create list of sequences across wards
        self.windows = []
        self.meta = []  # (ward_id, last_date_of_input)
        for ward, g in df.groupby(id_col):
            g = g.sort_values(date_col).reset_index(drop=True)
            arr = g[self.numeric_cols].values
            dates = g[date_col].values
            n = len(g)
            for start in range(0, n - input_window - horizon + 1):
                inp = arr[start:start + input_window]
                out = arr[start + input_window:start + input_window + horizon]
                # keep only target columns for output (in same order as targets)
                out_targets = out[:, [self.numeric_cols.index(t) for t in self.targets]] 
 
                last_date = dates[start + input_window - 1]
                self.windows.append((ward, inp, out_targets))
                self.meta.append((ward, pd.Timestamp(last_date)))
        logger.info(f"[Dataset] Built {len(self.windows)} windows from {df[id_col].nunique()} wards")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        ward, inp, out_targets = self.windows[idx]
        if self.scaler is not None:
            inp_scaled = self.scaler.transform(inp)
            # we need to produce scaled out_targets shaped (horizon, n_targets)
            # create full-row pad for each horizon row
            # pad to full numeric_cols then transform and extract target columns
            # create zeros for non-target columns
            H = out_targets.shape[0]
            full_rows = np.zeros((H, len(self.numeric_cols)), dtype=float)
            for i in range(H):
                for j, t in enumerate(self.targets):
                    full_rows[i, self.numeric_cols.index(t)] = out_targets[i, j]
            out_scaled_full = self.scaler.transform(full_rows)
            out_scaled = out_scaled_full[:, [self.numeric_cols.index(t) for t in self.targets]]
        else:
            inp_scaled = inp
            out_scaled = out_targets
        return {
            "ward": ward,
            "x": torch.tensor(inp_scaled, dtype=torch.float32),
            "y": torch.tensor(out_scaled.reshape(-1, len(self.targets)), dtype=torch.float32)  # (horizon, n_targets)
        }

# ----------------------------
# Rolling backtest utility
# ----------------------------
def rolling_backtest(df, targets, date_col, id_col):
    results = []
    cutoffs = pd.date_range("2021-01-01", "2024-01-01", freq="6MS")

    for cutoff in cutoffs:
        train = df[df[date_col] <= cutoff]
        test = df[(df[date_col] > cutoff) & (df[date_col] <= cutoff + pd.DateOffset(months=6))]

        if test.empty:
            continue

        for t in targets:
            mae = mean_absolute_error(
                test[t].values,
                train.groupby(id_col)[t].last().reindex(test[id_col]).values
            )
            results.append({
                "cutoff": str(cutoff.date()),
                "target": t,
                "MAE": float(mae)
            })

    return pd.DataFrame(results)

# ----------------------------
# Lightning Module: LitLSTM
# ----------------------------
class LitLSTM(pl.LightningModule):
    def __init__(self, n_features, n_targets, hidden_size=128, num_layers=2, lr=1e-3, horizon=12):
        super().__init__()
        self.save_hyperparameters()
        self.n_features = n_features
        self.n_targets = n_targets
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon
        self.lr = lr


        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, max(8, hidden_size//2)),
            nn.ReLU(),
            nn.Linear(max(8, hidden_size//2), horizon * n_targets)
        )
        
    def multiscale_loss(self,pred, target):
        loss = nn.functional.mse_loss(pred, target)
        loss += 0.3 * nn.functional.mse_loss(pred[:, ::3], target[:, ::3])
        loss += 0.1 * nn.functional.mse_loss(pred[:, ::6], target[:, ::6])
        return loss




    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        preds = self.fc(last)
        return preds.view(-1, self.horizon, self.n_targets)


    def training_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        y_hat = self(x)
        loss = self.multiscale_loss(y_hat, y)
        # gradient clipping handled by Trainer argument gradient_clip_val
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        y_hat = self(x)
        loss = self.multiscale_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        # add LR scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
    

# ----------------------------
# Main Execution
# ----------------------------
def main():
    pl.seed_everything(SEED, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # -------- load data --------
    logger.info(f"Loading cleaned data: {DATA_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    # ensure date col is datetime
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

    # -------- load scaler wrapper (or wrap existing scaler dict) --------
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
    logger.info(f"Loading scaler from: {SCALER_PATH}")
    with open(SCALER_PATH, "rb") as f:
        try:
            scaler_wrapper = pickle.load(f)
        except Exception:
            # try joblib
            scaler_wrapper = joblib.load(f)
    # If scaler_wrapper is a dict (older), convert to ScalerWrapper
    if isinstance(scaler_wrapper, dict):
        # expect keys 'x_scaler' (StandardScaler) and 'numeric_cols' and optionally 'target_cols'
        x_scaler = scaler_wrapper.get("x_scaler")
        numeric_cols = scaler_wrapper.get("numeric_cols")
        target_cols = scaler_wrapper.get("target_cols", DEFAULT_TARGET_COLS)
        if x_scaler is None or numeric_cols is None:
            raise KeyError("scalers.pkl dictionary missing expected keys 'x_scaler' or 'numeric_cols'. Re-run preprocessing to save proper scaler.")
        scaler_wrapper = ScalerWrapper(x_scaler=x_scaler, numeric_cols=numeric_cols, target_cols=target_cols)
    else:
        # ensure wrapper has required attributes; if missing, try to infer from meta
        if not hasattr(scaler_wrapper, "x_scaler") or not hasattr(scaler_wrapper, "numeric_cols"):
            raise AttributeError("Loaded scaler_wrapper lacks required attributes 'x_scaler' and 'numeric_cols'.")
        # if it's already a ScalerWrapper or similar, ensure target_idxs present
        if not hasattr(scaler_wrapper, "target_idxs") or not hasattr(scaler_wrapper, "col_to_idx"):
            # we will create them from meta below after loading meta
            pass

    # -------- load meta (json or pickle) --------
    if not META_PATH.exists():
        raise FileNotFoundError(f"Meta file not found: {META_PATH}")
    logger.info(f"Loading meta from: {META_PATH}")
    try:
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
            # if meta is bytes or str, handle below
            if isinstance(meta, (bytes, str)):
                meta = json.loads(meta)
    except Exception:
        with open(META_PATH, "r") as f:
            meta = json.load(f)

    tv_cols = meta.get("tv_cols") or getattr(scaler_wrapper, "numeric_cols", None)
    target_cols_meta = meta.get("target_cols", DEFAULT_TARGET_COLS)

    if tv_cols is None:
        raise KeyError("tv_cols not found in meta and not present in scaler wrapper.")

    # ensure scaler_wrapper has numeric_cols attribute matching tv_cols
    if not hasattr(scaler_wrapper, "numeric_cols"):
        scaler_wrapper.numeric_cols = list(tv_cols)
        scaler_wrapper.col_to_idx = {c: i for i, c in enumerate(tv_cols)}
    else:
        # if mismatch, prefer meta tv_cols but warn
        if list(scaler_wrapper.numeric_cols) != list(tv_cols):
            logger.warning("Warning: numeric_cols in scaler differs from meta tv_cols. Overriding scaler numeric_cols with meta tv_cols.")
            scaler_wrapper.numeric_cols = list(tv_cols)
            scaler_wrapper.col_to_idx = {c: i for i, c in enumerate(tv_cols)}

    # ensure scaler_wrapper has target info
    if not hasattr(scaler_wrapper, "target_idxs"):
        scaler_wrapper.target_cols = list(target_cols_meta)
        scaler_wrapper.target_idxs = [scaler_wrapper.col_to_idx[t] for t in scaler_wrapper.target_cols]

    # decide final TARGETS order (use meta ordering)
    TARGETS = list(scaler_wrapper.target_cols)

    # -------- split real vs future (history up to 2025-12) --------
    # real_df will include all rows up to and including 2025-12-01
    cut_date = pd.Timestamp("2025-12-01")
    real_df = df[df[DATE_COL] <= cut_date].copy().reset_index(drop=True)
    future_df = df[df[DATE_COL] > cut_date].copy().reset_index(drop=True)

    # If user doesn't have future rows, create future stub for 2026-01 -> 2035-12
    if len(future_df) == 0:
        logger.info("No future rows found in parquet. Creating future stub for 2026-01 to 2035-12 for all wards.")
        wards = real_df[ID_COL].unique()
        future_dates = pd.date_range("2026-01-01", "2035-12-01", freq="MS")
        rows = []
        for w in wards:
            for d in future_dates:
                rows.append({ID_COL: w, DATE_COL: d})
        future_df = pd.DataFrame(rows)

    logger.info(
        f"Real rows: {len(real_df)} | Future rows: {len(future_df)}")
    logger.info(f"Date ranges -> real: {real_df[DATE_COL].min()} -> {real_df[DATE_COL].max()}")

    # Sanity: ensure tv_cols exist
    for c in scaler_wrapper.numeric_cols:
        if c not in real_df.columns:
            raise KeyError(f"Numeric column {c} (from meta/scaler) not found in dataframe. Check preprocessing output. Missing: {c}")
        
    

    # ----------------------------
    # Prepare sliding window dataset
    # ----------------------------
    train_ds = SlidingWindowDataset(real_df, scaler_wrapper.numeric_cols, TARGETS, id_col=ID_COL, date_col=DATE_COL,
                                     input_window=INPUT_WINDOW, horizon=TRAIN_HORIZON, scaler=scaler_wrapper.x_scaler)
    n_windows = len(train_ds)
    if n_windows == 0:
        raise RuntimeError("No sliding windows constructed. Check INPUT_WINDOW/HORIZON vs data length per ward.")
    # ---- TIME-SAFE SPLIT ----
    train_indices = []
    val_indices = []

    VAL_CUTOFF = pd.Timestamp("2024-01-01")

    for i, (_, last_date) in enumerate(train_ds.meta):
        if last_date < VAL_CUTOFF:
            train_indices.append(i)
        else:
            val_indices.append(i)

    train_subset = torch.utils.data.Subset(train_ds, train_indices)
    val_subset = torch.utils.data.Subset(train_ds, val_indices)

    logger.info(
        f"Train windows: {len(train_indices)}, "
        f"Validation windows: {len(val_indices)}"
    )


    # DataLoaders - safe defaults for portability
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=False)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=False)

    # ----------------------------
    # Train
    # ----------------------------
    pl.seed_everything(SEED)
    n_features = len(scaler_wrapper.numeric_cols)
    n_targets = len(TARGETS)

    model = LitLSTM(n_features=n_features, n_targets=n_targets, hidden_size=HIDDEN_SIZE,
                     num_layers=NUM_LAYERS, lr=LR, horizon=TRAIN_HORIZON)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ARTIFACT_DIR),
        filename="lstm_global_{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=6, mode="min")

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if DEVICE == "cuda" else "cpu",
        devices=1 if DEVICE == "cuda" else None,
        callbacks=[checkpoint_callback, early_stop],
        log_every_n_steps=10,
        enable_progress_bar=True,
        gradient_clip_val=1.0
    )

    logger.info("Starting training... (this may take some time)")
    trainer.fit(model, train_loader, val_loader)

    ckpt_path = checkpoint_callback.best_model_path
    logger.info(f"Best checkpoint saved at: {ckpt_path}")

    # Save some trainer metrics
    try:
        # trainer.callback_metrics contains latest metrics
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        with open(ARTIFACT_DIR / "train_metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info(f"Saved training metrics to: {ARTIFACT_DIR / 'train_metrics.json'}")
    except Exception as e:
        logger.info(f"Could not save trainer metrics: {e}")

# NOTE:
# Model operates on raw scaled temporal features.
# No explicit lag or rolling features are used for LSTM.

# autoregressive forecasting. Instead, the model operates on scaled feature
# vectors propagated forward. This approximation trades feature fidelity
# for computational efficiency and is discussed in the limitations section.

    # ----------------------------
    # Prediction: autoregressive rollforward 2026-2035 using wrapper.inverse_y
    # ----------------------------
    unique_future_dates = sorted(pd.to_datetime(future_df[DATE_COL].unique()))
    TOTAL_HORIZON = len(unique_future_dates)
    logger.info(f"Total months to forecast (future): {TOTAL_HORIZON}")

    # load best model
    best_model = LitLSTM.load_from_checkpoint(ckpt_path)
    best_model.eval()
    best_model.to(DEVICE)

    rows_out = []
    # We'll iterate ward-by-ward using last INPUT_WINDOW months of real_df
    with torch.no_grad():
        for ward, g in real_df.groupby(ID_COL):
            g = g.sort_values(DATE_COL).reset_index(drop=True)
            if len(g) < INPUT_WINDOW:
                # skip wards with insufficient history
                logger.info(f"Skipping ward {ward} - insufficient history rows {len(g)} < {INPUT_WINDOW}")
                continue
            last_window = g[scaler_wrapper.numeric_cols].values[-INPUT_WINDOW:].copy()  # shape (input_window, n_features)
            # scale using saved scaler
            last_window_scaled = scaler_wrapper.x_scaler.transform(last_window)

            generated = []
            buffer = last_window_scaled.copy()
            months_generated = 0
            while months_generated < TOTAL_HORIZON:
                x = torch.tensor(buffer[-INPUT_WINDOW:].reshape(1, INPUT_WINDOW, n_features), dtype=torch.float32).to(DEVICE)
                y_hat = best_model(x)  # (1, TRAIN_HORIZON, n_targets) in scaled space
                pred = y_hat.squeeze(0).cpu().numpy()  # (TRAIN_HORIZON, n_targets)
                # Use wrapper.inverse_y to get predictions in original scale (B=1)
                inv_batch = scaler_wrapper.inverse_y(pred.reshape(1, pred.shape[0], pred.shape[1]))  # (1, H, T)
                inv_batch = inv_batch.squeeze(0)  # (H, T)

                for k in range(inv_batch.shape[0]):
                    inv_targets = inv_batch[k]  # (n_targets,) in original scale
                    # safety clean and clipping per variable name
                    inv_targets = np.nan_to_num(inv_targets, nan=0.0, posinf=0.0, neginf=0.0)
                    for j, tcol in enumerate(TARGETS):
                        if tcol in ("electricity_demand", "water_demand"):
                            if inv_targets[j] < 0:
                                inv_targets[j] = 0.0
                        if tcol == "congestion_index":
                            inv_targets[j] = float(np.clip(inv_targets[j], 0.0, 1.5))
                        if tcol == "pm25":
                            inv_targets[j] = float(max(inv_targets[j], 2.0))
                    # Build new raw row (original numeric_cols order) to append to buffer.
                    last_raw_row = scaler_wrapper.x_scaler.inverse_transform(buffer[-1].reshape(1, -1))[0]
                    new_row_raw = last_raw_row.copy()
                    # fill target positions in full numeric_cols order
                    for ti, idx in enumerate(scaler_wrapper.target_idxs):
                        new_row_raw[idx] = inv_targets[ti]


                    # update derived lag features if present (simple approach: recompute lags/rolling approximations is complex;
                    # here we rely on how dataset engineered features are structured in numeric_cols - they may contain lags computed from raw series.
                    # We append the new row and rely on the next iteration using scaled version.)
                    new_row_scaled = scaler_wrapper.x_scaler.transform(new_row_raw.reshape(1, -1))[0]
                    buffer = np.vstack([buffer, new_row_scaled])
                    generated.append(new_row_raw[scaler_wrapper.target_idxs])  # store only target original-scale values
                    months_generated += 1
                    if months_generated >= TOTAL_HORIZON:
                        break

            # attach ward_future_dates to generated
            ward_future_dates = unique_future_dates
            if len(generated) < len(ward_future_dates):
                # safety: pad last value
                last_val = generated[-1] if len(generated) > 0 else np.zeros(len(scaler_wrapper.target_idxs))
                while len(generated) < len(ward_future_dates):
                    generated.append(last_val)

            for i, dt in enumerate(ward_future_dates):
                out_row = {ID_COL: ward, DATE_COL: pd.Timestamp(dt)}
                for j, tcol in enumerate(TARGETS):
                    out_row[tcol] = float(generated[i][j])
                rows_out.append(out_row)

    pred_df = pd.DataFrame(rows_out)
    pred_csv = RESULTS_DIR / "lstm_predictions_2026_2035.csv"
    pred_df.to_csv(pred_csv, index=False)
    logger.info(f"Saved predictions to:{pred_csv}")

    run_meta = {
        "experiment": EXPERIMENT_NAME,
        "timestamp": EXPERIMENT_DATE,
        "git_commit": GIT_COMMIT,
        "input_window": INPUT_WINDOW,
        "train_horizon": TRAIN_HORIZON,
        "targets": TARGETS,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "device": DEVICE
    }

    with open(BASE_DIR / "artifacts/dl/metadata/lstm_run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)


    # ----------------------------
    # Evaluate (if actuals for early future exist in df) - compute metrics against any actual future months present
    # ----------------------------
    # merge with actual df if any overlap
    df_future_actual = df[(df[DATE_COL] >= pd.Timestamp("2026-01-01")) & (df[DATE_COL] <= pd.Timestamp("2035-12-01"))]
    if not df_future_actual.empty:
        actual = df_future_actual[[ID_COL, DATE_COL] + TARGETS]
        merged = pred_df.merge(actual, on=[ID_COL, DATE_COL], how="inner", suffixes=("_pred", "_true"))
        if not merged.empty:
            metrics = {}
            for t in TARGETS:
                yhat = merged[f"{t}_pred"].values
                y = merged[f"{t}_true"].values
                metrics[t] = {"MAE": float(mean_absolute_error(y, yhat)), "RMSE": float(mean_squared_error(y, yhat, squared=False))}
            with open(ARTIFACT_DIR / "lstm_eval_future_metrics.json", "w") as fh:
                json.dump(metrics, fh, indent=2)
            logger.info(f"Saved future-eval metrics to: { ARTIFACT_DIR / 'lstm_eval_future_metrics.json'}")

    if "merged" not in locals() or merged.empty:
        logger.warning("No future ground truth available — skipping error analyses")
        


    # ----------------------------
# Horizon-wise error growth
# ----------------------------
    horizon_errors = {}

    for t in TARGETS:
        horizon_errors[t] = {}
        for h in range(1, TRAIN_HORIZON + 1):
            sub = merged.groupby(ID_COL).nth(h - 1)
            horizon_errors[t][h] = float(
            mean_absolute_error(sub[f"{t}_true"], sub[f"{t}_pred"])
        )

    with open(RESULTS_DIR / "horizon_error_growth.json", "w") as f:
        json.dump(horizon_errors, f, indent=2)
    
    # ----------------------------
# Ward-level generalization
# ----------------------------
    ward_errors = []

    for ward in merged[ID_COL].unique():
        sub = merged[merged[ID_COL] == ward]
        if len(sub) > 0:
            ward_errors.append(
                mean_absolute_error(sub[f"{TARGETS[0]}_true"], sub[f"{TARGETS[0]}_pred"])
            )

    ward_stats = {
        "mean": float(np.mean(ward_errors)),
        "std": float(np.std(ward_errors)),
        "min": float(np.min(ward_errors)),
        "max": float(np.max(ward_errors))
    }

    with open(RESULTS_DIR / "ward_generalization_stats.json", "w") as f:
        json.dump(ward_stats, f, indent=2)

    # ----------------------------
# Forecast smoothness analysis
# ----------------------------
    smoothness = {}

    for t in TARGETS:
        diff_true = np.diff(merged[f"{t}_true"].values)
        diff_pred = np.diff(merged[f"{t}_pred"].values)
        smoothness[t] = {
            "true_volatility": float(np.mean(np.abs(diff_true))),
            "pred_volatility": float(np.mean(np.abs(diff_pred)))
        }

    with open(RESULTS_DIR / "forecast_smoothness.json", "w") as f:
        json.dump(smoothness, f, indent=2)

    # ----------------------------
    # Example plots for several wards
    # ----------------------------
    sample_wards = list(pred_df[ID_COL].unique())[:6]  # first 6 wards
    processed_df = real_df[[ID_COL, DATE_COL] + TARGETS].copy()
    for w in sample_wards:
        rp = processed_df[processed_df[ID_COL] == w].sort_values(DATE_COL)
        pp = pred_df[pred_df[ID_COL] == w].sort_values(DATE_COL)
        fig, axes = plt.subplots(len(TARGETS), 1, figsize=(10, 3 * len(TARGETS)), sharex=True)
        for i, t in enumerate(TARGETS):
            ax = axes[i]
            # plot last history (INPUT_WINDOW months)
            hist = rp.tail(INPUT_WINDOW)
            if not hist.empty:
                ax.plot(hist[DATE_COL], hist[t], label="history (last window)")
            ax.plot(pp[DATE_COL], pp[t], label="prediction", linestyle="--")
            ax.set_title(f"{t} - ward {w}")
            ax.legend()
        plt.tight_layout()
        fname = PLOTS_DIR / f"ward_{w}_forecast.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
    logger.info(f"Saved example plots to: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
