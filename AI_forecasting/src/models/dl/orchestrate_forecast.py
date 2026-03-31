#!/usr/bin/env python3
"""
orchestrate_forecast_cascaded_FINAL_bengaluru.py

Final corrected script:
- Integrates Bengaluru-specific realistic bounds, rainfall prior, population logistic growth and cascade constraints
- Keeps defensive coding for TFT/NBeats dataset creation and inference
- Adaptive smoothing, ensemble weights, dataset validation, and plotting samples retained
"""
import json
from pathlib import Path
from typing import Any, Mapping
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated")

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)

import hashlib
import logging
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import pytorch_lightning as pl

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, NBeats
from pytorch_forecasting.metrics import SMAPE
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor


BASE_DIR = Path("AI_forecasting")
ENRICHED_PATH = BASE_DIR / "data/input/masterdata_for_modeling.parquet"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
RESULTS_DIR   = BASE_DIR / "results"
REPORTS_DIR   = BASE_DIR / "reports"
LOGS_DIR      = BASE_DIR / "logs"
EVAL_DIR      = BASE_DIR / "evaluation"

# DL specific
ADV_ARTIFACTS_DIR = ARTIFACTS_DIR / "advanced"
ADV_RESULTS_DIR   = RESULTS_DIR / "advanced"
ADV_REPORTS_DIR   = REPORTS_DIR / "advanced"


for d in [
    ARTIFACTS_DIR, RESULTS_DIR, REPORTS_DIR, LOGS_DIR, EVAL_DIR,
    ADV_ARTIFACTS_DIR, ADV_RESULTS_DIR, ADV_REPORTS_DIR
]:
    d.mkdir(parents=True, exist_ok=True)

OUTDIR = ADV_RESULTS_DIR / "ensemble_bengaluru_forecasting"
OUTDIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = LOGS_DIR / "advanced"
LOG_DIR.mkdir(parents=True, exist_ok=True)

run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTDIR / f"run_{run_id}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Reproducibility
# ----------------------------
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
pl.seed_everything(GLOBAL_SEED, workers=True)



def file_hash(path: Path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"orchestration_{run_id}.log"),
        logging.StreamHandler()
    ]
)

log = logging.getLogger("FORECAST_ORCHESTRATOR")


MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 12
MIN_ENCODER_LENGTH = 30
BATCH_SIZE = 48
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 6

USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"

FUTURE_START = pd.Timestamp("2026-01-01")
FUTURE_END = pd.Timestamp("2035-12-01")
TOTAL_FUTURE_MONTHS = 12 * 10

PLOTS_DIR = ADV_REPORTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEPENDENCY_ORDER = [
    "population",
    "rainfall",
    "electricity_demand",
    "water_demand",
    "congestion_index",
    "pm25",
]

VARIABLE_PLAN = {
    "population": "prior_based",
    "rainfall": "prior_based",
    "electricity_demand": "ensemble",
    "water_demand": "ensemble",
    "congestion_index": "ensemble",
    "pm25": "ensemble",
}

ENSEMBLE_WEIGHTS = {
    "electricity_demand": (0.6, 0.4),
    "water_demand": (0.6, 0.4),
    "pm25": (0.5, 0.5),
    "congestion_index": (0.55, 0.45),
    "default": (0.6, 0.4)
}

# ----------------------------
# BENGALURU REALISTIC BOUNDS & ADAPTIVE CAPS (user-provided)
# ----------------------------
REALISTIC_BOUNDS = {
    "population": (8000, 500_000),
    "rainfall": (0.0, 400.0),
    "pm25": (25.0, 180.0),
    "congestion_index": (0.1, 2.5),
    "electricity_demand": (5000.0, 2_500_000.0),
    "water_demand": (2000.0, 1_500_000.0),
}

MAX_PCT_CHANGES = {
    "population": 0.025,
    "electricity_demand": 0.15,
    "water_demand": 0.12,
    "congestion_index": 0.20,
    "pm25": 0.25,
    "rainfall": None,
}

if USE_GPU:
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

# ----------------------------
# DeviceSafeWrapper
# ----------------------------
class DeviceSafeWrapper(pl.LightningModule):
    def __init__(self, inner: pl.LightningModule):
        super().__init__()
        self.inner = inner

    def _move(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        if isinstance(obj, Mapping):
            return {k: self._move(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._move(x) for x in obj)
        return obj

    def on_fit_start(self):
        try:
            if self.device.type != "cpu":
                self.inner.to(self.device)
        except Exception:
            pass
        try:
            self.inner.trainer = self.trainer
        except Exception:
            pass

    def _get_prediction(self, x):
        output = self.inner(x)
        return output[0] if isinstance(output, tuple) else output

    def training_step(self, batch, batch_idx):
        old_log = getattr(self.inner, "log", None)
        self.inner.log = lambda *a, **k: None
        try:
            out = self.inner.training_step(batch, batch_idx)
        finally:
            self.inner.log = old_log
        loss = out["loss"] if isinstance(out, dict) and "loss" in out else out
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        old_log = getattr(self.inner, "log", None)
        self.inner.log = lambda *a, **k: None
        try:
            out = self.inner.validation_step(batch, batch_idx)
        finally:
            self.inner.log = old_log
        loss = out["loss"] if isinstance(out, dict) and "loss" in out else out
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.inner.configure_optimizers() if hasattr(self.inner, "configure_optimizers") else None

    def forward(self, x, *args, **kwargs):
        return self._get_prediction(self._move(x))

# ----------------------------
# Feature Engineering
# ----------------------------
def ensure_time_idx(df, date_col="date"):
    df = df.copy()
    if pd.api.types.is_integer_dtype(df[date_col].dtype):
        df[date_col] = pd.to_datetime(df[date_col], unit="ms", errors="coerce")
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    global_min = df[date_col].min()
    if pd.isnull(global_min):
        raise ValueError("Date parsing failed")
    df["time_idx"] = ((df[date_col].dt.year - global_min.year) * 12 +
                      (df[date_col].dt.month - global_min.month)).astype(int)
    return df

def add_cyclical_features(df):
    df = df.copy()
    if "month" not in df.columns:
        df["month"] = pd.to_datetime(df["date"]).dt.month
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    df["quarter"] = pd.to_datetime(df["date"]).dt.quarter
    df["quarter_sin"] = np.sin(2 * np.pi * (df["quarter"] - 1) / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * (df["quarter"] - 1) / 4)
    return df

def create_lag_and_rolling_features(df, cols, lags=(1,2,3,6,12), windows=(3,6,12,24)):
    df = df.copy().sort_values(["ward_id", "time_idx"]).reset_index(drop=True)
    cols = [c for c in cols if c in df.columns]

    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby("ward_id")[col].shift(lag)

        for w in windows:
            df[f"{col}_roll_mean_{w}"] = (
                df.groupby("ward_id")[col]
                .rolling(window=w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            df[f"{col}_roll_std_{w}"] = (
                df.groupby("ward_id")[col]
                .rolling(window=w, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
                .fillna(0.0)
            )

        df[f"{col}_ema_12"] = (
            df.groupby("ward_id")[col]
            .transform(lambda x: x.ewm(span=12, adjust=False).mean())
        )

    return df

def fill_engineered_nans(df, engineered_cols):
    for c in engineered_cols:
        if c not in df.columns:
            continue
        df[c] = df.groupby("ward_id")[c].transform(lambda s: s.ffill().bfill())
        if df[c].isna().any():
            df[c] = df[c].fillna(df.groupby("ward_id")[c].transform("median"))
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
        if df[c].isna().any():
            df[c] = df[c].fillna(0)
    return df

# ----------------------------
# BENGALURU RAINFALL PRIOR (user-provided)
# ----------------------------
def get_rainfall_prior(base_last_date, n_months, ward_history=None):
    """
    Bengaluru-specific rainfall with bimodal monsoon pattern
    Pre-monsoon (Apr-May), SW monsoon (Jun-Sep), NE monsoon (Oct-Nov)
    """
    seasonal = [
        15,   # Jan - dry winter
        12,   # Feb - dry
        20,   # Mar - occasional showers
        60,   # Apr - pre-monsoon showers begin
        110,  # May - pre-monsoon peak (thunderstorms)
        90,   # Jun - SW monsoon arrives
        115,  # Jul - SW monsoon
        130,  # Aug - SW monsoon peak
        160,  # Sep - SW monsoon tail end
        180,  # Oct - NE monsoon peak
        95,   # Nov - NE monsoon continues
        35    # Dec - retreating monsoon
    ]
    
    scale = 1.0
    if ward_history is not None and "rainfall" in ward_history.columns and len(ward_history) >= 12:
        hist_mean = ward_history["rainfall"].mean()
        clim_mean = np.mean(seasonal)
        if clim_mean > 0:
            scale = np.clip(hist_mean / clim_mean, 0.7, 1.3)  # Limit scaling
    
    out = []
    for i in range(n_months):
        dt = base_last_date + pd.DateOffset(months=(i+1))
        base = seasonal[(dt.month - 1) % 12] * scale
        
        # Variability increases with base amount
        noise = np.random.normal(0, max(2.0, base * 0.18))
        val = max(0.0, base + noise)
        val = float(np.clip(val, REALISTIC_BOUNDS["rainfall"][0], REALISTIC_BOUNDS["rainfall"][1]))
        out.append(val)
    return out

# ----------------------------
# BENGALURU POPULATION PRIOR (user-provided)
# ----------------------------
def get_population_prior_logistic(base_pop, n_months, annual_r=0.022, carrying_factor=3.5):
    """
    Bengaluru population growth: ~2-2.5% annually
    Accounts for tech sector migration and urban expansion
    """
    base_pop = float(base_pop)
    K = base_pop * carrying_factor
    monthly_r = annual_r / 12.0
    out = []
    P = base_pop
    
    for i in range(n_months):
        # Stronger seasonal migration (Jan-Feb, Jul-Aug job changes)
        month = (i % 12) + 1
        if month in [1, 2, 7, 8]:
            season = 1.015
        else:
            season = 1.0 + 0.006 * np.sin(2 * np.pi * (i % 12) / 12.0)
        
        noise = np.random.normal(0, base_pop * 0.0008)
        dP = monthly_r * P * (1 - (P / K)) * season
        P = P + dP + noise
        P = float(np.clip(P, REALISTIC_BOUNDS["population"][0], REALISTIC_BOUNDS["population"][1]))
        out.append(P)
    return out

def clip_realistic(val, var):
    if var in REALISTIC_BOUNDS:
        lo, hi = REALISTIC_BOUNDS[var]
        return float(np.clip(val, lo, hi))
    return float(val)

# ----------------------------
# BENGALURU apply_cascade_constraints (user-provided)
# ----------------------------
def apply_cascade_constraints(pred_val, target, month_idx, ward, global_buffer, base_value, historical_avg=None):
    """
    BENGALURU-SPECIFIC physics-based constraints
    """
    val = float(pred_val)
    val = np.clip(val, REALISTIC_BOUNDS[target][0], REALISTIC_BOUNDS[target][1])
    
    if global_buffer is None or ward not in global_buffer:
        return val
    
    buf = global_buffer.setdefault(ward, {})

    
    def buf_get(varname, idx, default=None):
        if varname in buf:
            arr = buf[varname]
            if 0 <= idx < len(arr) and arr[idx] is not None:
                return float(arr[idx])
        return default
    
    month = (month_idx % 12) + 1
    
    # ========== ELECTRICITY DEMAND ==========
    if target == "electricity_demand":
        base_elec = historical_avg if historical_avg and historical_avg > 0 else val
        
        # Population scaling
        pop_base = buf_get("population", 0, None)
        pop_cur = buf_get("population", month_idx, pop_base)
        pop_factor = 1.0
        if pop_base and pop_base > 0 and pop_cur:
            pop_ratio = pop_cur / pop_base
            pop_factor = pop_ratio ** 0.80  # Strong correlation
        
        # Bengaluru climate seasonality
        seasonal = 1.0
        if month in [3, 4, 5]:  # Peak summer - heavy AC load
            seasonal = 1.35
        elif month in [6, 7, 8]:  # Monsoon - moderate AC
            seasonal = 1.10
        elif month in [12, 1]:  # Mild winter - moderate heating
            seasonal = 1.05
        
        # Economic growth (tech sector boom)
        monthly_econ = (1.045) ** (month_idx / 12.0)  # 4.5% annual growth
        
        val_adj = base_elec * pop_factor * seasonal * monthly_econ
        val_adj = np.clip(val_adj, REALISTIC_BOUNDS[target][0], REALISTIC_BOUNDS[target][1])
        return float(val_adj)
    
    # ========== WATER DEMAND ==========
    if target == "water_demand":
        base_water = historical_avg if historical_avg and historical_avg > 0 else val
        
        # Population scaling (high correlation)
        pop_base = buf_get("population", 0, None)
        pop_cur = buf_get("population", month_idx, pop_base)
        pop_factor = 1.0
        if pop_base and pop_base > 0 and pop_cur:
            pop_ratio = pop_cur / pop_base
            pop_factor = pop_ratio ** 0.92  # Very strong correlation
        
        # Rainfall effect
        rain = buf_get("rainfall", month_idx, 0.0) or 0.0
        rainfall_factor = 1.0
        if rain > 150:
            rainfall_factor = 0.85  # Good recharge
        elif rain < 30:
            rainfall_factor = 1.20  # Higher demand (less natural water)
        
        # Bengaluru summer water crisis
        summer_stress = 1.0
        if month in [3, 4, 5]:
            summer_stress = 1.28
            if rain < 20:  # Severe shortage
                summer_stress = 1.45
        
        val_adj = base_water * pop_factor * rainfall_factor * summer_stress
        val_adj = np.clip(val_adj, REALISTIC_BOUNDS[target][0], REALISTIC_BOUNDS[target][1])
        return float(val_adj)
    
    # ========== CONGESTION INDEX ==========
    if target == "congestion_index":
        base_cong = base_value if base_value and base_value > 0 else 0.5
        
        # Population pressure (accelerating beyond capacity)
        pop_base = buf_get("population", 0, None)
        pop_cur = buf_get("population", month_idx, pop_base)
        pop_factor = 1.0
        if pop_base and pop_base > 0 and pop_cur:
            pop_ratio = pop_cur / pop_base
            if pop_ratio <= 1.2:
                pop_factor = pop_ratio ** 0.75
            else:
                # Infrastructure strain accelerates
                pop_factor = (1.2 ** 0.75) * ((pop_ratio / 1.2) ** 1.35)
        
        # Rainfall impact (flooding, road damage)
        rain = buf_get("rainfall", month_idx, 0.0) or 0.0
        if rain > 180:
            pop_factor *= 1.15  # Heavy rain causes congestion
        elif rain > 250:
            pop_factor *= 1.25  # Flooding
        
        # Seasonal patterns (festival season, tech park cycles)
        if month in [9, 10, 11]:  # Post-monsoon, Diwali season
            pop_factor *= 1.12
        elif month in [1, 2]:  # Start of year, budget season
            pop_factor *= 1.08
        
        val_adj = base_cong * pop_factor
        val_adj = np.clip(val_adj, REALISTIC_BOUNDS[target][0], REALISTIC_BOUNDS[target][1])
        return float(val_adj)
    
    # ========== PM2.5 ==========
    if target == "pm25":

        # FIX 1: Use historical average as stable baseline
        if historical_avg and historical_avg > 0:
            base_pm = float(historical_avg)
        else:
            base_pm = 55.0  # Typical Bengaluru baseline

        # FIX 2: Long-term upward trend due to urbanization
        years_elapsed = month_idx / 12.0
        growth_factor = 1.0 + (0.015 * years_elapsed)

        # Multi-factor drivers
        cong = buf_get("congestion_index", month_idx, 0.5) or 0.5
        elec_cur = buf_get("electricity_demand", month_idx, None)
        elec_base = buf_get("electricity_demand", 0, elec_cur) or 1.0
        pop_cur = buf_get("population", month_idx, None)
        pop_base = buf_get("population", 0, pop_cur) or 1.0

        cong_factor = (cong / 0.5) ** 0.65
        elec_ratio = (elec_cur / elec_base) if elec_base > 0 else 1.0
        elec_factor = elec_ratio ** 0.40
        pop_ratio = (pop_cur / pop_base) if pop_base > 0 else 1.0
        pop_factor = pop_ratio ** 0.35

        pm = base_pm * (
            0.30 * cong_factor +
            0.30 * elec_factor +
            0.30 * pop_factor +
            0.10
        ) * growth_factor

        # FIX 3: Rainfall washout (less aggressive)
        rain = buf_get("rainfall", month_idx, 0.0) or 0.0
        if rain > 200:
            pm *= 0.60
        elif rain > 120:
            pm *= 0.70
        elif rain > 60:
            pm *= 0.82

        # Bengaluru seasonal effects
        if month in [12, 1, 2]:
            pm *= 1.20
        elif month in [3, 4, 5]:
            pm *= 1.35
        elif month in [10, 11]:
            pm *= 1.25

        # FIX 4: Prevent unrealistic lows
        pm = max(pm, base_pm * 0.7)

        pm = np.clip(pm, REALISTIC_BOUNDS["pm25"][0], REALISTIC_BOUNDS["pm25"][1])
        return float(pm)

    
    return float(val)

# ----------------------------
# Helper functions (unchanged)
# ----------------------------
def filter_groups_by_continuity(df, min_length):
    valid_wards = []
    for ward, grp in df.groupby("ward_id"):
        grp = grp.sort_values("time_idx")
        diffs = grp["time_idx"].diff().fillna(1).values
        seg_lengths = []
        cur = 1
        for d in diffs[1:]:
            cur = cur + 1 if d == 1 else 1
            if d != 1:
                seg_lengths.append(cur - 1)
        seg_lengths.append(cur)
        if seg_lengths and max(seg_lengths) >= min_length:
            valid_wards.append(ward)
    return df[df["ward_id"].isin(set(valid_wards))].copy()

def build_validation_windows(df, valid_wards, max_enc, max_pred):
    windows = []
    win_len = max_enc + max_pred
    for w in valid_wards:
        grp = df[df["ward_id"] == w].sort_values("time_idx")
        if grp.empty:
            continue
        last_idx = int(grp["time_idx"].max())
        start_idx = max(grp["time_idx"].min(), last_idx - win_len + 1)
        win = grp[(grp["time_idx"] >= start_idx) & (grp["time_idx"] <= last_idx)].copy()
        if not win.empty and win["time_idx"].nunique() >= win_len:
            windows.append(win)
    return pd.concat(windows, ignore_index=True) if windows else df.iloc[:0].copy()

def build_datasets(enriched_path, target, max_encoder_length, max_prediction_length,
                   min_encoder_length, batch_size, use_gpu):

    p = Path(enriched_path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {enriched_path}")

    df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p, parse_dates=["date"])

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Available columns: {df.columns.tolist()}")

    if "ward_id" not in df.columns:
        candidates = [c for c in df.columns if "ward" in c.lower()]
        df = df.rename(columns={candidates[0]: "ward_id"}) if candidates else df

    df["ward_id"] = df["ward_id"].astype(str)
    df = ensure_time_idx(df, date_col="date")

    if "month" not in df.columns:
        df["month"] = pd.to_datetime(df["date"]).dt.month
    if "t" not in df.columns:
        df = df.sort_values(["ward_id", "time_idx"])
        df["t"] = df.groupby("ward_id").cumcount()

    df = add_cyclical_features(df)

    available_cols = [c for c in DEPENDENCY_ORDER if c in df.columns]
    log.info(f"[FEATURES] Engineering features for: {available_cols}")

    cols_to_engineer = [target] + [c for c in available_cols if c != target]
    df = create_lag_and_rolling_features(df, cols_to_engineer)

    engineered_cols = [c for c in df.columns if any(s in c for s in ("_lag","_roll_","_ema_"))]
    df = fill_engineered_nans(df, engineered_cols)

    exogs = [c for c in DEPENDENCY_ORDER if c in df.columns and c != target]
    core_feats = {target, "time_idx", "month", "t", "ward_id"}
    core_feats.update(exogs)
    df = df.dropna(subset=[c for c in core_feats if c in df.columns]).copy()

    min_required = max_encoder_length + max_prediction_length
    df_filtered = filter_groups_by_continuity(df, min_required)

    if df_filtered["ward_id"].nunique() == 0:
        raise RuntimeError("No wards after filtering")

    max_t = int(df_filtered["time_idx"].max())
    cutoff = max_t - max_prediction_length
    train_df = df_filtered[df_filtered["time_idx"] <= cutoff].copy()

    ward_counts = train_df.groupby("ward_id").size()
    valid_wards = ward_counts[ward_counts >= min_encoder_length].index.tolist()

    train_df = train_df[train_df["ward_id"].isin(valid_wards)].copy()
    val_df = build_validation_windows(df_filtered, valid_wards, max_encoder_length, max_prediction_length)

    log.info(f"[DATA] Train: {len(train_df)}, Val: {len(val_df)}, Wards: {len(valid_wards)}")


    tft_known = ["time_idx","month","month_sin","month_cos","quarter_sin","quarter_cos","t"] + exogs
    lag_features = [c for c in train_df.columns if any(c.endswith(f"_lag{i}") for i in [1,2,3,6,12])]
    roll_features = [c for c in train_df.columns if "_roll_" in c or "_ema_" in c]

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

    training = TimeSeriesDataSet(train_df, **tft_kwargs)

    # ---- FIX: Extract schema safely from TimeSeriesDataSet ----
    training_columns = (
        training.time_varying_known_reals
        + training.time_varying_unknown_reals
        + [training.target]
    )

    # Remove duplicates while preserving order
    training_columns = list(dict.fromkeys(training_columns))

# Dtypes are inferred dynamically during prediction
    training_dtypes = None

    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)


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

    nbeats_training = TimeSeriesDataSet(train_df, **nbeats_kwargs)
    nbeats_validation = TimeSeriesDataSet.from_dataset(nbeats_training, val_df, predict=True, stop_randomization=True)

    pin_mem = use_gpu and torch.cuda.is_available()
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0, pin_memory=pin_mem)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0, pin_memory=pin_mem)
    nbeats_train = nbeats_training.to_dataloader(train=True, batch_size=batch_size, num_workers=0, pin_memory=pin_mem)
    nbeats_val = nbeats_validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0, pin_memory=pin_mem)

    original_df = pd.read_parquet(enriched_path) if Path(enriched_path).suffix.lower() == ".parquet" else pd.read_csv(enriched_path, parse_dates=["date"])
    original_df["ward_id"] = original_df["ward_id"].astype(str)
    original_df = ensure_time_idx(original_df, date_col="date")
    if "month" not in original_df.columns:
        original_df["month"] = pd.to_datetime(original_df["date"]).dt.month
    if "t" not in original_df.columns:
        original_df = original_df.sort_values(["ward_id", "time_idx"])
        original_df["t"] = original_df.groupby("ward_id").cumcount()

    return ((training, train_loader, val_loader), (nbeats_training, nbeats_train, nbeats_val), df_filtered, original_df,training_columns,training_dtypes)

# ----------------------------
# Prediction utilities (unchanged)
# ----------------------------
def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(v, device) for v in obj)
    return obj

def predict_with_model(dataloader, wrapped_model, device):
    model = wrapped_model.inner if hasattr(wrapped_model, "inner") else wrapped_model
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) and len(batch) >= 1 else batch
            x = move_to_device(x, device)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            arr = out.detach().cpu().numpy() if isinstance(out, torch.Tensor) else np.array(out)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            for row in arr:
                preds.append(row.flatten().tolist())
    return preds[0] if preds else None

# ----------------------------
# Train and Forecast (unchanged flow, using Bengaluru functions)
# ----------------------------
def train_and_forecast_target(enriched_path, target, strategy="ensemble",
                              global_pred_buffer=None, original_df=None):

    MODEL_DIR = ADV_ARTIFACTS_DIR / target
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    
    ckpt_tft = MODEL_DIR / f"tft_{target}.ckpt"
    ckpt_nbeats = MODEL_DIR / f"nbeats_{target}.ckpt"

    # For prior-based variables, skip training and use priors directly
    if strategy == "prior_based":
        log.info(f"[PRIOR] Using prior-based forecasting for {target}")

        if original_df is None:
            original_df = pd.read_parquet(enriched_path) if Path(enriched_path).suffix.lower() == ".parquet" else pd.read_csv(enriched_path, parse_dates=["date"])
            original_df["ward_id"] = original_df["ward_id"].astype(str)
            original_df = ensure_time_idx(original_df)

        raw = original_df.sort_values(["ward_id", "time_idx"]).copy()
        all_preds = []

        for ward, grp in raw.groupby("ward_id"):
            grp = grp.sort_values("time_idx")
            if len(grp) < 12:
                continue

            base_last_date = pd.to_datetime(grp["date"].iloc[-1])
            base_time_idx = int(grp["time_idx"].iloc[-1])

            # Generate priors

            if target == "rainfall":
                ward_hist = grp[grp["date"] < FUTURE_START]
                vals = get_rainfall_prior(base_last_date, TOTAL_FUTURE_MONTHS, ward_hist)
            elif target == "population":
                base_pop = float(grp[target].iloc[-1])
                vals = get_population_prior_logistic(base_pop, TOTAL_FUTURE_MONTHS)
            else:
                continue

            # Create predictions
            for i, val in enumerate(vals):
                ti = base_time_idx + 1 + i
                dt = base_last_date + pd.DateOffset(months=(i+1))
                all_preds.append({
                    "ward_id": ward,
                    "date": dt,
                    "time_idx": ti,
                    target: val
                })

            # Update global buffer
            if global_pred_buffer is not None:
                buf = global_pred_buffer.setdefault(ward, {})
                buf[target] = vals

        if all_preds:
            preds_all = pd.DataFrame(all_preds)
            preds_all["date"] = pd.to_datetime(preds_all["date"])
            preds_all = preds_all[(preds_all["date"] >= FUTURE_START) & (preds_all["date"] <= FUTURE_END)]
            preds_all = preds_all.sort_values(["ward_id","date"])
            out_csv = RUN_DIR / f"{target}_forecast_2026_2035.csv"
            preds_all.to_csv(out_csv, index=False)
            preds_all[target] = preds_all[target].astype(float)
            log.info(f"[SAVE] {target} forecasts: {out_csv}")

            # ALSO save to advanced results for consistency with other targets
            adv_out_csv = ADV_RESULTS_DIR / f"{target}_forecast_2026_2035.csv"
            preds_all.to_csv(adv_out_csv, index=False)
            log.info(f"[SAVE] {target} forecasts (advanced): {adv_out_csv}")


            # Plot samples
            sample_wards = list(pd.unique(preds_all["ward_id"]))[:6]
            for w in sample_wards:
                hist = original_df[original_df["ward_id"]==w].sort_values("date")
                fut = preds_all[preds_all["ward_id"]==w].sort_values("date")
                plt.figure(figsize=(10,4))
                if not hist.empty:
                    plt.plot(hist["date"], hist[target], label="history")
                if not fut.empty:
                    plt.plot(fut["date"], fut[target], linestyle="--", label="forecast")
                plt.title(f"{target} - ward {w}")
                plt.xlabel("date")
                plt.legend()
                plt.tight_layout()
                plt.savefig(PLOTS_DIR / f"{target}_ward_{w}.png")
                plt.close()

            meta = {"target": target, "strategy": strategy, "trained_at": str(datetime.utcnow()), "use_gpu": bool(USE_GPU)}

            # ---- SAVE PRIOR ARTIFACT (for consistency & reproducibility) ----
            prior_artifact = {
                "target": target,
                "strategy": "prior_based",
                "description": "Domain-driven prior (no ML training)",
                "generated_at": datetime.utcnow().isoformat(),
                "seed": GLOBAL_SEED,
                "method": (
                    "logistic_growth" if target == "population"
                        else "climatological_monsoon_prior"
                    ),
                "bounds": REALISTIC_BOUNDS.get(target),
            }

            prior_dir = ADV_ARTIFACTS_DIR / target
            prior_dir.mkdir(parents=True, exist_ok=True)

            with open(prior_dir / "prior_artifact.json", "w") as f:
                json.dump(prior_artifact, f, indent=2)


            return preds_all

    # ----------------------------
    # MODEL-BASED path (ensemble)
    # ----------------------------
    log.info(f"[DATA PREP] Building datasets for {target}")
    (tft_ds, tft_train, tft_val), (nbeats_ds, nbeats_train, nbeats_val), df_filtered, orig_df,training_columns,training_dtypes = build_datasets(
        enriched_path, target, max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        min_encoder_length=MIN_ENCODER_LENGTH,
        batch_size=BATCH_SIZE,
        use_gpu=USE_GPU
    )

    if original_df is None:
        original_df = orig_df.copy()
    else:
        original_df = original_df.copy()

    # instantiate models (TFT + N-BEATS)
    tft_model = TemporalFusionTransformer.from_dataset(
        tft_ds,
        learning_rate=3e-4,
        hidden_size=128,
        attention_head_size=4,
        dropout=0.15,
        hidden_continuous_size=64,
        output_size=1,
        loss=SMAPE(),
        log_interval=10,
        reduce_on_plateau_patience=3,
    )
    nbeats_model = NBeats.from_dataset(
        nbeats_ds,
        learning_rate=3e-4,
        log_interval=10,
        weight_decay=1e-5,
        widths=[256, 128],
        backcast_loss_ratio=0.1,
        loss=SMAPE()
    )

    wrapped_tft = DeviceSafeWrapper(tft_model)
    wrapped_nbeats = DeviceSafeWrapper(nbeats_model)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, mode="min"),
        ModelCheckpoint(dirpath=str(MODEL_DIR),filename=f"{target}_{{epoch}}_{{val_loss:.3f}}", save_top_k=2, monitor="val_loss", mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        default_root_dir=str(MODEL_DIR),
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if USE_GPU else "cpu",
        devices=1 if USE_GPU else None,
        callbacks=callbacks,
        enable_checkpointing=True,
        logger=pl.loggers.TensorBoardLogger(save_dir=str(LOG_DIR), name="tb_logs"),
    )

    # Train (or reuse checkpoints if exist)
    if ckpt_nbeats.exists():
        log.info(f"[SKIP] N-BEATS checkpoint exists for {target} at {ckpt_nbeats}")
    else:
        log.info(f"[TRAIN] N-BEATS for {target}")
        trainer.fit(wrapped_nbeats, train_dataloaders=nbeats_train, val_dataloaders=nbeats_val)
        try:
            trainer.save_checkpoint(str(ckpt_nbeats))
        except Exception:
            pass

    if strategy == "ensemble":
        if ckpt_tft.exists():
            log.info(f"[SKIP] TFT checkpoint exists for {target} at {ckpt_tft}")
        else:
            log.info(f"[TRAIN] TFT for {target}")
            trainer.fit(wrapped_tft, train_dataloaders=tft_train, val_dataloaders=tft_val)
            try:
                trainer.save_checkpoint(str(ckpt_tft))
            except Exception:
                pass
     
    try:
        small_ds = tft_val.dataset[:2000]  # or sample by ward
        interpretation = wrapped_tft.inner.interpret_output(small_ds)
        interpretation.to_csv(ADV_REPORTS_DIR / f"{target}_tft_feature_importance.csv")
    except Exception as e:
        log.warning(f"TFT interpretation skipped: {e}")



    # Build last encoder windows from original_df for ALL wards
    raw = original_df.sort_values(["ward_id", "time_idx"]).copy()
    last_windows = {}
    for ward, grp in raw.groupby("ward_id"):
        grp = grp.sort_values("time_idx").reset_index(drop=True)
        if len(grp) >= MAX_ENCODER_LENGTH:
            last_windows[ward] = grp.iloc[-MAX_ENCODER_LENGTH:].copy().reset_index(drop=True)
        else:
            pad_needed = MAX_ENCODER_LENGTH - len(grp)
            if len(grp) == 0:
                continue
            pad = pd.concat([grp.head(1)] * pad_needed, ignore_index=True)
            last_windows[ward] = pd.concat([pad, grp], ignore_index=True).reset_index(drop=True)

    wards = sorted(list(last_windows.keys()))
    log.info(f"[PREDICT] Will produce forecasts for {len(wards)} wards for target {target}")

    all_preds = []

    # helper: safe required_known extraction
    required_known = []
    try:
        if getattr(tft_ds, "time_varying_known_reals", None):
            required_known.extend(list(tft_ds.time_varying_known_reals))
    except Exception:
        pass
    try:
        if getattr(tft_ds, "static_reals", None):
            required_known.extend(list(tft_ds.static_reals))
    except Exception:
        pass
    try:
        if getattr(tft_ds, "static_categoricals", None):
            required_known.extend(list(tft_ds.static_categoricals))
    except Exception:
        pass

    # produce predictions per ward with cascaded injection
    for ward in wards:
        enc = last_windows[ward].copy().reset_index(drop=True)
        base_time_idx = int(enc["time_idx"].iloc[-1])
        base_t = int(enc["t"].iloc[-1]) if "t" in enc.columns else len(enc) - 1
        base_last_date = pd.to_datetime(enc["date"].iloc[-1])
        future_start_idx = base_time_idx + 1
        n_to_gen = TOTAL_FUTURE_MONTHS

        # Precompute priors
        ward_hist = orig_df[orig_df["ward_id"] == ward]
        pop_base = float(ward_hist["population"].iloc[-1]) if "population" in ward_hist.columns and len(ward_hist)>0 else (enc["population"].iloc[-1] if "population" in enc.columns else 0.0)
        rainfall_prior_full = get_rainfall_prior(base_last_date, TOTAL_FUTURE_MONTHS, ward_hist)
        population_prior_full = get_population_prior_logistic(pop_base, TOTAL_FUTURE_MONTHS)

        ward_generated = []

        # iterative blocks
        while n_to_gen > 0:
            block = min(MAX_PREDICTION_LENGTH, n_to_gen)
            future_rows = []
            offset0 = future_start_idx - (base_time_idx + 1)
            for i in range(block):
                ti = future_start_idx + i
                dt = base_last_date + pd.DateOffset(months=(ti - base_time_idx))
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

                row[target] = float(enc[target].iloc[-1])

                for ex in ("population", "rainfall", "electricity_demand", "water_demand", "congestion_index", "pm25"):
                    if ex in enc.columns:
                        val = None
                        if global_pred_buffer is not None and ward in global_pred_buffer and ex in global_pred_buffer[ward]:
                            buf = global_pred_buffer[ward][ex]
                            idx = offset0 + i
                            if 0 <= idx < len(buf) and buf[idx] is not None:
                                val = float(buf[idx])
                        if val is None:
                            if ex == "rainfall":
                                val = rainfall_prior_full[offset0 + i] if 0 <= offset0 + i < len(rainfall_prior_full) else float(enc[ex].iloc[-1])
                            elif ex == "population":
                                val = population_prior_full[offset0 + i] if 0 <= offset0 + i < len(population_prior_full) else float(enc[ex].iloc[-1])
                            else:
                                try:
                                    val = float(enc[ex].iloc[-1])
                                except Exception:
                                    val = None
                        row[ex] = val
                future_rows.append(row)

            future_df = pd.DataFrame(future_rows)
            future_df = future_df.dropna(axis=1, how="all")
            predict_df = pd.concat([enc, future_df], ignore_index=True)



            # engineered lag/roll features
            engineered_cols = [c for c in enc.columns if any(s in c for s in ("_lag","_roll_","_ema_"))]
            for c in engineered_cols:
                if c not in predict_df.columns:
                    predict_df[c] = None
                predict_df[c] = predict_df[c].ffill().bfill()
                if predict_df[c].isna().any():
                    predict_df[c] = predict_df[c].fillna(enc[c].iloc[-1] if c in enc.columns else 0.0)

            # Create TimeSeriesDataSet for prediction
            
            # SAFETY: TFT cannot handle NaNs in target
            if predict_df[target].isna().any():
                predict_df[target] = predict_df[target].ffill().bfill()

            
            # ---- Efficiently add missing columns in one shot (NO fragmentation) ----
            missing_cols = [c for c in training_columns if c not in predict_df.columns]

            if missing_cols:
                predict_df = pd.concat(
                    [
                        predict_df,
                        pd.DataFrame(
                            0.0,
                            index=predict_df.index,
                            columns=missing_cols,
                        ),
                    ],
                    axis=1,
                )


            # Drop extra columns + enforce order
            predict_df = predict_df[training_columns]

            predict_df = predict_df.copy()

            # Final safety checkif list(predict_df.columns) != training_columns:
            if list(predict_df.columns) != training_columns:
                raise RuntimeError("TFT schema mismatch detected during prediction")





            try:
                tft_pred_ds = TimeSeriesDataSet.from_dataset(tft_ds, predict_df, predict=True, stop_randomization=True)
                tft_dl = tft_pred_ds.to_dataloader(train=False, batch_size=min(32, max(1, block)), num_workers=0, pin_memory=USE_GPU)
            except Exception:
                tft_dl = None

            try:
                nbeats_pred_ds = TimeSeriesDataSet.from_dataset(nbeats_ds, predict_df, predict=True, stop_randomization=True)
                nbeats_dl = nbeats_pred_ds.to_dataloader(train=False, batch_size=min(32, max(1, block)), num_workers=0, pin_memory=USE_GPU)
            except Exception:
                nbeats_dl = None

            # Predict safely
            if strategy == "ensemble" and tft_dl is not None:
                tft_block_pred = predict_with_model(tft_dl, wrapped_tft, DEVICE)
                if tft_block_pred is None:
                    tft_block_pred = [float(enc[target].iloc[-1])] * block
            else:
                tft_block_pred = [float(enc[target].iloc[-1])] * block

            if nbeats_dl is not None:
                nbeats_block_pred = predict_with_model(nbeats_dl, wrapped_nbeats, DEVICE)
                if nbeats_block_pred is None:
                    nbeats_block_pred = [float(enc[target].iloc[-1])] * block
            else:
                nbeats_block_pred = [float(enc[target].iloc[-1])] * block

            # Ensemble weighting
            if strategy == "nbeats_only":
                block_pred = nbeats_block_pred
            elif strategy == "ensemble":
                w_tft, w_nbeats = ENSEMBLE_WEIGHTS.get(target, ENSEMBLE_WEIGHTS["default"])
                s = float(w_tft + w_nbeats) if (w_tft + w_nbeats) != 0 else 1.0
                w_tft /= s; w_nbeats /= s
                block_pred = []
                for j in range(block):
                    a = tft_block_pred[j] if j < len(tft_block_pred) else float(enc[target].iloc[-1])
                    b = nbeats_block_pred[j] if j < len(nbeats_block_pred) else float(enc[target].iloc[-1])
                    val = w_tft * a + w_nbeats * b
                    block_pred.append(float(val))
            else:
                block_pred = nbeats_block_pred

            # Apply cascade constraints + adaptive smoothing
            for j in range(len(block_pred)):
                month_idx = (future_start_idx - (base_time_idx + 1)) + j
                base_val = float(enc[target].iloc[-1]) if target in enc.columns else None
                historical_avg = enc[target].mean() if target in enc.columns else None
                adj = apply_cascade_constraints(block_pred[j], target, month_idx, ward, global_pred_buffer, base_val, historical_avg=historical_avg)

                max_pct = MAX_PCT_CHANGES.get(target, None)
                if base_val is not None and max_pct is not None:
                    pct = (adj - base_val) / (base_val if base_val != 0 else 1.0)
                    pct = np.clip(pct, -max_pct, max_pct)
                    block_pred[j] = float(base_val * (1.0 + pct))
                else:
                    block_pred[j] = float(adj)

            # Append predictions
            for j in range(block):
                ti = future_start_idx + j
                dt = base_last_date + pd.DateOffset(months=(ti - base_time_idx))
                ward_generated.append({
                    "ward_id": ward,
                    "date": dt,
                    "time_idx": ti,
                    target: float(block_pred[j])
                })

            if global_pred_buffer is not None:
                offset_idx = future_start_idx - (base_time_idx + 1)
                buf = global_pred_buffer.setdefault(ward, {})
                arr = buf.setdefault(target, [None] * TOTAL_FUTURE_MONTHS)
                for j in range(block):
                    idx = offset_idx + j
                    if 0 <= idx < TOTAL_FUTURE_MONTHS:
                        arr[idx] = float(block_pred[j])

            # Update encoder with appended predicted rows
            append_rows = []
            last_known = enc.iloc[-1].to_dict()
            offset_idx = future_start_idx - (base_time_idx + 1)
            for j in range(block):
                ti = future_start_idx + j
                new_row = last_known.copy()
                new_row["time_idx"] = ti
                new_row["date"] = pd.to_datetime(base_last_date + pd.DateOffset(months=(ti - base_time_idx)))
                new_row["t"] = base_t + (ti - base_time_idx)
                new_row[target] = block_pred[j]
                for ex in ("population", "rainfall", "electricity_demand", "water_demand", "congestion_index", "pm25"):
                    if ex in enc.columns:
                        val = None
                        if global_pred_buffer is not None and ward in global_pred_buffer and ex in global_pred_buffer[ward]:
                            arr2 = global_pred_buffer[ward][ex]
                            idx2 = offset_idx + j
                            if 0 <= idx2 < len(arr2) and arr2[idx2] is not None:
                                val = float(arr2[idx2])
                        if val is None:
                            try:
                                val = float(enc[ex].iloc[-1])
                            except Exception:
                                val = None
                        new_row[ex] = val
                append_rows.append(new_row)

            if append_rows:
                enc = pd.concat([enc, pd.DataFrame(append_rows)], ignore_index=True, sort=False)
                if len(enc) > MAX_ENCODER_LENGTH:
                    enc = enc.iloc[-MAX_ENCODER_LENGTH:].reset_index(drop=True)

            future_start_idx += block
            n_to_gen -= block

        if len(ward_generated) > 0:
            all_preds.append(pd.DataFrame(ward_generated))

    if len(all_preds) == 0:
        log.info(f"[WARN] No predictions produced for target: {target}")
        return None

    preds_all = pd.concat(all_preds, ignore_index=True)
    preds_all["date"] = pd.to_datetime(preds_all["date"])
    preds_all = preds_all[(preds_all["date"] >= FUTURE_START) & (preds_all["date"] <= FUTURE_END)]
    preds_all = preds_all.sort_values(["ward_id","date"])
    out_csv = ADV_RESULTS_DIR / f"{target}_forecast_2026_2035.csv"
    preds_all.to_csv(out_csv, index=False)
    log.info(f"[SAVE] Saved {target} forecasts to {out_csv}")


    joblib.dump(
        {
            "tft_ckpt": str(ckpt_tft) if ckpt_tft.exists() else None,
            "nbeats_ckpt": str(ckpt_nbeats) if ckpt_nbeats.exists() else None,
            "trained_at": str(datetime.utcnow()),
            "target": target,
            "seed": GLOBAL_SEED
        },
        MODEL_DIR / "model_bundle.joblib"
    )


    # Plot sample wards for quick check
    sample_wards = list(pd.unique(preds_all["ward_id"]))[:6]
    for w in sample_wards:
        hist = original_df[original_df["ward_id"]==w].sort_values("date")
        fut = preds_all[preds_all["ward_id"]==w].sort_values("date")
        plt.figure(figsize=(10,4))
        if not hist.empty:
            plt.plot(hist["date"], hist[target], label="history")
        if not fut.empty:
            plt.plot(fut["date"], fut[target], linestyle="--", label="forecast")
        plt.title(f"{target} - ward {w}")
        plt.xlabel("date")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{target}_ward_{w}.png")
        plt.close()

    meta = {"target": target, "strategy": strategy, "trained_at": str(datetime.utcnow()), "use_gpu": bool(USE_GPU),
            "ensemble_weights": ENSEMBLE_WEIGHTS.get(target, ENSEMBLE_WEIGHTS["default"])}
    with open(RUN_DIR / "run_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    return preds_all

# ----------------------------
# Orchestration (main)
# ----------------------------
def main():
    global_pred_buffer = {}

    original_df = pd.read_parquet(ENRICHED_PATH) if Path(ENRICHED_PATH).suffix.lower()==".parquet" else pd.read_csv(ENRICHED_PATH, parse_dates=["date"])
    original_df["ward_id"] = original_df["ward_id"].astype(str)
    original_df = ensure_time_idx(original_df, date_col="date")
    if "month" not in original_df.columns:
        original_df["month"] = pd.to_datetime(original_df["date"]).dt.month
    if "t" not in original_df.columns:
        original_df = original_df.sort_values(["ward_id","time_idx"])
        original_df["t"] = original_df.groupby("ward_id").cumcount()

    data_fingerprint = {
        "input_path": str(ENRICHED_PATH),
        "md5": file_hash(ENRICHED_PATH),
    }

    with open(RUN_DIR / "data_fingerprint.json", "w") as f:
        json.dump(data_fingerprint, f, indent=2)



    config_snapshot = {
        "run_id": run_id,
        "seed": GLOBAL_SEED,
        "max_encoder_length": MAX_ENCODER_LENGTH,
        "max_prediction_length": MAX_PREDICTION_LENGTH,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "dependency_order": DEPENDENCY_ORDER,
        "variable_plan": VARIABLE_PLAN,
        "realistic_bounds": REALISTIC_BOUNDS,
    }

    with open(RUN_DIR / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)


    combined_list = []
    for var in DEPENDENCY_ORDER:
        strategy = VARIABLE_PLAN.get(var, "ensemble")
        log.info(f"\n==== Processing variable: {var}  strategy: {strategy} ====")
        preds = train_and_forecast_target(ENRICHED_PATH, var, strategy=strategy,
                                         global_pred_buffer=global_pred_buffer,
                                         original_df=original_df)
        if preds is not None:
            preds["date"] = pd.to_datetime(preds["date"]).dt.to_period("M").dt.to_timestamp()
            combined_list.append(preds[["ward_id","date",var]])

    if len(combined_list) == 0:
        log.info("No variable forecasts generated.")
        return

    combined = combined_list[0]
    for df in combined_list[1:]:
        combined = combined.merge(df, on=["ward_id","date"], how="outer")
    combined = combined.sort_values(["ward_id","date"])
    combined_csv = OUTDIR / "combined_forecast_2026_2035.csv"
    combined.to_csv(combined_csv, index=False)
    log.info(f"Saved combined forecast CSV to: {combined_csv}")

    log.info("Orchestration complete.")

    with open(ADV_RESULTS_DIR / "meta.json", "w") as f:
        json.dump(
            {
                "last_run_id": run_id,
                "timestamp": datetime.utcnow().isoformat(),
                "output_dir": str(RUN_DIR)
            },
            f,
            indent=2
        )


if __name__ == "__main__":
    main()
