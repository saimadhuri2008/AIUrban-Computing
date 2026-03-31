# src/preprocess/prepare_features.py
"""
Prepare canonical modeling dataset, scalers and meta for Phase 3 models.

Outputs:
 - data/processed/for_modeling.parquet
 - data/processed/ward_<ward_id>.parquet  (one file per ward)
 - results/forecasting/dl/preprocessing/scalers.pkl
 - results/forecasting/dl/preprocessing/meta.pkl

Assumes input file:
 - data/processed/historical_2014_2025_all_wards.parquet
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime


np.random.seed(42)

BASE_DIR = Path("AI_forecasting")

# ---------- CONFIG ----------
INPUT_PATH = BASE_DIR / "data/input/historical_2014_2025_all_wards.parquet"
OUT_PROCESSED = BASE_DIR /"data/input/masterdata_for_modeling.parquet"


PREP_OUT = BASE_DIR / "artifacts/dl/metadata"
PREP_OUT.mkdir(parents=True, exist_ok=True)

PLOT_OUT = BASE_DIR / "reports/dl/preprocessing"
PLOT_OUT.mkdir(parents=True, exist_ok=True)

LOG_DIR = BASE_DIR / "logs/preprocessing"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY = BASE_DIR / "reports/dl/summary"
SUMMARY.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "prepare_features.log"),
        logging.StreamHandler()
    ]
)

log = logging.getLogger("PREPROCESS_DL")


# numeric target cols we will use (six core variables)
TARGETS = ["electricity_demand", "water_demand", "congestion_index", "pm25", "population", "rainfall"]

# time window features to compute
LAGS = [1,2,3,6,12]
ROLLS = [3,6,12]

# date handling
DATE_COL = "date"
ID_COL = "ward_id"

# safe fill thresholds
MAX_MISSING_RATIO_COL = 0.2   # if >20% missing for a target column, raise flag

# ------------------------------

def month_sin_cos(df):
    df["month"] = df[DATE_COL].dt.month
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] / 12))
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] / 12))
    df["is_monsoon"] = df["month"].isin([6,7,8,9]).astype(int)
    return df

def compute_lags_rolls(g):
    # g is df for single ward sorted by date
    for col in TARGETS:
        for lag in LAGS:
            g[f"{col}_lag{lag}"] = g[col].shift(lag)
        for r in ROLLS:
            g[f"{col}_rmean_{r}"] = g[col].rolling(window=r, min_periods=1, center=False).mean()
        # diffs
        g[f"{col}_diff1"] = g[col] - g[col].shift(1)
        g[f"{col}_diff12"] = g[col] - g[col].shift(12)
    return g

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    log.info(f"Loading historical data from {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    # normalize column names
    df.columns = [c if isinstance(c,str) else str(c) for c in df.columns]
    # ensure date col is datetime
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], unit='ms', origin='unix', errors='coerce') if df[DATE_COL].dtype.kind in ("i","u") else pd.to_datetime(df[DATE_COL], errors='coerce')
    # sort
    df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

    # -------- explicit time index (for trend learning) --------
    df["t_idx"] = (
        (df[DATE_COL].dt.year - df[DATE_COL].dt.year.min()) * 12
        + (df[DATE_COL].dt.month - 1)
    )

    df["t_idx_norm"] = df["t_idx"] / df["t_idx"].max()


    # Basic QA
    log.info(f"Unique wards: {df[ID_COL].nunique()}")
    log.info(f"Date range: {df[DATE_COL].min()} -> {df[DATE_COL].max()}")


    # missing check
    missing_report = df[TARGETS].isna().mean().to_dict()
    log.info("Missing ratios (per target):")
    for k,v in missing_report.items():
        log.info(f"  {k}: {v:.3f}")
        if v > MAX_MISSING_RATIO_COL:
            log.warning(f"  WARNING: {k} has >{MAX_MISSING_RATIO_COL*100:.0f}% missing values.")

    # fill simple missing: forward then backward per ward for each target
    df = df.groupby(ID_COL).apply(lambda g: g.sort_values(DATE_COL).ffill().bfill()).reset_index(drop=True)

    # after fill check
    missing_after = df[TARGETS].isna().mean().to_dict()
    log.info("Missing after forward/backfill:")
    for k,v in missing_after.items():
        log.info(f"  {k}: {v:.3f}")

    # add month sin/cos + monsoon flag
    df = month_sin_cos(df)

    # compute lags and rolling means per ward
    df = df.groupby(ID_COL, group_keys=False).apply(lambda g: compute_lags_rolls(g)).reset_index(drop=True)

    # If any remaining NaNs created by lags (start months), fill with reasonable defaults:
    for col in df.columns:
        if col.endswith(tuple([f"_lag{l}" for l in LAGS])) or "_rmean_" in col or col.endswith("_diff1") or col.endswith("_diff12"):
            # fill NaNs with column median per ward
            df[col] = df.groupby(ID_COL)[col].transform(lambda s: s.fillna(s.median()))
            # if still nan (all nan), fill overall median
            df[col] = df[col].fillna(df[col].median())

    # create numeric_cols ordering used by LSTM/TFT: tv_cols = time-varying reals (targets + engineered)
    tv_cols = []
    # include base targets (order preserved)
    tv_cols.extend(TARGETS)
    # lags & rolls
    for col in TARGETS:
        for lag in LAGS:
            tv_cols.append(f"{col}_lag{lag}")
        for r in ROLLS:
            tv_cols.append(f"{col}_rmean_{r}")
        tv_cols.append(f"{col}_diff1")
        tv_cols.append(f"{col}_diff12")
    tv_cols.append("t_idx_norm")

    # add month features
    tv_cols += ["month", "month_sin", "month_cos", "is_monsoon"]
    # ensure columns exist
    missing_tv = [c for c in tv_cols if c not in df.columns]
    if missing_tv:
        raise KeyError("Missing tv_cols after processing: " + ", ".join(missing_tv))

    log.info(f"Total time-varying cols: {len(tv_cols)}")

    # static cols - pick common statics from master if present
    static_candidates = ["population_2011_ward","updated_population_2023_ward","area_sqkm_ward","it_job_density_mean","ndvi_mean","income_index_mean","built_area_m2_sum"]
    static_cols = [c for c in static_candidates if c in df.columns]
    log.info(f"Static columns detected: {static_cols}")

    # produce scaler: fit on time-varying columns (tv_cols)
    scaler = StandardScaler()
    log.info("Fitting StandardScaler on tv_cols ...")
    # fit on training window only (2014-2023) to avoid data leakage; choose cutoff 2024-01-01
    train_mask = df[DATE_COL] <= pd.Timestamp("2024-12-01")
    scaler.fit(df.loc[train_mask, tv_cols].values)

    scaler_artifact = {
        "x_scaler": scaler,
        "numeric_cols": tv_cols,
        "target_cols": ["electricity_demand","water_demand","congestion_index","pm25"],
        "created_at": datetime.utcnow().isoformat(),
        "scaler_type": "StandardScaler",
        "fit_cutoff": "2024-12-01"
    }
    
    tv_cols = list(dict.fromkeys(tv_cols))


    SCALER_PATH = PREP_OUT / "scalers.joblib"
    joblib.dump(scaler_artifact, SCALER_PATH)

    log.info(f"Saved scaler artifact to {SCALER_PATH}")

   

    
    # ------------------------------
# Experimental time split (for reproducibility)
# ------------------------------
    split_meta = {
        "train_end": "2024-12-01",
        "validation_period": "2025-01 to 2025-12",
        "test_period": "future (model-specific forecasting)"
    }
    
    feature_inventory = {
        "base_targets": TARGETS,
        "lags": LAGS,
        "rolling_windows": ROLLS,
        "seasonal_features": ["month", "month_sin", "month_cos", "is_monsoon"],
        "derived_features": ["diff1", "diff12"]
    }

    # Save meta
    meta = {
        "created_at": datetime.utcnow().isoformat(),
        "preprocessing_version": "phase3_dl_v1",
        "tv_cols": tv_cols,
        "static_cols": static_cols,
        "target_cols": ["electricity_demand","water_demand","congestion_index","pm25"],
        "id_col": ID_COL,
        "date_col": DATE_COL,
        "time_splits": split_meta,
        "feature_inventory" : feature_inventory
    }

    meta["missing_data_policy"] = {
        "initial_check": f"Warn if missing ratio > {MAX_MISSING_RATIO_COL}",
        "imputation": "Forward-fill then backward-fill per ward",
        "lag_nan_handling": "Per-ward median, fallback to global median"
    }

    tv_cols = list(dict.fromkeys(tv_cols))


    META_PATH = PREP_OUT / "meta.json"

    with open(META_PATH, "w") as fh:
        json.dump(meta, fh, indent=2)

    log.info(f"Saved meta to {META_PATH}")


    # save per-ward parquet and combined
    df.to_parquet(OUT_PROCESSED, index=False)
    log.info(f"Saved processed combined to {OUT_PROCESSED}")

    

    # quick diagnostic plots for a sample ward (first ward)
    sample_ward = df[ID_COL].unique()[0]
    g = df[df[ID_COL]==sample_ward].sort_values(DATE_COL)
    for col in ["pm25","congestion_index","electricity_demand","water_demand"]:
        plt.figure(figsize=(10,3))
        sns.lineplot(x=DATE_COL, y=col, data=g)
        plt.title(f"{col} - {sample_ward}")
        plt.tight_layout()
        plt.savefig(PLOT_OUT / f"plot_{sample_ward}_{col}.png", dpi=150)
    log.info(f"Saved sample diagnostic plots to {PLOT_OUT}")

    # print small summary for user
    summary = {
        "n_wards": int(df[ID_COL].nunique()),
        "start_date": str(df[DATE_COL].min().date()),
        "end_date": str(df[DATE_COL].max().date()),
        "tv_cols_count": len(tv_cols),
        "static_cols": static_cols
    }
    with open(SUMMARY / "prep_summary.json","w") as fh:
        json.dump(summary, fh, indent=2)
    log.info(f"Prep summary written to {SUMMARY / 'prep_summary.json'}")
    print("PREPROCESSING COMPLETE.")

if __name__ == "__main__":
    main()
