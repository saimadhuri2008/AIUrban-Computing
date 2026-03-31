#!/usr/bin/env python3
"""
detect_failures_severity.py

Severity-based failure detection for cascade model.

Produces:
 - JSON per-run summary
 - CSV of per-row severity scores
 - joblib of numpy severity matrix
"""

import json
import os
from pathlib import Path
from datetime import datetime
import argparse
import math
import logging


import numpy as np
import pandas as pd
import joblib

BASE_DIR = Path("cascade_model")



ARTIFACTS_SEV_DIR = BASE_DIR / "artifacts"
RESULTS_DIR = BASE_DIR / "results" / "failure_detection" 
DIAG_DIR = BASE_DIR / "diagnostics" / "severity" 
LOGS_DIR = BASE_DIR / "logs"

for d in [ARTIFACTS_SEV_DIR, RESULTS_DIR, DIAG_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SUMMARY = BASE_DIR / "summary"
SUMMARY.mkdir(parents=True, exist_ok=True)
# ---------------------------
# Config / defaults
# ---------------------------
DEFAULT_THRESH_PATH = BASE_DIR / "artifacts/thresholds/thresholds.json"

log_file = LOGS_DIR / "detect_failures_severity.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

logger = logging.getLogger("failure_severity")
logger.info("Starting severity-based failure detection")



# ---------------------------
# Utilities
# ---------------------------
def load_thresholds(path=DEFAULT_THRESH_PATH):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Thresholds file not found: {p}")
    with open(p, "r") as fh:
        return json.load(fh)

def safe_div(a, b, eps=1e-9):
    try:
        return a / (b + eps)
    except Exception:
        return 0.0

# ---------------------------
# Severity calculation rules
# ---------------------------
def severity_from_absolute(value, threshold, median=None, p90=None):
    """
    For absolute-mode thresholds.
    - If value <= threshold -> severity 0
    - If value moderately above threshold -> value -> linear scale to 1.0 at (threshold + scale_span)
    - If very large above threshold -> cap at 2.0 (severe)
    scale_span chosen as max( (p90 - median), 0.1*threshold, or fixed small value ), whichever sensible.
    """
    value = float(value) if value is not None else float("nan")
    threshold = float(threshold)
    if math.isnan(value):
        return 0.0

    if value <= threshold:
        return 0.0

    # scale span heuristics
    span = None
    if p90 is not None and median is not None and p90 > median:
        span = float(max(1e-6, p90 - median))
    else:
        span = max(0.1 * threshold, 1.0)

    # Map:
    # threshold -> 0.0
    # threshold + span -> 1.0 (failure)
    # threshold + 2*span or more -> 2.0 (severe)
    rel = (value - threshold) / span
    if rel <= 0:
        return 0.0
    elif rel < 1.0:
        # 0 -> 1 maps to 0.0 -> 1.0
        return float(rel)
    elif rel < 2.0:
        # 1 -> 2 maps to 1.0 -> 2.0
        return 1.0 + float(rel - 1.0)
    else:
        return 2.0

def severity_from_relative_pct(value, baseline, alert_increase_pct, alert_decrease_pct=None):
    """
    For relative_pct mode thresholds.
    - baseline: typical historical baseline (can be median)
    - alert_increase_pct: e.g. 0.2 means 20% increase triggers 'failure' boundary
    - We compute pct = (value - baseline)/baseline
    - severity 0 if pct <= 0 (or low positive)
    - scale to 1.0 at alert_increase_pct and to 2.0 at 2*alert_increase_pct
    - If alert_decrease_pct provided and value < baseline by that proportion, produce negative severity (as stress due to drop; clipped >=0 unless you want negative)
    """
    if baseline is None or baseline == 0 or value is None or math.isnan(baseline):
        return 0.0
    pct = (float(value) - float(baseline)) / float(baseline)

    # If drop-detection configured and negative shocks matter
    if pct < 0 and alert_decrease_pct:
        # consider negative-direction severity (e.g., supply drop)
        threshold_neg = float(alert_decrease_pct)
        if abs(pct) <= threshold_neg:
            return 0.0
        # map abs(pct) to severity similarly
        rel = abs(pct) / max(1e-9, threshold_neg)
        if rel < 1.0:
            return float(rel)
        elif rel < 2.0:
            return 1.0 + float(rel - 1.0)
        else:
            return 2.0

    # Positive direction (growth pressure)
    if pct <= 0:
        return 0.0
    if not alert_increase_pct or alert_increase_pct <= 0:
        # fallback safe mapping: treat 20% as failure by default
        alert_increase_pct = 0.20

    rel = pct / float(alert_increase_pct)
    if rel < 1.0:
        return float(rel)
    elif rel < 2.0:
        return 1.0 + float(rel - 1.0)
    else:
        return 2.0

# ---------------------------
# Core API
# ---------------------------
def compute_row_severity(row: dict, thresholds: dict, baseline_map: dict = None):
    """
    row: dict-like containing node values (e.g. population, rainfall, pm25, ...)
    thresholds: loaded from thresholds.json
    baseline_map: optional dict of baselines for relative thresholds, key=node -> baseline_value (e.g., median)
    Returns: dict node -> severity_score (float, 0..2)
    """
    res = {}
    node_specs = thresholds.get("node_thresholds", thresholds)
    for node, spec in node_specs.items():
        val = row.get(node, None)
        if spec.get("mode") == "absolute":
            thr = spec.get("threshold")
            # supply median & p90 if available in notes
            notes = spec.get("notes", {})
            median = notes.get("median") or notes.get("p50") or None
            p90 = notes.get("p90") or None
            s = severity_from_absolute(val, thr, median=median, p90=p90)
            res[node] = float(s)
        elif spec.get("mode") == "relative_pct":
            # choose baseline: user-supplied baseline_map > notes.rel? > median historical absolute?
            baseline = None
            if baseline_map and node in baseline_map:
                baseline = baseline_map[node]
            else:
                # try notes
                notes = spec.get("notes", {})
                # notes may contain rel90/rel95 etc - but we need absolute baseline; leave None
                baseline = spec.get("baseline", None)
            alert_inc = spec.get("alert_pct_increase") or spec.get("alert_pct", spec.get("alert_pct_increase", None))
            alert_dec = spec.get("alert_pct_decrease", None)
            s = severity_from_relative_pct(val, baseline, alert_inc, alert_decrease_pct=alert_dec)
            res[node] = float(s)
        else:
            # unknown mode -> default 0
            res[node] = 0.0
    return res

def compute_severity_dataframe(df, thresholds, baseline_map=None, node_list=None):
    """
    df: pandas.DataFrame rows contain node columns (e.g., 'population','pm25',...)
    thresholds: loaded thresholds dict
    baseline_map: optional baseline values
    node_list: if provided, compute only these nodes (order preserved)
    Returns:
        severity_df: DataFrame with same index as df and columns node_list
    """
    if node_list is None:
        node_list = list(thresholds.get("node_thresholds", thresholds).keys())
    rows = []
    for idx, row in df.iterrows():
        rowd = row.to_dict()
        sev = compute_row_severity(rowd, thresholds, baseline_map=baseline_map)
        rows.append({n: sev.get(n, 0.0) for n in node_list})
    severity_df = pd.DataFrame(rows, index=df.index, columns=node_list)
    return severity_df

# ---------------------------
# CLI and IO
# ---------------------------
def run_detection(states_path, thresholds_path, outdir, baseline_map_path=None):
    thresholds = load_thresholds(thresholds_path)
    outdir = RESULTS_DIR


    # Load states: accept CSV or parquet. States must contain the node columns and optionally 'ward_id','date'
    p = Path(states_path)
    if not p.exists():
        raise FileNotFoundError(f"States file not found: {p}")
    if p.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, parse_dates=True)

    # Keep index for later merging
    df = df.reset_index(drop=True)

    # load baseline_map if provided (JSON mapping node->value)
    baseline_map = None
    if baseline_map_path:
        with open(baseline_map_path, "r") as fh:
            baseline_map = json.load(fh)

    node_list = list(thresholds.get("node_thresholds", thresholds).keys())
    missing = [n for n in node_list if n not in df.columns]
    if missing:
        logger.warning(f"[WARN] State rows missing these nodes; they will be filled with NaN: {missing}")
        for m in missing:
            df[m] = np.nan

    # Compute severity df
    severity_df = compute_severity_dataframe(df, thresholds, baseline_map=baseline_map, node_list=node_list)

    # add semantic label columns (optional)
    def label_from_score(x):
        if x == 0.0:
            return "healthy"
        elif 0.0 < x < 0.6:
            return "warning"
        elif 0.6 <= x < 1.5:
            return "failure"
        else:
            return "severe"

    labels_df = severity_df.map(label_from_score)

    # Save results
    tstamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    csv_out = RESULTS_DIR / "detection_results.csv"
    json_out = SUMMARY / "detection_summary.json"
    joblib_out = ARTIFACTS_SEV_DIR / "severity_matrix.joblib"

    # Merge severity into df for full context
    merged = pd.concat([df, severity_df.add_prefix("sev_")], axis=1)
    merged.to_csv(csv_out, index=False)

    # Save numpy severity matrix
    joblib.dump(severity_df.values, joblib_out)

    # Write a compact summary JSON
    summary = {
        "n_rows": int(len(df)),
        "nodes": node_list,
        "csv": str(csv_out),
        "joblib": str(joblib_out),
        "timestamp": tstamp,
        "severity_stats": {}
    }
    for n in node_list:
        arr = severity_df[n].dropna().values
        summary["severity_stats"][n] = {
            "mean": float(np.nanmean(arr)) if len(arr)>0 else None,
            "max": float(np.nanmax(arr)) if len(arr)>0 else None,
            "pct_above_0": float(np.mean(arr>0)) if len(arr)>0 else None,
            "pct_above_1": float(np.mean(arr>=1.0)) if len(arr)>0 else None,
            "pct_above_2": float(np.mean(arr>=2.0)) if len(arr)>0 else None,
        }

    with open(json_out, "w") as fh:
        json.dump(summary, fh, indent=2)

    # Also save readable diagnostics (per-node hist)
    diag_path = DIAG_DIR / "severity_diag.json"
    diag = {"node_histograms": {}}
    for n in node_list:
        arr = severity_df[n].fillna(0).values
        # basic histogram bins 0..2 with step 0.2
        bins = np.histogram(arr, bins=np.linspace(0,2,11))[0].tolist()
        diag["node_histograms"][n] = {"bins_counts": bins}
    with open(diag_path, "w") as fh:
        json.dump(diag, fh, indent=2)

    run_meta = {
        "analysis": "severity_based_failure_detection",
        "thresholds_used": str(thresholds_path),
        "states_used": str(states_path),
        "nodes": node_list,
        "n_rows": int(len(df)),
        "severity_scale": "0=healthy, 1=failure, 2=severe",
        "timestamp": tstamp
    }

    with open(SUMMARY / "failure_detection_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    logger.info("Saved run metadata")

    logger.info(f"[SAVED] CSV at {csv_out}")
    logger.info(f"[SAVED] Summary JSON at {json_out}")
    logger.info(f"[SAVED] Severity joblib at {joblib_out}")
    logger.info(f"[SAVED] Diagnostics at {diag_path}")

    return summary, merged, severity_df

    


# ---------------------------
# If run as script
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", type=str, default=str(DEFAULT_THRESH_PATH), help="Path to thresholds.json")
    parser.add_argument("--outdir",type=str,default = BASE_DIR/"results/failure_detection")
    parser.add_argument("--states", type=str, default=BASE_DIR/"data/combined_forecast_2026_2035.csv", help="Path to CSV/parquet containing state rows (ward/time) with node columns")
    parser.add_argument("--baseline", type=str, default=None, help="Optional baseline JSON mapping node->value")
    args = parser.parse_args()

    run_detection(args.states, args.thresholds,args.outdir, baseline_map_path=args.baseline)
