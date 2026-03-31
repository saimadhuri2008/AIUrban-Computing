#!/usr/bin/env python3
"""
run_phase3b.py

Orchestrate Phase 3B:
 - Ensure Phase3A data exists (all_wards_monthly.parquet)
 - Run SARIMAX forecasting for main targets
 - Run ETS/STL for rainfall & temperature
 - Save consolidated outputs and a short JSON summary with alarm years (quick heuristic)

Usage:
  python src/models/forecasting/run_phase3b.py --input data/time_series/all_wards_monthly.parquet
"""
import json
import logging
from datetime import datetime
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np

LOG_DIR = Path("AI_forecasting/logs/forecasting/orchestration")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "run_classical_models.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

BASE_DIR = Path("AI_forecasting")

MODELS_DIR = BASE_DIR / "src/models/classical"

SARIMAX_SCRIPT = MODELS_DIR / "sarimax_forecast.py"
ETS_SCRIPT = MODELS_DIR / "ets_stl_forecast.py"

RESULTS_DIR = BASE_DIR / "results"
CLASSICAL_RESULTS = RESULTS_DIR / "classical"
AGG_DIR = CLASSICAL_RESULTS / "aggregated"

ORCH_RESULTS = RESULTS_DIR / "orchestration"
ORCH_RESULTS.mkdir(parents=True, exist_ok=True)

def run_script(script_path, args_list):
    cmd = ["python", str(script_path)] + args_list
    logger.info(f"RUNNING: {' '.join(cmd)}")

    subprocess.check_call(cmd)

def quick_sustainability_alarms(forecast_agg_dir, thresholds):
    """
    Simple heuristic: for each ward and target, read forecast CSV and find first year threshold crossed.
    thresholds: dict, e.g. {"electricity_demand": capacity_value_by_ward_or_scalar}
    returns alarms dict
    """
    alarms = {}
    for target, thresh in thresholds.items():
        path = forecast_agg_dir / f"{target}_forecasts_all_wards.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=['date'])
        df['year'] = pd.to_datetime(df['date']).dt.year
        # group by ward-year and check annual sum/mean depending on variable; use monthly forecasts so aggregate
        if target in ["electricity_demand", "water_demand"]:
            grouped = (
                df.groupby(['ward_id','year'])['forecast']
                .sum()
                .reset_index()
            )
        else:
            grouped = (
                df.groupby(['ward_id','year'])['forecast']
                .mean()
                .reset_index()
            )
        if thresh < 1:
            thresh_value = grouped["forecast"].quantile(thresh)
        else:
            thresh_value = thresh


        first_cross = {}
        for ward, g in grouped.groupby('ward_id'):
            crossing = g[g["forecast"] > thresh_value]

            if not crossing.empty:
                first_cross[ward] = int(crossing['year'].iloc[0])
        alarms[target] = first_cross
    return alarms

def main():
    

   

    # 1) SARIMAX for core targets
    logger.info("TIER 2: Running SARIMAX (Top wards handled internally)")
    run_script(SARIMAX_SCRIPT, [])


    # 2) ETS / STL for rainfall & temperature
    logger.info("TIER 1: Running ETS/STL (city-level)")
    run_script(ETS_SCRIPT, [])


    # 3) quick alarm heuristics: threshold examples (scalars) — tweak per-ward later
    forecast_agg_dir = AGG_DIR

    thresholds = {
        "electricity_demand": 0.95, #ample: 1,000,000 MWh per year as threshold (adjust per real capacity)
        "water_demand": 0.95,     # example liters or cubic measure. Tweak accordingly
        "congestion_index": 0.75,   # congestion threshold
        "pm25": 60                   # annual avg threshold
    }
    alarms = quick_sustainability_alarms(forecast_agg_dir, thresholds)
    with open(ORCH_RESULTS / "quick_alarms.json","w") as f:
        json.dump(alarms, f, indent=2)

    run_meta = {
        "stage": "classical_forecasting",
        "methods": {
            "tier_1": "ETS/STL (city-level)",
            "tier_2": "SARIMAX (ward-level)",
            "tier_3": "heuristic sustainability alarms"
        },
        "thresholds": thresholds,
        "generated_at_utc": datetime.utcnow().isoformat()
    }

    with open(ORCH_RESULTS / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    logger.info("orchestration complete")
    print("Phase 3B complete.")
    print("Results:", ORCH_RESULTS)


if __name__ == "__main__":
    main()
