#!/usr/bin/env python3
"""
STEP 6.2 — Compute Sector-Level Stats from Future Forecasts

Inputs:
 - wards_with_sectors.geojson (from Step 6.1)
 - combined_forecast_2026_2035.csv (monthly forecasts per ward)

Output:
 - sector_future_profile.csv
 - sector_future_profile.json

Author: Urban Redesign (Phase 6)
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import json
import logging
import sys
import hashlib
import platform
from datetime import datetime

BASE_DIR = Path("redesign")

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "sector_future_stats.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def compute_sector_stats(wards_sectors_path, forecasts_path, outdir=ARTIFACTS_DIR/"sectors_forecast_2035"):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    # ----------------------------------------------------
    # 1. LOAD SECTORS + FORECASTS
    # ----------------------------------------------------
    logger.info("Loading wards with sector assignments")
    wards = gpd.read_file(wards_sectors_path)
    wards["ward_id"] = wards["ward_id"].astype(str)

    logger.info("Loading forecast data")
    df = pd.read_csv(forecasts_path)
    df["ward_id"] = df["ward_id"].astype(str)

    # parse date
    df["date"] = pd.to_datetime(df["date"])

    # ----------------------------------------------------
    # 2. FILTER ONLY YEAR 2035
    # ----------------------------------------------------
    logger.info("Filtering forecast data for year 2035")
    df_2035 = df[df["date"].dt.year == 2035].copy()

    if df_2035.empty:
        raise ValueError("❌ No rows found for year 2035 in the forecast file!")

    # ----------------------------------------------------
    # 3. AGGREGATE 2035 METRICS PER WARD
    # ----------------------------------------------------
    logger.info("Aggregating ward-level 2035 metrics")

    ward_2035 = df_2035.groupby("ward_id").agg(
        population_2035=("population", "last"),          # last month of 2035
        electricity_2035=("electricity_demand", "mean"), # average monthly consumption
        water_2035=("water_demand", "mean"),             # average
        traffic_2035=("congestion_index", "mean"),       # average
        pm25_2035=("pm25", "mean")                       # average pollution
    ).reset_index()


    # ----------------------------------------------------
# DERIVE FAILURE PROBABILITY FROM SEVERITY METRICS
# ----------------------------------------------------
    severity_cols = [
        "sev_population",
        "sev_rainfall",
        "sev_electricity_demand",
        "sev_water_demand",
        "sev_congestion_index",
        "sev_pm25"
    ]

    df_2035["fail_prob"] = df_2035[severity_cols].mean(axis=1)

    ward_2035["fail_prob"] = (
        df_2035.groupby("ward_id")["fail_prob"].mean().values
    )
 

    # ----------------------------------------------------
    # 4. MERGE WITH SECTOR ASSIGNMENTS
    # ----------------------------------------------------
    logger.info("Merging ward forecasts with sector assignments")
    merged = ward_2035.merge(
        wards[["ward_id", "sector"]],
        on="ward_id",
        how="left"
    )

    # sanity check
    if merged["sector"].isna().any():
        logger.warning("⚠️ Warning: Some wards have no assigned sector.")

    # ----------------------------------------------------
    # 5. COMPUTE SECTOR-LEVEL STATS
    # ----------------------------------------------------
    logger.info("Aggregating sector-level metrics")

    sector_profile = merged.groupby("sector").agg({
        "traffic_2035": "sum",
        "water_2035": "sum",
        "electricity_2035": "sum",
        "pm25_2035": "mean",
        "population_2035": "sum",
        "fail_prob": "mean"
    }).reset_index()

    # Rename for clarity
    sector_profile = sector_profile.rename(columns={
        "traffic_2035": "traffic_demand_2035",
        "water_2035": "water_demand_2035",
        "electricity_2035": "electricity_load_2035",
        "pm25_2035": "avg_pm25_2035",
        "fail_prob": "avg_failure_probability"
    })

    # ----------------------------------------------------
    # 6. SAVE OUTPUTS
    # ----------------------------------------------------
    csv_path = outdir / "sector_future_profile.csv"
    sector_profile.to_csv(csv_path, index=False)
    logger.info(f"Saved sector future profile CSV: {csv_path}")



    json_path = outdir / "sector_future_profile.json"
    with open(json_path, "w") as f:
        json.dump(sector_profile.to_dict(orient="records"), f, indent=4)
    logger.info(f"Saved sector future profile JSON: {json_path}")

    artifact_metadata = {
        "artifacts": [
            "sector_future_profile.csv",
            "sector_future_profile.json"
        ],
        "source_forecast_file": str(forecasts_path),
        "source_sector_file": str(wards_sectors_path),
        "aggregation_year": 2035,
        "metrics": [
            "traffic_demand",
            "water_demand",
            "electricity_load",
            "population",
            "pm25",
            "failure_probability"
        ],
        "method": "Ward-level aggregation followed by sector-level summation/averaging",
        "description": "Sector-level future demand and risk profile derived from AI forecasts"
    }

    with open(ARTIFACTS_DIR / "metadata/sector_future_profile_metadata.json", "w") as f:
        json.dump(artifact_metadata, f, indent=2)

    METADATA_DIR = BASE_DIR / "metadata"
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    def sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    run_manifest = {
        "script": "compute_sector_future_stats.py",
        "timestamp_utc": datetime.utcnow().isoformat(),
        "input_files": {
            "wards_sectors": str(wards_sectors_path),
            "forecasts": str(forecasts_path)
        },
        "output_files": [
            str(csv_path),
            str(json_path)
        ],
        "python_version": sys.version,
        "platform": platform.platform(),
        "phase": "Phase 6"
    }

    with open(METADATA_DIR / "run_manifest_sector_future_stats.json", "w") as f:
        json.dump(run_manifest, f, indent=2)


    logger.info("Sector-level stats computed successfully")
    return sector_profile

    


# ----------------------------------------------------
# SCRIPT ENTRY POINT
# ----------------------------------------------------
if __name__ == "__main__":
    compute_sector_stats(
        wards_sectors_path="redesign/data/processed/wards_with_sector_fixed.geojson",
        forecasts_path="cascade_model/results/failure_detection/detection_results.csv"
    )
