#!/usr/bin/env python3
"""
generate_phase3a_timeseries.py

Phase 3A — construct ward-level monthly time-series (2014-01 .. 2025-12).
If real timeseries are missing, this script synthesizes realistic series (seasonality + trend + noise + shocks).

Outputs:
 - per-ward parquet: data/time_series/ward_<ward_id>.parquet
 - combined parquet: data/time_series/all_wards_monthly.parquet
 - summary CSV: data/time_series/summary_stats.csv

Usage:
  python src/models/forecasting/generate_phase3a_timeseries.py \
      --wards data/processed/master/phase2/wards_phase2_enriched.geojson \
      --outdir data/time_series \
      --seed 42
"""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import logging

# =========================
# PHASE 3A CONFIGURATION
# =========================

SEED = 42
START_YEAR = 2014
END_YEAR = 2025

BASE_DIR = Path("AI_forecasting")

WARDS_PATH = Path(
    "AI_forecasting/data/input/wards_enriched.geojson"
)

CANONICAL_DIR = Path("AI_forecasting/data/input/timeseries")
WARD_DIR = Path("AI_forecasting/data/processed/timeseries/by_ward")



LOG_FILE = Path(
    "AI_forecasting/logs/data_generation.log"
)
SUMMARY = BASE_DIR / "summary"

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def seasonal_component(month, amplitude, phase=0):
    # month: 1..12 -> angle
    ang = 2 * np.pi * (month - 1) / 12.0 + phase
    return amplitude * np.sin(ang)

def generate_ward_series(seed, ward_meta, start_year=2014, end_year=2025):
    """
    ward_meta: dict with keys: ward_id (str), population_est (float), it_job_density_mean, built_area_m2_sum, income_index_mean
    returns dataframe with columns: year, month, electricity_demand, water_demand, congestion_index,
                                   pm25, rainfall, job_density, population, temperature, blackout_events
    """
    rng = np.random.RandomState(seed + abs(hash(ward_meta['ward_id'])) % (2**31))
    years = list(range(start_year, end_year + 1))
    months = list(range(1,13))
    recs = []
    # base scalars from ward metadata (fallback sensible defaults)
    pop0 = float(ward_meta.get('population_est', 40000.0))
    it_density = float(ward_meta.get('it_job_density_mean', 60.0))
    built_area = float(ward_meta.get('built_area_m2_sum', 50000.0))
    income = float(ward_meta.get('income_index_mean', 0.05))

    # set baseline levels shaped by metadata
    # electricity baseline MWh per month roughly proportional to population and income
    elec_base = 0.0008 * pop0 * (1 + (income - 0.05) * 3)  # ~ pop * factor
    water_base = 0.2 * pop0  # liters per month per person (scaled)
    congestion_base = 0.45 + (it_density - 60) * 0.001   # center near 0.536 typical
    pm25_base = 50 + (built_area / 100000.0) * 5  # baseline PM2.5
    temp_base = 24 + (income - 0.05) * 2  # slight variation
    job_density_base = it_density

    # trend: modest yearly growth
    annual_pop_growth = 0.01 + (rng.randn() * 0.002)  # ~1% +/- noise
    annual_demand_growth = 0.02 + (rng.randn() * 0.005)  # electricity/water trend

    # seasonal amplitudes
    elec_amp = max(0.05, 0.02 + abs(rng.randn() * 0.02))
    water_amp = max(0.05, 0.03 + abs(rng.randn() * 0.02))
    cong_amp = 0.03 + abs(rng.randn() * 0.02)
    pm25_amp = 15 + abs(rng.randn() * 5)
    rain_amp = 100 + abs(rng.randn() * 50)
    temp_amp = 3 + abs(rng.randn() * 2)

    # shock probability: occasional extreme events
    shock_years = rng.choice(years, size=max(1,int(len(years)*0.15)), replace=False)

    for y in years:
        # year-scale multipliers from trend
        year_idx = y - start_year
        pop = pop0 * ((1 + annual_pop_growth) ** year_idx)
        demand_trend_mul = (1 + annual_demand_growth) ** year_idx
        # monthly generation
        for m in months:
            # seasonality
            elec_seas = 1.0 + seasonal_component(m, elec_amp)
            water_seas = 1.0 + seasonal_component(m, water_amp, phase=0.5)
            cong_seas = 1.0 + seasonal_component(m, cong_amp, phase=1.0)
            pm25_seas = pm25_base + seasonal_component(m, pm25_amp, phase=-0.5)
            rain = max(0.0, seasonal_component(m, rain_amp, phase=0.2) + 50 + rng.randn()*20)  # mm/month

            # temperature seasonal + trend
            temp = temp_base + seasonal_component(m, temp_amp) + rng.randn()*0.8

            # electricity demand (MWh)
            elec = elec_base * elec_seas * demand_trend_mul
            # scale with temperature (AC usage spike if temp high)
            if temp > 28:
                elec *= (1 + 0.02 * (temp - 28))
            # add noise
            elec = elec * (1 + rng.randn() * 0.05)

            # water demand (cubic meters)
            water = (water_base * water_seas * demand_trend_mul) / 1000.0  # convert to kilo-cubic for scale
            water = water * (1 + rng.randn() * 0.06)

            # congestion index [0..1]
            cong = np.clip(congestion_base * cong_seas * (1 + (pop/pop0 - 1)*0.01) * (1 + rng.randn()*0.03), 0.0, 1.5)

            # pm25 (µg/m3)
            pm25 = max(5.0, pm25_seas + (pop / 50000.0)*5 + rng.randn()*6)

            # job_density (jobs per sq km proxy) evolves slowly; occasional shifts if ward is IT heavy
            job_density = job_density_base * (1 + 0.005*year_idx) + rng.randn()*1.0

            # blackout events: mostly low frequency integer (0/1/2 per month)
            # in shock years, increase probability
            base_blackout_prob = 0.01 + (built_area / 200000.0) * 0.005

            if not np.isfinite(base_blackout_prob):
                base_blackout_prob = 0.02
            if y in shock_years and rng.rand() < 0.3:
                # cluster of outages that year
                blackout = rng.poisson(lam=0.5)
            else:
                p = base_blackout_prob + rng.randn() * 0.01
                p = float(np.clip(p, 0.0, 1.0))
                blackout = rng.binomial(1, p)
            recs.append({
                'year': int(y),
                'month': int(m),
                'electricity_demand': float(np.round(elec, 6)),
                'water_demand': float(np.round(water, 6)),
                'congestion_index': float(np.round(cong, 6)),
                'pm25': float(np.round(pm25, 4)),
                'rainfall': float(np.round(rain, 3)),
                'job_density': float(np.round(job_density, 4)),
                'population': float(np.round(pop, 2)),
                'temperature': float(np.round(temp, 3)),
                'blackout_events': int(blackout)
            })
    df = pd.DataFrame.from_records(recs)
    # sanity corrections: fill any negatives
    numeric_cols = ['electricity_demand','water_demand','congestion_index','pm25','rainfall','job_density','population','temperature']
    for c in numeric_cols:
        df[c] = df[c].clip(lower=0.0)
    return df

def main():

    wards_path = WARDS_PATH
    ensure_dir(CANONICAL_DIR)
    ensure_dir(WARD_DIR)



    # load wards file (geojson/parquet/csv accepted)
    if wards_path.suffix.lower() in ['.geojson', '.json', '.gpkg', '.shp']:
        try:
            gdf = gpd.read_file(wards_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read wards file {wards_path}: {e}")
    elif wards_path.suffix.lower() in ['.parquet']:
        df_tmp = pd.read_parquet(wards_path)
        gdf = gpd.GeoDataFrame(df_tmp)
    else:
        gdf = gpd.read_file(wards_path)

    # identify ward id column
    if 'ward_id' in gdf.columns:
        ward_id_col = 'ward_id'
    elif 'ward_num_x' in gdf.columns:
        ward_id_col = 'ward_num_x'
    else:
        # fallback to index
        gdf = gdf.reset_index().rename(columns={'index':'ward_id'})
        ward_id_col = 'ward_id'

    # ensure numeric metadata columns exist
    meta_cols = ['population_est','it_job_density_mean','built_area_m2_sum','income_index_mean']
    for c in meta_cols:
        if c not in gdf.columns:
            gdf[c] = np.nan

    combined = []
    # generate per-ward series
    logger.info(f"Generating time-series for {len(gdf)} wards")

    for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):
        ward_meta = {
            'ward_id': str(row[ward_id_col]),
            'population_est': float(row.get('population_est', np.nan) if not pd.isna(row.get('population_est', np.nan)) else 40000.0),
            'it_job_density_mean': float(row.get('it_job_density_mean', 60.0) if not pd.isna(row.get('it_job_density_mean', 60.0)) else 60.0),
            'built_area_m2_sum': float(row.get('built_area_m2_sum', 50000.0) if not pd.isna(row.get('built_area_m2_sum', 50000.0)) else 50000.0),
            'income_index_mean': float(row.get('income_index_mean', 0.05) if not pd.isna(row.get('income_index_mean', 0.05)) else 0.05)
        }
        df_ts = generate_ward_series(SEED, ward_meta, start_year=START_YEAR, end_year=END_YEAR)
        df_ts['ward_id'] = ward_meta['ward_id']
        # save per-ward parquet
        p = WARD_DIR / f"ward_{ward_meta['ward_id']}.parquet"
        df_ts.to_parquet(p, index=False)

        
        combined.append(df_ts.assign(ward_id=ward_meta['ward_id']))

    # combined
    all_df = pd.concat(combined, axis=0, ignore_index=True)
    all_p = CANONICAL_DIR / "all_wards_monthly.parquet"
    all_df.to_parquet(all_p, index=False)

    # summary stats
    summary = all_df.groupby('ward_id').agg({
        'population': ['mean','first'],
        'electricity_demand': ['mean','max'],
        'water_demand': ['mean','max'],
        'congestion_index': ['mean','max'],
        'pm25': ['mean','max']
    })
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    summary.to_csv(WARD_DIR / "summary_stats.csv", index=False)

    logger.info(f"Per-ward time series written to {WARD_DIR}")
    logger.info(f"Combined file written: {all_p}")
    logger.info(f"Summary stats written: {WARD_DIR / 'summary_stats.csv'}")

    meta = {
        "stage": "Time-series data construction for forecasting",
        "seed": SEED,
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "num_wards": len(gdf),
        "input_wards_file": str(WARDS_PATH),
        "canonical_timeseries_dir": str(CANONICAL_DIR),
        "ward_timeseries_dir": str(WARD_DIR)
    }

    ensure_dir(SUMMARY)
    meta_path = SUMMARY / "timeseries_metadata.json"
    pd.Series(meta).to_json(meta_path, indent=2)

    logger.info("Data preprocessing metadata written")



if __name__ == "__main__":
    main()
