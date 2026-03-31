#!/usr/bin/env python3
"""
name_and_union_sectors.py

Heuristically name 5-sector GeoJSON features (East, West, North, South, Central),
union into single-sector polygons, and optionally spatial-join wards -> sectors.

Usage:
    python name_and_union_sectors.py --sectors bbmp_5sectors.geojson --wards wards_master_enriched.geojson --outdir src/cascade_model

Outputs:
 - bbmp_5sectors_named.geojson      (same features with sector names)
 - bbmp_5sectors_union.geojson      (one polygon per named sector)
 - wards_with_sectors.geojson       (if --wards provided)
 - sectors_summary.csv              (ward counts per sector)
"""
import argparse
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Point
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
import sys
import json
import hashlib
import platform
from datetime import datetime


# Bengaluru center (used for heuristic)
CITY_CENTER = (77.5946, 12.9716)  # (lon, lat)

BASE_DIR = Path("redesign")
INTERIM = BASE_DIR / "data/interim"
OUTDIR = BASE_DIR / "data/processed"
ARTIFACTS = BASE_DIR / "artifacts"

for d in [OUTDIR,ARTIFACTS]:
    d.mkdir(parents=True, exist_ok=True)

    LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "name_and_union_sectors.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def bearing_deg(center, pt):
    # center: (lon, lat), pt: shapely Point
    dx = pt.x - center[0]
    dy = pt.y - center[1]
    # angle in degrees: 0 = East, 90 = North, 180/-180 = West, -90 = South
    ang = math.degrees(math.atan2(dy, dx))
    return ang

def choose_sector_name(angle_deg):
    # map angle to cardinal sector
    # angle close to 0 -> East, 90 -> North, 180/-180 -> West, -90 -> South
    # compute absolute difference to each target
    targets = {"East": 0.0, "North": 90.0, "West": 180.0, "South": -90.0}
    best = min(targets.keys(), key=lambda k: min(abs(angle_deg - targets[k]), abs(angle_deg - (targets[k] - 360)), abs(angle_deg - (targets[k] + 360))))
    return best

def name_and_union(sectors_gdf, city_center=CITY_CENTER):
    # Ensure correct CRS and geometry types
    sectors = sectors_gdf.copy()
    if sectors.crs is None:
        sectors = sectors.set_crs(epsg=4326)
    sectors = sectors.to_crs(epsg=4326)

    # compute centroid for each feature
    centroids = sectors.geometry.centroid
    sectors["centroid_lon"] = centroids.x
    sectors["centroid_lat"] = centroids.y
    sectors["centroid_pt"] = [Point(x, y) for x, y in zip(sectors["centroid_lon"], sectors["centroid_lat"])]

    # if the features are exactly 1 and probably contain all sectors in one polygon, warn and exit
    n_feats = len(sectors)
    if n_feats == 0:
        raise RuntimeError("No polygon features found in sectors GeoJSON")
    if n_feats == 1:
        print("WARNING: Only 1 sector polygon found. Is this file already unioned? The script will still attempt to name it 'Central'.")
        sectors["sector"] = "Central"
        named = sectors
        # build union mapping trivially
        unioned = sectors.dissolve(by="sector", as_index=False)
        return named, unioned

    # compute distance to city center (lon,lat)
    cx, cy = city_center
    sectors["dist_to_center"] = sectors["centroid_pt"].apply(lambda p: p.distance(Point(cx, cy)))

    # choose Central = feature with smallest distance to city center
    idx_central = sectors["dist_to_center"].idxmin()
    sectors.loc[idx_central, "sector"] = "Central"

    # For remaining features, compute bearing from city center to centroid and assign East/West/North/South
    remaining = sectors[sectors["sector"].isna()].copy() if "sector" in sectors.columns else sectors.copy()
    if "sector" not in sectors.columns:
        sectors["sector"] = None

    for idx, row in remaining.iterrows():
        ang = bearing_deg(city_center, row["centroid_pt"])
        name = choose_sector_name(ang)
        sectors.loc[idx, "sector"] = name

    # After assignment, check counts per sector
    counts = sectors["sector"].value_counts().to_dict()

    # If any cardinal sector is missing (e.g., due to odd geometry), try to reassign the farthest from center
    needed = set(["East", "West", "North", "South", "Central"])
    present = set(sectors["sector"].unique())
    missing = needed - present
    if missing:
        print("Note: missing sector names detected:", missing)
        # try to fill by picking top-n by angle distance that are unassigned duplicates
        # (This is a gentle fallback; user should verify)
        for m in missing:
            # choose the feature not central with angle closest to target
            target_angle = {"East":0,"North":90,"West":180,"South":-90}[m]
            # compute angles for all features excluding central
            sectors["angle"] = sectors["centroid_pt"].apply(lambda p: bearing_deg(city_center, p))
            sectors["angle_diff"] = sectors["angle"].apply(lambda a: min(abs(a - target_angle), abs(a - (target_angle-360)), abs(a - (target_angle+360))))
            # pick smallest angle diff among those that are not Central and not yet labeled with missing (avoid overriding existing)
            candidate_idx = sectors[sectors["sector"] != "Central"].sort_values("angle_diff").index[0]
            sectors.loc[candidate_idx, "sector"] = m

    # union geometries per sector name (to get single polygon per sector)
    unioned_list = []
    for sec in ["Central", "East", "North", "West", "South"]:
        parts = sectors[sectors["sector"] == sec]
        if len(parts) == 0:
            continue
        union_geom = unary_union(parts.geometry.values)
        unioned_list.append({"sector": sec, "geometry": union_geom})

    unioned_gdf = gpd.GeoDataFrame(unioned_list, crs="EPSG:4326").set_geometry("geometry")

    # reorder columns and return
    named = sectors.drop(columns=["centroid_pt"], errors="ignore")
    return named, unioned_gdf

def spatial_join_wards(wards_path, unioned_gdf, outpath):
    wards = gpd.read_file(wards_path)
    # standardize ward id
    if "ward_id" not in wards.columns:
        for c in ("ward","wardcode","id","ward_name","OBJECTID","name"):
            if c in wards.columns:
                wards = wards.rename(columns={c:"ward_id"})
                break
    wards["ward_id"] = wards["ward_id"].astype(str)
    wards = wards.to_crs(epsg=4326)

    # spatial join: which union polygon contains ward centroid or intersects ward polygon
    wards["centroid"] = wards.geometry.centroid
    joined = gpd.sjoin(wards, unioned_gdf[["sector","geometry"]], how="left", predicate="intersects")
    joined = joined.drop(columns=["index_right"], errors="ignore")
    # fallback: assign by centroid if still NaN
    missing = joined["sector"].isna().sum()
    if missing > 0:
        missing_idx = joined[joined["sector"].isna()].index
        for idx in missing_idx:
            pt = joined.loc[idx, "centroid"]
            hit = unioned_gdf[unioned_gdf.geometry.contains(pt)]
            if len(hit) > 0:
                joined.at[idx, "sector"] = hit.iloc[0]["sector"]
    # final fill
    joined["sector"] = joined["sector"].fillna("UNMAPPED")
    # save
    joined = joined.drop(columns=["centroid"], errors="ignore")
    joined.to_file(outpath, driver="GeoJSON")
    return joined

def main():


    p = argparse.ArgumentParser()
    p.add_argument("--sectors", default=BASE_DIR/"data/interim/bbmp_5sectors.geojson", help="input sectors geojson/kml with polygons")
    p.add_argument("--wards", default="data/processed/wards/wards_master_enriched.geojson", help="(optional) wards geojson to assign sectors to wards")
    p.add_argument("--outdir", default=None, help="output directory")
    args = p.parse_args()

    sectors_path = Path(args.sectors)

    logger.info("PHASE 6 — Sector semantic naming and union started")
    logger.info(f"Input sectors file: {sectors_path}")

    # Read sectors (supports single-layer geojson or KML via geopandas)
    try:
        sectors_gdf = gpd.read_file(str(sectors_path))
    except Exception as e:
        raise RuntimeError(f"Could not read sectors file: {e}")

    # Ensure polygons only
    sectors_gdf = sectors_gdf[sectors_gdf.geometry.type.isin(["Polygon","MultiPolygon"])].reset_index(drop=True)
    if len(sectors_gdf) == 0:
        raise RuntimeError("No polygon features found in sectors input file.")

    # Name and union
    named_gdf, unioned_gdf = name_and_union(sectors_gdf)

    # Save outputs
    out_named = OUTDIR / "bbmp_5sectors_named.geojson"
    named_gdf.to_file(out_named, driver="GeoJSON")
    logger.info(f"[SAVED] named sectors at {out_named}")

    out_union = ARTIFACTS / "bbmp_5sectors_union.geojson"
    unioned_gdf.to_file(out_union, driver="GeoJSON")
    logger.info(f"[SAVED] unioned sector outlines at {out_union}")

    

    # Print human readable summary
    logger.info("\n---- Sector features (named) ----")
    logger.info(f"{named_gdf[['sector']].assign(count=1).groupby('sector').count().to_string()}")
    logger.info("\n---- Unioned sectors geometry ----")
    logger.info(f"{unioned_gdf[['sector','geometry']].to_string(index=False)}")

    ARTIFACTS_METADATA_DIR = BASE_DIR / "artifacts/metadata"
    ARTIFACTS_METADATA_DIR.mkdir(parents=True, exist_ok=True)

    sector_metadata = {
        "artifacts": [
            "bbmp_5sectors_named.geojson",
            "bbmp_5sectors_union.geojson"
        ],
        "method": "Centroid distance + bearing-based heuristic",
        "city_center": CITY_CENTER,
        "crs": "EPSG:4326",
        "num_input_features": len(sectors_gdf),
        "num_output_sectors": len(unioned_gdf),
        "description": "Semantic naming and canonical union of 5-sector city design"
    }

    with open(ARTIFACTS_METADATA_DIR / "sector_naming_metadata.json", "w") as f:
        json.dump(sector_metadata, f, indent=2)

    METADATA_DIR = BASE_DIR / "metadata"
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    def sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    run_manifest = {
        "script": "name_and_union_sectors.py",
        "timestamp_utc": datetime.utcnow().isoformat(),
        "input_sectors": str(sectors_path),
        "input_sectors_sha256": sha256(sectors_path),
        "outputs": [
            str(out_named),
            str(out_union)
        ],
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    with open(METADATA_DIR / "run_manifest_naming_sectors.json", "w") as f:
        json.dump(run_manifest, f, indent=2)

    logger.info("Sector naming and union completed successfully")


if __name__ == "__main__":
    main()
