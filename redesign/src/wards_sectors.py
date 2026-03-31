#!/usr/bin/env python3
# fix_wards_assign_sectors_fixed.py
import geopandas as gpd
from pathlib import Path
import pandas as pd
import numpy as np
import math

# ---------- EDIT PATHS ----------
WARDS_PATH = "data/processed/wards/wards_master_enriched.geojson"             # your raw wards file (198)
SECTORS_PATH = "redesign/data/processed/bbmp_5sectors_named.geojson" # your 5-sector polygon file (5)
OUT_GEOJSON = "redesign/data/processed/wards_with_sector_fixed.geojson"
OUT_SUMMARY = "redesign/results/sectors/wards_assign_summary.csv"
# ---------------------------------

def ensure_crs(gdf, epsg=4326):
    if gdf.crs is None:
        raise RuntimeError("Input GeoDataFrame has no CRS. Set it first.")
    if gdf.crs.to_epsg() != epsg:
        return gdf.to_crs(epsg=epsg)
    return gdf

def main():
    print("[INFO] Loading wards and sectors...")
    wards = gpd.read_file(WARDS_PATH)
    sectors = gpd.read_file(SECTORS_PATH)

    print(f"[INFO] wards CRS: {wards.crs}, rows: {len(wards)}")
    print(f"[INFO] sectors CRS: {sectors.crs}, rows: {len(sectors)}")

    wards = ensure_crs(wards, epsg=4326)
    sectors = ensure_crs(sectors, epsg=4326)

    # Ensure id columns
    if "ward_id" not in wards.columns:
        wards = wards.reset_index().rename(columns={"index":"ward_id"})
        wards["ward_id"] = wards["ward_id"].apply(lambda x: f"ward_{x+1}")
    if "sector" not in sectors.columns:
        if "sector_id" in sectors.columns:
            sectors["sector"] = sectors["sector_id"].astype(str)
        else:
            sectors = sectors.reset_index().rename(columns={"index":"sector_idx"})
            sectors["sector"] = sectors["sector_idx"].apply(lambda x: f"sector_{x+1}")

    # Project to metric CRS for accurate spatial ops
    wards_m = wards.to_crs(epsg=3857)
    sectors_m = sectors.to_crs(epsg=3857)

    # Attempt 'within' join first (fast)
    print("[INFO] Performing spatial join (within)...")
    joined_within = gpd.sjoin(wards_m, sectors_m[["sector","geometry"]], how="left", predicate="within")
    # Prepare result frame
    result = joined_within[[c for c in wards_m.columns]].copy()
    result["sector_assigned"] = joined_within.get("sector")

    # Identify unmatched
    unmatched_idx = result[result["sector_assigned"].isna()].index.tolist()
    print(f"[INFO] Unmatched after 'within': {len(unmatched_idx)} wards")

    # For each unmatched ward, compute intersection area with each sector and pick max
    if unmatched_idx:
        print("[INFO] Resolving unmatched wards by intersection area with sectors...")
        # ensure valid geometries
        sectors_m["geom_centroid"] = sectors_m.geometry.centroid
        for idx in unmatched_idx:
            ward_geom = result.at[idx, "geometry"]
            best_sector = None
            best_area = 0.0
            for sidx, srow in sectors_m.iterrows():
                try:
                    inter = ward_geom.intersection(srow.geometry)
                    if not inter.is_empty:
                        area = inter.area
                    else:
                        area = 0.0
                except Exception:
                    area = 0.0
                if area > best_area:
                    best_area = area
                    best_sector = srow["sector"]
            if best_sector is not None and best_area > 0:
                result.at[idx, "sector_assigned"] = best_sector
                result.at[idx, "assign_method"] = "intersection_max"
                result.at[idx, "assign_area_m2"] = best_area

    # Recompute list of still-unmatched
    still_unmatched = result[result["sector_assigned"].isna()].index.tolist()
    print(f"[INFO] Still unmatched after intersection area step: {len(still_unmatched)}")

    # For remaining unmatched, assign nearest sector by centroid distance
    if still_unmatched:
        print("[INFO] Assigning nearest sector by centroid distance for remaining unmatched...")
        # compute sector centroids once
        sector_centroids = sectors_m.copy()
        sector_centroids["centroid"] = sector_centroids.geometry.centroid
        for idx in still_unmatched:
            ward_centroid = result.at[idx, "geometry"].centroid
            min_dist = math.inf
            min_sector = None
            for sidx, srow in sector_centroids.iterrows():
                d = ward_centroid.distance(srow["centroid"])
                if d < min_dist:
                    min_dist = d
                    min_sector = srow["sector"]
            result.at[idx, "sector_assigned"] = min_sector
            result.at[idx, "assign_method"] = "nearest_centroid"
            result.at[idx, "assign_dist_m"] = float(min_dist)

    # Back to EPSG:4326 for export
    result = result.to_crs(epsg=4326)

    # Build final GeoDataFrame to export
    final = result.copy()
    # keep ward_id, geometry, sector_assigned
    final = final.rename(columns={"sector_assigned":"sector"})
    if "ward_id" not in final.columns:
        final["ward_id"] = final.index.astype(str)

    # Fill any still-missing sector mark
    final["sector"] = final["sector"].fillna("UNASSIGNED")
    final["assign_method"] = final.get("assign_method").fillna("within")
    # write outputs
    print(f"[INFO] Writing: {OUT_GEOJSON}")
    final.to_file(OUT_GEOJSON, driver="GeoJSON")

    # Write summary
    per_sector = final["sector"].value_counts().rename_axis("sector").reset_index(name="ward_count")
    per_sector["total_wards"] = len(final)
    per_sector.to_csv(OUT_SUMMARY, index=False)

    # diagnostics
    total = len(final)
    unassigned = (final["sector"] == "UNASSIGNED").sum()
    print("---- ASSIGNMENT SUMMARY ----")
    print("total wards:", total)
    print("unassigned wards:", unassigned)
    print("\nPer-sector counts:")
    print(per_sector.to_string(index=False))

    print("[DONE] wards_with_sector_fixed.geojson & summary written. Reload your map with this GeoJSON.")

if __name__ == "__main__":
    main()
