#!/usr/bin/env python3
"""
generate_candidates_and_travel.py

Generates:
 - candidates.geojson
 - travel_time.csv
 - costs.json (template)
 - optimisation_variables.json

Requirements:
    pip install geopandas pandas shapely pyproj numpy

Usage:
    python generate_candidates_and_travel.py --wards wards.geojson --sector_profile sector_future_profile.csv --out_dir ./ai_inputs --candidates_per_ward 3
"""

import argparse
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import json
import math
import uuid
from shapely.geometry import Point
import random

# haversine
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2*math.asin(min(1, math.sqrt(a)))
    return R * c

def jitter_point(x, y, max_meters=800):
    # jitter in meters -> convert to ~ degrees (approx)
    # 1 deg lat ~ 111km; 1 deg lon ~ 111km*cos(lat)
    # random distance and bearing
    d = random.random() * max_meters
    bearing = random.random() * 2*math.pi
    dy = d * math.cos(bearing)
    dx = d * math.sin(bearing)
    dlat = dy / 111000.0
    dlon = dx / (111000.0 * math.cos(math.radians(y)) + 1e-12)
    return x + dlon, y + dlat

def make_candidate_id(i):
    return f"cand_{i:05d}"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wards", required=True, help="wards.geojson with ward_id and geometry")
    p.add_argument("--sector_profile", required=True, help="sector_future_profile.csv")
    p.add_argument("--out_dir", default="./ai_inputs")
    p.add_argument("--candidates_per_ward", type=int, default=3)
    p.add_argument("--base_travel_speed_kmph", type=float, default=25.0)  # used to convert km->minutes for rough travel time
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading wards...")
    wards_gdf = gpd.read_file(args.wards)
    if 'ward_id' not in wards_gdf.columns:
        raise SystemExit("wards.geojson must have ward_id column")

    # ensure lon/lat
    wards_gdf = wards_gdf.to_crs(epsg=4326)
    wards_gdf['centroid'] = wards_gdf.geometry.centroid
    wards_gdf['lon'] = wards_gdf.centroid.x
    wards_gdf['lat'] = wards_gdf.centroid.y

    print("[INFO] Loading sector profile...")
    sector_df = pd.read_csv(args.sector_profile)
    # create sector priority as normalized population*failure or something simple
    # Here we compute sector_priority = normalized(population_2035 * avg_failure_probability)
    sector_df['priority_raw'] = sector_df['population_2035'] * sector_df['avg_failure_probability']
    sector_df['sector_priority'] = (sector_df['priority_raw'] - sector_df['priority_raw'].min()) / \
                                    max(1e-9, (sector_df['priority_raw'].max() - sector_df['priority_raw'].min()))
    sector_priority_map = sector_df.set_index('sector')['sector_priority'].to_dict()

    # build candidates
    candidates = []
    cid_counter = 1
    cand_rows_for_csv = []
    print("[INFO] Creating candidates (per ward)...")
    for _, w in wards_gdf.iterrows():
        wid = str(w['ward_id'])
        ward_lon = float(w['lon'])
        ward_lat = float(w['lat'])
        # simple ward population: if multiple rows per ward in your ward CSV, we will not sum here;
        # optimisation script later groups populations. For candidate scoring, use centroid-only.
        pop_proxy = float(w.get('population', 0)) if 'population' in w else 1.0
        sector = w.get('sector', None) if 'sector' in w else w.get('sector_id', None)
        sector = sector if sector is not None else "Unknown"
        # sector priority
        sec_pr = float(sector_priority_map.get(sector, 0.5))

        # choose candidate types in rotation (you can tune)
        types_pool = ['school','hospital','park','clinic','housing']
        # create N candidates
        for k in range(args.candidates_per_ward):
            cand_lon, cand_lat = jitter_point(ward_lon, ward_lat, max_meters=700)
            cand_id = make_candidate_id(cid_counter)
            cid_counter += 1

            # compute naive raw_score from population and sector priority and a bit of randomness
            raw_pop_n = min(1.0, math.log1p(max(1.0, pop_proxy)) / 12.0)  # small scaling
            randn = random.random() * 0.3
            # fail risk from ward mean fail_prob if present in attributes
            w_fail = float(w.get('fail_prob', 0.0)) if 'fail_prob' in w else 0.2
            raw_score = 0.4*raw_pop_n + 0.4*sec_pr + 0.2*(1.0 - w_fail) + randn*0.2
            raw_score = max(0.0, min(1.0, raw_score))

            # choose type: spread types - first candidate school, second hospital, third park, rotate
            typ = types_pool[k % len(types_pool)]
            # distance to nearest existing facility: if you have existing_facilities.geojson you can compute,
            # otherwise approximate with distance to ward centroid (makes sense if data not available)
            dist_to_existing_km = 0.0  # placeholder (update if you have existing facilities)

            # suitability score: combine raw_score, smaller penalty for proximity (we don't have existing), and sector priority
            suitability = 0.55*raw_score + 0.35*sec_pr + 0.1*(1.0 - w_fail)
            suitability = max(0.0, min(1.0, suitability))

            props = {
                "candidate_id": cand_id,
                "ward_id": wid,
                "type": typ,
                "raw_score": raw_score,
                "popn": raw_pop_n,
                "failn": w_fail,
                "sector": sector,
                "sector_priority": sec_pr,
                "dist_to_existing_km": dist_to_existing_km,
                "suitability_score": suitability
            }
            feature = {
                "type": "Feature",
                "properties": props,
                "geometry": {"type":"Point", "coordinates":[cand_lon, cand_lat]}
            }
            candidates.append(feature)
            cand_rows_for_csv.append({
                "candidate_id": cand_id,
                "ward_id": wid,
                "type": typ,
                "lon": cand_lon,
                "lat": cand_lat,
                "suitability_score": suitability
            })

    # write candidates.geojson
    cand_fc = {"type":"FeatureCollection","features":candidates}
    cand_path = out_dir / "candidates.geojson"
    print(f"[INFO] Writing {cand_path}")
    cand_path.write_text(json.dumps(cand_fc))

    # compute travel_time: ward -> candidate (minutes) using haversine and base speed (km/h)
    print("[INFO] Computing travel_time matrix (haversine -> minutes)")
    base_speed_kmph = float(args.base_travel_speed_kmph)
    travel = {}
    travel_csv_rows = []
    # build candidate index for quick access
    cand_df = pd.DataFrame(cand_rows_for_csv).set_index("candidate_id")
    for _, w in wards_gdf.iterrows():
        wid = str(w['ward_id'])
        travel[wid] = {}
        wlon = float(w['lon'])
        wlat = float(w['lat'])
        for cid, crow in cand_df.iterrows():
            clon = float(crow['lon'])
            clat = float(crow['lat'])
            km = haversine_km(wlon, wlat, clon, clat)
            # simple travel time estimation: time_minutes = km / speed_kmph * 60
            minutes = (km / max(0.1, base_speed_kmph)) * 60.0
            # add small network factor (urban road multiplier)
            minutes = minutes * 1.25
            # clamp
            minutes = float(round(minutes, 2))
            travel[wid][cid] = minutes
            travel_csv_rows.append({"ward_id": wid, "candidate_id": cid, "travel_min": minutes})

    # Save travel_time.csv (wide matrix) and JSON dictionary for optimisation input
    travel_csv = out_dir / "travel_time_long.csv"
    pd.DataFrame(travel_csv_rows).to_csv(travel_csv, index=False)
    # also write wide CSV (ward rows)
    cand_ids = list(cand_df.index)
    rows = []
    for wid in travel:
        row = {"ward_id": wid}
        for cid in cand_ids:
            row[cid] = travel[wid].get(cid, "")
        rows.append(row)
    wide_df = pd.DataFrame(rows).set_index("ward_id")
    wide_df.to_csv(out_dir / "travel_time_wide.csv")

    # costs.json template (you can tune numbers)
    costs = {
        "global_budget": 2.5e10,   # example: 25 billion
        "facilities": {
            "school": {"capex": 1e6, "capacity_students": 1200},
            "hospital": {"capex": 5e6, "capacity_beds": 200},
            "park": {"capex": 0.3e6, "area_hectares": 2},
            "clinic": {"capex": 0.5e6, "capacity_patients_per_day": 1000},
            "housing": {"capex": 0.8e6, "area_hectares": 1}
        },
        "weights": {"travel_time": 0.45, "capex": 0.3, "inequality": 0.15, "coverage": 0.1},
        "sector_minimum_requirements": {
            # example: require at least 2 hospitals in Central, 1 in others. tune for research
            "Central": {"hospitals": 2, "schools": 10},
            "East": {"hospitals": 1, "schools": 8},
            "West": {"hospitals": 2, "schools": 6},
            "South": {"hospitals": 1, "schools": 7},
            "North": {"hospitals": 1, "schools": 6}
        }
    }

    (out_dir / "costs.json").write_text(json.dumps(costs, indent=2))
    print(f"[INFO] Wrote costs.json -> {out_dir/'costs.json'}")

    # optimisation_variables.json (wards aggregated by ward_id population)
    print("[INFO] Building optimisation_variables.json")
    # If wards.json has multiple rows per ward (time series), we aggregate populations per ward by last value or sum
    # For safety we aggregate by taking average population if column exists
    ward_pop_map = {}
    if 'population' in wards_gdf.columns:
        # if population column present, take mean per ward_id
        for wid, grp in wards_gdf.groupby('ward_id'):
            # attempt numeric
            vals = pd.to_numeric(grp['population'], errors='coerce').dropna()
            ward_pop_map[wid] = float(vals.mean()) if len(vals)>0 else 0.0
    else:
        # fallback: assign 1
        for wid in wards_gdf['ward_id'].unique():
            ward_pop_map[wid] = 1.0

    wards_list = []
    for wid, pop in ward_pop_map.items():
        wards_list.append({"ward_id": wid, "sector": wards_gdf[wards_gdf['ward_id']==wid].iloc[0].get('sector', None), "population": pop})

    candidates_list = []
    for cid, crow in cand_df.reset_index().iterrows():
        candidates_list.append({
            "candidate_id": crow['candidate_id'] if 'candidate_id' in crow else cid,
            "ward_id": crow['ward_id'],
            "type": crow['type'],
            "lon": float(crow['lon']),
            "lat": float(crow['lat']),
            "suitability_score": float(crow['suitability_score'])
        })

    optimisation = {
        "wards": wards_list,
        "candidates": candidates_list,
        "travel_time": travel,
        "costs": costs,
        "sector_profile": sector_df.to_dict(orient='records')
    }

    (out_dir / "optimisation_variables.json").write_text(json.dumps(optimisation, indent=2))
    print(f"[SUCCESS] Wrote optimisation_variables.json -> {out_dir/'optimisation_variables.json'}")
    print(f"[SUCCESS] Wrote candidates.geojson -> {cand_path}")
    print(f"[SUCCESS] Wrote travel matrices -> {out_dir/'travel_time_long.csv'} and {out_dir/'travel_time_wide.csv'}")
    print("[DONE]")

if __name__ == "__main__":
    main()
