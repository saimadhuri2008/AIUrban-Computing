#!/usr/bin/env python3
"""
BANGALORE TRANSPORT NETWORK - CLEAN & MINIMAL
=============================================
FIXED: ORR now saved as LineString for proper visualization
"""

import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import unary_union, nearest_points
import numpy as np
import json
from pathlib import Path
import logging
import sys
import hashlib
import platform
from datetime import datetime


BASE_DIR = Path("redesign")
ARTIFACTS_TRANSPORT = BASE_DIR /"artifacts/transportlayer"
ARTIFACTS = BASE_DIR / "artifacts"
SUMMARY = BASE_DIR / "summary"

for d in [ARTIFACTS_TRANSPORT,SUMMARY]:
    d.mkdir(parents=True, exist_ok=True)

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "transport_layer.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


logger.info("Transport network (roads + metro) generation started")


# ============================================================================
# LOAD DATA
# ============================================================================

sectors = gpd.read_file(BASE_DIR/"data/processed/bbmp_5sectors_named.geojson").to_crs(4326)
sector_dict = {row.sector: row.geometry for _, row in sectors.iterrows()}

central_geom = sector_dict["Central"]
central_centroid = central_geom.centroid
city_boundary = unary_union(sectors.geometry)



# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_extreme_boundary_point(geom, direction):
    """Get the most extreme point on boundary in given direction"""
    bounds = geom.bounds
    centroid = geom.centroid
    
    # Target point far in that direction
    if direction == 'north':
        search_pt = Point(centroid.x, bounds[3] + 1)
    elif direction == 'south':
        search_pt = Point(centroid.x, bounds[1] - 1)
    elif direction == 'east':
        search_pt = Point(bounds[2] + 1, centroid.y)
    else:  # west
        search_pt = Point(bounds[0] - 1, centroid.y)
    
    # Sample boundary and find closest to search point
    boundary_samples = [geom.exterior.interpolate(i/200, normalized=True) for i in range(200)]
    return min(boundary_samples, key=lambda p: p.distance(search_pt))

def get_sector_central_boundary_point(sector_name):
    """Get point where sector boundary meets Central boundary"""
    sector_geom = sector_dict[sector_name]
    
    if sector_geom.touches(central_geom):
        intersection = sector_geom.boundary.intersection(central_geom.boundary)
        if not intersection.is_empty:
            if hasattr(intersection, 'geoms'):
                points = [g.centroid if g.geom_type == 'LineString' else g for g in intersection.geoms]
                return points[0]
            return intersection.centroid if intersection.geom_type == 'LineString' else intersection
    
    return nearest_points(sector_geom, central_geom)[1]

# ============================================================================
# ROAD NETWORK - ONLY ESSENTIAL ROADS
# ============================================================================

logger.info("PHASE 1: ESSENTIAL ROAD NETWORK")


road_features = []

# ────────────────────────────────────────────────────────────────────────────
# 1. OUTER RING ROAD (ORR) - FIXED TO BE LineString
# ────────────────────────────────────────────────────────────────────────────
logger.info(" Outer Ring Road")

# Get exterior as LinearRing, convert to LineString
orr_ring = city_boundary.exterior
orr_coords = list(orr_ring.coords)
# Create LineString (this ensures proper GeoJSON format)
orr_geom = LineString(orr_coords)
orr_length = orr_geom.length * 111

road_features.append({
    "type": "Feature",
    "geometry": orr_geom.__geo_interface__,
    "properties": {
        "road_type": "expressway",
        "category": "ring_road",
        "name": "Outer Ring Road (ORR)",
        "lanes": 10,
        "length_km": round(orr_length, 2)
    }
})

logger.info(f"ORR: {orr_length:.1f} km")
logger.info(f"Geometry type: {orr_geom.geom_type}")
logger.info(f"Coordinates: {len(orr_coords)} points")

# ────────────────────────────────────────────────────────────────────────────
# 2. FOUR ARTERIALS: Sector Boundary → Central Boundary
# ────────────────────────────────────────────────────────────────────────────
logger.info("2. Primary Arterials (4 roads: Boundary → Central)")

directions_map = {
    "North": "north",
    "South": "south",
    "East": "east",
    "West": "west"
}

for sector_name, direction in directions_map.items():
    sector_geom = sector_dict[sector_name]
    
    # Outer extreme point on sector boundary
    outer_point = get_extreme_boundary_point(sector_geom, direction)
    
    # Where sector meets Central
    central_point = get_sector_central_boundary_point(sector_name)
    
    # Create clean arterial
    arterial = LineString([outer_point, central_point])
    length_km = arterial.length * 111
    
    road_features.append({
        "type": "Feature",
        "geometry": arterial.__geo_interface__,
        "properties": {
            "road_type": "arterial",
            "category": "sector_to_central",
            "name": f"{sector_name} Arterial",
            "sector": sector_name,
            "lanes": 8,
            "length_km": round(length_km, 2)
        }
    })
    
    logger.info(f"{sector_name} Arterial: {length_km:.1f} km")

# ────────────────────────────────────────────────────────────────────────────
# 3. FOUR INTER-SECTOR ROADS: Proper boundary-to-boundary connections
# ────────────────────────────────────────────────────────────────────────────
logger.info("3.Inter-Sector Connectors (4 roads: Sector ↔ Sector)")

def get_sector_boundary_point_towards(from_sector, to_sector):
    """Get point on from_sector boundary that's closest to to_sector centroid"""
    from_geom = sector_dict[from_sector]
    to_centroid = sector_dict[to_sector].centroid
    
    # Sample boundary points
    boundary_samples = [from_geom.exterior.interpolate(i/300, normalized=True) 
                       for i in range(300)]
    
    # Find closest to target sector's centroid
    return min(boundary_samples, key=lambda p: p.distance(to_centroid))

# North ↔ East: Connect their facing boundaries
ne_north_pt = get_sector_boundary_point_towards("North", "East")
ne_east_pt = get_sector_boundary_point_towards("East", "North")
ne_road = LineString([ne_north_pt, ne_east_pt])

road_features.append({
    "type": "Feature",
    "geometry": ne_road.__geo_interface__,
    "properties": {
        "road_type": "collector",
        "category": "inter_sector",
        "name": "North-East Connector",
        "sector": "North, East",
        "lanes": 6,
        "length_km": round(ne_road.length * 111, 2)
    }
})
logger.info(f"North ↔ East: {ne_road.length * 111:.1f} km")

# East ↔ South: Connect their facing boundaries
es_east_pt = get_sector_boundary_point_towards("East", "South")
es_south_pt = get_sector_boundary_point_towards("South", "East")
es_road = LineString([es_east_pt, es_south_pt])

road_features.append({
    "type": "Feature",
    "geometry": es_road.__geo_interface__,
    "properties": {
        "road_type": "collector",
        "category": "inter_sector",
        "name": "East-South Connector",
        "sector": "East, South",
        "lanes": 6,
        "length_km": round(es_road.length * 111, 2)
    }
})
print(f"   ✓ East ↔ South: {es_road.length * 111:.1f} km")

# South ↔ West: Connect their facing boundaries
sw_south_pt = get_sector_boundary_point_towards("South", "West")
sw_west_pt = get_sector_boundary_point_towards("West", "South")
sw_road = LineString([sw_south_pt, sw_west_pt])

road_features.append({
    "type": "Feature",
    "geometry": sw_road.__geo_interface__,
    "properties": {
        "road_type": "collector",
        "category": "inter_sector",
        "name": "South-West Connector",
        "sector": "South, West",
        "lanes": 6,
        "length_km": round(sw_road.length * 111, 2)
    }
})
print(f"   ✓ South ↔ West: {sw_road.length * 111:.1f} km")

# West ↔ North: Connect their facing boundaries
wn_west_pt = get_sector_boundary_point_towards("West", "North")
wn_north_pt = get_sector_boundary_point_towards("North", "West")
wn_road = LineString([wn_west_pt, wn_north_pt])

road_features.append({
    "type": "Feature",
    "geometry": wn_road.__geo_interface__,
    "properties": {
        "road_type": "collector",
        "category": "inter_sector",
        "name": "West-North Connector",
        "sector": "West, North",
        "lanes": 6,
        "length_km": round(wn_road.length * 111, 2)
    }
})
print(f"   ✓ West ↔ North: {wn_road.length * 111:.1f} km")

# ────────────────────────────────────────────────────────────────────────────
# 4. TWO CENTRAL INTERNAL ROADS: N-S and E-W Boulevards
# ────────────────────────────────────────────────────────────────────────────
print("\n4. Central Sector Internal Roads (2 boulevards)")

# Get entry points to Central from each direction
north_entry = get_sector_central_boundary_point("North")
south_entry = get_sector_central_boundary_point("South")
east_entry = get_sector_central_boundary_point("East")
west_entry = get_sector_central_boundary_point("West")

# North-South boulevard through Central
central_ns = LineString([north_entry, south_entry])
road_features.append({
    "type": "Feature",
    "geometry": central_ns.__geo_interface__,
    "properties": {
        "road_type": "arterial",
        "category": "central_internal",
        "name": "Central Boulevard N-S",
        "sector": "Central",
        "lanes": 8,
        "length_km": round(central_ns.length * 111, 2)
    }
})

# East-West boulevard through Central
central_ew = LineString([east_entry, west_entry])
road_features.append({
    "type": "Feature",
    "geometry": central_ew.__geo_interface__,
    "properties": {
        "road_type": "arterial",
        "category": "central_internal",
        "name": "Central Boulevard E-W",
        "sector": "Central",
        "lanes": 8,
        "length_km": round(central_ew.length * 111, 2)
    }
})

print(f"   ✓ Central N-S: {central_ns.length * 111:.1f} km")
print(f"   ✓ Central E-W: {central_ew.length * 111:.1f} km")

# ============================================================================
# METRO NETWORK - TWO CLEAN LINES
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: METRO NETWORK")
print("="*80)

metro_features = []

# ────────────────────────────────────────────────────────────────────────────
# METRO LINE 1: Purple Line (North → Central → West)
# ────────────────────────────────────────────────────────────────────────────
print("\n1. Purple Line")

# Outer points with offset from roads
north_metro_start = get_extreme_boundary_point(sector_dict["North"], "north")
north_metro_start = Point(north_metro_start.x + 0.02, north_metro_start.y)

north_metro_entry = get_sector_central_boundary_point("North")
north_metro_entry = Point(north_metro_entry.x + 0.015, north_metro_entry.y)

# Central station northwest
central_purple = Point(central_centroid.x - 0.015, central_centroid.y + 0.015)

west_metro_exit = get_sector_central_boundary_point("West")
west_metro_exit = Point(west_metro_exit.x, west_metro_exit.y + 0.015)

west_metro_end = get_extreme_boundary_point(sector_dict["West"], "west")
west_metro_end = Point(west_metro_end.x, west_metro_end.y + 0.02)

purple_line = LineString([
    north_metro_start,
    north_metro_entry,
    central_purple,
    west_metro_exit,
    west_metro_end
])

purple_length = purple_line.length * 111
purple_stations = max(6, int(purple_length / 2))

metro_features.append({
    "type": "Feature",
    "geometry": purple_line.__geo_interface__,
    "properties": {
        "line_id": "Line_1",
        "line_name": "Purple Line",
        "color": "#9B59B6",
        "corridor": "North-Central-West",
        "sector": "North, Central, West",
        "length_km": round(purple_length, 2),
        "stations": purple_stations,
        "type": "underground",
        "ridership_daily": 450000
    }
})

print(f"   ✓ Purple: {purple_length:.1f} km, {purple_stations} stations")

# ────────────────────────────────────────────────────────────────────────────
# METRO LINE 2: Green Line (South → Central → East)
# ────────────────────────────────────────────────────────────────────────────
print("\n2. Green Line")

# Outer points with DIFFERENT offset
south_metro_start = get_extreme_boundary_point(sector_dict["South"], "south")
south_metro_start = Point(south_metro_start.x - 0.02, south_metro_start.y)

south_metro_entry = get_sector_central_boundary_point("South")
south_metro_entry = Point(south_metro_entry.x - 0.015, south_metro_entry.y)

# Central station southeast (SEPARATED from purple)
central_green = Point(central_centroid.x + 0.015, central_centroid.y - 0.015)

east_metro_exit = get_sector_central_boundary_point("East")
east_metro_exit = Point(east_metro_exit.x, east_metro_exit.y - 0.015)

east_metro_end = get_extreme_boundary_point(sector_dict["East"], "east")
east_metro_end = Point(east_metro_end.x, east_metro_end.y - 0.02)

green_line = LineString([
    south_metro_start,
    south_metro_entry,
    central_green,
    east_metro_exit,
    east_metro_end
])

green_length = green_line.length * 111
green_stations = max(6, int(green_length / 2))

metro_features.append({
    "type": "Feature",
    "geometry": green_line.__geo_interface__,
    "properties": {
        "line_id": "Line_2",
        "line_name": "Green Line",
        "color": "#27AE60",
        "corridor": "South-Central-East",
        "sector": "South, Central, East",
        "length_km": round(green_length, 2),
        "stations": green_stations,
        "type": "underground",
        "ridership_daily": 420000
    }
})

print(f"   ✓ Green: {green_length:.1f} km, {green_stations} stations")

separation = central_purple.distance(central_green) * 111
print(f"\n   ✓ Station separation: {separation:.2f} km")

# ============================================================================
# SAVE FILES
# ============================================================================
print("\n" + "="*80)
print("SAVING FILES")
print("="*80)

out = ARTIFACTS_TRANSPORT
out.mkdir(parents=True, exist_ok=True)

# Save roads
total_road_km = sum(f["properties"]["length_km"] for f in road_features)

with open(out / "transport_roads.geojson", "w") as f:
    json.dump({"type": "FeatureCollection", "features": road_features}, f, indent=2)

logger.info(f"Saved transport roads GeoJSON: {out / 'transport_roads.geojson'}")


print(f"✓ Roads: {len(road_features)} segments, {total_road_km:.1f} km")

# Save metro
total_metro_km = sum(f["properties"]["length_km"] for f in metro_features)
total_stations = sum(f["properties"]["stations"] for f in metro_features)

with open(out / "metro_network.geojson", "w") as f:
    json.dump({"type": "FeatureCollection", "features": metro_features}, f, indent=2)

logger.info(f"Saved metro network GeoJSON: {out / 'metro_network.geojson'}")

ARTIFACT_METADATA_DIR = ARTIFACTS / "metadata"
ARTIFACT_METADATA_DIR.mkdir(parents=True, exist_ok=True)

artifact_metadata = {
    "artifacts": [
        "transport_roads.geojson",
        "metro_network.geojson"
    ],
    "road_design": {
        "orr": "Single outer ring road",
        "arterials": 4,
        "inter_sector_connectors": 4,
        "central_internal_roads": 2
    },
    "metro_design": {
        "lines": 2,
        "corridors": [
            "North–Central–West",
            "South–Central–East"
        ]
    },
    "crs": "EPSG:4326",
    "phase": "Phase 6",
    "description": "Minimal, hierarchical transport network for redesigned 5-sector city"
}

with open(ARTIFACT_METADATA_DIR / "transport_layer_metadata.json", "w") as f:
    json.dump(artifact_metadata, f, indent=2)

METADATA_DIR = BASE_DIR / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

run_manifest = {
    "script": "build_transport_layer.py",
    "timestamp_utc": datetime.utcnow().isoformat(),
    "input_file": "bbmp_5sectors_named.geojson",
    "output_files": [
        str(out / "transport_roads.geojson"),
        str(out / "metro_network.geojson")
    ],
    "python_version": sys.version,
    "platform": platform.platform(),

}

with open(METADATA_DIR / "run_manifest_transportlayer.json", "w") as f:
    json.dump(run_manifest, f, indent=2)



logger.info(f"Metro: {len(metro_features)} lines, {total_metro_km:.1f} km, {total_stations} stations")

# ============================================================================
# SUMMARY
# ============================================================================

logger.info("CLEAN MINIMAL NETWORK - SUMMARY")

logging.info(
    "road_summary road_count=%d total_km=%.0f",
    len(road_features),
    total_road_km
)

logging.info("road_type name=ORR count=1 description=Outer_Ring_Road")
logging.info("road_type name=Arterial count=4 direction=Sector_to_Central")
logging.info("road_type name=Inter_Sector_Connector count=4")
logging.info("road_type name=Central_Internal_Boulevard count=2")

# Metro summary
logging.info(
    "metro_summary line_count=%d total_km=%.0f station_count=%d",
    len(metro_features),
    total_metro_km,
    total_stations
)

logging.info("metro_line name=Purple route=North_Central_West")
logging.info("metro_line name=Green route=South_Central_East")


logger.info(" Transport network generation completed successfully")
