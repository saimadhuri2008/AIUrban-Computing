#!/usr/bin/env python3
"""
Phase 6.7 — Facilities Placement (REVISED with Sector Rules)
- Emergency services distributed throughout sectors (not just borders)
- Schools allocated by sector function (minimal in Industrial)
- Smaller facility icons for better visualization
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import json
from pathlib import Path

BASE_DIR = Path("redesign")
ARTIFACTS = BASE_DIR /"artifacts"
SUMMARY = BASE_DIR / "summary"

for d in [ARTIFACTS,SUMMARY]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================
print("📊 Loading data...")

sectors = gpd.read_file(BASE_DIR/"data/processed/bbmp_5sectors_named.geojson").to_crs(4326)
wards = gpd.read_file(BASE_DIR/"artifacts/consumption_map.geojson").to_crs(4326)

# Load population forecast
pop_forecast = pd.read_csv("AI_forecasting/results/advanced/ensemble_bengaluru_forecasting/combined_forecast_2026_2035.csv")
pop_2035 = pop_forecast[pop_forecast['date'].str.startswith('2035-01')]
ward_pop_map = dict(zip(pop_2035['ward_id'], pop_2035['population']))

wards['ward_id'] = 'ward_' + wards.index.astype(str)
wards['population_2035'] = wards['ward_id'].map(ward_pop_map)
wards['population_2035'].fillna(wards['consumption_units'] * 50, inplace=True)

sector_pop = wards.groupby('sector')['population_2035'].sum().to_dict()
total_pop = sum(sector_pop.values())

# Define sector roles
SECTOR_ROLES = {
    'East': {'role': 'IT_Tech', 'school_priority': 'high', 'school_multiplier': 1.2},
    'West': {'role': 'Industrial', 'school_priority': 'low', 'school_multiplier': 0.4},
    'Central': {'role': 'Corporate_Govt', 'school_priority': 'medium', 'school_multiplier': 0.9},
    'South': {'role': 'Residential_Entertainment', 'school_priority': 'high', 'school_multiplier': 1.3},
    'North': {'role': 'Airport_Premium', 'school_priority': 'high', 'school_multiplier': 1.1}
}

print("\n" + "="*60)
print("📊 2035 POPULATION BY SECTOR")
print("="*60)
for sector, pop in sorted(sector_pop.items(), key=lambda x: x[1], reverse=True):
    role = SECTOR_ROLES[sector]['role']
    print(f"   {sector:12s} ({role:25s}): {pop:>12,.0f}")
print(f"   {'TOTAL':12s} {'':25s}: {total_pop:>12,.0f}")
print("="*60)

# ============================================================================
# HELPERS
# ============================================================================

def sample_points_in_polygon(polygon, n_points, seed=42, min_distance=0.005):
    """Generate random points distributed throughout polygon with minimum separation"""
    np.random.seed(seed)
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    max_attempts = n_points * 100  # Prevent infinite loops
    attempts = 0
    
    while len(points) < n_points and attempts < max_attempts:
        attempts += 1
        pnt = Point(
            np.random.uniform(minx, maxx),
            np.random.uniform(miny, maxy)
        )
        
        if not polygon.contains(pnt):
            continue
        
        # Check minimum distance from existing points
        too_close = False
        for existing_pt in points:
            if pnt.distance(existing_pt) < min_distance:
                too_close = True
                break
        
        if not too_close:
            points.append(pnt)
    
    # If we couldn't generate enough points with min_distance, fill remaining with no constraint
    if len(points) < n_points:
        print(f"   ⚠️  Generated {len(points)}/{n_points} points with separation, filling remainder")
        while len(points) < n_points:
            pnt = Point(
                np.random.uniform(minx, maxx),
                np.random.uniform(miny, maxy)
            )
            if polygon.contains(pnt):
                points.append(pnt)
    
    return points

def get_distributed_points(sector_name, n_points, seed=42, min_distance=0.005):
    """Get points distributed throughout sector with minimum separation (not just edges)"""
    geom = sectors[sectors['sector'] == sector_name].geometry.iloc[0]
    return sample_points_in_polygon(geom, n_points, seed, min_distance)

def create_geojson(features, filename):
    gdf = gpd.GeoDataFrame.from_features(features, crs=4326)
    
    invalid_count = 0
    for idx in gdf.index:
        if not gdf.loc[idx, 'geometry'].is_valid:
            gdf.loc[idx, 'geometry'] = gdf.loc[idx, 'geometry'].buffer(0)
            invalid_count += 1
    
    if invalid_count > 0:
        print(f"   ⚠️  Fixed {invalid_count} invalid geometries")
    
    output_path = f"redesign/artifacts/facilities/{filename}"
    gdf.to_file(output_path, driver='GeoJSON')
    print(f"   ✔ {filename} → {len(features)} features")
    return gdf

# ============================================================================
# 1. HOSPITALS (unchanged - population based)
# ============================================================================
print("\n🏥 Generating Hospitals...")

hospitals = []
hospital_id = 1
BEDS_PER_1000 = 2.5

for sector, pop in sector_pop.items():
    geom = sectors[sectors['sector'] == sector].geometry.iloc[0]
    required_beds = int(pop * BEDS_PER_1000 / 1000)
    
    # 1 super-specialty at center
    centroid = geom.centroid
    hospitals.append({
        'type': 'Feature',
        'geometry': {'type': 'Point', 'coordinates': [centroid.x, centroid.y]},
        'properties': {
            'id': f'H{hospital_id:03d}',
            'sector': sector,
            'hospital_type': 'super_specialty',
            'name': f'{sector} Super Specialty Hospital',
            'beds': 500,
            'specialties': 'Cardiology, Neurology, Oncology, Orthopedics, Trauma',
            'catchment_population': int(pop),
            'emergency': True
        }
    })
    hospital_id += 1
    
    # General hospitals distributed (with separation from super-specialty)
    remaining_beds = required_beds - 500
    n_general = max(int(remaining_beds / 200), 3)
    
    general_points = get_distributed_points(sector, n_general, seed=42+hospital_id, min_distance=0.008)
    
    for i, pt in enumerate(general_points):
        hospitals.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [pt.x, pt.y]},
            'properties': {
                'id': f'H{hospital_id:03d}',
                'sector': sector,
                'hospital_type': 'general',
                'name': f'{sector} General Hospital {i+1}',
                'beds': 200,
                'specialties': 'General Medicine, Surgery, Pediatrics, OB/GYN',
                'catchment_population': int(pop // n_general),
                'emergency': True
            }
        })
        hospital_id += 1

hospitals_gdf = create_geojson(hospitals, "hospitals_planned.geojson")
total_beds = sum(h['properties']['beds'] for h in hospitals)
print(f"   📌 Total: {len(hospitals)} hospitals, {total_beds:,} beds")

# ============================================================================
# 2. SCHOOLS (SECTOR-ROLE BASED) - REDUCED COUNT
# ============================================================================
print("\n🎓 Generating Schools (sector-role based)...")

schools = []
school_id = 1

# INCREASED standards to reduce total schools
PRIMARY_PER_PEOPLE = 5000  # Was 3000
SECONDARY_PER_PEOPLE = 8000  # Was 5000

for sector, pop in sector_pop.items():
    role_data = SECTOR_ROLES[sector]
    multiplier = role_data['school_multiplier']
    
    # Calculate schools with sector multiplier
    base_primary = int(pop * 0.25 / PRIMARY_PER_PEOPLE)
    base_secondary = int(pop * 0.25 / SECONDARY_PER_PEOPLE)
    
    n_primary = max(int(base_primary * multiplier), 1)
    n_secondary = max(int(base_secondary * multiplier), 1)
    
    # Industrial sector: minimal schools
    if sector == 'West':
        n_primary = max(3, int(n_primary * 0.3))  # Even fewer
        n_secondary = max(2, int(n_secondary * 0.3))
    
    # Get distributed points (with good separation)
    total_schools = n_primary + n_secondary
    school_points = get_distributed_points(sector, total_schools, seed=100+school_id, min_distance=0.012)  # Increased from 0.006
    
    # Primary schools
    for i, pt in enumerate(school_points[:n_primary]):
        schools.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [pt.x, pt.y]},
            'properties': {
                'id': f'S{school_id:03d}',
                'sector': sector,
                'school_type': 'primary',
                'name': f'{sector} Primary School {i+1}',
                'capacity': 1200,  # Increased from 600
                'grade_range': 'K-8',
                'priority': role_data['school_priority']
            }
        })
        school_id += 1
    
    # Secondary schools
    for i, pt in enumerate(school_points[n_primary:n_primary+n_secondary]):
        schools.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [pt.x, pt.y]},
            'properties': {
                'id': f'S{school_id:03d}',
                'sector': sector,
                'school_type': 'secondary',
                'name': f'{sector} Secondary School {i+1}',
                'capacity': 2000,  # Increased from 1000
                'grade_range': '9-12',
                'priority': role_data['school_priority']
            }
        })
        school_id += 1

schools_gdf = create_geojson(schools, "schools_planned.geojson")
total_capacity = sum(s['properties']['capacity'] for s in schools)
print(f"   📌 Total: {len(schools)} schools, {total_capacity:,} capacity")

# Print by sector
print("\n   Schools by sector:")
for sector in sectors['sector']:
    sector_schools = [s for s in schools if s['properties']['sector'] == sector]
    role = SECTOR_ROLES[sector]['role']
    priority = SECTOR_ROLES[sector]['school_priority']
    print(f"      {sector:12s} ({priority:6s}): {len(sector_schools):3d} schools")

# ============================================================================
# 3. PARKS (Smaller polygons - 15% coverage)
# ============================================================================
print("\n🌳 Generating Parks (15% green coverage)...")

parks = []
park_id = 1

for sector, pop in sector_pop.items():
    geom = sectors[sectors['sector'] == sector].geometry.iloc[0]
    sector_area_km2 = geom.area * (111**2)
    
    # 15% green space (reduced from 18%)
    green_target_km2 = sector_area_km2 * 0.15
    
    # One central park (25% of green space)
    central_park_area = green_target_km2 * 0.25
    central_radius = np.sqrt(central_park_area / np.pi) / 111
    centroid = geom.centroid
    central_geom = centroid.buffer(central_radius)
    
    parks.append({
        'type': 'Feature',
        'geometry': central_geom.__geo_interface__,
        'properties': {
            'id': f'P{park_id:03d}',
            'sector': sector,
            'park_type': 'regional',
            'name': f'{sector} Central Park',
            'area_km2': round(central_park_area, 3)
        }
    })
    park_id += 1
    
    # Small parks (remaining space)
    remaining = green_target_km2 - central_park_area
    avg_small_size = 0.02  # Smaller parks
    n_small = max(int(remaining*0.5 / avg_small_size), 8)
    
    small_points = get_distributed_points(sector, n_small, seed=200+park_id)
    
    for i, pt in enumerate(small_points):
        small_radius = np.sqrt(avg_small_size / np.pi) / 111
        small_geom = pt.buffer(small_radius)
        
        parks.append({
            'type': 'Feature',
            'geometry': small_geom.__geo_interface__,
            'properties': {
                'id': f'P{park_id:03d}',
                'sector': sector,
                'park_type': 'neighborhood',
                'name': f'{sector} Park {i+1}',
                'area_km2': round(avg_small_size, 3)
            }
        })
        park_id += 1

parks_gdf = create_geojson(parks, "parks_planned.geojson")
total_park_area = sum(p['properties']['area_km2'] for p in parks)
print(f"   📌 Total: {len(parks)} parks, {total_park_area:.1f} km²")

# ============================================================================
# 4. GOVERNMENT OFFICES
# ============================================================================
print("\n🏛️ Generating Government Offices...")

govt_offices = []
central_geom = sectors[sectors['sector'] == 'Central'].geometry.iloc[0]

govt_types = [
    {'name': 'City Municipal HQ', 'area': 0.15, 'emp': 2500},
    {'name': 'District Admin', 'area': 0.12, 'emp': 1800},
    {'name': 'Revenue Dept', 'area': 0.08, 'emp': 1000},
    {'name': 'PWD', 'area': 0.06, 'emp': 700}
]

govt_points = get_distributed_points('Central', len(govt_types), seed=400, min_distance=0.01)

for i, (pt, office) in enumerate(zip(govt_points, govt_types)):
    radius = np.sqrt(office['area'] / np.pi) / 111
    office_geom = pt.buffer(radius)
    
    govt_offices.append({
        'type': 'Feature',
        'geometry': office_geom.__geo_interface__,
        'properties': {
            'id': f'G{i+1:02d}',
            'sector': 'Central',
            'office_type': 'government',
            'name': office['name'],
            'area_km2': office['area'],
            'employees': office['emp']
        }
    })

govt_gdf = create_geojson(govt_offices, "govt_offices.geojson")
print(f"   📌 Total: {len(govt_offices)} offices")

# ============================================================================
# 5. EMERGENCY SERVICES (DISTRIBUTED THROUGHOUT)
# ============================================================================
print("\n🚨 Generating Emergency Services (distributed)...")

emergency_services = []
service_id = 1

for sector, pop in sector_pop.items():
    # Calculate based on population (REDUCED ratios)
    n_fire = max(int(pop / 80000), 2)  # Was 50000
    n_police = max(int(pop / 100000), 2)  # Was 75000
    
    # Get distributed points (NOT at borders, WITH MORE separation)
    fire_points = get_distributed_points(sector, n_fire, seed=500+service_id, min_distance=0.015)  # Increased from 0.01
    police_points = get_distributed_points(sector, n_police, seed=600+service_id, min_distance=0.015)  # Increased from 0.01
    
    # Fire stations
    for i, pt in enumerate(fire_points):
        emergency_services.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [pt.x, pt.y]},
            'properties': {
                'id': f'FS{service_id:02d}',
                'sector': sector,
                'service_type': 'fire_station',
                'name': f'{sector} Fire Station {i+1}',
                'vehicles': 6,
                'personnel': 35,
                'response_time_target_min': 5,
                'coverage_radius_km': 3.5
            }
        })
        service_id += 1
    
    # Police stations
    for i, pt in enumerate(police_points):
        emergency_services.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [pt.x, pt.y]},
            'properties': {
                'id': f'PS{service_id:02d}',
                'sector': sector,
                'service_type': 'police_station',
                'name': f'{sector} Police Station {i+1}',
                'officers': 60,
                'response_time_target_min': 8,
                'coverage_radius_km': 4.5
            }
        })
        service_id += 1

emergency_gdf = create_geojson(emergency_services, "police_fire_stations.geojson")

fire_count = len([e for e in emergency_services if e['properties']['service_type'] == 'fire_station'])
police_count = len([e for e in emergency_services if e['properties']['service_type'] == 'police_station'])
print(f"   📌 Fire: {fire_count}, Police: {police_count}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✅ REVISED FACILITIES (Reduced Count + Better Separation)")
print("="*70)
print(f"\n🏥 Hospitals: {len(hospitals)} ({total_beds:,} beds)")
print(f"🎓 Schools: {len(schools)} ({total_capacity:,} capacity)")
print(f"   └─ LARGER schools with 2× capacity, fewer total count")
print(f"🌳 Parks: {len(parks)} (12% green coverage, larger individual parks)")
print(f"🏛️ Govt: {len(govt_offices)}")
print(f"🚨 Emergency: {fire_count} fire + {police_count} police")
print(f"   └─ Reduced density for better spatial distribution")
print("="*70)