#!/usr/bin/env python3
"""
PHASE 6.6 — HOUSING & LAND-USE DISTRIBUTION
===========================================
Generates realistic land-use zones based on sector functional roles
"""

import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import numpy as np
import json
from pathlib import Path

BASE_DIR = Path("redesign")
ARTIFACTS = BASE_DIR /"artifacts/housing_landuse"
SUMMARY = BASE_DIR / "summary"

for d in [ARTIFACTS,SUMMARY]:
    d.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PHASE 6.6 — HOUSING & LAND-USE DISTRIBUTION")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📂 Loading data...")
sectors = gpd.read_file("redesign/data/processed/bbmp_5sectors_named.geojson").to_crs(4326)
wards = gpd.read_file(BASE_DIR/"artifacts/consumption_map.geojson").to_crs(4326)


sector_dict = {row.sector: row.geometry for _, row in sectors.iterrows()}

print("   ✓ Loaded")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_zones_in_sector(sector_name, num_zones, zone_size_deg=0.02):
    """Generate random non-overlapping zones within a sector"""
    sector_geom = sector_dict[sector_name]
    bounds = sector_geom.bounds
    zones = []
    
    attempts = 0
    max_attempts = num_zones * 50
    
    while len(zones) < num_zones and attempts < max_attempts:
        attempts += 1
        
        # Random center point
        x = np.random.uniform(bounds[0] + zone_size_deg, bounds[2] - zone_size_deg)
        y = np.random.uniform(bounds[1] + zone_size_deg, bounds[3] - zone_size_deg)
        center = Point(x, y)
        
        # Check if point is in sector
        if not sector_geom.contains(center):
            continue
        
        # Create zone polygon (roughly square)
        zone = Polygon([
            (x - zone_size_deg/2, y - zone_size_deg/2),
            (x + zone_size_deg/2, y - zone_size_deg/2),
            (x + zone_size_deg/2, y + zone_size_deg/2),
            (x - zone_size_deg/2, y + zone_size_deg/2)
        ])
        
        # Clip to sector boundary
        zone = zone.intersection(sector_geom)
        
        if zone.is_empty or zone.area < 0.0001:
            continue
        
        # Check overlap with existing zones
        overlaps = False
        for existing in zones:
            if zone.intersects(existing['geometry']):
                overlaps = True
                break
        
        if not overlaps:
            zones.append({'geometry': zone})
    
    return zones

def generate_corridor_zones(sector_names, num_zones, width=0.015):
    """Generate linear corridor zones along sector boundaries"""
    zones = []
    
    for i, sector_name in enumerate(sector_names):
        sector_geom = sector_dict[sector_name]
        
        # Get boundary line
        boundary = sector_geom.boundary
        
        # Sample points along boundary
        num_samples = num_zones // len(sector_names) + 1
        
        for j in range(num_samples):
            if len(zones) >= num_zones:
                break
            
            # Get point along boundary
            point = boundary.interpolate((j + 1) / (num_samples + 1), normalized=True)
            
            # Create corridor buffer
            corridor = point.buffer(width)
            corridor = corridor.intersection(sector_geom)
            
            if not corridor.is_empty and corridor.area > 0.0001:
                zones.append({'geometry': corridor})
    
    return zones[:num_zones]

def calculate_area_km2(geom):
    """Calculate approximate area in km²"""
    return geom.area * 111 * 111

# ============================================================================
# 1. RICH RESIDENTIAL (North + Central)
# ============================================================================
print("\n" + "="*80)
print("1. RICH RESIDENTIAL ZONES (North + Central)")
print("="*80)

rich_zones = []

# North sector: 12 zones (premium residential near airport)
north_zones = generate_zones_in_sector("North", 12, zone_size_deg=0.025)
for zone in north_zones:
    rich_zones.append({
        "type": "Feature",
        "geometry": zone['geometry'].__geo_interface__,
        "properties": {
            "land_use": "rich_residential",
            "category": "premium_villas",
            "sector": "North",
            "density": "low",
            "area_km2": round(calculate_area_km2(zone['geometry']), 2),
            "avg_plot_size_sqm": 500,
            "description": "Premium villas and gated communities"
        }
    })

# Central sector: 8 zones (high-rise luxury apartments)
central_zones = generate_zones_in_sector("Central", 8, zone_size_deg=0.02)
for zone in central_zones:
    rich_zones.append({
        "type": "Feature",
        "geometry": zone['geometry'].__geo_interface__,
        "properties": {
            "land_use": "rich_residential",
            "category": "luxury_apartments",
            "sector": "Central",
            "density": "medium",
            "area_km2": round(calculate_area_km2(zone['geometry']), 2),
            "avg_plot_size_sqm": 200,
            "description": "Luxury high-rise residential towers"
        }
    })

total_rich_area = sum(z['properties']['area_km2'] for z in rich_zones)
print(f"   ✓ Generated {len(rich_zones)} rich residential zones")
print(f"   ✓ Total area: {total_rich_area:.1f} km²")

# ============================================================================
# 2. MIDDLE INCOME (South + East)
# ============================================================================
print("\n" + "="*80)
print("2. MIDDLE INCOME HOUSING (South + East)")
print("="*80)

middle_zones = []

# South sector: 18 zones (standard apartments)
south_zones = generate_zones_in_sector("South", 18, zone_size_deg=0.018)
for zone in south_zones:
    middle_zones.append({
        "type": "Feature",
        "geometry": zone['geometry'].__geo_interface__,
        "properties": {
            "land_use": "middle_income",
            "category": "apartments",
            "sector": "South",
            "density": "medium-high",
            "area_km2": round(calculate_area_km2(zone['geometry']), 2),
            "avg_plot_size_sqm": 120,
            "description": "Standard apartment complexes"
        }
    })

# East sector: 15 zones (IT employee housing)
east_zones = generate_zones_in_sector("East", 15, zone_size_deg=0.018)
for zone in east_zones:
    middle_zones.append({
        "type": "Feature",
        "geometry": zone['geometry'].__geo_interface__,
        "properties": {
            "land_use": "middle_income",
            "category": "tech_housing",
            "sector": "East",
            "density": "medium-high",
            "area_km2": round(calculate_area_km2(zone['geometry']), 2),
            "avg_plot_size_sqm": 130,
            "description": "Housing for IT professionals"
        }
    })

total_middle_area = sum(z['properties']['area_km2'] for z in middle_zones)
print(f"   ✓ Generated {len(middle_zones)} middle income zones")
print(f"   ✓ Total area: {total_middle_area:.1f} km²")

# ============================================================================
# 3. AFFORDABLE HOUSING (West + South)
# ============================================================================
print("\n" + "="*80)
print("3. AFFORDABLE HOUSING (West + South)")
print("="*80)

affordable_zones = []

# West sector: 20 zones (worker housing near industrial)
west_zones = generate_zones_in_sector("West", 20, zone_size_deg=0.015)
for zone in west_zones:
    affordable_zones.append({
        "type": "Feature",
        "geometry": zone['geometry'].__geo_interface__,
        "properties": {
            "land_use": "affordable_housing",
            "category": "worker_housing",
            "sector": "West",
            "density": "high",
            "area_km2": round(calculate_area_km2(zone['geometry']), 2),
            "avg_plot_size_sqm": 60,
            "description": "Compact housing for industrial workers"
        }
    })

# South sector: 12 zones (mixed affordable)
south_afford_zones = generate_zones_in_sector("South", 12, zone_size_deg=0.015)
for zone in south_afford_zones:
    affordable_zones.append({
        "type": "Feature",
        "geometry": zone['geometry'].__geo_interface__,
        "properties": {
            "land_use": "affordable_housing",
            "category": "ews_housing",
            "sector": "South",
            "density": "high",
            "area_km2": round(calculate_area_km2(zone['geometry']), 2),
            "avg_plot_size_sqm": 50,
            "description": "Economically weaker section housing"
        }
    })

total_affordable_area = sum(z['properties']['area_km2'] for z in affordable_zones)
print(f"   ✓ Generated {len(affordable_zones)} affordable housing zones")
print(f"   ✓ Total area: {total_affordable_area:.1f} km²")

# ============================================================================
# 4. MIXED-USE (Central + East corridors)
# ============================================================================
print("\n" + "="*80)
print("4. MIXED-USE ZONES (Central + East Corridors)")
print("="*80)

mixed_zones = []

# Central sector: 10 zones (commercial + residential)
central_mixed = generate_zones_in_sector("Central", 10, zone_size_deg=0.02)
for zone in central_mixed:
    mixed_zones.append({
        "type": "Feature",
        "geometry": zone['geometry'].__geo_interface__,
        "properties": {
            "land_use": "mixed_use",
            "category": "commercial_residential",
            "sector": "Central",
            "density": "high",
            "area_km2": round(calculate_area_km2(zone['geometry']), 2),
            "commercial_pct": 40,
            "residential_pct": 60,
            "description": "Mixed commercial and residential"
        }
    })

# East sector corridors: 8 zones (tech parks + housing)
east_mixed = generate_corridor_zones(["East"], 8, width=0.018)
for zone in east_mixed:
    mixed_zones.append({
        "type": "Feature",
        "geometry": zone['geometry'].__geo_interface__,
        "properties": {
            "land_use": "mixed_use",
            "category": "tech_corridor",
            "sector": "East",
            "density": "medium-high",
            "area_km2": round(calculate_area_km2(zone['geometry']), 2),
            "commercial_pct": 50,
            "residential_pct": 50,
            "description": "Tech parks with residential"
        }
    })

total_mixed_area = sum(z['properties']['area_km2'] for z in mixed_zones)
print(f"   ✓ Generated {len(mixed_zones)} mixed-use zones")
print(f"   ✓ Total area: {total_mixed_area:.1f} km²")

# ============================================================================
# 5. INDUSTRIAL ZONES (West)
# ============================================================================
print("\n" + "="*80)
print("5. INDUSTRIAL ZONES (West)")
print("="*80)

industrial_zones = []

# West sector: 15 zones (factories, warehouses, logistics)
west_industrial = generate_zones_in_sector("West", 15, zone_size_deg=0.025)
for zone in west_industrial:
    industrial_zones.append({
        "type": "Feature",
        "geometry": zone['geometry'].__geo_interface__,
        "properties": {
            "land_use": "industrial",
            "category": "manufacturing",
            "sector": "West",
            "area_km2": round(calculate_area_km2(zone['geometry']), 2),
            "type": np.random.choice(["light_industry", "heavy_industry", "logistics"]),
            "description": "Industrial and manufacturing zones"
        }
    })

total_industrial_area = sum(z['properties']['area_km2'] for z in industrial_zones)
print(f"   ✓ Generated {len(industrial_zones)} industrial zones")
print(f"   ✓ Total area: {total_industrial_area:.1f} km²")

# ============================================================================
# 6. SLUM UPGRADATION (Scattered, density-based)
# ============================================================================
print("\n" + "="*80)
print("6. SLUM UPGRADATION ZONES (High Density Areas)")
print("="*80)

slum_zones = []

# Identify high-density wards (top 30%)
wards_sorted = wards.sort_values('consumption_units', ascending=False)
high_density_wards = wards_sorted.head(int(len(wards) * 0.3))

# Generate small zones in high-density areas
for _, ward in high_density_wards.iterrows():
    ward_geom = ward.geometry
    centroid = ward_geom.centroid
    
    # Small buffer around centroid
    slum_zone = centroid.buffer(0.008).intersection(ward_geom)
    
    if not slum_zone.is_empty and slum_zone.area > 0.0001:
        slum_zones.append({
            "type": "Feature",
            "geometry": slum_zone.__geo_interface__,
            "properties": {
                "land_use": "slum_upgradation",
                "category": "informal_settlement",
                "sector": ward['sector'],
                "area_km2": round(calculate_area_km2(slum_zone), 2),
                "status": "planned_upgradation",
                "description": "Slum upgradation and formalization"
            }
        })

total_slum_area = sum(z['properties']['area_km2'] for z in slum_zones)
print(f"   ✓ Generated {len(slum_zones)} slum upgradation zones")
print(f"   ✓ Total area: {total_slum_area:.1f} km²")

# ============================================================================
# SAVE FILES
# ============================================================================
print("\n" + "="*80)
print("SAVING FILES")
print("="*80)

out = ARTIFACTS
out.mkdir(parents=True, exist_ok=True)

# Save each land-use type
datasets = {
    "rich_residential.geojson": rich_zones,
    "middle_income.geojson": middle_zones,
    "affordable_housing.geojson": affordable_zones,
    "mixed_use.geojson": mixed_zones,
    "industrial_zones.geojson": industrial_zones,
    "slum_upgradation.geojson": slum_zones
}

for filename, zones in datasets.items():
    with open(ARTIFACTS / filename, "w") as f:
        json.dump({"type": "FeatureCollection", "features": zones}, f, indent=2)
    print(f"   ✓ {filename}: {len(zones)} zones")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("LAND-USE DISTRIBUTION SUMMARY")
print("="*80)

print(f"\n🏘️  HOUSING & LAND-USE:")
print(f"   • Rich Residential: {len(rich_zones)} zones, {total_rich_area:.1f} km²")
print(f"   • Middle Income: {len(middle_zones)} zones, {total_middle_area:.1f} km²")
print(f"   • Affordable: {len(affordable_zones)} zones, {total_affordable_area:.1f} km²")
print(f"   • Mixed-Use: {len(mixed_zones)} zones, {total_mixed_area:.1f} km²")
print(f"   • Industrial: {len(industrial_zones)} zones, {total_industrial_area:.1f} km²")
print(f"   • Slum Upgradation: {len(slum_zones)} zones, {total_slum_area:.1f} km²")

total_area = sum([total_rich_area, total_middle_area, total_affordable_area, 
                  total_mixed_area, total_industrial_area, total_slum_area])
print(f"\n   📊 Total Planned Area: {total_area:.1f} km²")

print("\n✅ PHASE 6.6 COMPLETE")
print("="*80 + "\n")