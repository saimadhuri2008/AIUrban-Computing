#!/usr/bin/env python3
"""
Phase 7.3 — Equity Optimization (REFINED VERSION)
Fixes MILP infeasibility and improves equity metrics computation

KEY IMPROVEMENTS:
1. Relaxed proximity constraints (preventing infeasibility)
2. Multi-stage optimization approach
3. Better handling of edge cases
4. More realistic facility placement
5. Enhanced visualization
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from ortools.linear_solver import pywraplp
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PHASE 7.3 — EQUITY OPTIMIZATION (REFINED)")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'hospital_per_pop': 100000,  # 1 hospital per 100k
    'school_per_pop': 10000,     # 1 school per 10k
    'emergency_per_pop': 50000,  # 1 emergency per 50k
    'min_facility_spacing_km': 1.0,  # Reduced from 2.0 for more flexibility
    'equity_weight': 2.0,        # Prioritize underserved areas
    'optimization_time_limit': 120,  # seconds
}

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading data...")

# Wards with socioeconomic data
wards = gpd.read_file("data/processed/wards/wards_2035_All_variables.geojson").to_crs(4326)

# Deduplicate
if 'ward_id' in wards.columns:
    wards = wards.drop_duplicates(subset='ward_id', keep='first')
elif 'Ward_No' in wards.columns:
    wards = wards.drop_duplicates(subset='Ward_No', keep='first')
wards = wards.reset_index(drop=True)

print(f"✓ Loaded {len(wards)} unique wards")

# Load optimized facilities from Phase 7.1
try:
    hospitals = gpd.read_file("results/optimization/facilities/hospital_optimized.geojson").to_crs(4326)
    schools = gpd.read_file("results/optimization/facilities/school_optimized.geojson").to_crs(4326)
    emergency = gpd.read_file("results/optimization/facilities/emergency_optimized.geojson").to_crs(4326)
except:
    # Fallback to phase7_outputs
    hospitals = gpd.read_file("phase7_outputs/hospital_optimized.geojson").to_crs(4326)
    schools = gpd.read_file("phase7_outputs/school_optimized.geojson").to_crs(4326)
    emergency = gpd.read_file("phase7_outputs/emergency_optimized.geojson").to_crs(4326)

print(f"✓ Loaded {len(hospitals)} hospitals, {len(schools)} schools, {len(emergency)} emergency")

# ============================================================================
# EXTRACT SOCIOECONOMIC DATA
# ============================================================================
print("\n" + "="*70)
print("EXTRACTING SOCIOECONOMIC INDICATORS")
print("="*70)

# Get population
pop_col = 'population_2035' if 'population_2035' in wards.columns else 'population'
population = wards[pop_col].values

# Estimate income from population density (inverse relationship for Indian cities)
area = wards.geometry.area * (111**2)  # km²
density = population / (area + 0.01)

# Inverse log relationship: higher density often = lower income (informal settlements)
# Normalize to realistic range (10k - 100k INR/month)
income = 100000 - (np.log1p(density) / np.log1p(density.max())) * 70000
income = np.clip(income, 10000, 100000)
wards['income_estimated'] = income

print(f"✓ Income range: ₹{income.min():,.0f} - ₹{income.max():,.0f} per month")

# Identify income groups
income_p30 = np.percentile(income, 30)
income_p90 = np.percentile(income, 90)

low_income = income <= income_p30
high_income = income >= income_p90

print(f"✓ Low-income wards (bottom 30%): {low_income.sum()} (₹{income_p30:,.0f}/mo)")
print(f"✓ High-income wards (top 10%): {high_income.sum()} (₹{income_p90:,.0f}/mo)")

# ============================================================================
# COMPUTE ACCESSIBILITY METRICS
# ============================================================================
print("\n" + "="*70)
print("COMPUTING ACCESSIBILITY METRICS")
print("="*70)

def compute_access(wards_gdf, facilities_gdf):
    """Compute travel time from each ward to nearest facility"""
    ward_coords = np.array([[w.geometry.centroid.x, w.geometry.centroid.y] 
                            for _, w in wards_gdf.iterrows()])
    
    facility_coords = []
    for _, f in facilities_gdf.iterrows():
        if f.geometry.geom_type == 'Point':
            facility_coords.append([f.geometry.x, f.geometry.y])
        else:
            c = f.geometry.centroid
            facility_coords.append([c.x, c.y])
    
    if len(facility_coords) == 0:
        return np.full(len(wards_gdf), 999)
    
    facility_coords = np.array(facility_coords)
    tree = cKDTree(facility_coords)
    distances, _ = tree.query(ward_coords, k=1)
    
    # Convert to travel time (35 km/h avg speed)
    distances_km = distances * 111
    travel_times = (distances_km / 35) * 60
    
    return travel_times

print("  Computing accessibility...")
hosp_access = compute_access(wards, hospitals)
school_access = compute_access(wards, schools)
emerg_access = compute_access(wards, emergency)

# Weighted composite (hospital 40%, school 30%, emergency 30%)
avg_access = hosp_access * 0.4 + school_access * 0.3 + emerg_access * 0.3

print(f"✓ Average access times:")
print(f"   Hospitals: {hosp_access.mean():.1f} min")
print(f"   Schools: {school_access.mean():.1f} min")
print(f"   Emergency: {emerg_access.mean():.1f} min")
print(f"   Composite: {avg_access.mean():.1f} min")

# ============================================================================
# BASELINE EQUITY METRICS
# ============================================================================
print("\n" + "="*70)
print("BASELINE EQUITY METRICS")
print("="*70)

def gini_coefficient(values):
    """Gini coefficient (0=equality, 1=inequality)"""
    sorted_vals = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_vals)
    return (2 * np.sum((np.arange(1, n+1)) * sorted_vals)) / (n * cumsum[-1]) - (n + 1) / n

def palma_ratio(values, income):
    """Top 10% vs bottom 40% service access ratio"""
    p90 = np.percentile(income, 90)
    p40 = np.percentile(income, 40)
    
    top_access = values[income >= p90].mean()
    bottom_access = values[income <= p40].mean()
    
    # Convert to service level (1/time)
    top_service = 1 / (top_access + 1)
    bottom_service = 1 / (bottom_access + 1)
    
    return top_service / (bottom_service + 0.0001)

baseline_gini = gini_coefficient(avg_access)
baseline_max_min = avg_access.max() - avg_access.min()
baseline_low_avg = avg_access[low_income].mean()
baseline_high_avg = avg_access[high_income].mean()
baseline_gap = baseline_low_avg - baseline_high_avg
baseline_palma = palma_ratio(avg_access, income)

print(f"\n📊 BASELINE METRICS:")
print(f"   Gini coefficient: {baseline_gini:.3f}")
print(f"   Max-min gap: {baseline_max_min:.1f} min")
print(f"   Low-income avg: {baseline_low_avg:.1f} min")
print(f"   High-income avg: {baseline_high_avg:.1f} min")
print(f"   Income gap: {baseline_gap:.1f} min")
print(f"   Palma ratio: {baseline_palma:.2f}")

# ============================================================================
# IDENTIFY UNDERSERVED WARDS
# ============================================================================
print("\n" + "="*70)
print("IDENTIFYING UNDERSERVED WARDS")
print("="*70)

# Underserved = low income AND poor access (worse than median)
access_median = np.median(avg_access)
underserved = low_income & (avg_access >= access_median)

n_underserved = underserved.sum()
underserved_pop = population[underserved].sum()

print(f"✓ Underserved wards: {n_underserved} ({n_underserved/len(wards)*100:.1f}%)")
print(f"   Population: {underserved_pop:,.0f}")
print(f"   Criteria: Bottom 30% income + worse than median access")

# ============================================================================
# COMPUTE FACILITY NEEDS
# ============================================================================
print("\n" + "="*70)
print("COMPUTING ADDITIONAL FACILITY NEEDS")
print("="*70)

# Count existing facilities in underserved areas
underserved_geom = wards[underserved].unary_union.buffer(0.01)

existing_hosp_underserved = len(hospitals[hospitals.intersects(underserved_geom)])
existing_school_underserved = len(schools[schools.intersects(underserved_geom)])
existing_emerg_underserved = len(emergency[emergency.intersects(underserved_geom)])

# Compute needs
needed_hosp = max(0, int(underserved_pop / CONFIG['hospital_per_pop']) - existing_hosp_underserved)
needed_school = max(0, int(underserved_pop / CONFIG['school_per_pop']) - existing_school_underserved)
needed_emerg = max(0, int(underserved_pop / CONFIG['emergency_per_pop']) - existing_emerg_underserved)

print(f"\nExisting in underserved areas:")
print(f"   Hospitals: {existing_hosp_underserved}")
print(f"   Schools: {existing_school_underserved}")
print(f"   Emergency: {existing_emerg_underserved}")

print(f"\nAdditional needed:")
print(f"   Hospitals: {needed_hosp}")
print(f"   Schools: {needed_school}")
print(f"   Emergency: {needed_emerg}")

# ============================================================================
# EQUITY-CONSTRAINED OPTIMIZATION (IMPROVED)
# ============================================================================
print("\n" + "="*70)
print("EQUITY-CONSTRAINED FACILITY PLACEMENT")
print("="*70)

def place_facilities_equity(wards_gdf, underserved_mask, existing_facilities, 
                           n_needed, facility_type, config):
    """
    Place facilities using improved MILP formulation
    """
    if n_needed == 0:
        print(f"  No additional {facility_type} needed")
        return existing_facilities.copy()
    
    print(f"\n  Optimizing {facility_type} placement ({n_needed} facilities)...")
    
    # Get underserved ward indices
    underserved_idx = np.where(underserved_mask)[0]
    
    if len(underserved_idx) == 0:
        print(f"  ⚠️  No underserved wards found")
        return existing_facilities.copy()
    
    # Get ward data
    ward_coords = np.array([[wards_gdf.iloc[i].geometry.centroid.x,
                            wards_gdf.iloc[i].geometry.centroid.y] 
                           for i in underserved_idx])
    ward_pop = np.array([population[i] for i in underserved_idx])
    ward_access = np.array([avg_access[i] for i in underserved_idx])
    
    # Create solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("  ⚠️  Solver not available, using greedy placement")
        return greedy_placement(wards_gdf, underserved_idx, existing_facilities, 
                               n_needed, facility_type)
    
    # Decision variables: place facility in ward i?
    x = {}
    for i, idx in enumerate(underserved_idx):
        x[i] = solver.BoolVar(f'{facility_type}_{idx}')
    
    # Objective: maximize weighted coverage (prioritize high-pop, poor-access wards)
    objective = solver.Objective()
    for i, idx in enumerate(underserved_idx):
        # Weight = population × current access time × equity weight
        weight = ward_pop[i] * ward_access[i] * config['equity_weight']
        objective.SetCoefficient(x[i], weight)
    objective.SetMaximization()
    
    # Constraint 1: Exactly n_needed facilities
    total_constraint = solver.Constraint(n_needed, n_needed)
    for var in x.values():
        total_constraint.SetCoefficient(var, 1)
    
    # Constraint 2: Minimum spacing (soft constraint via penalty)
    # Instead of hard constraint, we'll do post-processing
    
    # Solve
    solver.SetTimeLimit(config['optimization_time_limit'] * 1000)
    status = solver.Solve()
    
    if status not in [0, 1]:  # Not OPTIMAL or FEASIBLE
        print(f"  ⚠️  Optimization failed (status={status}), using greedy")
        return greedy_placement(wards_gdf, underserved_idx, existing_facilities, 
                               n_needed, facility_type)
    
    # Extract solution
    selected_wards = [underserved_idx[i] for i in range(len(underserved_idx)) 
                     if x[i].solution_value() > 0.5]
    
    print(f"  ✓ Placed {len(selected_wards)} facilities")
    
    # Apply spacing constraint post-hoc
    selected_wards = enforce_spacing(wards_gdf, selected_wards, 
                                    config['min_facility_spacing_km'])
    
    # If we removed too many, add back greedily
    while len(selected_wards) < n_needed and len(selected_wards) < len(underserved_idx):
        # Find best remaining ward
        remaining = [i for i in underserved_idx if i not in selected_wards]
        if not remaining:
            break
        
        scores = [(i, ward_pop[np.where(underserved_idx == i)[0][0]] * 
                      ward_access[np.where(underserved_idx == i)[0][0]]) 
                 for i in remaining]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_wards.append(scores[0][0])
        selected_wards = enforce_spacing(wards_gdf, selected_wards, 
                                        config['min_facility_spacing_km'])
    
    # Create new facilities
    new_facilities = []
    for idx in selected_wards:
        centroid = wards_gdf.iloc[idx].geometry.centroid
        new_facilities.append({
            'geometry': centroid,
            'ward_id': idx,
            'type': facility_type,
            'capacity': 100 if facility_type == 'hospital' else 500,
            'equity_placement': True
        })
    
    if new_facilities:
        new_gdf = gpd.GeoDataFrame(new_facilities, crs=4326)
        return pd.concat([existing_facilities, new_gdf], ignore_index=True)
    
    return existing_facilities.copy()

def enforce_spacing(wards_gdf, selected_indices, min_spacing_km):
    """Remove facilities that are too close together"""
    if len(selected_indices) <= 1:
        return selected_indices
    
    coords = np.array([[wards_gdf.iloc[i].geometry.centroid.x,
                       wards_gdf.iloc[i].geometry.centroid.y] 
                      for i in selected_indices])
    
    # Compute pairwise distances
    keep = [True] * len(selected_indices)
    for i in range(len(selected_indices)):
        if not keep[i]:
            continue
        for j in range(i+1, len(selected_indices)):
            if not keep[j]:
                continue
            dist = np.linalg.norm(coords[i] - coords[j]) * 111  # km
            if dist < min_spacing_km:
                # Keep the one with higher population
                pop_i = population[selected_indices[i]]
                pop_j = population[selected_indices[j]]
                if pop_i > pop_j:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    
    return [selected_indices[i] for i in range(len(selected_indices)) if keep[i]]

def greedy_placement(wards_gdf, candidate_indices, existing, n_needed, ftype):
    """Greedy fallback placement"""
    print(f"  Using greedy placement for {n_needed} {ftype}...")
    
    # Sort by population × access time (prioritize worst-served, high-pop)
    scores = [(i, population[i] * avg_access[i]) for i in candidate_indices]
    scores.sort(key=lambda x: x[1], reverse=True)
    
    selected = []
    for idx, _ in scores:
        if len(selected) >= n_needed:
            break
        
        # Check spacing
        ok = True
        for prev_idx in selected:
            c1 = wards_gdf.iloc[idx].geometry.centroid
            c2 = wards_gdf.iloc[prev_idx].geometry.centroid
            dist = np.linalg.norm([c1.x - c2.x, c1.y - c2.y]) * 111
            if dist < CONFIG['min_facility_spacing_km']:
                ok = False
                break
        
        if ok:
            selected.append(idx)
    
    new_facilities = []
    for idx in selected:
        new_facilities.append({
            'geometry': wards_gdf.iloc[idx].geometry.centroid,
            'ward_id': idx,
            'type': ftype,
            'capacity': 100 if ftype == 'hospital' else 500,
            'equity_placement': True
        })
    
    if new_facilities:
        new_gdf = gpd.GeoDataFrame(new_facilities, crs=4326)
        return pd.concat([existing, new_gdf], ignore_index=True)
    
    return existing.copy()

# Optimize each facility type
hospitals_eq = place_facilities_equity(wards, underserved, hospitals, needed_hosp, 
                                       'hospital', CONFIG)
schools_eq = place_facilities_equity(wards, underserved, schools, needed_school, 
                                     'school', CONFIG)
emergency_eq = place_facilities_equity(wards, underserved, emergency, needed_emerg, 
                                       'emergency', CONFIG)

print(f"\n✓ Equity optimization complete")
print(f"   Hospitals: {len(hospitals)} → {len(hospitals_eq)} (+{len(hospitals_eq)-len(hospitals)})")
print(f"   Schools: {len(schools)} → {len(schools_eq)} (+{len(schools_eq)-len(schools)})")
print(f"   Emergency: {len(emergency)} → {len(emergency_eq)} (+{len(emergency_eq)-len(emergency)})")

# ============================================================================
# RECOMPUTE EQUITY METRICS
# ============================================================================
print("\n" + "="*70)
print("IMPROVED EQUITY METRICS")
print("="*70)

hosp_access_new = compute_access(wards, hospitals_eq)
school_access_new = compute_access(wards, schools_eq)
emerg_access_new = compute_access(wards, emergency_eq)

avg_access_new = hosp_access_new * 0.4 + school_access_new * 0.3 + emerg_access_new * 0.3

new_gini = gini_coefficient(avg_access_new)
new_max_min = avg_access_new.max() - avg_access_new.min()
new_low_avg = avg_access_new[low_income].mean()
new_high_avg = avg_access_new[high_income].mean()
new_gap = new_low_avg - new_high_avg
new_palma = palma_ratio(avg_access_new, income)

print(f"\n📊 IMPROVED METRICS:")
print(f"   Gini: {new_gini:.3f} (was {baseline_gini:.3f})")
print(f"   Max-min gap: {new_max_min:.1f} min (was {baseline_max_min:.1f})")
print(f"   Low-income avg: {new_low_avg:.1f} min (was {baseline_low_avg:.1f})")
print(f"   Income gap: {new_gap:.1f} min (was {baseline_gap:.1f})")
print(f"   Palma: {new_palma:.2f} (was {baseline_palma:.2f})")

gini_reduction = ((baseline_gini - new_gini) / baseline_gini) * 100
gap_reduction = ((abs(baseline_gap) - abs(new_gap)) / abs(baseline_gap)) * 100 if baseline_gap != 0 else 0
time_saved = baseline_low_avg - new_low_avg

print(f"\n✅ IMPROVEMENTS:")
print(f"   Gini reduced: {gini_reduction:.1f}%")
print(f"   Income gap reduced: {gap_reduction:.1f}%")
print(f"   Low-income time saved: {time_saved:.1f} min")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

output_dir = Path("results/optimization/equity")
output_dir.mkdir(exist_ok=True, parents=True)

hospitals_eq.to_file(output_dir / "hospitals_equity_optimized.geojson", driver='GeoJSON')
schools_eq.to_file(output_dir / "schools_equity_optimized.geojson", driver='GeoJSON')
emergency_eq.to_file(output_dir / "emergency_equity_optimized.geojson", driver='GeoJSON')

# Ward analysis
wards['underserved'] = underserved
wards['low_income'] = low_income
wards['baseline_access'] = avg_access
wards['improved_access'] = avg_access_new
wards['improvement'] = avg_access - avg_access_new
wards['income'] = income

wards[['geometry', 'underserved', 'low_income', 'baseline_access', 
       'improved_access', 'improvement', 'income']].to_file(
    output_dir / "wards_equity_analysis.geojson", driver='GeoJSON')

# Summary JSON
summary = {
    'baseline': {
        'gini': float(baseline_gini),
        'max_min_gap': float(baseline_max_min),
        'low_income_avg': float(baseline_low_avg),
        'high_income_avg': float(baseline_high_avg),
        'income_gap': float(baseline_gap),
        'palma_ratio': float(baseline_palma)
    },
    'improved': {
        'gini': float(new_gini),
        'max_min_gap': float(new_max_min),
        'low_income_avg': float(new_low_avg),
        'high_income_avg': float(new_high_avg),
        'income_gap': float(new_gap),
        'palma_ratio': float(new_palma)
    },
    'improvements': {
        'gini_reduction_pct': float(gini_reduction),
        'gap_reduction_pct': float(gap_reduction),
        'time_saved_min': float(time_saved)
    },
    'facilities_added': {
        'hospitals': int(len(hospitals_eq) - len(hospitals)),
        'schools': int(len(schools_eq) - len(schools)),
        'emergency': int(len(emergency_eq) - len(emergency))
    },
    'underserved': {
        'count': int(n_underserved),
        'percentage': float(n_underserved/len(wards)*100),
        'population': int(underserved_pop)
    }
}

with open(output_dir / "equity_optimization_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✓ Saved all results to: {output_dir}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

sns.set_style("whitegrid")

# 1. Equity metrics comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Gini
axes[0, 0].bar(['Baseline', 'Improved'], [baseline_gini, new_gini], 
              color=['#e74c3c', '#27ae60'])
axes[0, 0].set_ylabel('Gini Coefficient', fontweight='bold')
axes[0, 0].set_title('Inequality Reduction', fontsize=12, fontweight='bold')
axes[0, 0].set_ylim(0, max(baseline_gini, new_gini) * 1.2)
for i, v in enumerate([baseline_gini, new_gini]):
    axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

# Income gap
axes[0, 1].bar(['Baseline', 'Improved'], [abs(baseline_gap), abs(new_gap)],
              color=['#e74c3c', '#27ae60'])
axes[0, 1].set_ylabel('Access Time Gap (min)', fontweight='bold')
axes[0, 1].set_title('Low-Income vs High-Income Gap', fontsize=12, fontweight='bold')
for i, v in enumerate([abs(baseline_gap), abs(new_gap)]):
    axes[0, 1].text(i, v + 0.05, f'{v:.1f}', ha='center', fontweight='bold')

# Access by income group
groups = ['Bottom\n30%', 'Middle\n40%', 'Top\n30%']
baseline_groups = [
    avg_access[low_income].mean(),
    avg_access[~low_income & ~high_income].mean(),
    avg_access[high_income].mean()
]
improved_groups = [
    avg_access_new[low_income].mean(),
    avg_access_new[~low_income & ~high_income].mean(),
    avg_access_new[high_income].mean()
]

x = np.arange(len(groups))
width = 0.35

axes[1, 0].bar(x - width/2, baseline_groups, width, label='Baseline', color='#e74c3c')
axes[1, 0].bar(x + width/2, improved_groups, width, label='Improved', color='#27ae60')
axes[1, 0].set_ylabel('Avg Access Time (min)', fontweight='bold')
axes[1, 0].set_title('Access Time by Income Group', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(groups)
axes[1, 0].legend()

# Palma ratio
axes[1, 1].bar(['Baseline', 'Improved'], [baseline_palma, new_palma],
              color=['#e74c3c', '#27ae60'])
axes[1, 1].set_ylabel('Palma Ratio', fontweight='bold')
axes[1, 1].set_title('Top 10% vs Bottom 40%', fontsize=12, fontweight='bold')
axes[1, 1].axhline(y=1, color='red', linestyle='--', label='Equality')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(output_dir / "equity_metrics_comparison.png", dpi=300, bbox_inches='tight')
print("✓ Saved: equity_metrics_comparison.png")
plt.close()

# 2. Access time distribution
fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(avg_access, bins=30, alpha=0.6, label='Baseline', color='#e74c3c', edgecolor='black')
ax.hist(avg_access_new, bins=30, alpha=0.6, label='After Equity Optimization', 
        color='#27ae60', edgecolor='black')
ax.axvline(avg_access.mean(), color='red', linestyle='--', linewidth=2, 
          label=f'Baseline Mean: {avg_access.mean():.1f} min')
ax.axvline(avg_access_new.mean(), color='green', linestyle='--', linewidth=2,
          label=f'Improved Mean: {avg_access_new.mean():.1f} min')

ax.set_xlabel('Access Time (minutes)', fontweight='bold')
ax.set_ylabel('Number of Wards', fontweight='bold')
ax.set_title('Distribution of Access Times', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "access_time_distribution.png", dpi=300, bbox_inches='tight')
print("✓ Saved: access_time_distribution.png")
plt.close()

print("\n" + "="*70)
print("✅ PHASE 7.3 COMPLETE")
