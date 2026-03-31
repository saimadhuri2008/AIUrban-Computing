#!/usr/bin/env python3
"""
Phase 7.2 — Network Optimization (OPTIMIZED VERSION)
Key optimizations:
1. Precompute all shortest paths once
2. Use betweenness centrality for road importance
3. Simplified benefit calculation
4. Faster MILP formulation
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from ortools.linear_solver import pywraplp
import json
from pathlib import Path
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("PHASE 7.2 — NETWORK OPTIMIZATION (OPTIMIZED)")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\nLoading Phase 6 & 7.1 data...")

sectors = gpd.read_file("data/processed/wards/bbmp_5sectors.geojson").to_crs(4326)
wards_raw = gpd.read_file("data/processed/wards/wards_2035_All_variables.geojson").to_crs(4326)

# Remove duplicate wards
print(f"  Raw wards loaded: {len(wards_raw)}")
if 'ward_id' in wards_raw.columns:
    wards = wards_raw.drop_duplicates(subset='ward_id', keep='first').reset_index(drop=True)
elif 'Ward_No' in wards_raw.columns:
    wards = wards_raw.drop_duplicates(subset='Ward_No', keep='first').reset_index(drop=True)
else:
    wards = wards_raw.drop_duplicates(subset=['geometry'], keep='first').reset_index(drop=True)

print(f"  Unique wards after deduplication: {len(wards)}")

# Transport networks
roads = gpd.read_file("results/redesign/transport_roads.geojson").to_crs(4326)
metro = gpd.read_file("results/redesign/metro_network.geojson").to_crs(4326)

# Optimized facilities from Phase 7.1
hospitals_opt = gpd.read_file("results/optimization/facilities/hospital_optimized.geojson").to_crs(4326)
schools_opt = gpd.read_file("results/optimization/facilities/school_optimized.geojson").to_crs(4326)
emergency_opt = gpd.read_file("results/optimization/facilities/emergency_optimized.geojson").to_crs(4326)

print(f"✓ Loaded {len(wards)} unique wards")
print(f"✓ Loaded {len(roads)} road segments")
print(f"✓ Loaded {len(metro)} metro lines")

# Extract population
population = wards['population_2035'].values if 'population_2035' in wards.columns else wards['population'].values
total_pop = population.sum()
print(f"✓ Total population: {total_pop:,.0f}")

# ============================================================================
# ENSURE NETWORK CONNECTIVITY
# ============================================================================
print("\n" + "="*70)
print("ENSURING NETWORK CONNECTIVITY")
print("="*70)

def ensure_network_connectivity(roads_gdf, wards_gdf, min_connections_per_ward=3):
    """Add synthetic roads to ensure connectivity"""
    print(f"\n  Input roads: {len(roads_gdf)}")
    
    ward_coords = np.array([[w.geometry.centroid.x, w.geometry.centroid.y] 
                            for _, w in wards_gdf.iterrows()])
    ward_tree = cKDTree(ward_coords)
    
    synthetic_roads = []
    
    for i, ward in wards_gdf.iterrows():
        centroid = ward.geometry.centroid
        k = min_connections_per_ward + 1
        dists, indices = ward_tree.query([centroid.x, centroid.y], k=k)
        
        for dist, j in zip(dists[1:], indices[1:]):
            if dist > 0 and i < j:
                neighbor = wards_gdf.iloc[j]
                neighbor_centroid = neighbor.geometry.centroid
                
                line = LineString([(centroid.x, centroid.y), 
                                  (neighbor_centroid.x, neighbor_centroid.y)])
                
                dist_km = dist * 111
                
                synthetic_roads.append({
                    'geometry': line,
                    'length_km': dist_km,
                    'lanes': 2,
                    'road_type': 'collector',
                    'source': 'synthetic',
                    'speed': 35
                })
    
    if len(synthetic_roads) > 0:
        synthetic_gdf = gpd.GeoDataFrame(synthetic_roads, crs=wards_gdf.crs)
        combined = pd.concat([roads_gdf, synthetic_gdf], ignore_index=True)
    else:
        combined = roads_gdf.copy()
    
    print(f"  Added {len(synthetic_roads)} synthetic roads")
    print(f"  Total roads: {len(combined)}")
    
    return combined

roads = ensure_network_connectivity(roads, wards, min_connections_per_ward=3)

# ============================================================================
# BUILD NETWORK GRAPH
# ============================================================================
print("\n" + "="*70)
print("BUILDING TRANSPORT NETWORK GRAPH")
print("="*70)

def build_transport_graph(roads_gdf, metro_gdf, wards_gdf):
    """Build unified transport graph"""
    G = nx.Graph()
    
    # Add ward centroids as nodes
    for idx, ward in wards_gdf.iterrows():
        centroid = ward.geometry.centroid
        G.add_node(idx, 
                  pos=(centroid.x, centroid.y),
                  population=ward.get('population_2035', ward.get('population', 0)),
                  sector=ward.get('sector', 'Unknown'))
    
    # Add road edges
    ward_coords = np.array([[w.geometry.centroid.x, w.geometry.centroid.y] 
                           for _, w in wards_gdf.iterrows()])
    ward_tree = cKDTree(ward_coords)
    
    road_edges = []
    
    for idx, road in roads_gdf.iterrows():
        try:
            coords = list(road.geometry.coords)
            start_pt = coords[0]
            end_pt = coords[-1]
            
            _, start_idx = ward_tree.query(start_pt, k=1)
            _, end_idx = ward_tree.query(end_pt, k=1)
            
            if start_idx != end_idx:
                length_km = road.get('length_km', 
                                    np.linalg.norm(np.array(start_pt) - np.array(end_pt)) * 111)
                
                if length_km <= 0 or not np.isfinite(length_km):
                    continue
                
                lanes = road.get('lanes', 2)
                road_type = road.get('road_type', 'collector')
                
                speed_map = {'arterial': 50, 'collector': 35, 'local': 25, 'highway': 80}
                speed = road.get('speed', speed_map.get(road_type, 35))
                
                travel_time = (length_km / speed) * 60  # minutes
                
                if travel_time > 0 and np.isfinite(travel_time):
                    road_edges.append({
                        'u': start_idx,
                        'v': end_idx,
                        'road_id': idx,
                        'length': length_km,
                        'lanes': lanes,
                        'speed': speed,
                        'time': travel_time,
                        'road_type': road_type,
                        'mode': 'road',
                        'upgradeable': lanes < 8
                    })
        except Exception as e:
            continue
    
    # Add metro edges
    for idx, line in metro_gdf.iterrows():
        try:
            coords = list(line.geometry.coords)
            
            for i in range(len(coords) - 1):
                _, start_idx = ward_tree.query(coords[i], k=1)
                _, end_idx = ward_tree.query(coords[i+1], k=1)
                
                if start_idx != end_idx:
                    dx = coords[i+1][0] - coords[i][0]
                    dy = coords[i+1][1] - coords[i][1]
                    dist = np.sqrt(dx**2 + dy**2) * 111
                    
                    if dist > 0 and np.isfinite(dist):
                        travel_time = (dist / 40) * 60  # 40 km/h avg metro speed
                        
                        if travel_time > 0 and np.isfinite(travel_time):
                            G.add_edge(start_idx, end_idx,
                                      line_id=idx,
                                      length=dist,
                                      time=travel_time,
                                      mode='metro')
        except Exception as e:
            continue
    
    # Add road edges to graph
    for edge in road_edges:
        G.add_edge(edge['u'], edge['v'], **{k: v for k, v in edge.items() if k not in ['u', 'v']})
    
    return G, road_edges

graph, road_edges_list = build_transport_graph(roads, metro, wards)

print(f"✓ Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

# Ensure connectivity
largest_cc = max(nx.connected_components(graph), key=len)
connectivity_pct = len(largest_cc) / len(wards) * 100
print(f"✓ Network connectivity: {len(largest_cc)} nodes ({connectivity_pct:.1f}%)")

# ============================================================================
# OPTIMIZED: PRECOMPUTE ALL SHORTEST PATHS
# ============================================================================
print("\n" + "="*70)
print("PRECOMPUTING SHORTEST PATHS (ONE TIME)")
print("="*70)

# Only compute for largest connected component
connected_nodes = list(largest_cc)
print(f"  Computing paths for {len(connected_nodes)} nodes...")

# Store all shortest paths
all_paths = {}
travel_times = np.full((len(wards), len(wards)), np.inf)

for source in tqdm(connected_nodes, desc="  Progress"):
    paths = nx.single_source_dijkstra_path(graph, source, weight='time')
    lengths = nx.single_source_dijkstra_path_length(graph, source, weight='time')
    
    for target in paths:
        all_paths[(source, target)] = paths[target]
        travel_times[source, target] = lengths[target]

np.fill_diagonal(travel_times, 0)

baseline_avg_time = np.mean(travel_times[np.isfinite(travel_times)])
baseline_max_time = np.max(travel_times[np.isfinite(travel_times)])

print(f"✓ Paths precomputed")
print(f"   Average travel time: {baseline_avg_time:.1f} minutes")
print(f"   Maximum travel time: {baseline_max_time:.1f} minutes")

# ============================================================================
# OPTIMIZED: COMPUTE OD DEMAND (SIMPLIFIED)
# ============================================================================
print("\n" + "="*70)
print("COMPUTING OD DEMAND (SIMPLIFIED)")
print("="*70)

# Simplified gravity model
od_demand = np.zeros((len(wards), len(wards)))

for i in range(len(wards)):
    for j in range(len(wards)):
        if i != j and np.isfinite(travel_times[i, j]):
            # Gravity model: demand proportional to populations, inversely proportional to time
            demand = (population[i] * population[j]) / (travel_times[i, j] + 1)
            od_demand[i, j] = demand * 0.0001  # Scale factor

total_trips = od_demand.sum()
print(f"✓ Total daily trips: {total_trips:,.0f}")

# ============================================================================
# OPTIMIZED: EDGE BETWEENNESS FOR ROAD IMPORTANCE
# ============================================================================
print("\n" + "="*70)
print("COMPUTING EDGE IMPORTANCE (BETWEENNESS)")
print("="*70)

print("  Computing edge betweenness centrality...")
edge_betweenness = nx.edge_betweenness_centrality(graph, weight='time', normalized=True)

# Map betweenness to road edges
road_importance = {}
for edge in road_edges_list:
    u, v = edge['u'], edge['v']
    
    # Get betweenness (handle both edge directions)
    bc = edge_betweenness.get((u, v), edge_betweenness.get((v, u), 0))
    
    road_importance[(u, v)] = bc

print(f"✓ Edge importance computed")

# ============================================================================
# OPTIMIZED: MARGINAL BENEFIT ESTIMATION (FAST)
# ============================================================================
print("\n" + "="*70)
print("PHASE 7.2.a — MARGINAL BENEFIT (OPTIMIZED)")
print("="*70)

print("\nEstimating road upgrade benefits...")
road_upgrades_catalog = []

for edge in tqdm(road_edges_list, desc="  Roads"):
    if edge['upgradeable']:
        u, v = edge['u'], edge['v']
        
        # Benefit = edge importance × OD demand using this edge × time savings
        importance = road_importance.get((u, v), 0)
        
        # Estimate demand using this edge (sum of OD pairs in nearby wards)
        edge_demand = 0
        for i in range(max(0, u-5), min(len(wards), u+6)):
            for j in range(max(0, v-5), min(len(wards), v+6)):
                edge_demand += od_demand[i, j]
        
        # Time savings from upgrade (20% reduction)
        time_saved = edge['time'] * 0.2
        
        # Benefit = importance × demand × time savings
        benefit = importance * edge_demand * time_saved * 1000  # Scale up
        
        length = edge['length']
        lanes = edge['lanes']
        
        # Cost: $5M per lane-km (2 lanes added)
        cost = 5_000_000 * length * 2
        
        # Benefit-cost ratio
        bc_ratio = benefit / cost if cost > 0 else 0
        
        road_upgrades_catalog.append({
            'type': 'road',
            'from_ward': u,
            'to_ward': v,
            'road_id': edge['road_id'],
            'current_lanes': lanes,
            'upgraded_lanes': lanes + 2,
            'length_km': length,
            'road_type': edge['road_type'],
            'importance': importance,
            'benefit': benefit,
            'cost': cost,
            'bc_ratio': bc_ratio
        })

print(f"✓ Evaluated {len(road_upgrades_catalog)} road upgrades")

print("\nEstimating metro extension benefits...")
metro_extensions_catalog = []

# Only consider high-demand corridors (top 20% by OD demand)
od_flat = []
for i in range(len(wards)):
    for j in range(i+1, len(wards)):
        if np.isfinite(travel_times[i, j]):
            od_flat.append((i, j, od_demand[i, j] + od_demand[j, i]))

od_flat.sort(key=lambda x: x[2], reverse=True)
top_20pct = od_flat[:int(len(od_flat) * 0.2)]

for i, j, demand in tqdm(top_20pct[:200], desc="  Metro"):  # Limit to top 200
    # Distance between wards
    dist_km = np.linalg.norm([
        wards.iloc[i].geometry.centroid.x - wards.iloc[j].geometry.centroid.x,
        wards.iloc[i].geometry.centroid.y - wards.iloc[j].geometry.centroid.y
    ]) * 111
    
    # Only consider 1-15 km extensions
    if 1 < dist_km < 15:
        # Metro travel time
        metro_time = (dist_km / 40) * 60
        
        # Current time
        current_time = travel_times[i, j]
        
        time_saved = max(0, current_time - metro_time)
        
        # Direct demand
        direct_demand = od_demand[i, j] + od_demand[j, i]
        
        # Metro attracts 20% more ridership
        metro_demand = direct_demand * 1.2
        
        benefit = metro_demand * time_saved * 1000  # Scale up
        
        # Cost: $150M per km
        cost = 150_000_000 * dist_km
        
        bc_ratio = benefit / cost if cost > 0 else 0
        
        metro_extensions_catalog.append({
            'type': 'metro',
            'from_ward': i,
            'to_ward': j,
            'length_km': dist_km,
            'benefit': benefit,
            'cost': cost,
            'bc_ratio': bc_ratio,
            'population_served': population[i] + population[j]
        })

# Keep top 100 by B/C ratio
metro_extensions_catalog = sorted(metro_extensions_catalog, 
                                  key=lambda x: x['bc_ratio'], reverse=True)[:100]

print(f"✓ Evaluated {len(metro_extensions_catalog)} metro extensions")

# ============================================================================
# OPTIMIZED: BUDGET-CONSTRAINED SELECTION (SIMPLIFIED MILP)
# ============================================================================
print("\n" + "="*70)
print("PHASE 7.2.b — BUDGET-CONSTRAINED SELECTION (OPTIMIZED)")
print("="*70)

# Pre-filter: only consider projects with positive B/C ratio
all_projects = [p for p in road_upgrades_catalog + metro_extensions_catalog 
                if p['bc_ratio'] > 0]

print(f"\nTotal projects evaluated: {len(all_projects)}")
print(f"  Road upgrades: {sum(1 for p in all_projects if p['type'] == 'road')}")
print(f"  Metro extensions: {sum(1 for p in all_projects if p['type'] == 'metro')}")

# Create MILP model
solver = pywraplp.Solver.CreateSolver('SCIP')
if not solver:
    print("❌ Solver not available - using greedy selection")
    # Fallback: greedy selection
    all_projects.sort(key=lambda x: x['bc_ratio'], reverse=True)
    selected_projects = []
    total_cost = 0
    TOTAL_BUDGET = 100_000_000_000
    
    for project in all_projects:
        if total_cost + project['cost'] <= TOTAL_BUDGET:
            selected_projects.append(project)
            total_cost += project['cost']
            
            if len(selected_projects) >= 100:  # Limit selections
                break
    
    print(f"✓ Greedy selection: {len(selected_projects)} projects")
else:
    print("\nCreating MILP model...")
    
    # Decision variables
    project_vars = []
    for i, project in enumerate(all_projects):
        var = solver.BoolVar(f'project_{i}')
        project_vars.append(var)
    
    print(f"✓ Variables: {len(project_vars)}")
    
    # Objective: Maximize total benefit
    objective = solver.Objective()
    for i, project in enumerate(all_projects):
        objective.SetCoefficient(project_vars[i], project['benefit'])
    objective.SetMaximization()
    
    # Constraint 1: Budget
    TOTAL_BUDGET = 100_000_000_000  # $100 billion
    budget_constraint = solver.Constraint(0, TOTAL_BUDGET)
    
    for i, project in enumerate(all_projects):
        budget_constraint.SetCoefficient(project_vars[i], project['cost'])
    
    print(f"✓ Budget constraint: ${TOTAL_BUDGET/1e9:.0f}B")
    
    # Constraint 2: Minimum metro coverage (at least 3 extensions)
    metro_constraint = solver.Constraint(3, solver.infinity())
    for i, project in enumerate(all_projects):
        if project['type'] == 'metro':
            metro_constraint.SetCoefficient(project_vars[i], 1)
    
    print(f"✓ Constraints: 2")
    
    # Solve with timeout
    print("\nSolving MILP...")
    solver.SetTimeLimit(60000)  # 1 minute timeout
    status = solver.Solve()
    
    status_map = {0: 'OPTIMAL', 1: 'FEASIBLE', 2: 'INFEASIBLE', 3: 'UNBOUNDED'}
    print(f"\n✓ Solver status: {status_map.get(status, 'UNKNOWN')}")
    print(f"✓ Solve time: {solver.WallTime()/1000:.1f}s")
    
    if status in [0, 1]:
        print(f"✓ Objective value: {solver.Objective().Value():,.0f}")
        
        # Extract selected projects
        selected_projects = []
        total_cost = 0
        
        for i, var in enumerate(project_vars):
            if var.solution_value() > 0.5:
                project = all_projects[i]
                selected_projects.append(project)
                total_cost += project['cost']
    else:
        # Fallback: greedy
        print("⚠️  MILP infeasible, using greedy fallback")
        all_projects.sort(key=lambda x: x['bc_ratio'], reverse=True)
        selected_projects = []
        total_cost = 0
        
        for project in all_projects:
            if total_cost + project['cost'] <= TOTAL_BUDGET:
                selected_projects.append(project)
                total_cost += project['cost']

# ============================================================================
# EXTRACT RESULTS
# ============================================================================
print("\n" + "="*70)
print("EXTRACTING SELECTED PROJECTS")
print("="*70)

total_benefit = sum(p['benefit'] for p in selected_projects)
selected_roads = [p for p in selected_projects if p['type'] == 'road']
selected_metro = [p for p in selected_projects if p['type'] == 'metro']

print(f"\n✅ Selected {len(selected_projects)} projects:")
print(f"   Road upgrades: {len(selected_roads)}")
print(f"   Metro extensions: {len(selected_metro)}")
print(f"\n💰 Total investment: ${total_cost/1e9:.2f}B")
print(f"📈 Total benefit score: {total_benefit:,.0f}")

if total_cost > 0:
    print(f"📊 Average B/C ratio: {total_benefit/total_cost:.2f}")

# ============================================================================
# ESTIMATE IMPROVEMENTS
# ============================================================================
print("\n" + "="*70)
print("ESTIMATING NETWORK IMPROVEMENTS")
print("="*70)

# Simplified estimation (avoid recomputing all paths)
# Assume 20% time reduction on upgraded roads
# Assume 30% time reduction on new metro corridors

total_time_savings = 0

for project in selected_roads:
    u, v = project['from_ward'], project['to_ward']
    importance = project.get('importance', 0.01)
    
    # Estimate trips affected
    affected_trips = total_trips * importance
    time_saved_per_trip = project['cost'] / 5_000_000 * 0.2 * 5  # rough estimate
    
    total_time_savings += affected_trips * time_saved_per_trip

for project in selected_metro:
    # More significant time savings for metro
    pop_served = project['population_served']
    trips = pop_served * 0.3  # 30% use metro
    time_saved = 15  # avg 15 min saved per trip
    
    total_time_savings += trips * time_saved

avg_improvement_pct = (total_time_savings / (total_trips * baseline_avg_time)) * 100

print(f"\n📊 Estimated Improvements:")
print(f"   Baseline avg time: {baseline_avg_time:.1f} min")
print(f"   Estimated reduction: ~{avg_improvement_pct:.1f}%")
print(f"   Improved avg time: ~{baseline_avg_time * (1 - avg_improvement_pct/100):.1f} min")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

output_dir = Path("results/network")
output_dir.mkdir(exist_ok=True, parents=True)

# Save GeoJSON files
if selected_roads:
    upgrade_features = []
    for project in selected_roads:
        try:
            road_id = project['road_id']
            road_geom = roads.iloc[road_id].geometry
            
            upgrade_features.append({
                'type': 'Feature',
                'geometry': road_geom.__geo_interface__,
                'properties': {k: v for k, v in project.items() if k not in ['type']}
            })
        except:
            pass
    
    if upgrade_features:
        upgrade_gdf = gpd.GeoDataFrame.from_features(upgrade_features, crs=4326)
        upgrade_gdf.to_file(output_dir / "roads_optimized_upgrades.geojson", driver='GeoJSON')
        print(f"✓ Saved: roads_optimized_upgrades.geojson")

if selected_metro:
    metro_features = []
    for project in selected_metro:
        try:
            from_pt = wards.iloc[project['from_ward']].geometry.centroid
            to_pt = wards.iloc[project['to_ward']].geometry.centroid
            
            line = LineString([(from_pt.x, from_pt.y), (to_pt.x, to_pt.y)])
            
            metro_features.append({
                'type': 'Feature',
                'geometry': line.__geo_interface__,
                'properties': {k: v for k, v in project.items() if k not in ['type']}
            })
        except:
            pass
    
    if metro_features:
        metro_ext_gdf = gpd.GeoDataFrame.from_features(metro_features, crs=4326)
        metro_ext_gdf.to_file(output_dir / "metro_optimized_extensions.geojson", driver='GeoJSON')
        print(f"✓ Saved: metro_optimized_extensions.geojson")

# Save summary
summary = {
    'methodology': 'Optimized Marginal Benefit Analysis',
    'total_projects': len(selected_projects),
    'road_upgrades': len(selected_roads),
    'metro_extensions': len(selected_metro),
    'total_investment_usd': float(total_cost),
    'total_benefit_score': float(total_benefit),
    'average_bc_ratio': float(total_benefit / total_cost if total_cost > 0 else 0),
    'baseline_avg_travel_time_min': float(baseline_avg_time),
    'estimated_improvement_pct': float(avg_improvement_pct),
    'optimization_method': 'MILP with betweenness centrality'
}

with open(output_dir / "network_optimization_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Saved: network_optimization_summary.json")

# Save CSV details
pd.DataFrame(selected_roads).to_csv(output_dir / "road_upgrades_detail.csv", index=False)
pd.DataFrame(selected_metro).to_csv(output_dir / "metro_extensions_detail.csv", index=False)
print(f"✓ Saved: CSV detail files")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# 1. Project selection summary
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Road\nUpgrades', 'Metro\nExtensions']
counts = [len(selected_roads), len(selected_metro)]
costs = [sum(p['cost'] for p in selected_roads)/1e9, 
         sum(p['cost'] for p in selected_metro)/1e9]

x = np.arange(len(categories))
width = 0.35

ax.bar(x - width/2, counts, width, label='Number of Projects', color='steelblue')
ax2 = ax.twinx()
ax2.bar(x + width/2, costs, width, label='Investment ($B)', color='green')

ax.set_xlabel('Project Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Projects', fontsize=11)
ax2.set_ylabel('Investment ($ Billion)', fontsize=11)
ax.set_title('Selected Projects Summary', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "selected_projects_summary.png", dpi=300, bbox_inches='tight')
print("✓ Saved: selected_projects_summary.png")
plt.close()

print("\n" + "="*70)
print("✅ PHASE 7.2 COMPLETE (OPTIMIZED)")
print("="*70)
print(f"\n📁 All outputs saved to: {output_dir.absolute()}")
print(f"\n📊 Key Results:")
print(f"   • {len(selected_projects)} projects selected")
print(f"   • ${total_cost/1e9:.1f}B investment")
print(f"   • ~{avg_improvement_pct:.1f}% estimated improvement")
print(f"   • B/C ratio: {total_benefit/total_cost:.2f}")
print(f"\n⏱️  Total runtime significantly reduced")
print(f"▶ Next: Phase 7.3 (Equity Optimization)")