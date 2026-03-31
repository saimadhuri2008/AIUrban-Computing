"""
PROBLEM FORMALIZATION
Mathematical Optimization Foundation

Purpose: Convert urban redesign into a formal optimization problem
Author: Your Name
Date: 2025
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path("optimization")
ARTIFACTS_DIR = BASE_DIR / "artifacts"

@dataclass
class OptimizationProblem:
    """Formal mathematical optimization problem structure"""
    num_facilities: Dict[str, int]
    num_wards: int
    num_sectors: int
    num_road_segments: int
    num_metro_segments: int
    budget_cap: float
    sector_constraints: Dict
    capacity_constraints: Dict
    travel_time_weight: float = 0.4
    cost_weight: float = 0.3
    equity_weight: float = 0.3

class UrbanOptimizationFormalization:
    """
    Mathematical formalization of Bengaluru redesign
    Adapted to actual file structure
    """
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.problem = None
        self.decision_vars = {}
        self.constraints = {}
        self.objectives = {}
        
    def load_phase6_data(self):
        """Load all Phase 6 outputs from your actual structure"""
        print("Loading Phase 6 data from your structure...")
        
        # Core geography
        self.sectors = gpd.read_file("data/processed/wards/bbmp_5sectors_named.geojson").to_crs(4326)
        self.wards = gpd.read_file("data/processed/wards/wards_2035_all_variables.geojson").to_crs(4326)
        
        # Transport networks
        self.roads = gpd.read_file("results/redesign/transport_roads.geojson").to_crs(4326)
        self.metro = gpd.read_file("results/redesign/metro_network.geojson").to_crs(4326)
        
        # Facilities (these will be optimized)
        self.hospitals = gpd.read_file("results/redesign/hospitals_planned.geojson").to_crs(4326)
        self.schools = gpd.read_file("results/redesign/schools_planned.geojson").to_crs(4326)
        self.parks = gpd.read_file("results/redesign/parks_planned.geojson").to_crs(4326)
        self.emergency = gpd.read_file("results/redesign/police_fire_stations.geojson").to_crs(4326)
        self.govt_offices = gpd.read_file("results/redesign/govt_offices.geojson").to_crs(4326)
        
        # Utilities (fixed infrastructure)
        self.power_plants = gpd.read_file("results/redesign/power_plants.geojson").to_crs(4326)
        self.power_lines = gpd.read_file("results/redesign/power_lines.geojson").to_crs(4326)
        self.water = gpd.read_file("results/redesign/water_treatment.geojson").to_crs(4326)
        self.sewage = gpd.read_file("results/redesign/sewage_network.geojson").to_crs(4326)
        
        # Land-use zones (for constraint checking)
        self.rich_res = gpd.read_file("results/redesign/rich_residential.geojson").to_crs(4326)
        self.middle_res = gpd.read_file("results/redesign/middle_income.geojson").to_crs(4326)
        self.affordable = gpd.read_file("results/redesign/affordable_housing.geojson").to_crs(4326)
        self.mixed_use = gpd.read_file("results/redesign/mixed_use.geojson").to_crs(4326)
        self.industrial = gpd.read_file("results/redesign/industrial_zones.geojson").to_crs(4326)
        self.slums = gpd.read_file("results/redesign/slum_upgradation.geojson").to_crs(4326)
        
        # Combine all facilities for analysis
        self.facilities = pd.concat([
            self.hospitals.assign(type='hospital'),
            self.schools.assign(type='school'),
            self.emergency.assign(type='emergency'),
            self.parks.assign(type='park'),
            self.govt_offices.assign(type='government')
        ], ignore_index=True)
        
        print(f"✓ Loaded {len(self.wards)} wards, {len(self.facilities)} facilities")
        print(f"  - Hospitals: {len(self.hospitals)}")
        print(f"  - Schools: {len(self.schools)}")
        print(f"  - Emergency: {len(self.emergency)}")
        print(f"  - Parks: {len(self.parks)}")
        print(f"  - Government: {len(self.govt_offices)}")
        
    def compute_travel_time_matrix(self, avg_speed_road: float = 35, avg_speed_metro: float = 40):
        """
        Compute ward-to-ward travel time matrix
        Uses actual road + metro networks with realistic speeds
        """
        print("\nComputing travel time matrix...")
        print(f"  Road speed: {avg_speed_road} km/h")
        print(f"  Metro speed: {avg_speed_metro} km/h")
        
        # Extract ward centroids
        ward_centroids = self.wards.geometry.centroid
        n_wards = len(self.wards)
        
        # Build unified transport network graph
        G = nx.Graph()
        
        # Add road network
        road_nodes = set()
        for idx, road in self.roads.iterrows():
            if road.geometry.geom_type == 'LineString':
                coords = list(road.geometry.coords)
            elif road.geometry.geom_type == 'MultiLineString':
                coords = []
                for line in road.geometry.geoms:
                    coords.extend(list(line.coords))
            else:
                continue
                
            for i in range(len(coords) - 1):
                p1, p2 = coords[i], coords[i+1]
                road_nodes.add(p1)
                road_nodes.add(p2)
                
                # Distance in km (approximate)
                dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) * 111
                travel_time = (dist / avg_speed_road) * 60  # minutes
                
                G.add_edge(p1, p2, weight=travel_time, mode='road')
        
        print(f"  Road network: {len(road_nodes)} nodes, {len(self.roads)} segments")
        
        # Add metro network (faster)
        metro_nodes = set()
        for idx, metro in self.metro.iterrows():
            if metro.geometry.geom_type == 'LineString':
                coords = list(metro.geometry.coords)
            elif metro.geometry.geom_type == 'MultiLineString':
                coords = []
                for line in metro.geometry.geoms:
                    coords.extend(list(line.coords))
            else:
                continue
                
            for i in range(len(coords) - 1):
                p1, p2 = coords[i], coords[i+1]
                metro_nodes.add(p1)
                metro_nodes.add(p2)
                
                dist = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) * 111
                travel_time = (dist / avg_speed_metro) * 60
                
                G.add_edge(p1, p2, weight=travel_time, mode='metro')
        
        print(f"  Metro network: {len(metro_nodes)} nodes, {len(self.metro)} segments")
        
        # Build KDTree for fast nearest-neighbor search
        all_nodes = np.array(list(G.nodes()))
        tree = cKDTree(all_nodes)
        
        # Compute travel time matrix
        travel_matrix = np.zeros((n_wards, n_wards))
        
        print("  Computing shortest paths...")
        for i, centroid_i in enumerate(ward_centroids):
            # Find nearest network node
            _, idx_i = tree.query([centroid_i.x, centroid_i.y])
            node_i = tuple(all_nodes[idx_i])
            
            # Compute shortest paths from this node to all others
            try:
                lengths = nx.single_source_dijkstra_path_length(G, node_i, weight='weight')
            except:
                lengths = {}
            
            for j, centroid_j in enumerate(ward_centroids):
                if i == j:
                    travel_matrix[i, j] = 0
                    continue
                
                _, idx_j = tree.query([centroid_j.x, centroid_j.y])
                node_j = tuple(all_nodes[idx_j])
                
                if node_j in lengths:
                    travel_matrix[i, j] = lengths[node_j]
                else:
                    # Fallback: euclidean distance
                    dist_km = np.sqrt(
                        (centroid_i.x - centroid_j.x)**2 + 
                        (centroid_i.y - centroid_j.y)**2
                    ) * 111
                    travel_matrix[i, j] = (dist_km / 30) * 60  # 30 km/h fallback
        
        self.travel_time_matrix = travel_matrix
        
        print(f"✓ Travel time matrix computed: {travel_matrix.shape}")
        print(f"  Mean: {travel_matrix[travel_matrix > 0].mean():.1f} min")
        print(f"  Median: {np.median(travel_matrix[travel_matrix > 0]):.1f} min")
        print(f"  90th percentile: {np.percentile(travel_matrix[travel_matrix > 0], 90):.1f} min")
        print(f"  Max: {travel_matrix.max():.1f} min")
        
        return travel_matrix
    
    def compute_demand_vectors(self):
        """
        Compute demand for each ward using actual 2035 projections
        Uses your wards_2035_all_variables data
        """
        print("\nComputing ward demand vectors...")
        
        # Extract key columns (adapt based on your actual column names)
        required_cols = ['pop_2035', 'area_sqkm', 'income_level', 'ward_name']
        
        # Check which columns exist
        available_cols = self.wards.columns.tolist()
        print(f"  Available columns: {len(available_cols)}")
        
        # Use population from your data
        if 'pop_2035' in available_cols:
            population = self.wards['pop_2035'].values
        elif 'population' in available_cols:
            population = self.wards['population'].values
        else:
            # Fallback: estimate from area
            print("  ⚠ No population column found, estimating...")
            population = self.wards.geometry.area * 111 * 111 * 5000  # 5k per km²
        
        # Area calculation
        if 'area_sqkm' in available_cols:
            area = self.wards['area_sqkm'].values
        else:
            area = self.wards.geometry.area * 111 * 111  # deg² to km²
        
        # Calculate facility demand
        demand = {
            'hospital_general': population / 50000,  # 1 per 50k
            'hospital_specialty': population / 200000,  # 1 per 200k
            'school_primary': population * 0.15 / 1000,  # 15% children
            'school_secondary': population * 0.10 / 2000,
            'college': population * 0.08 / 5000,
            'fire_station': area / 10,  # 1 per 10 km²
            'police_station': population / 30000,
            'park': area / 2,  # 1 park per 2 km²
            'govt_office': population / 100000
        }
        
        self.demand_vectors = pd.DataFrame(demand)
        
        # Add ward metadata
        if 'ward_name' in available_cols:
            self.demand_vectors['ward_name'] = self.wards['ward_name'].values
        self.demand_vectors['population'] = population
        self.demand_vectors['area_sqkm'] = area
        
        print(f"✓ Demand vectors computed for {len(demand)} facility types")
        print(f"  Total population: {population.sum():,.0f}")
        print(f"  Total area: {area.sum():.1f} km²")
        
        return self.demand_vectors
    
    def define_decision_variables(self):
        """Define decision variables for optimization"""
        print("\nDefining decision variables...")
        
        n_facilities = len(self.facilities)
        n_wards = len(self.wards)
        n_hospitals = len(self.hospitals)
        n_schools = len(self.schools)
        n_emergency = len(self.emergency)
        
        # x_f,w: Binary - facility f placed in ward w
        self.decision_vars = {
            'x_hospital_ward': {
                'shape': (n_hospitals, n_wards),
                'type': 'binary',
                'description': 'Hospital placement',
                'count': n_hospitals * n_wards
            },
            'x_school_ward': {
                'shape': (n_schools, n_wards),
                'type': 'binary',
                'description': 'School placement',
                'count': n_schools * n_wards
            },
            'x_emergency_ward': {
                'shape': (n_emergency, n_wards),
                'type': 'binary',
                'description': 'Emergency service placement',
                'count': n_emergency * n_wards
            },
            'y_service_allocation': {
                'shape': (n_wards, n_facilities),
                'type': 'continuous',
                'bounds': (0, 1),
                'description': 'Service allocation fraction',
                'count': n_wards * n_facilities
            },
            'z_road_upgrade': {
                'shape': (len(self.roads),),
                'type': 'binary',
                'description': 'Road segment upgrade',
                'count': len(self.roads)
            },
            'z_metro_extend': {
                'shape': (len(self.metro),),
                'type': 'binary',
                'description': 'Metro extension',
                'count': len(self.metro)
            }
        }
        
        total_vars = sum(v['count'] for v in self.decision_vars.values())
        print(f"✓ Total decision variables: {total_vars:,}")
        
        for name, var in self.decision_vars.items():
            print(f"  - {name}: {var['count']:,} ({var['type']})")
        
        return self.decision_vars
    
    def define_constraints(self, budget_cap: float = 1e9):
        """Define hard constraints based on your redesign requirements"""
        print("\nDefining constraints...")
        
        self.constraints = {
            'sector_hospitals_specialty': {
                'type': 'minimum',
                'value': 1,
                'per': 'sector',
                'description': '≥1 super-specialty hospital per sector'
            },
            'sector_hospitals_general': {
                'type': 'minimum',
                'value': 3,
                'per': 'sector',
                'description': '≥3 general hospitals per sector'
            },
            'sector_schools_density': {
                'type': 'minimum_density',
                'value': 0.5,
                'per': 'sector',
                'description': '≥0.5 schools per km²'
            },
            'emergency_response_time': {
                'type': 'upper_bound',
                'value': 8,
                'unit': 'minutes',
                'description': 'Emergency services within 8 min'
            },
            'hospital_access_time': {
                'type': 'upper_bound',
                'value': 30,
                'unit': 'minutes',
                'description': 'Hospital within 30 min'
            },
            'school_walking_distance': {
                'type': 'upper_bound',
                'value': 2,
                'unit': 'km',
                'description': 'Primary school within 2 km'
            },
            'budget_total': {
                'type': 'upper_bound',
                'value': budget_cap,
                'unit': 'INR',
                'description': f'Total cost ≤ ₹{budget_cap/1e9:.1f}B'
            },
            'demand_satisfaction': {
                'type': 'equality',
                'value': 1.0,
                'description': 'All ward demand must be met'
            },
            'facility_capacity': {
                'type': 'upper_bound',
                'description': 'Facility capacity limits'
            },
            'land_use_compatibility': {
                'type': 'compatibility',
                'description': 'Facilities in valid zones only'
            }
        }
        
        print(f"✓ Defined {len(self.constraints)} constraint types")
        return self.constraints
    
    def verify_constraints(self):
        """Verify Phase 6 design against constraints"""
        print("\nVerifying Phase 6 constraints...")
        
        violations = []
        
        # Check sector-level constraints
        for _, sector in self.sectors.iterrows():
            sector_geom = sector.geometry
            sector_name = sector.get('sector_name', f"Sector {sector.name}")
            
            # Count hospitals in this sector
            hospitals_in_sector = self.hospitals[
                self.hospitals.geometry.within(sector_geom)
            ]
            
            # Check specialty hospitals
            if 'hospital_type' in self.hospitals.columns:
                specialty = len(hospitals_in_sector[
                    hospitals_in_sector['hospital_type'].str.contains('specialty', case=False, na=False)
                ])
            else:
                specialty = len(hospitals_in_sector) // 4  # Estimate
            
            if specialty < 1:
                violations.append(
                    f"{sector_name}: Only {specialty} specialty hospital(s) (need ≥1)"
                )
            
            # Check general hospitals
            general = len(hospitals_in_sector) - specialty
            if general < 3:
                violations.append(
                    f"{sector_name}: Only {general} general hospital(s) (need ≥3)"
                )
            
            # Check school density
            schools_in_sector = self.schools[
                self.schools.geometry.within(sector_geom)
            ]
            sector_area = sector_geom.area * 111 * 111  # km²
            school_density = len(schools_in_sector) / sector_area if sector_area > 0 else 0
            
            if school_density < 0.5:
                violations.append(
                    f"{sector_name}: School density {school_density:.2f}/km² (need ≥0.5)"
                )
        
        if violations:
            print(f"⚠ Found {len(violations)} constraint violations:")
            for v in violations[:10]:
                print(f"  - {v}")
            if len(violations) > 10:
                print(f"  ... and {len(violations)-10} more")
        else:
            print("✓ All constraints satisfied in Phase 6 design")
        
        self.constraint_violations = violations
        return violations
    
    def formulate_optimization_problem(self):
        """Create formal optimization problem structure"""
        print("\n" + "="*60)
        print("FORMAL OPTIMIZATION PROBLEM SPECIFICATION")
        print("="*60)
        
        facility_counts = {
            'hospitals': len(self.hospitals),
            'schools': len(self.schools),
            'emergency': len(self.emergency),
            'parks': len(self.parks),
            'government': len(self.govt_offices)
        }
        
        self.problem = OptimizationProblem(
            num_facilities=facility_counts,
            num_wards=len(self.wards),
            num_sectors=len(self.sectors),
            num_road_segments=len(self.roads),
            num_metro_segments=len(self.metro),
            budget_cap=1e9,
            sector_constraints=self.constraints,
            capacity_constraints={}
        )
        
        print(f"\nProblem Scale:")
        print(f"  Wards: {self.problem.num_wards}")
        print(f"  Sectors: {self.problem.num_sectors}")
        for ftype, count in facility_counts.items():
            print(f"  {ftype.capitalize()}: {count}")
        print(f"  Road segments: {self.problem.num_road_segments}")
        print(f"  Metro segments: {self.problem.num_metro_segments}")
        
        total_vars = sum(v['count'] for v in self.decision_vars.values())
        print(f"\nOptimization Variables: {total_vars:,}")
        print(f"Constraints: {len(self.constraints)}")
        
        return self.problem
    
    def export_problem_specification(self, output_dir: str = BASE_DIR/"results"):
        """Export formal problem specification"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save travel time matrix
        np.save(ARTIFACTS_DIR / "travel_time_matrix.npy", self.travel_time_matrix)
        print(f"✓ Saved: travel_time_matrix.npy")
        
        # Save demand vectors
        self.demand_vectors.to_csv(output_path / "demand_vectors.csv", index=False)
        print(f"✓ Saved: demand_vectors.csv")
        
        # Save ward centroids for optimization
        ward_centroids = self.wards.copy()
        ward_centroids['centroid_x'] = ward_centroids.geometry.centroid.x
        ward_centroids['centroid_y'] = ward_centroids.geometry.centroid.y
        ward_centroids[['ward_name', 'centroid_x', 'centroid_y']].to_csv(
            output_path / "ward_centroids.csv", index=False
        )
        print(f"✓ Saved: ward_centroids.csv")
        
        # Save problem specification
        spec = {
            'problem_scale': {
                'num_wards': self.problem.num_wards,
                'num_sectors': self.problem.num_sectors,
                'num_facilities': self.problem.num_facilities,
                'num_variables': sum(v['count'] for v in self.decision_vars.values())
            },
            'decision_variables': {
                k: {
                    'shape': v['shape'],
                    'type': v['type'],
                    'description': v['description'],
                    'count': v['count']
                }
                for k, v in self.decision_vars.items()
            },
            'constraints': self.constraints,
            'constraint_violations_count': len(self.constraint_violations),
            'constraint_violations': self.constraint_violations[:20]  # First 20
        }
        
        with open(output_path / "optimization_problem.json", 'w') as f:
            json.dump(spec, f, indent=2)
        print(f"✓ Saved: optimization_problem.json")
        
        # Export current facilities for comparison
        self.facilities.to_file(output_path / "facilities.geojson", driver='GeoJSON')
        print(f"✓ Saved: facilities_phase6.geojson")
        
        print(f"\n{'='*60}")
        print(f"✓ All outputs saved to: {output_path.absolute()}")
        print(f"{'='*60}")
        
        return output_path

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("PHASE 7.0 — PROBLEM FORMALIZATION")
    print("Bengaluru Urban Planning Optimization")
    print("="*60)
    
    # Initialize
    optimizer = UrbanOptimizationFormalization()
    
    # Step 1: Load your actual Phase 6 data
    optimizer.load_phase6_data()
    
    # Step 2: Compute travel time matrix
    travel_matrix = optimizer.compute_travel_time_matrix(
        avg_speed_road=35,  # km/h
        avg_speed_metro=40   # km/h
    )
    
    # Step 3: Compute demand vectors
    demand = optimizer.compute_demand_vectors()
    
    # Step 4: Define decision variables
    decision_vars = optimizer.define_decision_variables()
    
    # Step 5: Define constraints
    constraints = optimizer.define_constraints(budget_cap=1e9)
    
    # Step 6: Verify Phase 6 design
    violations = optimizer.verify_constraints()
    
    # Step 7: Formulate problem
    problem = optimizer.formulate_optimization_problem()
    
    # Step 8: Export everything
    output_path = optimizer.export_problem_specification()
    
    print(f"\n📊 Summary:")
    print(f"  Wards analyzed: {len(optimizer.wards)}")
    print(f"  Facilities: {len(optimizer.facilities)}")
    print(f"  Decision variables: {sum(v['count'] for v in decision_vars.values()):,}")
    print(f"  Constraints: {len(constraints)}")
    if violations:
        print(f"  ⚠ Constraint violations: {len(violations)}")
    
    print(f"\n📁 Outputs: {output_path.absolute()}")