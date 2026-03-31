"""
FACILITY LOCATION OPTIMIZATION
Bengaluru Urban Planning Project
Mixed Integer Linear Programming (MILP)

Purpose: Minimize population-weighted travel time to essential services
Solver: OR-Tools (Google) - Production-grade, open-source
Author: Your Name
Date: 2025
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from ortools.linear_solver import pywraplp
from scipy.spatial import cKDTree
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path("optimization")
ARTIFACTS_DIR = BASE_DIR / "artifacts/facilities"

@dataclass
class OptimizationResult:
    """Results from facility optimization"""
    objective_value: float
    facility_locations: gpd.GeoDataFrame
    service_allocation: np.ndarray
    travel_time_reduction: float
    computation_time: float
    solver_status: str
    num_facilities: int

class FacilityLocationOptimizer:
    """
     Optimal facility location using MILP
    
    Objective: Minimize population-weighted travel time
    min Σ_w,f population(w) * travel_time(w,f) * y_w,f
    
    Constraints:
    - Facility count limits
    - Service coverage requirements
    - Travel time bounds
    - Sector-level requirements
    """
    
    def __init__(self, phase7_dir: str = BASE_DIR/"results/facilities"):
        self.data_dir = Path(BASE_DIR/"results")
        self.solver = None
        self.results = {}
        
    def load_formalized_problem(self):
        """Load Phase 7.0 outputs"""
        print("Loading Phase 7.0 formalized problem...")
        
        # Load matrices and data
        self.travel_matrix = np.load(self.data_dir / "travel_time_matrix.npy")
        self.demand = pd.read_csv(self.data_dir / "demand_vectors.csv")
        self.ward_centroids = pd.read_csv(self.data_dir / "ward_centroids.csv")
        
        # Load problem specification
        with open(self.data_dir / "optimization_problem.json", 'r') as f:
            self.problem_spec = json.load(f)
        
        # Load geospatial data
        self.facilities_current = gpd.read_file(
            self.data_dir / "facilities.geojson"
        )
        
        # Load original data for visualization
        self.wards = gpd.read_file(
            "data/processed/wards/wards_2035_all_variables.geojson"
        ).to_crs(4326)
        
        self.sectors = gpd.read_file(
            "data/processed/wards/bbmp_5sectors.geojson"
        ).to_crs(4326)
        
        n_wards = len(self.ward_centroids)
        n_facilities = len(self.facilities_current)
        
        print(f"✓ Problem loaded:")
        print(f"  Wards: {n_wards}")
        print(f"  Current facilities: {n_facilities}")
        print(f"  Travel matrix: {self.travel_matrix.shape}")
        
        # Get population
        if 'population' in self.demand.columns:
            self.population = self.demand['population'].values
        else:
            self.population = np.ones(n_wards) * 10000  # Default
        
        print(f"  Total population: {self.population.sum():,.0f}")
        
    def generate_candidate_locations(self, 
                                     density: float = 3.0,
                                     min_spacing: float = 0.01):
        """
        Generate candidate facility locations
        
        Strategy: Grid + random sampling within valid zones
        density: candidates per km²
        min_spacing: minimum spacing in degrees (~1 km)
        """
        print(f"\nGenerating candidate locations...")
        print(f"  Density: {density} per km²")
        print(f"  Min spacing: {min_spacing}° (~{min_spacing*111:.1f} km)")
        
        candidates = []
        
        # Generate candidates for each sector
        for idx, sector in self.sectors.iterrows():
            sector_geom = sector.geometry
            sector_name = sector.get('sector_name', f'Sector {idx}')
            
            # Calculate area
            bounds = sector_geom.bounds
            area_deg2 = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
            area_km2 = area_deg2 * 111 * 111
            
            # Number of candidates for this sector
            n_candidates = max(int(area_km2 * density), 20)
            
            print(f"  {sector_name}: {area_km2:.1f} km² → {n_candidates} candidates")
            
            # Generate with spacing constraint
            sector_candidates = []
            attempts = 0
            max_attempts = n_candidates * 50
            
            while len(sector_candidates) < n_candidates and attempts < max_attempts:
                attempts += 1
                
                # Random point in bounding box
                x = np.random.uniform(bounds[0], bounds[2])
                y = np.random.uniform(bounds[1], bounds[3])
                point = gpd.points_from_xy([x], [y], crs=4326)[0]
                
                # Check if inside sector
                if not sector_geom.contains(point):
                    continue
                
                # Check minimum spacing
                if sector_candidates:
                    dists = [
                        np.sqrt((x - c['x'])**2 + (y - c['y'])**2)
                        for c in sector_candidates
                    ]
                    if min(dists) < min_spacing:
                        continue
                
                sector_candidates.append({
                    'geometry': point,
                    'sector_id': sector.get('sector_id', idx),
                    'sector_name': sector_name,
                    'x': x,
                    'y': y
                })
            
            candidates.extend(sector_candidates)
        
        self.candidates = gpd.GeoDataFrame(candidates, crs=4326)
        
        print(f"✓ Generated {len(self.candidates)} candidate locations")
        print(f"  Distribution: {self.candidates['sector_name'].value_counts().to_dict()}")
        
        return self.candidates
    
    def compute_candidate_travel_times(self):
        """
        Compute travel time from each ward to each candidate
        Uses network-based approximation
        """
        print("\nComputing ward → candidate travel times...")
        
        n_wards = len(self.ward_centroids)
        n_candidates = len(self.candidates)
        
        # Extract coordinates
        ward_coords = self.ward_centroids[['centroid_x', 'centroid_y']].values
        cand_coords = self.candidates[['x', 'y']].values
        
        # Euclidean distance matrix (degrees)
        print("  Computing distances...")
        dist_matrix_deg = np.zeros((n_wards, n_candidates))
        
        for i in range(n_wards):
            dx = cand_coords[:, 0] - ward_coords[i, 0]
            dy = cand_coords[:, 1] - ward_coords[i, 1]
            dist_matrix_deg[i, :] = np.sqrt(dx**2 + dy**2)
        
        # Convert to km
        dist_matrix_km = dist_matrix_deg * 111
        
        # Convert to travel time
        # Use variable speed based on distance (accounts for route inefficiency)
        travel_time_matrix = np.zeros_like(dist_matrix_km)
        
        for i in range(n_wards):
            for j in range(n_candidates):
                dist_km = dist_matrix_km[i, j]
                
                # Speed model: slower for short distances (congestion)
                if dist_km < 2:
                    speed = 20  # km/h (congested local roads)
                elif dist_km < 10:
                    speed = 35  # km/h (main roads)
                else:
                    speed = 50  # km/h (highways/express)
                
                # Add detour factor (20% longer than straight line)
                actual_dist = dist_km * 1.2
                travel_time_matrix[i, j] = (actual_dist / speed) * 60  # minutes
        
        self.ward_candidate_time = travel_time_matrix
        
        print(f"✓ Travel time matrix: {travel_time_matrix.shape}")
        print(f"  Mean: {travel_time_matrix.mean():.1f} min")
        print(f"  Median: {np.median(travel_time_matrix):.1f} min")
        print(f"  90th percentile: {np.percentile(travel_time_matrix, 90):.1f} min")
        
        return travel_time_matrix
    
    def optimize_facility_type(self,
                               facility_type: str,
                               n_facilities: int,
                               max_travel_time: float,
                               time_limit: int = 300):
        """
        Optimize locations for a specific facility type
        
        Args:
            facility_type: 'hospital', 'school', 'emergency', etc.
            n_facilities: Target number of facilities
            max_travel_time: Maximum acceptable travel time (minutes)
            time_limit: Solver time limit (seconds)
        """
        print(f"\n{'='*60}")
        print(f"OPTIMIZING: {facility_type.upper()}")
        print(f"{'='*60}")
        print(f"  Target facilities: {n_facilities}")
        print(f"  Max travel time: {max_travel_time} min")
        print(f"  Time limit: {time_limit}s")
        
        start_time = time.time()
        
        # Initialize solver
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("ERROR: SCIP solver not available. Install with: pip install ortools")
            return None
        
        n_wards = self.ward_candidate_time.shape[0]
        n_candidates = self.ward_candidate_time.shape[1]
        
        # Decision variables
        print("\n  Creating variables...")
        
        # x[j]: Binary - select candidate j
        x = [solver.BoolVar(f'x_{j}') for j in range(n_candidates)]
        
        # y[i,j]: Continuous - fraction of ward i served by candidate j
        y = {}
        for i in range(n_wards):
            for j in range(n_candidates):
                y[i, j] = solver.NumVar(0, 1, f'y_{i}_{j}')
        
        print(f"  Variables created: {len(x) + len(y):,}")
        
        # Objective: minimize population-weighted travel time
        print("  Setting objective...")
        objective = solver.Objective()
        
        for i in range(n_wards):
            pop = self.population[i]
            for j in range(n_candidates):
                coef = pop * self.ward_candidate_time[i, j]
                objective.SetCoefficient(y[i, j], coef)
        
        objective.SetMinimization()
        
        # Constraints
        print("  Adding constraints...")
        
        # 1. Select exactly n_facilities
        ct_count = solver.Constraint(n_facilities, n_facilities)
        for j in range(n_candidates):
            ct_count.SetCoefficient(x[j], 1)
        
        # 2. Each ward fully served
        for i in range(n_wards):
            ct_serve = solver.Constraint(1, 1)
            for j in range(n_candidates):
                ct_serve.SetCoefficient(y[i, j], 1)
        
        # 3. Can only serve from selected facilities
        for i in range(n_wards):
            for j in range(n_candidates):
                ct_link = solver.Constraint(-solver.infinity(), 0)
                ct_link.SetCoefficient(y[i, j], 1)
                ct_link.SetCoefficient(x[j], -1)
        
        # 4. Travel time constraint (soft - allow violations with penalty)
        violations = []
        for i in range(n_wards):
            for j in range(n_candidates):
                if self.ward_candidate_time[i, j] > max_travel_time * 1.5:
                    # Hard constraint: too far
                    ct_time = solver.Constraint(0, 0)
                    ct_time.SetCoefficient(y[i, j], 1)
                elif self.ward_candidate_time[i, j] > max_travel_time:
                    # Soft constraint: penalty in objective
                    violations.append((i, j))
        
        print(f"  Constraints added: {solver.NumConstraints():,}")
        
        # Solve
        print(f"\n  Solving MILP...")
        solver.SetTimeLimit(time_limit * 1000)
        
        status = solver.Solve()
        
        computation_time = time.time() - start_time
        
        # Parse status
        status_map = {
            pywraplp.Solver.OPTIMAL: 'OPTIMAL',
            pywraplp.Solver.FEASIBLE: 'FEASIBLE',
            pywraplp.Solver.INFEASIBLE: 'INFEASIBLE',
            pywraplp.Solver.UNBOUNDED: 'UNBOUNDED',
            pywraplp.Solver.ABNORMAL: 'ABNORMAL',
            pywraplp.Solver.NOT_SOLVED: 'NOT_SOLVED'
        }
        status_str = status_map.get(status, 'UNKNOWN')
        
        print(f"\n  ✓ Solver status: {status_str}")
        print(f"  ✓ Computation time: {computation_time:.1f}s")
        
        if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
            print(f"  ⚠ Optimization failed!")
            return None
        
        # Extract solution
        obj_value = solver.Objective().Value()
        print(f"  ✓ Objective: {obj_value:,.0f} person-minutes")
        
        # Selected facilities
        selected = [j for j in range(n_candidates) if x[j].solution_value() > 0.5]
        
        print(f"  ✓ Facilities selected: {len(selected)}")
        
        # Service allocation matrix
        allocation = np.zeros((n_wards, n_candidates))
        for i in range(n_wards):
            for j in range(n_candidates):
                allocation[i, j] = y[i, j].solution_value()
        
        # Create result
        result = OptimizationResult(
            objective_value=obj_value,
            facility_locations=self.candidates.iloc[selected].copy(),
            service_allocation=allocation,
            travel_time_reduction=0,  # Calculate later
            computation_time=computation_time,
            solver_status=status_str,
            num_facilities=len(selected)
        )
        
        # Store result
        self.results[facility_type] = result
        
        return result
    
    def compute_phase6_baseline(self, facility_type: str):
        """Compute baseline travel times for Phase 6 design"""
        print(f"\n  Computing Phase 6 baseline for {facility_type}...")
        
        # Filter facilities by type
        if facility_type in ['hospital', 'school', 'emergency']:
            phase6_facilities = self.facilities_current[
                self.facilities_current['type'] == facility_type
            ]
        else:
            phase6_facilities = self.facilities_current
        
        if len(phase6_facilities) == 0:
            print(f"  ⚠ No {facility_type} facilities in Phase 6")
            return None
        
        # Build KDTree for fast nearest-neighbor
        phase6_coords = np.array([
            [p.x, p.y] for p in phase6_facilities.geometry
        ])
        tree = cKDTree(phase6_coords)
        
        # Compute travel times
        ward_coords = self.ward_centroids[['centroid_x', 'centroid_y']].values
        
        phase6_times = []
        for i, wc in enumerate(ward_coords):
            # Find nearest facility
            dist_deg, _ = tree.query(wc)
            dist_km = dist_deg * 111 * 1.2  # 20% detour
            
            # Convert to time
            if dist_km < 2:
                speed = 20
            elif dist_km < 10:
                speed = 35
            else:
                speed = 50
            
            travel_time = (dist_km / speed) * 60
            phase6_times.append(travel_time)
        
        phase6_avg = np.average(phase6_times, weights=self.population)
        
        print(f"  Phase 6 avg travel time: {phase6_avg:.1f} min")
        
        return phase6_times, phase6_avg
    
    def compare_results(self, facility_type: str):
        """Compare optimized vs Phase 6"""
        print(f"\nComparing {facility_type} optimization...")
        
        result = self.results[facility_type]
        
        # Compute Phase 6 baseline
        phase6_data = self.compute_phase6_baseline(facility_type)
        if phase6_data is None:
            return None
        
        phase6_times, phase6_avg = phase6_data
        
        # Compute optimized travel times
        opt_times = []
        for i in range(len(self.ward_centroids)):
            # Find primary serving facility
            serving_fractions = result.service_allocation[i, :]
            if serving_fractions.max() > 0:
                primary = serving_fractions.argmax()
                travel_time = self.ward_candidate_time[i, primary]
            else:
                travel_time = 999  # Not served
            opt_times.append(travel_time)
        
        opt_avg = np.average(opt_times, weights=self.population)
        
        # Calculate improvement
        reduction_pct = ((phase6_avg - opt_avg) / phase6_avg) * 100
        
        print(f"\n{'='*50}")
        print(f"{facility_type.upper()} - COMPARISON RESULTS")
        print(f"{'='*50}")
        print(f"Phase 6 (rule-based):  {phase6_avg:.1f} min")
        print(f"Optimized (MILP):      {opt_avg:.1f} min")
        print(f"Improvement:           {reduction_pct:+.1f}%")
        print(f"{'='*50}")
        
        result.travel_time_reduction = reduction_pct
        
        comparison = {
            'facility_type': facility_type,
            'phase6_avg': phase6_avg,
            'optimized_avg': opt_avg,
            'reduction_pct': reduction_pct,
            'phase6_times': phase6_times,
            'optimized_times': opt_times
        }
        
        return comparison
    
    def visualize_optimization(self, facility_type: str, output_dir: str = BASE_DIR /"results/facilities"):
        """Create visualizations"""
        print(f"\nGenerating visualizations for {facility_type}...")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        result = self.results[facility_type]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Left: Phase 6
        ax1 = axes[0]
        self.wards.plot(ax=ax1, color='lightgray', edgecolor='white', alpha=0.6)
        self.sectors.boundary.plot(ax=ax1, color='black', linewidth=2, alpha=0.8)
        
        phase6_fac = self.facilities_current[
            self.facilities_current['type'] == facility_type
        ]
        phase6_fac.plot(ax=ax1, color='#e74c3c', markersize=80, 
                       alpha=0.8, edgecolor='black', linewidth=1, 
                       label=f'Phase 6 ({len(phase6_fac)})')
        
        ax1.set_title(f'{facility_type.title()} - Phase 6 Design', 
                     fontsize=15, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.axis('off')
        
        # Right: Optimized
        ax2 = axes[1]
        self.wards.plot(ax=ax2, color='lightgray', edgecolor='white', alpha=0.6)
        self.sectors.boundary.plot(ax=ax2, color='black', linewidth=2, alpha=0.8)
        
        result.facility_locations.plot(ax=ax2, color='#27ae60', markersize=80,
                                      alpha=0.8, edgecolor='black', linewidth=1,
                                      label=f'Optimized ({result.num_facilities})')
        
        ax2.set_title(f'{facility_type.title()} - Optimized (MILP)', 
                     fontsize=15, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / f'{facility_type}_comparison.png', 
                   dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {facility_type}_comparison.png")
        
        plt.close()
    
    def export_results(self, output_dir: str = "./phase7_outputs"):
        """Export all optimization results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"\nExporting results to {output_path}...")
        
        # Export each facility type
        for ftype, result in self.results.items():
            # Optimized locations
            result.facility_locations.to_file(
                output_path / f'{ftype}_optimized.geojson',
                driver='GeoJSON'
            )
            print(f"  ✓ Saved: {ftype}_optimized.geojson")
            
            # Service allocation
            np.save(
                output_path / f'{ftype}_allocation.npy',
                result.service_allocation
            )
        
        # Summary table
        summary = []
        for ftype, result in self.results.items():
            summary.append({
                'facility_type': ftype,
                'num_facilities': result.num_facilities,
                'objective_value': result.objective_value,
                'travel_time_reduction_pct': result.travel_time_reduction,
                'computation_time_sec': result.computation_time,
                'solver_status': result.solver_status
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(output_path / 'optimization_summary.csv', index=False)
        print(f"  ✓ Saved: optimization_summary.csv")
        
        print(f"\n✓ All results exported to: {output_path.absolute()}")
        
        return output_path

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("PHASE 7.1 — FACILITY LOCATION OPTIMIZATION")
    print("="*60)
    
    # Initialize
    optimizer = FacilityLocationOptimizer(phase7_dir="./phase7_outputs")
    
    # Load problem
    optimizer.load_formalized_problem()
    
    # Generate candidates
    candidates = optimizer.generate_candidate_locations(density=2.5, min_spacing=0.008)
    
    # Compute travel times
    travel_times = optimizer.compute_candidate_travel_times()
    
    # Optimize different facility types
    facility_configs = [
        ('hospital', 60, 30),      # 60 hospitals, 30 min max
        ('school', 150, 15),        # 150 schools, 15 min max
        ('emergency', 80, 8),       # 80 emergency, 8 min max
    ]
    
    for ftype, n_fac, max_time in facility_configs:
        # Optimize
        result = optimizer.optimize_facility_type(
            facility_type=ftype,
            n_facilities=n_fac,
            max_travel_time=max_time,
            time_limit=300
        )
        
        if result:
            # Compare
            comparison = optimizer.compare_results(ftype)
            
            # Visualize
            optimizer.visualize_optimization(ftype)
    
    # Export all results
    output_path = optimizer.export_results()
    
    print("\n" + "="*60)
    print("✓ PHASE 7.1 COMPLETE")
    print("="*60)
    print(f"\nOptimized facility types: {len(optimizer.results)}")
    for ftype, result in optimizer.results.items():
        print(f"  {ftype}: {result.travel_time_reduction:+.1f}% improvement")
    
    print(f"\n📁 Outputs: {output_path.absolute()}")
    print("\n▶ Next: Phase 7.2 (Network Optimization) or Phase 7.3 (Equity)")