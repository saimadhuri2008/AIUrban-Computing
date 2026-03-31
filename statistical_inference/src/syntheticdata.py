#!/usr/bin/env python3
"""
synthetic_data.py

Generates synthetic ward-level data and a small panel for DiD.
Outputs:
 - wards_synthetic.csv  (ward-level attributes including a WKT polygon string 'wkt')
 - panel_synthetic.csv  (synthetic panel with yearly outcomes for DiD)
"""
import numpy as np, pandas as pd, math, random
from pathlib import Path
OUT = Path("phase2d_synthetic_output")
OUT.mkdir(parents=True, exist_ok=True)

N = 198
random.seed(0); np.random.seed(0)
angles = np.linspace(0, 2*math.pi, N, endpoint=False)
radii = 0.8 + 0.6 * np.random.rand(N)
xs = (radii * np.cos(angles)) * 0.1 + 77.57 + 0.02*np.random.randn(N)
ys = (radii * np.sin(angles)) * 0.08 + 12.97 + 0.02*np.random.randn(N)

population_est = np.abs(np.random.normal(40000, 35000, N)).astype(int) + (np.abs(xs-77.575)*1e6).astype(int)%5000
it_job_density_mean = np.clip(55 + 12*(np.sin(angles)+np.random.randn(N)*0.3), 45, 75)
income_index_mean = np.clip(0.04 + 0.04*(np.cos(angles) + np.random.randn(N)*0.2), 0.02, 0.18)
built_area_m2_sum = np.abs(np.random.lognormal(11, 1.2, N)).astype(int)
aqi_mean = 73 + 5*np.sin(angles*2) + np.random.randn(N)
congestion_index_mean = 0.536 + 0.005*((it_job_density_mean-60)/10) + np.random.randn(N)*0.003
total_outflow_sum = (population_est * (0.02 + 0.001*(it_job_density_mean-55))).astype(int) + (np.random.poisson(200, N))
distance_to_transit = np.clip(5*np.abs(xs-77.57) + 3*np.abs(ys-12.97) + np.random.randn(N), 0.2, 20.0)
landuse_mix = np.clip(0.4 + 0.3*(np.cos(angles)+np.random.randn(N)*0.2), 0.1, 0.9)
commute_time = np.clip(20 + 10*(distance_to_transit/5) + (population_est/60000)*10 + np.random.randn(N)*3, 10, 120)
job_accessibility = np.clip(1/(1+distance_to_transit) * (it_job_density_mean/60) * 100, 0, 200)
car_dependence_proxy = (distance_to_transit > 5).astype(int)
electricity_demand = (population_est * (0.5 + 0.01*income_index_mean) + np.random.randn(N)*1000).astype(int)

wkt_polys = []
for x,y in zip(xs, ys):
    dx = 0.005; dy = 0.004
    coords = [(x-dx,y-dy),(x+dx,y-dy),(x+dx,y+dy),(x-dx,y+dy),(x-dx,y-dy)]
    poly = "POLYGON((" + ",".join([f"{cx} {cy}" for cx,cy in coords]) + "))"
    wkt_polys.append(poly)

df = pd.DataFrame({
    'ward_id':[f'ward_{i+1}' for i in range(N)],
    'centroid_lon': xs, 'centroid_lat': ys,
    'population_est': population_est,
    'it_job_density_mean': it_job_density_mean,
    'income_index_mean': income_index_mean,
    'built_area_m2_sum': built_area_m2_sum,
    'aqi_mean': aqi_mean,
    'congestion_index_mean': congestion_index_mean,
    'total_outflow_sum': total_outflow_sum,
    'distance_to_transit': distance_to_transit,
    'landuse_mix': landuse_mix,
    'commute_time': commute_time,
    'job_accessibility': job_accessibility,
    'car_dependence_proxy': car_dependence_proxy,
    'electricity_demand': electricity_demand,
    'wkt': wkt_polys
})
df.to_csv(OUT/"wards_synthetic.csv", index=False)

years = [2019, 2022]
treated_units = df.loc[df['distance_to_transit']>8, 'ward_id'].tolist()
panel_rows = []
for ward in df['ward_id']:
    for y in years:
        base = df.loc[df['ward_id']==ward, 'congestion_index_mean'].values[0]
        val = base + np.random.randn()*0.002 + ( -0.01 if (ward in treated_units and y>=2022) else 0.0)
        panel_rows.append({'ward_id': ward, 'year': y, 'treated_flag': 1 if ward in treated_units else 0, 'congestion': val})
panel = pd.DataFrame(panel_rows)
panel.to_csv(OUT/"panel_synthetic.csv", index=False)

print("Synthetic outputs written to:", OUT)
