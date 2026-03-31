#!/usr/bin/env python3
"""
Phase 6.4–6.7 — Complete Master Visualization
Includes: Utilities + Transport + Land-Use + Facilities
"""

import geopandas as gpd
import json
from pathlib import Path

BASE_DIR = Path("redesign")
ARTIFACTS = BASE_DIR /"artifacts"
SUMMARY = BASE_DIR / "summary"

for d in [ARTIFACTS,SUMMARY]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------- LOAD ALL DATA ----------------
sectors = gpd.read_file(BASE_DIR/"data/processed/bbmp_5sectors_named.geojson").to_crs(4326)
wards = gpd.read_file(ARTIFACTS/"consumption_map.geojson").to_crs(4326)

# Utilities
power_pts = gpd.read_file(ARTIFACTS/"powerlayer/power_plants.geojson").to_crs(4326)
power_lines = gpd.read_file(ARTIFACTS/"powerlayer/power_lines.geojson").to_crs(4326)
water = gpd.read_file(ARTIFACTS/"water_treatment.geojson").to_crs(4326)
sewage = gpd.read_file(ARTIFACTS/"sewage_network.geojson").to_crs(4326)

# Transport
roads = gpd.read_file(ARTIFACTS/"transportlayer/transport_roads.geojson").to_crs(4326)
metro = gpd.read_file(ARTIFACTS/"transportlayer/metro_network.geojson").to_crs(4326)

# Land-use (Phase 6.6)
rich_res = gpd.read_file(ARTIFACTS/"housing_landuse/rich_residential.geojson").to_crs(4326)
middle_res = gpd.read_file(ARTIFACTS/"housing_landuse/middle_income.geojson").to_crs(4326)
affordable = gpd.read_file(ARTIFACTS/"housing_landuse/affordable_housing.geojson").to_crs(4326)
mixed_use = gpd.read_file(ARTIFACTS/"housing_landuse/mixed_use.geojson").to_crs(4326)
industrial = gpd.read_file(ARTIFACTS/"housing_landuse/industrial_zones.geojson").to_crs(4326)
slums = gpd.read_file(ARTIFACTS/"housing_landuse/slum_upgradation.geojson").to_crs(4326)

# Facilities (Phase 6.7)
hospitals = gpd.read_file(ARTIFACTS/"facilities/hospitals_planned.geojson").to_crs(4326)
schools = gpd.read_file(ARTIFACTS/"facilities/schools_planned.geojson").to_crs(4326)
parks = gpd.read_file(ARTIFACTS/"facilities/parks_planned.geojson").to_crs(4326)
govt_offices = gpd.read_file(ARTIFACTS/"facilities/govt_offices.geojson").to_crs(4326)
emergency = gpd.read_file(ARTIFACTS/"facilities/police_fire_stations.geojson").to_crs(4326)


# ---------------- HELPERS ----------------
def to_features(gdf):
    features = []
    for _, r in gdf.iterrows():
        if r.geometry is None or r.geometry.is_empty:
            continue
        features.append({
            "type": "Feature",
            "geometry": r.geometry.__geo_interface__,
            "properties": {
                k: (str(v) if isinstance(v, (list, dict)) else v) 
                for k, v in r.items() if k != "geometry"
            }
        })
    return features

# ---------------- COLORS ----------------
SECTOR_COLORS = {
    "West":    [140, 90, 180, 160],
    "Central": [200, 60, 60, 170],
    "East":    [70, 170, 160, 160],
    "South":   [70, 140, 200, 160],
    "North":   [230, 190, 60, 160]
}

# ---------------- FEATURES ----------------
sector_feats = [{
    "type": "Feature",
    "geometry": r.geometry.__geo_interface__,
    "properties": {
        "sector": r["sector"],
        "_color": SECTOR_COLORS[r["sector"]]
    }
} for _, r in sectors.iterrows()]

# ---------------- MAP CENTER ----------------
from shapely.ops import unary_union
c = unary_union(sectors.geometry).centroid
lon, lat = c.x, c.y

# Calculate statistics
total_hospitals = len(hospitals)
total_beds = sum(hospitals.get('beds', 0) if isinstance(hospitals, dict) else getattr(hospitals.iloc[i], 'beds', 0) for i in range(len(hospitals)))
total_schools = len(schools)
total_parks = len(parks)
total_park_area = sum(float(parks.iloc[i].get('area_km2', 0)) for i in range(len(parks)))

# ---------------- HTML ----------------
html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Bangalore Master Plan - Phase 6.7 Complete</title>

<script src="https://unpkg.com/deck.gl@8.9.0/dist.min.js"></script>
<script src="https://unpkg.com/maplibre-gl@2.4.0/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@2.4.0/dist/maplibre-gl.css" rel="stylesheet"/>

<style>
body {{ margin:0; background:#0b0f1a; font-family:'Inter', sans-serif; }}
#map {{ position:absolute; inset:0; }}

#tooltip {{
  position:absolute;
  pointer-events:none;
  background:rgba(0,0,0,0.95);
  color:white;
  padding:14px;
  border-radius:8px;
  font-size:13px;
  line-height:1.7;
  max-width:320px;
  display:none;
  z-index:1000;
  border:1px solid rgba(255,255,255,0.25);
  box-shadow:0 8px 16px rgba(0,0,0,0.4);
}}

#info {{
  position:absolute;
  right:20px;
  top:20px;
  width:300px;
  background:linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
  color:white;
  padding:18px;
  border-radius:10px;
  font-size:13px;
  line-height:1.7;
  box-shadow: 0 8px 20px rgba(0,0,0,0.4);
  border:1px solid rgba(255,255,255,0.1);
}}

#info h3 {{ 
  margin:0 0 16px 0; 
  color:#60A5FA; 
  font-size:18px; 
  font-weight:700;
  letter-spacing:0.5px;
}}

#info .section {{ margin:12px 0; padding:12px 0; border-top:1px solid rgba(255,255,255,0.1); }}
#info .section:first-child {{ border-top:none; padding-top:0; }}
#info .section-title {{ 
  color:#94A3B8; 
  font-size:11px; 
  text-transform:uppercase; 
  letter-spacing:1px; 
  margin-bottom:8px;
  font-weight:600;
}}
#info .stat {{ margin:6px 0; display:flex; justify-content:space-between; }}
#info .label {{ color:#CBD5E1; font-size:12px; }}
#info .value {{ color:#F1F5F9; font-size:13px; font-weight:700; }}

.controls {{
  position:absolute;
  left:20px;
  top:20px;
  background:linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
  color:white;
  padding:16px;
  border-radius:10px;
  font-size:12px;
  box-shadow: 0 8px 20px rgba(0,0,0,0.4);
  border:1px solid rgba(255,255,255,0.1);
}}

.controls h4 {{ 
  margin:0 0 14px 0; 
  font-size:15px; 
  color:#60A5FA; 
  font-weight:700;
  letter-spacing:0.5px;
}}
.controls label {{ 
  display:block; 
  margin:9px 0; 
  cursor:pointer; 
  transition:color 0.2s;
  font-size:13px;
}}
.controls label:hover {{ color:#60A5FA; }}
.controls input {{ margin-right:10px; cursor:pointer; }}

.legend {{
  position:absolute;
  left:20px;
  bottom:20px;
  background:linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
  color:white;
  padding:14px;
  border-radius:10px;
  font-size:11px;
  box-shadow: 0 8px 20px rgba(0,0,0,0.4);
  max-width: 240px;
  border:1px solid rgba(255,255,255,0.1);
}}

.legend-item {{ display:flex; align-items:center; margin:7px 0; }}
.legend-color {{ height:3px; margin-right:10px; border-radius:2px; }}
.legend-point {{ 
  width:10px; 
  height:10px; 
  border-radius:50%; 
  margin-right:10px;
  border:2px solid rgba(255,255,255,0.3);
}}
.legend-title {{ 
  font-weight:700; 
  margin:10px 0 6px 0; 
  font-size:12px; 
  color:#94A3B8;
  text-transform:uppercase;
  letter-spacing:0.5px;
}}
</style>
</head>

<body>
<div id="map"></div>
<div id="tooltip"></div>

<div class="controls">
  <h4>🗺️ Layers</h4>
  <label><input type="checkbox" checked onchange="toggle('sectors')">Sectors</label>
  <label><input type="checkbox" checked onchange="toggle('landuse')">Land Use</label>
  <label><input type="checkbox" checked onchange="toggle('facilities')">Facilities</label>
  <label><input type="checkbox" checked onchange="toggle('roads')">Roads</label>
  <label><input type="checkbox" checked onchange="toggle('metro')">Metro</label>
  <label><input type="checkbox" onchange="toggle('power')">Power Grid</label>
  <label><input type="checkbox" onchange="toggle('water')">Water & Sewage</label>
</div>

<div id="info">
  <h3>🏙️ Master Plan</h3>
  
  <div class="section">
    <div class="section-title">Healthcare</div>
    <div class="stat">
      <span class="label">Hospitals</span>
      <span class="value">{total_hospitals}</span>
    </div>
    <div class="stat">
      <span class="label">Total Beds</span>
      <span class="value">{sum(h.get('beds', 0) for h in to_features(hospitals))}</span>
    </div>
  </div>
  
  <div class="section">
    <div class="section-title">Education</div>
    <div class="stat">
      <span class="label">Schools</span>
      <span class="value">{total_schools}</span>
    </div>
    <div class="stat">
      <span class="label">Capacity</span>
      <span class="value">{sum(s.get('capacity', 0) for s in to_features(schools)):,}</span>
    </div>
  </div>
  
  <div class="section">
    <div class="section-title">Green Space</div>
    <div class="stat">
      <span class="label">Parks</span>
      <span class="value">{total_parks}</span>
    </div>
    <div class="stat">
      <span class="label">Total Area</span>
      <span class="value">{total_park_area:.1f} km²</span>
    </div>
  </div>
  
  <div class="section">
    <div class="section-title">Safety</div>
    <div class="stat">
      <span class="label">Fire Stations</span>
      <span class="value">{len([e for e in to_features(emergency) if e['properties'].get('service_type') == 'fire_station'])}</span>
    </div>
    <div class="stat">
      <span class="label">Police Stations</span>
      <span class="value">{len([e for e in to_features(emergency) if e['properties'].get('service_type') == 'police_station'])}</span>
    </div>
  </div>
</div>

<div class="legend">
  <div class="legend-title">Facilities</div>
  <div class="legend-item">
    <div class="legend-point" style="background:#EF4444;"></div>
    <span>Hospitals</span>
  </div>
  <div class="legend-item">
    <div class="legend-point" style="background:#3B82F6;"></div>
    <span>Schools</span>
  </div>
  <div class="legend-item">
    <div class="legend-point" style="background:#10B981;"></div>
    <span>Parks</span>
  </div>
  <div class="legend-item">
    <div class="legend-point" style="background:#8B5CF6;"></div>
    <span>Government</span>
  </div>
  <div class="legend-item">
    <div class="legend-point" style="background:#F59E0B;"></div>
    <span>Emergency Services</span>
  </div>
  
  <div class="legend-title">Land Use</div>
  <div class="legend-item">
    <div class="legend-color" style="background:#FFD700; width:40px;"></div>
    <span>Rich Residential</span>
  </div>
  <div class="legend-item">
    <div class="legend-color" style="background:#64B4FF; width:40px;"></div>
    <span>Middle Income</span>
  </div>
  <div class="legend-item">
    <div class="legend-color" style="background:#78C878; width:40px;"></div>
    <span>Affordable</span>
  </div>
  <div class="legend-item">
    <div class="legend-color" style="background:#B478DC; width:40px;"></div>
    <span>Mixed Use</span>
  </div>
  <div class="legend-item">
    <div class="legend-color" style="background:#8B5A2B; width:40px;"></div>
    <span>Industrial</span>
  </div>
</div>

<script>
const {{DeckGL, PolygonLayer, ScatterplotLayer, PathLayer}} = deck;

const data = {{
  sectors: {json.dumps(sector_feats)},
  powerPts: {json.dumps(to_features(power_pts))},
  powerLines: {json.dumps(to_features(power_lines))},
  water: {json.dumps(to_features(water))},
  sewage: {json.dumps(to_features(sewage))},
  roads: {json.dumps(to_features(roads))},
  metro: {json.dumps(to_features(metro))},
  richRes: {json.dumps(to_features(rich_res))},
  middleRes: {json.dumps(to_features(middle_res))},
  affordable: {json.dumps(to_features(affordable))},
  mixedUse: {json.dumps(to_features(mixed_use))},
  industrial: {json.dumps(to_features(industrial))},
  slums: {json.dumps(to_features(slums))},
  hospitals: {json.dumps(to_features(hospitals))},
  schools: {json.dumps(to_features(schools))},
  parks: {json.dumps(to_features(parks))},
  govt: {json.dumps(to_features(govt_offices))},
  emergency: {json.dumps(to_features(emergency))}
}};

let visibility = {{
  sectors:true, landuse:true, facilities:true, roads:true, metro:true, power:false, water:false
}};

const tooltip = document.getElementById('tooltip');

function showTooltip(info, x, y) {{
  if (!info.object) {{
    tooltip.style.display = 'none';
    return;
  }}
  
  const p = info.object.properties;
  let html = '';
  
  if (p.hospital_type) {{
    html = `<strong>🏥 HOSPITAL</strong><br/>
            ${{p.name}}<br/>
            Type: ${{p.hospital_type.replace('_', ' ').toUpperCase()}}<br/>
            Beds: ${{p.beds}}<br/>
            Sector: ${{p.sector}}`;
  }} else if (p.school_type) {{
    html = `<strong>🎓 SCHOOL</strong><br/>
            ${{p.name}}<br/>
            Type: ${{p.school_type.toUpperCase()}}<br/>
            Capacity: ${{p.capacity}} students<br/>
            Sector: ${{p.sector}}`;
  }} else if (p.park_type) {{
    html = `<strong>🌳 PARK</strong><br/>
            ${{p.name}}<br/>
            Type: ${{p.park_type.toUpperCase()}}<br/>
            Area: ${{p.area_km2}} km²<br/>
            ${{p.facilities}}`;
  }} else if (p.office_type) {{
    html = `<strong>🏛️ GOVERNMENT</strong><br/>
            ${{p.name}}<br/>
            Employees: ${{p.employees}}<br/>
            Area: ${{p.area_km2}} km²`;
  }} else if (p.service_type) {{
    const emoji = p.service_type === 'fire_station' ? '🚒' : '🚓';
    html = `<strong>${{emoji}} ${{p.service_type.replace('_', ' ').toUpperCase()}}</strong><br/>
            ${{p.name}}<br/>
            Response Time: ${{p.response_time_target_min}} min<br/>
            Coverage: ${{p.coverage_radius_km}} km`;
  }} else if (p.land_use) {{
    html = `<strong>LAND USE</strong><br/>
            Type: ${{p.land_use.replace('_', ' ').toUpperCase()}}<br/>
            Sector: ${{p.sector}}<br/>
            Area: ${{p.area_km2}} km²`;
  }} else if (p.road_type) {{
    html = `<strong>ROAD</strong><br/>
            ${{p.name}}<br/>
            Type: ${{p.road_type.toUpperCase()}}<br/>
            Lanes: ${{p.lanes}}<br/>
            Length: ${{p.length_km}} km`;
  }} else if (p.line_name) {{
    html = `<strong>METRO</strong><br/>
            ${{p.line_name}}<br/>
            Stations: ${{p.stations}}<br/>
            Ridership: ${{p.ridership_daily.toLocaleString()}}/day`;
  }} else if (p.sector) {{
    html = `<strong>SECTOR</strong><br/>${{p.sector}}`;
  }}
  
  tooltip.innerHTML = html;
  tooltip.style.display = 'block';
  tooltip.style.left = x + 'px';
  tooltip.style.top = y + 'px';
}}

function toggle(k) {{
  visibility[k] = !visibility[k];
  render();
}}

function render() {{
  const layers = [];

  // Sectors (base)
  if (visibility.sectors)
    layers.push(new PolygonLayer({{
      id:'sectors',
      data:data.sectors,
      getPolygon:d=>{{
        const coords = d.geometry.coordinates;
        return d.geometry.type === 'MultiPolygon' ? coords[0] : coords;
      }},
      getFillColor:d=>d.properties._color,
      stroked:true,
      getLineColor:[255,255,255,100],
      lineWidthMinPixels:1,
      pickable:true,
      onHover:showTooltip
    }}));

  // Land use
  if (visibility.landuse) {{
    [
      {{data:data.richRes, color:[255,215,0,120]}},
      {{data:data.middleRes, color:[100,180,255,100]}},
      {{data:data.affordable, color:[120,200,120,100]}},
      {{data:data.mixedUse, color:[180,120,220,100]}},
      {{data:data.industrial, color:[139,90,43,100]}},
      {{data:data.slums, color:[255,140,60,100]}}
    ].forEach((layer, i) => {{
      layers.push(new PolygonLayer({{
        id:`landuse-${{i}}`,
        data:layer.data,
        getPolygon:d=>{{
          const coords = d.geometry.coordinates;
          return d.geometry.type === 'MultiPolygon' ? coords[0] : coords;
        }},
        getFillColor:layer.color,
        stroked:true,
        getLineColor:[...layer.color.slice(0,3), 180],
        lineWidthMinPixels:1,
        pickable:true,
        onHover:showTooltip
      }}));
    }});
  }}

  // Parks (green polygons)
  if (visibility.facilities)
    layers.push(new PolygonLayer({{
      id:'parks',
      data:data.parks,
      getPolygon:d=>{{
        const coords = d.geometry.coordinates;
        return d.geometry.type === 'MultiPolygon' ? coords[0] : coords;
      }},
      getFillColor:[16,185,129,140],
      stroked:true,
      getLineColor:[16,185,129,220],
      lineWidthMinPixels:1,
      pickable:true,
      onHover:showTooltip
    }}));

  // Government offices (purple polygons)
  if (visibility.facilities)
    layers.push(new PolygonLayer({{
      id:'govt',
      data:data.govt,
      getPolygon:d=>{{
        const coords = d.geometry.coordinates;
        return d.geometry.type === 'MultiPolygon' ? coords[0] : coords;
      }},
      getFillColor:[139,92,246,130],
      stroked:true,
      getLineColor:[139,92,246,220],
      lineWidthMinPixels:1,
      pickable:true,
      onHover:showTooltip
    }}));

  // Power
  if (visibility.power) {{
    layers.push(new PathLayer({{
      id:'powerLines',
      data:data.powerLines,
      getPath:d=>d.geometry.coordinates,
      getColor:[255,210,80,180],
      getWidth:30,
      widthMinPixels:1
    }}));
    layers.push(new ScatterplotLayer({{
      id:'powerPts',
      data:data.powerPts,
      getPosition:d=>d.geometry.coordinates,
      getRadius:d=>d.properties.type==='power_plant'?900:450,
      getFillColor:d=>d.properties.type==='power_plant'?[255,80,80]:[255,210,80],
      pickable:true,
      onHover:showTooltip
    }}));
  }}

  // Water & Sewage
  if (visibility.water) {{
    layers.push(new ScatterplotLayer({{
      id:'water',
      data:data.water,
      getPosition:d=>d.geometry.coordinates,
      getRadius:600,
      getFillColor:[80,160,255],
      pickable:true,
      onHover:showTooltip
    }}));
    layers.push(new ScatterplotLayer({{
      id:'sewage',
      data:data.sewage,
      getPosition:d=>d.geometry.coordinates,
      getRadius:550,
      getFillColor:[150,150,150],
      pickable:true,
      onHover:showTooltip
    }}));
  }}

  // Roads
  if (visibility.roads) {{
    layers.push(new PathLayer({{
      id:'roads-regular',
      data:data.roads.filter(d => d.properties.category !== 'ring_road'),
      getPath:d=>d.geometry.coordinates,
      getColor:d=>{{
        const cat = d.properties.category;
        const type = d.properties.road_type;
        if(type==='arterial' || cat==='sector_to_central') return[255,60,60,255];
        if(cat==='central_internal') return[255,200,50,255];
        if(type==='collector' || cat==='inter_sector') return[50,220,255,255];
        return[200,200,200,255];
      }},
      getWidth:d=>{{
        const type = d.properties.road_type;
        if(type==='arterial') return 110;
        if(type==='collector') return 95;
        return 50;
      }},
      widthMinPixels:4,
      pickable:true,
      onHover:showTooltip
    }}));

    const orrData = data.roads.filter(d => d.properties.category === 'ring_road');
    if(orrData.length > 0) {{
      layers.push(new PathLayer({{
        id:'roads-orr',
        data:orrData,
        getPath:d=>d.geometry.coordinates,
        getColor:[255,255,255,255],
        getWidth:100,
        widthMinPixels:5,
        pickable:true,
        onHover:showTooltip
      }}));
    }}
  }}

  // Metro (dashed)
  if (visibility.metro)
    layers.push(new PathLayer({{
      id:'metro',
      data:data.metro,
      getPath:d=>d.geometry.coordinates,
      getColor:d=>{{
        const lineId = d.properties.line_id;
        if(lineId==='Line_1') return[155,89,182,255];
        if(lineId==='Line_2') return[39,174,96,255];
        return[120,220,220,255];
      }},
      getWidth:80,
      widthMinPixels:3,
      getDashArray:[10,5],
      dashJustified:true,
      pickable:true,
      onHover:showTooltip
    }}));

  // Facilities (on top)
  if (visibility.facilities) {{
    // Hospitals - red
    layers.push(new ScatterplotLayer({{
      id:'hospitals',
      data:data.hospitals,
      getPosition:d=>d.geometry.coordinates,
      getRadius:d=>d.properties.hospital_type==='super_specialty'?450:250,
      getFillColor:[239,68,68],
      stroked:true,
      getLineColor:[255,255,255],
      lineWidthMinPixels:2,
      pickable:true,
      onHover:showTooltip
    }}));

    // Schools - blue
    layers.push(new ScatterplotLayer({{
      id:'schools',
      data:data.schools,
      getPosition:d=>d.geometry.coordinates,
      getRadius:100,
      getFillColor:[59,130,246],
      stroked:true,
      getLineColor:[255,255,255],
      lineWidthMinPixels:1,
      pickable:true,
      onHover:showTooltip
    }}));

    // Emergency services - orange
    layers.push(new ScatterplotLayer({{
      id:'emergency',
      data:data.emergency,
      getPosition:d=>d.geometry.coordinates,
      getRadius:100,
      getFillColor:d=>d.properties.service_type==='fire_station'?[245,158,11]:[234,179,8],
      stroked:true,
      getLineColor:[255,255,255],
      lineWidthMinPixels:1,
      pickable:true,
      onHover:showTooltip
    }}));
  }}

  deckgl.setProps({{layers}});
}}

const deckgl = new DeckGL({{
  container:'map',
  mapStyle:'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
  initialViewState:{{
    longitude:{lon},
    latitude:{lat},
    zoom:10.6,
    pitch:35
  }},
  controller:true
}});

render();
</script>
</body>
</html>
"""

Path(BASE_DIR/"scripts/redesign_map.html").write_text(html, encoding="utf-8")
print("\n✅ Complete Phase 6.4-6.7 master visualization created!")
print("   📁 File: reports/redesign_map.html")
print("\n📊 Layer Stack (bottom to top):")
print("   1. Sectors (background)")
print("   2. Land-use zones")
print("   3. Parks (green polygons)")
print("   4. Government offices (purple polygons)")
print("   5. Utilities (power, water)")
print("   6. Roads & ORR")
print("   7. Metro (dashed lines)")
print("   8. Facilities (hospitals, schools, emergency)")