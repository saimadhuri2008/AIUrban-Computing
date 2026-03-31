import geopandas as gpd
from shapely.geometry import LineString
from shapely.affinity import translate
from pathlib import Path
import logging
import sys
import json
import hashlib
import platform
from datetime import datetime


BASE_DIR = Path("redesign")
ARTIFACTS = BASE_DIR /"artifacts"
ARTIFACTS_POWERLAYER = ARTIFACTS /"powerlayer"
SUMMARY = BASE_DIR / "summary"

for d in [ARTIFACTS,SUMMARY,ARTIFACTS_POWERLAYER]:
    d.mkdir(parents=True, exist_ok=True)

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "power_layer.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------- LOAD + PROJECT ----------
sectors = gpd.read_file(
    "redesign/data/processed/bbmp_5sectors_named.geojson"
).to_crs(epsg=32643)  # meters

power_nodes = []
lines = []

logger.info("Power infrastructure layer generation started")


# ---------- 1.1 POWER PLANT ----------
west = sectors[sectors["sector"] == "West"].geometry.iloc[0]

# place OUTSIDE dense wards → push SW
plant_pt = translate(west.centroid, xoff=-6000, yoff=-3000)

power_nodes.append({
    "type": "Feature",
    "geometry": plant_pt,
    "properties": {
        "type": "power_plant",
        "sector": "West",
        "capacity_mw": 1200
    }
})

# ---------- 1.2 SUBSTATIONS ----------
offsets = {
    "West":    ( 3000,   0),
    "Central": ( 2500, 2000),
    "North":   (   0, -2500),
    "East":    (-2500,   0),
    "South":   (-2000, 2500)
}

substations = {}

for _, row in sectors.iterrows():
    dx, dy = offsets[row["sector"]]
    sub = translate(row.geometry.centroid, dx, dy)
    substations[row["sector"]] = sub

    power_nodes.append({
        "type": "Feature",
        "geometry": sub,
        "properties": {
            "type": "substation",
            "sector": row["sector"],
            "capacity_mw": 300
        }
    })

# ---------- 1.3 TRANSMISSION LINES ----------
for sector, sub in substations.items():
    lines.append({
        "type": "Feature",
        "geometry": LineString([plant_pt, sub]),
        "properties": {
            "type": "transmission_line",
            "to_sector": sector
        }
    })

# ---------- WRITE ----------
gpd.GeoDataFrame.from_features(power_nodes, crs=32643)\
    .to_crs(4326)\
    .to_file(ARTIFACTS_POWERLAYER/"power_plants.geojson")
logger.info(f"Saved power plants GeoJSON: {ARTIFACTS_POWERLAYER / 'power_plants.geojson'}")

gpd.GeoDataFrame.from_features(lines, crs=32643)\
    .to_crs(4326)\
    .to_file(ARTIFACTS_POWERLAYER/"power_lines.geojson")
logger.info(f"Saved power transmission lines GeoJSON: {ARTIFACTS_POWERLAYER / 'power_lines.geojson'}")

ARTIFACT_METADATA_DIR = ARTIFACTS / "metadata"
ARTIFACT_METADATA_DIR.mkdir(parents=True, exist_ok=True)

artifact_metadata = {
    "artifacts": [
        "power_plants.geojson",
        "power_lines.geojson"
    ],
    "method": "Rule-based placement using sector centroids with fixed spatial offsets",
    "capacity_assumptions": {
        "power_plant_mw": 1200,
        "substation_mw": 300
    },
    "crs": "EPSG:4326",
    "phase": "Phase 6",
    "description": "Conceptual power generation and transmission layout for redesigned city"
}

with open(ARTIFACT_METADATA_DIR / "power_layer_metadata.json", "w") as f:
    json.dump(artifact_metadata, f, indent=2)

METADATA_DIR = BASE_DIR / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

run_manifest = {
    "script": "build_power_layer.py",
    "timestamp_utc": datetime.utcnow().isoformat(),
    "input_file": "bbmp_5sectors_named.geojson",
    "output_files": [
        str(ARTIFACTS_POWERLAYER / "power_plants.geojson"),
        str(ARTIFACTS_POWERLAYER / "power_lines.geojson")
    ],
    "python_version": sys.version,
    "platform": platform.platform(),
}

with open(METADATA_DIR / "run_manifest_power_layer.json", "w") as f:
    json.dump(run_manifest, f, indent=2)



print("✅ Power system generated")
