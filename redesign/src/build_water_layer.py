import geopandas as gpd
from shapely.affinity import translate
from shapely.geometry import LineString
from pathlib import Path
import logging
import sys
import json
import hashlib
import platform
from datetime import datetime

BASE_DIR = Path("redesign")
ARTIFACTS = BASE_DIR /"artifacts"
SUMMARY = BASE_DIR / "summary"

for d in [ARTIFACTS,SUMMARY]:
    d.mkdir(parents=True, exist_ok=True)

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "water_layer.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

logger.info("Water treatment infrastructure layer generation started")


sectors = gpd.read_file(
    BASE_DIR/"data/processed/bbmp_5sectors_named.geojson"
).to_crs(epsg=32643)

wtp_offsets = {
    "North":   (  0,  4000),
    "West":    (-3000, 3500),
    "Central": ( 2000, 3000),
    "East":    ( 3000, 3500),
    "South":   ( 3000, -3000)
}

features = []

for _, row in sectors.iterrows():
    dx, dy = wtp_offsets[row["sector"]]
    wtp = translate(row.geometry.centroid, dx, dy)

    features.append({
        "type": "Feature",
        "geometry": wtp,
        "properties": {
            "type": "water_treatment_plant",
            "sector": row["sector"],
            "capacity_mld": 250
        }
    })

gpd.GeoDataFrame.from_features(features, crs=32643)\
    .to_crs(4326)\
    .to_file(ARTIFACTS/"water_treatment.geojson")

logger.info(f"Saved water treatment GeoJSON: {ARTIFACTS / 'water_treatment.geojson'}")

ARTIFACT_METADATA_DIR = ARTIFACTS / "metadata"
ARTIFACT_METADATA_DIR.mkdir(parents=True, exist_ok=True)

artifact_metadata = {
    "artifact": "water_treatment.geojson",
    "infrastructure_type": "Water Treatment Plants (WTP)",
    "placement_method": "Sector centroid with fixed spatial offsets",
    "capacity_assumption_mld": 250,
    "crs": "EPSG:4326",
    "num_facilities": len(features),
    "phase": "Phase 6",
    "description": "Conceptual water treatment plant placement for redesigned city"
}

with open(ARTIFACT_METADATA_DIR / "water_layer_metadata.json", "w") as f:
    json.dump(artifact_metadata, f, indent=2)

METADATA_DIR = BASE_DIR / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

out_path = ARTIFACTS / "water_treatment.geojson"

run_manifest = {
    "script": "build_water_layer.py",
    "timestamp_utc": datetime.utcnow().isoformat(),
    "input_file": "bbmp_5sectors_named.geojson",
    "output_file": str(out_path),
    "output_sha256": sha256(out_path),
    "python_version": sys.version,
    "platform": platform.platform(),
}

with open(METADATA_DIR / "run_manifest_water_layer.json", "w") as f:
    json.dump(run_manifest, f, indent=2)


logger.info("Water treatment infrastructure layer generation completed successfully")

