import geopandas as gpd
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
SUMMARY = BASE_DIR / "summary"

for d in [ARTIFACTS,SUMMARY]:
    d.mkdir(parents=True, exist_ok=True)


LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "sewage_layer.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

logger.info("Sewage (STP) infrastructure layer generation started")


sectors = gpd.read_file(
    BASE_DIR/"data/processed/bbmp_5sectors_named.geojson"
).to_crs(epsg=32643)

stp_offsets = {
    "North":   ( 2000, -4000),
    "West":    (-4000, -3000),
    "East":    ( 4000,   0),
    "South":   ( 2000, -5000)
}

features = []

for _, row in sectors.iterrows():
    if row["sector"] == "Central":
        continue  # NO STP in central

    dx, dy = stp_offsets[row["sector"]]
    stp = translate(row.geometry.centroid, dx, dy)

    features.append({
        "type": "Feature",
        "geometry": stp,
        "properties": {
            "type": "STP",
            "sector": row["sector"],
            "capacity_mld": 180
        }
    })

gpd.GeoDataFrame.from_features(features, crs=32643)\
    .to_crs(4326)\
    .to_file(ARTIFACTS/"sewage_network.geojson")

logger.info(f"Saved sewage network GeoJSON: {ARTIFACTS / 'sewage_network.geojson'}")


ARTIFACT_METADATA_DIR = ARTIFACTS / "metadata"
ARTIFACT_METADATA_DIR.mkdir(parents=True, exist_ok=True)

artifact_metadata = {
    "artifact": "sewage_network.geojson",
    "infrastructure_type": "Sewage Treatment Plants (STP)",
    "placement_method": "Sector centroid with fixed spatial offsets",
    "capacity_assumption_mld": 180,
    "excluded_sectors": ["Central"],
    "crs": "EPSG:4326",
    "num_facilities": len(features),
    "phase": "Phase 6",
    "description": "Conceptual sewage treatment plant placement for redesigned city"
}

with open(ARTIFACT_METADATA_DIR / "sewage_layer_metadata.json", "w") as f:
    json.dump(artifact_metadata, f, indent=2)

METADATA_DIR = BASE_DIR / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

out_path = ARTIFACTS / "sewage_network.geojson"

run_manifest = {
    "script": "build_sewage_layer.py",
    "timestamp_utc": datetime.utcnow().isoformat(),
    "input_file": "bbmp_5sectors_named.geojson",
    "output_file": str(out_path),
    "output_sha256": sha256(out_path),
    "python_version": sys.version,
    "platform": platform.platform(),
    "phase": "Phase 6"
}

with open(METADATA_DIR / "run_manifest_sewage_layer.json", "w") as f:
    json.dump(run_manifest, f, indent=2)



logger.info(" Sewage infrastructure layer generation completed successfully")

