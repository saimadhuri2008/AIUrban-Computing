import geopandas as gpd
import logging
import sys
import json
import hashlib
import platform
from datetime import datetime
from pathlib import Path

BASE_DIR = Path("redesign")

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "consumption_map.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

out_path = ARTIFACTS_DIR / "consumption_map.geojson"


logger.info("Sector-based consumption map generation started")

wards = gpd.read_file("data/processed/wards/wards_with_sector_fixed.geojson").to_crs(4326)

def consumption(row):
    if row["sector"] == "West":
        return 100
    if row["sector"] == "Central":
        return 50
    if row["sector"] == "South":
        return 40
    if row["sector"] == "East":
        return 40
    return 10

wards["consumption_units"] = wards.apply(consumption, axis=1)

wards.to_file(out_path)
logger.info(f"Saved consumption map: {out_path}")

artifact_metadata = {
    "artifact": "consumption_map.geojson",
    "method": "Rule-based sector-wise assignment",
    "sector_consumption_units": {
        "West": 100,
        "Central": 50,
        "South": 40,
        "East": 40,
        "Other": 10
    },
    "units": "relative consumption units",
    "crs": "EPSG:4326",
    "num_features": len(wards),
    "phase": "Phase 6",
    "description": "Sector-level consumption intensity map for redesigned city"
}

with open(ARTIFACTS_DIR / "metadata/consumption_map_metadata.json", "w") as f:
    json.dump(artifact_metadata, f, indent=2)

METADATA_DIR = BASE_DIR / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

run_manifest = {
    "script": "consumption_map.py",
    "timestamp_utc": datetime.utcnow().isoformat(),
    "input_file": "wards_with_sector_fixed.geojson",
    "output_file": str(out_path),
    "output_sha256": sha256(out_path),
    "python_version": sys.version,
    "platform": platform.platform(),
    "phase": "Phase 6"
}

with open(METADATA_DIR / "run_manifest_consumptionmap.json", "w") as f:
    json.dump(run_manifest, f, indent=2)

logger.info("Consumption map generation completed successfully")

