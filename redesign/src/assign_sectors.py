#!/usr/bin/env python3
"""
kml_to_bbmp_5sectors.py

Reads an official 5-sector KML file and outputs:
 - bbmp_5sectors.geojson
"""

from pathlib import Path
import geopandas as gpd
import logging
import sys
from datetime import datetime
import json
import hashlib
import platform


BASE_DIR = Path("redesign")
INTERIM = BASE_DIR / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path("redesign/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "kml_to_sectors.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)



def read_kml_try(path):
    """
    Robustly read a KML file.
    Handles multiple layers if present.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"KML not found: {path}")

    # Try direct read
    try:
        gdf = gpd.read_file(p)
        if len(gdf) > 0:
            return gdf
    except Exception:
        pass

    # Try reading individual layers
    try:
        import fiona
        layers = fiona.listlayers(p)
        if not layers:
            raise RuntimeError("No layers found in KML")

        for layer in layers:
            try:
                gdf = gpd.read_file(p, layer=layer)
                if len(gdf) > 0:
                    return gdf
            except Exception:
                continue
    except Exception:
        raise RuntimeError(
            "Failed to read KML. Ensure GDAL/Fiona are installed correctly."
        )

    raise RuntimeError("Could not read any valid layer from KML")


def ensure_sector_column(gdf):
    """
    Ensure a column named 'sector' exists.
    """
    candidates = [
        "sector", "name", "Name", "Name_1",
        "description", "title", "Name_0"
    ]

    for col in candidates:
        if col in gdf.columns:
            gdf = gdf.rename(columns={col: "sector"})
            break

    if "sector" not in gdf.columns:
        non_geom_cols = [c for c in gdf.columns if c != gdf.geometry.name]
        if non_geom_cols:
            gdf = gdf.rename(columns={non_geom_cols[0]: "sector"})
        else:
            gdf["sector"] = [f"sector_{i+1}" for i in range(len(gdf))]

    gdf["sector"] = gdf["sector"].astype(str)
    return gdf


def main(kml_path):
    logger.info(f"Reading KML: {kml_path}")

    sectors = read_kml_try(kml_path)

    # Keep only polygon geometries
    sectors = sectors[
        sectors.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    if sectors.empty:
        raise RuntimeError("No polygon geometries found in KML")

    sectors = ensure_sector_column(sectors)

    # Clean KML HTML from names if present
    sectors["sector"] = (
        sectors["sector"]
        .str.replace(r"<[^>]*>", "", regex=True)
        .str.strip()
    )

    # Force WGS84
    sectors = sectors.to_crs(epsg=4326)

    # Output GeoJSON
    out_path = INTERIM / "bbmp_5sectors.geojson"
    sectors.to_file(out_path, driver="GeoJSON")

    logger.info(f"[SAVED] {out_path}")
    logger.info(f"Total sectors: {len(sectors)}")

    ARTIFACTS_DIR = Path("redesign/artifacts/metadata")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    sector_metadata = {
        "artifact": "bbmp_5sectors.geojson",
        "source": str(kml_path),
        "geometry_type": "Polygon/MultiPolygon",
        "crs": "EPSG:4326",
        "num_features": len(sectors),
        "phase": "City Redesign",
        "description": "Standardized official BBMP 5-sector spatial design"
    }

    with open(ARTIFACTS_DIR / "sector_metadata.json", "w") as f:
        json.dump(sector_metadata, f, indent=2)

    METADATA_DIR = Path("redesign/metadata")
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    def sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    run_manifest = {
        "script": "kml_to_bbmp_5sectors.py",
        "timestamp_utc": datetime.utcnow().isoformat(),
        "input_kml": str(kml_path),
        "input_kml_sha256": sha256(Path(kml_path)),
        "output_geojson": str(out_path),
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    with open(METADATA_DIR / "run_manifest_sectors.json", "w") as f:
        json.dump(run_manifest, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert BBMP 5-sector KML to GeoJSON"
    )
    parser.add_argument(
        "--kml",
        default=BASE_DIR / "data" / "raw" / "bengaluru-5sectors.kml",
        help="Official 5-sector KML file"
    )

    args = parser.parse_args()
    main(args.kml)
