#!/usr/bin/env python3
"""
ward_failure_map.py

Research-grade ward failure timeline map for Bengaluru.

Logic:
- Uses cascade_history (W × T × M)
- First failure time = first timestep where any variable crosses FAIL_LEVEL
- Time index is converted to year assuming monthly steps from START_YEAR
- Earlier failure → red
- Later failure → yellow
- No failure → green

Outputs:
- Clean, interpretable Folium HTML map
"""

from pathlib import Path
import json
import joblib
import numpy as np
import geopandas as gpd
import folium
from branca.colormap import linear

# ---------------- CONFIG ----------------
BASE = Path("cascade_model")

CASCADE_HISTORY = BASE / "artifacts/cascade/cascade_history_20251229T182054Z.joblib"
WARDS_GEO = Path("data/processed/wards/wards_2035_all_variables.geojson")

OUT_HTML = BASE / "reports/maps/ward_failure_timeline.html"

FAIL_LEVEL = 1.0
START_YEAR = 2026          # Forecast start year
STEPS_PER_YEAR = 12        # Monthly resolution
# ---------------------------------------


def main():
    print("Loading data...")

    # Load cascade history
    history = joblib.load(CASCADE_HISTORY)   # shape: (W, T, M)
    W, T, M = history.shape

    # Load wards
    gdf = gpd.read_file(WARDS_GEO).to_crs(epsg=4326)
    gdf["ward_id"] = gdf["ward_id"].astype(str)

    if len(gdf) != W:
        raise ValueError(
            f"Ward count mismatch: GeoJSON={len(gdf)} vs Cascade={W}"
        )

    # ---------------- FIRST FAILURE TIME ----------------
    first_fail_t = np.full(W, np.nan)

    for w in range(W):
        hits = np.where(history[w] >= FAIL_LEVEL)
        if len(hits[0]) > 0:
            first_fail_t[w] = hits[0][0]   # earliest timestep

    # Convert timestep → year
    first_fail_year = [
        int(START_YEAR + t // STEPS_PER_YEAR) if not np.isnan(t) else None
        for t in first_fail_t
    ]

    # Attach results safely by index
    gdf["first_fail_year"] = first_fail_year

    # ---------------- COLOR SCALE ----------------
    valid_years = gdf["first_fail_year"].dropna()

    if valid_years.empty:
        raise RuntimeError("No failures detected in cascade history.")

    min_year = int(valid_years.min())
    max_year = int(valid_years.max())

    cmap = linear.RdYlGn_11.scale(min_year, max_year)
    cmap.caption = "Ward failure timing (earlier = more critical)"

    def style_fn(feature):
        year = feature["properties"]["first_fail_year"]
        if year is None:
            return {
                "fillColor": "#2ecc71",  # green
                "color": "#444",
                "weight": 0.6,
                "fillOpacity": 0.85,
            }
        return {
            "fillColor": cmap(year),
            "color": "#444",
            "weight": 0.6,
            "fillOpacity": 0.85,
        }

    # ---------------- MAP ----------------
    center = gdf.geometry.centroid
    m = folium.Map(
        location=[center.y.mean(), center.x.mean()],
        zoom_start=11,
        tiles="CartoDB positron",
        control_scale=True
    )

    folium.GeoJson(
        gdf,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["ward_id", "first_fail_year"],
            aliases=["Ward", "Estimated first failure year"],
            localize=True,
            sticky=True
        )
    ).add_to(m)

    cmap.add_to(m)

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(OUT_HTML)

    print(f"[SAVED] Ward failure map → {OUT_HTML}")


if __name__ == "__main__":
    main()
