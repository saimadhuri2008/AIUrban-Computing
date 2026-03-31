import geopandas as gpd
import pandas as pd

# ---------------- LOAD FILES ----------------
wards_gdf = gpd.read_file(
    "data/processed/wards/wards_with_sectors.geojson" 
).to_crs(4326)

forecast = pd.read_csv(
    "results/forecasting/ensemble_bengaluru_forecasting/combined_forecast_2026_2035.csv"
)

# ---------------- PROCESS FORECAST ----------------
forecast["date"] = pd.to_datetime(forecast["date"])

# Filter only 2035
forecast_2035 = forecast[forecast["date"].dt.year == 2035]

# Aggregate all required variables per ward
forecast_2035_agg = (
    forecast_2035
    .groupby("ward_id")[[
        "population",
        "electricity_demand",
        "water_demand",
        "congestion_index",
        "pm25"
    ]]
    .mean()
    .reset_index()
)

# ---------------- MERGE WITH WARDS ----------------
wards_2035 = wards_gdf.merge(
    forecast_2035_agg,
    on="ward_id",
    how="left"
)

# ---------------- HANDLE MISSING VALUES ----------------
for col in [
    "population",
    "electricity_demand",
    "water_demand",
    "congestion_index",
    "pm25"
]:
    wards_2035[col] = wards_2035[col].fillna(
        wards_2035[col].mean()
    )

# ---------------- SAVE OUTPUT ----------------
wards_2035.to_file(
    "data/processed/wards/wards_2035_all_variables.geojson",
    driver="GeoJSON"
)

print("✅ Wards merged with 2035 population, demand, congestion, PM2.5, and geometry")
