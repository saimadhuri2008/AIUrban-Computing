#!/usr/bin/env python3
"""
spatial_statistics.py

Spatial statistical diagnostics and clustering analysis:
- Descriptive statistics
- Correlation diagnostics
- KMeans / DBSCAN / Hierarchical clustering
- Global Moran’s I, LISA, Getis-Ord Gi*
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
import warnings
warnings.filterwarnings("ignore")

# spatial libs
import libpysal
from esda import Moran, Moran_Local, G_Local


# clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


import logging
BASE_DIR = Path("statistical_inference")
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "spatial_statistics.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logging.info("Spatial statistics module initialized")


sns.set(style="whitegrid", context="notebook")



path = BASE_DIR / "data/derived/wards_enriched.geojson"

ART_DESCRIPTIVE = BASE_DIR / "artifacts/descriptive"
ART_SPATIAL = BASE_DIR / "artifacts/spatial"
ART_CLUSTERING = BASE_DIR / "artifacts/clustering"

FIG_DIAGNOSTICS = BASE_DIR / "figures/diagnostics"
FIG_CLUSTERS = BASE_DIR / "figures/clusters"
FIG_MAPS = BASE_DIR / "figures/maps"

DERIVED = BASE_DIR / "data/derived"

for d in [
    ART_DESCRIPTIVE, ART_SPATIAL, ART_CLUSTERING,
    FIG_DIAGNOSTICS, FIG_CLUSTERS, FIG_MAPS
]:
    d.mkdir(parents=True, exist_ok=True)

MODELS = BASE_DIR / "models"
MODELS.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "population_est",
    "it_job_density_mean",
    "congestion_index_mean",
    "aqi_mean",
    "built_area_m2_sum",
]

KMEANS_K = [3, 4, 5]

DBSCAN_EPS = 0.5
DBSCAN_MIN_SAMPLES = 5


def load_wards(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"wards file not found: {path}")
    # Accept parquet/geojson/gpkg/csv
    suffix = p.suffix.lower()
    if suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(p)
        # if geometry bytes present, convert
        if "geometry" in df.columns and not isinstance(df.loc[0, "geometry"], object):
            # geopandas can interpret WKB bytes via GeoSeries.from_wkb if necessary,
            # but GeoPandas read_parquet often preserves geometry; attempt direct
            try:
                gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
            except Exception:
                gdf = gpd.GeoDataFrame(df)
        else:
            gdf = gpd.GeoDataFrame(df)
    elif suffix in [".geojson", ".json", ".gpkg", ".gpkg"]:
        gdf = gpd.read_file(p)
    else:
        df = pd.read_csv(p)
        gdf = gpd.GeoDataFrame(df)
    # Ensure geometry column is set
    if "geometry" in gdf.columns:
        try:
            gdf = gdf.set_geometry("geometry")
        except Exception:
            # sometimes geometry is WKB bytes: attempt to convert
            try:
                gdf["geometry"] = gpd.GeoSeries.from_wkb(gdf["geometry"])
                gdf = gdf.set_geometry("geometry")
            except Exception:
                pass
    # enforce CRS
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf


def descriptive_stats(gdf, features):
    df = gdf.copy()
    stats = df[features].describe().transpose()
    stats["missing"] = df[features].isnull().sum()
    stats.to_csv(ART_DESCRIPTIVE / "descriptive_stats.csv")
    # correlation matrix
    corr = df[features].corr()
    corr.to_csv(ART_DESCRIPTIVE / "correlation_matrix.csv")
    # save correlation heatmap
    plt.figure(figsize=(7,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1)
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIAGNOSTICS / "correlation_heatmap.png", dpi=200)
    plt.close()
    logging.info("Computed descriptive statistics and correlation diagnostics")


    # violin + boxplots for key features
    fig, axs = plt.subplots(1, len(features), figsize=(4*len(features),4), squeeze=False)
    for i, f in enumerate(features):
        sns.violinplot(y=df[f], ax=axs[0,i], inner="quartile")
        axs[0,i].set_title(f)
    plt.tight_layout()
    plt.savefig(FIG_DIAGNOSTICS / "violin_features.png", dpi=200)
    plt.close()

    # histograms
    df[features].hist(bins=20, figsize=(4*len(features),4))
    plt.tight_layout()
    plt.savefig(FIG_DIAGNOSTICS / "hist_features.png", dpi=200)
    plt.close()

    return stats, corr


def clustering_analysis(gdf, features, k_list=(3,4,5), db_eps=0.5, db_min_samples=5):
    df = gdf.copy()
    X = df[features].fillna(df[features].median()).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    cluster_results = {}

    # KMeans for multiple k
    kmeans_labels = {}
    for k in k_list:
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = km.fit_predict(Xs)
        kmeans_labels[f"k{k}"] = labels
        # silhouette if >1 cluster
        sil = None
        if len(np.unique(labels)) > 1:
            try:
                sil = silhouette_score(Xs, labels)
            except Exception:
                sil = None
        cluster_results[f"k{k}"] = {"labels": labels, "inertia": float(km.inertia_), "silhouette": sil}
        df[f"kmeans_k{k}"] = labels
        dump(km, MODELS / f"kmeans_k{k}.joblib")
    logging.info(f"KMeans clustering completed for k={k_list}")

    # DBSCAN
    db = DBSCAN(eps=db_eps, min_samples=db_min_samples)
    db_labels = db.fit_predict(Xs)
    df["dbscan"] = db_labels
    cluster_results["dbscan"] = {"labels": db_labels}

    dump(db, MODELS / "dbscan.joblib")

    logging.info("DBSCAN clustering completed")


    # Agglomerative clustering (hierarchical)
    agg = AgglomerativeClustering(n_clusters=4)
    agg_labels = agg.fit_predict(Xs)
    df["hier_4"] = agg_labels
    cluster_results["hier_4"] = {"labels": agg_labels}

    dump(agg, MODELS / "hierarchical_4.joblib")

    logging.info("Hierarchical clustering completed")


    # Save clustering membership
    df_out = df.copy()
    cols_save = ["ward_id"] + [c for c in df_out.columns if c.startswith("kmeans_k") or c in ("dbscan","hier_4")]
    df_out.to_csv(ART_CLUSTERING / "cluster_memberships.csv", index=False)
    # Save kmeans centroids (scaled back)
    centroids = {}
    for k in k_list:
        key = f"k{k}"
        km = KMeans(n_clusters=int(k), random_state=0, n_init=10).fit(Xs)
        cent = scaler.inverse_transform(km.cluster_centers_)
        cent_df = pd.DataFrame(cent, columns=features)
        cent_df.to_csv(ART_CLUSTERING / f"kmeans_centroids_k{k}.csv", index=False)
        centroids[key] = cent_df.to_dict(orient="list")

    # Map clusters on the geometry and save geojsons
    geo = gdf.copy()
    for k in k_list:
        geo[f"kmeans_k{k}"] = df[f"kmeans_k{k}"].values
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        geo.plot(column=f"kmeans_k{k}", categorical=True, legend=True, ax=ax)
        ax.set_title(f"KMeans k={k}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(FIG_CLUSTERS / f"kmeans_k{k}_map.png", dpi=200)
        plt.close()
    # DBSCAN map
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    geo["dbscan"] = df["dbscan"].values
    # cluster colors: noise (-1) separate color
    geo.plot(column="dbscan", categorical=True, legend=True, ax=ax)
    ax.set_title("DBSCAN clusters")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(FIG_CLUSTERS / "dbscan_map.png", dpi=200)
    plt.close()

    # hierarchical dendrogram (for a small set; use a sample)
    try:
        Z = linkage(Xs, method="ward")
        plt.figure(figsize=(10,4))
        dendrogram(Z, no_labels=True, count_sort='ascending', truncate_mode='level', p=6)
        plt.title("Hierarchical clustering dendrogram (truncated)")
        plt.tight_layout()
        plt.savefig(FIG_DIAGNOSTICS / "hier_dendrogram.png", dpi=200)
        plt.close()
    except Exception:
        pass

    with open(ART_CLUSTERING / "clustering_summary.json", "w") as f:
        json.dump(cluster_results, f, indent=2, default=str)


    return cluster_results, df_out
   


def spatial_analysis(gdf, target_vars):
    """
    Computes global Moran's I, local Moran (LISA), and Getis-Ord Gi* (G_Local).
    Saves CSVs and GeoJSON with LISA categories and Gi* z-scores.
    """
    g = gdf.copy()
    # project to geographic/equal-area? Moran works in geographic but recommended weights in planar:
    # We'll compute weights using queen contiguity on the polygon index (no reprojection needed for contiguity).
    # Build weights (queen)
    w = libpysal.weights.contiguity.Queen.from_dataframe(g)
    w.transform = "r"  # row-standardize
    out_summary = {"global_moran": {}, "lisa_files": [], "gi_files": []}

    for var in target_vars:
        vals = g[var].fillna(g[var].median()).values
        # global Moran
        try:
            mi = Moran(vals, w, two_tailed=True)
            out_summary["global_moran"][var] = {"I": float(mi.I), "p_norm": float(mi.p_norm), "z_norm": float(mi.z_norm)}
        except Exception as e:
            out_summary["global_moran"][var] = {"error": str(e)}
        # local Moran (LISA)
        try:
            local = Moran_Local(vals, w)
            # construct categories for LISA (HH, LL, HL, LH, not significant)
            sig = local.p_sim < 0.05
            quadrant = local.q
            lisa_cat = np.array(["NotSig"] * len(g), dtype=object)
            # q: 1 = HH, 2 = LH, 3 = LL, 4 = HL (esda doc)
            mask_hh = (quadrant == 1) & sig
            mask_ll = (quadrant == 3) & sig
            mask_hl = (quadrant == 4) & sig
            mask_lh = (quadrant == 2) & sig
            lisa_cat[mask_hh] = "HH"
            lisa_cat[mask_ll] = "LL"
            lisa_cat[mask_hl] = "HL"
            lisa_cat[mask_lh] = "LH"
            g[f"{var}_lisa_cat"] = lisa_cat
            g[f"{var}_lisa_z"] = local.z_sim
            # save LISA CSV
            g[["ward_id", var, f"{var}_lisa_cat", f"{var}_lisa_z"]].to_csv(ART_SPATIAL / f"lisa_{var}.csv", index=False)
            out_summary["lisa_files"].append(str(ART_SPATIAL / f"lisa_{var}.csv"))
        except Exception as e:
            out_summary["lisa_" + var] = {"error": str(e)}
        # Getis-Ord Gi* (G_Local)
        try:
            gl = G_Local(vals, w, transform="r")
            g[f"{var}_gi_z"] = gl.Zs
            g[f"{var}_gi_p"] = gl.p_sim
            g[["ward_id", var, f"{var}_gi_z", f"{var}_gi_p"]].to_csv(ART_SPATIAL / f"gi_{var}.csv", index=False)
            out_summary["gi_files"].append(str(ART_SPATIAL / f"gi_{var}.csv"))
        except Exception as e:
            out_summary["gi_" + var] = {"error": str(e)}
    logging.info(f"Computed Moran, LISA, Gi* for variable: {var}")

    # Save the enriched geojson
    g.to_file(DERIVED / "wards_spatial_enriched.geojson", driver="GeoJSON")

    # Also save a CSV summary of all LISA / Gi values
    cols_keep = ["ward_id"] + [c for c in g.columns if any(s in c for s in ["_lisa_", "_gi_z", "_gi_p"])]
    g[cols_keep].to_csv(ART_SPATIAL / "spatial_stats_summary.csv", index=False)

    # Create choropleth maps for one variable examples
    for var in target_vars:
        # LISA map
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        catcol = f"{var}_lisa_cat"
        if catcol in g.columns:
            # color mapping
            cmap = {"NotSig":"lightgrey","HH":"red","LL":"blue","HL":"orange","LH":"cyan"}
            g.plot(column=catcol, categorical=True, legend=True, ax=ax, color=g[catcol].map(cmap))
            ax.set_title(f"LISA categories: {var}")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(FIG_MAPS / f"lisa_map_{var}.png", dpi=200)
            plt.close()
        # Gi* z-score map
        zcol = f"{var}_gi_z"
        if zcol in g.columns:
            fig, ax = plt.subplots(1,1,figsize=(8,6))
            g.plot(column=zcol, cmap="RdBu_r", legend=True, ax=ax)
            ax.set_title(f"Getis-Ord Gi* z-score: {var}")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(FIG_MAPS / f"gi_z_map_{var}.png", dpi=200)
            plt.close()
        

    # save summary JSON
    with open(ART_SPATIAL / "spatial_summary.json","w") as f:
        json.dump(out_summary, f, indent=2)

    return out_summary, g


def produce_basic_maps(gdf, var):
    g = gdf.copy()
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    g.plot(column=var, cmap="viridis", legend=True, ax=ax)
    ax.set_title(var)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIG_MAPS / f"map_{var}.png", dpi=200)
    plt.close()


def main():
    logging.info("Starting spatial statistics pipeline")

    gdf = load_wards(path)

    if "ward_id" not in gdf.columns:
        gdf = gdf.reset_index().rename(columns={"index": "ward_id"})

    descriptive_stats(gdf, FEATURES)

    cluster_results, _ = clustering_analysis(
        gdf,
        FEATURES,
        k_list=KMEANS_K,
        db_eps=DBSCAN_EPS,
        db_min_samples=DBSCAN_MIN_SAMPLES
    )

    spatial_analysis(gdf, FEATURES)

    gdf.to_file(DERIVED / "wards_spatial_clustering_enriched.geojson", driver="GeoJSON")

    logging.info("Spatial statistics pipeline completed successfully")

if __name__ == "__main__":
    main()
