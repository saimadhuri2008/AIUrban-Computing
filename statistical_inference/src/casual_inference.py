#!/usr/bin/env python3
"""
phase2d_final.py

Phase 2D — Causal analysis (final): synthesise missing causal variables, run PSM/IV/DiD/spatial lag,
and save outputs to results/phase2d_final.

Usage:
    python src/models/statistical/phase2d_final.py \
        --wards data/processed/master/phase2/wards_phase2_enriched.geojson \
        --outdir results/phase2d_final \
        --seed 0

Dependencies:
  pip install pandas geopandas numpy scipy matplotlib seaborn statsmodels scikit-learn linearmodels libpysal networkx
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from linearmodels.iv import IV2SLS
import libpysal
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path("statistical_inference")

DATA_DERIVED = BASE_DIR / "data/derived"
ART_CASUAL   = BASE_DIR / "artifacts/casual"
ART_SYNTH    = BASE_DIR / "artifacts/synthetic"
FIG_DAGS     = BASE_DIR / "figures/dags"
FIG_DIAG     = BASE_DIR / "figures/diagnostics"
LOG_DIR      = BASE_DIR / "logs"
SUMMARY_DIR  = BASE_DIR / "summary"
ART_SPATIAL = BASE_DIR / "artifacts/spatial"

for d in [ART_CASUAL,ART_SPATIAL, ART_SYNTH, FIG_DAGS, FIG_DIAG, LOG_DIR, SUMMARY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

import logging

logging.basicConfig(
    filename=LOG_DIR / "causal_inference.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger(__name__)


INPUT_WARDS = DATA_DERIVED / "wards_enriched.geojson"

def load_wards(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
        try:
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        except Exception:
            gdf = gpd.GeoDataFrame(df)
    elif p.suffix.lower() in [".geojson", ".json", ".gpkg", ".gpkg"]:
        gdf = gpd.read_file(p)
    else:
        df = pd.read_csv(p)
        gdf = gpd.GeoDataFrame(df)
    return gdf

def safe_get(df, col, default):
    if col in df.columns:
        return df[col].values
    else:
        return np.full(len(df), default)

def synth_variables(gdf, seed=0):
    """Create synthetic variables needed for causal analysis.
    Design choices:
      - Correlate new vars plausibly with existing vars so regressions make sense.
      - Use small random noise to avoid perfect collinearity.
    Returns augmented GeoDataFrame (non-destructive).
    """

   

    rng = np.random.RandomState(seed)
    df = pd.DataFrame(gdf.drop(columns=[c for c in gdf.columns if c == 'geometry']))

    n = len(df)
    # ensure essential existing variables exist by fallback to sensible values
    pop = safe_get(df, 'population_est', 30000).astype(float)
    built = safe_get(df, 'built_area_m2_sum', np.maximum(1e3, pop * 2.0)).astype(float)
    it_job = safe_get(df, 'it_job_density_mean', 60.0).astype(float)
    income = safe_get(df, 'income_index_mean', 0.05).astype(float)
    
    # Standardized versions for building correlated synthetic variables
    scaler = StandardScaler()
    base_stack = np.vstack([np.log1p(pop), np.log1p(built), it_job, income]).T
    base_scaled = scaler.fit_transform(base_stack)

    # F2: land-use + mobility mismatch variables
    # landuse_mix: higher where built is high and income moderate — synthetic diversity index
    landuse_mix = (0.3*base_scaled[:,1] - 0.1*base_scaled[:,3] + 0.2*rng.randn(n))
    landuse_mix = (landuse_mix - landuse_mix.min()) / (landuse_mix.max() - landuse_mix.min() + 1e-9)

    # distance_to_jobs: shorter in high IT areas, longer where pop high & built low
    distance_to_jobs = np.clip(5.0 - 2.0*(it_job - it_job.mean())/it_job.std() + 1.0*(pop/pop.mean()) + rng.randn(n)*0.5, 0.5, 20.0)

    # job_accessibility: inverse of distance_to_jobs, scaled 0-1
    job_accessibility = 1.0 / (1.0 + distance_to_jobs)
    job_accessibility = (job_accessibility - job_accessibility.min()) / (job_accessibility.max() - job_accessibility.min() + 1e-9)

    # commute_time (minutes): longer where distance_to_jobs large and car dependence high
    # car_dependence proxy: more car use where distance to transit is high and income high
    distance_to_transit = np.clip(1.0 + (pop/pop.mean())*0.2 + rng.randn(n)*0.5 + (1.0 - landuse_mix)*2.0, 0.1, 10.0)
    car_dependence = (income*5.0) + 0.3*distance_to_transit + rng.randn(n)*0.3
    car_dependence = (car_dependence - car_dependence.min()) / (car_dependence.max() - car_dependence.min() + 1e-9)
    commute_time = np.clip(10.0 + distance_to_jobs*4.0 + (1.0-car_dependence)*5.0 + rng.randn(n)*5.0, 5.0, 180.0)

    # F3: climate cascade variables
    rainfall_shock = rng.gamma(2.0, 1.0, size=n)  # positive skew
    # water_disruption increases with rainfall_shock but also with poor built infra (low built density)
    built_density_proxy = built / (pop + 1.0)
    water_disruption = np.clip(0.2*rainfall_shock + 0.1*(1.0/(1.0+built_density_proxy/pop.mean())) + rng.randn(n)*0.05, 0.0, 1.0)

    # substation failure random but more likely where built is high AND population high (stress)
    substation_failure_prob = sigmoid((np.log1p(pop) + np.log1p(built))/ (np.log1p(pop).mean()*2.0) + rng.randn(n)*0.5)
    substation_failure = rng.binomial(1, p=np.clip(substation_failure_prob, 0.01, 0.5))
    blackout = substation_failure  # simplified cascade

    # mobility failure increases with water_disruption and blackout
    mobility_failure = np.clip(0.6*water_disruption + 0.6*blackout + rng.randn(n)*0.1, 0.0, 1.0)

    # F4: redesign scenario variables (sector design)
    # create a 5-sector categorization (0..4) and design effects
    sector_design = rng.randint(0, 5, size=n)
    # sector effect on mean commute: some sectors (mixed-use) reduce commute
    sector_effect_map = {0: 1.0, 1: 0.9, 2: 0.8, 3: 1.1, 4: 0.7}
    sector_commute_multiplier = np.array([sector_effect_map[s] for s in sector_design])
    mean_commute = commute_time * sector_commute_multiplier

    # access equity: higher when job accessibility high and landuse_mix high
    access_equity = 0.5*job_accessibility + 0.5*landuse_mix + rng.randn(n)*0.05
    access_equity = (access_equity - access_equity.min()) / (access_equity.max() - access_equity.min() + 1e-9)

    # F5: BAU forecast variables
    population_growth = np.clip(0.01 + 0.001*(pop/pop.mean()) + rng.randn(n)*0.002, -0.01, 0.1)
    vehicle_growth = np.clip(0.02 + 0.005*car_dependence + rng.randn(n)*0.01, 0.0, 0.2)
    electricity_demand = pop*0.0002 * (1.0 + population_growth*10.0) + built*1e-6 + rng.randn(n)*10.0
    water_demand = pop*0.0003 * (1.0 + population_growth*8.0) + rng.randn(n)*5.0
    built_area_growth = built * (0.01 + population_growth*0.1 + rng.randn(n)*0.01)

    # create columns into df
    synth = {
        'landuse_mix': landuse_mix,
        'distance_to_jobs': distance_to_jobs,
        'job_accessibility': job_accessibility,
        'distance_to_transit': distance_to_transit,
        'car_dependence': car_dependence,
        'commute_time': commute_time,
        'rainfall_shock': rainfall_shock,
        'water_disruption': water_disruption,
        'substation_failure': substation_failure,
        'blackout': blackout,
        'mobility_failure': mobility_failure,
        'sector_design': sector_design,
        'mean_commute': mean_commute,
        'access_equity': access_equity,
        'population_growth': population_growth,
        'vehicle_growth': vehicle_growth,
        'electricity_demand': electricity_demand,
        'water_demand': water_demand,
        'built_area_growth': built_area_growth
    }
    log.info(
        "Synthetic variables generated | seed=%d | n=%d | vars=%s",
        seed,
        len(gdf),
        list(synth.keys())
    )
    for k, v in synth.items():
        df[k] = v

    # include back geometry
    gdf_out = gpd.GeoDataFrame(df, geometry=gdf.geometry.values, crs=gdf.crs)
    return gdf_out

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
def save_gdf_csv(gdf):
    """
    Save Phase 2D synthetic wards data.
    Fixed output location: artifacts/synthetic/
    """
    gdf.to_file(
        ART_SYNTH / "wards_phase2d_synthetic.geojson",
        driver="GeoJSON"
    )

    gdf.drop(columns="geometry").to_csv(
        ART_SYNTH / "wards_phase2d_synthetic.csv",
        index=False
    )

    log.info("Synthetic Phase 2D ward data saved (GeoJSON + CSV)")


def plot_dag_summaries():
    # Create simple images to represent DAG names (text nodes). Minimal but informative.
    import matplotlib.pyplot as plt
    dags = {
        "F1_stress": ["population_est", "income_index_mean", "built_area_m2_sum", "congestion_index_mean", "aqi_mean"],
        "F2_landuse_mismatch": ["landuse_mix", "distance_to_jobs", "commute_time", "job_accessibility", "car_dependence"],
        "F3_climate_cascade": ["rainfall_shock", "water_disruption", "mobility_failure", "substation_failure", "blackout"],
        "F4_redesign": ["sector_design", "mean_commute", "access_equity", "congestion_index_mean"],
        "F5_bau_forecast": ["population_growth", "vehicle_growth", "electricity_demand", "water_demand"]
    }
    for name, nodes in dags.items():
        plt.figure(figsize=(6,4))
        G=nx.DiGraph()
        for n in nodes:
            G.add_node(n)
        # Add a couple edges to visualise links (simple)
        edges = []
        if name=="F2_landuse_mismatch":
            edges=[("landuse_mix","commute_time"),("distance_to_jobs","commute_time"),("commute_time","job_accessibility")]
        elif name=="F3_climate_cascade":
            edges=[("rainfall_shock","water_disruption"),("water_disruption","mobility_failure"),("substation_failure","blackout")]
        elif name=="F4_redesign":
            edges=[("sector_design","mean_commute"),("sector_design","access_equity"),("mean_commute","congestion_index_mean")]
        elif name=="F5_bau_forecast":
            edges=[("population_growth","electricity_demand"),("vehicle_growth","congestion_index_mean")]
        else:
            edges=[("population_est","congestion_index_mean"),("income_index_mean","congestion_index_mean")]
        G.add_edges_from(edges)
        pos=nx.spring_layout(G, seed=1)
        nx.draw(G, pos, with_labels=True, node_size=900, node_color='lightblue', arrowsize=12)
        plt.title(name)
        plt.tight_layout()
        plt.savefig(FIG_DAGS / f"dag_{name}.png", dpi=150)
        plt.close()

def propensity_score_matching_and_att(df,treat_col='landuse_mix_high', covariates=None, outcome='congestion_index_mean'):
    

    """Do PSM: logistic pscore on covariates, NN matching and ATT"""
    df = df.copy()
    if covariates is None:
        covariates = ['income_index_mean','population_est','built_area_m2_sum','aqi_mean']
    X = df[covariates].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    y = df[treat_col].values
    if len(np.unique(y)) < 2:
        raise ValueError("Treatment has <2 unique values, cannot run PSM")
    logit = LogisticRegression(max_iter=500)
    logit.fit(Xs, y)
    pscore = logit.predict_proba(Xs)[:,1]
    df['pscore'] = pscore

    log.info(
        "PSM overlap | min_pscore=%.3f max_pscore=%.3f",
        df["pscore"].min(),
        df["pscore"].max()
    )
    
    log.info(
        "PSM groups | treated=%d control=%d",
        (df[treat_col] == 1).sum(),
        (df[treat_col] == 0).sum()
    )

    # matching
    treated_idx = np.where(y==1)[0]
    control_idx = np.where(y==0)[0]
    if len(control_idx) == 0 or len(treated_idx) == 0:
        raise ValueError("No treated or no control units for PSM")
    nbrs = NearestNeighbors(n_neighbors=1).fit(pscore[control_idx].reshape(-1,1))
    dists, inds = nbrs.kneighbors(pscore[treated_idx].reshape(-1,1))
    matched_control = control_idx[inds.flatten()]
    matched_idx = np.concatenate([treated_idx, matched_control])
    matched_df = df.iloc[matched_idx].reset_index(drop=True)
    att = matched_df[matched_df[treat_col]==1][outcome].mean() - matched_df[matched_df[treat_col]==0][outcome].mean()
    out = {
        'att': float(att),
        'treated_mean': float(matched_df[matched_df[treat_col]==1][outcome].mean()),
        'control_mean': float(matched_df[matched_df[treat_col]==0][outcome].mean()),
        'n_treated': int(len(treated_idx)),
        'n_control_matched': int(len(matched_control))
    }
    (ART_CASUAL / "psm_matched.csv").write_text(matched_df.to_csv(index=False))
    with open(ART_CASUAL / "psm_att.json","w") as f:
        json.dump(out, f, indent=2)
    # save pscore distribution plot
    plt.figure(figsize=(6,3))
    sns.kdeplot(df.loc[df[treat_col]==1,'pscore'], label='treated')
    sns.kdeplot(df.loc[df[treat_col]==0,'pscore'], label='control')
    plt.title("Propensity score distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIAG / "pscore_distribution.png", dpi=150)
    plt.close()
    return out

def two_stage_iv(df,outcome='congestion_index_mean', endog='car_dependence', instrument='distance_to_transit', exog=None):
    if exog is None:
        exog = ['population_est','built_area_m2_sum','income_index_mean','aqi_mean']
    # prepare data - drop rows with NA in required vars
    cols = [outcome, endog, instrument] + exog
    data = df[cols].dropna().copy()
    if data.shape[0] < 10:
        raise ValueError("Too few observations for IV")
    # first stage: endog ~ instrument + exog
    X1 = sm.add_constant(data[[instrument] + exog])
    fs = sm.OLS(data[endog], X1).fit(cov_type='HC3')
    # compute F-stat for instrument: use the instrument coefficient t-test or partial F
    try:
        fstat = fs.f_test(np.eye(len(X1.columns))[1:2])  # not perfect but try
    except Exception:
        fstat = None

    if fstat is not None:
        log.info("IV first-stage F-stat = %.2f", float(fstat.fvalue))

    # second stage via linearmodels IV2SLS
    try:
        formula = f"{outcome} ~ 1 + " + " + ".join(exog) + f" + [{endog} ~ {instrument}]"
        iv_res = IV2SLS.from_formula(formula, data).fit(cov_type='robust')
        with open(ART_CASUAL / "iv_2sls_summary.txt","w") as f:
            f.write(iv_res.summary.as_text())
    except Exception as e:
        iv_res = None
        with open(ART_CASUAL / "iv_2sls_summary.txt","w") as f:
            f.write("IV failed: " + str(e))
    # save first-stage
    with open(ART_CASUAL / "iv_first_stage.txt","w") as f:
        f.write(fs.summary().as_text())
    return {'first_stage': fs, 'iv_res': iv_res, 'first_stage_rows': data.shape[0]}

def diff_in_diff_synthetic(df, treat_selector=None):
    """
    Build a simple synthetic panel: years 2018-2023, treatment occurs in 2021 for treated wards.
    treat_selector: boolean mask selecting treated wards (same length as df)
    """
    yrs = np.arange(2018, 2024)
    n = len(df)
    if treat_selector is None:
        # treat top 20% by distance_to_transit as 'metro open' treatment
        treat_selector = df['distance_to_transit'] < np.percentile(df['distance_to_transit'], 20)
    rows = []
    rng = np.random.RandomState(1)
    for i, row in df.reset_index().iterrows():
        for y in yrs:
            base = row.get('commute_time', 30.0)
            # small upward BAU trend
            trend = (y - 2018) * (1.0 + 0.02 * (row.get('population_growth', 0.01)*100))
            # if treated and y>=2021, reduce commute by 10% (treatment effect)
            treat = 1 if (treat_selector.iloc[i] and y >= 2021) else 0
            commute = base + trend + rng.randn()*2.0 - 5.0*treat  # 5 minute improvement post-treatment
            rows.append({
                'ward_id': row.get('ward_id', f'ward_{i+1}'),
                'year': int(y),
                'commute_time': float(commute),
                'treated_group': int(treat_selector.iloc[i])
            })
    panel = pd.DataFrame(rows)
    log.warning(
        "DiD uses synthetic panel with assumed parallel trends (documented assumption)"
    )

    # DiD regression commute_time ~ treated_group * post + year FE
    panel['post'] = (panel['year'] >= 2021).astype(int)
    panel['did'] = panel['treated_group'] * panel['post']
    mod = smf.ols("commute_time ~ treated_group + post + did", data=panel).fit(cov_type='HC3')
    with open(ART_CASUAL / "did_commute_summary.txt","w") as f:
        f.write(mod.summary().as_text())
    panel.to_csv(ART_CASUAL / "did_panel_commute.csv", index=False)
    return mod

def spatial_lag_test(gdf, y_col='congestion_index_mean', x_cols=None):
    if x_cols is None:
        x_cols = ['income_index_mean','it_job_density_mean','built_area_m2_sum','aqi_mean','population_est']
    df = gdf.dropna(subset=[y_col] + x_cols).reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("Too few observations for spatial lag")
   

    # create weights via libpysal queen contiguity if polygons; otherwise kNN on centroids
    try:
        w = libpysal.weights.Queen.from_dataframe(df)
    except Exception:
        coords = np.vstack([df.geometry.centroid.x.values, df.geometry.centroid.y.values]).T
        w = libpysal.weights.KNN.from_array(coords, k=5)
    w.transform = 'r'
    # spatially lag y
    y = df[y_col].values
    Wy = w.sparse.dot(y)
    df['Wy'] = Wy
    X = sm.add_constant(df[x_cols + ['Wy']])
    model = sm.OLS(y, X).fit(cov_type='HC3')
    with open(ART_SPATIAL / "spatial_lag_summary.txt","w") as f:
        f.write(model.summary().as_text())
    log.info(
        "Spatial weights | type=%s | n=%d",
        type(w).__name__,
        len(df)
    )
    return model

def explain_and_save_summaries(psm_out, iv_out, did_out, spatial_out):
    summary = {
        'psm': psm_out,
        'iv_first_stage_rows': iv_out.get('first_stage_rows', None),
        'did_coef': None,
        'spatial_coef_Wy': None
    }
    if did_out is not None:
        summary['did_coef'] = float(did_out.params.get('did', np.nan))
    if spatial_out is not None:
        summary['spatial_coef_Wy'] = float(spatial_out.params.get('Wy', np.nan))
    with open(SUMMARY_DIR / "casual_inference_summary.json","w") as f:
        json.dump(summary, f, indent=2)

def main():
    log.info("=" * 80)
    log.info("CAUSAL INFERENCE STARTED")

    
    SEED = 0
    log.info("Random seed fixed at %d for reproducibility", SEED)


    gdf = load_wards(INPUT_WARDS)
    log.info("Loaded %d wards", len(gdf))

    gdf_synth = synth_variables(gdf, seed=SEED)
    save_gdf_csv(gdf_synth)

    plot_dag_summaries()

    gdf_synth["landuse_mix_high"] = (
        gdf_synth["landuse_mix"] >= np.percentile(gdf_synth["landuse_mix"], 70)
    ).astype(int)

    log.info(
        "Treatment split | treated=%d control=%d",
        gdf_synth["landuse_mix_high"].sum(),
        (1 - gdf_synth["landuse_mix_high"]).sum()
    )

    try:
        psm_out = propensity_score_matching_and_att(gdf_synth)
        log.info("PSM ATT = %.4f", psm_out["att"])
    except Exception:
        log.exception("PSM failed")
        psm_out = {"error": "PSM failed"}

    try:
        iv_out = two_stage_iv(gdf_synth)
        log.info("IV first-stage rows = %d", iv_out["first_stage_rows"])
    except Exception:
        log.exception("IV failed")
        iv_out = {"error": "IV failed"}

    try:
        did_out = diff_in_diff_synthetic(gdf_synth)
        log.info("DiD coef = %.4f", did_out.params.get("did", np.nan))
    except Exception:
        log.exception("DiD failed")
        did_out = None

    try:
        spatial_out = spatial_lag_test(gdf_synth)
        log.info("Spatial lag Wy coef = %.4f", spatial_out.params.get("Wy", np.nan))
    except Exception:
        log.exception("Spatial lag failed")
        spatial_out = None

    gdf_synth.to_file(ART_CASUAL / "wards_phase2d_final.geojson", driver="GeoJSON")
    gdf_synth.drop(columns="geometry").to_csv(
        ART_CASUAL / "wards_phase2d_final.csv", index=False
    )

    explain_and_save_summaries(psm_out, iv_out, did_out, spatial_out)

    log.info("PHASE 2D — CAUSAL INFERENCE COMPLETED")
    log.info("=" * 80)

if __name__ == "__main__":
    main()
