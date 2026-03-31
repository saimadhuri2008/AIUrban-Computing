#!/usr/bin/env python3
"""
phase2e_2f.py

Phase 2E (Robustness & Sensitivity) and Phase 2F (Results summary + visualization).
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

import logging
BASE_DIR = Path("statistical_inference")
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=BASE_DIR / "logs" / "robustness_results.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

log = logging.getLogger(__name__)




path = BASE_DIR / "data/derived/wards_enriched.geojson"

ART_DESCRIPTIVE = BASE_DIR / "artifacts/descriptive"
ART_SPATIAL = BASE_DIR / "artifacts/spatial"
ART_CLUSTERING = BASE_DIR / "artifacts/clustering"
ART_REG = BASE_DIR / "artifacts/regression"
ART_ROBUSTNESS = BASE_DIR / "artifacts/robustness"

FIG_DIAGNOSTICS = BASE_DIR / "figures/diagnostics"
FIG_CLUSTERS = BASE_DIR / "figures/clusters"
FIG_MAPS = BASE_DIR / "figures/maps"

SUMMARY = BASE_DIR / "summary"

DERIVED = BASE_DIR / "data/derived"

for d in [
    ART_DESCRIPTIVE, ART_SPATIAL, ART_CLUSTERING,ART_ROBUSTNESS,ART_REG,
    FIG_DIAGNOSTICS, FIG_CLUSTERS, FIG_MAPS
]:
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Helper functions
# -----------------------------
def load_inputs(path, phase2d_dir):
    wards = gpd.read_file(path)
    phase2d_dir = Path(phase2d_dir)
    results = {}
    for fname in ['psm_att.json','psm_matched.csv','iv_2sls_congestion.txt',
                  'poisson_outflow.txt','placebo_atts.npy','bootstrap_atts.npy','phase2d_summary.json']:
        p = phase2d_dir / fname
        if p.exists():
            if p.suffix=='.json':
                results[fname] = json.loads(p.read_text())
            elif p.suffix=='.npy':
                results[fname] = np.load(p, allow_pickle=True)
            else:
                results[fname] = p.read_text()
    return wards, results

def ols_with_variance(df, formula, cluster_col=None, hac_lags=4):

    model = smf.ols(formula=formula, data=df).fit()
    out = {'coef': model.params, 'summary': model.summary().as_text(), 'model': model}
    out['white'] = model.get_robustcov_results(cov_type='HC3').summary().as_text()
    if cluster_col is not None and cluster_col in df.columns:
        try:
            cl = df[cluster_col]
            out['clustered'] = model.get_robustcov_results(cov_type='cluster', groups=cl).summary().as_text()
        except Exception as e:
            out['clustered'] = f"clustered failed: {e}"
    try:
        out['hac'] = model.get_robustcov_results(cov_type='HAC', maxlags=hac_lags).summary().as_text()
    except Exception as e:
        out['hac'] = f"HAC failed: {e}"
    return out

def influence_diagnostics(model):
    inf = model.get_influence()
    cooks = inf.cooks_distance[0]
    leverage = inf.hat_matrix_diag
    resid = model.resid
    return pd.DataFrame({'cooks_d': cooks, 'leverage': leverage, 'resid': resid})

def missingness_sensitivity(df, formula, target_cols, niter=100):
    rng = np.random.RandomState(0)
    coef_records = []
    for i in range(niter):
        tmp = df.copy()
        frac = rng.uniform(0.05, 0.20)
        for c in target_cols:
            mask = rng.rand(len(tmp)) < frac
            tmp.loc[mask, c] = np.nan
        try:
            mod_cc = smf.ols(formula=formula, data=tmp.dropna()).fit()
            cc_coef = mod_cc.params
        except Exception:
            cc_coef = pd.Series(dtype=float)
        imputer = IterativeImputer(random_state=i, max_iter=20)
        numeric = tmp.select_dtypes(include=[np.number])
        imputed = pd.DataFrame(imputer.fit_transform(numeric), columns=numeric.columns, index=numeric.index)
        tmp2 = tmp.copy()
        tmp2[numeric.columns] = imputed
        try:
            mod_imp = smf.ols(formula=formula, data=tmp2).fit()
            imp_coef = mod_imp.params
        except Exception:
            imp_coef = None
        coef_records.append({
            'cc_coef': cc_coef.to_json() if isinstance(cc_coef, pd.Series) else None,
            'imp_coef': imp_coef.to_json() if imp_coef is not None else None
        })
    return coef_records

def permutation_test_ATT(df, treat_col, outcome_col, covariates, nperm=1000):
    treated = df[df[treat_col]==1]
    control = df[df[treat_col]==0]
    if len(treated)==0 or len(control)==0:
        log.info(f"Skipping permutation ATT: no treated/control units for '{treat_col}'")
        return {'obs_att': None, 'perm_atts': [], 'p_emp': None}
    # Propensity score
    X = df[covariates].fillna(0).values
    logit = LogisticRegression(max_iter=200).fit(X, df[treat_col].values)
    df2 = df.copy()
    df2['pscore'] = logit.predict_proba(X)[:,1]
    treated = df2[df2[treat_col]==1]
    control = df2[df2[treat_col]==0]
    # 1-to-1 NN matching
    nbrs = NearestNeighbors(n_neighbors=1).fit(control[['pscore']])
    dist, idx = nbrs.kneighbors(treated[['pscore']])
    matched_control_idx = control.iloc[idx.flatten()].index
    obs_att = treated[outcome_col].mean() - control.loc[matched_control_idx][outcome_col].mean()
    # Permutations
    perm_atts = []
    rng = np.random.RandomState(0)
    for i in range(nperm):
        perm = df2.copy()
        perm[treat_col] = rng.permutation(perm[treat_col].values)
        try:
            logit2 = LogisticRegression(max_iter=200).fit(perm[covariates].fillna(0).values, perm[treat_col].values)
            perm['pscore'] = logit2.predict_proba(perm[covariates].fillna(0).values)[:,1]
            treated_p = perm[perm[treat_col]==1]
            control_p = perm[perm[treat_col]==0]
            if len(control_p)==0 or len(treated_p)==0:
                perm_atts.append(0.0)
                continue
            nbrs2 = NearestNeighbors(n_neighbors=1).fit(control_p[['pscore']])
            dist2, idx2 = nbrs2.kneighbors(treated_p[['pscore']])
            matched_ctrl_idx = control_p.iloc[idx2.flatten()].index
            att_p = treated_p[outcome_col].mean() - control_p.loc[matched_ctrl_idx][outcome_col].mean()
            perm_atts.append(att_p)
        except Exception:
            perm_atts.append(0.0)
    perm_atts = np.array(perm_atts)
    p_emp = (np.sum(np.abs(perm_atts) >= np.abs(obs_att)) + 1) / (len(perm_atts) + 1) if obs_att is not None else None
    return {'obs_att': float(obs_att) if obs_att is not None else None, 'perm_atts': perm_atts.tolist(), 'p_emp': float(p_emp) if p_emp is not None else None}

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w') as f:
        json.dump(obj, f, indent=2, default=lambda x: x.tolist() if hasattr(x,'tolist') else str(x))

def plot_maps(gdf,columns):
    for c in columns:
        if c in gdf.columns:
            fig, ax = plt.subplots(1,1, figsize=(7,7))
            gdf.plot(column=c, ax=ax, legend=True, cmap='viridis', linewidth=0.1, edgecolor='white')
            ax.set_title(c)
            ax.axis('off')
            fig.savefig(FIG_MAPS / f"map_{c}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main():
    log.info("=" * 80)
    log.info("PHASE 2E–2F (ROBUSTNESS + SUMMARY) STARTED")

    wards_path = path
    phase2d_dir = BASE_DIR / "artifacts/casual"


    wards, phase2d_results = load_inputs(wards_path, phase2d_dir)
    log.info("Loaded wards GeoDataFrame: %d rows, %d columns", *wards.shape)

    log.info(
        "Loaded Phase 2D artifacts: %s",
        list(phase2d_results.keys())
    )



    
    df = pd.DataFrame(wards.drop(columns=[c for c in wards.columns if c=='geometry']))

    ols_formula = "congestion_index_mean ~ population_est + built_area_m2_sum + income_index_mean + it_job_density_mean + aqi_mean"
    ols_res = ols_with_variance(df, ols_formula, cluster_col='ward_num_x', hac_lags=4)
    log.info("Running OLS with robustness checks")
    log.info("OLS formula: %s", ols_formula)

    for k in ['summary','white','clustered','hac']:
        val = ols_res.get(k, f'no {k}')
        with open(ART_REG / f"ols_{k}.txt", "w") as f:

            f.write(val)

    infl = influence_diagnostics(ols_res['model'])
    infl.to_csv(ART_ROBUSTNESS / "influence_with_flag.csv", index=False)
    cooks_thr = 4.0 / len(df)
    infl['outlier_cook'] = infl['cooks_d'] > cooks_thr

    log.info(
        "Influence diagnostics | Cook threshold=%.5f | flagged=%d",
        cooks_thr,
        infl["outlier_cook"].sum()
    )

    log.info("Running missingness MCAR sensitivity (niter=60)")

    sens = missingness_sensitivity(df, ols_formula, target_cols=['population_est','built_area_m2_sum','income_index_mean'], niter=60)
    save_json(sens, ART_ROBUSTNESS / "sensitivity/missingness_mcar_results.json")

    # Permutation ATT
    if "high_it" not in df.columns:
        log.warning("Skipping permutation ATT: 'high_it' treatment not found")
        perm = {"error": "high_it not found"}
    else:
        perm = permutation_test_ATT(
            df,
            treat_col="high_it",
            outcome_col="congestion_index_mean",
            covariates=["income_index_mean","population_est","built_area_m2_sum"],
            nperm=400
        )
        save_json(perm, ART_ROBUSTNESS / "sensitivity/att_permutation.json")

    # Residual diagnostics plots
    model = ols_res['model']
    fig = plt.figure(figsize=(10,6))
    sm.graphics.plot_regress_exog(model, 'income_index_mean', fig=fig)
    fig.savefig(FIG_DIAGNOSTICS / "regress_income_index.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    sns.histplot(model.resid, ax=ax[0], kde=True)
    sm.qqplot(model.resid, line='45', ax=ax[1])
    fig.savefig(FIG_MAPS / "residuals_hist_qq.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    plot_cols = ['congestion_index_mean','population_est','it_job_density_mean','aqi_mean','built_area_m2_sum','income_index_mean']
    plot_maps(wards,plot_cols)

    summary = {'ols': {'params': model.params.to_dict(), 'pvalues': model.pvalues.to_dict(), 'rsquared_adj': getattr(model,'rsquared_adj',None)},
               'diagnostics': {'cooks_threshold': cooks_thr,'n_obs': len(df)}}
    save_json(summary, SUMMARY / "robustness_summary.json")

    coef_df = pd.DataFrame({'coef': model.params,'se': model.bse,'pval': model.pvalues})
    coef_df.to_csv(ART_REG / "ols_coef_table.csv")


    wards2 = wards.copy().reset_index(drop=True)
    wards2['cooks_d'] = infl['cooks_d'].values
    wards2['influential'] = infl['outlier_cook'].values
    wards2.to_file(DERIVED/ "wards_phase2_final.geojson", driver='GeoJSON')

    log.info("robustness check and results summary finished")

if __name__ == "__main__":
    main()
