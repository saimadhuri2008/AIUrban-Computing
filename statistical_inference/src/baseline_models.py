#!/usr/bin/env python3
"""
phase2c_baseline.py

Phase 2C — Statistical Baseline Modeling

Usage:
    python src/models/statistical/phase2c_baseline.py \
        --wards data/processed/master/phase2/wards_phase2_enriched.geojson \
        --outdir results/statistical_baseline \
        --ts-dir data/processed/master/phase2/timeseries  # optional

Produces:
 - OLS / GLM / Poisson / NegBin model summaries (text files)
 - Spatial-Lag (approx) and Spatial-Durbin (approx) models
 - Residual Moran I checks
 - VIF, condition number diagnostics
 - Time-series STL + ARIMA + ETS for supplied timeseries
 - Plots (residuals, ppc, decomposition) and CSV outputs for tables

Dependencies:
  pip install pandas geopandas numpy scipy matplotlib seaborn statsmodels scikit-learn libpysal esda
"""

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler
from libpysal.weights import Queen, lag_spatial
from esda.moran import Moran
from scipy import stats
import logging
from joblib import dump


warnings.filterwarnings("ignore")
sns.set(style="whitegrid", rc={"figure.dpi": 150})

# -------------------------------------------------------------------
# Fixed configuration (research pipeline – no CLI args)
# -------------------------------------------------------------------

BASE_DIR = Path("statistical_inference")

INPUT_WARDS = BASE_DIR / "data/derived/wards_enriched.geojson"

ART_DESCRIPTIVE = BASE_DIR / "artifacts/descriptive"
ART_REGRESSION  = BASE_DIR / "artifacts/regression"
ART_SPATIAL     = BASE_DIR / "artifacts/spatial"
ART_TS          = BASE_DIR / "artifacts/timeseries"

FIG_DIAG = BASE_DIR / "figures/diagnostics"
FIG_TS   = BASE_DIR / "figures/timeseries"

SUMMARY = BASE_DIR / "summary"

for d in [
    ART_DESCRIPTIVE,
    ART_REGRESSION,
    ART_SPATIAL,
    ART_TS,
    FIG_DIAG,
    FIG_TS
]:
    d.mkdir(parents=True, exist_ok=True)

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "baseline_models.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logging.info("=" * 80)
logging.info("Baseline statistical modeling — START")
logging.info(f"Input wards: {INPUT_WARDS}")

def load_wards(path):
    p = Path(path)
    if p.suffix.lower() in [".parquet"]:
        df = pd.read_parquet(p)
        try:
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")
        except Exception:
            gdf = gpd.GeoDataFrame(df)
    else:
        gdf = gpd.read_file(p)
    return gdf

def prepare_df(gdf, required):
    # drop geometry into a normal DataFrame but keep index
    df = pd.DataFrame(gdf.drop(columns=[c for c in gdf.columns if c == 'geometry']))
    # ensure required exist
    for r in required:
        if r not in df.columns:
            df[r] = np.nan
    df = df.dropna(subset=[required[0]])  # ensure at least primary outcome exists
    return df

def compute_vif(df, features):
    X = df[features].copy().dropna()
    X = sm.add_constant(X)
    vifs = {}
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        try:
            vifs[col] = float(variance_inflation_factor(X.values, i))
        except Exception:
            vifs[col] = np.nan
    return vifs

def cond_number(df, features):
    X = df[features].copy().dropna().values
    # standardize
    Xs = StandardScaler().fit_transform(X)
    u, s, vh = np.linalg.svd(Xs, full_matrices=False)
    cond = float(s.max()/s.min())
    return cond

def fit_ols(df, formula, cov_type="HC3"):
    model = smf.ols(formula=formula, data=df).fit(cov_type=cov_type)
    return model
    

def fit_glm_poisson(df, formula):
    model = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()
    return model


def fit_neg_binomial(df, formula):
    model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial()).fit()
    return model

def spatial_weights_from_gdf(gdf):
    # Builds Queen contiguity weights; returns W (libpysal) and row-standardized weights matrix via lag_spatial
    gdf = gdf.reset_index(drop=True)
    w = Queen.from_dataframe(gdf)
    w.transform = 'r'  # row-standardize
    return w

def add_spatial_lags(df, gdf, features):
    # Compute Wy (spatial lag of y) and WX (spatial lag of features) and add to df
    w = spatial_weights_from_gdf(gdf)
    # ensure index alignment
    df2 = df.reset_index(drop=True)
    wy = lag_spatial(w, df2['congestion_index_mean'].values) if 'congestion_index_mean' in df2.columns else None
    if wy is not None:
        df2['Wy_congestion'] = wy
    for f in features:
        if f in df2.columns:
            df2[f"Wx_{f}"] = lag_spatial(w, df2[f].values)
    # save weights info
    with open(ART_SPATIAL / "spatial_weights_meta.json", "w") as f:
        json.dump({"n": w.n, "neighbors_summary": {str(k): len(w.neighbors[k]) for k in w.neighbors}}, f, indent=2)
    return df2, w

def spatial_lag_model_via_ols(df, formula_without_Wy, w, outcome='congestion_index_mean', cov_type='HC3'):
    """
    Approximate spatial lag by adding Wy as regressor (Wy calculated externally).
    Returns fitted model and diagnostics.
    """
    df = df.copy()
    if 'Wy_congestion' not in df.columns:
        df['Wy_congestion'] = lag_spatial(w, df[outcome].values)
    formula = formula_without_Wy + " + Wy_congestion"
    mod = smf.ols(formula=formula, data=df).fit(cov_type=cov_type)
    logging.info("Running spatial lag models")
    return mod

def spatial_durbin_via_ols(df, formula_main, wx_list, cov_type='HC3'):
    """
    Spatial Durbin via including WX terms explicitly in the regressor set
    """
    # formula_main example: "y ~ x1 + x2 + x3"
    # append Wx columns
    for wx in wx_list:
        if wx not in df.columns:
            continue
    # build the formula automatically using columns present
    all_vars = df.columns.tolist()
    # keep only variables from formula_main and any Wx_* present
    # we'll just create a design matrix and run OLS manually for stability
    y_var = formula_main.split("~")[0].strip()
    exog_vars = [v.strip() for v in formula_main.split("~")[1].split("+")]
    exog_vars = [v for v in exog_vars if v in df.columns]
    # include Wx_*
    exog_vars += [c for c in df.columns if c.startswith("Wx_")]
    X = sm.add_constant(df[exog_vars].astype(float).dropna())
    y = df.loc[X.index, y_var].astype(float)
    mod = sm.OLS(y, X).fit(cov_type=cov_type)
    logging.info("Running spatial Durbin models")
    return mod

def residual_moran_on_model(mod, gdf, outcome='congestion_index_mean'):
    # Input mod is statsmodels fitted model. Need residuals aligned with geometry rows.
    resid = mod.resid
    # ensure length matches gdf
    if len(resid) != len(gdf):
        # try aligning by index
        raise ValueError("Residual length does not match geometry length; ensure the model used full dataset without drops.")
    w = spatial_weights_from_gdf(gdf)
    mi = Moran(resid.values, w)
    return {"I": float(mi.I), "p_sim": float(mi.p_sim), "z": float(mi.z_norm), "n": int(mi.n)}

def save_text(path, text):
    with open(path, "w", encoding="utf8") as f:
        f.write(text)


def timeseries_baselines(ts_dir, aggregate=True):
    """
    ts_dir: directory with CSVs. Each CSV expects columns: date, ward_id (optional), value
      Example filenames: electricity_ward_YYYY.csv or electricity_city.csv
    If aggregate=True, script will aggregate across wards to city-level for ARIMA.
    """
   
    ts_dir = Path(ts_dir)
    if not ts_dir.exists():
        print("No timeseries dir found:", ts_dir)
        return
    for csv in ts_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv, parse_dates=['date'])
        except Exception:
            continue
        # detect value column
        val_cols = [c for c in df.columns if c not in ('date','ward_id','ward','ward_code')]
        if not val_cols:
            continue
        val = val_cols[0]
        if 'ward_id' in df.columns and not aggregate:
            # optionally do per-ward ARIMA (but this can be slow) -> instead do city-level by default
            groups = df.groupby('ward_id')
            # skip detailed per-ward loop here for speed
            series = groups[val].sum(axis=0).resample('M').sum() if False else None
        # aggregate to city-level
        ts = df.set_index('date')[val].resample('M').sum()
        ts = ts.fillna(method='ffill').dropna()
        if len(ts) < 24:
            # short series: use ETS only
            try:
                ets = ExponentialSmoothing(ts, trend='add', seasonal=None).fit()
                preds = ets.predict(start=ts.index[0], end=ts.index[-1])
                plt.figure(figsize=(8,3))
                ts.plot(label='observed')
                preds.plot(label='ETS_fitted')
                plt.title(f"{csv.name} ETS fit")
                plt.legend()
                plt.tight_layout()
                plt.savefig(FIG_TS / f"{csv.stem}_ets.png", dpi=150)
                plt.close()
            except Exception as e:
                print("ETS failed", csv, e)
            continue
        # STL decomposition
        try:
            stl = STL(ts, period=12, robust=True).fit()
            fig = stl.plot()
            fig.set_size_inches(10,6)
            plt.tight_layout()
            plt.savefig(FIG_TS / f"{csv.stem}_stl.png", dpi=150)
            plt.close()
        except Exception as e:
            print("STL failed", csv, e)
        # ARIMA automatic order selection: use simple heuristics (p,d,q)=(1,1,1) as baseline
        try:
            arima = ARIMA(ts, order=(1,1,1)).fit()
            with open(ART_TS / f"{csv.stem}_arima.txt","w") as f:
                f.write(arima.summary().as_text())
            # forecast 12 months
            fc = arima.get_forecast(steps=12)
            fc_mean = fc.predicted_mean
            fc_ci = fc.conf_int()
            fc.to_frame().to_csv(ART_TS / f"{csv.stem}_arima_forecast.csv")
            plt.figure(figsize=(8,3))
            ts.plot(label='observed')
            fc_mean.plot(label='forecast')
            plt.fill_between(fc_ci.index, fc_ci.iloc[:,0], fc_ci.iloc[:,1], alpha=0.2)
            plt.title(f"{csv.name} ARIMA forecast")
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIG_TS / f"{csv.stem}_arima_forecast.png", dpi=150)
            plt.close()
        except Exception as e:
            print("ARIMA failed for", csv, e)

def main():
    logging.info("Baseline modeling started")

    gdf = load_wards(INPUT_WARDS)

    # primary variables we expect from Phase2: congestion_index_mean, total_outflow_sum, population_est, income_index_mean, it_job_density_mean, built_area_m2_sum, aqi_mean
    required = ['congestion_index_mean','total_outflow_sum','population_est',
                'income_index_mean','it_job_density_mean','built_area_m2_sum','aqi_mean']
    df = prepare_df(gdf, required)

    # quick descriptive save
    df.describe().to_csv(SUMMARY/ "baseline_descriptive_summary.csv")

    # compute VIF and condition number
    features = ['income_index_mean','it_job_density_mean','built_area_m2_sum','congestion_index_mean','aqi_mean','population_est']
    vifs = compute_vif(df, features)
    cond = cond_number(df, features)
    with open(ART_REGRESSION / "vif_and_cond.txt","w") as f:
        f.write("VIFs:\n")
        json.dump(vifs, f, indent=2)
        f.write("\n\nCondition number (SVD on standardized X):\n")
        f.write(str(cond))

    # baseline OLS - model for log(population) as in your earlier run
    df['log_population_est'] = np.log(df['population_est'].replace(0, np.nan)).fillna(0)
    ols_formula = "log_population_est ~ income_index_mean + it_job_density_mean + built_area_m2_sum + congestion_index_mean + aqi_mean"
    logging.info("Fitting OLS population baseline model")
    ols_mod = fit_ols(df, ols_formula, cov_type="HC3")
    dump(ols_mod, MODELS_DIR / "ols_population.joblib")

    save_text(
        ART_REGRESSION / "ols_population_baseline_summary.txt",
        ols_mod.summary().as_text()
    )


    # residual spatial autocorrelation check
    rm = None
    try:
        rm = residual_moran_on_model(ols_mod, gdf, outcome='log_population_est')
        with open(ART_SPATIAL / "ols_residual_moran.json","w") as f:
            json.dump(rm, f, indent=2)
    except Exception as e:
        print("Residual Moran failed:", e)

    # GLM Gaussian for travel-time-like outcome: if travel_time variable present use it else use congestion_index_mean as proxy
    travel_var = 'travel_time_mean' if 'travel_time_mean' in df.columns else 'congestion_index_mean'
    glm_gauss_formula = f"{travel_var} ~ income_index_mean + it_job_density_mean + built_area_m2_sum + population_est + aqi_mean"
    try:
        logging.info("Fitting GLM Gaussian travel model")
        glm_gauss = smf.glm(
            formula=glm_gauss_formula,
            data=df,
            family=sm.families.Gaussian()
        ).fit()
        dump(glm_gauss, MODELS_DIR / "glm_gaussian_travel.joblib")

        save_text(
            ART_REGRESSION / "glm_gaussian_travel_summary.txt",
            glm_gauss.summary().as_text()
        )

    except Exception as e:
        save_text(ART_REGRESSION / "glm_gaussian_travel_summary.txt", f"GLM Gaussian failed: {e}")

    # Poisson regression for counts - example: total_outflow_sum
    pois_formula = "total_outflow_sum ~ income_index_mean + it_job_density_mean + built_area_m2_sum + population_est + aqi_mean"
    try:
        logging.info("Fitting Poisson outflow model")
        pois_mod = fit_glm_poisson(df, pois_formula)
        dump(pois_mod, MODELS_DIR / "poisson_outflow.joblib")
        save_text(
            ART_REGRESSION / "poisson_outflow_baseline_summary.txt", 
            pois_mod.summary().as_text()
        )
    except Exception as e:
        save_text(ART_REGRESSION / "poisson_outflow_baseline_summary.txt", f"Poisson failed: {e}")

    # Negative Binomial for utility failure counts; assume 'utility_failures' column exists; fallback to total_outflow_sum if not
    nb_outcome = 'utility_failures' if 'utility_failures' in df.columns else 'total_outflow_sum'
    nb_formula = f"{nb_outcome} ~ income_index_mean + it_job_density_mean + built_area_m2_sum + population_est + aqi_mean"
    try:
        logging.info("Fitting Negative Binomial model")
        nb_mod = fit_neg_binomial(df, nb_formula)
        dump(nb_mod, MODELS_DIR / "neg_binomial.joblib")
        save_text(ART_REGRESSION / "neg_binomial_summary.txt", nb_mod.summary().as_text())
    except Exception as e:
        save_text(ART_REGRESSION / "neg_binomial_summary.txt", f"Negative Binomial failed: {e}")

    # Spatial augmentations: create W and lagged variables
    df_with_wx, w = add_spatial_lags(df, gdf, ['income_index_mean','it_job_density_mean','built_area_m2_sum','aqi_mean','population_est'])
    df_with_wx.to_csv(ART_SPATIAL / "wards_with_spatial_lags.csv", index=False)

    # Spatial Lag (approx) by adding Wy to OLS for congestion
    try:
        logging.info("Running spatial lag model")
        sl_mod = spatial_lag_model_via_ols(df_with_wx, "congestion_index_mean ~ income_index_mean + it_job_density_mean + built_area_m2_sum + population_est + aqi_mean", w, outcome='congestion_index_mean')
        dump(sl_mod, MODELS_DIR / "spatial_lag_congestion.joblib")
        save_text(ART_SPATIAL, "spatial_lag_approx_congestion.txt", sl_mod.summary().as_text())
        sl_rm = residual_moran_on_model(sl_mod, gdf, outcome='congestion_index_mean')
        with open(ART_SPATIAL / "spatial_lag_resid_moran.json","w") as f:
            json.dump(sl_rm, f, indent=2)
    except Exception as e:
        save_text(ART_SPATIAL / "spatial_lag_approx_congestion.txt", f"Spatial-lag approximation failed: {e}")

    # Spatial Durbin (approx) by including Wx_* terms - build OLS design and fit
    try:
        logging.info("Running spatial Durbin model")
        sd_mod = spatial_durbin_via_ols(df_with_wx, "congestion_index_mean ~ income_index_mean + it_job_density_mean + built_area_m2_sum + population_est + aqi_mean", wx_list=['Wx_income_index_mean','Wx_it_job_density_mean','Wx_built_area_m2_sum'], cov_type='HC3')
        dump(sd_mod, MODELS_DIR / "spatial_durbin_congestion.joblib")
        save_text(ART_SPATIAL / "spatial_durbin_approx_congestion.txt", sd_mod.summary().as_text())
        sd_rm = None
        try:
            sd_rm = residual_moran_on_model(sd_mod, gdf, outcome='congestion_index_mean')
            with open(ART_SPATIAL / "spatial_durbin_resid_moran.json","w") as f:
                json.dump(sd_rm, f, indent=2)
        except Exception:
            pass
    except Exception as e:
        save_text(ART_SPATIAL / "spatial_durbin_approx_congestion.txt", f"Spatial-Durbin approximation failed: {e}")

    TS_DIR = BASE_DIR / "data/derived/timeseries"

    if TS_DIR.exists():
        logging.info(f"Running time-series baselines from {TS_DIR}")
        timeseries_baselines(TS_DIR)
    else:
        logging.info("No time-series directory found; skipping TS baselines")


    # Save model coefficients to CSV for easy reporting
    def save_params_to_csv(mod, filename):
        dfp = pd.DataFrame({
            "coef": mod.params,
            "se": mod.bse,
            "pval": mod.pvalues
        })
        dfp.to_csv(ART_REGRESSION / filename)


    save_params_to_csv(ols_mod, "ols_population")
    try:
        save_params_to_csv(glm_gauss, "glm_gauss_travel")
    except Exception:
        pass
    try:
        save_params_to_csv(pois_mod, "poisson_outflow")
    except Exception:
        pass
    try:
        save_params_to_csv(nb_mod, "neg_binomial")
    except Exception:
        pass
    try:
        save_params_to_csv(sl_mod, "spatial_lag")
    except Exception:
        pass
    try:
        save_params_to_csv(sd_mod, "spatial_durbin")
    except Exception:
        pass

    # Residual plots for OLS
    try:
        fig, ax = plt.subplots(1,2,figsize=(10,3))
        sm.graphics.plot_regress_exog(ols_mod, "income_index_mean", fig=fig)
        plt.tight_layout()
        plt.savefig(FIG_DIAG / "ols_diagnostics_income.png", dpi=150)
        plt.close()
    except Exception:
        pass

    # Save quick correlation heatmap for features used
    try:
        corr = df[features].corr()
        plt.figure(figsize=(6,4))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Feature correlation - Phase2C")
        plt.tight_layout()
        plt.savefig(FIG_DIAG / "baseline_correlation_heatmap.png", dpi=150)
        plt.close()
    except Exception:
        pass

    # Save a JSON master summary
    master = {
        "vif": vifs,
        "condition_number": cond,
        "ols_adjR2": float(ols_mod.rsquared_adj),
        "ols_aic": float(ols_mod.aic),
    }
    if rm:
        master["ols_residual_moran"] = rm
    try:
        with open(SUMMARY / "baselinemodels_master_summary.json","w") as f:
            json.dump(master, f, indent=2)
    except Exception:
        pass


    logging.info(f"Baseline modeling complete. Base directory: {BASE_DIR}")


if __name__ == "__main__":
    main()
