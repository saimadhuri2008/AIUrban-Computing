#!/usr/bin/env python3
"""
monte_carlo_cascade.py

Run Monte-Carlo ensemble of realistic monotonic cascade simulations,
aggregate failure probabilities and save summary outputs.

Outputs:
 - montecarlo/severity_prob_matrix_<ts>.joblib  (W x T float32)
 - montecarlo/collapse_prob_ts_<ts>.csv
 - montecarlo/top_wards_prob_<ts>.csv
 - montecarlo/summary_<ts>.json

Author: Assistant (adapted for your project)
"""
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from datetime import datetime
import os
import sys
import math
import logging

BASE_DIR = Path("cascade_model")


RESULTS_DIR = BASE_DIR / "results" / "montecarlo_cascade" 
ARTIFACTS_DIR = BASE_DIR / "artifacts" 

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY = BASE_DIR / "summary"
LOGS_DIR = BASE_DIR / "logs"

for d in [ARTIFACTS_DIR, RESULTS_DIR,SUMMARY]:
    d.mkdir(parents=True, exist_ok=True)

log_file = LOGS_DIR / "montecarlo_cascade.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

logger = logging.getLogger("montecarlo_cascade")

# ----------------------------
# Utility: load detection CSV
# ----------------------------
def load_detection(det_csv):
    df = pd.read_csv(det_csv, parse_dates=["date"], infer_datetime_format=True)
    sev_cols = [c for c in df.columns if c.startswith("sev_")]
    if not sev_cols:
        raise RuntimeError("No sev_ columns found in detection CSV.")
    wards = sorted(df["ward_id"].unique())
    dates = sorted(df["date"].unique())
    W = len(wards); T = len(dates); M = len(sev_cols)
    sev_baseline = np.zeros((W, T, M), dtype=np.float32)
    ward_to_idx = {w:i for i,w in enumerate(wards)}
    date_to_idx = {pd.Timestamp(d).to_pydatetime(): i for i,d in enumerate(dates)}
    # build mapping for dates that may have timezone info
    date_lookup = {pd.Timestamp(d): i for i,d in enumerate(dates)}
    for _, row in df.iterrows():
        wi = ward_to_idx[row["ward_id"]]
        ti = date_lookup[pd.Timestamp(row["date"])]
        sev_baseline[wi, ti, :] = np.asarray([row[c] for c in sev_cols], dtype=np.float32)
    return sev_baseline, wards, dates, sev_cols

# ----------------------------
# Core simulation (vectorized per-ward loop)
# ----------------------------
def run_realistic_sim_small(sev_baseline, P,
                            alpha, compound_coef, cap, fail_level,
                            trend_rate, season_amp,
                            shock_prob0, shock_growth, shock_scale,
                            random_seed=0):
    """
    A compact, deterministic + stochastic, monotonic simulation.
    Returns: history (W, T, M) float32
    """
    np.random.seed(int(random_seed) & 0xFFFFFFFF)
    W, T, M = sev_baseline.shape
    history = np.zeros((W, T, M), dtype=np.float32)
    history[:,0,:] = sev_baseline[:,0,:].astype(np.float32)

    months = np.arange(T) % 12
    for t in range(1, T):
        prev = history[:, t-1, :]  # W x M
        frac = t / max(1, T-1)
        trend_multiplier = 1.0 + trend_rate * frac
        month = months[t]
        season_mult = 1.0 + season_amp * math.sin(2*math.pi*(month/12.0))
        shock_prob = float(min(0.5, shock_prob0 + shock_growth * frac))

        # propagated effect per ward: (s @ P.T)  -> shape (W, M)
        propagated = (prev @ P.T) * trend_multiplier

        # compound: per-ward scalar times vector
        severe_count = (prev >= fail_level).sum(axis=1).astype(np.float32)  # shape (W,)
        compound = compound_coef * severe_count * (1.0 + 0.5 * frac)            # shape (W,)
        compound_vec = compound.reshape((W,1))                                 # W x 1

        # baseline increases: only positive deltas from sev_baseline
        baseline_inc = sev_baseline[:, t, :] - sev_baseline[:, t-1, :]
        baseline_inc = np.maximum(baseline_inc, 0.0) * (0.5 + 0.5 * frac)

        # shocks: sample per-ward boolean then per-indicator noise
        shock_flags = np.random.rand(W) < shock_prob
        # noise in [0, shock_scale)*trend_multiplier
        shocks = np.random.rand(W, M).astype(np.float32) * shock_scale * trend_multiplier
        shocks[~shock_flags, :] = 0.0

        cand_increase = alpha * propagated + compound_vec + baseline_inc + shocks
        candidate = prev + cand_increase
        candidate = np.clip(candidate, 0.0, cap)
        # enforce monotonic (no recovery)
        candidate = np.maximum(prev, candidate)
        history[:, t, :] = candidate.astype(np.float32)
    return history

# ----------------------------
# Single sim worker (used by Parallel)
# ----------------------------
def simulate_one(idx,
                 sev_baseline, P,
                 fail_level,
                 param_sampler,
                 perturb_P_scale=0.0):
    """
    param_sampler: function(index, rng) -> dict of parameters
    perturb_P_scale: gaussian noise std applied elementwise to P (small)
    Returns: boolean failure matrix W x T (uint8 or bool)
    """
    rng = np.random.RandomState(abs(hash((idx, int(time_seed := (idx<<16) ^ 0xABCDEF))) ) % (2**32))
    params = param_sampler(idx, rng)
    # optionally perturb P
    if perturb_P_scale and perturb_P_scale > 0.0:
        P_ = P + rng.normal(0.0, perturb_P_scale, size=P.shape).astype(np.float32)
    else:
        P_ = P
    hist = run_realistic_sim_small(
        sev_baseline, P_,
        alpha=params["alpha"],
        compound_coef=params["compound"],
        cap=params["cap"],
        fail_level=fail_level,
        trend_rate=params["trend_rate"],
        season_amp=params["season_amp"],
        shock_prob0=params["shock_prob0"],
        shock_growth=params["shock_growth"],
        shock_scale=params["shock_scale"],
        random_seed=int(rng.randint(0, 2**31))
    )
    # ward failing at t if any indicator >= fail_level
    fail_bool = (hist >= fail_level).any(axis=2)  # W x T boolean
    # convert to uint8 for memory & easy summation
    return fail_bool.astype(np.uint8)

# ----------------------------
# Parameter sampler (default)
# ----------------------------
def default_param_sampler_factory(param_priors, rng_global_seed=0):
    """
    param_priors: dict with keys -> (low, high) or (mean, std) depending on 'type'
    returns a function(idx, rng) -> params dict
    We'll sample uniformly within ranges for robustness.
    """
    def sampler(idx, rng):
        p = {}
        for k,v in param_priors.items():
            if v.get("dist","uniform") == "uniform":
                lo, hi = v["range"]
                p[k] = float(rng.uniform(lo, hi))
            elif v.get("dist","normal"):
                mu, sd = v["range"]
                p[k] = float(max(0.0, rng.normal(mu, sd)))
            else:
                lo, hi = v["range"]
                p[k] = float(rng.uniform(lo, hi))
        return p
    return sampler

# ----------------------------
# Main orchestrator
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_csv", default=BASE_DIR/"results/failure_detection/detection_results.csv", help="detection CSV with sev_ columns")
    parser.add_argument("--prop", default=BASE_DIR/"artifacts/propagation_matrix.joblib", help="propagation matrix (.joblib)")
    parser.add_argument("--outdir", default=None, help="output dir")
    parser.add_argument("--n_sims", type=int, default=200, help="number of Monte-Carlo sims")
    parser.add_argument("--n_jobs", type=int, default=4, help="parallel jobs")
    parser.add_argument("--fail_level", type=float, default=1.0)
    parser.add_argument("--perturb_P_scale", type=float, default=0.0, help="std of gaussian noise to perturb P")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = RESULTS_DIR

    # load inputs
    logging.info("Loading detection CSV...")
    sev_baseline, wards, dates, sev_cols = load_detection(args.det_csv)
    logging.info(f"Loaded sev baseline: W={sev_baseline.shape[0]}, T={sev_baseline.shape[1]}, M={sev_baseline.shape[2]}")
    P = joblib.load(args.prop)
    if P.shape[0] != P.shape[1]:
        raise RuntimeError("Propagation matrix must be square M x M")
    M = P.shape[0]
    if sev_baseline.shape[2] != M:
        logging.info("Warning: M mismatch: sev M =", sev_baseline.shape[2], "P M =", M, file=sys.stderr)
    # define param priors (tune to the region found by sweep)
    param_priors = {
        "alpha": {"dist":"uniform", "range": (0.02, 0.12)},       # small influence
        "compound": {"dist":"uniform", "range": (0.0, 0.06)},
        "cap": {"dist":"uniform", "range": (1.5, 2.5)},
        "trend_rate": {"dist":"uniform", "range": (0.25, 0.9)},   # how much fragility grows
        "season_amp": {"dist":"uniform", "range": (0.02, 0.10)},
        "shock_prob0": {"dist":"uniform", "range": (0.005, 0.03)},
        "shock_growth": {"dist":"uniform", "range": (0.01, 0.06)},
        "shock_scale": {"dist":"uniform", "range": (0.08, 0.32)}
    }
    sampler = default_param_sampler_factory(param_priors, rng_global_seed=args.seed)

    # run sims in parallel, accumulating counts
    W, T, Mb = sev_baseline.shape
    counts = np.zeros((W, T), dtype=np.uint32)

    logging.info(f"Running {args.n_sims} simulations with {args.n_jobs} jobs ...")
    # prepare parallel job list indices
    sim_indices = list(range(args.n_sims))

    # wrapper for joblib to avoid huge pickles: pass small items only by reference (sev_baseline and P are picklable)
    results = Parallel(n_jobs=args.n_jobs, prefer="threads")(
        delayed(simulate_one)(i, sev_baseline, P, args.fail_level, sampler, args.perturb_P_scale)
        for i in sim_indices
    )

    # sum boolean arrays to counts
    for i, arr in enumerate(results):
        counts += arr.astype(np.uint32)

    # compute probabilities
    prob = counts.astype(np.float32) / float(max(1, args.n_sims))
    ts_frac = (prob.sum(axis=0) / float(W))  # fraction of wards failing at each timestep (averaged across wards)

    ts_df = pd.DataFrame({"date": pd.to_datetime(dates), "frac_fail_mean": ts_frac})
    ts_csv = RESULTS_DIR / "collapse_prob_ts.csv"
    ts_df.to_csv(ts_csv, index=False)

    # save severity probability matrix
    sev_prob_path = ARTIFACTS_DIR / "severity_prob_matrix_montecarlo.joblib"
    joblib.dump({"prob": prob, "wards": wards, "dates": [str(d) for d in dates], "sev_cols": sev_cols}, sev_prob_path, compress=3)

    # top wards by mean failure probability (averaged across timesteps)
    mean_prob = prob.mean(axis=1)
    top_idx = np.argsort(-mean_prob)[:100]
    top_df = pd.DataFrame({
        "ward_index": top_idx,
        "ward_name": [wards[i] for i in top_idx],
        "mean_prob": mean_prob[top_idx]
    })
    top_df.to_csv(RESULTS_DIR / f"top_wards_prob.csv", index=False)

    summary = {
        "n_sims": args.n_sims,
        "n_wards": int(W),
        "n_timesteps": int(T),
        "params_priors": param_priors,
        "perturb_P_scale": float(args.perturb_P_scale),
        "fail_level": float(args.fail_level),
        "sev_cols": sev_cols,
        "created": datetime.utcnow().isoformat() + "Z"
    }
    with open(SUMMARY / f"summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    logging.info(f"[SAVED] severity probability matrix at {sev_prob_path}")
    logging.info(f"[SAVED] collapse time series at {ts_csv}")
    logging.info(f"[SAVED] top wards at {outdir}")
    logging.info("Done.")

if __name__ == "__main__":
    main()
