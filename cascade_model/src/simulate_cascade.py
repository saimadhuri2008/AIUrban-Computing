#!/usr/bin/env python3
"""
simulate_cascade.py

Deterministic cascade failure simulator for urban indicators.

Core principles:
- Stress accumulates monotonically
- Functional state F(w,i,t) ∈ {0,1}
- Failed indicators do NOT propagate stress
- Cascades arise from load redistribution, not random shocks
- Designed for extracting logical failure chains (not prediction)

Mathematics:
Let S(w,i,t) = accumulated stress
Let F(w,i,t) ∈ {0,1} = functional state
Let A_ij = propagation matrix (stress j → i)

Exogenous stress:
  ΔS_exo(w,i,t) = max( sev(w,i,t) − sev(w,i,t−1), 0 )

Endogenous stress:
  ΔS_endo(w,i,t) = α · Σ_j A_ij · F(w,j,t−1) · S(w,j,t−1)

Stress update:
  S(w,i,t) = (1 − δ)·S(w,i,t−1) + ΔS_exo + ΔS_endo

Failure rule (monotonic):
  if S(w,i,t) ≥ θ_i  →  F(w,i,t) = 0
  else              →  F(w,i,t) = F(w,i,t−1)

Inputs:
- detection CSV (sev_* columns)
- propagation_matrix.joblib
- indicator thresholds θ_i

Outputs:
- functional_state.joblib
- stress_history.joblib
- cascade_metrics.csv
- failure_times.csv
- collapse_curve.png
- cascade_summary.json
"""

import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List

# =============================
# PATHS
# =============================
BASE_DIR = Path("cascade_model")
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RESULTS_DIR = BASE_DIR / "results/cascade"
REPORTS_DIR = BASE_DIR / "reports"

for d in [ARTIFACTS_DIR, RESULTS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SUMMARY = BASE_DIR / "summary"
SUMMARY.mkdir(parents=True, exist_ok=True)

# =============================
# DATA LOADING
# =============================
def load_detection_csv(path: Path):
    df = pd.read_csv(path, parse_dates=["date"])
    sev_cols = [c for c in df.columns if c.startswith("sev_")]
    if not sev_cols:
        raise ValueError("No sev_ columns found")

    wards = sorted(df["ward_id"].unique())
    dates = sorted(df["date"].unique())
    indicators = [c.replace("sev_", "") for c in sev_cols]

    W, T, M = len(wards), len(dates), len(indicators)

    ward_idx = {w: i for i, w in enumerate(wards)}
    date_idx = {d: i for i, d in enumerate(dates)}

    S = np.zeros((W, T, M))
    for _, r in df.iterrows():
        S[ward_idx[r["ward_id"]],
          date_idx[r["date"]], :] = r[sev_cols].values

    return S, wards, [d.to_pydatetime() for d in dates], indicators

# =============================
# CASCADE SIMULATION
# =============================
def simulate_cascade(
    sev_baseline: np.ndarray,
    A: np.ndarray,
    thresholds: np.ndarray,
    alpha: float,
    stress_decay: float
):
    W, T, M = sev_baseline.shape

    F = np.ones((W, T, M))        # functional state
    S = np.zeros((W, T, M))       # accumulated stress

    # initial condition
    S[:, 0, :] = sev_baseline[:, 0, :]
    F[:, 0, :] = (S[:, 0, :] < thresholds).astype(float)

    for t in range(1, T):
        ΔS_exo = np.maximum(sev_baseline[:, t, :] - sev_baseline[:, t-1, :], 0)

        for w in range(W):
            active_stress = S[w, t-1, :] * F[w, t-1, :]
            ΔS_endo = alpha * (A @ active_stress)

            S[w, t, :] = (1 - stress_decay) * S[w, t-1, :] + ΔS_exo[w] + ΔS_endo

        F[:, t, :] = np.minimum(F[:, t-1, :], (S[:, t, :] < thresholds).astype(float))

    return F, S

# =============================
# METRICS
# =============================
def compute_metrics(F, S, dates):
    W, T, M = F.shape
    rows = []
    prev_failed = 0

    for t in range(T):
        failed = (F[:, t, :] == 0)
        total_failed = failed.sum()
        velocity = total_failed - prev_failed
        prev_failed = total_failed

        rows.append({
            "date": dates[t],
            "fraction_failed": total_failed / (W * M),
            "failed_wards": failed.any(axis=1).sum(),
            "mean_stress": S[:, t, :].mean(),
            "cascade_velocity": velocity
        })

    return pd.DataFrame(rows)

# =============================
# FAILURE TIMES (FOR CHAINS)
# =============================
def compute_failure_times(F, wards, indicators):
    W, T, M = F.shape
    ft = np.full((W, M), np.nan)

    for w in range(W):
        for i in range(M):
            t_fail = np.where(F[w, :, i] == 0)[0]
            if len(t_fail) > 0:
                ft[w, i] = t_fail[0]

    return pd.DataFrame(ft, index=wards, columns=indicators)

# =============================
# PLOTTING
# =============================
def plot_collapse(metrics, outpath):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(metrics["date"], metrics["fraction_failed"], lw=2)
    ax.fill_between(metrics["date"], metrics["fraction_failed"], alpha=0.25)
    ax.set_ylabel("Fraction Failed")
    ax.set_xlabel("Time")
    ax.set_title("Cascade Collapse Curve")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

# =============================
# MAIN
# =============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_csv", default=BASE_DIR /"results/failure_detection/detection_results.csv")
    parser.add_argument("--prop_matrix", default=BASE_DIR/"artifacts/propagation_matrix.joblib")
    parser.add_argument("--thresholds", default=BASE_DIR/"artifacts/thresholds/cascade_thresholds.json")

    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--stress_decay", type=float, default=0.02)
    args = parser.parse_args()

    sev, wards, dates, indicators = load_detection_csv(Path(args.det_csv))
    A = joblib.load(args.prop_matrix)

    with open(args.thresholds) as f:
        thresh_map = json.load(f)

    missing = set(indicators) - set(thresh_map.keys())
    if missing:
        raise ValueError(f"Missing thresholds for indicators: {missing}")

    thresholds = np.array([thresh_map[i] for i in indicators], dtype=float)


    F, S = simulate_cascade(sev, A, thresholds, args.alpha, args.stress_decay)

    metrics = compute_metrics(F, S, dates)
    failure_times = compute_failure_times(F, wards, indicators)


    joblib.dump(F, ARTIFACTS_DIR / "functional_state.joblib")
    joblib.dump(S, ARTIFACTS_DIR / "stress_history.joblib")

    metrics.to_csv(RESULTS_DIR / "cascade_metrics.csv", index=False)
    failure_times.to_csv(RESULTS_DIR / "failure_times.csv")

    plot_collapse(metrics, REPORTS_DIR / "collapse_curve.png")

    summary = {
        "alpha": args.alpha,
        "stress_decay": args.stress_decay,
        "final_fraction_failed": float(metrics.iloc[-1]["fraction_failed"]),
        "timesteps": len(dates)
    }

    with open(SUMMARY / "cascade_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Cascade simulation completed.")

if __name__ == "__main__":
    main()
