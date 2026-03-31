#!/usr/bin/env python3
"""
extract_failure_chains.py

Extracts temporal failure sequences from cascade simulation results.

Definition (strict, research-safe):
A failure sequence j → i exists if and only if:

1. F[w, t-1, j] == 0   (indicator j already failed)
2. F[w, t, i]   == 0   (indicator i is failed at time t)
3. F[w, t-1, i] == 1   (indicator i failed for the first time at t)
4. A[i, j] > ε         (non-trivial dependency exists)

Notes:
- Observational only (NO causal claims)
- Single timestep lag (t-1 → t)
- Sparse outputs are valid scientific outcomes
"""

import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# =====================================================
# PATHS & LOGGING (DETERMINISTIC)
# =====================================================
BASE_DIR = Path("cascade_model")
RESULTS_DIR = BASE_DIR / "results" / "failure_chains"
LOGS_DIR = BASE_DIR / "logs" / "failure_chains"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY = BASE_DIR / "summary"
SUMMARY.mkdir(parents=True, exist_ok=True)

log_file = LOGS_DIR / "extract_failure_chains.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("failure_chain_extraction")

# =====================================================
# CORE EXTRACTION
# =====================================================
def extract_failure_sequences(
    F, A, indicators, wards, dates, weight_threshold=0.05
):
    """
    Extract observed failure sequences from cascade simulation.

    Returns:
        List of dicts with keys:
        source, target, ward, timestep, date, weight
    """
    W, T, M = F.shape
    sequences = []
    total_new_failures = 0

    logger.info(f"Extracting failure sequences (weight_threshold={weight_threshold})")
    logger.info(f"Data shape: {W} wards × {T} timesteps × {M} indicators")

    for t in range(1, T):
        for w in range(W):
            for i in range(M):

                # NEW failure at time t
                if F[w, t, i] == 0 and F[w, t-1, i] == 1:
                    total_new_failures += 1

                    for j in range(M):
                        if (
                            F[w, t-1, j] == 0 and
                            A[i, j] > weight_threshold
                        ):
                            sequences.append({
                                "source": indicators[j],
                                "target": indicators[i],
                                "ward": wards[w],
                                "timestep": t,
                                "date": str(dates[t]),
                                "weight": float(A[i, j])
                            })

    logger.info(f"Total new failures observed: {total_new_failures}")
    logger.info(f"Total failure sequences extracted: {len(sequences)}")

    if not sequences:
        logger.warning("No failure sequences detected.")
        logger.warning("Possible reasons:")
        logger.warning("- Weak or absent cascades")
        logger.warning("- weight_threshold too high")
        logger.warning("- Sparse propagation matrix")

    return sequences

# =====================================================
# AGGREGATION
# =====================================================
def aggregate_sequences(sequences, min_frequency=2):
    """
    Aggregate failure sequences across wards and time.
    """
    if not sequences:
        return pd.DataFrame()

    df = pd.DataFrame(sequences)

    agg = (
        df.groupby(["source", "target"], as_index=False)
          .agg(
              frequency=("ward", "count"),
              avg_weight=("weight", "mean"),
              first_timestep=("timestep", "min"),
              last_timestep=("timestep", "max")
          )
    )

    agg = agg[agg["frequency"] >= min_frequency]
    agg = agg.sort_values("frequency", ascending=False).reset_index(drop=True)

    return agg

# =====================================================
# MAIN
# =====================================================
def main():
    parser = argparse.ArgumentParser(
        description="Extract observed failure chains from cascade simulation"
    )
    parser.add_argument("--functional_state", default=BASE_DIR/"artifacts/functional_state.joblib",
                        help="Path to functional_state.joblib (W×T×M)")
    parser.add_argument("--prop_matrix", default=BASE_DIR / "artifacts/propagation_matrix.joblib",
                        help="Path to propagation matrix")
    parser.add_argument("--detection_csv", default=BASE_DIR/"results/failure_detection/detection_results.csv",
                        help="Detection CSV (for wards, dates, indicators)")
    parser.add_argument("--weight_threshold", type=float, default=0.05,
                        help="Minimum dependency weight ε (default: 0.05)")
    parser.add_argument("--min_frequency", type=int, default=2,
                        help="Minimum frequency to retain a chain (default: 2)")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("FAILURE CHAIN EXTRACTION")
    logger.info("=" * 60)

    # Load data
    F = joblib.load(args.functional_state)
    A = joblib.load(args.prop_matrix)

    det_df = pd.read_csv(args.detection_csv, parse_dates=["date"])
    sev_cols = [c for c in det_df.columns if c.startswith("sev_")]
    indicators = [c.replace("sev_", "") for c in sev_cols]
    wards = sorted(det_df["ward_id"].unique())
    dates = sorted(det_df["date"].unique())

    W, T, M = F.shape
    logger.info(f"Functional state shape: {F.shape}")
    logger.info(f"Indicators: {indicators}")

    if M != len(indicators):
        raise ValueError("Indicator mismatch between F and detection CSV")

    # Extract sequences
    sequences = extract_failure_sequences(
        F, A, indicators, wards, dates,
        weight_threshold=args.weight_threshold
    )

    # Save raw sequences (always)
    raw_path = RESULTS_DIR / "raw_failure_sequences.csv"
    pd.DataFrame(sequences).to_csv(raw_path, index=False)
    logger.info(f"Saved raw sequences at {raw_path}")

    # Aggregate
    chains_df = aggregate_sequences(
        sequences,
        min_frequency=args.min_frequency
    )

    if chains_df.empty:
        logger.warning("No aggregated chains passed frequency threshold.")
    else:
        chains_path = RESULTS_DIR / "failure_chains.csv"
        chains_df.to_csv(chains_path, index=False)
        logger.info(f"Saved aggregated chains at {chains_path}")

    # Metadata
    meta = {
        "method": "Observed temporal failure sequence extraction",
        "assumptions": [
            "Single timestep lag (t-1 → t)",
            "Non-zero propagation weight implies dependency",
            "No causal inference"
        ],
        "parameters": {
            "weight_threshold": args.weight_threshold,
            "min_frequency": args.min_frequency
        },
        "data_shape": {
            "wards": W,
            "timesteps": T,
            "indicators": M
        },
        "results": {
            "total_sequences": len(sequences),
            "unique_chains": len(chains_df)
        }
    }

    meta_path = SUMMARY / "failure_chain_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved metadata at {meta_path}")
    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
