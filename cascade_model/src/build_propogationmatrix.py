#!/usr/bin/env python3
"""
propagation_matrix.py

Builds a research-grade base propagation matrix from dependency rules.

Key properties:
- Column-normalized (outgoing influence conserved)
- Supports signed dependencies (positive / negative)
- Designed to be used as BASE matrix for state-dependent cascades
- Fully reproducible and paper-safe

Outputs:
- propagation_matrix.joblib
- node_index_map.json
- propagation_matrix_meta.json
"""

import json
import numpy as np
import joblib
from pathlib import Path
import logging

# ===============================
# PATHS
# ===============================
BASE_DIR = Path("cascade_model")
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SUMMARY = BASE_DIR / "summary"

GRAPH_PATH = ARTIFACTS_DIR / "dependency_graph/dependency_rules.json"
OUT_MATRIX_PATH = ARTIFACTS_DIR / "propagation_matrix.joblib"
OUT_INDEX_PATH = ARTIFACTS_DIR / "node_index_map.json"
META_PATH = SUMMARY / "propagation_matrix_meta.json"

# ===============================
# LOGGING
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("propagation_matrix")

# ===============================
# CORE LOGIC
# ===============================
def build_propagation_matrix():
    # --------------------------------------------------
    # Load dependency rules
    # --------------------------------------------------
    with open(GRAPH_PATH, "r") as f:
        graph = json.load(f)

    if not graph:
        raise ValueError("Dependency rules are empty")

    # Validate weights
    for src, targets in graph.items():
        for dst, weight in targets.items():
            if not isinstance(weight, (int, float)):
                raise TypeError(f"Non-numeric weight: {src} -> {dst}")

    # --------------------------------------------------
    # Node indexing
    # --------------------------------------------------
    nodes = sorted(
        set(graph.keys()) |
        {dst for targets in graph.values() for dst in targets}
    )

    n = len(nodes)
    index_map = {node: i for i, node in enumerate(nodes)}

    # --------------------------------------------------
    # Build raw matrix A (dst, src)
    # --------------------------------------------------
    A = np.zeros((n, n), dtype=float)

    for src, targets in graph.items():
        j = index_map[src]
        for dst, weight in targets.items():
            i = index_map[dst]
            A[i, j] = float(weight)

    # --------------------------------------------------
    # COLUMN NORMALIZATION (CRITICAL)
    # --------------------------------------------------
    # Each column represents how influence from src is distributed
    col_sums = np.sum(np.abs(A), axis=0, keepdims=True)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        A_norm = np.divide(A, col_sums, where=col_sums != 0)

    # Columns with no outgoing influence remain zero
    A_norm[:, col_sums.flatten() == 0] = 0.0

    # --------------------------------------------------
    # Spectral radius check (diagnostic only)
    # --------------------------------------------------
    eigvals = np.linalg.eigvals(A_norm)
    spectral_radius = float(np.max(np.abs(eigvals)))

    # --------------------------------------------------
    # Save artifacts
    # --------------------------------------------------
    joblib.dump(A_norm, OUT_MATRIX_PATH)

    with open(OUT_INDEX_PATH, "w") as f:
        json.dump(index_map, f, indent=2)

    meta = {
        "matrix_shape": A_norm.shape,
        "normalization": "column_l1_norm (sum abs = 1)",
        "graph_source": str(GRAPH_PATH),
        "ordering": "node_index_map.json",
        "spectral_radius": spectral_radius,
        "interpretation": (
            "A_norm[i,j] represents fraction of stress from node j "
            "transmitted to node i under full functionality."
        ),
        "note": (
            "This is a BASE propagation matrix. "
            "State-dependent attenuation is applied during cascade simulation."
        )
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------
    logger.info(f"Propagation matrix saved to {OUT_MATRIX_PATH}")
    logger.info(f"Node index map saved to {OUT_INDEX_PATH}")
    logger.info(f"Spectral radius (diagnostic): {spectral_radius:.3f}")

    if spectral_radius > 1.0:
        logger.warning(
            "Spectral radius > 1.0. "
            "This does NOT imply instability because failures attenuate propagation."
        )

    logger.info("Propagation matrix construction complete.")

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    build_propagation_matrix()
