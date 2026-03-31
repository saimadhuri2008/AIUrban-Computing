import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
from joblib import dump
import pandas as pd


# ===============================
# RUN SETUP
# ===============================



BASE_DIR = Path("cascade_model")
ARTIFACTS_DIR = BASE_DIR / "artifacts/dependency_graph"
RESULTS_DIR = BASE_DIR / "results/dependency_graph"
SUMMARY = BASE_DIR / "summary"

LOGS_DIR = BASE_DIR / "logs"

for d in [ARTIFACTS_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===============================
# LOGGING SETUP
# ===============================
log_file = LOGS_DIR / "dependency_graph.log"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("dependency_graph")
logger.info("Starting Dependency Graph Construction")

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ===============================
# DEPENDENCY RULES (DOMAIN-INFORMED)
# ===============================
DEPENDENCY_RULES = {
    "population": {
        "electricity_demand": 0.8,
        "water_demand": 0.9,
        "congestion_index": 0.7
    },
    "rainfall": {
        "water_demand": -0.6,
        "pm25": -0.7
    },
    "electricity_demand": {
        "pm25": 0.4,
        "congestion_index": 0.3
    },
    "water_demand": {
        "congestion_index": 0.2,
        "population": -0.25        # feedback: water stress limits population
    },
    "congestion_index": {
        "pm25": 0.6,
        "electricity_demand": 0.2  # feedback: congestion raises energy use
    },
    "pm25": {
        "population": -0.2         # feedback: pollution affects livability
    }
}


DEPENDENCY_RULES_VERSION = "v1.0"

# Save dependency rules
rules_path = ARTIFACTS_DIR / "dependency_rules.json"
with open(rules_path, "w") as f:
    json.dump(DEPENDENCY_RULES, f, indent=4)

logger.info(f"Saved dependency rules at {rules_path}")

ASSUMPTIONS = {
    "graph_structure": "directed",
    "dependency_nature": "domain-informed",
    "weights_meaning": "relative influence strength",
    "weight_calibration": "not data-fitted",
    "negative_weights": "mitigating effects allowed",
    "temporal_scope": "static dependency structure",
    "layout_type": "topological (non-spatial)"
}

assumptions_path = ARTIFACTS_DIR / "assumptions.json"
with open(assumptions_path, "w") as f:
    json.dump(ASSUMPTIONS, f, indent=4)

logger.info(f"Saved modeling assumptions at {assumptions_path}")


# ===============================
# BUILD DEPENDENCY GRAPH
# ===============================
def build_dependency_graph(rules: dict) -> nx.DiGraph:
    G = nx.DiGraph()

    for src, targets in rules.items():
        G.add_node(src)
        for dst, weight in targets.items():
            G.add_node(dst)
            G.add_edge(src, dst, weight=weight)

    return G

G = build_dependency_graph(DEPENDENCY_RULES)
logger.info(
    f"Constructed dependency graph with "
    f"{G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
)

# Save graph artifact
graph_path = ARTIFACTS_DIR / "dependency_graph.joblib"
dump(G, graph_path)
logger.info(f"Saved dependency graph artifact at {graph_path}")

isolated = list(nx.isolates(G))
if isolated:
    logger.warning(f"Isolated nodes detected: {isolated}")
else:
    logger.info("No isolated nodes detected in dependency graph")


centrality = nx.betweenness_centrality(G, weight="weight")

centrality_df = (
    pd.DataFrame.from_dict(centrality, orient="index", columns=["betweenness"])
      .sort_values("betweenness", ascending=False)
)

centrality_df.to_csv(RESULTS_DIR / "node_centrality.csv")
logger.info("Saved node centrality diagnostics")


fingerprint = {
    "dependency_rules.json": sha256(rules_path),
    "dependency_graph.joblib": sha256(graph_path),
    "assumptions.json": sha256(assumptions_path)
}

with open(SUMMARY / "data_fingerprint.json", "w") as f:
    json.dump(fingerprint, f, indent=4)

logger.info("Artifact fingerprints saved")


# ===============================
# VISUALIZATION (TOPOLOGICAL)
# ===============================
def plot_dependency_graph(G: nx.DiGraph, outpath: Path):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color="skyblue")
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Infrastructure Dependency Graph (Topological)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    logger.info(f"Saved dependency graph visualization at {outpath}")

plot_dependency_graph(G, RESULTS_DIR / "dependency_graph.png")

# ===============================
# RUN METADATA
# ===============================
run_meta = {
    "analysis_stage": "cascading_dependency_modeling",
    "city": "Bengaluru",
    "dependency_type": "Directed, weighted, domain-informed",
    "dependency_rules_version": DEPENDENCY_RULES_VERSION,
    "weights_interpretation": "Relative influence strength",
    "calibration": "expert/domain-informed",
    "graph_nodes": G.number_of_nodes(),
    "graph_edges": G.number_of_edges(),
    "visualization": "Spring layout (topological, not spatial)",
    "diagnostics": {
        "centrality_metric": "betweenness",
        "centrality_weighted": True
    }

}

with open(RESULTS_DIR / "run_meta.json", "w") as f:
    json.dump(run_meta, f, indent=4)

logger.info("Run metadata saved")
logger.info("dependency graph construction completed successfully")

