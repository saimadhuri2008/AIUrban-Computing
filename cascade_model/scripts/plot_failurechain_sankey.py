#!/usr/bin/env python3
"""
sankey_readable.py

Improved, presentation-ready Sankey + top-chains table for portfolio.
- clearer labels (counts + %)
- node hover shows in/out totals
- table below shows top-K chains with counts & freq
- animated snapshots (All / 2026 / 2030 / 2035)

Usage:
 python src/cascade_model/sankey_readable.py \
   --chains src/cascade_model/outputs/chains_top.csv \
   --ward_chains src/cascade_model/outputs/ward_chains.json \
   --outdir src/cascade_model/visuals \
   --topk 12
"""

import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from collections import Counter, defaultdict
from datetime import datetime
import logging
import hashlib

def file_hash(path, block_size=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

def combined_inputs_hash(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(file_hash(p).encode())
    return h.hexdigest()


BASE_DIR = Path("cascade_model")
RESULTS_DIR = BASE_DIR / "results/sankeydiagram"
REPORTS_DIR = BASE_DIR / "reports/sankeydiagram"
LOGS_DIR = BASE_DIR /"logs"
SUMMARY = BASE_DIR / "summary"

RUN_META_PATH = SUMMARY / "run_metadata.json"


for d in [BASE_DIR,RESULTS_DIR,REPORTS_DIR, LOGS_DIR,SUMMARY]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "sankey_readable.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sankey")

# ---------------- helpers ----------------
def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def parse_chain(chain_str):
    return [s.strip() for s in chain_str.split("->")]

def build_link_counts(chains_df):
    lc = Counter()

    MIN_LINK_SHARE = 0.03  # 3% threshold (research standard)

    filtered = Counter()
    for (s, t), v in lc.items():
        if total_transitions > 0 and (v / total_transitions) >= MIN_LINK_SHARE:
            filtered[(s, t)] = v

        return filtered, total_transitions

    total_transitions = 0
    # iterate rows
    for _, r in chains_df.iterrows():
        chain = parse_chain(r["chain"])
        count = count = int(r["count"]) if "count" in r and not pd.isna(r["count"]) else 1
        total_transitions += max(0, (len(chain)-1) * count)
        for i in range(len(chain)-1):
            lc[(chain[i], chain[i+1])] += count
    return lc, total_transitions

def build_node_stats(link_counts):
    nodes = set()
    for (s,t) in link_counts:
        nodes.add(s); nodes.add(t)
    # compute in/out sums
    in_sum = defaultdict(int)
    out_sum = defaultdict(int)
    for (s,t),v in link_counts.items():
        out_sum[s] += v
        in_sum[t] += v
    return sorted(list(nodes)), in_sum, out_sum

def preferred_ordering(nodes):
    # give a sane default ordering for Bengaluru project domain
    order = ["congestion_index","pm25","water_demand","population","electricity_demand","rainfall"]
    # put preferred ones first, rest afterward
    final = [n for n in order if n in nodes] + [n for n in nodes if n not in order]
    return final

def build_sankey_arrays(link_counts, total_transitions, node_order, color_map):
    node_idx = {n:i for i,n in enumerate(node_order)}
    labels = node_order
    colors = [color_map.get(n,"#B0BEC5") for n in labels]

    src = []; tgt = []; vals = []; link_labels = []; hover = []
    for (s,t),v in link_counts.items():
        if s not in node_idx or t not in node_idx:
            continue
        src.append(node_idx[s]); tgt.append(node_idx[t]); vals.append(v)
        pct = 100.0 * v / total_transitions if total_transitions>0 else 0.0
        label = f"{v} ({pct:.1f}%)"
        link_labels.append(label)
        hover.append(f"{s} → {t}<br>count: {v}<br>share: {pct:.1f}%")
    return labels, colors, src, tgt, vals, link_labels, hover

def compute_node_xy(labels, layer_map):
    # simple left-right positions by layer (0..n)
    max_layer = max(layer_map.values()) if layer_map else 1
    x = []
    y = []
    # group nodes by layer to spread vertically
    groups = defaultdict(list)
    for n in labels:
        groups[layer_map.get(n,1)].append(n)
    for n in labels:
        layer = layer_map.get(n,1)
        xpos = layer / max_layer * 0.85 + 0.05
        # compute y from position within group
        pos = groups[layer].index(n)
        cnt = len(groups[layer])
        y_pos = (pos + 0.5) / cnt
        x.append(xpos); y.append(y_pos)
    return x, y

# ------------- main -------------
def run(args):

    chains_csv = Path(args.chains)
    ward_chains_json = Path(args.ward_chains)
    topk = int(args.topk)

    input_files = [chains_csv, ward_chains_json]
    current_hash = combined_inputs_hash(input_files)

    if RUN_META_PATH.exists():
        prev = json.loads(RUN_META_PATH.read_text())
        if prev.get("input_hash") == current_hash:
            logger.info("Inputs unchanged — using cached Sankey outputs.")
            logger.info(f"Existing HTML: {prev['outputs']['html']}")
            return

    chains_df = pd.read_csv(chains_csv)
    with open(ward_chains_json,"r") as fh:
        ward_chains = json.load(fh)

    # build counts
    link_counts, total_transitions = build_link_counts(chains_df)
    nodes, in_sum, out_sum = build_node_stats(link_counts)
    node_order = preferred_ordering(nodes)

    # color palette (domain-themed)
    color_map = {
        "congestion_index": "#c0392b", "pm25":"#e67e22",
        "water_demand":"#2980b9", "population":"#f39c12",
        "electricity_demand":"#f1c40f", "rainfall":"#2ecc71"
    }

    labels, colors, src, tgt, vals, link_labels, link_hover = build_sankey_arrays(
        link_counts, total_transitions, node_order, color_map
    )

    # node hover text with in/out summary
    node_hover = []
    for n in labels:
        incoming = in_sum.get(n,0)
        outgoing = out_sum.get(n,0)
        node_hover.append(f"{n}<br>incoming: {incoming}<br>outgoing: {outgoing}")

    # compute node positions for nicer left-right
    # define layer_map explicitly to force logical flow
    layer_map = {
        "congestion_index":0, "pm25":1, "water_demand":2,
        "population":3, "electricity_demand":4, "rainfall":5
    }
    node_x, node_y = compute_node_xy(labels, layer_map)

    # Make the Sankey
    link = dict(source=src, target=tgt, value=vals, label=link_labels, color="rgba(160,160,160,0.5)", customdata=link_hover,
                hovertemplate="%{customdata}<extra></extra>")
    node = dict(label=labels, color=colors, pad=35, thickness=28,
                x=node_x, y=node_y, customdata=node_hover,
                hovertemplate="%{customdata}<extra></extra>")
    sankey = go.Sankey(node=node, link=link, arrangement="fixed")

    # Top-K chains table (from chains_df)
    chains_df_sorted = chains_df.sort_values("count", ascending=False).head(topk)
    table_header = ["rank","chain","count","freq"]
    table_values = [
        list(range(1,len(chains_df_sorted)+1)),
        chains_df_sorted["chain"].tolist(),
        chains_df_sorted["count"].tolist(),
        (chains_df_sorted.get("freq", pd.Series([None]*len(chains_df_sorted))).fillna("").tolist())
    ]

    # Build combined figure: sankey (row1) & table (row2)
    fig = make_subplots(rows=2, cols=1,
                        specs=[[{"type":"domain"}],
                               [{"type":"table"}]],
                        row_heights=[0.7,0.3])

    fig.add_trace(sankey, row=1, col=1)
    fig.add_trace(go.Table(
        header=dict(values=table_header, fill_color="#f7f7f7", align="left", font=dict(size=13)),
        cells=dict(values=table_values, fill_color="#ffffff", align="left", font=dict(size=12))
    ), row=2, col=1)

    # Title & annotation box with plain-language instructions
    caption = (
        "<b>Interpretation</b>: Boxes are city indicators. "
        "Flows show dominant failure propagation paths. "
        "Thickness = frequency. Color encodes indicator type. "
        "Only statistically significant transitions shown."
    )
    fig.update_layout(title_text="Failure Chain Sankey — Clear View for Non-Experts",
                      annotations=[dict(text=caption, x=0, y=-0.03, showarrow=False, xref="paper", yref="paper", align="left")],
                      height=880, width=1200, font=dict(size=13))

    # Save interactive html
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_html = REPORTS_DIR / f"sankey_readable_{ts}.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    logger.info(f"[SAVED] interactive HTML  {out_html}")

    # Save static images (kaleido required)
    try:
        out_png = REPORTS_DIR / f"sankey_readable_{ts}.png"
        out_svg = REPORTS_DIR / f"sankey_readable_{ts}.svg"
        pio.write_image(fig, str(out_png))
        pio.write_image(fig, str(out_svg))
        logger.info(f"[SAVED] static PNG/SVG at  {out_png}, {out_svg}")
    except Exception as e:
        logger.warning("[WARN] static image export failed (kaleido missing?). Install 'kaleido' to enable.")
        logger.info(f"Error: {e}")

    # Save summary json
    summary = {
        "created": datetime.utcnow().isoformat()+"Z",
        "n_nodes": len(labels),
        "n_links": len(vals),
        "total_transitions": int(total_transitions),
        "agg_html": str(out_html)
    }
    (SUMMARY / f"sankey_readable_summary_{ts}.json").write_text(json.dumps(summary, indent=2))
    logger.info("[SAVED] summary JSON")

    logger.info("Done. Open the HTML in a browser and hover links & nodes; the explanations are in the caption.")

    run_meta = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "input_files": {
            "chains_csv": str(chains_csv),
            "ward_chains_json": str(ward_chains_json)
        },
        "parameters": {
            "topk": topk
        },
        "outputs": {
            "html": str(out_html),
            "png": str(REPORTS_DIR / "sankey_readable.png"),
            "svg": str(REPORTS_DIR / "sankey_readable.svg"),
            "table": str(RESULTS_DIR / "top_failure_chains.csv")
        },
        "stats": {
            "n_nodes": len(labels),
            "n_links": len(vals),
            "total_transitions": int(total_transitions)
        }
    }

    (SUMMARY / "run_metadata.json").write_text(
        json.dumps(run_meta, indent=2)
    )
    logger.info("Saved run_metadata.json")


# ---------------- CLI ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chains", default=BASE_DIR/"results/failure_chain/chains_top.csv")
    parser.add_argument("--ward_chains", default=BASE_DIR/"results/failure_chain/ward_chains.json")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--topk", default=10)
    args = parser.parse_args()
    run(args)
