import streamlit as st
from pathlib import Path

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Urban Systems Digital Twin — Bengaluru",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================
# PATHS
# =========================================
BASE  = Path("statistical_inference/figures")
MAPS  = BASE / "maps"
DIAG  = BASE / "diagnostics"
DAGS  = BASE / "dags"
CLUST = BASE / "clusters"
FORE  = BASE / "forecasting"
OPT   = BASE / "optimisation"
RISK  = BASE / "risk"

# =========================================
# CSS
# =========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ---- BASE ---- */
.stApp { background-color: #09090f !important; color: #b8b8cc !important; }
.stApp > header { background: transparent !important; }
section[data-testid="stSidebar"] { background: #09090f !important; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: #b8b8cc;
}
h1, h2, h3, h4 {
    font-family: 'DM Serif Display', serif !important;
    color: #ededf5 !important;
    font-weight: 400 !important;
}

/* ---- TABS ---- */
.stTabs [data-baseweb="tab-list"] {
    background: #0d0d17 !important;
    border-bottom: 1px solid #1e1e30 !important;
    gap: 0 !important;
    padding: 0 2rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #44445e !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.85rem 1.5rem !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #cf4a4a !important;
    border-bottom: 2px solid #cf4a4a !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 2.5rem !important;
}

/* ---- MISC ---- */
.stRadio > label { color: #55556e !important; font-size: 0.8rem !important; font-family: 'DM Mono', monospace !important; }
.streamlit-expanderHeader {
    background: #0e0e1a !important;
    color: #55556e !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    border: 1px solid #1e1e30 !important;
    border-radius: 4px !important;
}
.streamlit-expanderContent {
    background: #0e0e1a !important;
    border: 1px solid #1e1e30 !important;
    border-top: none !important;
}
hr { border-color: #1e1e30 !important; }
.stAlert { background: #0e0e1a !important; border: 1px solid #1e1e30 !important; }
.stCaption, [data-testid="stCaptionContainer"] {
    color: #33334a !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.04em !important;
}
img { border-radius: 4px !important; }

/* ---- HERO ---- */
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #cf4a4a;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.4rem;
    font-weight: 400;
    line-height: 1.08;
    color: #ededf5;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}
.hero-title em { font-style: italic; color: #cf4a4a; }
.hero-para {
    font-size: 0.95rem;
    font-weight: 300;
    color: #6666888;
    line-height: 1.85;
    max-width: 680px;
    margin-bottom: 1.5rem;
    color: #666688;
}
.problem-frame {
    border-left: 2px solid #cf4a4a;
    padding: 1rem 1.4rem;
    background: rgba(207, 74, 74, 0.05);
    border-radius: 0 4px 4px 0;
    margin: 1.5rem 0;
    font-family: 'DM Serif Display', serif;
    font-size: 1rem;
    font-style: italic;
    color: #c0c0d4;
    line-height: 1.75;
}

/* ---- STAT CARDS ---- */
.stat-card {
    background: #0e0e1a;
    border: 1px solid #1e1e30;
    border-radius: 6px;
    padding: 1.1rem 1.25rem;
    text-align: center;
    margin-bottom: 10px;
}
.stat-number {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    font-weight: 400;
    color: #cf4a4a;
    line-height: 1.1;
}
.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #33334a;
    margin-top: 0.3rem;
    line-height: 1.6;
}

/* ---- CONTRAST ---- */
.contrast-block {
    background: #0e0e1a;
    border: 1px solid #1e1e30;
    border-radius: 6px;
    padding: 1.1rem 1.4rem;
}
.contrast-block.positive { border-left: 3px solid #3ab87a; }
.contrast-block.negative { border-left: 3px solid #cf4a4a; }
.contrast-item { font-size: 0.85rem; line-height: 2.2; margin: 0; }
.contrast-item.neg { color: #55556a; }
.contrast-item.pos { color: #8888a8; }

/* ---- PIPELINE ---- */
.pipeline-outer { overflow-x: auto; padding: 1.5rem 0 2rem 0; }
.pipeline-track {
    display: flex;
    align-items: flex-start;
    gap: 0;
    min-width: 900px;
}
.pipeline-node {
    flex: 1;
    background: #0e0e1a;
    border: 1px solid #1e1e30;
    border-radius: 6px;
    padding: 1.1rem 1rem;
    min-width: 155px;
}
.pipeline-node:hover { border-color: #cf4a4a; transition: border-color 0.2s; }
.pipeline-phase-id {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #cf4a4a;
    margin-bottom: 0.4rem;
}
.pipeline-title {
    font-family: 'DM Serif Display', serif;
    font-size: 0.82rem;
    font-weight: 400;
    color: #d8d8ec;
    line-height: 1.35;
    margin-bottom: 0.5rem;
}
.pipeline-methods {
    font-size: 0.67rem;
    color: #33334a;
    font-family: 'DM Mono', monospace;
    line-height: 1.75;
    margin-bottom: 0.5rem;
}
.pipeline-output {
    font-size: 0.67rem;
    color: #3ab87a;
    font-family: 'DM Mono', monospace;
    line-height: 1.75;
}
.pipeline-arrow {
    color: #2a2a40;
    font-size: 1rem;
    padding: 0 0.3rem;
    flex-shrink: 0;
    align-self: center;
    margin-top: -0.8rem;
}

/* ---- INSIGHT PANEL ---- */
.eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #cf4a4a;
    margin-bottom: 0.3rem;
}
.insight-panel {
    background: #0e0e1a;
    border: 1px solid #1e1e30;
    border-radius: 6px;
    padding: 1.25rem 1.5rem;
    font-size: 0.86rem;
    color: #7777908;
    line-height: 1.9;
    color: #666688;
}
.insight-panel strong { color: #cf4a4a; font-weight: 500; }
.insight-panel em { color: #b0b0c8; font-style: italic; }

/* ---- MODEL BADGE ---- */
.model-cascade {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin: 1rem 0;
}
.model-row {
    display: flex;
    align-items: center;
    gap: 10px;
}
.model-horizon {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #cf4a4a;
    width: 80px;
    flex-shrink: 0;
}
.model-badge {
    background: #131320;
    border: 1px solid #1e1e30;
    border-radius: 4px;
    padding: 0.35rem 0.7rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #9999b8;
    white-space: nowrap;
}
.model-badge.highlight { border-color: #3ab87a; color: #3ab87a; }
.model-connector { color: #1e1e30; font-size: 0.8rem; }

/* ---- FORECAST VAR CHIPS ---- */
.var-chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 0.8rem 0; }
.var-chip {
    background: #131320;
    border: 1px solid #1e1e30;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #7777a0;
    letter-spacing: 0.06em;
}
.var-chip.exo { border-color: #2a2a50; color: #5555880; color: #555580; font-style: italic; }

/* ---- STAT GRID ---- */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 1.5rem 0;
}

/* ---- MILP COMPARISON ---- */
.milp-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: #0e0e1a;
    border: 1px solid #1e1e30;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin-bottom: 10px;
}
.milp-facility {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #55556a;
    width: 100px;
    flex-shrink: 0;
}
.milp-before {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #cf4a4a;
    width: 60px;
    text-align: right;
}
.milp-arrow { color: #2a2a40; font-size: 1.1rem; padding: 0 0.3rem; }
.milp-after {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #3ab87a;
    width: 60px;
}
.milp-desc {
    font-size: 0.78rem;
    color: #44445e;
    line-height: 1.55;
    font-family: 'DM Mono', monospace;
    flex: 1;
}
.milp-saving {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #3ab87a;
    text-align: right;
    white-space: nowrap;
}

/* ---- PLACEHOLDER ---- */
.fig-placeholder {
    background: #0b0b15;
    border: 1px dashed #1e1e30;
    border-radius: 6px;
    min-height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #22223a;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    text-align: center;
    padding: 1.5rem;
    line-height: 1.8;
}

/* ---- BEFORE / AFTER ---- */
.before-after-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 3px;
    display: inline-block;
    margin-bottom: 0.5rem;
}
.label-before { background: rgba(207,74,74,0.1); color: #cf4a4a; }
.label-after  { background: rgba(58,184,122,0.08); color: #3ab87a; }

/* ---- QUOTE ---- */
.reviewer-quote {
    font-family: 'DM Serif Display', serif;
    font-style: italic;
    font-size: 1.05rem;
    font-weight: 400;
    color: #2e2e44;
    text-align: center;
    padding: 2rem 1rem;
    border-top: 1px solid #141422;
    margin-top: 2.5rem;
    line-height: 1.75;
}
.reviewer-quote span { color: #cf4a4a; }

/* ---- SECTION DIVIDER ---- */
.section-rule {
    border: none;
    border-top: 1px solid #141422;
    margin: 2.5rem 0;
}

/* ---- RISK LEGEND ---- */
.risk-legend {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin: 1rem 0;
}
.risk-legend-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
}
.risk-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 3px;
}
.risk-legend-text {
    font-size: 0.8rem;
    line-height: 1.5;
    color: #6666888;
    color: #555578;
    font-family: 'DM Mono', monospace;
}
.risk-legend-text strong { color: #9999b8; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# =========================================
# HELPER
# =========================================
def safe_image(path, caption="", placeholder_text=""):
    p = Path(path)
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        label = placeholder_text or str(path)
        st.markdown(f'<div class="fig-placeholder">{label}</div>', unsafe_allow_html=True)
        if caption:
            st.caption(caption)


# =========================================
# TABS
# =========================================
tab1, tab2, tab3 = st.tabs([
    "01 · Vision & Architecture",
    "02 · Analysis & Forecasting",
    "03 · Redesign & Optimisation",
])


# ================================================================
# TAB 1 — VISION & PIPELINE
# ================================================================
with tab1:
    st.markdown('<p class="hero-eyebrow">MSc Urban Informatics &nbsp;·&nbsp; Research Portfolio &nbsp;·&nbsp; 2025</p>', unsafe_allow_html=True)
    st.markdown("""
    <h1 class="hero-title">
        Urban Systems<br>
        <em>Digital Twin</em><br>
        for Bengaluru
    </h1>
    """, unsafe_allow_html=True)

    hero_left, hero_right = st.columns([3, 1])
    with hero_left:
        st.markdown("""
        <p class="hero-para">
        Cities are not collections of data points — they are dynamic, interdependent systems where
        transport, employment, housing, and environment co-evolve under uncertainty.
        This project constructs a research-grade digital twin of Bengaluru across five phases:
        empirical grounding, causal inference, AI-driven forecasting, systemic risk stress-testing,
        and evidence-based spatial redesign.
        </p>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="problem-frame">
        "Bengaluru's urban stress is not a congestion problem — it is a spatial mismatch problem.
        No amount of traffic prediction will fix a land-use failure."
        </div>
        """, unsafe_allow_html=True)

    with hero_right:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">198</div>
            <div class="stat-label">Wards modelled</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">5</div>
            <div class="stat-label">Research phases</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">12+</div>
            <div class="stat-label">Methods deployed</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<p class="eyebrow">What this is not</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="contrast-block negative">
        <p class="contrast-item neg">✗ &nbsp; Traffic prediction for a single corridor</p>
        <p class="contrast-item neg">✗ &nbsp; Clustering wards as an end goal</p>
        <p class="contrast-item neg">✗ &nbsp; A correlation study dressed as insight</p>
        <p class="contrast-item neg">✗ &nbsp; A black-box model with no causal grounding</p>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<p class="eyebrow">What this is</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="contrast-block positive">
        <p class="contrast-item pos">✓ &nbsp; Urban systems modelling under uncertainty</p>
        <p class="contrast-item pos">✓ &nbsp; Digital twin for policy-scale reasoning</p>
        <p class="contrast-item pos">✓ &nbsp; Causal inference before prediction</p>
        <p class="contrast-item pos">✓ &nbsp; Redesign grounded in counterfactual analysis</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown('<p class="eyebrow">Research Architecture</p>', unsafe_allow_html=True)
    st.markdown("### Full Research Pipeline")
    st.markdown(
        '<p style="font-size:0.78rem;color:#22223a;margin-bottom:1.5rem;font-family:\'DM Mono\',monospace">'
        '← scroll if needed &nbsp;·&nbsp; each phase builds on validated outputs from the prior one'
        '</p>',
        unsafe_allow_html=True
    )

    phases = [
        {
            "id": "Phase A",
            "title": "Empirical\nGrounding",
            "methods": ["GeoPandas", "PySAL", "LISA", "Moran's I", "K-Means"],
            "output": ["Ward geodataset", "Urban typology map", "Spatial stats"],
        },
        {
            "id": "Phase B",
            "title": "Statistical &\nCausal Baselines",
            "methods": ["OLS / GLM", "Robust SEs", "DID", "IV Regression", "DAGs"],
            "output": ["Causal estimates", "Regression baselines", "Causal DAGs"],
        },
        {
            "id": "Phase C",
            "title": "AI\nForecasting",
            "methods": ["SARIMAX (1yr)", "LGB · RF (5yr)", "LSTM · N-BEATS", "TFT (10yr)"],
            "output": ["10-yr forecasts", "Uncertainty bounds", "Future stress map"],
        },
        {
            "id": "Phase D",
            "title": "Systemic Risk\n& Cascades",
            "methods": ["Network analysis", "Monte Carlo", "Fragility curves"],
            "output": ["Vulnerability maps", "Cascade scenarios", "Risk rankings"],
        },
        {
            "id": "Phase E",
            "title": "Redesign &\nOptimisation",
            "methods": ["MILP", "Multi-obj. opt.", "Counterfactual", "Spatial re-alloc."],
            "output": ["Optimised scenarios", "Redesign maps", "Policy brief"],
        },
    ]

    nodes_html = ""
    for i, p in enumerate(phases):
        methods_str = "<br>".join(p["methods"])
        output_str  = "<br>".join(p["output"])
        title_str   = p["title"].replace("\n", "<br>")
        nodes_html += f"""
        <div class="pipeline-node">
            <div class="pipeline-phase-id">{p["id"]}</div>
            <div class="pipeline-title">{title_str}</div>
            <div class="pipeline-methods">{methods_str}</div>
            <div class="pipeline-output">{output_str}</div>
        </div>"""
        if i < len(phases) - 1:
            nodes_html += '<div class="pipeline-arrow">→</div>'

    st.markdown(f"""
    <div class="pipeline-outer">
        <div class="pipeline-track">{nodes_html}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="reviewer-quote">
    "This project wasn't hacked together. <span>It was architected.</span>"
    </div>
    """, unsafe_allow_html=True)


# ================================================================
# TAB 2 — ANALYSIS & FORECASTING (merged from old tabs 2 & 3)
# ================================================================
with tab2:

    st.markdown('<p class="eyebrow">Phase A → B → C</p>', unsafe_allow_html=True)
    st.markdown("### Analysis & Forecasting")
    st.markdown("""
    <p style="font-size:0.92rem;color:#555578;line-height:1.85;max-width:760px;margin-bottom:2rem">
    Interpretable statistical baselines were established before any AI model ran — this is how
    you know whether a model is learning something real or simply memorising spatial autocorrelation.
    Causal structure was specified via DAGs, validated through difference-in-differences and
    instrumental variable regression, and only then handed to a cascade of forecasting models
    with increasing time horizons.
    </p>
    """, unsafe_allow_html=True)

    # ---- SPATIAL MAPS ----
    st.markdown('<p class="eyebrow">Observed City — 2024 Baseline</p>', unsafe_allow_html=True)
    st.markdown("#### Population · Employment · Congestion · Air Quality")

    m1, m2 = st.columns(2)
    with m1:
        safe_image(MAPS / "map_population_est.png", caption="Population distribution — 2024",
                   placeholder_text="map_population_est.png")
    with m2:
        safe_image(MAPS / "map_it_job_density_mean.png", caption="IT job density — spatial mismatch visible",
                   placeholder_text="map_it_job_density_mean.png")

    m3, m4 = st.columns(2)
    with m3:
        safe_image(MAPS / "map_congestion_index_mean.png", caption="Congestion index — observed baseline",
                   placeholder_text="map_congestion_index_mean.png")
    with m4:
        safe_image(MAPS / "map_aqi_mean.png", caption="AQI — observed baseline",
                   placeholder_text="map_aqi_mean.png")

    st.markdown("""
    <div class="insight-panel" style="margin-top:1rem">
    <strong>Structural insight</strong> &nbsp; Population, employment, congestion, and pollution are not co-located.
    The spatial mismatch between where people live and where jobs concentrate generates systemic commute burden —
    a finding that directly shaped the causal model in Phase B and redesign priorities in Phase E.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ---- CORRELATION + DAGS ----
    st.markdown('<p class="eyebrow">Causal Grounding</p>', unsafe_allow_html=True)
    st.markdown("#### Correlation Structure & DAG-Driven Regression Strategy")

    cg1, cg2 = st.columns([3, 2])
    with cg1:
        safe_image(
            DIAG / "correlation_heatmap.png",
            caption="Correlation matrix — key urban variables",
            placeholder_text="correlation_heatmap.png"
        )
    with cg2:
        st.markdown("""
        <div class="insight-panel">
        <strong>Key correlations</strong><br><br>
        · Population tracks built-up area strongly — density is structural, not a policy choice<br><br>
        · Job density is <em>spatially decoupled</em> from residential density — the mismatch is measurable<br><br>
        · Congestion shows weak raw correlation with ward attributes — pointing to structural, not local, drivers<br><br>
        · This motivated DID and IV specifications over naive OLS
        </div>
        """, unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        safe_image(DAGS / "dag_F1_stress.png", caption="Urban stress mechanisms — causal pathway",
                   placeholder_text="dag_F1_stress.png")
    with d2:
        safe_image(DAGS / "dag_F2_landuse_mismatch.png", caption="Land-use mismatch & mobility — causal pathway",
                   placeholder_text="dag_F2_landuse_mismatch.png")

    st.markdown("""
    <div class="insight-panel" style="margin-top:0.5rem">
    <strong>Why the DAGs matter</strong> &nbsp; Correlation alone flags AQI and congestion as directly linked.
    The DAG reveals they share a common cause — employment decentralisation — and that naive regression
    overstates the direct effect. DID was specified accordingly. The model is grounded in <em>mechanism</em>, not coincidence.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ---- CLUSTERING ----
    st.markdown('<p class="eyebrow">Urban Typologies</p>', unsafe_allow_html=True)
    st.markdown("#### K-Means Clustering — Recurring Urban Forms")
    st.caption("K-Means + Hierarchical Clustering (scikit-learn) · feeds directly into Phase E redesign targeting")

    k = st.radio("Clustering resolution", ["k = 3 (recommended)", "k = 4 (sensitivity check)"], horizontal=True)

    k_left, k_right = st.columns([3, 2])
    with k_left:
        if k.startswith("k = 3"):
            safe_image(CLUST / "kmeans_k3_map.png", placeholder_text="kmeans_k3_map.png")
        else:
            safe_image(CLUST / "kmeans_k4_map.png", placeholder_text="kmeans_k4_map.png")

    with k_right:
        st.markdown("""
        <div class="insight-panel">
        <strong>Three stable archetypes</strong><br><br>
        <span style="color:#cf4a4a">■</span> &nbsp;<strong style="color:#9999b8">Residential periphery</strong><br>
        <span style="font-size:0.75rem;color:#33334a">High population · low jobs · high commute stress</span><br><br>
        <span style="color:#c8a050">■</span> &nbsp;<strong style="color:#9999b8">Mixed-use central</strong><br>
        <span style="font-size:0.75rem;color:#33334a">Balanced but overloaded infrastructure</span><br><br>
        <span style="color:#3ab87a">■</span> &nbsp;<strong style="color:#9999b8">Stressed employment zones</strong><br>
        <span style="font-size:0.75rem;color:#33334a">High jobs · inadequate housing · peak AQI</span><br><br>
        These typologies directly guide redesign targeting in Phase E.
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Hierarchical dendrogram — validation"):
            safe_image(DIAG / "hier_dendrogram.png", placeholder_text="hier_dendrogram.png")

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ---- LISA ----
    st.markdown('<p class="eyebrow">Spatial Dependence</p>', unsafe_allow_html=True)
    st.markdown("#### LISA Analysis — Confirming Non-Random Clustering")
    st.caption("Moran's I & LISA (PySAL) · Moran's I > 0.6 confirms spatial non-independence across wards")

    lisa1, lisa2 = st.columns(2)
    with lisa1:
        safe_image(MAPS / "lisa_map_population_est.png",
                   caption="LISA — Population (High-High / Low-Low clusters)",
                   placeholder_text="lisa_map_population_est.png")
    with lisa2:
        safe_image(MAPS / "lisa_map_it_job_density_mean.png",
                   caption="LISA — Employment density clustering",
                   placeholder_text="lisa_map_it_job_density_mean.png")

    st.markdown("""
    Strong positive spatial autocorrelation confirms urban processes are **networked, not independent**.
    Treating wards as independent observations would produce biased standard errors. Spatial structure is not noise — it is signal.
    """)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ---- AI FORECASTING ----
    st.markdown('<p class="eyebrow">Phase C — AI Forecasting Cascade</p>', unsafe_allow_html=True)
    st.markdown("#### Multi-Horizon Forecasting — 1 Year → 5 Years → 10 Years")

    fc_left, fc_right = st.columns([2, 3])
    with fc_left:
        st.markdown("""
        <div class="insight-panel">
        <strong>Forecasted variables</strong><br><br>
        <div class="var-chips">
            <span class="var-chip">Water demand</span>
            <span class="var-chip">Electricity</span>
            <span class="var-chip">PM2.5 / AQI</span>
            <span class="var-chip">Congestion index</span>
            <span class="var-chip exo">Rainfall ↗ exogenous</span>
            <span class="var-chip exo">Population ↗ exogenous</span>
        </div>
        <br>
        Rainfall and population growth enter as exogenous regressors — they drive the system but are not
        themselves modelled endogenously. This prevents target leakage and respects causal structure.
        </div>
        """, unsafe_allow_html=True)

    with fc_right:
        st.markdown("""
        <div class="insight-panel">
        <strong>Model cascade by horizon</strong><br><br>
        <div class="model-cascade">
            <div class="model-row">
                <span class="model-horizon">1-year</span>
                <span class="model-badge">SARIMAX</span>
                <span style="font-size:0.7rem;color:#33334a;font-family:'DM Mono',monospace">seasonal decomposition · interpretable coefficients</span>
            </div>
            <div class="model-row">
                <span class="model-horizon">5-year</span>
                <span class="model-badge">LightGBM</span>
                <span class="model-badge">Random Forest</span>
                <span style="font-size:0.7rem;color:#33334a;font-family:'DM Mono',monospace">ensemble · SHAP explainability</span>
            </div>
            <div class="model-row">
                <span class="model-horizon">10-year</span>
                <span class="model-badge">LSTM</span>
                <span class="model-badge">N-BEATS</span>
                <span class="model-badge highlight">TFT ✓ primary</span>
            </div>
        </div>
        <br>
        <strong>TFT (Temporal Fusion Transformer)</strong> was selected as the primary 10-year model —
        its attention mechanism makes the forecast interpretable: variable importance, temporal patterns,
        and uncertainty quantiles are all recoverable. The 10-year TFT output directly generated the
        <em>future stress map</em> used in Phase E redesign.
        </div>
        """, unsafe_allow_html=True)

    fc_img1, fc_img2 = st.columns(2)
    with fc_img1:
        safe_image(
            FORE / "forecast_congestion_2035.png",
            caption="Congestion index — 10-year ward-level forecast (TFT)",
            placeholder_text="forecast_congestion_2035.png"
        )
    with fc_img2:
        safe_image(
            FORE / "forecast_aqi_2035.png",
            caption="PM2.5 / AQI — 10-year forecast, baseline + intervention scenarios",
            placeholder_text="forecast_aqi_2035.png"
        )

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ---- FUTURE STRESS MAP ----
    st.markdown('<p class="eyebrow">TFT Output → Future Stress Map</p>', unsafe_allow_html=True)
    st.markdown("#### Ward-Level Mean Failure Probability — 2026–2035")
    st.caption("Built from TFT 10-year forecasts · wards coloured by mean predicted failure probability")

    fs_left, fs_right = st.columns([3, 2])
    with fs_left:
        # Use the uploaded red hotspot map
        st.image(r"C:\Users\jbhuv\OneDrive\Pictures\Screenshots\Screenshot 2026-01-02 131833.png",
                 caption="Mean failure probability by ward — darker red = higher future stress",
                 use_container_width=True)

    with fs_right:
        st.markdown("""
        <div class="insight-panel">
        <strong>How to read this map</strong><br><br>
        <div class="risk-legend">
            <div class="risk-legend-item">
                <div class="risk-dot" style="background:#7a0000"></div>
                <div class="risk-legend-text"><strong>Dark crimson (0.17–0.23)</strong><br>Critical — highest predicted failure probability. Immediate redesign priority. Central and south-central wards concentrated here.</div>
            </div>
            <div class="risk-legend-item">
                <div class="risk-dot" style="background:#cc3333"></div>
                <div class="risk-legend-text"><strong>Red (0.09–0.17)</strong><br>High stress — likely to breach system thresholds by 2030–2032. Phase E interventions targeted here.</div>
            </div>
            <div class="risk-legend-item">
                <div class="risk-dot" style="background:#f0aaaa"></div>
                <div class="risk-legend-text"><strong>Light pink (0.00–0.06)</strong><br>Currently manageable — peripheral wards with lower near-term risk but vulnerable to cascade contagion.</div>
            </div>
            <div class="risk-legend-item">
                <div class="risk-dot" style="background:#ffffff;border:1px solid #333"></div>
                <div class="risk-legend-text"><strong>White</strong><br>Missing or insufficient data for the ward.</div>
            </div>
        </div>
        <br>
        The spatial concentration of high-failure probability in the urban core — not the periphery —
        directly contradicts the common assumption that congestion is a boundary problem.
        <em>The core is failing. The periphery is the symptom.</em>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="reviewer-quote">
    "She didn't just run a model. <span>She built a causal scaffold, then ran the model on top of it.</span>"
    </div>
    """, unsafe_allow_html=True)


# ================================================================
# TAB 3 — REDESIGN & OPTIMISATION
# ================================================================
with tab3:

    st.markdown('<p class="eyebrow">Phase D + E — Intervention Layer</p>', unsafe_allow_html=True)
    st.markdown("### Redesign & Optimisation")
    st.markdown("""
    <p style="font-size:0.92rem;color:#555578;line-height:1.85;max-width:760px;margin-bottom:2rem">
    The TFT-generated future stress map identified which wards will fail under the baseline trajectory.
    Phase E responds: multi-objective MILP optimisation determines the minimum set of strategically
    placed facilities that satisfies coverage, equity, and cost constraints simultaneously —
    then the redesigned city is mapped against the observed 2024 baseline.
    </p>
    """, unsafe_allow_html=True)

    # ---- MASTER PLAN REDESIGN MAP ----
    st.markdown('<p class="eyebrow">Redesigned City — Master Plan</p>', unsafe_allow_html=True)
    st.markdown("#### Spatial Redesign — Land Use, Facilities, Transit, Infrastructure")

    st.image(r"C:\Users\jbhuv\OneDrive\Pictures\Screenshots\Screenshot 2026-03-31 230931.png",
             caption="Bengaluru Master Plan redesign — sectors, land use, facilities, metro corridors overlaid",
             use_container_width=True)

    st.markdown("""
    <div class="insight-panel" style="margin-top:1rem">
    <strong>What the redesign encodes</strong> &nbsp;
    The master plan map layers sectors, land-use zoning (residential by income tier, mixed-use, industrial),
    facility placement (hospitals, schools, parks, government, emergency services), and proposed metro corridors.
    Each layer was derived from the MILP optimal solution — not from intuition. The coloured sectors correspond
    directly to the urban typologies identified in Phase A clustering:
    <em>residential periphery, mixed-use central, and stressed employment zones.</em>
    Facility placement was constrained to maximise coverage within each typology's demand surface.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ---- MILP RESULTS ----
    st.markdown('<p class="eyebrow">Phase E — MILP Optimisation Results</p>', unsafe_allow_html=True)
    st.markdown("#### Multi-Objective Facility Re-allocation")
    st.caption("Objective: simultaneous coverage maximisation · equity (underserved wards) · cost minimisation")

    st.markdown("""
    <div class="milp-row">
        <span class="milp-facility">Emergency</span>
        <span class="milp-before">239</span>
        <span class="milp-arrow">→</span>
        <span class="milp-after">80</span>
        <span class="milp-desc">Phase 6 plan had 239 dispersed stations with overlapping coverage zones.
        MILP identified 80 optimally placed stations achieving equivalent or superior population coverage
        with eliminated redundancy.</span>
        <span class="milp-saving">−66% facilities<br>coverage maintained</span>
    </div>
    <div class="milp-row">
        <span class="milp-facility">Schools</span>
        <span class="milp-before">638</span>
        <span class="milp-arrow">→</span>
        <span class="milp-after">150</span>
        <span class="milp-desc">638 schools in the Phase 6 plan showed severe clustering in central wards
        with peripheral underservice. 150 optimised locations achieve full ward coverage with
        equity constraints enforced for underserved zones.</span>
        <span class="milp-saving">−76% facilities<br>equity improved</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ---- COMPARISON MAPS ----
    st.markdown('<p class="eyebrow">Before → After Comparison</p>', unsafe_allow_html=True)
    st.markdown("#### Emergency Services — Phase 6 Design vs. Optimised (MILP)")

    st.markdown('<span class="before-after-label label-before">Phase 6 — 239 locations (current plan)</span> &nbsp; <span class="before-after-label label-after">Optimised — 80 locations (MILP)</span>', unsafe_allow_html=True)
    st.image(r"C:\Users\jbhuv\OneDrive\Pictures\Screenshots\Screenshot 2026-03-31 232606.png",
             caption="Emergency services: Phase 6 design (239 dispersed) vs. MILP-optimised (80 strategic placements)",
             use_container_width=True)

    st.markdown("""
    <div class="insight-panel" style="margin-top:0.5rem;margin-bottom:1.5rem">
    The Phase 6 plan shows <strong>dense overlapping coverage</strong> in central areas with thin peripheral presence —
    a pattern driven by historical incremental placement rather than system-level design.
    The MILP solution redistributes stations to maximise the number of wards within the response-time threshold,
    with explicit weight given to high-failure-probability wards identified in the TFT stress map.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Schools — Phase 6 Design vs. Optimised (MILP)")
    st.markdown('<span class="before-after-label label-before">Phase 6 — 638 schools</span> &nbsp; <span class="before-after-label label-after">Optimised — 150 schools</span>', unsafe_allow_html=True)
    st.image("C:\\Users\\jbhuv\\OneDrive\\Pictures\\Screenshots\\Screenshot 2026-03-31 232702.png",
             caption="School placement: Phase 6 design (638, heavily clustered centrally) vs. MILP-optimised (150, equitably distributed)",
             use_container_width=True)

    st.markdown("""
    <div class="insight-panel" style="margin-top:0.5rem">
    School placement under Phase 6 replicates the same spatial mismatch diagnosed in Phase A:
    central-ward concentration while peripheral high-population wards remain underserved.
    The MILP solution enforces a <em>minimum coverage constraint per ward</em>, eliminating the equity gap
    while reducing total facility count by 76% — demonstrating that the problem is placement, not quantity.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    # ---- KEY FINDINGS ----
    st.markdown('<p class="eyebrow">Summary Findings</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-number">−66%</div>
            <div class="stat-label">Emergency stations<br>without coverage loss</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">−76%</div>
            <div class="stat-label">School facilities<br>with improved equity</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">3×</div>
            <div class="stat-label">Higher return from<br>placement vs. quantity</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-panel" style="margin-top:0.5rem">
    <strong>Central finding</strong> &nbsp; Bengaluru's infrastructure deficit is not a quantity problem — it is a
    <em>placement problem</em>. The Phase 6 master plan over-provisions central wards while under-serving
    high-stress peripheral zones that the TFT model flags as future failure points.
    Multi-objective MILP demonstrates that coverage, equity, and cost objectives can be satisfied simultaneously
    with dramatically fewer facilities, provided placement is optimised against the actual demand surface
    rather than historical precedent.
    <br><br>
    <strong>This finding is consistent across all forecasting horizons and robust to the full Monte Carlo ensemble.</strong>
    It is not a modelling artefact. <em>It is what the city is telling us.</em>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="reviewer-quote">
    "She understands cities as systems, not datasets —<br>
    <span>and she built the evidence to prove it.</span>"
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.caption(
        "Urban Systems Digital Twin · Bengaluru · 2025 &nbsp;·&nbsp; "
        "Figures generated from original spatial analysis pipeline &nbsp;·&nbsp; "
        "Code and data available on request"
    )