"""
Model Diagnostics for Narrative Structure of the Popol Wuj
"""

import os
import yaml
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy as scipy_entropy

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
LANG_LABELS  = cfg["languages"]


def find_token_file(src_id: str) -> str | None:
    candidates = [
        os.path.join(APP_DIR, src_id, f"{src_id}-TOKEN.csv"),
        os.path.join(APP_DIR, "ensemble", f"{src_id}-TOKEN.csv"),
        os.path.join(APP_DIR, f"{src_id}-TOKEN.csv"),
        os.path.join(APP_DIR, "..", src_id, f"{src_id}-TOKEN.csv"),
    ]
    for p in candidates:
        norm = os.path.normpath(p)
        if os.path.exists(norm):
            return norm
    return None


@st.cache_data(show_spinner=False)
def load_tokens(src_id: str, token_path: str) -> pd.DataFrame:
    TOKEN = pd.read_csv(token_path)
    idx_offset = TOKEN.columns.to_list().index("token_str")
    ohco = TOKEN.columns.to_list()[:idx_offset]
    return TOKEN.set_index(ohco)


@st.cache_data(show_spinner="Running model…")
def run_model(src_id, token_path, chunk_size, overlap_int, min_df, max_df, n_topics):
    TOKEN = load_tokens(src_id, token_path)
    tokens = TOKEN["term_str"].dropna().to_list()

    step = max(1, chunk_size - overlap_int)
    token_arr = np.array(tokens)
    if len(token_arr) < chunk_size:
        return None, None, None
    windows = np.lib.stride_tricks.sliding_window_view(token_arr, chunk_size)[::step]
    chunks_s = pd.Series(
        np.apply_along_axis(lambda row: " ".join(row), axis=1, arr=windows)
    ).loc[lambda s: s.str.split().str.len() >= 50]
    chunks_list = chunks_s.tolist()

    if len(chunks_list) < 2:
        return None, None, None

    try:
        vec = TfidfVectorizer(lowercase=True, max_df=max_df, min_df=min_df,
                              strip_accents=None, norm="l2")
        X = vec.fit_transform(chunks_list)
    except ValueError:
        return None, None, None

    nmf = NMF(n_components=n_topics, init="nndsvd", max_iter=500)
    THETA = pd.DataFrame(nmf.fit_transform(X))
    THETA.index.name, THETA.columns.name = "chunk_id", "topic_id"
    PHI_sim = cosine_similarity(nmf.components_)

    return THETA, PHI_sim, X


@st.cache_data(show_spinner="Running elbow analysis…")
def run_elbow(src_id, token_path, chunk_size, overlap_int, min_df, max_df, max_topics):
    TOKEN = load_tokens(src_id, token_path)
    tokens = TOKEN["term_str"].dropna().to_list()
    step = max(1, chunk_size - overlap_int)
    token_arr = np.array(tokens)
    if len(token_arr) < chunk_size:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(token_arr, chunk_size)[::step]
    chunks_s = pd.Series(
        np.apply_along_axis(lambda row: " ".join(row), axis=1, arr=windows)
    ).loc[lambda s: s.str.split().str.len() >= 50]
    chunks_list = chunks_s.tolist()
    if len(chunks_list) < 2:
        return None
    try:
        vec = TfidfVectorizer(lowercase=True, max_df=max_df, min_df=min_df,
                              strip_accents=None, norm="l2")
        X = vec.fit_transform(chunks_list)
    except ValueError:
        return None
    rows = []
    for n in range(2, max_topics + 1):
        m = NMF(n_components=n, init="nndsvd", max_iter=500)
        m.fit(X)
        rows.append({"n_topics": n, "error": m.reconstruction_err_})
    return pd.DataFrame(rows)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Model Diagnostics — Popol Wuj",
    page_icon="🔬",
    layout="wide",
)

st.markdown(f"""
<style>
.block-container {{ padding-top: 1rem; padding-bottom: 1rem; }}
h3 {{ margin-bottom: 1rem; }}
@media (min-width: 768px) {{
    section[data-testid="stSidebar"] {{
        min-width: {cfg['sidebar']['min_width_px']}px;
        width: {cfg['sidebar']['width_vw']}vw;
    }}
}}
</style>
""", unsafe_allow_html=True)

st.title("Model Diagnostics — Popol Wuj")

# ── Controls ──────────────────────────────────────────────────────────────────
src_ids = list(SOURCES_META.keys())
cols = st.columns(cfg["layout"]["column_ratios"])

_c = cfg["controls"]
src_id     = cols[0].selectbox(
    "Source", src_ids, index=src_ids.index(_c["default_source"]),
    format_func=lambda x: f"{SOURCES_META[x]['label']} ({LANG_LABELS[SOURCES_META[x]['lang']]})")
chunk_size = cols[1].number_input("Chunk size", _c["chunk_size"]["min"], _c["chunk_size"]["max"], _c["chunk_size"]["default"], step=_c["chunk_size"]["step"])
overlap    = cols[2].number_input("Overlap", _c["overlap"]["min"], _c["overlap"]["max"], _c["overlap"]["default"], step=_c["overlap"]["step"], format="%.2f")
min_df     = cols[3].number_input("min_df", _c["min_df"]["min"], _c["min_df"]["max"], _c["min_df"]["default"], step=_c["min_df"]["step"])
max_df     = cols[4].number_input("max_df", _c["max_df"]["min"], _c["max_df"]["max"], _c["max_df"]["default"], step=_c["max_df"]["step"], format="%.2f")
n_topics   = cols[5].number_input("Topics", _c["n_topics"]["min"], _c["n_topics"]["max"], _c["n_topics"]["default"], step=_c["n_topics"]["step"])

overlap_int = int(overlap * chunk_size)

st.divider()

# ── File resolution ───────────────────────────────────────────────────────────
token_path = find_token_file(src_id)
if token_path is None:
    st.warning(f"Token file not found for `{src_id}`.")
    st.stop()

# ── Run models ────────────────────────────────────────────────────────────────
THETA, PHI_sim, _ = run_model(src_id, token_path, chunk_size, overlap_int, min_df, max_df, n_topics)
elbow_df = run_elbow(src_id, token_path, chunk_size, overlap_int, min_df, max_df, _c["n_topics"]["max"])

if THETA is None:
    st.warning("Model couldn't run — try adjusting chunk size, min_df, or max_df.")
    st.stop()

# ── Diagnostics grid ──────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
_m = dict(l=10, r=10, t=30, b=10)

with col1:
    st.subheader("Topic Overlap")
    topic_labels = [f"Topic {i}" for i in range(n_topics)]
    fig_sim = px.imshow(
        PHI_sim,
        x=topic_labels, y=topic_labels,
        color_continuous_scale="RdBu_r",
        zmin=0, zmax=1,
        aspect="auto",
    )
    fig_sim.update_layout(height=350, margin=_m, coloraxis_showscale=True)
    st.plotly_chart(fig_sim, use_container_width=True)

with col2:
    st.subheader("Chunk Topic Entropy")
    theta_norm = THETA.div(THETA.sum(axis=1), axis=0).fillna(0)
    entropy = theta_norm.apply(scipy_entropy, axis=1)
    fig_ent = px.histogram(
        entropy, nbins=30,
        labels={"value": "Entropy", "count": "Chunks"},
    )
    fig_ent.update_layout(height=350, margin=_m, showlegend=False)
    st.plotly_chart(fig_ent, use_container_width=True)

st.divider()

col3, col4 = st.columns(2)

if elbow_df is not None:
    elbow_df["marginal_gain"] = elbow_df["error"].diff(-1).fillna(0)

    with col3:
        st.subheader("Reconstruction Error")
        fig_elbow = px.line(
            elbow_df, x="n_topics", y="error", markers=True,
            labels={"n_topics": "Number of topics", "error": "Reconstruction error"},
        )
        fig_elbow.add_vline(x=n_topics, line_dash="dash", line_color="gray",
                            annotation_text="current", annotation_position="top right")
        fig_elbow.update_layout(height=350, margin=_m)
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col4:
        st.subheader("Marginal Gain")
        fig_gain = px.bar(
            elbow_df, x="n_topics", y="marginal_gain",
            labels={"n_topics": "Number of topics", "marginal_gain": "Drop in error per topic added"},
        )
        fig_gain.add_vline(x=n_topics, line_dash="dash", line_color="gray",
                           annotation_text="current", annotation_position="top right")
        fig_gain.update_layout(height=350, margin=_m)
        st.plotly_chart(fig_gain, use_container_width=True)
