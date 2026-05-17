"""
Narrative DCT — Discrete Cosine Transform of cluster presence signals.

Applies DCT-II to each cluster's binary narrative presence signal using three
clustering methods in parallel: TF-IDF → HAC, TF-IDF → PCA/LSA → HAC, and
TF-IDF → NMF. Low-frequency DCT coefficients capture long narrative arcs;
high-frequency coefficients capture rapid oscillation. Low-pass reconstruction
(IDCT of the first N coefficients) gives the structural envelope of each cluster.

Layout: for each analysis section, method is a row and clusters are columns.
"""

import os
import yaml
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.fft import dct, idct
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from toc import render_toc
from utils import find_token_file, load_tokens, make_chunks, threshold_for_k

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
LANG_LABELS  = cfg["languages"]
NMF_MAX_ITER = cfg["model"]["nmf_max_iter"]

st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)


def _tfidf_matrix(chunks_list, min_df, max_df):
    if len(chunks_list) < 3:
        return None
    try:
        vec = TfidfVectorizer(lowercase=True, max_df=max_df, min_df=min_df,
                              strip_accents=None, norm="l2")
        return vec.fit_transform(chunks_list)
    except ValueError:
        return None


@st.cache_data(show_spinner=False)
def run_tfidf_hac(src_id, token_path, n_chunks, min_df, max_df):
    chunks = make_chunks(src_id, token_path, n_chunks)
    X = _tfidf_matrix(chunks, min_df, max_df)
    if X is None:
        return None
    Z = linkage(pdist(X.toarray(), metric="euclidean"), method="ward")
    return {"Z": Z, "n_chunks": len(chunks)}


@st.cache_data(show_spinner=False)
def run_pca_hac(src_id, token_path, n_chunks, min_df, max_df, n_components):
    chunks = make_chunks(src_id, token_path, n_chunks)
    X = _tfidf_matrix(chunks, min_df, max_df)
    if X is None:
        return None
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1] - 1)
    if n_comp < 2:
        return None
    THETA = TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(X)
    return {"THETA": THETA, "n_chunks": len(chunks)}


@st.cache_data(show_spinner=False)
def run_nmf(src_id, token_path, n_chunks, min_df, max_df, k):
    chunks = make_chunks(src_id, token_path, n_chunks)
    if len(chunks) < max(3, k):
        return None
    X = _tfidf_matrix(chunks, min_df, max_df)
    if X is None or X.shape[1] < k:
        return None
    try:
        model = NMF(n_components=k, init="nndsvda", max_iter=NMF_MAX_ITER)
        THETA = model.fit_transform(X)
    except Exception:
        return None
    return {"labels": np.argmax(THETA, axis=1), "n_chunks": len(chunks)}


@st.cache_data(show_spinner=False)
def run_sim_hac(src_id, token_path, n_chunks, min_df, max_df):
    chunks = make_chunks(src_id, token_path, n_chunks)
    X = _tfidf_matrix(chunks, min_df, max_df)
    if X is None:
        return None
    SIM = (X @ X.T).toarray()
    Z = linkage(pdist(SIM, metric="euclidean"), method="ward")
    return {"Z": Z, "n_chunks": len(chunks)}


def normalize_labels(raw):
    """Map any label array to 0-based first-appearance order."""
    mapping = {}
    out = np.zeros(len(raw), dtype=int)
    for i, lbl in enumerate(raw):
        if lbl not in mapping:
            mapping[lbl] = len(mapping)
        out[i] = mapping[lbl]
    return out


def hac_labels(Z, k, n):
    raw = fcluster(Z, threshold_for_k(Z, k, n), criterion="distance")
    return normalize_labels(raw)


def pca_labels(THETA, k):
    Z = linkage(pdist(THETA, metric="euclidean"), method="ward")
    n = THETA.shape[0]
    return hac_labels(Z, k, n)


def binary_signals(labels, k):
    n = len(labels)
    mat = np.zeros((k, n))
    for t, c in enumerate(labels):
        if c < k:
            mat[c, t] = 1.0
    return mat


def low_pass(signal, n_keep):
    coeffs = dct(signal, type=2, norm="ortho")
    filtered = np.zeros_like(coeffs)
    filtered[:n_keep] = coeffs[:n_keep]
    return idct(filtered, type=2, norm="ortho")


# ── Controls ──────────────────────────────────────────────────────────────────
st.title("Narrative DCT Analysis — Popol Wuj")
render_toc([
    ("DCT Spectra",             "dct-spectra"),
    ("Low-pass Reconstruction", "low-pass"),
    ("Cross-edition Envelopes", "cross-edition"),
])
st.caption(
    "Applies the Discrete Cosine Transform to each cluster's binary presence signal "
    "across narrative time, using three clustering methods in parallel. "
    "**Low-frequency coefficients** = long narrative arcs; "
    "**high-frequency** = rapid oscillation. "
    "Within each section, rows are methods and columns are clusters."
)

src_ids = list(SOURCES_META.keys())
_c      = cfg["controls"]
_col_r  = cfg["layout"]["column_ratios"]
cols    = st.columns(_col_r[:5])

src_id    = cols[0].selectbox(
    "Source", src_ids, index=src_ids.index(_c["default_source"]),
    format_func=lambda x: f"{SOURCES_META[x]['label']} ({LANG_LABELS[SOURCES_META[x]['lang']]})")
n_chunks  = cols[1].number_input("n_chunks",  min_value=_c["n_chunks"]["min"], max_value=_c["n_chunks"]["max"], value=_c["n_chunks"]["default"], step=_c["n_chunks"]["step"])
min_df    = cols[2].number_input("min_df",    _c["min_df"]["min"], _c["min_df"]["max"],
                                  _c["min_df"]["default"], step=_c["min_df"]["step"])
max_df    = cols[2].number_input("max_df",    _c["max_df"]["min"], _c["max_df"]["max"],
                                  _c["max_df"]["default"], step=_c["max_df"]["step"], format="%.2f")
k         = cols[3].number_input("k (clusters)", _c["n_clusters"]["min"], _c["n_clusters"]["max"],
                                  _c["n_clusters"]["default"], step=1)
n_components = cols[4].number_input("PCA components", 2, _c["n_topics"]["max"], 10, step=1)

n_keep_max = max(2, min(12, int(n_chunks) // 2))
n_keep = st.slider("Low-pass cutoff (DCT coefficients to keep)", 1, n_keep_max,
                   min(4, n_keep_max), step=1,
                   help="Keep the first N DCT coefficients; reconstruct with IDCT. Lower = smoother.")

st.divider()

# ── Load token file ────────────────────────────────────────────────────────────
token_path = find_token_file(src_id)
if token_path is None:
    st.warning(f"Token file not found for `{src_id}`.")
    st.stop()

_k  = int(k)
_nc = int(n_chunks)
_mn = int(min_df)
_mx = float(max_df)
_np = int(n_components)

# ── Run all three models for the selected edition ──────────────────────────────
res_tfidf = run_tfidf_hac(src_id, token_path, _nc, _mn, _mx)
res_pca   = run_pca_hac(src_id, token_path, _nc, _mn, _mx, _np)
res_nmf   = run_nmf(src_id, token_path, _nc, _mn, _mx, _k)
res_sim   = run_sim_hac(src_id, token_path, _nc, _mn, _mx)

if res_tfidf is None and res_pca is None and res_nmf is None and res_sim is None:
    st.warning("All models failed — try adjusting parameters.")
    st.stop()

n_act     = (res_tfidf or res_pca or res_nmf or res_sim)["n_chunks"]
x_pos     = np.linspace(1, 100, n_act)
freq_bins = np.arange(n_act)
_show_bins = min(n_act, 16)
_palette   = px.colors.qualitative.Plotly


def _build_model_data(res_t, res_p, res_n, res_s, k_val):
    """Return list of (method_label, signals, coefficients, k_act) for each available model."""
    methods = []

    if res_t is not None:
        lbl = hac_labels(res_t["Z"], k_val, res_t["n_chunks"])
        ka  = int(lbl.max()) + 1
        sig = binary_signals(lbl, ka)
        methods.append(("TF-IDF → HAC", sig, np.array([dct(r, type=2, norm="ortho") for r in sig]), ka))

    if res_p is not None:
        lbl = pca_labels(res_p["THETA"], k_val)
        ka  = int(lbl.max()) + 1
        sig = binary_signals(lbl, ka)
        methods.append(("PCA/LSA → HAC", sig, np.array([dct(r, type=2, norm="ortho") for r in sig]), ka))

    if res_n is not None:
        lbl = normalize_labels(res_n["labels"])
        ka  = int(lbl.max()) + 1
        sig = binary_signals(lbl, ka)
        methods.append(("NMF", sig, np.array([dct(r, type=2, norm="ortho") for r in sig]), ka))

    if res_s is not None:
        lbl = hac_labels(res_s["Z"], k_val, res_s["n_chunks"])
        ka  = int(lbl.max()) + 1
        sig = binary_signals(lbl, ka)
        methods.append(("Cosine-Sim → HAC", sig, np.array([dct(r, type=2, norm="ortho") for r in sig]), ka))

    return methods


model_data = _build_model_data(res_tfidf, res_pca, res_nmf, res_sim, _k)

# ── Section 1: DCT Spectra ─────────────────────────────────────────────────────
st.subheader("DCT Spectra", anchor="dct-spectra")
st.caption(
    "Magnitude of each DCT coefficient for each cluster's binary presence signal. "
    "Bin 0 = DC (mean presence); bins 1–N = increasing frequency. "
    "The dashed line marks the low-pass cutoff."
)

for mi, (method_name, signals_m, coeffs_m, k_act_m) in enumerate(model_data):
    cluster_names_m = [f"Cluster {chr(65 + i)}" for i in range(k_act_m)]
    colors_m = [_palette[i % len(_palette)] for i in range(k_act_m)]

    st.markdown(f"**{method_name}**")
    fig = go.Figure()
    for i in range(k_act_m):
        fig.add_trace(go.Scatter(
            x=freq_bins[:_show_bins],
            y=np.abs(coeffs_m[i, :_show_bins]),
            mode="lines+markers",
            name=cluster_names_m[i],
            line=dict(color=colors_m[i], width=2),
            marker=dict(size=6),
            hovertemplate=f"{cluster_names_m[i]}<br>bin %{{x}}<br>|coeff| = %{{y:.4f}}<extra></extra>",
        ))
    fig.add_vline(x=n_keep - 0.5, line_dash="dash", line_color="gray", line_width=1.5,
                  annotation_text=f"cutoff = {n_keep}", annotation_position="top right")
    fig.update_layout(
        height=280,
        margin=dict(l=60, r=30, t=20, b=50),
        plot_bgcolor="white",
        xaxis=dict(title="DCT frequency bin", dtick=1, showgrid=False, zeroline=False,
                   range=[-0.5, _show_bins - 0.5]),
        yaxis=dict(title="|DCT coefficient|", showgrid=True, gridcolor="#EEEEEE", zeroline=False),
        legend=dict(x=0.75, y=0.98),
    )
    st.plotly_chart(fig, width="stretch", key=f"spec_{mi}")

# ── Section 2: Low-pass Reconstruction ────────────────────────────────────────
st.divider()
st.subheader("Low-pass Reconstruction", anchor="low-pass")
st.caption(
    f"IDCT of the first **{n_keep}** coefficients — the structural envelope of each cluster's "
    "narrative presence. Dashed = raw binary signal; solid = smooth reconstruction."
)

for mi, (method_name, signals_m, _, k_act_m) in enumerate(model_data):
    cluster_names_m = [f"Cluster {chr(65 + i)}" for i in range(k_act_m)]
    colors_m = [_palette[i % len(_palette)] for i in range(k_act_m)]
    n_cols = min(k_act_m, 3)
    n_rows = math.ceil(k_act_m / n_cols)

    st.markdown(f"**{method_name}**")
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cluster_names_m,
                        horizontal_spacing=0.06, vertical_spacing=0.18)

    for ci in range(k_act_m):
        row = ci // n_cols + 1
        col = ci % n_cols + 1
        recon = low_pass(signals_m[ci], n_keep)
        fig.add_trace(go.Scatter(
            x=x_pos, y=signals_m[ci], mode="lines",
            line=dict(color=colors_m[ci], width=1, dash="dot"),
            opacity=0.3, showlegend=False, hoverinfo="skip",
        ), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=x_pos, y=recon, mode="lines",
            name=cluster_names_m[ci],
            line=dict(color=colors_m[ci], width=2.5),
            showlegend=False,
            hovertemplate=f"{cluster_names_m[ci]}<br>pos %{{x:.0f}}<br>env = %{{y:.3f}}<extra></extra>",
        ), row=row, col=col)

    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE", zeroline=True, zerolinecolor="#CCCCCC")
    fig.update_layout(
        height=200 * n_rows,
        margin=dict(l=50, r=20, t=40, b=40),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, width="stretch", key=f"lp_{mi}")

# ── Section 3: Cross-edition Envelopes ────────────────────────────────────────
st.divider()
st.subheader("Cross-edition Envelopes", anchor="cross-edition")
st.caption(
    "Low-pass reconstruction for every available edition using the same parameters. "
    "Clusters are aligned by **order of first appearance**. "
    "Convergence of lines = editions agree on the narrative rhythm; divergence = structural disagreement."
)

# Collect all-edition results for each method
all_ed = {"tfidf": {}, "pca": {}, "nmf": {}, "sim": {}}
missing = []
src_list = list(SOURCES_META.keys())
progress = st.progress(0, text="Computing cross-edition models…")

for ei, sid in enumerate(src_list):
    progress.progress((ei + 1) / len(src_list), text=f"{SOURCES_META[sid]['label']}")
    tp = find_token_file(sid)
    if tp is None:
        missing.append(sid)
        continue

    r = run_tfidf_hac(sid, tp, _nc, _mn, _mx)
    if r is not None:
        all_ed["tfidf"][sid] = hac_labels(r["Z"], _k, r["n_chunks"])

    r = run_pca_hac(sid, tp, _nc, _mn, _mx, _np)
    if r is not None:
        all_ed["pca"][sid] = pca_labels(r["THETA"], _k)

    r = run_nmf(sid, tp, _nc, _mn, _mx, _k)
    if r is not None:
        all_ed["nmf"][sid] = normalize_labels(r["labels"])

    r = run_sim_hac(sid, tp, _nc, _mn, _mx)
    if r is not None:
        all_ed["sim"][sid] = hac_labels(r["Z"], _k, r["n_chunks"])

progress.empty()

if missing:
    st.caption(f"Editions skipped (no token file): {', '.join(missing)}")

_edition_colors = px.colors.qualitative.Dark24


def _render_cross_edition(method_name, ed_labels_dict, k_target, key):
    if len(ed_labels_dict) < 2:
        st.info(f"Need at least 2 editions for {method_name}.")
        return

    # Low-pass envelope per edition: shape (k_target, n_act)
    ed_envelopes = {}
    for sid, labels in ed_labels_dict.items():
        k_ed = int(labels.max()) + 1
        sig  = binary_signals(labels, k_ed)
        ed_envelopes[sid] = np.array([
            low_pass(sig[i], n_keep) if i < k_ed else np.zeros(n_act)
            for i in range(k_target)
        ])

    cluster_names_m = [f"Cluster {chr(65 + i)}" for i in range(k_target)]
    n_cols = min(k_target, 3)
    n_rows = math.ceil(k_target / n_cols)

    st.markdown(f"**{method_name}**")
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=cluster_names_m,
                        horizontal_spacing=0.06, vertical_spacing=0.14)

    for ci in range(k_target):
        row = ci // n_cols + 1
        col = ci % n_cols + 1
        for ei, (sid, recons) in enumerate(ed_envelopes.items()):
            meta = SOURCES_META[sid]
            fig.add_trace(go.Scatter(
                x=x_pos, y=recons[ci],
                mode="lines",
                name=meta["label"],
                legendgroup=sid,
                showlegend=(ci == 0),
                line=dict(color=_edition_colors[ei % len(_edition_colors)], width=1.8),
                hovertemplate=(
                    f"{meta['label']}<br>pos %{{x:.0f}}<br>envelope = %{{y:.3f}}<extra></extra>"
                ),
            ), row=row, col=col)

    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE", zeroline=True, zerolinecolor="#CCCCCC")
    fig.update_layout(
        height=280 * n_rows,
        margin=dict(l=40, r=180, t=40, b=40),
        plot_bgcolor="white",
        legend=dict(
            x=1.02, y=0.5,
            xanchor="left", yanchor="middle",
            font=dict(size=10),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#CCCCCC",
            borderwidth=1,
        ),
    )
    st.plotly_chart(fig, width="stretch", key=f"ce_{key}")


_render_cross_edition("TF-IDF → HAC",     all_ed["tfidf"], _k, "tfidf")
_render_cross_edition("PCA/LSA → HAC",   all_ed["pca"],   _k, "pca")
_render_cross_edition("NMF",             all_ed["nmf"],   _k, "nmf")
_render_cross_edition("Cosine-Sim → HAC", all_ed["sim"],  _k, "sim")
