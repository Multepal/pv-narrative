"""
Parameter Grid Search (PCA/LSA) — same as grid_search.py but inserts
TruncatedSVD (n_components=10) between TF-IDF and HAC.

k is swept cheaply post-linkage (fcluster only). Expensive SVD+linkage runs
are cached, so the full grid costs one upfront computation then stays instant.
"""

import os
import yaml
import itertools
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import pdist, hamming
from scipy.cluster.hierarchy import linkage, fcluster
from toc import render_toc

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
LANG_LABELS  = cfg["languages"]
K_VALS       = list(range(2, 21))
N_COMPONENTS = 10


def find_token_file(src_id: str) -> str | None:
    candidates = [
        os.path.join(APP_DIR, f"../../notebooks/{src_id}/{src_id}-TOKEN.csv"),
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


@st.cache_data(show_spinner=False)
def run_linkage(src_id, token_path, n_chunks, min_df, max_df, ngram_range=(1, 1)):
    TOKEN = load_tokens(src_id, token_path)
    token_reset = TOKEN.reset_index()
    token_reset["chunk_num"] = pd.cut(
        token_reset.index, n_chunks, labels=list(range(n_chunks))
    )
    chunks_s = (
        token_reset.groupby("chunk_num", observed=True)["term_str"]
        .apply(lambda x: " ".join(x.dropna()))
    )
    chunks_list = chunks_s.tolist()
    if len(chunks_list) < 3:
        return None
    try:
        vec = TfidfVectorizer(
            lowercase=True, max_df=max_df, min_df=min_df,
            strip_accents=None, norm="l2", ngram_range=ngram_range,
        )
        X = vec.fit_transform(chunks_list)
    except ValueError:
        return None
    n_comp = min(N_COMPONENTS, X.shape[0] - 1, X.shape[1] - 1)
    if n_comp < 2:
        return None
    X_pca = TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(X)
    Z = linkage(pdist(X_pca, metric="euclidean"), method="ward")
    return {"Z": Z, "n_chunks": len(chunks_list)}


def threshold_for_k(Z, k, n):
    k = max(2, min(k, n - 1))
    return float((Z[n - k - 1, 2] + Z[n - k, 2]) / 2)


def cluster_string(labels) -> str:
    mapping: dict = {}
    result = []
    for lbl in labels:
        if lbl not in mapping:
            mapping[lbl] = chr(65 + len(mapping))
        result.append(mapping[lbl])
    return "".join(result)


def mean_pairwise_hamming(strings: list[str]) -> float:
    n = len(strings)
    if n < 2:
        return float("nan")
    return float(np.mean([
        hamming(list(strings[i]), list(strings[j]))
        for i in range(n) for j in range(i + 1, n)
    ]))


# ── Controls ──────────────────────────────────────────────────────────────────
st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)

st.title("Parameter Grid Search — PCA/LSA")
render_toc([
    ("Hamming vs. k",          "hamming-chart"),
    ("Optimal k Distribution", "k-distribution"),
    ("Summary",                "summary"),
])
st.caption(
    f"Same as Parameter Grid Search but uses TF-IDF → LSA (TruncatedSVD, "
    f"n_components={N_COMPONENTS}) → HAC instead of raw TF-IDF → HAC. "
    "For each combination of non-k parameters, SVD+linkage is computed once per edition "
    "then k is swept from 2–20 cheaply. Results are cached after the first run."
)

col1, col2 = st.columns(2)

nc_range = col1.slider("n_chunks range", min_value=5, max_value=50, value=(10, 40), step=5)
nc_vals  = list(range(nc_range[0], nc_range[1] + 1, 5))

maxdf_range = col2.slider("max_df range", min_value=0.20, max_value=0.95, value=(0.30, 0.70), step=0.05, format="%.2f")
_n_maxdf    = round((maxdf_range[1] - maxdf_range[0]) / 0.05) + 1
maxdf_vals  = [round(maxdf_range[0] + i * 0.05, 2) for i in range(_n_maxdf)]

FIXED_MIN_DF  = 5
FIXED_NGRAM   = (1, 1)

n_combos = len(nc_vals) * len(maxdf_vals)
st.caption(f"**{n_combos}** combinations · **{n_combos * len(SOURCES_META)}** SVD+linkage runs (est.)  ·  min_df={FIXED_MIN_DF} · ngram=(1,1) · n_components={N_COMPONENTS} fixed")

# ── Grid computation ───────────────────────────────────────────────────────────
combos      = list(itertools.product(nc_vals, maxdf_vals))
token_files = {src_id: find_token_file(src_id) for src_id in SOURCES_META}

curve_rows   = []
summary_rows = []
progress     = st.progress(0, text="Running grid…")

for ci, (nc, mxdf) in enumerate(combos):
    progress.progress(
        (ci + 1) / len(combos),
        text=f"combo {ci + 1}/{len(combos)}  ·  n_chunks={nc}  max_df={mxdf}",
    )

    linkage_cache = {}
    for src_id in SOURCES_META:
        tp = token_files[src_id]
        if tp is None:
            continue
        result = run_linkage(src_id, tp, nc, FIXED_MIN_DF, mxdf, FIXED_NGRAM)
        if result is not None:
            linkage_cache[src_id] = result

    if len(linkage_cache) < 2:
        continue

    combo_label = f"nc={nc} · max_df={mxdf:.2f}"
    combo_k_rows = []

    for k in K_VALS:
        strings = []
        for src_id, result in linkage_cache.items():
            Z, n = result["Z"], result["n_chunks"]
            labels = fcluster(Z, threshold_for_k(Z, k, n), criterion="distance")
            strings.append(cluster_string(labels))
        mh = mean_pairwise_hamming(strings)
        combo_k_rows.append({"k": k, "mean_hamming": mh})
        curve_rows.append({
            "combo_label": combo_label,
            "n_chunks": nc, "max_df": mxdf,
            "k": k, "mean_hamming": mh,
        })

    best = min(combo_k_rows, key=lambda r: r["mean_hamming"])
    summary_rows.append({
        "combo_label": combo_label,
        "n_chunks": nc, "max_df": mxdf,
        "k*": int(best["k"]),
        "min_hamming": round(float(best["mean_hamming"]), 4),
    })

progress.empty()

if not curve_rows:
    st.warning("No combinations produced valid results. Try adjusting the grid.")
    st.stop()

df_curves  = pd.DataFrame(curve_rows)
df_summary = pd.DataFrame(summary_rows).sort_values("min_hamming").reset_index(drop=True)

# ── Spaghetti plot ─────────────────────────────────────────────────────────────
st.subheader("Hamming vs. k — All Combinations", anchor="hamming-chart")
st.caption(
    "Each gray curve = one parameter combination. "
    "Bold blue = mean across all combinations. "
    "A consistent trough indicates a robust optimal k."
)

fig = go.Figure()

for label, gdf in df_curves.groupby("combo_label", sort=False):
    fig.add_trace(go.Scatter(
        x=gdf["k"], y=gdf["mean_hamming"],
        mode="lines",
        line=dict(color="#CCCCCC", width=1),
        showlegend=False,
        hovertemplate=f"{label}<br>k=%{{x}}<br>Hamming=%{{y:.3f}}<extra></extra>",
    ))

mean_curve = df_curves.groupby("k")["mean_hamming"].mean().reset_index()
fig.add_trace(go.Scatter(
    x=mean_curve["k"], y=mean_curve["mean_hamming"],
    mode="lines+markers",
    line=dict(color="#1f77b4", width=3),
    marker=dict(size=6),
    name="Mean across combos",
    hovertemplate="mean · k=%{x}<br>Hamming=%{y:.3f}<extra></extra>",
))

fig.update_layout(
    height=400,
    margin=dict(l=60, r=30, t=10, b=50),
    plot_bgcolor="white",
    xaxis=dict(title="k (clusters)", dtick=1, showgrid=False, zeroline=False),
    yaxis=dict(title="Mean Hamming", range=[0, 1],
               showgrid=True, gridcolor="#EEEEEE", zeroline=False),
    legend=dict(x=0.02, y=0.98),
)
st.plotly_chart(fig, width="stretch")

# ── k* distribution ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Distribution of Optimal k*", anchor="k-distribution")
st.caption("How many parameter combinations achieve their minimum Hamming distance at each k.")

k_star_counts = (
    df_summary["k*"].value_counts()
    .reindex(range(2, 21), fill_value=0)
    .reset_index()
)
k_star_counts.columns = ["k*", "count"]

fig_bar = px.bar(k_star_counts, x="k*", y="count",
                 labels={"k*": "k (clusters)", "count": "# combinations"})
fig_bar.update_layout(
    height=280,
    margin=dict(l=60, r=30, t=10, b=50),
    plot_bgcolor="white",
    xaxis=dict(dtick=1, showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False),
)
st.plotly_chart(fig_bar, width="stretch")

# ── Summary table ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("Summary — Sorted by Minimum Hamming Distance", anchor="summary")
st.caption("Most coherent parameter combinations first.")
st.dataframe(df_summary, use_container_width=True, hide_index=True)
