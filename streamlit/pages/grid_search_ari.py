"""
Parameter Grid Search (ARI) — sweep all combinations of non-k parameters and
report how cross-edition Adjusted Rand Index varies with k for each combination.

k is swept cheaply post-linkage (fcluster only). Expensive linkage runs are
cached, so the full grid costs one upfront computation then stays instant.

ARI operates on raw integer label arrays and is permutation-invariant, so it
correctly scores structurally identical clusterings regardless of label order.
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
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from toc import render_toc

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
LANG_LABELS  = cfg["languages"]
K_VALS       = list(range(2, 21))


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
    tfidf_dense = X.toarray()
    Z = linkage(pdist(tfidf_dense, metric="euclidean"), method="ward")
    return {"Z": Z, "n_chunks": len(chunks_list)}


def threshold_for_k(Z, k, n):
    k = max(2, min(k, n - 1))
    return float((Z[n - k - 1, 2] + Z[n - k, 2]) / 2)


def mean_pairwise_ari(label_arrays: list[np.ndarray]) -> float:
    n = len(label_arrays)
    if n < 2:
        return float("nan")
    scores = [
        adjusted_rand_score(label_arrays[i], label_arrays[j])
        for i in range(n) for j in range(i + 1, n)
    ]
    return float(np.mean(scores))


# ── Controls ──────────────────────────────────────────────────────────────────
st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)

st.title("Parameter Grid Search (ARI)")
render_toc([
    ("ARI vs. k",             "ari-chart"),
    ("Optimal k Distribution","k-distribution"),
    ("Best Parameters",       "best-params"),
    ("Summary",               "summary"),
])
st.caption(
    "For each combination of non-k parameters, linkage is computed once per edition "
    "then k is swept from 2–20 cheaply. Reports the optimal k* and maximum mean pairwise "
    "Adjusted Rand Index for each combination. "
    "ARI is permutation-invariant: structurally identical clusterings always score 1.0 "
    "regardless of label assignment order. Results are cached after the first run."
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
st.caption(f"**{n_combos}** combinations · **{n_combos * len(SOURCES_META)}** linkage runs (est.)  ·  min_df={FIXED_MIN_DF} · ngram=(1,1) fixed")

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
        label_arrays = []
        for src_id, result in linkage_cache.items():
            Z, n = result["Z"], result["n_chunks"]
            labels = fcluster(Z, threshold_for_k(Z, k, n), criterion="distance")
            label_arrays.append(labels)
        ari = mean_pairwise_ari(label_arrays)
        combo_k_rows.append({"k": k, "mean_ari": ari})
        curve_rows.append({
            "combo_label": combo_label,
            "n_chunks": nc, "max_df": mxdf,
            "k": k, "mean_ari": ari,
        })

    best = max(combo_k_rows, key=lambda r: r["mean_ari"])
    summary_rows.append({
        "combo_label": combo_label,
        "n_chunks": nc, "max_df": mxdf,
        "k*": int(best["k"]),
        "max_ari": round(float(best["mean_ari"]), 4),
    })

progress.empty()

if not curve_rows:
    st.warning("No combinations produced valid results. Try adjusting the grid.")
    st.stop()

df_curves  = pd.DataFrame(curve_rows)
df_summary = pd.DataFrame(summary_rows).sort_values("max_ari", ascending=False).reset_index(drop=True)

# ── Spaghetti plot ─────────────────────────────────────────────────────────────
st.subheader("ARI vs. k — All Combinations", anchor="ari-chart")
st.caption(
    "Each gray curve = one parameter combination. "
    "Bold blue = mean across all combinations. "
    "A consistent peak indicates a robust optimal k."
)

fig = go.Figure()

for label, gdf in df_curves.groupby("combo_label", sort=False):
    fig.add_trace(go.Scatter(
        x=gdf["k"], y=gdf["mean_ari"],
        mode="lines",
        line=dict(color="#CCCCCC", width=1),
        showlegend=False,
        hovertemplate=f"{label}<br>k=%{{x}}<br>ARI=%{{y:.3f}}<extra></extra>",
    ))

mean_curve = df_curves.groupby("k")["mean_ari"].mean().reset_index()
fig.add_trace(go.Scatter(
    x=mean_curve["k"], y=mean_curve["mean_ari"],
    mode="lines+markers",
    line=dict(color="#1f77b4", width=3),
    marker=dict(size=6),
    name="Mean across combos",
    hovertemplate="mean · k=%{x}<br>ARI=%{y:.3f}<extra></extra>",
))

fig.update_layout(
    height=400,
    margin=dict(l=60, r=30, t=10, b=50),
    plot_bgcolor="white",
    xaxis=dict(title="k (clusters)", dtick=1, showgrid=False, zeroline=False),
    yaxis=dict(title="Mean ARI", range=[-0.05, 1.05],
               showgrid=True, gridcolor="#EEEEEE", zeroline=False),
    legend=dict(x=0.02, y=0.02),
)
st.plotly_chart(fig, width="stretch")

# ── k* distribution ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Distribution of Optimal k*", anchor="k-distribution")
st.caption("How many parameter combinations achieve their maximum ARI at each k.")

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

# ── Best Parameters ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Best Parameters", anchor="best-params")

# Exclude k=2: ARI peaks trivially at the minimum k; find the genuine secondary maximum
_df_excl2   = df_curves[df_curves["k"] >= 3]
_mean_by_k  = _df_excl2.groupby("k")["mean_ari"].mean()
_k_star     = int(_mean_by_k.idxmax())
_best       = _df_excl2[_df_excl2["k"] == _k_star].nlargest(1, "mean_ari").iloc[0]

st.caption(
    f"k=2 is excluded from the optimal search — ARI peaks trivially at the minimum k, "
    f"dips as k increases, then rises to a genuine maximum at k={_k_star} before declining at higher k."
)

_c1, _c2, _c3, _c4 = st.columns(4)
_c1.metric("Best n_chunks",    int(_best["n_chunks"]))
_c2.metric("Best max_df",      f"{_best['max_df']:.2f}")
_c3.metric("Optimal k*",       _k_star)
_c4.metric("Max ARI at k*",    f"{_best['mean_ari']:.4f}")

# ARI at k* specifically (keeps heatmap, callouts, and marginal bars consistent)
_df_at_kstar = (
    df_curves[df_curves["k"] == _k_star]
    [["n_chunks", "max_df", "mean_ari"]]
    .reset_index(drop=True)
)

_pivot = _df_at_kstar.pivot(index="n_chunks", columns="max_df", values="mean_ari")
_fig_heat = px.imshow(
    _pivot,
    labels=dict(x="max_df", y="n_chunks", color=f"ARI at k={_k_star}"),
    color_continuous_scale="Blues",
    aspect="auto",
    text_auto=".3f",
)
_fig_heat.update_layout(height=300, margin=dict(l=60, r=30, t=30, b=50))
st.plotly_chart(_fig_heat, width="stretch")

_col_nc, _col_mdf = st.columns(2)
_by_nc  = _df_at_kstar.groupby("n_chunks")["mean_ari"].mean().reset_index()
_by_mdf = _df_at_kstar.groupby("max_df")["mean_ari"].mean().reset_index()

_fig_nc = px.bar(_by_nc, x="n_chunks", y="mean_ari",
                 labels={"n_chunks": "n_chunks", "mean_ari": f"Mean ARI at k={_k_star}"})
_fig_nc.update_layout(
    height=240, margin=dict(l=60, r=30, t=10, b=50),
    plot_bgcolor="white",
    xaxis=dict(dtick=5, showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False),
)
_col_nc.plotly_chart(_fig_nc, width="stretch")

_fig_mdf = px.bar(_by_mdf, x="max_df", y="mean_ari",
                  labels={"max_df": "max_df", "mean_ari": f"Mean ARI at k={_k_star}"})
_fig_mdf.update_layout(
    height=240, margin=dict(l=60, r=30, t=10, b=50),
    plot_bgcolor="white",
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False),
)
_col_mdf.plotly_chart(_fig_mdf, width="stretch")

# Per-k table
st.subheader("Best Parameters by k")
st.caption("For each k, the (n_chunks, max_df) combination yielding the highest mean pairwise ARI.")

_best_by_k = (
    df_curves
    .loc[df_curves.groupby("k")["mean_ari"].idxmax()]
    [["k", "n_chunks", "max_df", "mean_ari"]]
    .sort_values("k")
    .reset_index(drop=True)
    .rename(columns={"mean_ari": "ARI"})
)
st.dataframe(_best_by_k, use_container_width=True, hide_index=True)

# ── Summary table ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("Summary — Sorted by Maximum ARI", anchor="summary")
st.caption("Most coherent parameter combinations first (higher ARI = more agreement across editions).")
st.dataframe(df_summary, use_container_width=True, hide_index=True)
