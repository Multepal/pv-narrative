"""
Parameter Grid Search — Boundary Concordance (TF-IDF/HAC)
"""

import os
import yaml
import itertools
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster.hierarchy import fcluster
from toc import render_toc
from grid_search_boundary_core import (
    find_token_file, load_tokens, run_linkage_tfidf,
    threshold_for_k, get_boundaries, boundary_f1, mean_pairwise_boundary_f1,
    FIXED_MIN_DF, FIXED_NGRAM,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
K_VALS       = list(range(2, 21))

# ── Controls ──────────────────────────────────────────────────────────────────
st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)

st.title("Grid Search — Boundary Concordance (TF-IDF)")
render_toc([
    ("Concordance vs. k",      "concordance-chart"),
    ("Optimal k Distribution", "k-distribution"),
    ("Best Parameters",        "best-params"),
    ("Summary",                "summary"),
])
st.caption(
    "TF-IDF → Ward HAC (no SVD). "
    "Each edition's cluster assignment is converted to normalized boundary positions in [0, 1]. "
    "Pairwise boundary F1 is averaged across all edition pairs. "
    "**Tolerance scales with k**: `tol = scale / (2 × (k − 1))` — adjust the scale slider to "
    "tighten or loosen matching. k is swept cheaply post-linkage; TF-IDF+linkage results are cached."
)

with st.expander("Tolerance parameter — justification"):
    st.markdown(
        r"""
The tolerance is set as $\tau = \alpha \;/\; 2(k-1)$, where $\alpha$ is the scale
factor below (default **0.75**). The matching rule is: boundary $b_1$ (from one
edition) matches boundary $b_2$ (from another) if $|b_1 - b_2| \leq \tau$.
For this assignment to be *unambiguous* — so that no single boundary in one
edition could plausibly match two distinct boundaries in the other — the matching
window $2\tau$ must be strictly smaller than the minimum distance between adjacent
boundaries. Under $k$ clusters evenly spaced at $1/(k-1)$, the **non-aliasing
condition** is:

$$2\tau < \frac{1}{k-1} \;\Longrightarrow\; \alpha < 1.0$$

The natural default $\alpha = 1.0$ sits exactly at the aliasing boundary: a
boundary placed midway between two true boundaries is equidistant from both, and
its assignment is decided only by a tiebreak. Setting $\alpha = 0.75$ gives a
matching window of $0.75/(k-1)$, leaving a 12.5% dead zone on either side of
each expected boundary position — every boundary unambiguously belongs to one
side or the other. In concrete terms, at $n = 25$ chunks and $k = 6$ this
corresponds to a window of roughly ±1.9 chunks, which accommodates the
one-to-two-chunk positional variation attributable to differences in segmentation
and translation between editions without reaching the next segment boundary.
        """
    )

col1, col2, col3 = st.columns(3)

nc_range = col1.slider("n_chunks range", min_value=15, max_value=50, value=(15, 40), step=5)
nc_vals  = list(range(nc_range[0], nc_range[1] + 1, 5))

maxdf_range = col2.slider("max_df range", min_value=0.20, max_value=0.95, value=(0.30, 0.70), step=0.05, format="%.2f")
_n_maxdf    = round((maxdf_range[1] - maxdf_range[0]) / 0.05) + 1
maxdf_vals  = [round(maxdf_range[0] + i * 0.05, 2) for i in range(_n_maxdf)]

tol_scale = col3.slider("Tolerance scale", min_value=0.25, max_value=2.0, value=0.75, step=0.25,
    help="Multiplier on τ = 1/(2(k−1)). Values < 1 tighten matching; > 1 loosen it.")

n_combos = len(nc_vals) * len(maxdf_vals)
st.caption(
    f"**{n_combos}** combinations · **{n_combos * len(SOURCES_META)}** TF-IDF+linkage runs (est.)  "
    f"·  min_df={FIXED_MIN_DF} · ngram=(1,1) fixed"
)

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
        result = run_linkage_tfidf(src_id, tp, nc, FIXED_MIN_DF, mxdf, FIXED_NGRAM)
        if result is not None:
            linkage_cache[src_id] = result

    if len(linkage_cache) < 2:
        continue

    combo_label  = f"nc={nc} · max_df={mxdf:.2f}"
    combo_k_rows = []

    for k in K_VALS:
        tol = tol_scale / (2 * max(k - 1, 1))
        label_arrays = []
        for src_id, result in linkage_cache.items():
            Z, n = result["Z"], result["n_chunks"]
            labels = fcluster(Z, threshold_for_k(Z, k, n), criterion="distance")
            label_arrays.append(labels)

        f1 = mean_pairwise_boundary_f1(label_arrays, tol)
        combo_k_rows.append({"k": k, "mean_f1": f1})
        curve_rows.append({
            "combo_label": combo_label,
            "n_chunks": nc, "max_df": mxdf,
            "k": k, "mean_f1": f1,
        })

    best = max(combo_k_rows, key=lambda r: r["mean_f1"])
    summary_rows.append({
        "combo_label": combo_label,
        "n_chunks": nc, "max_df": mxdf,
        "k*": int(best["k"]),
        "max_f1": round(float(best["mean_f1"]), 4),
    })

progress.empty()

if not curve_rows:
    st.warning("No combinations produced valid results. Try adjusting the grid.")
    st.stop()

df_curves  = pd.DataFrame(curve_rows)
df_summary = pd.DataFrame(summary_rows).sort_values("max_f1", ascending=False).reset_index(drop=True)

# ── Spaghetti plot ─────────────────────────────────────────────────────────────
st.subheader("Boundary Concordance vs. k — All Combinations", anchor="concordance-chart")
st.caption(
    "Each gray curve = one parameter combination. "
    "Bold blue = mean across all combinations. "
    "A consistent peak indicates a robust optimal k."
)

fig = go.Figure()
for label, gdf in df_curves.groupby("combo_label", sort=False):
    fig.add_trace(go.Scatter(
        x=gdf["k"], y=gdf["mean_f1"], mode="lines",
        line=dict(color="#CCCCCC", width=1), showlegend=False,
        hovertemplate=f"{label}<br>k=%{{x}}<br>F1=%{{y:.3f}}<extra></extra>",
    ))
mean_curve = df_curves.groupby("k")["mean_f1"].mean().reset_index()
fig.add_trace(go.Scatter(
    x=mean_curve["k"], y=mean_curve["mean_f1"], mode="lines+markers",
    line=dict(color="#1f77b4", width=3), marker=dict(size=6),
    name="Mean across combos",
    hovertemplate="mean · k=%{x}<br>F1=%{y:.3f}<extra></extra>",
))
fig.update_layout(
    height=400, margin=dict(l=60, r=30, t=10, b=50), plot_bgcolor="white",
    xaxis=dict(title="k (clusters)", dtick=1, showgrid=False, zeroline=False),
    yaxis=dict(title="Mean Boundary F1", range=[-0.05, 1.05],
               showgrid=True, gridcolor="#EEEEEE", zeroline=False),
    legend=dict(x=0.02, y=0.02),
)
st.plotly_chart(fig, width="stretch")

# ── k* distribution ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Distribution of Optimal k*", anchor="k-distribution")
st.caption("How many parameter combinations achieve their maximum boundary F1 at each k.")

k_star_counts = (
    df_summary["k*"].value_counts().reindex(range(2, 21), fill_value=0).reset_index()
)
k_star_counts.columns = ["k*", "count"]
fig_kbar = px.bar(k_star_counts, x="k*", y="count",
                  labels={"k*": "k (clusters)", "count": "# combinations"})
fig_kbar.update_layout(
    height=280, margin=dict(l=60, r=30, t=10, b=50), plot_bgcolor="white",
    xaxis=dict(dtick=1, showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False),
)
st.plotly_chart(fig_kbar, width="stretch")

# ── Best Parameters ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Best Parameters", anchor="best-params")

_df_excl2  = df_curves[df_curves["k"] >= 3]
_mean_by_k = _df_excl2.groupby("k")["mean_f1"].mean()
_k_star    = int(_mean_by_k.idxmax())
_best      = _df_excl2[_df_excl2["k"] == _k_star].nlargest(1, "mean_f1").iloc[0]

st.caption(f"k=2 excluded — F1 peaks trivially at minimum k. Genuine maximum at k={_k_star}.")

_c1, _c2, _c3, _c4 = st.columns(4)
_c1.metric("Best n_chunks", int(_best["n_chunks"]))
_c2.metric("Best max_df",   f"{_best['max_df']:.2f}")
_c3.metric("Optimal k*",    _k_star)
_c4.metric("Max F1 at k*",  f"{_best['mean_f1']:.4f}")

_df_at_kstar = (
    df_curves[df_curves["k"] == _k_star][["n_chunks", "max_df", "mean_f1"]].reset_index(drop=True)
)
_pivot = _df_at_kstar.pivot(index="n_chunks", columns="max_df", values="mean_f1")
_fig_heat = px.imshow(
    _pivot,
    labels=dict(x="max_df", y="n_chunks", color=f"Boundary F1 at k={_k_star}"),
    color_continuous_scale="Blues", aspect="auto", text_auto=".3f",
)
_fig_heat.update_layout(height=300, margin=dict(l=60, r=30, t=30, b=50))
st.plotly_chart(_fig_heat, width="stretch")

_col_nc, _col_mdf = st.columns(2)
_by_nc  = _df_at_kstar.groupby("n_chunks")["mean_f1"].mean().reset_index()
_by_mdf = _df_at_kstar.groupby("max_df")["mean_f1"].mean().reset_index()

_fig_nc = px.bar(_by_nc, x="n_chunks", y="mean_f1",
                 labels={"n_chunks": "n_chunks", "mean_f1": f"Mean F1 at k={_k_star}"})
_fig_nc.update_layout(height=240, margin=dict(l=60, r=30, t=10, b=50), plot_bgcolor="white",
    xaxis=dict(dtick=5, showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False))
_col_nc.plotly_chart(_fig_nc, width="stretch")

_fig_mdf = px.bar(_by_mdf, x="max_df", y="mean_f1",
                  labels={"max_df": "max_df", "mean_f1": f"Mean F1 at k={_k_star}"})
_fig_mdf.update_layout(height=240, margin=dict(l=60, r=30, t=10, b=50), plot_bgcolor="white",
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False))
_col_mdf.plotly_chart(_fig_mdf, width="stretch")

st.subheader("Best Parameters by k")
st.caption("For each k, the (n_chunks, max_df) combination yielding the highest mean boundary F1.")
_best_by_k = (
    df_curves.loc[df_curves.groupby("k")["mean_f1"].idxmax()]
    [["k", "n_chunks", "max_df", "mean_f1"]].sort_values("k").reset_index(drop=True)
    .rename(columns={"mean_f1": "Boundary F1"})
)
st.dataframe(_best_by_k, use_container_width=True, hide_index=True)

# ── Summary ────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Summary — Sorted by Maximum Boundary F1", anchor="summary")
st.caption("Most concordant parameter combinations first.")
st.dataframe(df_summary, use_container_width=True, hide_index=True)
