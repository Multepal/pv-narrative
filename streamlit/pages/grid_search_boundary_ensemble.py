"""
Boundary Concordance Ensemble — cross-method parameter robustness analysis.

Runs all four boundary concordance pipelines (TF-IDF, LSA, Cosine-Sim, NMF) over
a shared parameter grid and identifies which (n_chunks, max_df) settings are
consistently strong across modeling choices.

Three diagnostics beyond what individual pages show:
  1. Consensus score — mean cross-method rank at each method's own k*. Breaks
     ties among saturated cells by revealing which combos top *every* method, not
     just one.
  2. Peak sharpness — F1(k*) − mean(F1, k≥3) averaged across methods. Near-zero
     sharpness flags combinations where the metric is saturated or flat; high
     sharpness means a genuine, narrow peak.
  3. Tolerance stress-test — a scale slider multiplies τ = 1/(2(k−1)) globally.
     Setting it to 0.25–0.50 reveals which 1.0 cells were coasting on a loose
     tolerance and which survive strict matching.

Cache is shared with the individual method pages: visiting any page first warms
the cache for this one (and vice versa).
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
    find_token_file, run_linkage_lsa, run_linkage_tfidf, run_linkage_sim,
    run_nmf_labels, threshold_for_k, mean_pairwise_boundary_f1,
    FIXED_MIN_DF, FIXED_NGRAM, N_COMPONENTS,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
K_VALS       = list(range(2, 21))

METHOD_COLORS = {
    "TF-IDF":     "#1f77b4",
    "LSA":        "#ff7f0e",
    "Cosine-Sim": "#2ca02c",
    "NMF":        "#d62728",
}

HAC_METHODS = {
    "TF-IDF":     run_linkage_tfidf,
    "LSA":        run_linkage_lsa,
    "Cosine-Sim": run_linkage_sim,
}

# ── Controls ──────────────────────────────────────────────────────────────────
st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)

st.title("Grid Search — Boundary Concordance (Ensemble)")
render_toc([
    ("Mean F1 Curves by Method",  "mean-curves"),
    ("Consensus Score",           "consensus"),
    ("Peak Sharpness",            "sharpness"),
    ("Recommended Parameters",    "recommendation"),
])
st.caption(
    "Runs TF-IDF, LSA, Cosine-Sim, and NMF boundary concordance over the same grid. "
    "**Consensus score** = 1 − normalized mean rank across methods at each method's own k* "
    "(1.0 = top-ranked by every method). "
    "**Peak sharpness** = F₁(k*) − mean(F₁, k≥3), averaged across methods "
    "(near-zero = saturated or flat; high = genuine peak). "
    "**Tolerance scale** < 1 tightens matching globally — stress-tests 1.000 cells. "
    "Cache is shared with individual method pages."
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
    help="Multiplier on τ = 1/(2(k−1)). Reduce to stress-test saturated cells.")

n_combos = len(nc_vals) * len(maxdf_vals)
st.caption(
    f"**{n_combos}** (n_chunks, max_df) combinations × 4 methods  ·  "
    f"min_df={FIXED_MIN_DF} · ngram=(1,1) fixed"
)

# ── Grid computation ───────────────────────────────────────────────────────────
combos      = list(itertools.product(nc_vals, maxdf_vals))
token_files = {src_id: find_token_file(src_id) for src_id in SOURCES_META}

all_curves  = {m: [] for m in [*HAC_METHODS, "NMF"]}
progress    = st.progress(0, text="Running grid…")

for ci, (nc, mxdf) in enumerate(combos):
    progress.progress(
        (ci + 1) / len(combos),
        text=f"combo {ci + 1}/{len(combos)}  ·  n_chunks={nc}  max_df={mxdf}",
    )
    combo_label = f"nc={nc} · max_df={mxdf:.2f}"

    # ── HAC methods (cheap k sweep post-linkage) ───────────────────────────
    for method_name, run_fn in HAC_METHODS.items():
        linkage_cache = {}
        for src_id in SOURCES_META:
            tp = token_files[src_id]
            if tp is None:
                continue
            result = run_fn(src_id, tp, nc, FIXED_MIN_DF, mxdf, FIXED_NGRAM)
            if result is not None:
                linkage_cache[src_id] = result
        if len(linkage_cache) < 2:
            continue
        for k in K_VALS:
            tol = tol_scale / (2 * max(k - 1, 1))
            label_arrays = [
                fcluster(r["Z"], threshold_for_k(r["Z"], k, r["n_chunks"]), criterion="distance")
                for r in linkage_cache.values()
            ]
            f1 = mean_pairwise_boundary_f1(label_arrays, tol)
            all_curves[method_name].append({
                "combo_label": combo_label, "n_chunks": nc, "max_df": mxdf,
                "k": k, "mean_f1": f1,
            })

    # ── NMF (full fit per k) ───────────────────────────────────────────────
    for k in K_VALS:
        tol = tol_scale / (2 * max(k - 1, 1))
        label_arrays = []
        for src_id in SOURCES_META:
            tp = token_files[src_id]
            if tp is None:
                continue
            labels = run_nmf_labels(src_id, tp, nc, FIXED_MIN_DF, mxdf, k, FIXED_NGRAM)
            if labels is not None:
                label_arrays.append(labels)
        if len(label_arrays) >= 2:
            f1 = mean_pairwise_boundary_f1(label_arrays, tol)
            all_curves["NMF"].append({
                "combo_label": combo_label, "n_chunks": nc, "max_df": mxdf,
                "k": k, "mean_f1": f1,
            })

progress.empty()

dfs = {m: pd.DataFrame(rows) for m, rows in all_curves.items() if rows}

if len(dfs) < 2:
    st.warning("Fewer than two methods produced valid results. Try adjusting the grid.")
    st.stop()

# ── Per-method k* (exclude k=2) ────────────────────────────────────────────────
k_star = {}
for m, df in dfs.items():
    mean_by_k = df[df["k"] >= 3].groupby("k")["mean_f1"].mean()
    k_star[m] = int(mean_by_k.idxmax())

# ── Mean F1 Curves by Method ───────────────────────────────────────────────────
st.subheader("Mean Boundary F1 vs. k — by Method", anchor="mean-curves")
st.caption(
    "Mean across all parameter combinations for each method. "
    "Dashed vertical lines mark each method's k* (excluding k=2). "
    "Overlapping peaks across methods indicate a robust consensus k."
)

fig_curves = go.Figure()
for m, df in dfs.items():
    mean_c = df.groupby("k")["mean_f1"].mean().reset_index()
    fig_curves.add_trace(go.Scatter(
        x=mean_c["k"], y=mean_c["mean_f1"], mode="lines+markers",
        line=dict(color=METHOD_COLORS.get(m, "#888"), width=2),
        marker=dict(size=5), name=m,
        hovertemplate=f"{m} · k=%{{x}}<br>F1=%{{y:.3f}}<extra></extra>",
    ))
    fig_curves.add_vline(
        x=k_star[m], line_dash="dash",
        line_color=METHOD_COLORS.get(m, "#888"), opacity=0.5,
    )

fig_curves.update_layout(
    height=380, margin=dict(l=60, r=30, t=10, b=50), plot_bgcolor="white",
    xaxis=dict(title="k", dtick=1, showgrid=False, zeroline=False),
    yaxis=dict(title="Mean Boundary F1", range=[-0.05, 1.05],
               showgrid=True, gridcolor="#EEEEEE", zeroline=False),
    legend=dict(x=0.75, y=0.05),
)
st.plotly_chart(fig_curves, width="stretch")

_kstar_summary = "  ·  ".join(f"**{m}**: k*={k_star[m]}" for m in dfs)
st.caption(_kstar_summary)

# ── Consensus score ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Consensus Score", anchor="consensus")
st.caption(
    "For each method, parameter combinations are ranked by F1 at that method's k* "
    "(rank 1 = highest F1). **Consensus score** = 1 − (mean rank − 1) / (N − 1), "
    "so 1.0 means top-ranked by every method and 0.0 means bottom-ranked by every method. "
    "Heatmap breaks ties among saturated cells: all 1.000s look the same in individual pages, "
    "but their consensus scores differ when even one method is less impressed."
)

f1_at_kstar = {}
for m, df in dfs.items():
    slice_df = (
        df[df["k"] == k_star[m]]
        [["combo_label", "n_chunks", "max_df", "mean_f1"]]
        .set_index("combo_label")
    )
    f1_at_kstar[m] = slice_df

# Merge on combo_label — inner join so only combos present in all methods appear
base = f1_at_kstar[list(dfs.keys())[0]][["n_chunks", "max_df"]].copy()
merged = base.copy()
for m, sl in f1_at_kstar.items():
    merged[m] = sl["mean_f1"]
merged = merged.dropna().reset_index()

for m in dfs:
    if m in merged.columns:
        merged[f"rank_{m}"] = merged[m].rank(ascending=False, method="average")

rank_cols = [f"rank_{m}" for m in dfs if f"rank_{m}" in merged.columns]
n_combos_merged = len(merged)
merged["mean_rank"]       = merged[rank_cols].mean(axis=1)
merged["consensus_score"] = 1 - (merged["mean_rank"] - 1) / max(n_combos_merged - 1, 1)

pivot_consensus = merged.pivot(index="n_chunks", columns="max_df", values="consensus_score")
fig_con = px.imshow(
    pivot_consensus,
    labels=dict(x="max_df", y="n_chunks", color="Consensus score"),
    color_continuous_scale="RdYlGn", zmin=0, zmax=1,
    aspect="auto", text_auto=".2f",
)
fig_con.update_layout(height=300, margin=dict(l=60, r=30, t=30, b=50))
st.plotly_chart(fig_con, width="stretch")

# Per-method F1 at k* table
st.caption("F1 at each method's own k* and within-method rank, for every parameter combination.")
display_cols = ["combo_label", "n_chunks", "max_df"] + list(dfs.keys()) + ["consensus_score"]
display_cols = [c for c in display_cols if c in merged.columns]
st.dataframe(
    merged[display_cols]
    .sort_values("consensus_score", ascending=False)
    .reset_index(drop=True)
    .rename(columns={"consensus_score": "Consensus"}),
    use_container_width=True, hide_index=True,
)

# ── Peak sharpness ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("Peak Sharpness", anchor="sharpness")
st.caption(
    "Sharpness = F₁(k*) − mean(F₁, k≥3) for each method, averaged across methods. "
    "**Near zero** = metric is flat or saturated across k — the parameter combination "
    "cannot discriminate between 3 and 20 segments. "
    "**Positive** = a genuine peak exists at k*. "
    "Prefer combinations in the upper-right of the consensus heatmap "
    "that *also* have meaningful sharpness."
)

sharpness_per_method = {}
for m, df in dfs.items():
    df3 = df[df["k"] >= 3]
    mean_f1_combo = df3.groupby("combo_label")["mean_f1"].mean()
    f1_k = df3[df3["k"] == k_star[m]].set_index("combo_label")["mean_f1"]
    sharpness_per_method[m] = (f1_k - mean_f1_combo).rename(m)

sharpness_df = pd.concat(sharpness_per_method.values(), axis=1).dropna()
sharpness_df["mean_sharpness"] = sharpness_df.mean(axis=1)

# Attach n_chunks / max_df back
combo_coords = (
    dfs[list(dfs.keys())[0]]
    [["combo_label", "n_chunks", "max_df"]]
    .drop_duplicates("combo_label")
    .set_index("combo_label")
)
sharpness_df = sharpness_df.join(combo_coords)

pivot_sharp = sharpness_df.reset_index().pivot(
    index="n_chunks", columns="max_df", values="mean_sharpness"
)
fig_sharp = px.imshow(
    pivot_sharp,
    labels=dict(x="max_df", y="n_chunks", color="Mean sharpness"),
    color_continuous_scale="Blues", aspect="auto", text_auto=".3f",
)
fig_sharp.update_layout(height=300, margin=dict(l=60, r=30, t=30, b=50))
st.plotly_chart(fig_sharp, width="stretch")

# ── Recommendation ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("Recommended Parameters", anchor="recommendation")
st.caption(
    "Best combination by a joint score combining consensus (cross-method agreement) "
    "and sharpness (genuine peak vs. flat plateau). "
    "Score = 0.6 × consensus + 0.4 × normalized sharpness."
)

joint = merged[["combo_label", "n_chunks", "max_df", "consensus_score"]].set_index("combo_label")
sharp_norm = sharpness_df["mean_sharpness"]
sharp_min, sharp_max = sharp_norm.min(), sharp_norm.max()
if sharp_max > sharp_min:
    sharp_norm = (sharp_norm - sharp_min) / (sharp_max - sharp_min)
joint["sharpness"]   = sharp_norm
joint["joint_score"] = 0.6 * joint["consensus_score"] + 0.4 * joint["sharpness"].fillna(0)
joint = joint.dropna(subset=["joint_score"]).sort_values("joint_score", ascending=False)

best_row = joint.iloc[0]
_c1, _c2, _c3, _c4 = st.columns(4)
_c1.metric("Best n_chunks",    int(best_row["n_chunks"]))
_c2.metric("Best max_df",      f"{best_row['max_df']:.2f}")
_c3.metric("Consensus score",  f"{best_row['consensus_score']:.3f}")
_c4.metric("Sharpness (norm)", f"{best_row['sharpness']:.3f}")

_kstar_str = "  ·  ".join(f"{m}: k*={k_star[m]}" for m in dfs)
st.caption(f"Method-specific k*:  {_kstar_str}")

st.dataframe(
    joint.reset_index()[["combo_label", "n_chunks", "max_df", "consensus_score", "sharpness", "joint_score"]]
    .rename(columns={"consensus_score": "Consensus", "joint_score": "Joint score"})
    .round(4),
    use_container_width=True, hide_index=True,
)
