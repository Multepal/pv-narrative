"""
Parameter Grid Search — Boundary Concordance (method selector).

Sweeps n_chunks × max_df; reports mean pairwise boundary F1 vs. k.
Tolerance τ = scale / (2(k−1)) scales with k to remain non-aliasing.

Methods:
  TF-IDF / LSA / Cosine-Sim — k swept cheaply post-linkage
  NMF                        — full fit per (combo, k); cached but slow first run
  Ensemble                   — runs all four methods; adds consensus + sharpness analysis
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
    find_token_file,
    run_linkage_tfidf, run_linkage_lsa, run_linkage_sim, run_nmf_labels,
    run_resonance_boundaries, resonance_boundaries_at_pct, mean_pairwise_f1_from_positions,
    threshold_for_k, mean_pairwise_boundary_f1,
    FIXED_MIN_DF, FIXED_NGRAM, N_COMPONENTS,
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
_v           = cfg["visualization"]
K_VALS       = list(range(2, 21))

HAC_RUN_FNS = {
    "TF-IDF":     run_linkage_tfidf,
    "LSA":        run_linkage_lsa,
    "Cosine-Sim": run_linkage_sim,
}

METHOD_LABELS = {
    "TF-IDF":         "TF-IDF → Ward HAC",
    "LSA":            f"TF-IDF → LSA (SVD, n={N_COMPONENTS}) → Ward HAC",
    "Cosine-Sim":     "TF-IDF → cosine-similarity matrix → Ward HAC",
    "NMF":            "TF-IDF → NMF (full fit per k)",
    "KLD Resonance":  "KLD Resonance — novelty/transience peaks across editions",
    "Ensemble":       "Ensemble — all four HAC/NMF methods, consensus + sharpness",
}

METHOD_COLORS = {
    "TF-IDF":     "#1f77b4",
    "LSA":        "#ff7f0e",
    "Cosine-Sim": "#2ca02c",
    "NMF":        "#d62728",
}

TOL_EXPANDER_TEXT = r"""
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

# ── Controls ──────────────────────────────────────────────────────────────────
st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)

st.title("Grid Search — Boundary Concordance")
render_toc([
    ("Concordance vs. k",      "concordance-chart"),
    ("Optimal k Distribution", "k-distribution"),
    ("Best Parameters",        "best-params"),
    ("Summary",                "summary"),
])
st.caption(
    "Each edition's cluster assignment is converted to normalized boundary positions in [0, 1]. "
    "Pairwise boundary F1 is averaged across all edition pairs. "
    "**Tolerance scales with k**: `tol = scale / (2 × (k − 1))`. "
    "HAC method results are cached after the first run; NMF requires one fit per (combo, k)."
)

with st.expander("Tolerance parameter — justification"):
    st.markdown(TOL_EXPANDER_TEXT)

col_m, col1, col2, col3 = st.columns(4)
method = col_m.selectbox("Method", list(METHOD_LABELS.keys()),
                          format_func=lambda x: METHOD_LABELS[x])

nc_range = col1.slider("n_chunks range", min_value=15, max_value=50, value=(15, 40), step=5)
nc_vals  = list(range(nc_range[0], nc_range[1] + 1, 5))

maxdf_range = col2.slider("max_df range", min_value=0.20, max_value=0.95, value=(0.30, 0.70),
                           step=0.05, format="%.2f")
_n_maxdf    = round((maxdf_range[1] - maxdf_range[0]) / 0.05) + 1
maxdf_vals  = [round(maxdf_range[0] + i * 0.05, 2) for i in range(_n_maxdf)]

tol_scale = col3.slider("Tolerance scale", min_value=0.25, max_value=2.0, value=0.75, step=0.25,
    help="Multiplier on τ = 1/(2(k−1)). Values < 1 tighten matching; > 1 loosen it.")

if method == "KLD Resonance":
    _rcols = st.columns(2)
    window    = _rcols[0].number_input("Window (w)", min_value=1, max_value=20, value=3, step=1,
                                        help="Number of preceding/following chunks for novelty/transience.")
    pct_range = _rcols[1].slider("Percentile range", min_value=60, max_value=99,
                                  value=(70, 95), step=5,
                                  help="Sweep of threshold percentiles applied to the resonance signal.")
    pct_vals  = list(range(pct_range[0], pct_range[1] + 1, 5))
else:
    window   = 3
    pct_vals = []

n_combos = len(nc_vals) * len(maxdf_vals)
if method == "NMF":
    n_fits = n_combos * len(K_VALS) * len(SOURCES_META)
    st.caption(
        f"**{n_combos}** combinations × **{len(K_VALS)}** k values × **{len(SOURCES_META)}** editions "
        f"= **{n_fits}** NMF fits (est.)  ·  min_df={FIXED_MIN_DF} · ngram=(1,1) fixed"
    )
elif method == "KLD Resonance":
    st.caption(
        f"**{n_combos}** (n_chunks, max_df) combinations × **{len(pct_vals)}** percentile thresholds "
        f"· window w={window} · min_df={FIXED_MIN_DF} fixed"
    )
elif method == "Ensemble":
    st.caption(
        f"**{n_combos}** (n_chunks, max_df) combinations × 4 methods  ·  "
        f"min_df={FIXED_MIN_DF} · ngram=(1,1) fixed"
    )
else:
    st.caption(
        f"**{n_combos}** combinations · **{n_combos * len(SOURCES_META)}** linkage runs (est.)  "
        f"·  min_df={FIXED_MIN_DF} · ngram=(1,1) fixed"
    )

# ── Grid computation ───────────────────────────────────────────────────────────
combos      = list(itertools.product(nc_vals, maxdf_vals))
token_files = {src_id: find_token_file(src_id) for src_id in SOURCES_META}

if method == "Ensemble":
    # ── Ensemble: run all four methods ─────────────────────────────────────────
    all_curves = {m: [] for m in [*HAC_RUN_FNS, "NMF"]}
    progress   = st.progress(0, text="Running grid…")

    for ci, (nc, mxdf) in enumerate(combos):
        progress.progress(
            (ci + 1) / len(combos),
            text=f"combo {ci + 1}/{len(combos)}  ·  n_chunks={nc}  max_df={mxdf}",
        )
        combo_label = f"nc={nc} · max_df={mxdf:.2f}"

        for m_name, run_fn in HAC_RUN_FNS.items():
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
                all_curves[m_name].append({"combo_label": combo_label, "n_chunks": nc,
                                           "max_df": mxdf, "k": k, "mean_f1": f1})

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
                all_curves["NMF"].append({"combo_label": combo_label, "n_chunks": nc,
                                          "max_df": mxdf, "k": k, "mean_f1": f1})

    progress.empty()
    dfs = {m: pd.DataFrame(rows) for m, rows in all_curves.items() if rows}

    if len(dfs) < 2:
        st.warning("Fewer than two methods produced valid results. Try adjusting the grid.")
        st.stop()

    # Per-method k* (exclude k=2)
    k_star = {}
    for m, df in dfs.items():
        mean_by_k = df[df["k"] >= 3].groupby("k")["mean_f1"].mean()
        k_star[m] = int(mean_by_k.idxmax())

    render_toc([
        ("Mean F1 Curves by Method", "mean-curves"),
        ("Consensus Score",          "consensus"),
        ("Peak Sharpness",           "sharpness"),
        ("Recommended Parameters",   "recommendation"),
    ])

    # ── Mean F1 Curves by Method ───────────────────────────────────────────────
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
        fig_curves.add_vline(x=k_star[m], line_dash="dash",
                             line_color=METHOD_COLORS.get(m, "#888"), opacity=0.5)
    fig_curves.update_layout(
        height=380, margin=dict(l=60, r=30, t=10, b=50), plot_bgcolor="white",
        xaxis=dict(title="k", dtick=1, showgrid=False, zeroline=False),
        yaxis=dict(title="Mean Boundary F1", range=[-0.05, 1.05],
                   showgrid=True, gridcolor="#EEEEEE", zeroline=False),
        legend=dict(x=0.75, y=0.05),
    )
    st.plotly_chart(fig_curves, width="stretch")
    st.caption("  ·  ".join(f"**{m}**: k*={k_star[m]}" for m in dfs))

    # ── Consensus score ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Consensus Score", anchor="consensus")
    st.caption(
        "For each method, combinations are ranked by F1 at that method's k* "
        "(rank 1 = highest F1). **Consensus score** = 1 − (mean rank − 1) / (N − 1), "
        "so 1.0 means top-ranked by every method. "
        "Breaks ties among saturated cells that all appear as 1.000 in individual pages."
    )
    f1_at_kstar = {}
    for m, df in dfs.items():
        f1_at_kstar[m] = (
            df[df["k"] == k_star[m]]
            [["combo_label", "n_chunks", "max_df", "mean_f1"]]
            .set_index("combo_label")
        )
    base   = f1_at_kstar[list(dfs.keys())[0]][["n_chunks", "max_df"]].copy()
    merged = base.copy()
    for m, sl in f1_at_kstar.items():
        merged[m] = sl["mean_f1"]
    merged = merged.dropna().reset_index()
    for m in dfs:
        if m in merged.columns:
            merged[f"rank_{m}"] = merged[m].rank(ascending=False, method="average")
    rank_cols = [f"rank_{m}" for m in dfs if f"rank_{m}" in merged.columns]
    n_merged  = len(merged)
    merged["mean_rank"]       = merged[rank_cols].mean(axis=1)
    merged["consensus_score"] = 1 - (merged["mean_rank"] - 1) / max(n_merged - 1, 1)

    pivot_con = merged.pivot(index="n_chunks", columns="max_df", values="consensus_score")
    fig_con   = px.imshow(pivot_con,
                          labels=dict(x="max_df", y="n_chunks", color="Consensus score"),
                          color_continuous_scale=_v["colorscale_boundary"], zmin=0, zmax=1,
                          aspect="auto", text_auto=".2f")
    fig_con.update_layout(height=300, margin=dict(l=60, r=30, t=30, b=50))
    st.plotly_chart(fig_con, width="stretch")

    display_cols = ["combo_label", "n_chunks", "max_df"] + list(dfs.keys()) + ["consensus_score"]
    display_cols = [c for c in display_cols if c in merged.columns]
    st.dataframe(
        merged[display_cols]
        .sort_values("consensus_score", ascending=False)
        .reset_index(drop=True)
        .rename(columns={"consensus_score": "Consensus"}),
        use_container_width=True, hide_index=True,
    )

    # ── Peak sharpness ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Peak Sharpness", anchor="sharpness")
    st.caption(
        "Sharpness = F₁(k*) − mean(F₁, k≥3) for each method, averaged across methods. "
        "**Near zero** = metric is flat or saturated; **positive** = genuine peak at k*."
    )
    sharpness_per_method = {}
    for m, df in dfs.items():
        df3            = df[df["k"] >= 3]
        mean_f1_combo  = df3.groupby("combo_label")["mean_f1"].mean()
        f1_k           = df3[df3["k"] == k_star[m]].set_index("combo_label")["mean_f1"]
        sharpness_per_method[m] = (f1_k - mean_f1_combo).rename(m)
    sharpness_df = pd.concat(sharpness_per_method.values(), axis=1).dropna()
    sharpness_df["mean_sharpness"] = sharpness_df.mean(axis=1)
    combo_coords = (
        dfs[list(dfs.keys())[0]][["combo_label", "n_chunks", "max_df"]]
        .drop_duplicates("combo_label").set_index("combo_label")
    )
    sharpness_df = sharpness_df.join(combo_coords)

    pivot_sharp = sharpness_df.reset_index().pivot(
        index="n_chunks", columns="max_df", values="mean_sharpness"
    )
    fig_sharp = px.imshow(pivot_sharp,
                          labels=dict(x="max_df", y="n_chunks", color="Mean sharpness"),
                          color_continuous_scale=_v["colorscale_ari"], aspect="auto", text_auto=".3f")
    fig_sharp.update_layout(height=300, margin=dict(l=60, r=30, t=30, b=50))
    st.plotly_chart(fig_sharp, width="stretch")

    # ── Recommendation ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Recommended Parameters", anchor="recommendation")
    st.caption(
        "Best combination by joint score: 0.6 × consensus + 0.4 × normalized sharpness."
    )
    joint      = merged[["combo_label", "n_chunks", "max_df", "consensus_score"]].set_index("combo_label")
    sharp_norm = sharpness_df["mean_sharpness"]
    s_min, s_max = sharp_norm.min(), sharp_norm.max()
    if s_max > s_min:
        sharp_norm = (sharp_norm - s_min) / (s_max - s_min)
    joint["sharpness"]   = sharp_norm
    joint["joint_score"] = 0.6 * joint["consensus_score"] + 0.4 * joint["sharpness"].fillna(0)
    joint = joint.dropna(subset=["joint_score"]).sort_values("joint_score", ascending=False)

    best_row = joint.iloc[0]
    _c1, _c2, _c3, _c4 = st.columns(4)
    _c1.metric("Best n_chunks",    int(best_row["n_chunks"]))
    _c2.metric("Best max_df",      f"{best_row['max_df']:.2f}")
    _c3.metric("Consensus score",  f"{best_row['consensus_score']:.3f}")
    _c4.metric("Sharpness (norm)", f"{best_row['sharpness']:.3f}")
    st.caption("  ·  ".join(f"{m}: k*={k_star[m]}" for m in dfs))

    st.dataframe(
        joint.reset_index()[["combo_label", "n_chunks", "max_df", "consensus_score",
                              "sharpness", "joint_score"]]
        .rename(columns={"consensus_score": "Consensus", "joint_score": "Joint score"})
        .round(4),
        use_container_width=True, hide_index=True,
    )

elif method == "KLD Resonance":
    # ── KLD Resonance ──────────────────────────────────────────────────────────
    curve_rows   = []
    summary_rows = []
    progress     = st.progress(0, text="Running grid…")

    for ci, (nc, mxdf) in enumerate(combos):
        progress.progress((ci + 1) / len(combos),
                          text=f"combo {ci + 1}/{len(combos)}  ·  n_chunks={nc}  max_df={mxdf}")
        resonance_cache = {}
        for src_id in SOURCES_META:
            tp = token_files[src_id]
            if tp is None:
                continue
            res = run_resonance_boundaries(src_id, tp, nc, FIXED_MIN_DF, mxdf, window)
            if res is not None:
                resonance_cache[src_id] = res
        if len(resonance_cache) < 2:
            continue

        combo_label    = f"nc={nc} · max_df={mxdf:.2f}"
        combo_pct_rows = []
        for pct in pct_vals:
            position_arrays = []
            n_b_list        = []
            for res in resonance_cache.values():
                pos = resonance_boundaries_at_pct(res["resonance"], res["n_chunks"], pct)
                position_arrays.append(pos)
                n_b_list.append(len(pos))
            k_eff = float(np.mean(n_b_list)) + 1
            tol   = tol_scale / (2 * max(k_eff - 1, 1))
            f1    = mean_pairwise_f1_from_positions(position_arrays, tol)
            combo_pct_rows.append({"percentile": pct, "mean_f1": f1, "k_eff": round(k_eff, 1)})
            curve_rows.append({"combo_label": combo_label, "n_chunks": nc,
                               "max_df": mxdf, "percentile": pct, "mean_f1": f1})
        if combo_pct_rows:
            best = max(combo_pct_rows, key=lambda r: r["mean_f1"])
            summary_rows.append({"combo_label": combo_label, "n_chunks": nc, "max_df": mxdf,
                                  "pct*": int(best["percentile"]),
                                  "k_eff": best["k_eff"],
                                  "max_f1": round(float(best["mean_f1"]), 4)})

    progress.empty()

    if not curve_rows:
        st.warning("No combinations produced valid results. Try adjusting the grid.")
        st.stop()

    df_curves  = pd.DataFrame(curve_rows)
    df_summary = pd.DataFrame(summary_rows).sort_values("max_f1", ascending=False).reset_index(drop=True)

    # ── Concordance vs. percentile ─────────────────────────────────────────────
    st.subheader("Boundary Concordance vs. Threshold Percentile", anchor="concordance-chart")
    st.caption(
        "Each gray curve = one (n_chunks, max_df) combination. "
        "Bold green = mean across all combinations. "
        "Tolerance adapts to the mean number of detected boundaries at each percentile."
    )
    fig = go.Figure()
    for label, gdf in df_curves.groupby("combo_label", sort=False):
        fig.add_trace(go.Scatter(
            x=gdf["percentile"], y=gdf["mean_f1"], mode="lines",
            line=dict(color="#CCCCCC", width=1), showlegend=False,
            hovertemplate=f"{label}<br>pct=%{{x}}<br>F1=%{{y:.3f}}<extra></extra>",
        ))
    mean_curve = df_curves.groupby("percentile")["mean_f1"].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=mean_curve["percentile"], y=mean_curve["mean_f1"], mode="lines+markers",
        line=dict(color="#00CC96", width=3), marker=dict(size=6),
        name="Mean across combos",
        hovertemplate="mean · pct=%{x}<br>F1=%{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=400, margin=dict(l=60, r=30, t=10, b=50), plot_bgcolor="white",
        xaxis=dict(title="Threshold percentile", dtick=5, showgrid=False, zeroline=False),
        yaxis=dict(title="Mean Boundary F1", range=[-0.05, 1.05],
                   showgrid=True, gridcolor="#EEEEEE", zeroline=False),
        legend=dict(x=0.02, y=0.02),
    )
    st.plotly_chart(fig, width="stretch")

    # ── Optimal threshold distribution ────────────────────────────────────────
    st.divider()
    st.subheader("Distribution of Optimal Threshold (pct*)", anchor="k-distribution")
    st.caption("How many parameter combinations achieve their maximum concordance at each percentile.")
    pct_star_counts = (
        df_summary["pct*"].value_counts().reindex(pct_vals, fill_value=0).reset_index()
    )
    pct_star_counts.columns = ["pct*", "count"]
    fig_bar = px.bar(pct_star_counts, x="pct*", y="count",
                     labels={"pct*": "Threshold percentile", "count": "# combinations"})
    fig_bar.update_layout(
        height=280, margin=dict(l=60, r=30, t=10, b=50), plot_bgcolor="white",
        xaxis=dict(dtick=5, showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False),
    )
    st.plotly_chart(fig_bar, width="stretch")

    # ── Best Parameters ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Best Parameters", anchor="best-params")
    _mean_by_pct = df_curves.groupby("percentile")["mean_f1"].mean()
    _pct_star    = int(_mean_by_pct.idxmax())
    _best        = df_curves[df_curves["percentile"] == _pct_star].nlargest(1, "mean_f1").iloc[0]
    st.caption(f"Optimal threshold percentile: **{_pct_star}**.")

    _c1, _c2, _c3, _c4 = st.columns(4)
    _c1.metric("Best n_chunks",   int(_best["n_chunks"]))
    _c2.metric("Best max_df",     f"{_best['max_df']:.2f}")
    _c3.metric("Optimal pct*",    _pct_star)
    _c4.metric("Max F1 at pct*",  f"{_best['mean_f1']:.4f}")

    _df_at_pct = (
        df_curves[df_curves["percentile"] == _pct_star][["n_chunks", "max_df", "mean_f1"]]
        .reset_index(drop=True)
    )
    _pivot = _df_at_pct.pivot(index="n_chunks", columns="max_df", values="mean_f1")
    _fig_heat = px.imshow(_pivot,
                          labels=dict(x="max_df", y="n_chunks",
                                      color=f"Boundary F1 at pct={_pct_star}"),
                          color_continuous_scale=_v["colorscale_ari"], aspect="auto", text_auto=".3f")
    _fig_heat.update_layout(height=300, margin=dict(l=60, r=30, t=30, b=50))
    st.plotly_chart(_fig_heat, width="stretch")

    # ── Summary ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Summary — Sorted by Maximum Boundary F1", anchor="summary")
    st.caption("Most concordant parameter combinations first. k_eff = mean detected boundaries + 1.")
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

else:
    # ── Single method ──────────────────────────────────────────────────────────
    curve_rows   = []
    summary_rows = []
    progress     = st.progress(0, text="Running grid…")

    if method == "NMF":
        total_steps = len(combos) * len(K_VALS)
        step = 0
        for nc, mxdf in combos:
            combo_label  = f"nc={nc} · max_df={mxdf:.2f}"
            combo_k_rows = []
            for k in K_VALS:
                step += 1
                progress.progress(step / total_steps,
                                  text=f"{combo_label}  ·  k={k}  ({step}/{total_steps})")
                tol = tol_scale / (2 * max(k - 1, 1))
                label_arrays = []
                for src_id in SOURCES_META:
                    tp = token_files[src_id]
                    if tp is None:
                        continue
                    labels = run_nmf_labels(src_id, tp, nc, FIXED_MIN_DF, mxdf, k, FIXED_NGRAM)
                    if labels is not None:
                        label_arrays.append(labels)
                if len(label_arrays) < 2:
                    continue
                f1 = mean_pairwise_boundary_f1(label_arrays, tol)
                combo_k_rows.append({"k": k, "mean_f1": f1})
                curve_rows.append({"combo_label": combo_label, "n_chunks": nc,
                                   "max_df": mxdf, "k": k, "mean_f1": f1})
            if combo_k_rows:
                best = max(combo_k_rows, key=lambda r: r["mean_f1"])
                summary_rows.append({"combo_label": combo_label, "n_chunks": nc, "max_df": mxdf,
                                     "k*": int(best["k"]),
                                     "max_f1": round(float(best["mean_f1"]), 4)})
    else:
        run_fn = HAC_RUN_FNS[method]
        for ci, (nc, mxdf) in enumerate(combos):
            progress.progress((ci + 1) / len(combos),
                              text=f"combo {ci + 1}/{len(combos)}  ·  n_chunks={nc}  max_df={mxdf}")
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
            combo_label  = f"nc={nc} · max_df={mxdf:.2f}"
            combo_k_rows = []
            for k in K_VALS:
                tol = tol_scale / (2 * max(k - 1, 1))
                label_arrays = [
                    fcluster(r["Z"], threshold_for_k(r["Z"], k, r["n_chunks"]), criterion="distance")
                    for r in linkage_cache.values()
                ]
                f1 = mean_pairwise_boundary_f1(label_arrays, tol)
                combo_k_rows.append({"k": k, "mean_f1": f1})
                curve_rows.append({"combo_label": combo_label, "n_chunks": nc,
                                   "max_df": mxdf, "k": k, "mean_f1": f1})
            best = max(combo_k_rows, key=lambda r: r["mean_f1"])
            summary_rows.append({"combo_label": combo_label, "n_chunks": nc, "max_df": mxdf,
                                  "k*": int(best["k"]),
                                  "max_f1": round(float(best["mean_f1"]), 4)})

    progress.empty()

    if not curve_rows:
        st.warning("No combinations produced valid results. Try adjusting the grid.")
        st.stop()

    df_curves  = pd.DataFrame(curve_rows)
    df_summary = pd.DataFrame(summary_rows).sort_values("max_f1", ascending=False).reset_index(drop=True)
    k_label    = "k (topics)" if method == "NMF" else "k (clusters)"

    # ── Spaghetti plot ─────────────────────────────────────────────────────────
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
        xaxis=dict(title=k_label, dtick=1, showgrid=False, zeroline=False),
        yaxis=dict(title="Mean Boundary F1", range=[-0.05, 1.05],
                   showgrid=True, gridcolor="#EEEEEE", zeroline=False),
        legend=dict(x=0.02, y=0.02),
    )
    st.plotly_chart(fig, width="stretch")

    # ── k* distribution ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Distribution of Optimal k*", anchor="k-distribution")
    st.caption("How many parameter combinations achieve their maximum boundary F1 at each k.")
    k_star_counts = (
        df_summary["k*"].value_counts().reindex(range(2, 21), fill_value=0).reset_index()
    )
    k_star_counts.columns = ["k*", "count"]
    fig_kbar = px.bar(k_star_counts, x="k*", y="count",
                      labels={"k*": k_label, "count": "# combinations"})
    fig_kbar.update_layout(
        height=280, margin=dict(l=60, r=30, t=10, b=50), plot_bgcolor="white",
        xaxis=dict(dtick=1, showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False),
    )
    st.plotly_chart(fig_kbar, width="stretch")

    # ── Best Parameters ────────────────────────────────────────────────────────
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
        df_curves[df_curves["k"] == _k_star][["n_chunks", "max_df", "mean_f1"]]
        .reset_index(drop=True)
    )
    _pivot = _df_at_kstar.pivot(index="n_chunks", columns="max_df", values="mean_f1")
    _fig_heat = px.imshow(_pivot,
                          labels=dict(x="max_df", y="n_chunks", color=f"Boundary F1 at k={_k_star}"),
                          color_continuous_scale=_v["colorscale_ari"], aspect="auto", text_auto=".3f")
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

    # ── Summary ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Summary — Sorted by Maximum Boundary F1", anchor="summary")
    st.caption("Most concordant parameter combinations first.")
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
