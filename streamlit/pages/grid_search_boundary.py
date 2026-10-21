"""
Parameter Grid Search — Boundary Concordance (method selector).

Sweeps n_chunks × max_df; for each k, scores cross-edition boundary F1 (boundaries match
within ± a fixed number of chunks) as excess over a cyclic-shift chance baseline, and
chooses k* by that excess, optionally on K'iche' pairs only with translations held out.

Methods:
  TF-IDF / LSA / Cosine-Sim — k swept cheaply post-linkage
  NMF                        — full fit per (combo, k); cached but slow first run
  KLD Resonance              — boundaries from novelty/transience peaks (no k; no chance baseline)
  Ensemble                   — runs all four methods; k*, robustness and held-out score by method
"""

import os
import yaml
import itertools
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from toc import render_toc
from grid_search_boundary_core import (
    agreement_curve,
    find_token_file,
    run_resonance_boundaries, resonance_boundaries_at_pct, mean_pairwise_f1_from_positions,
    FIXED_MIN_DF, N_COMPONENTS,
)
from grid_search_ui import (K_VALS, K_MIN, SELECTION_LABELS, run_grid, choose_k, selection_control,
                            reporting_setting,
                            render_selection_results)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
_v           = cfg["visualization"]
_tc          = cfg["controls"]["boundary_tol_chunks"]
ENSEMBLE_METHODS = ["TF-IDF", "LSA", "Cosine-Sim", "NMF"]

METHOD_LABELS = {
    "TF-IDF":         "TF-IDF → Ward HAC",
    "LSA":            f"TF-IDF → PCA/LSA (centered, n={N_COMPONENTS}) → Ward HAC",
    "Cosine-Sim":     "TF-IDF → cosine-similarity matrix → Ward HAC",
    "NMF":            "TF-IDF → NMF (full fit per k)",
    "KLD Resonance":  "KLD Resonance — novelty/transience peaks across editions",
    "Ensemble":       "Ensemble — all four HAC/NMF methods, k* and robustness by method",
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

TOL_CHUNKS_EXPANDER_TEXT = """
Two boundaries match if they fall within ± the tolerance, in whole chunks. A fixed
tolerance replaces the earlier τ = α / 2(k−1), which assumed k−1 boundaries per edition.
Clusters recur in narrative order (A B C D C A …), so editions have more boundaries than
that (about 7 at k = 6, 15 at k = 12); k-scaled windows then overlap and even misaligned
sequences score near 1. At high k any tolerance saturates, because boundaries fall every
chunk or two; the excess-over-chance score shows where that happens rather than hiding it.
"""

# ── Controls ──────────────────────────────────────────────────────────────────
st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)

st.title("Grid Search — Boundary Concordance")
st.caption(
    "Each edition's cluster assignment is converted to boundary positions; pairwise boundary F1 "
    "is scored against a cyclic-shift chance baseline. HAC results are cached after the first "
    "run; NMF requires one fit per (combo, k)."
)

col_m, col1, col2, col3 = st.columns(4)
method = col_m.selectbox("Method", list(METHOD_LABELS.keys()),
                          format_func=lambda x: METHOD_LABELS[x])

nc_range = col1.slider("n_chunks range", min_value=15, max_value=50, value=(15, 40), step=5)
nc_vals  = list(range(nc_range[0], nc_range[1] + 1, 5))
maxdf_range = col2.slider("max_df range", min_value=0.20, max_value=0.95, value=(0.30, 0.70),
                           step=0.05, format="%.2f")
_n_maxdf    = round((maxdf_range[1] - maxdf_range[0]) / 0.05) + 1
maxdf_vals  = [round(maxdf_range[0] + i * 0.05, 2) for i in range(_n_maxdf)]

if method == "KLD Resonance":
    tol_scale = col3.slider("Tolerance scale", min_value=0.25, max_value=2.0, value=0.75, step=0.25,
        help="Multiplier on τ = 1/(2(k_eff−1)), k_eff = detected boundaries + 1.")
    with st.expander("Tolerance parameter — justification"):
        st.markdown(TOL_EXPANDER_TEXT)
    _rcols = st.columns(2)
    window    = _rcols[0].number_input("Window (w)", min_value=1, max_value=20, value=3, step=1,
                                        help="Number of preceding/following chunks for novelty/transience.")
    pct_range = _rcols[1].slider("Percentile range", min_value=60, max_value=99,
                                  value=(70, 95), step=5,
                                  help="Sweep of threshold percentiles applied to the resonance signal.")
    pct_vals  = list(range(pct_range[0], pct_range[1] + 1, 5))
else:
    tol_chunks = col3.slider("Boundary tolerance (± chunks)", _tc["min"], _tc["max"], _tc["default"],
                             _tc["step"], help="0 = boundaries must fall at the same chunk.")
    with st.expander("Tolerance — justification"):
        st.markdown(TOL_CHUNKS_EXPANDER_TEXT)
    sel = selection_control(st, key="gs_boundary_sel")

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
    # ── Ensemble: run all four methods, score each by excess over chance ───────
    ex  = f"{sel}_excess"
    dfs = {}
    for m in ENSEMBLE_METHODS:
        df_m = run_grid(m, combos, "F1", tol_chunks, progress_label=f"{m} · ")
        if not df_m.empty:
            dfs[m] = df_m

    if len(dfs) < 2:
        st.warning("Fewer than two methods produced valid results. Try adjusting the grid.")
        st.stop()

    k_star = {m: choose_k(df, sel)[0] for m, df in dfs.items()}

    render_toc([
        ("Excess Curves by Method",    "mean-curves"),
        ("k* by Method",               "k-by-method"),
        ("Robustness Across the Grid", "robustness"),
    ])

    # ── Excess curves by method ────────────────────────────────────────────────
    st.subheader("Boundary F1 Excess over Chance vs. k — by Method", anchor="mean-curves")
    st.caption(
        f"Pairs: **{SELECTION_LABELS[sel]}**. Mean excess (F1 − chance 95th percentile) across all "
        "parameter combinations for each method. Dashed lines mark each method's k*. "
        "Overlapping peaks across methods indicate a robust consensus k."
    )
    fig_curves = go.Figure()
    for m, df in dfs.items():
        mean_c = df.groupby("k")[ex].mean().reset_index()
        fig_curves.add_trace(go.Scatter(
            x=mean_c["k"], y=mean_c[ex], mode="lines+markers",
            line=dict(color=METHOD_COLORS.get(m, "#888"), width=2),
            marker=dict(size=5), name=m,
            hovertemplate=f"{m} · k=%{{x}}<br>excess=%{{y:+.3f}}<extra></extra>",
        ))
        fig_curves.add_vline(x=k_star[m], line_dash="dash",
                             line_color=METHOD_COLORS.get(m, "#888"), opacity=0.5)
    fig_curves.add_hline(y=0, line_color="gray", line_width=1, line_dash="dot",
                         annotation_text="chance (95th pct)", annotation_position="bottom right")
    fig_curves.update_layout(
        height=380, margin=dict(l=60, r=30, t=10, b=50), plot_bgcolor="white",
        xaxis=dict(title="k", dtick=1, showgrid=False, zeroline=False),
        yaxis=dict(title="Boundary F1 − chance", showgrid=True, gridcolor="#EEEEEE", zeroline=False),
        legend=dict(x=0.75, y=0.98),
    )
    st.plotly_chart(fig_curves, width="stretch")
    st.caption("  ·  ".join(f"**{m}**: k*={k_star[m]}" for m in dfs))

    # ── k* by method ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("k* by Method", anchor="k-by-method")
    st.caption(
        "Each method's k* (highest mean excess across the grid), how many parameter combinations "
        "choose it independently, and its excess at the reporting setting from config. The grid "
        "chooses k, not a single best combination: the top cell is usually the luckiest."
    )
    rep = reporting_setting()
    rows = []
    for m, df in dfs.items():
        elig = df[df["k"] >= K_MIN]
        per_combo = elig.loc[elig.groupby("combo_label")[ex].idxmax()]
        at_k = df[df["k"] == k_star[m]]
        rep_row = next((r for r in agreement_curve(m, rep["n_chunks"], rep["max_df"], "F1", tol_chunks,
                                                   K_VALS, min_df=rep["min_df"],
                                                   ngram_range=(rep["ngram_min"], rep["ngram_max"]))
                        if r["k"] == k_star[m]), None)
        row = {"Method": m, "k*": k_star[m], "Mean excess at k*": at_k[ex].mean(),
               "Excess SD": at_k[ex].std(),
               "Combinations choosing k*": f"{int((per_combo['k'] == k_star[m]).sum())} of {len(per_combo)}",
               f"Excess at reporting setting (n = {rep['n_chunks']})": rep_row[ex] if rep_row else np.nan}
        if sel == "within":
            row["Held-out excess at reporting setting"] = rep_row["cross_excess"] if rep_row else np.nan
            row["Held-out excess, grid mean"] = at_k["cross_excess"].mean()
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).round(3), width="stretch", hide_index=True)
    if sel == "within":
        st.caption(f"Held-out = {SELECTION_LABELS['within'].split(' only')[0]} ↔ translation boundary F1 "
                   "excess. Translations played no part in choosing k.")

    # ── Robustness by method ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Robustness Across the Grid", anchor="robustness")
    st.caption(
        "Excess at each method's k*, by n_chunks (mean ± SD across max_df). Excess isn't "
        "comparable across n_chunks — chance falls as chunks get fewer — so read this for whether "
        "agreement holds across resolutions, not for a best n. n_chunks is chosen by stability "
        "(Choose k → Resolution)."
    )
    fig_rob = go.Figure()
    for m, df in dfs.items():
        g = df[df["k"] == k_star[m]].groupby("n_chunks")[ex].agg(["mean", "std"]).reset_index()
        fig_rob.add_trace(go.Scatter(
            x=g["n_chunks"], y=g["mean"], mode="lines+markers", name=f"{m} (k* = {k_star[m]})",
            line=dict(color=METHOD_COLORS.get(m, "#888"), width=2),
            error_y=dict(type="data", array=g["std"].fillna(0), visible=True, thickness=1),
        ))
    fig_rob.add_hline(y=0, line_color="gray", line_width=1, line_dash="dot")
    fig_rob.update_layout(height=340, margin=dict(l=60, r=30, t=10, b=50), plot_bgcolor="white",
                          xaxis=dict(title="n_chunks", dtick=5, showgrid=False, zeroline=False),
                          yaxis=dict(title="Boundary F1 − chance", showgrid=True, gridcolor="#EEEEEE",
                                     zeroline=False),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig_rob, width="stretch")

elif method == "KLD Resonance":
    # ── KLD Resonance ──────────────────────────────────────────────────────────
    render_toc([
        ("Concordance vs. Percentile", "concordance-chart"),
        ("Optimal Threshold",          "k-distribution"),
        ("Best Parameters",            "best-params"),
        ("Summary",                    "summary"),
    ])
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
    st.dataframe(df_summary, width="stretch", hide_index=True)

else:
    # ── Single method ──────────────────────────────────────────────────────────
    render_toc([
        ("Boundary F1 Excess vs. k", "agreement-chart"),
        ("Optimal k Distribution",   "k-distribution"),
        ("k* and Robustness",        "best-params"),
        ("Summary",                  "summary"),
    ])
    df_curves = run_grid(method, combos, "F1", tol_chunks)
    if df_curves.empty:
        st.warning("No combinations produced valid results. Try adjusting the grid.")
        st.stop()
    render_selection_results(df_curves, sel, "F1",
                             "k (topics)" if method == "NMF" else "k (clusters)", method, tol_chunks)
