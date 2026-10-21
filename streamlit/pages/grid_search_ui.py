"""
Shared controls and result views for the grid search pages.

Both grid searches choose k (and n_chunks, max_df) by **excess over chance**:
mean pairwise agreement minus the 95th percentile of the cyclic-shift baseline,
computed on the selection pairs. With translations held out, the K'iche' ↔
translation agreement at the chosen k is reported as a validation score.
"""

import os
import yaml
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from grid_search_boundary_core import agreement_curve, FIXED_MIN_DF, FIXED_NGRAM
from utils import TFIDF_HANDOFF_KEY, TFIDF_PAGE
from resolution import resolution_stability, stability_plateau, RES

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

_c          = cfg["controls"]
_v          = cfg["visualization"]
_src        = cfg["languages"][cfg["source_lang"]]
K_VALS      = tuple(range(_c["n_clusters"]["min"], _c["n_clusters"]["max"] + 1))
K_MIN       = _c["k_selection_min"]
METRIC_NAME = {"ARI": "ARI", "F1": "boundary F1"}

SELECTION_LABELS = {
    "within": f"{_src} ↔ {_src} only (translations held out)",
    "all":    "All edition pairs",
}

SELECTION_EXPANDER_TEXT = f"""
**Why excess over chance.** Raw agreement isn't comparable across k: at k = 2 any two
segmentations agree almost perfectly, and chance agreement falls as k grows. Choosing
the k with the highest raw score therefore favors small k. The score used here is
observed mean agreement minus the 95th percentile of agreement between the same
sequences cyclically shifted out of alignment, so k\\* is where editions agree *most
beyond* what their block structure alone would produce.

**Why hold out translations.** Choosing k by cross-edition agreement and then reporting
that agreement as evidence is circular. With *{SELECTION_LABELS['within']}*, k and the
parameters are chosen on the {_src} transcriptions alone; agreement between {_src} and
the Spanish and English translations, which played no part in the choice, is then an
independent test.

**Why k\\* starts at {_c['k_selection_min']}.** k = 2 asks only where the single main division
falls. Editions can agree on that strongly (boundary F1 peaks there) without saying anything
about the episode structure, so k\\* is chosen among k ≥ {_c['k_selection_min']}. The k = 2 value
is still plotted (shaded) and listed among the local peaks.

**Why k is capped.** k is limited to n_chunks / {_c['min_chunks_per_cluster']}, so clusters
average at least {_c['min_chunks_per_cluster']} chunks. Near k = n_chunks, partitions approach
one chunk per cluster: editions that merge the same single pair agree perfectly while shifted
sequences score ≈ 0, which inflates the excess without reflecting narrative structure.
"""


def selection_control(container, key):
    """Radio for the pairs used to choose k. Returns 'within' or 'all'."""
    options = list(SELECTION_LABELS)
    return container.radio(
        "Choose k on", options, index=options.index(_c["k_selection_pairs"]["default"]),
        format_func=SELECTION_LABELS.get, key=key, horizontal=True,
    )


def run_grid(method, combos, metric, tol_chunks=0, progress_label=""):
    """agreement_curve for every (n_chunks, max_df) combination, as one long DataFrame."""
    rows = []
    progress = st.progress(0, text="Running grid…")
    for ci, (nc, mxdf) in enumerate(combos):
        progress.progress((ci + 1) / len(combos),
                          text=f"{progress_label}combo {ci + 1}/{len(combos)}  ·  "
                               f"n_chunks={nc}  max_df={mxdf}")
        for r in agreement_curve(method, nc, mxdf, metric, tol_chunks, K_VALS):
            rows.append({"combo_label": f"nc={nc} · max_df={mxdf:.2f}",
                         "n_chunks": nc, "max_df": mxdf, **r})
    progress.empty()
    return pd.DataFrame(rows)


def choose_k(df, sel):
    """k* = argmax over k ≥ K_MIN of the mean (across combinations) excess on the selection
    pairs, and the best combination at k*."""
    ex = f"{sel}_excess"
    eligible = df[df["k"] >= K_MIN]
    k_star = int(eligible.groupby("k")[ex].mean().idxmax())
    best = eligible[eligible["k"] == k_star].nlargest(1, ex).iloc[0]
    return k_star, best


def local_peaks(df, sel):
    """k values where the mean excess curve is a local maximum (endpoints included)."""
    curve = df.groupby("k")[f"{sel}_excess"].mean()
    vals, ks = curve.to_numpy(), curve.index.to_numpy()
    return [(int(ks[i]), float(vals[i])) for i in range(len(vals))
            if (i == 0 or vals[i] > vals[i - 1]) and (i == len(vals) - 1 or vals[i] > vals[i + 1])]


def reporting_setting():
    """The config's reporting setting (the `controls` defaults used for Tables 1 and 3)."""
    return {
        "n_chunks":  _c["n_chunks"]["default"], "min_df": _c["min_df"]["default"],
        "max_df":    _c["max_df"]["default"],
        "ngram_min": _c["ngram_range"]["default_min"], "ngram_max": _c["ngram_range"]["default_max"],
    }


def render_validation(row, metric, k_star, where):
    """Held-out score: source ↔ translation agreement at k*."""
    m = METRIC_NAME[metric]
    st.markdown(f"**Held-out validation — {_src} ↔ translation {m} at k = {k_star}, {where}**")
    _v1, _v2, _v3 = st.columns(3)
    _v1.metric("Observed", f"{row['cross_obs']:.3f}")
    _v2.metric("Chance 95th pct", f"{row['cross_p95']:.3f}")
    _v3.metric("Excess", f"{row['cross_excess']:+.3f}")
    verdict = ("above chance — the structure chosen on the original carries over to the translations"
               if row["cross_excess"] > 0 else
               "not above chance — the chosen structure does not carry over to the translations")
    st.caption(f"Translations played no part in choosing k. Agreement is {verdict}.")


def render_handoff(k_star, method, sel):
    """Button that opens Clustering (TF-IDF) at the reporting setting with k* for this session."""
    rep = reporting_setting()
    label = (f"Open in Clustering (TF-IDF) — reporting setting (n_chunks = {rep['n_chunks']}, "
             f"max_df = {rep['max_df']:.2f}), k = {k_star}")
    if st.button(label, key="handoff_to_tfidf", type="primary"):
        st.session_state[TFIDF_HANDOFF_KEY] = {
            **rep, "k": int(k_star), "source": f"k* from the {method} grid, {SELECTION_LABELS[sel]}",
        }
        st.switch_page(TFIDF_PAGE)
    st.caption("Carries k* to the clustering page at the reporting setting from config. The grid "
               "chooses k; n_chunks is a choice of resolution that agreement can't optimize "
               "(see *Robustness across the grid*).")


RESOLUTION_EXPANDER_TEXT = r"""
**Why not choose n_chunks by agreement?** Excess over chance isn't comparable across n_chunks:
the chance baseline falls as chunks get fewer, so small n can look better without agreeing
more. Agreement chooses **k** at each fixed n — and k\* doesn't depend on n.

**Stability instead.** At each n, every edition's k\* segmentation (TF-IDF → Ward, the primary
method) is compared token by token with its segmentations at nearby resolutions, n ± 10%
(relative, so large n isn't favored). Where small changes of resolution reorganize the
partition, n is too coarse to have settled; on the plateau, they don't. The rule takes the
**onset of the plateau**: the coarsest stable resolution, since larger chunks give more
reliable vocabulary profiles and more chunks past the onset add no stability. Stability
compares the model only with itself and never uses the editors, so the editorial test stays
independent (stability-based model selection: Ben-Hur, Elisseeff & Guyon 2002; Lange, Roth,
Braun & Buhmann 2004).

**Order.** Stability depends on k, so k is chosen first. That order is harmless here because
k\* is the same at every n on the grid.
"""


def render_resolution(k_star):
    """Stability of the k* partition across n_chunks (TF-IDF), its plateau, and where the
    reporting setting falls. Anchor: resolution."""
    st.divider()
    st.subheader("Resolution: Choosing n_chunks", anchor="resolution")
    rep = reporting_setting()
    stab = resolution_stability("TF-IDF", int(k_star), rep["min_df"], rep["max_df"],
                                (rep["ngram_min"], rep["ngram_max"]))
    onset, end = stability_plateau(stab)
    n_rep = rep["n_chunks"]
    st.caption(
        f"Token-level ARI between each edition's k = {k_star} partition at n and at n ± "
        f"{int(RES['neighborhood'] * 100)}% (TF-IDF → Ward; mean and least stable edition). "
        "Shaded: the stability plateau. **Recommended n_chunks = its onset**, the coarsest "
        "stable resolution."
    )
    _sm = dict(zip(stab["n_chunks"], stab["smoothed"]))
    _r1, _r2, _r3 = st.columns(3)
    _r1.metric("Plateau onset (recommended)", onset)
    _r2.metric("Plateau", f"{onset}–{end}")
    _r3.metric(f"Reporting setting (n = {n_rep})",
               "on the plateau" if onset <= n_rep <= end else "off the plateau",
               help=(f"Smoothed stability {_sm[n_rep]:.3f} vs. {_sm[onset]:.3f} at the onset"
                     if n_rep in _sm else "Outside the measured range"))
    fig = go.Figure()
    fig.add_vrect(x0=onset - 0.5, x1=end + 0.5, fillcolor="#00CC96", opacity=0.10, line_width=0,
                  annotation_text="plateau", annotation_position="top left")
    fig.add_trace(go.Scatter(x=stab["n_chunks"], y=stab["stability"], mode="lines+markers",
                             name="Mean over editions", line=dict(color="#1f77b4", width=1.5),
                             marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=stab["n_chunks"], y=stab["smoothed"], mode="lines",
                             name=f"Smoothed ({RES['smooth']})", line=dict(color="#1f77b4", width=3)))
    fig.add_trace(go.Scatter(x=stab["n_chunks"], y=stab["stability_min"], mode="lines",
                             name="Least stable edition", line=dict(color="gray", width=1, dash="dot")))
    fig.add_vline(x=onset, line_color="#00CC96", line_width=2,
                  annotation_text=f"onset {onset}", annotation_position="bottom right")
    if n_rep != onset:
        fig.add_vline(x=n_rep, line_color="crimson", line_dash="dash", line_width=1.5,
                      annotation_text=f"reporting {n_rep}", annotation_position="top right")
    fig.update_layout(height=340, margin=dict(l=60, r=30, t=10, b=50), plot_bgcolor="white",
                      xaxis=dict(title="n_chunks", dtick=2, showgrid=False, zeroline=False),
                      yaxis=dict(title="Stability (ARI across resolutions)", showgrid=True,
                                 gridcolor="#EEEEEE", zeroline=False),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig, width="stretch")
    with st.expander("Why stability chooses the resolution"):
        st.markdown(RESOLUTION_EXPANDER_TEXT)


def render_selection_results(df, sel, metric, k_label, method, tol_chunks=0, handoff=False,
                             resolution=False):
    """Agreement curves, k* distribution, k* with its robustness across the grid, held-out
    validation, and summary tables. With handoff, adds a button that opens Clustering (TF-IDF)
    at the reporting setting with k*; with resolution, adds the n_chunks stability section.
    Anchors: agreement-chart, k-distribution, best-params, summary (resolution). Returns k*."""
    m      = METRIC_NAME[metric]
    ex     = f"{sel}_excess"
    sel_lb = SELECTION_LABELS[sel]

    # ── Excess curves ──────────────────────────────────────────────────────────
    st.subheader(f"{m} Excess over Chance vs. k — All Combinations", anchor="agreement-chart")
    st.caption(
        f"Pairs: **{sel_lb}**. Excess = mean pairwise {m} − chance 95th percentile (cyclic-shift "
        "baseline). Above 0 = agreement beyond chance. Each gray curve = one (n_chunks, max_df) "
        "combination; bold = mean across combinations."
    )
    fig = go.Figure()
    for label, gdf in df.groupby("combo_label", sort=False):
        fig.add_trace(go.Scatter(
            x=gdf["k"], y=gdf[ex], mode="lines", line=dict(color="#CCCCCC", width=1),
            showlegend=False, customdata=gdf[[f"{sel}_obs", f"{sel}_p95"]].values,
            hovertemplate=(f"{label}<br>k=%{{x}}<br>excess=%{{y:+.3f}}<br>"
                           "observed=%{customdata[0]:.3f} · chance=%{customdata[1]:.3f}<extra></extra>"),
        ))
    mean_c = df.groupby("k")[[ex, f"{sel}_obs", f"{sel}_p95"]].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=mean_c["k"], y=mean_c[ex], mode="lines+markers", name="Mean across combos",
        line=dict(color="#1f77b4", width=3), marker=dict(size=6),
        customdata=mean_c[[f"{sel}_obs", f"{sel}_p95"]].values,
        hovertemplate=("mean · k=%{x}<br>excess=%{y:+.3f}<br>"
                       "observed=%{customdata[0]:.3f} · chance=%{customdata[1]:.3f}<extra></extra>"),
    ))
    fig.add_hline(y=0, line_color="gray", line_width=1, line_dash="dot",
                  annotation_text="chance (95th pct)", annotation_position="bottom right")
    if K_MIN > min(K_VALS):
        fig.add_vrect(x0=min(K_VALS) - 0.5, x1=K_MIN - 0.5, fillcolor="gray", opacity=0.08,
                      line_width=0, annotation_text="not used for k*", annotation_position="top left")
    fig.update_layout(
        height=400, margin=dict(l=60, r=30, t=10, b=50), plot_bgcolor="white",
        xaxis=dict(title=k_label, dtick=1, showgrid=False, zeroline=False),
        yaxis=dict(title=f"{m} − chance", showgrid=True, gridcolor="#EEEEEE", zeroline=False),
        legend=dict(x=0.75, y=0.98),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Local peaks of the mean excess: " + " · ".join(
        f"**k = {k}** ({v:+.3f})" for k, v in local_peaks(df, sel)))
    with st.expander("Why excess over chance, and why hold out translations?"):
        st.markdown(SELECTION_EXPANDER_TEXT)

    # ── k* distribution ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Distribution of Optimal k*", anchor="k-distribution")
    st.caption(f"How many parameter combinations reach their maximum {m} excess at each k "
               f"(k ≥ {K_MIN}).")
    _elig = df[df["k"] >= K_MIN]
    per_combo = _elig.loc[_elig.groupby("combo_label")[ex].idxmax()]
    counts = per_combo["k"].value_counts().reindex(K_VALS, fill_value=0).reset_index()
    counts.columns = ["k*", "count"]
    fig_bar = px.bar(counts, x="k*", y="count", labels={"k*": k_label, "count": "# combinations"})
    fig_bar.update_layout(height=280, margin=dict(l=60, r=30, t=10, b=50), plot_bgcolor="white",
                          xaxis=dict(dtick=1, showgrid=False, zeroline=False),
                          yaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False))
    st.plotly_chart(fig_bar, width="stretch")

    # ── k* and its robustness ──────────────────────────────────────────────────
    st.divider()
    st.subheader("k* and Its Robustness Across the Grid", anchor="best-params")
    k_star, _ = choose_k(df, sel)
    at_k = df[df["k"] == k_star]
    rep = reporting_setting()
    rep_rows = [r for r in agreement_curve(method, rep["n_chunks"], rep["max_df"], metric, tol_chunks,
                                           K_VALS, min_df=rep["min_df"],
                                           ngram_range=(rep["ngram_min"], rep["ngram_max"]))
                if r["k"] == k_star]
    st.caption(
        f"k* = the k with the highest mean excess across all {df['combo_label'].nunique()} "
        f"combinations (selection pairs: {sel_lb}). The grid chooses **k**, not n_chunks or max_df: "
        "see the note below the robustness table."
    )
    _c1, _c2, _c3, _c4 = st.columns(4)
    _c1.metric("Optimal k*", k_star)
    _c2.metric("Mean excess at k*", f"{at_k[ex].mean():+.3f}",
               help=f"Across the grid. Observed {at_k[f'{sel}_obs'].mean():.3f} vs. chance "
                    f"{at_k[f'{sel}_p95'].mean():.3f}")
    _c3.metric("Combinations with k* = this k", f"{int((per_combo['k'] == k_star).sum())} of {len(per_combo)}")
    if rep_rows:
        _c4.metric(f"Excess at reporting setting", f"{rep_rows[0][ex]:+.3f}",
                   help=f"n_chunks = {rep['n_chunks']}, max_df = {rep['max_df']:.2f} (config defaults)")
    if sel == "within":
        if rep_rows:
            render_validation(rep_rows[0], metric, k_star,
                              f"reporting setting (n_chunks = {rep['n_chunks']}, max_df = {rep['max_df']:.2f})")
        st.caption(f"Across the whole grid, held-out excess at k = {k_star}: mean "
                   f"{at_k['cross_excess'].mean():+.3f}, above chance in "
                   f"{int((at_k['cross_excess'] > 0).sum())} of {len(at_k)} combinations.")
    if handoff:
        render_handoff(k_star, method, sel)

    pivot = at_k.pivot(index="n_chunks", columns="max_df", values=ex)
    fig_heat = px.imshow(pivot, labels=dict(x="max_df", y="n_chunks", color=f"Excess at k={k_star}"),
                         color_continuous_scale=_v["colorscale_ari"], aspect="auto", text_auto=".3f")
    fig_heat.update_layout(height=300, margin=dict(l=60, r=30, t=30, b=50))
    st.plotly_chart(fig_heat, width="stretch")

    # Robustness by n_chunks: observed, chance and excess (mean ± SD across max_df)
    st.markdown(f"**Robustness across the grid at k = {k_star}**")
    rob = (at_k.groupby("n_chunks")
               .agg(observed=(f"{sel}_obs", "mean"), chance=(f"{sel}_p95", "mean"),
                    excess=(ex, "mean"), excess_sd=(ex, "std"), excess_min=(ex, "min"),
                    excess_max=(ex, "max"))
               .reset_index())
    rob["k* here"] = [f"{int((per_combo[per_combo['n_chunks'] == n]['k'] == k_star).sum())} of "
                      f"{int((per_combo['n_chunks'] == n).sum())}" for n in rob["n_chunks"]]
    fig_rob = go.Figure()
    fig_rob.add_trace(go.Scatter(x=rob["n_chunks"], y=rob["observed"], mode="lines+markers",
                                 name=f"Observed {m}", line=dict(color="#1f77b4", width=2)))
    fig_rob.add_trace(go.Scatter(x=rob["n_chunks"], y=rob["chance"], mode="lines+markers",
                                 name="Chance (95th pct)", line=dict(color="gray", width=2, dash="dot")))
    fig_rob.add_trace(go.Scatter(x=rob["n_chunks"], y=rob["excess"], mode="lines+markers",
                                 name="Excess (± SD across max_df)", line=dict(color="#EF553B", width=2),
                                 error_y=dict(type="data", array=rob["excess_sd"].fillna(0), visible=True)))
    fig_rob.update_layout(height=320, margin=dict(l=60, r=30, t=10, b=50), plot_bgcolor="white",
                          xaxis=dict(title="n_chunks", dtick=5, showgrid=False, zeroline=False),
                          yaxis=dict(title=m, showgrid=True, gridcolor="#EEEEEE", zeroline=False),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    st.plotly_chart(fig_rob, width="stretch")
    st.dataframe(rob.rename(columns={"n_chunks": "n_chunks", "observed": f"{m} observed",
                                     "chance": "Chance 95th pct", "excess": "Excess (mean)",
                                     "excess_sd": "Excess SD", "excess_min": "Excess min",
                                     "excess_max": "Excess max"}).round(3),
                 width="stretch", hide_index=True)
    st.caption(
        "**Excess isn't comparable across n_chunks.** The chance baseline falls as chunks get "
        "fewer (fewer rotations, coarser sequences), so small n_chunks can show a higher excess "
        "without higher agreement. Use excess to compare k at a fixed n_chunks, which is how k* "
        "is chosen; read this table for whether k* and its agreement hold across resolutions, and "
        "prefer settings where agreement is high *and* steady (low SD). The highest single cell "
        "in the grid is usually the luckiest, not the best."
    )

    if resolution:
        render_resolution(k_star)

    # ── Tables ─────────────────────────────────────────────────────────────────
    cols = {f"{sel}_obs": f"{m} observed", f"{sel}_p95": "Chance 95th pct", ex: "Excess"}
    if sel == "within":
        cols.update({"cross_obs": f"Held-out {m}", "cross_p95": "Held-out chance",
                     "cross_excess": "Held-out excess"})

    st.subheader("Agreement by k (mean across the grid)")
    by_k = df.groupby("k")[list(cols)].mean().reset_index()
    by_k.insert(len(by_k.columns), "Excess SD", df.groupby("k")[ex].std().to_numpy())
    st.dataframe(by_k.rename(columns=cols).round(4), width="stretch", hide_index=True)

    st.divider()
    st.subheader(f"Summary — Every Combination at Its Own k*", anchor="summary")
    st.caption("Raw detail, sorted by excess. Single cells are noisy; compare with the robustness "
               "view above before reading anything into the top rows.")
    summary = per_combo.rename(columns={"k": "k*"}).sort_values(ex, ascending=False)
    st.dataframe(summary[["combo_label", "n_chunks", "max_df", "k*", *cols]].rename(columns=cols)
                 .round(4), width="stretch", hide_index=True)
    return k_star
