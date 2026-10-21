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
from grid_search_boundary_core import agreement_curve

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


def _bar(df, x, y, y_label, dtick=None):
    fig = px.bar(df, x=x, y=y, labels={x: x, y: y_label})
    fig.update_layout(height=240, margin=dict(l=60, r=30, t=10, b=50), plot_bgcolor="white",
                      xaxis=dict(dtick=dtick, showgrid=False, zeroline=False),
                      yaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False))
    return fig


def render_validation(row, metric, k_star):
    """Held-out score: source ↔ translation agreement at k* for the chosen combination."""
    m = METRIC_NAME[metric]
    st.markdown(f"**Held-out validation — {_src} ↔ translation {m} at k = {k_star}**")
    _v1, _v2, _v3 = st.columns(3)
    _v1.metric("Observed", f"{row['cross_obs']:.3f}")
    _v2.metric("Chance 95th pct", f"{row['cross_p95']:.3f}")
    _v3.metric("Excess", f"{row['cross_excess']:+.3f}")
    verdict = ("above chance — the structure chosen on the original carries over to the translations"
               if row["cross_excess"] > 0 else
               "not above chance — the chosen structure does not carry over to the translations")
    st.caption(f"Translations played no part in choosing k or the parameters. Agreement is {verdict}.")


def render_selection_results(df, sel, metric, k_label):
    """Agreement curves, k* distribution, best parameters, validation and summary tables.
    Anchors: agreement-chart, k-distribution, best-params, summary. Returns (k*, best row)."""
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

    # ── Best parameters ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Best Parameters", anchor="best-params")
    k_star, best = choose_k(df, sel)
    st.caption(f"k* = the k with the highest mean excess across combinations; best combination "
               f"= highest excess at k*. Selection pairs: {sel_lb}.")
    _c1, _c2, _c3, _c4 = st.columns(4)
    _c1.metric("Best n_chunks", int(best["n_chunks"]))
    _c2.metric("Best max_df", f"{best['max_df']:.2f}")
    _c3.metric("Optimal k*", k_star)
    _c4.metric(f"Excess at k*", f"{best[ex]:+.3f}",
               help=f"Observed {best[f'{sel}_obs']:.3f} vs. chance {best[f'{sel}_p95']:.3f}")
    if sel == "within":
        render_validation(best, metric, k_star)

    at_k = df[df["k"] == k_star]
    pivot = at_k.pivot(index="n_chunks", columns="max_df", values=ex)
    fig_heat = px.imshow(pivot, labels=dict(x="max_df", y="n_chunks", color=f"Excess at k={k_star}"),
                         color_continuous_scale=_v["colorscale_ari"], aspect="auto", text_auto=".3f")
    fig_heat.update_layout(height=300, margin=dict(l=60, r=30, t=30, b=50))
    st.plotly_chart(fig_heat, width="stretch")

    _col_nc, _col_mdf = st.columns(2)
    _col_nc.plotly_chart(_bar(at_k.groupby("n_chunks")[ex].mean().reset_index(), "n_chunks", ex,
                              f"Mean excess at k={k_star}", dtick=5), width="stretch")
    _col_mdf.plotly_chart(_bar(at_k.groupby("max_df")[ex].mean().reset_index(), "max_df", ex,
                               f"Mean excess at k={k_star}"), width="stretch")

    # ── Tables ─────────────────────────────────────────────────────────────────
    cols = {f"{sel}_obs": f"{m} observed", f"{sel}_p95": "Chance 95th pct", ex: "Excess"}
    if sel == "within":
        cols.update({"cross_obs": f"Held-out {m}", "cross_p95": "Held-out chance",
                     "cross_excess": "Held-out excess"})

    st.subheader("Best Parameters by k")
    st.caption(f"For each k, the (n_chunks, max_df) combination with the highest {m} excess.")
    best_by_k = df.loc[df.groupby("k")[ex].idxmax()].sort_values("k")
    st.dataframe(best_by_k[["k", "n_chunks", "max_df", *cols]].rename(columns=cols).round(4),
                 width="stretch", hide_index=True)

    st.divider()
    st.subheader(f"Summary — Sorted by Maximum {m} Excess", anchor="summary")
    st.caption("Each combination at its own best k (k*).")
    summary = per_combo.rename(columns={"k": "k*"}).sort_values(ex, ascending=False)
    st.dataframe(summary[["combo_label", "n_chunks", "max_df", "k*", *cols]].rename(columns=cols)
                 .round(4), width="stretch", hide_index=True)
    return k_star, best
