"""
Cross-Edition Cluster Comparison — hold one parameter fixed and sweep the rest,
comparing cluster sequences across all editions by ARI and by boundary F1 (with
positional tolerance), each against a cyclic-shift chance baseline and split into
source-language and source ↔ translation pairs.

Method selector chooses the vectorization / clustering backend:
  TF-IDF     — TF-IDF matrix → Ward HAC
  LSA        — TF-IDF → TruncatedSVD (n=10) → Ward HAC
  Cosine-Sim — TF-IDF → cosine-similarity matrix → Ward HAC
  NMF        — TF-IDF → NMF → argmax(THETA) labels (k is a model parameter)
"""

import os
import yaml
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from toc import render_toc
from utils import (find_token_file, load_tokens, cluster_string,
                   pairwise_hamming_matrix, pairwise_ari_matrix)
from grid_search_boundary_core import (N_COMPONENTS, SOURCE_LANG, cluster_labels,
                                       ari_summary, f1_summary, boundary_tol, pairwise_f1_matrix)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
LANG_LABELS  = cfg["languages"]
_v           = cfg["visualization"]

st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)

METHOD_LABELS = {
    "TF-IDF":     "TF-IDF → Ward HAC",
    "LSA":        f"TF-IDF → LSA (SVD, n={N_COMPONENTS}) → Ward HAC",
    "Cosine-Sim": "TF-IDF → cosine-similarity matrix → Ward HAC",
    "NMF":        "TF-IDF → NMF → argmax(THETA)",
}

METHOD_CAPTIONS = {
    "TF-IDF": (
        "Runs Ward HAC on raw TF-IDF vectors. Each edition is divided into `n_chunks` equal bins "
        "via `pd.cut`. Select a parameter to sweep; all others stay fixed."
    ),
    "LSA": (
        f"TF-IDF → TruncatedSVD (n_components={N_COMPONENTS}) → Ward HAC. "
        "Latent-semantic projection before clustering reduces noise and vocabulary mismatch."
    ),
    "Cosine-Sim": (
        "TF-IDF → cosine-similarity matrix → Ward HAC. Each chunk is represented by its "
        "similarity profile to all other chunks, capturing second-order distributional structure."
    ),
    "NMF": (
        "TF-IDF → NMF → argmax(THETA) labels. `k` is a model parameter here — "
        "sweeping k requires a full NMF re-fit per edition, not just a dendrogram re-cut."
    ),
}

HOLDOUT_DEFS = {
    "k":         {"dtype": int,   "vals": list(range(2, 21))},
    "n_chunks":  {"dtype": int,   "vals": list(range(10, 55, 5))},
    "min_df":    {"dtype": int,   "vals": list(range(1, 11))},
    "max_df":    {"dtype": float, "vals": [round(v, 2) for v in np.arange(0.20, 0.95, 0.10)]},
    "ngram_max": {"dtype": int,   "vals": list(range(1, 5))},
}


def render_agreement_chart(df, prefix, x_col, x_label, y_title, key):
    """Within / cross / all-pairs agreement lines over a shaded chance band."""
    _src = LANG_LABELS[SOURCE_LANG]
    series = [
        ("null_p95",  "Chance (95th pct)", dict(mode="lines", line=dict(width=0), fill="tozeroy",
                                                fillcolor="rgba(150,150,150,0.2)")),
        ("null_mean", "Chance (mean)",     dict(mode="lines", line=dict(color="gray", width=1, dash="dot"))),
        ("all",       "All pairs",         dict(mode="lines", line=dict(color="gray", width=1.5, dash="dash"))),
        ("within",    f"{_src} ↔ {_src}",  dict(mode="lines+markers", line=dict(color="#1f77b4", width=2),
                                                marker=dict(size=6))),
        ("cross",     f"{_src} ↔ translation", dict(mode="lines+markers", line=dict(color="#EF553B", width=2),
                                                    marker=dict(size=6))),
    ]
    fig = go.Figure()
    for col, name, style in series:
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[f"{prefix}_{col}"], name=name,
            hovertemplate=f"{x_label}=%{{x}}<br>{name}=%{{y:.3f}}<extra></extra>",
            **style,
        ))
    fig.update_layout(
        height=360, margin=dict(l=60, r=30, t=20, b=50), plot_bgcolor="white",
        xaxis=dict(title=x_label, showgrid=False, zeroline=False),
        yaxis=dict(title=y_title, range=[-0.05, 1.05],
                   showgrid=True, gridcolor="#EEEEEE", zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, width="stretch", key=key)


# ── Controls ──────────────────────────────────────────────────────────────────
st.title("Cross-Edition Cluster Comparison")
render_toc([
    ("Cluster Agreement (ARI)",    "agreement-chart"),
    ("Boundary Agreement (F1)",    "boundary-chart"),
    ("Pairwise Agreement Matrices", "pairwise-matrices"),
])

_c         = cfg["controls"]
_col_ratios = cfg["layout"]["column_ratios"]
cols        = st.columns(_col_ratios[:6])

method = cols[0].selectbox("Method", list(METHOD_LABELS.keys()),
                            format_func=lambda x: METHOD_LABELS[x])

k_label = "k (topics)" if method == "NMF" else "k (clusters)"
holdout_labels = {
    "k": k_label, "n_chunks": "n_chunks", "min_df": "min_df",
    "max_df": "max_df", "ngram_max": "ngram max",
}

holdout_param = cols[1].selectbox(
    "Hold-out parameter", list(HOLDOUT_DEFS.keys()),
    format_func=lambda x: holdout_labels[x],
)

n_chunks  = cols[2].number_input("n_chunks",  min_value=_c["n_chunks"]["min"], max_value=_c["n_chunks"]["max"],
                                  value=_c["n_chunks"]["default"], step=_c["n_chunks"]["step"],
                                  disabled=(holdout_param == "n_chunks"))
min_df    = cols[3].number_input("min_df",    _c["min_df"]["min"], _c["min_df"]["max"],
                                  _c["min_df"]["default"], step=_c["min_df"]["step"],
                                  disabled=(holdout_param == "min_df"))
max_df    = cols[3].number_input("max_df",    _c["max_df"]["min"], _c["max_df"]["max"],
                                  _c["max_df"]["default"], step=_c["max_df"]["step"],
                                  format="%.2f", disabled=(holdout_param == "max_df"))
_ng       = _c["ngram_range"]
ngram_min = cols[4].number_input("ngram min", _ng["min_n"], _ng["max_n"], _ng["default_min"], step=1)
ngram_max = cols[4].number_input("ngram max", _ng["min_n"], _ng["max_n"], _ng["default_max"], step=1,
                                  disabled=(holdout_param == "ngram_max"))
ngram_max = max(ngram_max, ngram_min)
k         = cols[5].number_input(k_label, _c["n_clusters"]["min"], _c["n_clusters"]["max"],
                                  _c["n_clusters"]["default"], step=1,
                                  disabled=(holdout_param == "k"))

st.caption(METHOD_CAPTIONS[method])
st.divider()

# ── Sweep ─────────────────────────────────────────────────────────────────────
sweep_vals = HOLDOUT_DEFS[holdout_param]["vals"]

all_rows     = []
summary_rows = []
progress     = st.progress(0, text="Running…")

for step_i, val in enumerate(sweep_vals):
    progress.progress(
        (step_i + 1) / len(sweep_vals),
        text=f"{holdout_labels[holdout_param]} = {val}  ({step_i + 1}/{len(sweep_vals)})",
    )

    p_n_chunks  = int(val)   if holdout_param == "n_chunks"  else int(n_chunks)
    p_min_df    = int(val)   if holdout_param == "min_df"    else int(min_df)
    p_max_df    = float(val) if holdout_param == "max_df"    else float(max_df)
    p_ngram_max = max(int(ngram_min), int(val)) if holdout_param == "ngram_max" else int(ngram_max)
    p_k         = int(val)   if holdout_param == "k"         else int(k)

    edition_label_arrays = []
    edition_langs        = []

    for src_id, meta in SOURCES_META.items():
        token_path = find_token_file(src_id)
        if token_path is None:
            continue
        TOKEN = load_tokens(src_id, token_path)
        labels = cluster_labels(method, src_id, token_path, p_n_chunks, p_min_df, p_max_df,
                                p_k, (int(ngram_min), p_ngram_max))
        if labels is None:
            continue
        cstr = cluster_string(labels)
        edition_label_arrays.append(labels)
        edition_langs.append(meta["lang"])
        all_rows.append({
            holdout_param:    val,
            "edition":        meta["label"],
            "lang":           LANG_LABELS[meta["lang"]],
            "lang_code":      meta["lang"],
            "n_tokens":       len(TOKEN),
            "k_actual":       int(len(np.unique(labels))),
            "cluster_string": cstr,
            "labels":         labels.tolist(),
        })

    summary_rows.append({holdout_param: val,
                         **{f"ari_{key}": v for key, v in
                            ari_summary(edition_label_arrays, edition_langs).items()}})

progress.empty()

if not all_rows:
    st.warning("No editions could be processed with the current parameters.")
    st.stop()

df_all     = pd.DataFrame(all_rows)
df_summary = pd.DataFrame(summary_rows)

# ── Agreement charts ──────────────────────────────────────────────────────────
x_label  = holdout_labels[holdout_param]
_src_lbl = LANG_LABELS[SOURCE_LANG]

st.subheader(f"Cluster Agreement (ARI) vs. {x_label}", anchor="agreement-chart")
st.caption(
    f"Mean pairwise ARI between editions' cluster sequences. **{_src_lbl} ↔ {_src_lbl}** compares "
    f"transcriptions of the original; **{_src_lbl} ↔ translation** compares the original with "
    "Spanish and English versions, which share almost no vocabulary with it. "
    "The gray band is chance: agreement below its upper edge is no better than misaligned sequences."
)
render_agreement_chart(df_summary, "ari", holdout_param, x_label, "Mean pairwise ARI", key="ari_chart")

st.subheader(f"Boundary Agreement (F1) vs. {x_label}", anchor="boundary-chart")
st.caption(
    "Mean pairwise F1 between the positions where each edition's cluster label changes. Two "
    "boundaries match if they lie within the tolerance of each other, so a boundary one chunk "
    "off — common when translations expand or contract a passage — still counts as agreement. "
    "ARI would count it as disagreement. At high k, boundaries fall every chunk or two and any "
    "tolerance matches almost everything; the chance band shows where that happens."
)
_tc = _c["boundary_tol_chunks"]
tol_chunks = st.slider("Boundary tolerance (± chunks)", _tc["min"], _tc["max"], _tc["default"],
                       _tc["step"], help="0 = boundaries must fall at the same chunk.")

f1_rows = []
for val, grp in df_all.groupby(holdout_param, sort=False):
    _tol = boundary_tol(len(grp["labels"].iloc[0]), tol_chunks)
    f1_rows.append({holdout_param: val,
                    **{f"f1_{key}": v for key, v in
                       f1_summary([np.array(l) for l in grp["labels"]], grp["lang_code"].tolist(),
                                  _tol).items()}})
df_summary = df_summary.merge(pd.DataFrame(f1_rows), on=holdout_param, how="left")
render_agreement_chart(df_summary, "f1", holdout_param, x_label, "Mean pairwise boundary F1",
                       key="f1_chart")

_table_cols = {holdout_param: x_label}
for prefix, metric in [("ari", "ARI"), ("f1", "F1")]:
    _table_cols.update({
        f"{prefix}_within":    f"{metric} {_src_lbl} ↔ {_src_lbl}",
        f"{prefix}_cross":     f"{metric} {_src_lbl} ↔ translation",
        f"{prefix}_all":       f"{metric} all pairs",
        f"{prefix}_null_p95":  f"{metric} chance 95th pct",
    })
st.dataframe(df_summary[list(_table_cols)].rename(columns=_table_cols).round(4),
             width="stretch", hide_index=True)

with st.expander("How the chance baselines work"):
    st.markdown(
        "ARI ≈ 0 is chance only for randomly *shuffled* labels. Narrative cluster sequences come "
        "in contiguous blocks, and any two block segmentations of the same text share structure "
        "just by being contiguous, so unrelated sequences can score well above 0.\n\n"
        "The baselines keep each edition's sequence intact and only break the alignment: for "
        "every pair of editions, one sequence is rotated by every possible offset and the score "
        "recomputed. The band's upper edge is the 95th percentile of those scores, pooled over "
        "all pairs. Agreement above it means editions put their blocks in the *same places*, "
        "not merely that both are blocky.\n\n"
        "For ARI, every offset from 1 to n−1 chunks is used. For boundary F1, offsets within the "
        "tolerance are skipped, since the F1 by definition treats those as matches.\n\n"
        "The boundary tolerance is fixed in chunks rather than scaled by k as on Grid Search "
        "(Boundary). That page's τ = α/2(k−1) assumes k−1 boundaries, but recurring clusters "
        "produce more (about 7 at k = 6, 15 at k = 12), so k-scaled windows overlap and even "
        "misaligned sequences score near 1.\n\n"
        "Caveats: the two Ximénez transcriptions are close to the same text, which lifts "
        f"{_src_lbl} ↔ {_src_lbl} agreement. Chunks are aligned by token position, and "
        "translations expand and contract unevenly. Boundary F1 absorbs drift within the tolerance; "
        "ARI does not, so its cross-language scores are conservative."
    )

# ── Pairwise matrices ──────────────────────────────────────────────────────────
st.divider()
st.subheader("Pairwise Agreement Matrices", anchor="pairwise-matrices")
st.caption(
    "Select a sweep value to inspect how each pair of editions compares. "
    "Both scores: 1 = identical; compare with the chance values shown below each matrix."
)

sel_val = st.selectbox(x_label, sweep_vals, index=len(sweep_vals) - 1)

if HOLDOUT_DEFS[holdout_param]["dtype"] == float:
    df_sel = df_all[np.isclose(df_all[holdout_param].astype(float), float(sel_val))]
    sum_sel = df_summary[np.isclose(df_summary[holdout_param].astype(float), float(sel_val))]
else:
    df_sel = df_all[df_all[holdout_param] == sel_val]
    sum_sel = df_summary[df_summary[holdout_param] == sel_val]

if len(df_sel) >= 2:
    edition_labels = [f"{e} ({l})" for e, l in zip(df_sel["edition"], df_sel["lang"])]
    _sel_labels = [np.array(lbl) for lbl in df_sel["labels"].tolist()]
    _sel_tol    = boundary_tol(len(_sel_labels[0]), tol_chunks)
    ari_df = pd.DataFrame(pairwise_ari_matrix(_sel_labels),
                          index=edition_labels, columns=edition_labels).round(3)
    f1_df  = pd.DataFrame(pairwise_f1_matrix(_sel_labels, _sel_tol),
                          index=edition_labels, columns=edition_labels).round(3)

    _h      = max(300, len(edition_labels) * 50 + 100)
    _margin = dict(l=20, r=20, t=40, b=20)
    _s      = sum_sel.iloc[0] if not sum_sel.empty else None
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**ARI** (cluster structure)")
        fig_ari = px.imshow(ari_df, color_continuous_scale=_v["colorscale_cluster"], zmin=0, zmax=1,
                            text_auto=".3f", aspect="auto")
        fig_ari.update_layout(height=_h, margin=_margin, coloraxis_colorbar=dict(title="ARI"))
        fig_ari.update_traces(textfont_size=11)
        st.plotly_chart(fig_ari, width="stretch")
        if _s is not None:
            st.caption(f"Chance: mean {_s['ari_null_mean']:.3f}, 95th percentile {_s['ari_null_p95']:.3f}.")

    with col_right:
        st.markdown(f"**Boundary F1** (± {tol_chunks} chunk{'s' if tol_chunks != 1 else ''})")
        fig_f1 = px.imshow(f1_df, color_continuous_scale=_v["colorscale_boundary"], zmin=0, zmax=1,
                           text_auto=".3f", aspect="auto")
        fig_f1.update_layout(height=_h, margin=_margin, coloraxis_colorbar=dict(title="F1"))
        fig_f1.update_traces(textfont_size=11)
        st.plotly_chart(fig_f1, width="stretch")
        if _s is not None:
            st.caption(f"Chance: mean {_s['f1_null_mean']:.3f}, 95th percentile {_s['f1_null_p95']:.3f}.")

    with st.expander("Hamming distance matrix (label-order sensitive)"):
        st.caption(
            "Position-by-position mismatch of the letter sequences (0 = identical). Letters are "
            "assigned by first appearance, so one early difference relabels every later chunk; "
            "treat this as a rough view and prefer ARI or boundary F1."
        )
        ham_mat = pairwise_hamming_matrix(df_sel["cluster_string"].tolist())
        ham_df  = pd.DataFrame(ham_mat, index=edition_labels, columns=edition_labels).round(3)
        fig_ham = px.imshow(ham_df, color_continuous_scale=_v["colorscale_ari"], zmin=0, zmax=1,
                            text_auto=".3f", aspect="auto")
        fig_ham.update_layout(height=_h, margin=_margin, coloraxis_colorbar=dict(title="Hamming"))
        fig_ham.update_traces(textfont_size=11)
        st.plotly_chart(fig_ham, width="stretch")

# ── Detailed cluster strings ───────────────────────────────────────────────────
st.divider()
with st.expander("Cluster sequences by edition and parameter value"):
    st.dataframe(
        df_all[[holdout_param, "edition", "lang", "n_tokens", "k_actual", "cluster_string"]],
        width="stretch", hide_index=True,
    )
