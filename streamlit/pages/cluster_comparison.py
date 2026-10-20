"""
Cross-Edition Cluster Comparison — hold one parameter fixed and sweep the rest,
comparing cluster sequences across all editions using ARI against a cyclic-shift
chance baseline, split into source-language and source ↔ translation pairs.

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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from toc import render_toc
from sklearn.metrics import adjusted_rand_score
from utils import (find_token_file, load_tokens, threshold_for_k, cluster_string,
                   pairwise_hamming_matrix, pairwise_ari_matrix, cyclic_shift_ari)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
LANG_LABELS  = cfg["languages"]
SOURCE_LANG  = cfg["source_lang"]
_v           = cfg["visualization"]
NMF_MAX_ITER = cfg["model"]["nmf_max_iter"]
N_COMPONENTS = 10

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


@st.cache_data(show_spinner=False)
def run_hac(src_id, token_path, n_chunks, min_df, max_df, method, ngram_range=(1, 1)):
    TOKEN = load_tokens(src_id, token_path)
    tr = TOKEN.reset_index()
    tr["chunk_num"] = pd.cut(tr.index, n_chunks, labels=list(range(n_chunks)))
    chunks_list = (
        tr.groupby("chunk_num", observed=True)["term_str"]
        .apply(lambda x: " ".join(x.dropna()))
        .tolist()
    )
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
    if method == "LSA":
        n_comp = min(N_COMPONENTS, X.shape[0] - 1, X.shape[1] - 1)
        if n_comp < 2:
            return None
        mat = TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(X)
    elif method == "Cosine-Sim":
        mat = (X @ X.T).toarray()
    else:
        mat = X.toarray()
    Z = linkage(pdist(mat, metric="euclidean"), method="ward")
    return {"Z": Z, "n_chunks": len(chunks_list)}


@st.cache_data(show_spinner=False)
def run_nmf_labels(src_id, token_path, n_chunks, min_df, max_df, k, ngram_range=(1, 1)):
    TOKEN = load_tokens(src_id, token_path)
    tr = TOKEN.reset_index()
    tr["chunk_num"] = pd.cut(tr.index, n_chunks, labels=list(range(n_chunks)))
    chunks_list = (
        tr.groupby("chunk_num", observed=True)["term_str"]
        .apply(lambda x: " ".join(x.dropna()))
        .tolist()
    )
    if len(chunks_list) < max(3, k):
        return None
    try:
        vec = TfidfVectorizer(
            lowercase=True, max_df=max_df, min_df=min_df,
            strip_accents=None, norm="l2", ngram_range=ngram_range,
        )
        X = vec.fit_transform(chunks_list)
    except ValueError:
        return None
    if X.shape[1] < k:
        return None
    try:
        THETA = NMF(n_components=k, init="nndsvda", max_iter=NMF_MAX_ITER).fit_transform(X)
    except Exception:
        return None
    return {"labels": np.argmax(THETA, axis=1), "n_chunks": len(chunks_list)}


def agreement_summary(label_arrays, langs):
    """Mean pairwise ARI overall, within the source language, and source ↔ translation,
    plus a cyclic-shift chance baseline pooled over all pairs."""
    within, cross, every, null = [], [], [], []
    for i in range(len(label_arrays)):
        for j in range(i + 1, len(label_arrays)):
            a, b = label_arrays[i], label_arrays[j]
            ari = adjusted_rand_score(a, b)
            every.append(ari)
            n_src = (langs[i] == SOURCE_LANG) + (langs[j] == SOURCE_LANG)
            if n_src == 2:
                within.append(ari)
            elif n_src == 1:
                cross.append(ari)
            null.extend(cyclic_shift_ari(a, b))
    _mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return {
        "ari_all":    _mean(every),
        "ari_within": _mean(within),
        "ari_cross":  _mean(cross),
        "null_mean":  _mean(null),
        "null_p95":   float(np.percentile(null, 95)) if null else float("nan"),
    }


# ── Controls ──────────────────────────────────────────────────────────────────
st.title("Cross-Edition Cluster Comparison")
render_toc([
    ("Agreement vs. Parameter",    "agreement-chart"),
    ("Pairwise Agreement Matrix",  "pairwise-matrices"),
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
        if method == "NMF":
            result = run_nmf_labels(src_id, token_path, p_n_chunks, p_min_df, p_max_df,
                                    p_k, (int(ngram_min), p_ngram_max))
            if result is None:
                continue
            labels = result["labels"]
        else:
            result = run_hac(src_id, token_path, p_n_chunks, p_min_df, p_max_df,
                             method, (int(ngram_min), p_ngram_max))
            if result is None:
                continue
            Z      = result["Z"]
            n_act  = result["n_chunks"]
            labels = fcluster(Z, threshold_for_k(Z, p_k, n_act), criterion="distance")
        cstr = cluster_string(labels)
        edition_label_arrays.append(labels)
        edition_langs.append(meta["lang"])
        all_rows.append({
            holdout_param:    val,
            "edition":        meta["label"],
            "lang":           LANG_LABELS[meta["lang"]],
            "n_tokens":       len(TOKEN),
            "k_actual":       int(len(np.unique(labels))),
            "cluster_string": cstr,
            "labels":         labels.tolist(),
        })

    summary_rows.append({holdout_param: val,
                         **agreement_summary(edition_label_arrays, edition_langs)})

progress.empty()

if not all_rows:
    st.warning("No editions could be processed with the current parameters.")
    st.stop()

df_all     = pd.DataFrame(all_rows)
df_summary = pd.DataFrame(summary_rows)

# ── Sweep line chart ──────────────────────────────────────────────────────────
x_label  = holdout_labels[holdout_param]
_src_lbl = LANG_LABELS[SOURCE_LANG]
st.subheader(f"Cross-Edition Agreement vs. {x_label}", anchor="agreement-chart")
st.caption(
    f"Mean pairwise ARI between editions' cluster sequences. **{_src_lbl} ↔ {_src_lbl}** compares "
    f"transcriptions of the original; **{_src_lbl} ↔ translation** compares the original with "
    "Spanish and English versions, which share almost no vocabulary with it. "
    "The gray band is chance: agreement below its upper edge is no better than misaligned sequences."
)

fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=df_summary[holdout_param], y=df_summary["null_p95"],
    mode="lines", line=dict(width=0), fill="tozeroy", fillcolor="rgba(150,150,150,0.2)",
    name="Chance (95th pct)",
    hovertemplate=f"{x_label}=%{{x}}<br>Chance 95th pct=%{{y:.3f}}<extra></extra>",
))
fig_line.add_trace(go.Scatter(
    x=df_summary[holdout_param], y=df_summary["null_mean"],
    mode="lines", line=dict(color="gray", width=1, dash="dot"),
    name="Chance (mean)",
    hovertemplate=f"{x_label}=%{{x}}<br>Chance mean=%{{y:.3f}}<extra></extra>",
))
fig_line.add_trace(go.Scatter(
    x=df_summary[holdout_param], y=df_summary["ari_all"],
    mode="lines", line=dict(color="gray", width=1.5, dash="dash"),
    name="All pairs",
    hovertemplate=f"{x_label}=%{{x}}<br>All pairs=%{{y:.3f}}<extra></extra>",
))
fig_line.add_trace(go.Scatter(
    x=df_summary[holdout_param], y=df_summary["ari_within"],
    mode="lines+markers", line=dict(color="#1f77b4", width=2), marker=dict(size=6),
    name=f"{_src_lbl} ↔ {_src_lbl}",
    hovertemplate=f"{x_label}=%{{x}}<br>{_src_lbl} ↔ {_src_lbl}=%{{y:.3f}}<extra></extra>",
))
fig_line.add_trace(go.Scatter(
    x=df_summary[holdout_param], y=df_summary["ari_cross"],
    mode="lines+markers", line=dict(color="#EF553B", width=2), marker=dict(size=6),
    name=f"{_src_lbl} ↔ translation",
    hovertemplate=f"{x_label}=%{{x}}<br>{_src_lbl} ↔ translation=%{{y:.3f}}<extra></extra>",
))
fig_line.update_layout(
    height=360, margin=dict(l=60, r=30, t=20, b=50), plot_bgcolor="white",
    xaxis=dict(title=x_label, showgrid=False, zeroline=False),
    yaxis=dict(title="Mean pairwise ARI", range=[-0.05, 1.05],
               showgrid=True, gridcolor="#EEEEEE", zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
st.plotly_chart(fig_line, width="stretch")

st.dataframe(
    df_summary[[holdout_param, "ari_within", "ari_cross", "ari_all", "null_mean", "null_p95"]].rename(columns={
        holdout_param: x_label,
        "ari_within":  f"ARI {_src_lbl} ↔ {_src_lbl}",
        "ari_cross":   f"ARI {_src_lbl} ↔ translation",
        "ari_all":     "ARI all pairs",
        "null_mean":   "Chance mean",
        "null_p95":    "Chance 95th pct",
    }).round(4),
    width="stretch", hide_index=True,
)

with st.expander("How the chance baseline works"):
    st.markdown(
        "ARI ≈ 0 is chance only for randomly *shuffled* labels. Narrative cluster sequences come "
        "in contiguous blocks, and any two block segmentations of the same text share structure "
        "just by being contiguous, so unrelated sequences can score well above 0.\n\n"
        "The baseline here keeps each edition's sequence intact and only breaks the alignment: for "
        "every pair of editions, one sequence is rotated by every possible offset (1 to n−1 chunks) "
        "and the ARI recomputed. The band's upper edge is the 95th percentile of those scores, "
        "pooled over all pairs. Agreement above it means editions put their blocks in the "
        "*same places*, not merely that both are blocky.\n\n"
        "Caveats: the two Ximénez transcriptions are close to the same text, which lifts "
        f"{_src_lbl} ↔ {_src_lbl} agreement. Chunks are aligned by token position, and "
        "translations expand and contract unevenly, so cross-language scores are conservative."
    )

# ── Pairwise matrices ──────────────────────────────────────────────────────────
st.divider()
st.subheader("Pairwise Agreement Matrix", anchor="pairwise-matrices")
st.caption(
    "Select a sweep value to inspect how each pair of editions compares. "
    "ARI: 1 = identical structure; compare with the chance values shown below the matrix."
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
    ari_mat = pairwise_ari_matrix([np.array(lbl) for lbl in df_sel["labels"].tolist()])
    ari_df  = pd.DataFrame(ari_mat, index=edition_labels, columns=edition_labels).round(3)

    _h      = max(300, len(edition_labels) * 50 + 100)
    _margin = dict(l=20, r=20, t=40, b=20)

    fig_ari = px.imshow(ari_df, color_continuous_scale=_v["colorscale_cluster"], zmin=0, zmax=1,
                        text_auto=".3f", aspect="auto")
    fig_ari.update_layout(height=_h, margin=_margin, coloraxis_colorbar=dict(title="ARI"))
    fig_ari.update_traces(textfont_size=11)
    st.plotly_chart(fig_ari, width="stretch")
    if not sum_sel.empty:
        _s = sum_sel.iloc[0]
        st.caption(f"Chance at {x_label} = {sel_val}: mean {_s['null_mean']:.3f}, "
                   f"95th percentile {_s['null_p95']:.3f}.")

    with st.expander("Hamming distance matrix (label-order sensitive)"):
        st.caption(
            "Position-by-position mismatch of the letter sequences (0 = identical). Letters are "
            "assigned by first appearance, so one early difference relabels every later chunk; "
            "treat this as a rough view and prefer ARI."
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
