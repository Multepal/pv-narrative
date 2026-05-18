"""
Episode Boundaries — KLD-based novelty, transience, and resonance.

Implements the Novelty / Transience / Resonance framework from:
  Jurafsky et al. (2018) PNAS 115(10) doi:10.1073/pnas.1717729115

Each chunk is treated as a probability distribution over vocabulary
(normalized raw TF + Laplace smoothing). KL-divergence between
consecutive chunks gives a novelty signal; high-resonance chunks
are candidate episode boundaries.
"""

import os
import yaml
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from toc import render_toc
from utils import find_token_file, load_tokens, threshold_for_k, make_cluster_table, build_dendrogram_figure

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
LANG_LABELS  = cfg["languages"]

st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)


def _make_dist_fn(distance: str):
    if distance == "JSD":
        def fn(p, q):
            m = 0.5 * (p + q)
            return float(0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m)))
    elif distance == "Cosine":
        def fn(p, q):
            denom = np.linalg.norm(p) * np.linalg.norm(q)
            return float(1.0 - np.dot(p, q) / denom) if denom > 0 else 0.0
    else:  # KLD (default)
        def fn(p, q):
            return float(np.sum(p * np.log2(p / q)))
    return fn


@st.cache_data(show_spinner="Computing novelty curves…")
def compute_novelty(src_id, token_path, n_chunks, min_df, max_df, window, distance):
    TOKEN = load_tokens(src_id, token_path)
    token_reset = TOKEN.reset_index()
    token_reset["chunk_num"] = pd.cut(
        token_reset.index, n_chunks, labels=list(range(n_chunks))
    )
    chunks_list = (
        token_reset.groupby("chunk_num", observed=True)["term_str"]
        .apply(lambda x: " ".join(x.dropna()))
        .tolist()
    )
    if len(chunks_list) < 3:
        return None

    try:
        vec = TfidfVectorizer(lowercase=True, min_df=min_df, max_df=max_df,
                              strip_accents=None, norm=None)
        X = vec.fit_transform(chunks_list).toarray().astype(float)
    except ValueError:
        return None

    # Laplace smoothing → L1-normalize to probability distributions
    X += 1e-9
    P = X / X.sum(axis=1, keepdims=True)

    n    = len(P)
    dist = _make_dist_fn(distance)

    novelty    = np.full(n, np.nan)
    transience = np.full(n, np.nan)

    for j in range(n):
        back = [dist(P[j], P[j - d]) for d in range(1, window + 1) if j - d >= 0]
        fwd  = [dist(P[j], P[j + d]) for d in range(1, window + 1) if j + d < n]
        if back:
            novelty[j]    = float(np.mean(back))
        if fwd:
            transience[j] = float(np.mean(fwd))

    resonance = novelty - transience

    words = vec.get_feature_names_out()
    top_terms = []
    for j in range(n):
        idx = np.argsort(P[j])[::-1][:8]
        top_terms.append(", ".join(words[idx]))

    return {
        "novelty":    novelty,
        "transience": transience,
        "resonance":  resonance,
        "chunks_list": chunks_list,
        "top_terms":   top_terms,
        "n_chunks":    n,
    }


_CURVES_EXPANDER_TEXT = r"""
Each chunk is treated as a **probability distribution over vocabulary** — TF-IDF
weights (unnormalized, so IDF amplification of distinctive terms is preserved),
with a small Laplace smoothing constant ($\epsilon = 10^{-9}$) added before
L1-normalizing each chunk vector to sum to 1. This ensures the novelty signal
and the HAC clustering operate on the same underlying representation.

The three signals follow **Barron, Huang, Spang, and DeDeo (2018)**
([PNAS 115(18):4607–4612, doi:10.1073/pnas.1717729115](https://doi.org/10.1073/pnas.1717729115)),
equations 1–3, originally developed for speeches in the French Revolutionary debates:

**Novelty** $\mathcal{N}_w(j)$ — how much chunk $j$ diverges from the $w$ chunks
preceding it:

$$\mathcal{N}_w(j) = \frac{1}{w} \sum_{d=1}^{w} \mathrm{KLD}\!\left(s^{(j)} \,\|\, s^{(j-d)}\right)$$

where $\mathrm{KLD}(p \| q) = \sum_i p_i \log_2 (p_i / q_i)$ is the
Kullback–Leibler divergence (in bits).

**Transience** $\mathcal{T}_w(j)$ — the same quantity looking *forward*: how much
the $w$ chunks after $j$ diverge from $j$ itself.

**Resonance** $\mathcal{R}_w(j) = \mathcal{N}_w(j) - \mathcal{T}_w(j)$ — a chunk
is resonant when it is simultaneously novel relative to the past *and* its
novelty persists into the future. Pure novelty that immediately reverts (a
one-off lexical spike) will score high on Novelty but also high on Transience,
leaving Resonance near zero. A true narrative pivot — one that opens a new
episode — should peak on Resonance.

The **threshold** is set as a percentile of the chosen signal over all chunks
with sufficient context (chunks within $w$ of either end have partial windows
and receive `NaN`). Chunks at or above the threshold are flagged as candidate
episode boundaries and overlaid as dashed lines on the HAC heatmap below.
"""

@st.cache_data(show_spinner="Running HAC linkage…")
def run_linkage(src_id, token_path, n_chunks, min_df, max_df):
    TOKEN = load_tokens(src_id, token_path)
    token_reset = TOKEN.reset_index()
    token_reset["chunk_num"] = pd.cut(
        token_reset.index, n_chunks, labels=list(range(n_chunks))
    )
    chunks_list = (
        token_reset.groupby("chunk_num", observed=True)["term_str"]
        .apply(lambda x: " ".join(x.dropna()))
        .tolist()
    )
    if len(chunks_list) < 3:
        return None
    try:
        vec = TfidfVectorizer(lowercase=True, max_df=max_df, min_df=min_df,
                              strip_accents=None, norm="l2")
        X = vec.fit_transform(chunks_list)
    except ValueError:
        return None
    tfidf_dense = X.toarray()
    words = vec.get_feature_names_out()
    Z = linkage(pdist(tfidf_dense, metric="euclidean"), method="ward")
    return {"tfidf_dense": tfidf_dense, "words": words, "Z": Z,
            "n_chunks": len(chunks_list)}


# ── Controls ──────────────────────────────────────────────────────────────────
st.title("Episode Boundaries — Popol Wuj")
render_toc([
    ("Novelty / Transience / Resonance",          "curves"),
    ("HAC Cluster Membership with Boundaries",     "hac-heatmap"),
    ("Detected Boundaries",                        "boundaries"),
])

src_ids = list(SOURCES_META.keys())
_col_ratios = cfg["layout"]["column_ratios"]
cols = st.columns(_col_ratios[:4])

_c = cfg["controls"]
src_id  = cols[0].selectbox("Source", src_ids,
                             format_func=lambda s: SOURCES_META[s]["label"])
n_chunks = cols[1].number_input("Chunks", min_value=_c["n_chunks"]["min"],
                                 max_value=_c["n_chunks"]["max"],
                                 value=_c["n_chunks"]["default"], step=_c["n_chunks"]["step"])
min_df   = cols[2].number_input("min_df", min_value=_c["min_df"]["min"],
                                 max_value=_c["min_df"]["max"],
                                 value=_c["min_df"]["default"], step=_c["min_df"]["step"])
max_df   = cols[3].number_input("max_df", min_value=_c["max_df"]["min"],
                                 max_value=_c["max_df"]["max"],
                                 value=_c["max_df"]["default"], step=_c["max_df"]["step"],
                                 format="%.2f")

cols2 = st.columns(_col_ratios[:4])
window        = cols2[0].number_input("Window (w)", min_value=1, max_value=20, value=3, step=1)
distance      = cols2[1].selectbox("Distance", ["KLD", "JSD", "Cosine"])
signal_opt    = cols2[2].selectbox("Boundary signal", ["Resonance", "Novelty"])
threshold_pct = cols2[3].slider("Threshold (percentile)", min_value=50, max_value=99,
                                 value=85, step=1)

cols3 = st.columns(_col_ratios[:2])
k_clusters    = cols3[0].number_input("k (clusters)", min_value=_c["n_clusters"]["min"],
                                       max_value=_c["n_clusters"]["max"],
                                       value=_c["n_clusters"]["default"],
                                       step=_c["n_clusters"]["step"])

token_path = find_token_file(src_id)
if token_path is None:
    st.error(f"TOKEN file not found for source '{src_id}'.")
    st.stop()

result = compute_novelty(src_id, token_path, int(n_chunks), int(min_df),
                         float(max_df), int(window), distance)
if result is None:
    st.warning("Not enough data — try reducing min_df or increasing chunks.")
    st.stop()

novelty    = result["novelty"]
transience = result["transience"]
resonance  = result["resonance"]
chunks_list = result["chunks_list"]
top_terms   = result["top_terms"]
n           = result["n_chunks"]
xs          = list(range(n))

signal = resonance if signal_opt == "Resonance" else novelty
valid  = signal[~np.isnan(signal)]
thresh = float(np.nanpercentile(valid, threshold_pct))
boundary_mask = (~np.isnan(signal)) & (signal >= thresh)

# ── Section 1: Curves ─────────────────────────────────────────────────────────
st.divider()
st.subheader("Novelty / Transience / Resonance", anchor="curves")
st.caption(
    f"Window w={window}. Novelty = mean KLD looking back; "
    "Transience = mean KLD looking forward; Resonance = Novelty − Transience."
)
with st.expander("Method — Novelty, Transience, and Resonance"):
    st.markdown(_CURVES_EXPANDER_TEXT)

fig = go.Figure()
fig.add_trace(go.Scatter(x=xs, y=novelty,    name="Novelty",
                          line=dict(color="#636EFA", width=1.8)))
fig.add_trace(go.Scatter(x=xs, y=transience, name="Transience",
                          line=dict(color="#EF553B", width=1.8)))
fig.add_trace(go.Scatter(x=xs, y=resonance,  name="Resonance",
                          line=dict(color="#00CC96", width=2.2)))
fig.add_hline(y=thresh, line_dash="dot", line_color="rgba(0,0,0,0.45)",
              line_width=1.5,
              annotation_text=f"{threshold_pct}th pct ({thresh:.3f})",
              annotation_position="top right", annotation_font_size=9)
fig.update_layout(
    height=340,
    margin=dict(l=60, r=30, t=20, b=50),
    plot_bgcolor="white",
    legend=dict(orientation="h", y=1.08, x=0),
    xaxis=dict(title="Chunk (narrative order)", showgrid=True,
               gridcolor="#EEEEEE", zeroline=False),
    yaxis=dict(title=f"{distance} distance", showgrid=True,
               gridcolor="#EEEEEE", zeroline=False),
)
st.plotly_chart(fig, width="stretch")

boundary_chunks = [j for j in xs if boundary_mask[j]]

# ── Section 2: HAC heatmap segmented by resonance boundaries ─────────────────
st.divider()
st.subheader("HAC Cluster Membership with Episode Boundaries", anchor="hac-heatmap")
st.caption(
    f"Ward linkage on L2-normalized TF-IDF vectors · k={k_clusters} clusters. "
    "Dashed lines mark chunks where the boundary signal exceeds the threshold above."
)

hac = run_linkage(src_id, token_path, int(n_chunks), int(min_df), float(max_df))
if hac is None:
    st.warning("HAC couldn't run — try adjusting parameters.")
else:
    _Z           = hac["Z"]
    _n           = hac["n_chunks"]
    _thresh      = threshold_for_k(_Z, int(k_clusters), _n)
    _labels      = fcluster(_Z, _thresh, criterion="distance")
    _first_seen  = {c: int(np.argmax(_labels == c)) for c in np.unique(_labels)}
    _unique      = sorted(np.unique(_labels), key=lambda c: _first_seen[c])
    _n_clusters  = len(_unique)
    _palette     = px.colors.qualitative.Plotly
    _c2color     = {c: _palette[i % len(_palette)] for i, c in enumerate(_unique)}

    _cluster_table = make_cluster_table(hac["tfidf_dense"], hac["words"],
                                        _labels, _c["n_top_words"]["default"])
    _y_labels = [_cluster_table.loc[c, "top_terms"] for c in _unique]

    _z_heat = np.zeros((_n_clusters, _n), dtype=float)
    for _i, _c_lbl in enumerate(_unique):
        _z_heat[_i, _labels == _c_lbl] = _i + 1

    _N  = _n_clusters + 1
    _cs = [[0, "white"], [1 / _N, "white"]]
    for _i, _c_lbl in enumerate(_unique):
        _cs += [[(_i + 1) / _N, _c2color[_c_lbl]], [(_i + 2) / _N, _c2color[_c_lbl]]]

    _row_h = max(40, cfg["layout"]["heatmap_row_height_px"])
    fig_hac = go.Figure(go.Heatmap(
        z=_z_heat, x=list(range(_n)), y=_y_labels,
        colorscale=_cs, zmin=-0.5, zmax=_n_clusters + 0.5,
        showscale=False, xgap=0, ygap=2,
    ))
    fig_hac.update_layout(
        height=_n_clusters * _row_h + 80,
        margin=dict(l=250, r=20, t=20, b=60),
        plot_bgcolor="white",
        xaxis=dict(title="Chunk (narrative order)", showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
    )
    for _i, _c_lbl in enumerate(_unique):
        _pos  = np.where(_labels == _c_lbl)[0]
        _gaps = np.where(np.diff(_pos) > 1)[0] + 1
        for _run in np.split(_pos, _gaps):
            fig_hac.add_annotation(
                x=float(np.mean(_run)), y=_y_labels[_i],
                text=f"<b>{chr(65 + _i)}</b>",
                showarrow=False, xanchor="center", yanchor="middle",
                font=dict(size=13, color="white"),
            )
    for _bj in boundary_chunks:
        fig_hac.add_vline(
            x=_bj,
            line_dash="dot", line_color="rgba(0,0,0,0.55)", line_width=1.5,
            annotation_text=str(_bj),
            annotation_position="top",
            annotation_font_size=8,
        )
    st.plotly_chart(fig_hac, width="stretch")

# ── Section 3: Detected boundaries table ─────────────────────────────────────
st.divider()
st.subheader("Detected Boundaries", anchor="boundaries")
st.caption(
    f"Chunks where **{signal_opt}** ≥ {threshold_pct}th percentile "
    f"({thresh:.3f} bits). {len(boundary_chunks)} boundary chunk(s) detected."
)

if not boundary_chunks:
    st.info("No boundaries detected at this threshold — try lowering the percentile.")
else:
    _preview_len = cfg["visualization"].get("preview_len", 200)
    rows = []
    for j in boundary_chunks:
        prev_text = chunks_list[j - 1][:_preview_len] if j > 0 else ""
        curr_text = chunks_list[j][:_preview_len]
        rows.append({
            "Chunk": j,
            f"{signal_opt}": round(float(signal[j]), 4),
            "Novelty":    round(float(novelty[j]), 4)    if not np.isnan(novelty[j])    else None,
            "Transience": round(float(transience[j]), 4) if not np.isnan(transience[j]) else None,
            "Top terms":  top_terms[j],
            "Prev chunk (preview)": prev_text + ("…" if len(chunks_list[j - 1]) > _preview_len and j > 0 else ""),
            "This chunk (preview)": curr_text + ("…" if len(curr_text) == _preview_len else ""),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Chunk"), width="stretch")
