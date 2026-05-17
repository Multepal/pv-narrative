"""
Chunk Clustering (n-chunks) — HAC clustering using pd.cut for exact chunk counts.
"""

import os
import yaml
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from toc import render_toc
from utils import find_token_file, load_tokens, threshold_for_k, make_cluster_table, build_dendrogram_figure, load_boundaries, add_boundary_vlines

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
LANG_LABELS  = cfg["languages"]

st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Running linkage…")
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
        vec = TfidfVectorizer(lowercase=True, max_df=max_df, min_df=min_df,
                              strip_accents=None, norm="l2", ngram_range=ngram_range)
        X = vec.fit_transform(chunks_list)
    except ValueError:
        return None
    tfidf_dense = X.toarray()
    words = vec.get_feature_names_out()
    dist_condensed = pdist(tfidf_dense, metric='euclidean')
    Z = linkage(dist_condensed, method='ward')
    return {
        'tfidf_dense': tfidf_dense,
        'words': words,
        'Z': Z,
        'chunks_list': chunks_list,
        'n_chunks': len(chunks_list),
    }


# ── Controls ──────────────────────────────────────────────────────────────────
st.title("Chunk Clustering (n-chunks) — Popol Wuj")
render_toc([
    ("Merge Height Scree Plot", "scree-plot"),
    ("Chunk Dendrogram",        "dendrogram"),
    ("Cluster Membership",      "cluster-membership"),
    ("Cluster Word Clouds",     "word-clouds"),
])

src_ids = list(SOURCES_META.keys())
_col_ratios = cfg["layout"]["column_ratios"]
cols = st.columns(_col_ratios[:4])  # source | n_chunks | min_df+max_df | ngram min+max

_c = cfg["controls"]
src_id = cols[0].selectbox(
    "Source", src_ids, index=src_ids.index(_c["default_source"]),
    format_func=lambda x: f"{SOURCES_META[x]['label']} ({LANG_LABELS[SOURCES_META[x]['lang']]})")

_early_token_path = find_token_file(src_id)
_n_tokens_early = len(load_tokens(src_id, _early_token_path)) if _early_token_path else None

n_chunks = cols[1].number_input("n_chunks", min_value=_c["n_chunks"]["min"], max_value=_c["n_chunks"]["max"], value=_c["n_chunks"]["default"], step=_c["n_chunks"]["step"])

min_df = cols[2].number_input("min_df", _c["min_df"]["min"], _c["min_df"]["max"], _c["min_df"]["default"], step=_c["min_df"]["step"])
max_df = cols[2].number_input("max_df", _c["max_df"]["min"], _c["max_df"]["max"], _c["max_df"]["default"], step=_c["max_df"]["step"], format="%.2f")

_ng = _c["ngram_range"]
ngram_min = cols[3].number_input("ngram min", _ng["min_n"], _ng["max_n"], _ng["default_min"], step=1)
ngram_max = cols[3].number_input("ngram max", _ng["min_n"], _ng["max_n"], _ng["default_max"], step=1)
ngram_max = max(ngram_max, ngram_min)

n_top_words = _c["n_top_words"]["default"]

if _n_tokens_early:
    st.caption(f"n_chunks = {n_chunks} · approx. {int(_n_tokens_early / n_chunks):,} tokens per chunk")

st.divider()

# ── Token file resolution ─────────────────────────────────────────────────────
token_path = find_token_file(src_id)
if token_path is None:
    st.warning(f"Token file not found for `{src_id}`.")
    st.stop()

# ── Run linkage ───────────────────────────────────────────────────────────────
result = run_linkage(src_id, token_path, n_chunks, min_df, max_df, (ngram_min, ngram_max))

if result is None:
    st.warning("Clustering couldn't run — try adjusting n_chunks, min_df, or max_df.")
    st.stop()

n_chunks    = result['n_chunks']
tfidf_dense = result['tfidf_dense']
words       = result['words']
Z           = result['Z']
chunks_list = result['chunks_list']
meta        = SOURCES_META[src_id]

_linkage_key = f"{src_id}_{n_chunks}_{min_df}_{max_df}_{ngram_min}_{ngram_max}"
_z_max = float(Z[:, 2].max())

# ── Session state: reset k when linkage params change ────────────────────────
if st.session_state.get("_cluster_nc_linkage_key") != _linkage_key:
    st.session_state["_cluster_nc_linkage_key"] = _linkage_key
    st.session_state["cluster_nc_k"] = cfg["controls"]["n_clusters"]["default"]
_selected_k = st.session_state.get("cluster_nc_k", cfg["controls"]["n_clusters"]["default"])

# ── Section 1: Scree plot ─────────────────────────────────────────────────────
st.subheader("Merge Height Scree Plot", anchor="scree-plot")
st.caption(
    "Ward merge distance required to reduce k clusters to k−1. "
    "Look for the elbow where the line flattens. **Click a point to select k.**"
)

_max_k_scree = min(cfg["controls"]["n_clusters"]["max"], n_chunks - 1)
_k_vals = list(range(2, _max_k_scree + 1))
_heights = [float(Z[n_chunks - k, 2]) for k in _k_vals]

fig_scree = go.Figure(go.Scatter(
    x=_k_vals, y=_heights,
    mode="lines+markers",
    line=dict(color="#636EFA"),
    marker=dict(size=7),
    showlegend=False,
))
fig_scree.add_vline(
    x=_selected_k, line_dash="dash", line_color="crimson", line_width=1.5,
    annotation_text=f"k = {_selected_k}", annotation_position="top right",
)
fig_scree.update_layout(
    height=320,
    margin=dict(l=60, r=30, t=20, b=50),
    plot_bgcolor="white",
    xaxis=dict(title="Number of clusters (k)", dtick=1, showgrid=False, zeroline=False),
    yaxis=dict(title="Ward merge distance", showgrid=True, gridcolor="#EEEEEE", zeroline=False),
)
_scree_event = st.plotly_chart(fig_scree, width='stretch', key=f"nc_scree_{_linkage_key}", on_select="rerun")
if _scree_event.selection.points:
    _clicked_k = int(_scree_event.selection.points[0]["x"])
    if _clicked_k != st.session_state.get("cluster_nc_k"):
        st.session_state["cluster_nc_k"] = _clicked_k
        st.rerun()
st.success(f"k = {_selected_k} selected — dendrogram below ↓")

# ── Section 2: Dendrogram ─────────────────────────────────────────────────────
threshold  = threshold_for_k(Z, _selected_k, n_chunks)
labels     = fcluster(Z, threshold, criterion='distance')
n_clusters = int(len(np.unique(labels)))

_first_seen      = {c: int(np.argmax(labels == c)) for c in np.unique(labels)}
unique_labels    = sorted(np.unique(labels), key=lambda c: _first_seen[c])
_palette         = px.colors.qualitative.Plotly
cluster_to_color = {c: _palette[i % len(_palette)] for i, c in enumerate(unique_labels)}

_key_sfx = f"{_linkage_key}_{_selected_k}"

st.divider()
st.subheader("Chunk Dendrogram", anchor="dendrogram")
st.caption("Ward linkage on Euclidean distances between L2-normalized TF-IDF chunk vectors. Dashed line = cut threshold.")

_show_labels = n_chunks <= 60
_chunk_labels = [str(i) for i in range(n_chunks)] if _show_labels else [""] * n_chunks
fig_dend, _dend_leaves, _root_x = build_dendrogram_figure(Z, labels, n_chunks, cluster_to_color, _chunk_labels)

fig_rot = go.Figure()
for _tr in fig_dend.data:
    fig_rot.add_trace(go.Scatter(
        x=list(_tr.y), y=list(_tr.x),
        mode='lines',
        line=_tr.line,
        showlegend=False, hoverinfo='none',
    ))

fig_rot.add_vline(x=threshold, line_dash="dash", line_color="crimson", line_width=1.5)

for _i, _c in enumerate(unique_labels):
    _ly = [10 * _j + 5 for _j, _orig in enumerate(_dend_leaves) if labels[_orig] == _c]
    _y  = _root_x.get(_c, float(np.mean(_ly)) if _ly else 0)
    fig_rot.add_annotation(
        x=threshold, y=_y,
        xshift=6, yshift=8,
        text=f"<b>{chr(65 + _i)}</b>",
        showarrow=False, xanchor='left', yanchor='bottom',
        font=dict(size=14, color='black'),
    )

_dx = fig_dend.layout.xaxis
fig_rot.update_layout(
    height=500,
    margin=dict(l=50, r=30, t=10, b=50),
    plot_bgcolor='white',
    xaxis=dict(
        range=[0, _z_max * 1.12],
        showline=False, showgrid=False, zeroline=False,
    ),
    yaxis=dict(
        tickvals=list(_dx.tickvals) if _dx.tickvals is not None else [],
        ticktext=list(_dx.ticktext) if _dx.ticktext is not None else [],
        range=list(_dx.range) if _dx.range is not None else [-5, 10 * n_chunks + 5],
        showline=False, showgrid=False, zeroline=False,
    ),
)
st.plotly_chart(fig_rot, width='stretch', key=f"nc_dend_{_key_sfx}")
st.caption(
    f"**{meta['label']}** ({LANG_LABELS[meta['lang']]}) · "
    f"{_n_tokens_early:,} tokens · {n_chunks} chunks · "
    f"**{n_clusters} clusters** at threshold {threshold:.4f} · "
    f"approx. {int(_n_tokens_early / n_chunks):,} tokens per chunk"
)

# ── Section 3: Cluster membership ─────────────────────────────────────────────
cluster_table = make_cluster_table(tfidf_dense, words, labels, n_top_words)

st.divider()
st.subheader("Cluster Membership in Narrative Order", anchor="cluster-membership")
st.caption(
    "Rows = clusters labeled by top terms, columns = chunks in sequential narrative order. "
    "Colors match the dendrogram branches."
)
_show_bounds = st.checkbox("Show episode boundaries", value=False, key=f"nc_bounds_{_key_sfx}")
_boundaries  = load_boundaries(APP_DIR) if _show_bounds else []

y_labels = [cluster_table.loc[c, 'top_terms'] for c in unique_labels]

z = np.zeros((n_clusters, n_chunks), dtype=float)
for i, c in enumerate(unique_labels):
    z[i, labels == c] = i + 1

_N = n_clusters + 1
_colorscale = [[0, 'white'], [1 / _N, 'white']]
for i, c in enumerate(unique_labels):
    _colorscale += [[(i + 1) / _N, cluster_to_color[c]], [(i + 2) / _N, cluster_to_color[c]]]

_row_h = max(40, cfg["layout"]["heatmap_row_height_px"])
fig_seq = go.Figure(go.Heatmap(
    z=z,
    x=list(range(n_chunks)),
    y=y_labels,
    colorscale=_colorscale,
    zmin=-0.5,
    zmax=n_clusters + 0.5,
    showscale=False,
    xgap=0,
    ygap=2,
))
fig_seq.update_layout(
    height=n_clusters * _row_h + 80,
    margin=dict(l=250, r=20, t=20, b=60),
    plot_bgcolor='white',
    xaxis=dict(title="Chunk (narrative order)", showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False),
)
for _i, _c in enumerate(unique_labels):
    _pos = np.where(labels == _c)[0]
    if len(_pos) == 0:
        continue
    _gaps = np.where(np.diff(_pos) > 1)[0] + 1
    for _run in np.split(_pos, _gaps):
        fig_seq.add_annotation(
            x=float(np.mean(_run)), y=y_labels[_i],
            text=f"<b>{chr(65 + _i)}</b>",
            showarrow=False, xanchor='center', yanchor='middle',
            font=dict(size=13, color='white'),
        )
if _boundaries:
    add_boundary_vlines(fig_seq, _boundaries, n_chunks)
st.plotly_chart(fig_seq, width='stretch', key=f"nc_seq_{_key_sfx}")

_lbl_to_alpha = {c: chr(65 + i) for i, c in enumerate(unique_labels)}
_export_df = pd.DataFrame({
    "chunk":      list(range(n_chunks)),
    "cluster":    [_lbl_to_alpha[l] for l in labels],
    "top_terms":  [cluster_table.loc[l, "top_terms"] for l in labels],
})
st.download_button(
    "↓ Download cluster assignments (CSV)",
    data=_export_df.to_csv(index=False),
    file_name=f"clusters_{src_id}_{n_chunks}_k{_selected_k}.csv",
    mime="text/csv",
)

# ── Section 4: Word clouds ─────────────────────────────────────────────────────
st.divider()
st.subheader("Cluster Word Clouds", anchor="word-clouds")

_v = cfg["visualization"]
_wc_means = (
    pd.DataFrame(tfidf_dense, columns=words)
    .groupby(pd.Series(labels, name='cluster'))
    .mean()
)
_n_wc_cols = min(_v["wordcloud_cols"], n_clusters)
for _row_start in range(0, n_clusters, _n_wc_cols):
    _row_idxs = range(_row_start, min(_row_start + _n_wc_cols, n_clusters))
    _grid_cols = st.columns(_n_wc_cols)
    for _col_idx, _lbl_idx in enumerate(_row_idxs):
        _clust = unique_labels[_lbl_idx]
        _wc = WordCloud(
            width=_v["wordcloud_width"], height=_v["wordcloud_height"],
            background_color="white",
            colormap=_v["wordcloud_colormap"],
            prefer_horizontal=_v["wordcloud_prefer_horizontal"],
        ).generate_from_frequencies(_wc_means.loc[_clust].to_dict())
        _grid_cols[_col_idx].image(
            _wc.to_array(),
            caption=f"**{chr(65 + _lbl_idx)}** · {cluster_table.loc[_clust, 'top_terms']}",
            width='stretch',
        )
