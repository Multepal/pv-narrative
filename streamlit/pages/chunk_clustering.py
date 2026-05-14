"""
Chunk Clustering — HAC clustering of TF-IDF chunk vectors.
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
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list, dendrogram as scipy_dendrogram
from toc import render_toc

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
LANG_LABELS  = cfg["languages"]

st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)


def find_token_file(src_id: str) -> str | None:
    candidates = [
        os.path.join(APP_DIR, f"../../notebooks/{src_id}/{src_id}-TOKEN.csv"),
    ]
    for p in candidates:
        norm = os.path.normpath(p)
        if os.path.exists(norm):
            return norm
    return None


@st.cache_data(show_spinner=False)
def load_tokens(src_id: str, token_path: str) -> pd.DataFrame:
    TOKEN = pd.read_csv(token_path)
    idx_offset = TOKEN.columns.to_list().index("token_str")
    ohco = TOKEN.columns.to_list()[:idx_offset]
    return TOKEN.set_index(ohco)


@st.cache_data(show_spinner=False)
def compute_vocab_stats(src_id, token_path, chunk_size, overlap_int):
    """Return (n_chunks, suggested_min_df, suggested_max_df) from a vocab frequency scan."""
    TOKEN = load_tokens(src_id, token_path)
    tokens = TOKEN["term_str"].dropna().to_list()
    token_arr = np.array(tokens)
    step = max(1, chunk_size - overlap_int)
    if len(token_arr) < chunk_size:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(token_arr, chunk_size)[::step]
    chunks = [" ".join(row) for row in windows]
    n_chunks = len(chunks)
    if n_chunks < 2:
        return None
    vec = TfidfVectorizer(lowercase=True, max_df=1.0, min_df=1, strip_accents=None)
    try:
        X = vec.fit_transform(chunks)
    except ValueError:
        return None
    doc_freqs = np.asarray((X > 0).sum(axis=0)).flatten() / n_chunks
    min_df_sug = max(2, round(0.02 * n_chunks))
    high = np.sort(doc_freqs[doc_freqs >= 0.05])[::-1]
    if len(high) >= 3:
        x = np.linspace(0, 1, len(high))
        y_range = high[0] - high[-1]
        y_norm = (high - high[-1]) / y_range if y_range > 1e-10 else np.ones(len(high))
        line = np.array([1.0, -1.0])
        vecs = np.column_stack([x, y_norm]) - np.array([0.0, 1.0])
        line3 = np.append(line / np.linalg.norm(line), 0)
        vecs3 = np.hstack([vecs, np.zeros((len(vecs), 1))])
        dists = np.abs(np.cross(line3, vecs3)[:, 2])
        max_df_sug = float(np.clip(round(high[np.argmax(dists)], 2), 0.05, 0.95))
    else:
        max_df_sug = 0.35
    return n_chunks, min_df_sug, max_df_sug


@st.cache_data(show_spinner="Running linkage…")
def run_linkage(src_id, token_path, chunk_size, overlap_int, min_df, max_df, ngram_range=(1, 1)):
    """Build TF-IDF, cosine similarity, and Ward linkage. Clustering is applied later."""
    TOKEN = load_tokens(src_id, token_path)
    tokens = TOKEN["term_str"].dropna().to_list()
    step = max(1, chunk_size - overlap_int)
    token_arr = np.array(tokens)
    if len(token_arr) < chunk_size:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(token_arr, chunk_size)[::step]
    chunks_s = pd.Series(
        np.apply_along_axis(lambda row: " ".join(row), axis=1, arr=windows)
    ).loc[lambda s: s.str.split().str.len() >= 50]
    chunks_list = chunks_s.tolist()
    n_chunks = len(chunks_list)
    if n_chunks < 3:
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
        'n_chunks': n_chunks,
    }


def make_cluster_table(tfidf_dense, words, labels, n_top_words):
    """Top terms per cluster by mean TF-IDF weight."""
    tfidf_df = pd.DataFrame(tfidf_dense, columns=words)
    cluster_s = pd.Series(labels, name='cluster')
    cluster_tfidf = tfidf_df.groupby(cluster_s).mean()
    top_terms = cluster_tfidf.apply(
        lambda row: ", ".join(row.sort_values(ascending=False).head(n_top_words).index), axis=1
    )
    n_per = cluster_s.value_counts().sort_index().rename('n_chunks')
    return pd.DataFrame({'n_chunks': n_per, 'top_terms': top_terms})


def threshold_for_k(Z, k, n):
    """Return a threshold value that yields exactly k clusters via distance criterion."""
    heights = Z[:, 2]  # already sorted ascending by scipy
    k = max(2, min(k, n - 1))
    idx_low  = n - k - 1
    idx_high = n - k
    return float((heights[idx_low] + heights[idx_high]) / 2)


def build_dendrogram_figure(Z, labels, n, cluster_to_color, chunk_labels):
    """Plotly dendrogram with branch colors matching cluster_to_color assignments."""
    gray = '#CCCCCC'
    node_clusters: dict = {}

    def leaf_clusters(node):
        if node in node_clusters:
            return node_clusters[node]
        if node < n:
            result = frozenset([labels[node]])
        else:
            i = int(node - n)
            left, right = int(Z[i, 0]), int(Z[i, 1])
            result = leaf_clusters(left) | leaf_clusters(right)
        node_clusters[node] = result
        return result

    # Precompute all internal nodes so link_color_func has no recursion overhead
    for i in range(len(Z)):
        leaf_clusters(n + i)

    def link_color_func(k):
        cs = leaf_clusters(k)
        if len(cs) == 1:
            return cluster_to_color[next(iter(cs))]
        return gray

    dend = scipy_dendrogram(Z, no_plot=True, link_color_func=link_color_func)

    # For each cluster, find the x of the leftmost vertical line of its root merge.
    # Strategy: map Z heights → Z index, then match dend icoord entries to find the root merge.
    _height_to_z = {float(Z[i, 2]): i for i in range(len(Z))}
    cluster_root_height: dict = {}
    for i in range(len(Z)):
        cs = node_clusters[n + i]
        if len(cs) == 1:
            c = next(iter(cs))
            h = float(Z[i, 2])
            if c not in cluster_root_height or h > cluster_root_height[c]:
                cluster_root_height[c] = h

    cluster_root_x: dict = {}
    for xs, ys in zip(dend['icoord'], dend['dcoord']):
        h = float(ys[1])
        z_idx = _height_to_z.get(h)
        if z_idx is None:
            continue
        cs = node_clusters.get(n + z_idx, frozenset())
        if len(cs) == 1:
            c = next(iter(cs))
            if cluster_root_height.get(c) == h:
                cluster_root_x[c] = (float(xs[1]) + float(xs[2])) / 2  # centre of root merge = x of vertical above threshold

    fig = go.Figure()
    for xs, ys, color in zip(dend['icoord'], dend['dcoord'], dend['color_list']):
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color=color, width=2.5),
            showlegend=False, hoverinfo='none',
        ))

    # Leaves sit at x = 5, 15, 25, … in scipy's coordinate system
    leaf_x    = [10 * i + 5 for i in range(n)]
    tick_text = [chunk_labels[orig] for orig in dend['leaves']]
    fig.update_layout(
        xaxis=dict(tickvals=leaf_x, ticktext=tick_text,
                   range=[-5, 10 * n + 5],
                   showline=False, showgrid=False, zeroline=False),
        yaxis=dict(showline=False, showgrid=False, zeroline=False),
        plot_bgcolor='white',
    )
    return fig, dend['leaves'], cluster_root_x


# ── Controls ──────────────────────────────────────────────────────────────────
st.title("Chunk Clustering — Popol Wuj")
render_toc([
    ("Merge Height Scree Plot", "scree-plot"),
    ("Chunk Dendrogram",        "dendrogram"),
    ("Cluster Membership",      "cluster-membership"),
    ("Cluster Word Clouds",     "word-clouds"),
])

src_ids = list(SOURCES_META.keys())
_col_ratios = cfg["layout"]["column_ratios"]
cols = st.columns(_col_ratios[:4])  # source | chunk%+overlap | min_df+max_df | ngram min+max

_c = cfg["controls"]
src_id = cols[0].selectbox(
    "Source", src_ids, index=src_ids.index(_c["default_source"]),
    format_func=lambda x: f"{SOURCES_META[x]['label']} ({LANG_LABELS[SOURCES_META[x]['lang']]})")

_early_token_path = find_token_file(src_id)
_n_tokens_early = len(load_tokens(src_id, _early_token_path)) if _early_token_path else None

_cp = _c["chunk_pct"]
chunk_pct  = cols[1].number_input("Chunk %", _cp["min"], _cp["max"], _cp["default"], step=_cp["step"], format="%.3f")
chunk_size = max(50, int(chunk_pct * _n_tokens_early)) if _n_tokens_early else 100
overlap    = cols[1].number_input("Overlap", _c["overlap"]["min"], _c["overlap"]["max"], _c["overlap"]["default"], step=_c["overlap"]["step"], format="%.2f")
overlap_int = int(overlap * chunk_size)

_vstats = compute_vocab_stats(src_id, _early_token_path, chunk_size, overlap_int) if _early_token_path else None

min_df = cols[2].number_input("min_df", _c["min_df"]["min"], _c["min_df"]["max"], _c["min_df"]["default"], step=_c["min_df"]["step"])
max_df = cols[2].number_input("max_df", _c["max_df"]["min"], _c["max_df"]["max"], _c["max_df"]["default"], step=_c["max_df"]["step"], format="%.2f")

_ng = _c["ngram_range"]
ngram_min = cols[3].number_input("ngram min", _ng["min_n"], _ng["max_n"], _ng["default_min"], step=1)
ngram_max = cols[3].number_input("ngram max", _ng["min_n"], _ng["max_n"], _ng["default_max"], step=1)
ngram_max = max(ngram_max, ngram_min)

n_top_words = _c["n_top_words"]["default"]

_info_parts = []
if _n_tokens_early:
    _info_parts += [f"chunk = {int(chunk_pct * _n_tokens_early):,} tokens",
                    f"overlap = {int(overlap * chunk_size):,} tokens"]
if _vstats:
    _info_parts += [f"min_df → {_vstats[1]} (2% of {_vstats[0]} chunks)",
                    f"max_df → {_vstats[2]:.2f} (vocab knee)"]
if _info_parts:
    st.caption(" · ".join(_info_parts))

st.divider()

# ── Token file resolution ─────────────────────────────────────────────────────
token_path = find_token_file(src_id)
if token_path is None:
    st.warning(f"Token file not found for `{src_id}`.")
    st.stop()

# ── Run linkage ───────────────────────────────────────────────────────────────
result = run_linkage(src_id, token_path, chunk_size, overlap_int, min_df, max_df, (ngram_min, ngram_max))

if result is None:
    st.warning("Clustering couldn't run — try adjusting chunk size, min_df, or max_df.")
    st.stop()

n_chunks    = result['n_chunks']
tfidf_dense = result['tfidf_dense']
words       = result['words']
Z           = result['Z']
chunks_list = result['chunks_list']
meta        = SOURCES_META[src_id]

_linkage_key = f"{src_id}_{chunk_size}_{overlap_int}_{min_df}_{max_df}_{ngram_min}_{ngram_max}"
_z_max = float(Z[:, 2].max())

# ── Session state: reset k when linkage params change ────────────────────────
if st.session_state.get("_cluster_linkage_key") != _linkage_key:
    st.session_state["_cluster_linkage_key"] = _linkage_key
    st.session_state["cluster_k"] = cfg["controls"]["n_clusters"]["default"]
_selected_k = st.session_state.get("cluster_k", cfg["controls"]["n_clusters"]["default"])

# ── Section 1: Scree plot (click to select k) ────────────────────────────────
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
_scree_event = st.plotly_chart(fig_scree, width='stretch', key=f"scree_{_linkage_key}", on_select="rerun")
if _scree_event.selection.points:
    _clicked_k = int(_scree_event.selection.points[0]["x"])
    if _clicked_k != st.session_state.get("cluster_k"):
        st.session_state["cluster_k"] = _clicked_k
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

# Build dendrogram in default orientation, then rotate 90° CCW by swapping x↔y
_show_labels = n_chunks <= 60
_chunk_labels = [str(i) for i in range(n_chunks)] if _show_labels else [""] * n_chunks
fig_dend, _dend_leaves, _root_x = build_dendrogram_figure(Z, labels, n_chunks, cluster_to_color, _chunk_labels)

fig_rot = go.Figure()
for _tr in fig_dend.data:
    fig_rot.add_trace(go.Scatter(
        x=list(_tr.y), y=list(_tr.x),  # swap: height → x, leaf position → y
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
st.plotly_chart(fig_rot, width='stretch', key=f"dend_{_key_sfx}")
st.caption(
    f"**{meta['label']}** ({LANG_LABELS[meta['lang']]}) · "
    f"{_n_tokens_early:,} tokens · {n_chunks} chunks · "
    f"**{n_clusters} clusters** at threshold {threshold:.4f} · "
    f"chunk = {chunk_pct * 100:.1f}% ({chunk_size:,} tokens)"
)

# ── Section 2: Cluster membership ────────────────────────────────────────────
cluster_table = make_cluster_table(tfidf_dense, words, labels, n_top_words)

st.divider()
st.subheader("Cluster Membership in Narrative Order", anchor="cluster-membership")
st.caption(
    "Rows = clusters labeled by top terms, columns = chunks in sequential narrative order. "
    "Colors match the dendrogram branches."
)
y_labels = [cluster_table.loc[c, 'top_terms'] for c in unique_labels]

# z[i, j] = i+1 if chunk j is in cluster unique_labels[i], else 0
z = np.zeros((n_clusters, n_chunks), dtype=float)
for i, c in enumerate(unique_labels):
    z[i, labels == c] = i + 1

# Stepped colorscale built from cluster_to_color — same mapping as the dendrogram
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
st.plotly_chart(fig_seq, width='stretch', key=f"seq_{_key_sfx}")

# ── Section 4: Cluster word clouds ───────────────────────────────────────────
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

# ── Methods note ──────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
**Pipeline summary.**

The clustering is derived from the selected edition's token file (`TOKEN.csv`), which records
every token in the text in narrative order with a normalized term string (`term_str`).

A sliding window of **chunk_size** tokens is stepped across that sequence with stride
`chunk_size − overlap`, producing a set of overlapping text chunks; windows shorter than
50 tokens are discarded.

Each chunk is converted to an L2-normalized TF-IDF vector by `TfidfVectorizer`
(vocabulary filtered by `min_df` and `max_df`), yielding a matrix of shape
*(n_chunks × vocab_size)* in which every row is a unit vector in term space.

Pairwise Euclidean distances between those unit vectors are computed and passed to
**Ward's hierarchical agglomerative clustering**, which at each step merges the pair of
clusters whose union minimizes the increase in total within-cluster variance.

The result is the dendrogram above.

Cutting the dendrogram at the chosen threshold assigns each chunk to exactly one cluster;
the membership chart shows those assignments in narrative order.
"""
)
