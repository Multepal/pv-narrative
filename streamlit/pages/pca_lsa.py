"""
PCA / LSA Analysis — Latent Semantic Analysis with HAC clustering on component scores.
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
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram as scipy_dendrogram

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


def _make_chunks(src_id, token_path, n_chunks):
    TOKEN = load_tokens(src_id, token_path)
    token_reset = TOKEN.reset_index()
    token_reset["chunk_num"] = pd.cut(
        token_reset.index, n_chunks, labels=list(range(n_chunks))
    )
    chunks_s = (
        token_reset.groupby("chunk_num", observed=True)["term_str"]
        .apply(lambda x: " ".join(x.dropna()))
    )
    return chunks_s.tolist()


@st.cache_data(show_spinner="Running LSA…")
def run_lsa(src_id, token_path, n_chunks, min_df, max_df, n_components, ngram_range=(1, 1)):
    chunks_list = _make_chunks(src_id, token_path, n_chunks)
    if len(chunks_list) < 3:
        return None
    try:
        vec = TfidfVectorizer(lowercase=True, max_df=max_df, min_df=min_df,
                              strip_accents=None, norm="l2", ngram_range=ngram_range)
        X = vec.fit_transform(chunks_list)
    except ValueError:
        return None
    n_components = min(n_components, X.shape[0] - 1, X.shape[1] - 1)
    if n_components < 2:
        return None
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    THETA = svd.fit_transform(X)
    return {
        'THETA': THETA,
        'components': svd.components_,
        'words': vec.get_feature_names_out(),
        'explained': svd.explained_variance_ratio_,
        'n_chunks': len(chunks_list),
        'n_components': n_components,
        'chunks_list': chunks_list,
    }


def threshold_for_k(Z, k, n):
    k = max(2, min(k, n - 1))
    return float((Z[n - k - 1, 2] + Z[n - k, 2]) / 2)


def build_dendrogram_figure(Z, labels, n, cluster_to_color, chunk_labels):
    gray = '#CCCCCC'
    node_clusters: dict = {}

    def leaf_clusters(node):
        if node in node_clusters:
            return node_clusters[node]
        if node < n:
            result = frozenset([labels[node]])
        else:
            i = int(node - n)
            result = leaf_clusters(int(Z[i, 0])) | leaf_clusters(int(Z[i, 1]))
        node_clusters[node] = result
        return result

    for i in range(len(Z)):
        leaf_clusters(n + i)

    def link_color_func(k):
        cs = leaf_clusters(k)
        return cluster_to_color[next(iter(cs))] if len(cs) == 1 else gray

    dend = scipy_dendrogram(Z, no_plot=True, link_color_func=link_color_func)

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
                cluster_root_x[c] = (float(xs[1]) + float(xs[2])) / 2

    fig = go.Figure()
    for xs, ys, color in zip(dend['icoord'], dend['dcoord'], dend['color_list']):
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color=color, width=2.5),
            showlegend=False, hoverinfo='none',
        ))

    leaf_x = [10 * i + 5 for i in range(n)]
    fig.update_layout(
        xaxis=dict(tickvals=leaf_x, ticktext=[chunk_labels[o] for o in dend['leaves']],
                   range=[-5, 10 * n + 5],
                   showline=False, showgrid=False, zeroline=False),
        yaxis=dict(showline=False, showgrid=False, zeroline=False),
        plot_bgcolor='white',
    )
    return fig, dend['leaves'], cluster_root_x


# ── Controls ──────────────────────────────────────────────────────────────────
st.title("PCA / LSA Analysis — Popol Wuj")
st.caption(
    "LSA = TF-IDF + truncated SVD. "
    "Unlike NMF, components are **bipolar**: each has a positive pole (dominant terms) "
    "and a negative pole (contrasting terms). HAC runs on chunk PCA score vectors."
)

src_ids = list(SOURCES_META.keys())
_col_ratios = cfg["layout"]["column_ratios"]
cols = st.columns(_col_ratios[:4])

_c = cfg["controls"]
src_id = cols[0].selectbox(
    "Source", src_ids, index=src_ids.index(_c["default_source"]),
    format_func=lambda x: f"{SOURCES_META[x]['label']} ({LANG_LABELS[SOURCES_META[x]['lang']]})")

_early_token_path = find_token_file(src_id)
_n_tokens_early = len(load_tokens(src_id, _early_token_path)) if _early_token_path else None

n_chunks  = cols[1].number_input("n_chunks", min_value=5, max_value=200, value=20, step=1)
min_df    = cols[2].number_input("min_df", _c["min_df"]["min"], _c["min_df"]["max"], _c["min_df"]["default"], step=_c["min_df"]["step"])
max_df    = cols[2].number_input("max_df", _c["max_df"]["min"], _c["max_df"]["max"], _c["max_df"]["default"], step=_c["max_df"]["step"], format="%.2f")
_ng       = _c["ngram_range"]
ngram_min = cols[3].number_input("ngram min", _ng["min_n"], _ng["max_n"], _ng["default_min"], step=1)
ngram_max = cols[3].number_input("ngram max", _ng["min_n"], _ng["max_n"], _ng["default_max"], step=1)
ngram_max = max(ngram_max, ngram_min)

if _n_tokens_early:
    st.caption(f"n_chunks = {n_chunks} · approx. {int(_n_tokens_early / n_chunks):,} tokens per chunk")

st.divider()

token_path = find_token_file(src_id)
if token_path is None:
    st.warning(f"Token file not found for `{src_id}`.")
    st.stop()

_max_components = min(_c["n_topics"]["max"], n_chunks - 1)
result = run_lsa(src_id, token_path, n_chunks, min_df, max_df, _max_components, (ngram_min, ngram_max))

if result is None:
    st.warning("LSA couldn't run — try adjusting n_chunks, min_df, or max_df.")
    st.stop()

_lsa_key = f"{src_id}_{n_chunks}_{min_df}_{max_df}_{ngram_min}_{ngram_max}"

if st.session_state.get("_lsa_key") != _lsa_key:
    st.session_state["_lsa_key"] = _lsa_key
    st.session_state["lsa_n_components"] = min(_c["n_topics"]["default"], result['n_components'])
    st.session_state["lsa_k"] = _c["n_clusters"]["default"]

_selected_n = st.session_state.get("lsa_n_components", _c["n_topics"]["default"])
_selected_k = st.session_state.get("lsa_k", _c["n_clusters"]["default"])
_selected_n = min(_selected_n, result['n_components'])

# ── Data prep ─────────────────────────────────────────────────────────────────
THETA      = result['THETA'][:, :_selected_n]
components = result['components'][:_selected_n]
words      = result['words']
n_actual   = result['n_chunks']
explained  = result['explained']
_cum_var   = np.cumsum(explained)

# ── Section 1: HAC dendrogram on PCA scores ────────────────────────────────────
_dist = pdist(THETA, metric='euclidean')
_Z    = linkage(_dist, method='ward')
_z_max = float(_Z[:, 2].max())

st.subheader("HAC Dendrogram (Ward on PCA Score Vectors)")
st.caption(
    "Hierarchical clustering of chunks using their PCA scores as features. "
    "Click a point to select k. Use the scree plot at the bottom to change n_components."
)

_max_k = min(cfg["controls"]["n_clusters"]["max"], n_actual - 1)
_k_vals  = list(range(2, _max_k + 1))
_heights = [float(_Z[n_actual - k, 2]) for k in _k_vals]

fig_hac = go.Figure(go.Scatter(
    x=_k_vals, y=_heights,
    mode="lines+markers", line=dict(color="#636EFA"), marker=dict(size=7),
    showlegend=False,
))
fig_hac.add_vline(
    x=_selected_k, line_dash="dash", line_color="crimson", line_width=1.5,
    annotation_text=f"k = {_selected_k}", annotation_position="top right",
)
fig_hac.update_layout(
    height=260,
    margin=dict(l=60, r=30, t=10, b=50),
    plot_bgcolor="white",
    xaxis=dict(title="Number of clusters (k)", dtick=1, showgrid=False, zeroline=False),
    yaxis=dict(title="Ward merge distance", showgrid=True, gridcolor="#EEEEEE", zeroline=False),
)
_hac_evt = st.plotly_chart(fig_hac, width='stretch', key=f"lsa_hac_{_lsa_key}_{_selected_n}", on_select="rerun")
if _hac_evt.selection.points:
    _clicked_k = int(_hac_evt.selection.points[0]["x"])
    if _clicked_k != st.session_state.get("lsa_k"):
        st.session_state["lsa_k"] = _clicked_k
        st.rerun()

_threshold    = threshold_for_k(_Z, _selected_k, n_actual)
_labels       = fcluster(_Z, _threshold, criterion='distance')
_n_clusters   = int(len(np.unique(_labels)))
_first_seen   = {c: int(np.argmax(_labels == c)) for c in np.unique(_labels)}
_unique       = sorted(np.unique(_labels), key=lambda c: _first_seen[c])
_palette      = px.colors.qualitative.Plotly
c2color       = {c: _palette[i % len(_palette)] for i, c in enumerate(_unique)}
_key_sfx      = f"{_lsa_key}_{_selected_n}_{_selected_k}"

_show_lbl = n_actual <= 60
_clabels  = [str(i) for i in range(n_actual)] if _show_lbl else [""] * n_actual
fig_dend, _leaves, _root_x = build_dendrogram_figure(_Z, _labels, n_actual, c2color, _clabels)

fig_rot = go.Figure()
for _tr in fig_dend.data:
    fig_rot.add_trace(go.Scatter(
        x=list(_tr.y), y=list(_tr.x),
        mode='lines', line=_tr.line,
        showlegend=False, hoverinfo='none',
    ))
fig_rot.add_vline(x=_threshold, line_dash="dash", line_color="crimson", line_width=1.5)

for _i, _c in enumerate(_unique):
    _ly = [10 * _j + 5 for _j, _orig in enumerate(_leaves) if _labels[_orig] == _c]
    _y  = _root_x.get(_c, float(np.mean(_ly)) if _ly else 0)
    fig_rot.add_annotation(
        x=_threshold, y=_y, xshift=6, yshift=8,
        text=f"<b>{chr(65 + _i)}</b>",
        showarrow=False, xanchor='left', yanchor='bottom',
        font=dict(size=14, color='black'),
    )

_dx = fig_dend.layout.xaxis
fig_rot.update_layout(
    height=500,
    margin=dict(l=50, r=30, t=10, b=50),
    plot_bgcolor='white',
    xaxis=dict(range=[0, _z_max * 1.12], showline=False, showgrid=False, zeroline=False),
    yaxis=dict(
        tickvals=list(_dx.tickvals) if _dx.tickvals is not None else [],
        ticktext=list(_dx.ticktext) if _dx.ticktext is not None else [],
        range=list(_dx.range) if _dx.range is not None else [-5, 10 * n_actual + 5],
        showline=False, showgrid=False, zeroline=False,
    ),
)
st.plotly_chart(fig_rot, width='stretch', key=f"lsa_dend_{_key_sfx}")

meta = SOURCES_META[src_id]
st.caption(
    f"**{meta['label']}** ({LANG_LABELS[meta['lang']]}) · "
    f"{_n_tokens_early:,} tokens · {n_actual} chunks · "
    f"{_n_clusters} clusters · {_selected_n} PCA components"
)

# ── Section 2: Cluster membership strip ────────────────────────────────────────
st.divider()
st.subheader("Cluster Membership in Narrative Order")
st.caption("Colors match the dendrogram branches.")

_row_h = max(40, cfg["layout"]["heatmap_row_height_px"])
z_strip = np.zeros((_n_clusters, n_actual), dtype=float)
for _i, _c in enumerate(_unique):
    z_strip[_i, _labels == _c] = _i + 1

_N = _n_clusters + 1
_cscale = [[0, 'white'], [1 / _N, 'white']]
for _i, _c in enumerate(_unique):
    _cscale += [[(_i + 1) / _N, c2color[_c]], [(_i + 2) / _N, c2color[_c]]]

fig_strip = go.Figure(go.Heatmap(
    z=z_strip,
    x=list(range(n_actual)),
    y=[f"Cluster {chr(65 + i)}" for i in range(_n_clusters)],
    colorscale=_cscale,
    zmin=-0.5, zmax=_n_clusters + 0.5,
    showscale=False, xgap=0, ygap=2,
))
fig_strip.update_layout(
    height=_n_clusters * _row_h + 80,
    margin=dict(l=100, r=20, t=20, b=60),
    plot_bgcolor='white',
    xaxis=dict(title="Chunk (narrative order)", showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False),
)
for _i, _c in enumerate(_unique):
    _pos = np.where(_labels == _c)[0]
    if len(_pos) == 0:
        continue
    _gaps = np.where(np.diff(_pos) > 1)[0] + 1
    for _run in np.split(_pos, _gaps):
        fig_strip.add_annotation(
            x=float(np.mean(_run)), y=f"Cluster {chr(65 + _i)}",
            text=f"<b>{chr(65 + _i)}</b>",
            showarrow=False, xanchor='center', yanchor='middle',
            font=dict(size=13, color='white'),
        )
st.plotly_chart(fig_strip, width='stretch', key=f"lsa_strip_{_key_sfx}")

# ── Cluster centroid table ─────────────────────────────────────────────────────
st.subheader("Cluster Component Profiles")
st.caption(
    "Mean PCA score per cluster and component. "
    "Red = positive pole, blue = negative pole. "
    "Read each row to see which component poles define that cluster."
)

_cluster_labels_alpha = [chr(65 + i) for i in range(_n_clusters)]
_centroid = np.array([
    THETA[_labels == _c].mean(axis=0) for _c in _unique
])  # shape: (n_clusters, n_components)

_hover_n = 5
_pos_pole = []
_neg_pole = []
for _ci in range(_selected_n):
    _cv = components[_ci]
    _si = np.argsort(_cv)
    _pos_pole.append(", ".join(words[j] for j in _si[::-1][:_hover_n] if _cv[j] > 0))
    _neg_pole.append(", ".join(words[j] for j in _si[:_hover_n]      if _cv[j] < 0))

_cdata = np.empty((_n_clusters, _selected_n, 2), dtype=object)
for _ci in range(_selected_n):
    _cdata[:, _ci, 0] = _pos_pole[_ci]
    _cdata[:, _ci, 1] = _neg_pole[_ci]

fig_centroid = px.imshow(
    _centroid,
    x=[f"PC{i + 1}" for i in range(_selected_n)],
    y=_cluster_labels_alpha,
    aspect="auto",
    color_continuous_scale="RdBu_r",
    color_continuous_midpoint=0.0,
    text_auto=".2f",
    labels=dict(x="Component", y="Cluster", color="Mean score"),
)
fig_centroid.update_layout(
    height=max(160, _n_clusters * 50 + 60),
    margin=dict(l=60, r=20, t=20, b=40),
    coloraxis_showscale=True,
)
fig_centroid.update_traces(
    customdata=_cdata,
    hovertemplate=(
        "<b>Cluster %{y} · %{x}</b><br>"
        "Mean score: %{z:.3f}<br>"
        "(+) %{customdata[0]}<br>"
        "(−) %{customdata[1]}"
        "<extra></extra>"
    ),
    textfont_size=11,
)
st.plotly_chart(fig_centroid, width='stretch', key=f"lsa_centroid_{_key_sfx}")

# ── Section 3: Bipolar component word clouds ───────────────────────────────────
st.divider()
st.subheader("Component Poles (Bipolar Word Clouds)")
st.caption(
    "**Left (blue) = positive pole**: terms most strongly associated with this component. "
    "**Right (red) = negative pole**: terms that contrast with the positive pole. "
    "The two poles together define the semantic axis of each component."
)

_v      = cfg["visualization"]
_n_top  = cfg["controls"]["n_top_words"]["default"]
_wc_w, _wc_h = 200, 125

for comp_idx in range(_selected_n):
    comp_vec = components[comp_idx]
    sorted_idx = np.argsort(comp_vec)
    pos_words = {words[j]: float(comp_vec[j])  for j in sorted_idx[::-1][:50] if comp_vec[j] > 0}
    neg_words = {words[j]: float(-comp_vec[j]) for j in sorted_idx[:50]       if comp_vec[j] < 0}

    pos_top = list(pos_words.keys())[:_n_top]
    neg_top = list(neg_words.keys())[:_n_top]
    st.markdown(
        f"**PC{comp_idx + 1}** &nbsp;&nbsp; "
        f"(+) {', '.join(pos_top)} &nbsp;|&nbsp; "
        f"(−) {', '.join(neg_top)}"
    )
    wc_cols = st.columns(2)
    if pos_words:
        wc_pos = WordCloud(
            width=_wc_w, height=_wc_h,
            background_color="white", colormap="Blues",
            prefer_horizontal=_v["wordcloud_prefer_horizontal"],
        ).generate_from_frequencies(pos_words)
        wc_cols[0].image(wc_pos.to_array(), caption=f"PC{comp_idx + 1} — positive pole", width='stretch')
    if neg_words:
        wc_neg = WordCloud(
            width=_wc_w, height=_wc_h,
            background_color="white", colormap="Reds",
            prefer_horizontal=_v["wordcloud_prefer_horizontal"],
        ).generate_from_frequencies(neg_words)
        wc_cols[1].image(wc_neg.to_array(), caption=f"PC{comp_idx + 1} — negative pole", width='stretch')

# ── Section 4: Variance Explained (Scree Plot) ────────────────────────────────
st.divider()
st.subheader("Variance Explained (Scree Plot)")
st.caption("Per-component and cumulative variance explained. **Click a bar or point to select n_components.**")

_comp_vals = list(range(1, len(explained) + 1))

fig_scree = go.Figure()
fig_scree.add_trace(go.Bar(
    x=_comp_vals, y=explained * 100,
    name="Per-component", marker_color="#636EFA", opacity=0.7, showlegend=True,
))
fig_scree.add_trace(go.Scatter(
    x=_comp_vals, y=_cum_var * 100,
    name="Cumulative", line=dict(color="#EF553B"), mode="lines+markers", showlegend=True,
))
fig_scree.add_vline(
    x=_selected_n, line_dash="dash", line_color="crimson", line_width=1.5,
    annotation_text=f"n = {_selected_n}", annotation_position="top right",
)
fig_scree.update_layout(
    height=320,
    margin=dict(l=60, r=30, t=20, b=50),
    plot_bgcolor="white",
    xaxis=dict(title="Component", dtick=1, showgrid=False, zeroline=False),
    yaxis=dict(title="Variance explained (%)", showgrid=True, gridcolor="#EEEEEE", zeroline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)
_scree_evt = st.plotly_chart(fig_scree, width='stretch', key=f"lsa_scree_{_lsa_key}", on_select="rerun")
if _scree_evt.selection.points:
    _clicked_n = int(_scree_evt.selection.points[0]["x"])
    if _clicked_n != st.session_state.get("lsa_n_components") and 1 <= _clicked_n <= result['n_components']:
        st.session_state["lsa_n_components"] = _clicked_n
        st.rerun()
st.success(f"n_components = {_selected_n} · {_cum_var[_selected_n - 1] * 100:.1f}% variance explained")

# ── Section 5: Component score heatmap ────────────────────────────────────────
st.divider()
st.subheader("Component Scores Across Chunks")
st.caption(
    "Diverging colorscale (red = positive, blue = negative). "
    "Chunks with large absolute scores are strongly associated with a component's pole."
)

_x_pos = np.round(np.linspace(1, 100, n_actual)).astype(int)
fig_heat = px.imshow(
    THETA.T,
    y=[f"PC{i + 1}" for i in range(_selected_n)],
    x=_x_pos,
    aspect="auto",
    color_continuous_scale="RdBu_r",
    color_continuous_midpoint=0.0,
    labels=dict(x="Position (1–100)", y="Component"),
)
fig_heat.update_layout(
    height=max(180, _selected_n * cfg["layout"]["heatmap_row_height_px"] + 60),
    margin=dict(l=60, r=20, t=20, b=40),
)
st.plotly_chart(fig_heat, width='stretch', key=f"lsa_heat_{_lsa_key}_{_selected_n}")
