"""
Topic Modeling (n-chunks) — NMF/LDA diagnostics using pd.cut for exact chunk counts.
"""

import os
import yaml
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_distances
from toc import render_toc
from utils import find_token_file, load_tokens, make_chunks, wrap_text, load_boundaries, add_boundary_vlines

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
LANG_LABELS  = cfg["languages"]

st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)


def _umass_coherence(X_bin, feature_names, top_words):
    name_idx = {w: i for i, w in enumerate(feature_names)}
    indices = [name_idx[w] for w in top_words if w in name_idx]
    if len(indices) < 2:
        return 0.0
    score, count = 0.0, 0
    for i in range(1, len(indices)):
        for j in range(i):
            co   = X_bin[:, indices[i]].multiply(X_bin[:, indices[j]]).sum()
            df_j = X_bin[:, indices[j]].sum()
            if df_j > 0:
                score += np.log((co + 1.0) / df_j)
                count += 1
    return score / count if count > 0 else 0.0


@st.cache_data(show_spinner="Running elbow analysis…")
def run_elbow(src_id, token_path, n_chunks, min_df, max_df, max_topics, model_type, ngram_range=(1, 1)):
    chunks_list = make_chunks(src_id, token_path, n_chunks)
    if len(chunks_list) < 2:
        return None
    try:
        if model_type == "NMF":
            vec = TfidfVectorizer(lowercase=True, max_df=max_df, min_df=min_df,
                                  strip_accents=None, norm="l2", ngram_range=ngram_range)
        else:
            vec = CountVectorizer(lowercase=True, max_df=max_df, min_df=min_df,
                                  strip_accents=None, ngram_range=ngram_range)
        X = vec.fit_transform(chunks_list)
    except ValueError:
        return None

    X_bin = (X > 0).tocsc()
    words = vec.get_feature_names_out()
    n_top = 10

    rows = []
    for n in range(2, max_topics + 1):
        if model_type == "NMF":
            m = NMF(n_components=n, init="nndsvda", max_iter=cfg["model"]["nmf_max_iter"])
            m.fit(X)
        else:
            m = LatentDirichletAllocation(n_components=n, random_state=42, max_iter=20)
            m.fit(X)
        phi = [
            {words[j]: m.components_[i][j] for j in m.components_[i].argsort()[::-1][:n_top]}
            for i in range(n)
        ]
        coherence = float(np.mean([_umass_coherence(X_bin, words, list(p.keys())) for p in phi]))
        dists = cosine_distances(m.components_)
        idx = np.triu_indices(n, k=1)
        independence = float(np.mean(dists[idx])) if n > 1 else 0.0
        rows.append({"n_topics": n, "coherence": coherence, "independence": independence})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Running topic model…")
def run_model(src_id, token_path, n_chunks, min_df, max_df, n_topics, model_type, n_top_words, ngram_range=(1, 1)):
    chunks_list = make_chunks(src_id, token_path, n_chunks)
    if len(chunks_list) < 2:
        return None, None, None
    try:
        if model_type == "NMF":
            vec = TfidfVectorizer(lowercase=True, max_df=max_df, min_df=min_df,
                                  strip_accents=None, norm="l2", ngram_range=ngram_range)
            X = vec.fit_transform(chunks_list)
            model = NMF(n_components=n_topics, init="nndsvda", max_iter=cfg["model"]["nmf_max_iter"])
        else:
            vec = CountVectorizer(lowercase=True, max_df=max_df, min_df=min_df,
                                  strip_accents=None, ngram_range=ngram_range)
            X = vec.fit_transform(chunks_list)
            model = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
    except ValueError:
        return None, None, None
    THETA = pd.DataFrame(model.fit_transform(X))
    THETA.index.name, THETA.columns.name = "chunk_id", "topic_id"
    words = vec.get_feature_names_out()
    PHI = [
        {words[j]: model.components_[i][j] for j in model.components_[i].argsort()[::-1][:n_top_words]}
        for i in range(n_topics)
    ]
    return THETA, PHI, chunks_list


def render_heatmap(THETA, PHI, chunks_list, key="heatmap", boundaries=None):
    _v = cfg["visualization"]
    n_chunks = len(chunks_list)
    _show_bounds = st.checkbox("Show episode boundaries", value=False, key=f"diag_bounds_{key}")
    _active_bounds = (boundaries or []) if _show_bounds else []
    n_topics = THETA.shape[1]
    _row_h = cfg["layout"]["heatmap_row_height_px"]
    height = n_topics * _row_h + 60
    _preview_len = _v["preview_len"]

    _wrap = lambda text: wrap_text(text, _v["wrap_width"])
    _chunk_previews = [
        _wrap((c[:_preview_len] + "…") if len(c) > _preview_len else c)
        for c in chunks_list
    ]
    _topic_seq = THETA.idxmax(axis=1).tolist()
    _topic_order = []
    for t in _topic_seq:
        if t not in _topic_order:
            _topic_order.append(t)
    for t in range(THETA.shape[1]):
        if t not in _topic_order:
            _topic_order.append(t)
    _topic_order_plot = list(reversed(_topic_order))
    _topic_labels = [f"Topic {i}" for i in _topic_order_plot]

    _x_scaled = np.round(np.linspace(1, 100, n_chunks)).astype(int)
    fig = px.imshow(
        THETA.T.loc[_topic_order_plot].values,
        y=_topic_labels,
        x=_x_scaled,
        aspect="auto",
        color_continuous_scale=_v["heatmap_color_scale"],
        labels=dict(x="Position (1–100)", y="Topic"),
    )
    _topic_words = [", ".join(list(PHI[t].keys())[:_v["hover_top_words"]]) for t in _topic_order_plot]
    _customdata = np.empty((len(_topic_order_plot), n_chunks, 2), dtype=object)
    _customdata[:, :, 0] = np.tile(_chunk_previews, (len(_topic_order_plot), 1))
    _customdata[:, :, 1] = np.array(_topic_words)[:, np.newaxis]
    fig.update_traces(
        customdata=_customdata,
        hovertemplate=(
            "<b>Topic %{y} · Position %{x}</b><br>"
            "Weight: %{z:.3f}<br>"
            "<i>%{customdata[1]}</i><br><br>"
            "%{customdata[0]}<extra></extra>"
        ),
    )
    fig.update_layout(height=height, margin=dict(l=60, r=20, t=20, b=40),
                      coloraxis_showscale=False)
    if _active_bounds:
        add_boundary_vlines(fig, _active_bounds, n_chunks, x_is_pct=True)
    st.plotly_chart(fig, width='stretch', key=key)

    _theta_export = THETA.copy()
    _theta_export.columns = [f"T{c}" for c in _theta_export.columns]
    _theta_export.insert(0, "chunk", list(range(n_chunks)))
    st.download_button(
        "↓ Download topic weights (CSV)",
        data=_theta_export.to_csv(index=False),
        file_name=f"nmf_theta_{key}.csv",
        mime="text/csv",
    )


# ── Controls ──────────────────────────────────────────────────────────────────
st.title("Topic Modeling (n-chunks) — Popol Wuj")
render_toc([
    ("Coherence & Independence", "coherence-chart"),
    ("Topic Heatmap",            "heatmap"),
    ("Topic Word Clouds",        "word-clouds"),
])

src_ids = list(SOURCES_META.keys())
_col_ratios = cfg["layout"]["column_ratios"]
cols = st.columns(_col_ratios[:5])  # source | n_chunks | min_df+max_df | ngram min+max | model

_c = cfg["controls"]
src_id = cols[0].selectbox(
    "Source", src_ids, index=src_ids.index(_c["default_source"]),
    format_func=lambda x: f"{SOURCES_META[x]['label']} ({LANG_LABELS[SOURCES_META[x]['lang']]})")

_early_token_path = find_token_file(src_id)
_n_tokens_early = len(load_tokens(src_id, _early_token_path)) if _early_token_path else None

n_chunks   = cols[1].number_input("n_chunks", min_value=_c["n_chunks"]["min"], max_value=_c["n_chunks"]["max"], value=_c["n_chunks"]["default"], step=_c["n_chunks"]["step"])
min_df     = cols[2].number_input("min_df", _c["min_df"]["min"], _c["min_df"]["max"], _c["min_df"]["default"], step=_c["min_df"]["step"])
max_df     = cols[2].number_input("max_df", _c["max_df"]["min"], _c["max_df"]["max"], _c["max_df"]["default"], step=_c["max_df"]["step"], format="%.2f")
_ng        = _c["ngram_range"]
ngram_min  = cols[3].number_input("ngram min", _ng["min_n"], _ng["max_n"], _ng["default_min"], step=1)
ngram_max  = cols[3].number_input("ngram max", _ng["min_n"], _ng["max_n"], _ng["default_max"], step=1)
model_type = cols[4].selectbox("Model", ["NMF", "LDA"])
ngram_max  = max(ngram_max, ngram_min)

if _n_tokens_early:
    st.caption(f"n_chunks = {n_chunks} · approx. {int(_n_tokens_early / n_chunks):,} tokens per chunk")

st.divider()

token_path = find_token_file(src_id)
if token_path is None:
    st.warning(f"Token file not found for `{src_id}`.")
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
_params_key = f"{src_id}_{n_chunks}_{min_df}_{max_df}_{model_type}_{ngram_min}_{ngram_max}"
if st.session_state.get("_diag_nc_params_key") != _params_key:
    st.session_state["_diag_nc_params_key"] = _params_key
    st.session_state["selected_nc_k"] = None

# ── Phase 1: coherence + independence ─────────────────────────────────────────
elbow_df = run_elbow(src_id, token_path, n_chunks, min_df, max_df,
                     _c["n_topics"]["max"], model_type, (ngram_min, ngram_max))

_m = dict(l=10, r=120, t=40, b=40)
_selected_k = st.session_state.get("selected_nc_k")

st.subheader("Coherence & Independence vs. Number of Topics", anchor="coherence-chart")
st.caption(
    "UMass coherence (blue, left axis): higher = more coherent. "
    "Mean pairwise cosine distance (red, right axis): higher = more independent. "
    "**Click a point to select k.**"
)
if elbow_df is not None:
    fig_coh = make_subplots(specs=[[{"secondary_y": True}]])
    fig_coh.add_trace(
        go.Scatter(x=elbow_df["n_topics"], y=elbow_df["coherence"],
                   mode="lines+markers", name="Coherence",
                   line=dict(color="#636EFA")),
        secondary_y=False,
    )
    fig_coh.add_trace(
        go.Scatter(x=elbow_df["n_topics"], y=elbow_df["independence"],
                   mode="lines+markers", name="Independence",
                   line=dict(color="#EF553B")),
        secondary_y=True,
    )
    if _selected_k is not None:
        fig_coh.add_vline(x=_selected_k, line_dash="dash", line_color="gray", line_width=2,
                          annotation_text=f"k = {_selected_k}", annotation_position="top right")
    fig_coh.update_yaxes(title_text="Mean UMass coherence", secondary_y=False)
    fig_coh.update_yaxes(title_text="Mean pairwise cosine distance", secondary_y=True, showgrid=False)
    fig_coh.update_xaxes(title_text="Number of topics (k)")
    fig_coh.update_layout(height=380, margin=_m,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    _coh_event = st.plotly_chart(fig_coh, width='stretch',
                                 key=f"nc_coh_{_params_key}", on_select="rerun")
    if _coh_event.selection.points:
        _clicked_k = int(_coh_event.selection.points[0]["x"])
        if _clicked_k != st.session_state.get("selected_nc_k"):
            st.session_state["selected_nc_k"] = _clicked_k
            st.rerun()
    if elbow_df is not None:
        _peak_k = int(elbow_df.loc[elbow_df["coherence"].idxmax(), "n_topics"])
        st.caption(f"→ k = {_peak_k} (coherence peak)")
    if _selected_k is not None:
        st.success(f"k = {_selected_k} selected — heatmap below ↓")
    else:
        st.info("Click a point on the curve to select k.")
else:
    st.warning("Elbow analysis could not run — try adjusting n_chunks, min_df, or max_df.")

# ── Phase 2: heatmap + word clouds ────────────────────────────────────────────
_selected_k = st.session_state.get("selected_nc_k")
if _selected_k is not None:
    st.divider()
    st.subheader(f"Heatmap — k = {_selected_k}", anchor="heatmap")

    _n_top_words = _c["n_top_words"]["default"]
    _v = cfg["visualization"]
    _ctl_cols = st.columns(3)
    _h_n_chunks  = _ctl_cols[0].number_input("n_chunks", min_value=_c["n_chunks"]["min"], max_value=_c["n_chunks"]["max"],
                                              value=n_chunks, step=_c["n_chunks"]["step"], key="ha_nc_n_chunks")
    _h_min_df    = _ctl_cols[1].number_input("min_df", _c["min_df"]["min"], _c["min_df"]["max"],
                                              min_df, step=_c["min_df"]["step"], key="ha_nc_min_df")
    _h_max_df    = _ctl_cols[1].number_input("max_df", _c["max_df"]["min"], _c["max_df"]["max"],
                                              max_df, step=_c["max_df"]["step"], format="%.2f",
                                              key="ha_nc_max_df")
    _h_ngram_min = _ctl_cols[2].number_input("ngram min", _ng["min_n"], _ng["max_n"],
                                              ngram_min, step=1, key="ha_nc_ngram_min")
    _h_ngram_max = _ctl_cols[2].number_input("ngram max", _ng["min_n"], _ng["max_n"],
                                              ngram_max, step=1, key="ha_nc_ngram_max")
    _h_ngram_max = max(_h_ngram_max, _h_ngram_min)
    st.caption(
        f"n_chunks = {_h_n_chunks} · approx. {int(_n_tokens_early / _h_n_chunks):,} tokens per chunk"
        if _n_tokens_early else ""
    )

    THETA, PHI, chunks_list = run_model(
        src_id, token_path, _h_n_chunks,
        _h_min_df, _h_max_df, _selected_k, model_type, _n_top_words,
        (int(_h_ngram_min), int(_h_ngram_max))
    )
    if THETA is None:
        st.warning("Model couldn't run — try adjusting parameters.")
    else:
        st.caption(f"{len(chunks_list)} chunks · {THETA.shape[0]} × {THETA.shape[1]}")
        render_heatmap(THETA, PHI, chunks_list, key=f"nc_heat_{_selected_k}",
                       boundaries=load_boundaries(APP_DIR))

        st.divider()
        st.subheader("Topic Word Clouds", anchor="word-clouds")
        _wc_topic_seq = THETA.idxmax(axis=1).tolist()
        _wc_topic_order = []
        for t in _wc_topic_seq:
            if t not in _wc_topic_order:
                _wc_topic_order.append(t)
        for t in range(_selected_k):
            if t not in _wc_topic_order:
                _wc_topic_order.append(t)
        n_wc_cols = min(_v["wordcloud_cols"], _selected_k)
        for row_start in range(0, _selected_k, n_wc_cols):
            row_topics = _wc_topic_order[row_start:row_start + n_wc_cols]
            grid_cols = st.columns(n_wc_cols)
            for col_idx, topic_idx in enumerate(row_topics):
                wc = WordCloud(
                    width=_v["wordcloud_width"], height=_v["wordcloud_height"],
                    background_color="white",
                    colormap=_v["wordcloud_colormap"],
                    prefer_horizontal=_v["wordcloud_prefer_horizontal"],
                ).generate_from_frequencies(PHI[topic_idx])
                grid_cols[col_idx].image(wc.to_array(), caption=f"Topic {topic_idx}", width='stretch')
