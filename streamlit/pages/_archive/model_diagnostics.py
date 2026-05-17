"""
Model Diagnostics for Narrative Structure of the Popol Wuj
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
from utils import find_token_file, load_tokens

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
LANG_LABELS  = cfg["languages"]

st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)


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
    # Restrict knee detection to words appearing in ≥5% of chunks — the range
    # where function words and common content words live. The full distribution's
    # long rare-word tail would pull the knee to a meaninglessly low value.
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


def _umass_coherence(X_bin, feature_names, top_words):
    """UMass coherence for one topic (Mimno et al. 2011)."""
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
def run_elbow(src_id, token_path, chunk_size, overlap_int, min_df, max_df, max_topics, model_type, ngram_range=(1, 1)):
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
def run_model(src_id, token_path, chunk_size, overlap_int, min_df, max_df, n_topics, model_type, n_top_words, ngram_range=(1, 1)):
    TOKEN = load_tokens(src_id, token_path)
    tokens = TOKEN["term_str"].dropna().to_list()
    step = max(1, chunk_size - overlap_int)
    token_arr = np.array(tokens)
    if len(token_arr) < chunk_size:
        return None, None, None
    windows = np.lib.stride_tricks.sliding_window_view(token_arr, chunk_size)[::step]
    chunks_s = pd.Series(
        np.apply_along_axis(lambda row: " ".join(row), axis=1, arr=windows)
    ).loc[lambda s: s.str.split().str.len() >= 50]
    chunks_list = chunks_s.tolist()
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


def render_heatmap(THETA, PHI, chunks_list, key="heatmap"):
    """Render a topic heatmap into the current Streamlit container."""
    _v = cfg["visualization"]
    n_chunks = len(chunks_list)
    n_topics = THETA.shape[1]
    _row_h = cfg["layout"]["heatmap_row_height_px"]
    height = n_topics * _row_h + 60  # 60px = top+bottom margin overhead
    _preview_len = _v["preview_len"]

    def _wrap(text, width=_v["wrap_width"]):
        words, lines, line = text.split(), [], []
        for word in words:
            if sum(len(w) for w in line) + len(line) + len(word) > width:
                lines.append(" ".join(line))
                line = [word]
            else:
                line.append(word)
        if line:
            lines.append(" ".join(line))
        return "<br>".join(lines)

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
    st.plotly_chart(fig, width='stretch', key=key)


st.title("Topic Modeling — Popol Wuj")
render_toc([
    ("Coherence & Independence", "coherence-chart"),
    ("Topic Heatmap",            "heatmap"),
    ("Topic Word Clouds",        "word-clouds"),
])

# ── Phase 1 controls (no n_topics — k is chosen by clicking the coherence plot) ──
src_ids = list(SOURCES_META.keys())
_col_ratios = cfg["layout"]["column_ratios"]
cols = st.columns(_col_ratios[:5])  # source | chunk%+overlap | min_df+max_df | model | ngram min+max

_c = cfg["controls"]
src_id     = cols[0].selectbox(
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

min_df     = cols[2].number_input("min_df", _c["min_df"]["min"], _c["min_df"]["max"], _c["min_df"]["default"], step=_c["min_df"]["step"])
max_df     = cols[2].number_input("max_df", _c["max_df"]["min"], _c["max_df"]["max"], _c["max_df"]["default"], step=_c["max_df"]["step"], format="%.2f")
_ng = _c["ngram_range"]
ngram_min = cols[3].number_input("ngram min", _ng["min_n"], _ng["max_n"], _ng["default_min"], step=1)
ngram_max = cols[3].number_input("ngram max", _ng["min_n"], _ng["max_n"], _ng["default_max"], step=1)
model_type = cols[4].selectbox("Model", ["NMF", "LDA"])
ngram_max = max(ngram_max, ngram_min)

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

# ── File resolution ───────────────────────────────────────────────────────────
token_path = find_token_file(src_id)
if token_path is None:
    st.warning(f"Token file not found for `{src_id}`.")
    st.stop()

# ── Session state: clear selected k when Phase 1 parameters change ────────────
_params_key = f"{src_id}_{chunk_size}_{overlap_int}_{min_df}_{max_df}_{model_type}_{ngram_min}_{ngram_max}"
if st.session_state.get("_diag_params_key") != _params_key:
    st.session_state["_diag_params_key"] = _params_key
    st.session_state["selected_k"] = None

# ── Phase 1: coherence curve + topic similarity dendrogram ───────────────────
elbow_df = run_elbow(src_id, token_path, chunk_size, overlap_int, min_df, max_df,
                     _c["n_topics"]["max"], model_type, (ngram_min, ngram_max))

_m = dict(l=10, r=120, t=40, b=40)
_selected_k = st.session_state.get("selected_k")

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
                                 key=f"coh_{_params_key}", on_select="rerun")
    if _coh_event.selection.points:
        _clicked_k = int(_coh_event.selection.points[0]["x"])
        if _clicked_k != st.session_state.get("selected_k"):
            st.session_state["selected_k"] = _clicked_k
            st.rerun()
    if elbow_df is not None:
        _peak_k = int(elbow_df.loc[elbow_df["coherence"].idxmax(), "n_topics"])
        st.caption(f"→ k = {_peak_k} (coherence peak)")
    if _selected_k is not None:
        st.success(f"k = {_selected_k} selected — heatmap below ↓")
    else:
        st.info("Click a point on the curve to select k.")
else:
    st.warning("Elbow analysis could not run — try adjusting chunk size, min_df, or max_df.")

# ── Phase 2: heatmap + word clouds triggered by click ─────────────────────────
_selected_k = st.session_state.get("selected_k")
if _selected_k is not None:
    st.divider()
    st.subheader(f"Heatmap — k = {_selected_k}", anchor="heatmap")

    _n_top_words = _c["n_top_words"]["default"]
    _v = cfg["visualization"]
    _ctl_cols = st.columns(3)  # chunk%+overlap | min_df+max_df | ngram min+max
    _h_chunk_pct = _ctl_cols[0].number_input("Chunk %", _cp["min"], _cp["max"],
                                              chunk_pct, step=_cp["step"], format="%.3f",
                                              key="ha_chunk")
    _h_chunk = max(50, int(_h_chunk_pct * _n_tokens_early))
    _h_overlap = _ctl_cols[0].number_input("Overlap", _c["overlap"]["min"], _c["overlap"]["max"],
                                           overlap, step=_c["overlap"]["step"], format="%.2f",
                                           key="ha_overlap")
    _h_min_df = _ctl_cols[1].number_input("min_df", _c["min_df"]["min"], _c["min_df"]["max"],
                                          min_df, step=_c["min_df"]["step"],
                                          key="ha_min_df")
    _h_max_df = _ctl_cols[1].number_input("max_df", _c["max_df"]["min"], _c["max_df"]["max"],
                                          max_df, step=_c["max_df"]["step"], format="%.2f",
                                          key="ha_max_df")
    _h_ngram_min = _ctl_cols[2].number_input("ngram min", _ng["min_n"], _ng["max_n"],
                                             ngram_min, step=1, key="ha_ngram_min")
    _h_ngram_max = _ctl_cols[2].number_input("ngram max", _ng["min_n"], _ng["max_n"],
                                             ngram_max, step=1, key="ha_ngram_max")
    _h_ngram_max = max(_h_ngram_max, _h_ngram_min)
    _h_overlap_int = int(_h_overlap * _h_chunk)
    st.caption(
        f"chunk = {_h_chunk:,} tokens · overlap = {int(_h_overlap * _h_chunk):,} tokens"
    )

    THETA, PHI, chunks_list = run_model(
        src_id, token_path, _h_chunk, _h_overlap_int,
        _h_min_df, _h_max_df, _selected_k, model_type, _n_top_words,
        (int(_h_ngram_min), int(_h_ngram_max))
    )
    if THETA is None:
        st.warning("Model couldn't run — try adjusting parameters.")
    else:
        n_chunks = len(chunks_list)
        st.caption(f"{n_chunks} chunks · {THETA.shape[0]} × {THETA.shape[1]} · chunk = {_h_chunk_pct * 100:.1f}% ({_h_chunk:,} tokens)")
        render_heatmap(THETA, PHI, chunks_list, key=f"heat_ha_{_selected_k}")

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
