"""
Structural Explorer — main analysis page.
"""

import os
import yaml
import streamlit as st
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance
from utils import load_tokens

# Point back to streamlit/ so config.yaml and token file paths resolve identically
# to when this code lived in app.py
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(APP_DIR, "config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
LANG_LABELS  = cfg["languages"]


def find_token_file(src_id: str) -> str | None:
    candidates = [
        os.path.join(APP_DIR, f"../notebooks/{src_id}/{src_id}-TOKEN.csv"),
        os.path.join(APP_DIR, f"../../notebooks/{src_id}/{src_id}-TOKEN.csv"),
    ]
    for p in candidates:
        norm = os.path.normpath(p)
        if os.path.exists(norm):
            return norm
    return None


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


@st.cache_data(show_spinner="Running model…")
def run_model(src_id, token_path, chunk_size, overlap_int, min_df, max_df, n_topics, n_top_words):
    TOKEN = load_tokens(src_id, token_path)
    tokens = TOKEN["term_str"].dropna().to_list()

    step = max(1, chunk_size - overlap_int)
    token_arr = np.array(tokens)
    if len(token_arr) < chunk_size:
        chunks_list = []
    else:
        windows = np.lib.stride_tricks.sliding_window_view(token_arr, chunk_size)[::step]
        chunks_s = pd.Series(
            np.apply_along_axis(lambda row: " ".join(row), axis=1, arr=windows)
        ).loc[lambda s: s.str.split().str.len() >= 50]
        chunks_list = chunks_s.tolist()

    if len(chunks_list) < 2:
        return None, None, None, None, len(TOKEN)

    try:
        vec = TfidfVectorizer(lowercase=True, max_df=max_df, min_df=min_df,
                              strip_accents=None, norm="l2")
        X = vec.fit_transform(chunks_list)
    except ValueError:
        return None, None, None, None, len(TOKEN)

    nmf = NMF(n_components=n_topics, init="nndsvda", max_iter=cfg["model"]["nmf_max_iter"])
    THETA = pd.DataFrame(nmf.fit_transform(X))
    THETA.index.name, THETA.columns.name = "chunk_id", "topic_id"

    words = vec.get_feature_names_out()
    PHI = [
        {words[j]: nmf.components_[i][j] for j in nmf.components_[i].argsort()[::-1][:n_top_words]}
        for i in range(n_topics)
    ]
    scaler = MinMaxScaler((0, 1))
    D = pd.concat([THETA.shift(1), THETA.shift(0)], axis=1, keys=["a", "b"]).dropna() \
        .apply(lambda x: distance.cosine(x.a, x.b), axis=1).to_frame("d")
    D["scaled"] = scaler.fit_transform(D[["d"]])

    return THETA, PHI, D, chunks_list, len(TOKEN)


st.title(cfg["app"]["title"])

# ── Controls ──────────────────────────────────────────────────────────────────
src_ids = list(SOURCES_META.keys())
cols = st.columns(cfg["layout"]["column_ratios"])

_c = cfg["controls"]
src_id      = cols[0].selectbox(
    "Source", src_ids, index=src_ids.index(_c["default_source"]),
    format_func=lambda x: f"{SOURCES_META[x]['label']} ({LANG_LABELS[SOURCES_META[x]['lang']]})")

_early_token_path = find_token_file(src_id)
_n_tokens_early = len(load_tokens(src_id, _early_token_path)) if _early_token_path else None

_cp = _c["chunk_pct"]
chunk_pct   = cols[1].number_input("Chunk %", _cp["min"], _cp["max"], _cp["default"], step=_cp["step"], format="%.3f")
if _n_tokens_early:
    cols[1].caption(f"{int(chunk_pct * _n_tokens_early):,} tokens")
chunk_size  = max(50, int(chunk_pct * _n_tokens_early)) if _n_tokens_early else 100
overlap     = cols[2].number_input("Overlap", _c["overlap"]["min"], _c["overlap"]["max"], _c["overlap"]["default"], step=_c["overlap"]["step"], format="%.2f")
if _n_tokens_early:
    cols[2].caption(f"{int(overlap * chunk_size):,} tokens")
overlap_int = int(overlap * chunk_size)

_vstats = compute_vocab_stats(src_id, _early_token_path, chunk_size, overlap_int) if _early_token_path else None

min_df      = cols[3].number_input("min_df", _c["min_df"]["min"], _c["min_df"]["max"], _c["min_df"]["default"], step=_c["min_df"]["step"])
if _vstats:
    cols[3].caption(f"→ {_vstats[1]} (2% of {_vstats[0]} chunks)")
max_df      = cols[4].number_input("max_df", _c["max_df"]["min"], _c["max_df"]["max"], _c["max_df"]["default"], step=_c["max_df"]["step"], format="%.2f")
if _vstats:
    cols[4].caption(f"→ {_vstats[2]:.2f} (vocab knee)")
n_topics    = cols[5].number_input("Topics", _c["n_topics"]["min"], _c["n_topics"]["max"], _c["n_topics"]["default"], step=_c["n_topics"]["step"])
n_top_words = cols[6].number_input("Top words", _c["n_top_words"]["min"], _c["n_top_words"]["max"], _c["n_top_words"]["default"], step=_c["n_top_words"]["step"])

st.divider()

# ── File resolution ───────────────────────────────────────────────────────────
token_path = find_token_file(src_id)

if token_path is None:
    tried = [
        os.path.normpath(os.path.join(APP_DIR, src_id, f"{src_id}-TOKEN.csv")),
        os.path.normpath(os.path.join(APP_DIR, "ensemble", f"{src_id}-TOKEN.csv")),
        os.path.normpath(os.path.join(APP_DIR, f"{src_id}-TOKEN.csv")),
    ]
    st.warning(
        f"**Token file not found for `{src_id}`.**\n\nLooked in:\n" +
        "\n".join(f"- `{p}`" for p in tried),
        icon="📂",
    )
    uploaded = st.file_uploader(f"Upload `{src_id}-TOKEN.csv`", type="csv", key=f"upload_{src_id}")
    if uploaded is not None:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded.read())
        tmp.flush()
        token_path = tmp.name
    else:
        st.stop()

# ── Run model ─────────────────────────────────────────────────────────────────
THETA, PHI, D, chunks_list, n_tokens = run_model(
    src_id, token_path, chunk_size, overlap_int, min_df, max_df, n_topics, n_top_words
)

if THETA is None:
    st.warning("Model couldn't run — try adjusting chunk size, min_df, or max_df.")
    st.stop()

meta = SOURCES_META[src_id]
n_chunks = len(chunks_list)

st.caption(
    f"**{meta['label']}** ({LANG_LABELS[meta['lang']]}) · "
    f"{n_tokens:,} tokens · {n_chunks} chunks · {THETA.shape[1]} topics · "
    f"chunk = {chunk_pct * 100:.1f}% ({chunk_size:,} tokens)"
)

# ── Heatmap ───────────────────────────────────────────────────────────────────
_v = cfg["visualization"]
margin = dict(**cfg["layout"]["margin"])

# session state for bar→heatmap vlines
_chart_key = f"{src_id}_{chunk_size}_{overlap_int}_{min_df}_{max_df}_{n_topics}"
if st.session_state.get("vline_chart_key") != _chart_key:
    st.session_state["active_chunks"]    = set()
    st.session_state["last_bar_sel"]     = None
    st.session_state["vline_chart_key"]  = _chart_key
if "active_chunks" not in st.session_state:
    st.session_state["active_chunks"] = set()
if "last_bar_sel" not in st.session_state:
    st.session_state["last_bar_sel"] = None

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

_preview_len = _v["preview_len"]
_chunk_previews = [
    _wrap((c[:_preview_len] + "…") if len(c) > _preview_len else c)
    for c in chunks_list
]

# Topic order by first dominance: collect topics in the order they first have the highest weight
_topic_seq = THETA.idxmax(axis=1).tolist()
_topic_order = []
for t in _topic_seq:
    if t not in _topic_order:
        _topic_order.append(t)
# Append any topics that never dominated a chunk
for t in range(THETA.shape[1]):
    if t not in _topic_order:
        _topic_order.append(t)
# Reverse so earliest-appearing topic is at the bottom of the chart
_topic_order_plot = list(reversed(_topic_order))
_topic_labels = [f"Topic {i}" for i in _topic_order_plot]

_x_scaled = np.round(np.linspace(0, 100, n_chunks)).astype(int)
_chunk_to_scaled = dict(enumerate(_x_scaled))

fig1 = px.imshow(
    THETA.T.loc[_topic_order_plot].values,  # numpy array avoids Plotly treating int index as continuous axis
    y=_topic_labels,
    x=_x_scaled,
    aspect="auto",
    color_continuous_scale=_v["heatmap_color_scale"],
    labels=dict(x="Position (0–100)", y="Paradigm / Structure"),
)
_topic_words = [", ".join(list(PHI[t].keys())[:_v["hover_top_words"]]) for t in _topic_order_plot]
_customdata = np.empty((len(_topic_order_plot), n_chunks, 2), dtype=object)
_customdata[:, :, 0] = np.tile(_chunk_previews, (len(_topic_order_plot), 1))
_customdata[:, :, 1] = np.array(_topic_words)[:, np.newaxis]
fig1.update_traces(
    customdata=_customdata,
    hovertemplate=(
        "<b>Topic %{y} · Position %{x}</b><br>"
        "Weight: %{z:.3f}<br>"
        "<i>%{customdata[1]}</i><br><br>"
        "%{customdata[0]}<extra></extra>"
    ),
)
fig1.update_layout(height=cfg["layout"]["heatmap_height"], margin=margin, coloraxis_showscale=False)
_heatmap_slot = st.empty()  # filled after bar click is processed

# ── Cosine distance bar ───────────────────────────────────────────────────────
_active_chunks = set(st.session_state["active_chunks"])
_default_color = "#636EFA"
_active_color  = "#EF553B"
_bar_colors = [_active_color if i in _active_chunks else _default_color for i in D.index]

fig2 = px.bar(
    D["scaled"],
    labels={"value": "Scaled cosine distance", "chunk_id": "Syntagm / Event (chunk_id)"},
)
_bar_previews = [_wrap((chunks_list[i][:_preview_len] + "…") if len(chunks_list[i]) > _preview_len else chunks_list[i])
                 for i in D.index]
fig2.update_traces(
    customdata=_bar_previews,
    hovertemplate=(
        "<b>Chunk %{x}</b><br>"
        "Distance: %{y:.3f}<br><br>"
        "%{customdata}<extra></extra>"
    ),
    marker_color=_bar_colors,
    selected={"marker": {"color": _active_color, "opacity": 1.0}},
    unselected={"marker": {"opacity": 1.0}},
)
fig2.update_layout(height=cfg["layout"]["bar_height"], margin=margin, showlegend=False,
                   clickmode="event+select")
_bar_event = st.plotly_chart(fig2, width="stretch", key=f"bardist_{_chart_key}", on_select="rerun")

# Toggle logic — compare current Plotly selection to last known selection
_cur = _bar_event.selection.points[0]["x"] if _bar_event.selection.points else None
_last = st.session_state["last_bar_sel"]
if _cur != _last:
    if _cur is not None:
        if _cur in _active_chunks:
            _active_chunks.discard(_cur)
        else:
            _active_chunks.add(_cur)
        st.session_state["active_chunks"] = _active_chunks
    st.session_state["last_bar_sel"]  = _cur

# Fill heatmap slot now that active_chunks is current
for _cid in st.session_state["active_chunks"]:
    fig1.add_vline(x=_chunk_to_scaled.get(_cid, _cid), line_dash="dash", line_color="lightgray", line_width=1.5)
with _heatmap_slot:
    st.plotly_chart(fig1, width='stretch', key=f"heatmap_{_chart_key}")

# ── Topic word clouds ─────────────────────────────────────────────────────────
st.divider()
st.subheader("Topic Word Clouds")
n_cols = min(_v["wordcloud_cols"], n_topics)
topic_indices = _topic_order
for row_start in range(0, n_topics, n_cols):
    row_topics = topic_indices[row_start:row_start + n_cols]
    grid_cols = st.columns(n_cols)
    for col_idx, topic_idx in enumerate(row_topics):
        wc = WordCloud(
            width=_v["wordcloud_width"], height=_v["wordcloud_height"],
            background_color="white",
            colormap=_v["wordcloud_colormap"],
            prefer_horizontal=_v["wordcloud_prefer_horizontal"],
        ).generate_from_frequencies(PHI[topic_idx])
        grid_cols[col_idx].image(wc.to_array(), caption=f"Topic {topic_idx}", width='stretch')

# ── Chunk viewer ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("Chunk Viewer")
chunk_id = st.slider("Chunk", 0, n_chunks - 1, 0)
st.markdown(
    f"<div style='font-family: Georgia, serif; font-size: 14pt; line-height: 1.8;'>"
    f"{chunks_list[chunk_id]}"
    f"</div>",
    unsafe_allow_html=True,
)
