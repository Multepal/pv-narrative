"""
Narrative Structure of the Popol Wuj
Streamlit conversion of overlap.ipynb
"""

import os
import re
import streamlit as st
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance

APP_DIR = os.path.dirname(os.path.abspath(__file__))

SOURCES_META = {
    "ajtzibab":            {"lang": "quc", "label": "Ajtzibab 2025"},
    "christenson":         {"lang": "quc", "label": "Christenson 2007"},
    "colop":               {"lang": "quc", "label": "Colop 2012"},
    "christenson_ximenez": {"lang": "quc", "label": "Christenson's Ximénez"},
    "ximenez":             {"lang": "quc", "label": "Ximénez"},
    "recinos":             {"lang": "spa", "label": "Recinos 1947"},
    "tedlock":             {"lang": "eng", "label": "Tedlock 1983"},
}
LANG_LABELS = {"quc": "K'iche'", "spa": "Spanish", "eng": "English"}


def find_token_file(src_id: str) -> str | None:
    candidates = [
        os.path.join(APP_DIR, src_id, f"{src_id}-TOKEN.csv"),
        os.path.join(APP_DIR, "ensemble", f"{src_id}-TOKEN.csv"),
        os.path.join(APP_DIR, f"{src_id}-TOKEN.csv"),
        os.path.join(APP_DIR, "..", src_id, f"{src_id}-TOKEN.csv"),
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

    nmf = NMF(n_components=n_topics, init="nndsvd", max_iter=500)
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


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Narrative Structure of the Popol Wuj",
    page_icon="📜",
    layout="wide",
)

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
h3 { margin-bottom: 1rem; }
@media (min-width: 768px) {
    section[data-testid="stSidebar"] {
        min-width: 280px;
        width: 25vw;
    }
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
_about_path = os.path.join(APP_DIR, "about.md")
if os.path.exists(_about_path):
    with open(_about_path, encoding="utf-8") as _f:
        _about_text = _f.read()
    with st.sidebar:
        st.markdown("# About this App")
        for _section in re.split(r'\n(?=## )', _about_text)[1:]:
            _lines = _section.split('\n', 1)
            _title = _lines[0].lstrip('#').strip()
            _body = _lines[1].strip() if len(_lines) > 1 else ''
            with st.expander(_title):
                st.markdown(_body)

st.title("The Narrative Structure of the Popol Wuj")

# ── Controls ──────────────────────────────────────────────────────────────────
src_ids = list(SOURCES_META.keys())
cols = st.columns([2, 1.5, 1.2, 1.2, 1.2, 1.2, 1.2])

src_id      = cols[0].selectbox(
    "Source", src_ids, index=src_ids.index("colop"),
    format_func=lambda x: f"{SOURCES_META[x]['label']} ({LANG_LABELS[SOURCES_META[x]['lang']]})")
chunk_size  = cols[1].number_input("Chunk size", 100, 2000, 1000, step=50)
overlap     = cols[2].number_input("Overlap", 0.0, 0.9, 0.9, step=0.05, format="%.2f")
min_df      = cols[3].number_input("min_df", 1, 20, 5, step=1)
max_df      = cols[4].number_input("max_df", 0.1, 1.0, 0.35, step=0.05, format="%.2f")
n_topics    = cols[5].number_input("Topics", 2, 20, 8, step=1)
n_top_words = cols[6].number_input("Top words", 3, 15, 7, step=1)

overlap_int = int(overlap * chunk_size)

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
    f"{n_tokens:,} tokens · {n_chunks} chunks · {THETA.shape[1]} topics"
)

# ── Heatmap ───────────────────────────────────────────────────────────────────
margin = dict(l=80, r=150, t=20, b=75)

def _wrap(text, width=60):
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

_preview_len = 300
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

fig1 = px.imshow(
    THETA.T.loc[_topic_order_plot].values,  # numpy array avoids Plotly treating int index as continuous axis
    y=_topic_labels,
    x=list(range(n_chunks)),
    aspect="auto",
    color_continuous_scale="YlGnBu",
    labels=dict(x="Syntagm / Event", y="Paradigm / Structure"),
)
_customdata = np.tile(_chunk_previews, (len(_topic_order_plot), 1))
fig1.update_traces(
    customdata=_customdata,
    hovertemplate=(
        "<b>Topic %{y} · Chunk %{x}</b><br>"
        "Weight: %{z:.3f}<br><br>"
        "%{customdata}<extra></extra>"
    ),
)
_chart_key = f"{src_id}_{chunk_size}_{overlap_int}_{min_df}_{max_df}_{n_topics}"
fig1.update_layout(height=400, margin=margin, coloraxis_showscale=False)
st.plotly_chart(fig1, width="stretch", key=f"heatmap_{_chart_key}")

# ── Cosine distance bar ───────────────────────────────────────────────────────
fig2 = px.bar(
    D["scaled"],
    labels={"value": "Scaled cosine distance", "index": "Chunk"},
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
)
fig2.update_layout(height=200, margin=margin, showlegend=False)
st.plotly_chart(fig2, width="stretch", key=f"bardist_{_chart_key}")

# ── Topic word clouds ─────────────────────────────────────────────────────────
st.divider()
st.subheader("Topic Word Clouds")
n_cols = min(4, n_topics)
topic_indices = _topic_order
for row_start in range(0, n_topics, n_cols):
    row_topics = topic_indices[row_start:row_start + n_cols]
    grid_cols = st.columns(n_cols)
    for col_idx, topic_idx in enumerate(row_topics):
        wc = WordCloud(
            width=400, height=250,
            background_color="white",
            colormap="Blues",
            prefer_horizontal=0.9,
        ).generate_from_frequencies(PHI[topic_idx])
        grid_cols[col_idx].image(wc.to_array(), caption=f"Topic {topic_idx}", use_container_width=True)


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
