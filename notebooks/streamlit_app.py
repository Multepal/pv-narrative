"""
Overlapping Chunks Topic Modeler
Streamlit conversion of overlap.ipynb
"""

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# ── Path resolution ───────────────────────────────────────────────────────────
# All paths are resolved relative to this file's own location on disk,
# matching the notebook layout: app.py is a sibling of the source folders.
# e.g.  project/overlap/app.py  →  project/recinos/recinos-TOKEN.csv
APP_DIR = os.path.dirname(os.path.abspath(__file__))

def find_token_file(src_id: str) -> str | None:
    candidates = [
        os.path.join(APP_DIR, "..", src_id, f"{src_id}-TOKEN.csv"),  # notebook layout
        os.path.join(APP_DIR, src_id, f"{src_id}-TOKEN.csv"),        # sub-folder of app
        os.path.join(APP_DIR, f"{src_id}-TOKEN.csv"),                 # flat, same dir
    ]
    for p in candidates:
        norm = os.path.normpath(p)
        if os.path.exists(norm):
            return norm
    return None

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Overlapping Chunks Topic Modeler",
    page_icon="📜",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Source+Code+Pro:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Libre Baskerville', Georgia, serif; }
h1, h2, h3 { font-family: 'Libre Baskerville', Georgia, serif; letter-spacing: -0.01em; }
.stDataFrame, code, pre { font-family: 'Source Code Pro', monospace !important; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ── Source catalog ────────────────────────────────────────────────────────────
SOURCES_META = {
    "ajtzibab":            {"lang": "quc", "label": "Ajtzibab"},
    "christenson":         {"lang": "quc", "label": "Christenson"},
    "colop":               {"lang": "quc", "label": "Colop"},
    "christenson_ximenez": {"lang": "quc", "label": "Christenson Ximenez"},
    "ximenez":             {"lang": "quc", "label": "Ximenez"},
    "recinos":             {"lang": "spa", "label": "Recinos"},
    "tedlock":             {"lang": "eng", "label": "Tedlock"},
}
LANG_LABELS = {"quc": "K'iche'", "spa": "Spanish", "eng": "English"}

# ── Helper functions ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_tokens(src_id: str, token_path: str) -> pd.DataFrame:
    TOKEN = pd.read_csv(token_path)
    idx_offset = TOKEN.columns.to_list().index("token_str")
    ohco = TOKEN.columns.to_list()[:idx_offset]
    return TOKEN.set_index(ohco)


def chunk_tokens(TOKEN: pd.DataFrame, chunk_size: int, overlap: int, min_len: int = 50):
    tokens = TOKEN["term_str"].dropna().to_list()
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(tokens), step):
        chunk = " ".join(tokens[i : i + chunk_size])
        if len(chunk.split()) >= min_len:
            chunks.append(chunk)
    return chunks


def get_nmf_topics(nmf_model, tfidf_vectorizer, n_top_words: int) -> pd.DataFrame:
    words = tfidf_vectorizer.get_feature_names_out()
    topic_words = {}
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_idx = topic.argsort()[: -n_top_words - 1 : -1]
        topic_words[f"Topic {topic_idx}"] = [words[i] for i in top_idx]
    return pd.DataFrame(topic_words)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📜 Topic Modeler")
    st.caption("Overlapping Chunks + NMF")
    st.divider()

    lang_filter = st.radio(
        "Filter by language",
        ["All"] + list(LANG_LABELS.values()),
        horizontal=True,
    )
    lang_code_map = {v: k for k, v in LANG_LABELS.items()}
    filtered_sources = {
        sid: meta
        for sid, meta in SOURCES_META.items()
        if lang_filter == "All" or meta["lang"] == lang_code_map.get(lang_filter)
    }

    src_id = st.selectbox(
        "Source text",
        options=list(filtered_sources.keys()),
        format_func=lambda x: f"{SOURCES_META[x]['label']} ({LANG_LABELS[SOURCES_META[x]['lang']]})",
    )

    st.divider()
    st.subheader("Chunking")
    chunk_size = st.slider("Chunk size (tokens)", 100, 2000, 1000, step=5)
    overlap_pct = st.slider("Overlap", 0.0, 0.9, 0.9, step=0.01, format="%.2f")
    overlap_int = int(overlap_pct * chunk_size)
    st.caption(f"≈ {overlap_int} token overlap per chunk")

    st.divider()
    st.subheader("TF-IDF")
    min_df = st.slider("min_df", 1, 20, 5)
    max_df = st.slider("max_df", 0.10, 1.00, 0.35, step=0.01, format="%.2f")

    st.divider()
    st.subheader("NMF")
    n_topics = st.slider("Number of topics", 2, 20, 8)
    n_top_words = st.slider("Top words per topic", 3, 15, 7)

    st.divider()
    run_btn = st.button("▶  Run Model", type="primary", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Overlapping Chunks Topic Modeler")
st.markdown(
    "Explore topic structure across **overlapping text windows** using "
    "TF-IDF vectorisation and Non-negative Matrix Factorisation (NMF)."
)

# ── File resolution ───────────────────────────────────────────────────────────
token_path = find_token_file(src_id)

if token_path is None:
    tried = [
        os.path.normpath(os.path.join(APP_DIR, "..", src_id, f"{src_id}-TOKEN.csv")),
        os.path.normpath(os.path.join(APP_DIR, src_id, f"{src_id}-TOKEN.csv")),
        os.path.normpath(os.path.join(APP_DIR, f"{src_id}-TOKEN.csv")),
    ]
    st.warning(
        f"**Token file not found for `{src_id}`.**\n\n"
        "Looked in:\n" + "\n".join(f"- `{p}`" for p in tried),
        icon="📂",
    )
    uploaded = st.file_uploader(
        f"Upload `{src_id}-TOKEN.csv`", type="csv", key=f"upload_{src_id}"
    )
    if uploaded is not None:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded.read())
        tmp.flush()
        token_path = tmp.name
    else:
        st.stop()

# ── Load tokens ───────────────────────────────────────────────────────────────
with st.spinner("Loading token file…"):
    try:
        TOKEN = load_tokens(src_id, token_path)
    except Exception as e:
        st.error(f"Could not load token file: {e}")
        st.stop()

meta = SOURCES_META[src_id]
col1, col2, col3 = st.columns(3)
col1.metric("Source", meta["label"])
col2.metric("Language", LANG_LABELS[meta["lang"]])
col3.metric("Total tokens", f"{len(TOKEN):,}")
st.divider()

# ── Run model ─────────────────────────────────────────────────────────────────
if run_btn or "theta" not in st.session_state:
    with st.spinner("Chunking tokens…"):
        chunks = chunk_tokens(TOKEN, chunk_size=chunk_size, overlap=overlap_int)

    if len(chunks) < 2:
        st.warning("Too few chunks — try a smaller chunk size or less overlap.")
        st.stop()

    with st.spinner("Fitting TF-IDF…"):
        try:
            count_engine = TfidfVectorizer(
                lowercase=True, max_df=max_df, min_df=min_df,
                strip_accents=None, norm="l2",
            )
            X = count_engine.fit_transform(chunks)
        except ValueError as e:
            st.error(f"TF-IDF error: {e}\n\nTry adjusting min_df or max_df.")
            st.stop()

    with st.spinner("Fitting NMF…"):
        topic_engine = NMF(n_components=n_topics, init="nndsvd", max_iter=500)
        THETA = pd.DataFrame(topic_engine.fit_transform(X))
        THETA.index.name = "chunk_id"
        THETA.columns.name = "topic_id"
        TOPICS = get_nmf_topics(topic_engine, count_engine, n_top_words)

    st.session_state.update({"theta": THETA, "topics": TOPICS, "n_chunks": len(chunks)})

# ── Display results ───────────────────────────────────────────────────────────
if "theta" in st.session_state:
    THETA    = st.session_state["theta"]
    TOPICS   = st.session_state["topics"]
    n_chunks = st.session_state["n_chunks"]

    st.subheader("Topic Distribution Heatmap")
    st.caption(
        f"{n_chunks} chunks · {THETA.shape[1]} topics  |  "
        "X = syntagm (narrative position) · Y = paradigm (topic structure)"
    )

    fig, ax = plt.subplots(figsize=(20, 4))
    sns.heatmap(
        THETA.T, cmap="YlGnBu", ax=ax, linewidths=0,
        xticklabels=max(1, n_chunks // 40),
        yticklabels=[f"T{i}" for i in range(THETA.shape[1])],
    )
    ax.set_title(meta["label"], fontsize=18, fontweight="bold", pad=12)
    ax.set_xlabel("Syntagm / Event (chunk index)", fontsize=13)
    ax.set_ylabel("Paradigm / Structure (topic)", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()
    st.subheader("Top Terms per Topic")
    topic_display = TOPICS.copy()
    topic_display.index = [f"Rank {i+1}" for i in range(len(TOPICS))]
    st.dataframe(topic_display, use_container_width=True, height=min(40 + 35 * len(TOPICS), 500))

    st.download_button(
        "⬇  Download THETA (chunk × topic weights)",
        data=THETA.to_csv().encode(),
        file_name=f"{src_id}_theta.csv",
        mime="text/csv",
    )

# ── Bibliography ──────────────────────────────────────────────────────────────
with st.expander("Bibliography"):
    st.markdown(
        "Recinos, A. (1947). *Popol Vuh: Las Antiguas Historias Del Quiché*. "
        "Fondo de Cultura Económica. "
        "[Google Books](https://books.google.com/books?hl=en&id=p9hpEAAAQBAJ)"
    )
