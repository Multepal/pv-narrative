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


site_title = "The Narrative Structure of the Popol Wuj"

# ── Path resolution ───────────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))

def find_token_file(src_id: str) -> str | None:
    candidates = [
        os.path.join(APP_DIR, "..", src_id, f"{src_id}-TOKEN.csv"),
        os.path.join(APP_DIR, src_id, f"{src_id}-TOKEN.csv"),
        os.path.join(APP_DIR, f"{src_id}-TOKEN.csv"),
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

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
div[data-testid="stHorizontalBlock"] { align-items: end; }
</style>
""", unsafe_allow_html=True)

# ── Source catalog ────────────────────────────────────────────────────────────
SOURCES_META = {
    "ajtzibab":            {"lang": "quc", "label": "Ajtzibab"},
    "christenson":         {"lang": "quc", "label": "Christenson"},
    "colop":               {"lang": "quc", "label": "Colop"},
    "christenson_ximenez": {"lang": "quc", "label": "Christenson's Ximénez"},
    "ximenez":             {"lang": "quc", "label": "Ximénez"},
    "recinos":             {"lang": "spa", "label": "Recinos 1947"},
    "tedlock":             {"lang": "eng", "label": "Tedlock 1983"},
}
LANG_LABELS = {"quc": "K'iche'", "spa": "Spanish", "eng": "English"}

# ── Helper functions ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_tokens(src_id: str, token_path: str) -> pd.DataFrame:
    TOKEN = pd.read_csv(token_path)
    idx_offset = TOKEN.columns.to_list().index("token_str")
    ohco = TOKEN.columns.to_list()[:idx_offset]
    return TOKEN.set_index(ohco)

@st.cache_data(show_spinner="Running model…")
def run_model(src_id, token_path, chunk_size, overlap_int, min_df, max_df, n_topics, n_top_words):
    TOKEN = load_tokens(src_id, token_path)
    # Chunk
    tokens = TOKEN["term_str"].dropna().to_list()
    chunks, step = [], max(1, chunk_size - overlap_int)
    for i in range(0, len(tokens), step):
        chunk = " ".join(tokens[i : i + chunk_size])
        if len(chunk.split()) >= 50:
            chunks.append(chunk)
    if len(chunks) < 2:
        return None, None, None, len(TOKEN)
    # TF-IDF
    try:
        vec = TfidfVectorizer(lowercase=True, max_df=max_df, min_df=min_df,
                              strip_accents=None, norm="l2")
        X = vec.fit_transform(chunks)
    except ValueError:
        return None, None, None, len(TOKEN)
    # NMF
    nmf = NMF(n_components=n_topics, init="nndsvd", max_iter=500)
    THETA = pd.DataFrame(nmf.fit_transform(X))
    THETA.index.name, THETA.columns.name = "chunk_id", "topic_id"
    words = vec.get_feature_names_out()
    TOPICS = pd.DataFrame({
        f"Topic {i}": [words[j] for j in topic.argsort()[:-n_top_words-1:-1]]
        for i, topic in enumerate(nmf.components_)
    })
    return THETA, TOPICS, len(chunks), len(TOKEN)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title(site_title)

# ── Controls — compact inline row, like ipywidgets interact ───────────────────
src_ids = list(SOURCES_META.keys())
c = st.columns([2, 1.2, 1, 1, 1, 1, 1])

src_id     = c[0].selectbox("Source", src_ids,
                 index=src_ids.index("colop"),
                 format_func=lambda x: f"{SOURCES_META[x]['label']} ({LANG_LABELS[SOURCES_META[x]['lang']]})")
chunk_size = c[1].number_input("Chunk size", 100, 2000, 1000, step=50)
overlap    = c[2].number_input("Overlap", 0.0, 0.9, 0.9, step=0.05, format="%.2f")
min_df     = c[3].number_input("min_df", 1, 20, 5, step=1)
max_df     = c[4].number_input("max_df", 0.1, 1.0, 0.35, step=0.05, format="%.2f")
n_topics   = c[5].number_input("Topics", 2, 20, 8, step=1)
n_top_words= c[6].number_input("Top words", 3, 15, 7, step=1)

overlap_int = int(overlap * chunk_size)

st.divider()

# ── File resolution ───────────────────────────────────────────────────────────
token_path = find_token_file(src_id)

if token_path is None:
    tried = [
        os.path.normpath(os.path.join(APP_DIR, "..", src_id, f"{src_id}-TOKEN.csv")),
        os.path.normpath(os.path.join(APP_DIR, src_id, f"{src_id}-TOKEN.csv")),
        os.path.normpath(os.path.join(APP_DIR, f"{src_id}-TOKEN.csv")),
    ]
    st.warning("**Token file not found for `" + src_id + "`.**\n\nLooked in:\n" +
               "\n".join(f"- `{p}`" for p in tried), icon="📂")
    uploaded = st.file_uploader(f"Upload `{src_id}-TOKEN.csv`", type="csv", key=f"upload_{src_id}")
    if uploaded is not None:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded.read()); tmp.flush()
        token_path = tmp.name
    else:
        st.stop()

# ── Run (auto, cached by params) ──────────────────────────────────────────────
THETA, TOPICS, n_chunks, n_tokens = run_model(
    src_id, token_path, chunk_size, overlap_int, min_df, max_df, n_topics, n_top_words
)

meta = SOURCES_META[src_id]

if THETA is None:
    st.warning("Model couldn't run — try adjusting chunk size, min_df, or max_df.")
    st.stop()

# ── Heatmap ───────────────────────────────────────────────────────────────────
st.caption(
    f"**{meta['label']}** ({LANG_LABELS[meta['lang']]}) · "
    f"{n_tokens:,} tokens · {n_chunks} chunks · {THETA.shape[1]} topics"
)
fig, ax = plt.subplots(figsize=(20, 4))
sns.heatmap(THETA.T, cmap="YlGnBu", ax=ax, linewidths=0,
            xticklabels=max(1, n_chunks // 40),
            yticklabels=[f"T{i}" for i in range(THETA.shape[1])])
ax.set_title(meta["label"], fontsize=18, fontweight="bold", pad=12)
ax.set_xlabel("Syntagm / Event (chunk index)", fontsize=13)
ax.set_ylabel("Paradigm / Structure (topic)", fontsize=13)
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ── Topic terms ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("Top Terms per Topic")
topic_display = TOPICS.copy()
topic_display.index = [f"Rank {i+1}" for i in range(len(TOPICS))]
st.dataframe(topic_display, use_container_width=True, height=min(40 + 35 * len(TOPICS), 500))

st.download_button("⬇  Download THETA", data=THETA.to_csv().encode(),
                   file_name=f"{src_id}_theta.csv", mime="text/csv")

# ── Bibliography ──────────────────────────────────────────────────────────────
# with st.expander("Bibliography"):
#     st.markdown(
#         "Recinos, A. (1947). *Popol Vuh: Las Antiguas Historias Del Quiché*. "
#         "Fondo de Cultura Económica. "
#         "[Google Books](https://books.google.com/books?hl=en&id=p9hpEAAAQBAJ)"
#     )