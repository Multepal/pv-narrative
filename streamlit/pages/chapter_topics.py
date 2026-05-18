"""
Chapter Topics — NMF topic model applied directly to Christenson chapters.

Applies the trained PHI matrix (topics × vocabulary) to each chapter as a
document, bypassing the chunk/chapter alignment problem.  Produces a
topics × chapters heatmap showing the activation of each NMF topic across
the narrative at chapter resolution.

Approach (from notebooks/ensemble/ensemble.ipynb, "Test model on K'iche' DOCs"):
  1. Build chapter-level term-count matrix B (vocab × chapters)
     restricted to PHI's vocabulary.
  2. Compute L2-normalised TF-IDF of B.
  3. Project: C = PHI @ TFIDF  →  topics × chapters activation matrix.
  4. Column-normalise C so each chapter's activations sum to 1.
"""

import os
import yaml
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

TOKEN_PATH = os.path.normpath(
    os.path.join(APP_DIR, "../../notebooks/christenson/christenson-TOKEN.csv")
)
PHI_PATH = os.path.normpath(
    os.path.join(APP_DIR, "../../notebooks/christenson/christenson-PHI.csv")
)
TOPIC_PATH = os.path.normpath(
    os.path.join(APP_DIR, "../../notebooks/christenson/christenson-TOPIC.csv")
)
CHAP_PATH = os.path.normpath(
    os.path.join(APP_DIR, "../../notebooks/christenson/christenson-CHAP-with-text.csv")
)

# User-proposed divisions (Roman numeral, label, first chapter)
USER_DIVISIONS = [
    ("I",    "First Creation",          1),
    ("II",   "7 Macaw",                 9),
    ("III",  "Father",                 15),
    ("IV",   "Mother",                 19),
    ("V",    "Two Boys Defeat Death",  22),
    ("VI",   "Second Creation",        41),
    ("VII",  "Religion",               48),
    ("VIII", "Politics",               65),
]


@st.cache_data(show_spinner="Applying topic model to chapters…")
def compute_chapter_topics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    TOKEN = pd.read_csv(TOKEN_PATH)
    PHI   = pd.read_csv(PHI_PATH, index_col=0)
    TOPIC = pd.read_csv(TOPIC_PATH)
    CHAP  = pd.read_csv(CHAP_PATH)[["chap_num", "chap_title"]]

    phi_vocab = PHI.columns.tolist()

    # Chapter-level term counts, restricted to PHI vocabulary
    tok_filtered = TOKEN[TOKEN["term_str"].isin(phi_vocab)]
    B = (
        tok_filtered.groupby(["chap_num", "term_str"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=phi_vocab, fill_value=0)
    )   # shape: chapters × vocab

    # TF-IDF (same formula as TestModel.compute_tfidf)
    TF  = B  # chapters × vocab
    DF  = (TF > 0).sum(axis=0)
    IDF = np.log2((len(TF) + 1) / (DF + 1) + 1)
    TFIDF = TF * IDF

    # L2 normalise row-wise
    l2 = np.sqrt((TFIDF ** 2).sum(axis=1)).replace(0, 1)
    TFIDF_norm = TFIDF.div(l2, axis=0)   # chapters × vocab

    # Project: C = PHI @ TFIDF.T  →  topics × chapters
    C = PHI.values @ TFIDF_norm.T.values   # (n_topics, n_chapters)
    C_df = pd.DataFrame(C, index=PHI.index, columns=B.index)

    # Column-normalise: each chapter's scores sum to 1
    C_norm = C_df.div(C_df.sum(axis=0).replace(0, 1), axis=1)

    return C_norm, TOPIC, CHAP


# ── Page ──────────────────────────────────────────────────────────────────────
st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)

st.title("Chapter Topics — NMF Applied at Chapter Resolution")
st.caption(
    "The trained NMF topic model (PHI) is applied directly to each of Christenson's 87 "
    "chapters as documents, with no chunking.  Each column sums to 1 — the heatmap shows "
    "topic activation share per chapter."
)

with st.expander("How this works", expanded=False):
    st.markdown("""
Normally the NMF model is fitted on 60 text *chunks* (fixed-size windows) and the
resulting THETA matrix gives topic proportions per chunk.  Here we bypass the
chunk/chapter alignment problem by applying the *vocabulary weights* (PHI, topics × vocab)
directly to chapters:

1. Build a chapter-level TF-IDF matrix using only PHI's 500-word vocabulary.
2. Project each chapter's TF-IDF vector through PHI: `C = PHI @ TFIDF.T`.
3. Column-normalise so each chapter's topic activations sum to 1.

This is the approach used in `notebooks/ensemble/ensemble.ipynb` ("Test model on K'iche' DOCs").
The result is a topics × chapters matrix that reveals dominant topics at chapter resolution,
without any chunk boundary artefacts.
""")

C_norm, TOPIC, CHAP = compute_chapter_topics()

topic_labels = dict(zip(TOPIC["topic_id"].astype(int), TOPIC["gloss"]))
chap_titles  = dict(zip(CHAP["chap_num"], CHAP["chap_title"]))

all_chaps   = sorted(C_norm.columns)
all_topics  = list(C_norm.index)
y_labels    = [topic_labels.get(t, f"T{t}") for t in all_topics]
x_labels    = [f"{c}" for c in all_chaps]

# Hover text
hover = []
for ti in all_topics:
    row = []
    for ch in all_chaps:
        val   = C_norm.loc[ti, ch]
        title = chap_titles.get(ch, "")
        row.append(
            f"<b>Ch. {ch}</b> {title}<br>"
            f"Topic {ti} ({topic_labels.get(ti,'?')}): {val:.1%}"
        )
    hover.append(row)

# ── Controls ──────────────────────────────────────────────────────────────────
with st.sidebar:
    show_divisions = st.checkbox("Show division lines", value=True)
    colorscale = st.selectbox(
        "Colour scale",
        ["YlOrRd", "Blues", "Viridis", "Plasma", "YlGnBu"],
        index=0,
    )
    st.divider()
    st.markdown("**Divisions (dashed lines)**")
    for rom, label, chap in USER_DIVISIONS:
        st.markdown(f"{rom}. {label} — ch. {chap}")

# ── Heatmap ───────────────────────────────────────────────────────────────────
fig = go.Figure(go.Heatmap(
    z=C_norm.values,
    x=x_labels,
    y=y_labels,
    customdata=hover,
    hovertemplate="%{customdata}<extra></extra>",
    colorscale=colorscale,
    showscale=True,
    xgap=1, ygap=2,
    hoverlabel=dict(align="left", font=dict(family="monospace", size=11)),
))

# Division lines
if show_divisions:
    chap_positions = {str(c): i for i, c in enumerate(all_chaps)}
    for rom, label, chap in USER_DIVISIONS[1:]:
        x_pos = chap_positions.get(str(chap), None)
        if x_pos is None:
            continue
        # line falls between x_pos-1 and x_pos (0-indexed)
        x_line = x_pos - 0.5
        fig.add_shape(
            type="line",
            x0=x_line, x1=x_line,
            y0=-0.5,   y1=len(all_topics) - 0.5,
            line=dict(color="white", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=x_line, y=len(all_topics) - 0.3,
            xref="x", yref="y",
            text=rom,
            showarrow=False,
            font=dict(color="white", size=11, family="Arial Black"),
            xanchor="center",
        )

row_h = cfg["layout"].get("heatmap_row_height_px", 40)
fig.update_layout(
    height=max(250, len(all_topics) * row_h + 80),
    margin=dict(l=100, r=60, t=20, b=60),
    xaxis=dict(
        title="Chapter (narrative order)",
        tickmode="array",
        tickvals=list(range(0, len(all_chaps), 5)),
        ticktext=[str(all_chaps[i]) for i in range(0, len(all_chaps), 5)],
        showgrid=False,
    ),
    yaxis=dict(
        autorange="reversed",
        categoryorder="array",
        categoryarray=y_labels,
        tickmode="array",
        tickvals=list(range(len(y_labels))),
        ticktext=y_labels,
    ),
)

st.plotly_chart(fig, width="stretch")

st.caption(
    "Rows = NMF topics · Columns = chapters 1–87 · "
    "Colour = normalised topic activation (per-chapter shares sum to 1) · "
    "Dashed white lines = your proposed division starts."
)

# ── Dominant topic per chapter ────────────────────────────────────────────────
st.subheader("Dominant Topic per Chapter", anchor=False)
st.caption("Chapter assigned to the topic with the highest activation score.")

dom = C_norm.idxmax(axis=0).rename("dominant_topic")
dom_label = dom.map(lambda t: topic_labels.get(t, f"T{t}"))
dom_score = C_norm.max(axis=0).rename("score")

summary = pd.DataFrame({
    "chapter":       all_chaps,
    "title":         [chap_titles.get(c, "") for c in all_chaps],
    "dominant_topic": dom.values,
    "topic_gloss":   dom_label.values,
    "score":         dom_score.round(3).values,
}).set_index("chapter")

# Find where dominant topic changes — these are the model's "natural" chapter boundaries
summary["new_topic"] = summary["dominant_topic"] != summary["dominant_topic"].shift()
boundaries = summary[summary["new_topic"] & (summary.index > 1)].index.tolist()
st.caption(f"Topic changes at chapters: {boundaries}")

st.dataframe(summary.drop(columns="new_topic"), width="stretch")
