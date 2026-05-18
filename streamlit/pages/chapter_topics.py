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
from scipy.stats import entropy as scipy_entropy

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

# ── Topic entropy per chapter ─────────────────────────────────────────────────
st.subheader("Topic Entropy & Dominance per Chapter", anchor=False)
st.caption(
    "**Entropy** (left axis, dark): Shannon entropy of the topic distribution. "
    "High = multiple topics co-active. "
    "**Dominance** (right axis, colour): share held by the single strongest topic. "
    "Low dominance = structurally ambiguous; high dominance = structurally pure."
)

H = C_norm.apply(
    lambda col: float(scipy_entropy(col.clip(lower=1e-9))), axis=0
)
H_max    = float(np.log(len(all_topics)))
dom_score = C_norm.max(axis=0)   # highest single-topic share per chapter

fig_ent = go.Figure()

# Shaded background bands per division
div_starts = [chap for _, _, chap in USER_DIVISIONS]
div_ends   = div_starts[1:] + [all_chaps[-1] + 1]
band_colors = ["rgba(75,171,201,0.08)", "rgba(224,108,75,0.08)",
               "rgba(142,107,191,0.08)", "rgba(107,155,210,0.08)",
               "rgba(142,107,191,0.08)", "rgba(75,171,201,0.08)",
               "rgba(90,171,107,0.08)",  "rgba(201,168,76,0.08)"]
for (rom, label, cs), ce, bc in zip(USER_DIVISIONS, div_ends, band_colors):
    fig_ent.add_vrect(
        x0=cs - 0.5, x1=ce - 0.5,
        fillcolor=bc, line_width=0,
        annotation_text=rom,
        annotation_position="top left",
        annotation_font=dict(size=10, color="#777"),
    )

# Max-entropy reference
fig_ent.add_hline(
    y=H_max, line_dash="dot", line_color="#ccc", line_width=1,
    annotation_text=f"max H={H_max:.2f}",
    annotation_position="right",
    annotation_font=dict(size=8, color="#aaa"),
)

hover_text = [
    f"<b>Ch. {c}</b> {chap_titles.get(c,'')}<br>"
    f"entropy: {H[c]:.3f}<br>"
    f"dominance: {dom_score[c]:.3f}<br>"
    f"dominant topic: {topic_labels.get(int(C_norm[c].idxmax()),'?')}"
    for c in all_chaps
]

# Entropy line (left y-axis)
fig_ent.add_trace(go.Scatter(
    x=all_chaps, y=H.values,
    mode="lines+markers",
    name="Entropy",
    line=dict(color="#333", width=1.8),
    marker=dict(size=5, color="#333"),
    customdata=hover_text,
    hovertemplate="%{customdata}<extra></extra>",
    yaxis="y",
))

# Dominance bars (right y-axis, coloured by dominant topic index)
_topic_palette = ["#4babc9", "#e06c4b", "#8e6bbf", "#6b9bd2", "#5aab6b", "#c9a84c"]
dom_colors = [
    _topic_palette[int(C_norm[c].idxmax()) % len(_topic_palette)]
    for c in all_chaps
]

fig_ent.add_trace(go.Bar(
    x=all_chaps, y=dom_score.values,
    name="Dominance",
    marker_color=dom_colors, opacity=0.35,
    customdata=hover_text,
    hovertemplate="%{customdata}<extra></extra>",
    yaxis="y2",
))

# Division lines
if show_divisions:
    for rom, label, chap in USER_DIVISIONS[1:]:
        fig_ent.add_vline(
            x=chap - 0.5,
            line_dash="dash", line_color="#888", line_width=1.2,
        )

fig_ent.update_layout(
    height=300,
    margin=dict(l=60, r=80, t=20, b=50),
    plot_bgcolor="white",
    barmode="overlay",
    xaxis=dict(
        title="Chapter (narrative order)",
        showgrid=True, gridcolor="#eee", zeroline=False,
        range=[0.5, all_chaps[-1] + 0.5],
    ),
    yaxis=dict(
        title="Shannon entropy (nats)",
        showgrid=True, gridcolor="#eee", zeroline=False,
        range=[1.3, H_max * 1.08],
        side="left",
    ),
    yaxis2=dict(
        title="Dominance (max topic share)",
        overlaying="y", side="right",
        range=[0, 0.65],
        showgrid=False,
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    hoverlabel=dict(align="left", font=dict(family="monospace", size=11)),
)

st.plotly_chart(fig_ent, width="stretch")

col_top, col_bot = st.columns(2)
with col_top:
    st.caption("**Highest entropy** (most topic-mixed)")
    top_h = H.nlargest(10).reset_index()
    top_h.columns = ["chapter", "entropy"]
    top_h["title"] = top_h["chapter"].map(chap_titles)
    top_h["dom_topic"] = top_h["chapter"].map(
        lambda c: topic_labels.get(int(C_norm[c].idxmax()), "?"))
    top_h["entropy"] = top_h["entropy"].round(3)
    st.dataframe(top_h.set_index("chapter").sort_index(), width="stretch")
with col_bot:
    st.caption("**Lowest dominance** (no single topic wins)")
    bot_d = dom_score.nsmallest(10).reset_index()
    bot_d.columns = ["chapter", "dominance"]
    bot_d["title"] = bot_d["chapter"].map(chap_titles)
    bot_d["dom_topic"] = bot_d["chapter"].map(
        lambda c: topic_labels.get(int(C_norm[c].idxmax()), "?"))
    bot_d["dominance"] = bot_d["dominance"].round(3)
    st.dataframe(bot_d.set_index("chapter").sort_index(), width="stretch")

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
