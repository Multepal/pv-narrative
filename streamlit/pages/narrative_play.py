"""
Narrative Play — Chapter / Cluster Boundary Alignment

Shows the mismatch between HAC cluster boundaries (computed at chunk level)
and chapter boundaries, revealing where an analyst has interpretive "play"
when mapping clusters to narrative divisions.

Two horizontal bands share a proportional x-axis (token position):
  Top:    33 chunks coloured by cluster A–F
  Bottom: 87 chapters coloured by their per-chapter cluster composition
          (split rectangles for chapters that straddle a cluster boundary)

Two sets of vertical division markers:
  Dashed:  user-proposed boundaries (Christenson chapters)
  Solid:   model majority-vote boundaries
  Gap between them = the interpretive play
"""

import os
import yaml
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, fcluster

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

TOKEN_PATH = os.path.normpath(
    os.path.join(APP_DIR, "../../notebooks/christenson/christenson-TOKEN.csv")
)
CHAP_PATH = os.path.normpath(
    os.path.join(APP_DIR, "../../notebooks/christenson/christenson-CHAP-with-text.csv")
)

# Cluster colours (assigned by first appearance in narrative)
CLUSTER_COLORS = ["#4babc9", "#e06c4b", "#8e6bbf", "#6b9bd2", "#5aab6b", "#c9a84c"]

# User-proposed divisions (Roman numeral, label, first chapter of this division)
USER_DIVISIONS = [
    ("I",    "First Creation",             1),
    ("II",   "7 Macaw",                    9),
    ("III",  "Father",                    15),
    ("IV",   "Mother",                    19),
    ("V",    "Two Boys Defeat Death",      22),
    ("VI",   "Second Creation",           41),
    ("VII",  "Religion",                  48),
    ("VIII", "Politics",                  65),
]

# Model majority-vote divisions (first chapter of each division)
MODEL_DIVISIONS = [
    ("I",    "First Creation",             1),
    ("II",   "7 Macaw",                    8),
    ("III",  "Father",                    15),
    ("IV",   "Mother",                    21),
    ("V",    "Two Boys Defeat Death",      25),
    ("VI",   "Second Creation",           39),
    ("VII",  "Religion",                  48),
    ("VIII", "Politics",                  65),
]


@st.cache_data(show_spinner="Computing cluster assignments…")
def compute_data(n_chunks: int, k: int, min_df: int, max_df: float):
    TOKEN = pd.read_csv(TOKEN_PATH)
    CHAP  = pd.read_csv(CHAP_PATH)[["chap_num", "chap_title"]]

    TOKEN["chunk_num"] = pd.cut(TOKEN.index, n_chunks, labels=range(n_chunks)).astype(int)
    chunk_texts = (
        TOKEN.groupby("chunk_num")["term_str"]
        .apply(lambda x: " ".join(x.dropna()))
        .reindex(range(n_chunks), fill_value="")
    )
    X = TfidfVectorizer(norm="l2", min_df=min_df, max_df=max_df).fit_transform(chunk_texts).toarray()
    raw_labels = fcluster(linkage(X, method="ward"), k, criterion="maxclust")

    seen, letter_map = [], {}
    for l in raw_labels:
        if l not in seen:
            seen.append(l)
            letter_map[l] = chr(65 + len(seen) - 1)

    TOKEN["cluster"] = TOKEN["chunk_num"].map(lambda c: letter_map[raw_labels[c]])

    # Per-chapter cluster fractions
    chap_cluster = TOKEN.groupby(["chap_num", "cluster"]).size().unstack(fill_value=0)
    chap_frac = chap_cluster.div(chap_cluster.sum(axis=1), axis=0)

    # Per-chapter token counts and cumulative positions
    chap_sizes = TOKEN.groupby("chap_num").size().rename("n_tokens")
    chap_sizes_df = chap_sizes.reset_index()
    chap_sizes_df["cum_start"] = chap_sizes_df["n_tokens"].cumsum().shift(fill_value=0)
    chap_sizes_df["cum_end"]   = chap_sizes_df["n_tokens"].cumsum()
    chap_sizes_df = chap_sizes_df.set_index("chap_num")

    # Per-chunk cumulative positions
    chunk_sizes = TOKEN.groupby("chunk_num").size().rename("n_tokens")
    chunk_df = chunk_sizes.reset_index()
    chunk_df["cum_start"] = chunk_df["n_tokens"].cumsum().shift(fill_value=0)
    chunk_df["cum_end"]   = chunk_df["n_tokens"].cumsum()
    chunk_df["cluster"]   = [letter_map[l] for l in raw_labels]
    chunk_df = chunk_df.set_index("chunk_num")

    total = TOKEN.index.max() + 1

    # Map each chapter to its token start position (for division lines)
    chap_tok_start = (
        TOKEN.groupby("chap_num").apply(lambda g: g.index.min())
        .rename("tok_start")
    )

    return (
        TOKEN, CHAP, chap_frac, chap_sizes_df, chunk_df, total,
        chap_tok_start, letter_map, raw_labels,
    )


# ── Page layout ───────────────────────────────────────────────────────────────
st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)

st.title("Narrative Play — Chapter/Cluster Alignment")
st.caption(
    "Where do HAC cluster boundaries (chunk-level) land relative to chapter boundaries? "
    "The gap between **dashed** (your proposed) and **solid** (majority-vote model) division "
    "lines is the interpretive play."
)

with st.sidebar:
    n_chunks = st.number_input("Chunks", 10, 60, 33, 1)
    k        = st.number_input("k clusters", 2, 12, 6, 1)
    min_df   = st.number_input("min_df", 1, 20, 5, 1)
    max_df   = st.number_input("max_df", 0.1, 1.0, 0.5, 0.05)
    st.divider()
    st.markdown(
        "**Dashed lines** — your proposed boundaries  \n"
        "**Solid lines** — majority-vote model boundaries  \n"
        "**Split bars** — chapters that straddle a cluster boundary"
    )

(TOKEN, CHAP, chap_frac, chap_sizes_df, chunk_df, total,
 chap_tok_start, letter_map, raw_labels) = compute_data(
    int(n_chunks), int(k), int(min_df), float(max_df)
)

chap_titles = dict(zip(CHAP["chap_num"], CHAP["chap_title"]))
all_clusters = sorted(set(letter_map.values()))
color_map = {cl: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, cl in enumerate(all_clusters)}

shapes = []
annotations = []

# ── Band 1: chunk-level cluster sequence (y = 1.15 to 2.0) ───────────────────
for chunk_id, row in chunk_df.iterrows():
    x0 = row["cum_start"] / total
    x1 = row["cum_end"]   / total
    cl = row["cluster"]
    shapes.append(dict(
        type="rect", x0=x0, x1=x1, y0=1.12, y1=1.98,
        fillcolor=color_map[cl], opacity=0.85,
        line=dict(color="white", width=0.5),
    ))

# Cluster letter labels in band 1 (one per run)
prev_cl = None
for chunk_id, row in chunk_df.iterrows():
    cl = row["cluster"]
    if cl != prev_cl:
        x_mid = ((row["cum_start"] + row["cum_end"]) / 2) / total
        annotations.append(dict(
            x=x_mid, y=1.55, xref="x", yref="y",
            text=f"<b>{cl}</b>", showarrow=False,
            font=dict(color="white", size=13),
        ))
        prev_cl = cl

# ── Band 2: chapter-level cluster composition (y = 0 to 0.95) ────────────────
for chap_num, frac_row in chap_frac.iterrows():
    if chap_num not in chap_sizes_df.index:
        continue
    sz  = chap_sizes_df.loc[chap_num]
    x0c = sz["cum_start"] / total
    x1c = sz["cum_end"]   / total
    width = x1c - x0c

    nonzero = frac_row[frac_row > 0].sort_values(ascending=False)
    cur_x = x0c
    for cl, frac in nonzero.items():
        seg_w = width * frac
        shapes.append(dict(
            type="rect", x0=cur_x, x1=cur_x + seg_w, y0=0.02, y1=0.95,
            fillcolor=color_map.get(cl, "#aaa"), opacity=0.85,
            line=dict(color="white", width=0.5),
        ))
        cur_x += seg_w

# ── Division boundary lines ───────────────────────────────────────────────────
def _chap_x(chap_num):
    tok = chap_tok_start.get(chap_num, None)
    return tok / total if tok is not None else None

# User boundaries (dashed black)
for rom, label, chap in USER_DIVISIONS[1:]:
    x = _chap_x(chap)
    if x is None:
        continue
    shapes.append(dict(
        type="line", x0=x, x1=x, y0=-0.08, y1=2.1,
        line=dict(color="#222", width=2, dash="dash"),
    ))
    annotations.append(dict(
        x=x, y=2.12, xref="x", yref="y",
        text=rom, showarrow=False,
        font=dict(color="#222", size=11),
        xanchor="center",
    ))

# Model boundaries (solid grey)
for rom, label, chap in MODEL_DIVISIONS[1:]:
    x = _chap_x(chap)
    if x is None:
        continue
    shapes.append(dict(
        type="line", x0=x, x1=x, y0=-0.08, y1=2.1,
        line=dict(color="#888", width=1.5, dash="solid"),
    ))

# ── Chapter boundary ticks (thin lines at bottom of band 2) ──────────────────
for chap_num, sz in chap_sizes_df.iterrows():
    x = sz["cum_start"] / total
    shapes.append(dict(
        type="line", x0=x, x1=x, y0=0.0, y1=0.05,
        line=dict(color="white", width=0.5),
    ))

# ── Hover trace (invisible scatter over band 2) for chapter tooltips ─────────
hover_x, hover_text = [], []
for chap_num, sz in chap_sizes_df.iterrows():
    x_mid = (sz["cum_start"] + sz["cum_end"]) / 2 / total
    fracs = chap_frac.loc[chap_num] if chap_num in chap_frac.index else pd.Series(dtype=float)
    nonzero = fracs[fracs > 0].sort_values(ascending=False)
    frac_str = "  ".join(f"{cl}: {v*100:.0f}%" for cl, v in nonzero.items())
    title = chap_titles.get(chap_num, "")
    hover_x.append(x_mid)
    hover_text.append(f"<b>Ch. {chap_num}</b> {title}<br>{frac_str}")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hover_x, y=[0.48] * len(hover_x),
    mode="markers",
    marker=dict(opacity=0, size=8),
    hovertemplate="%{customdata}<extra></extra>",
    customdata=hover_text,
    showlegend=False,
))

# Legend traces (one per cluster)
for cl in all_clusters:
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color=color_map[cl], size=12, symbol="square"),
        name=f"Cluster {cl}",
        showlegend=True,
    ))

fig.update_layout(
    shapes=shapes,
    annotations=annotations,
    height=280,
    margin=dict(l=40, r=20, t=40, b=50),
    plot_bgcolor="white",
    xaxis=dict(
        title="Narrative position (proportional to token count)",
        showgrid=False, zeroline=False,
        range=[0, 1],
        tickformat=".0%",
    ),
    yaxis=dict(
        showticklabels=False, showgrid=False, zeroline=False,
        range=[-0.15, 2.3],
    ),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
        font=dict(size=11),
    ),
    hoverlabel=dict(align="left", font=dict(family="monospace", size=11)),
)

# Band labels on y-axis
fig.add_annotation(x=-0.01, y=1.55, xref="paper", yref="y",
    text="Chunks", showarrow=False, xanchor="right",
    font=dict(size=10, color="#555"))
fig.add_annotation(x=-0.01, y=0.48, xref="paper", yref="y",
    text="Chapters", showarrow=False, xanchor="right",
    font=dict(size=10, color="#555"))

st.plotly_chart(fig, width="stretch")

st.caption(
    "Top band = 33-chunk HAC cluster sequence · "
    "Bottom band = per-chapter cluster composition (split bars = chapters straddling a boundary) · "
    "**Dashed** = your proposed division start · **Solid** = majority-vote model boundary · "
    "Hover over bottom band for chapter details."
)

# ── Play detail table ─────────────────────────────────────────────────────────
st.subheader("The Play at Each Transition", anchor=False)

play_data = []
transitions = [
    ("A→B", 8,  "The Fall of the Effigies of Carved Wood",  9,  91, "A", "B", "I → II"),
    ("B→C", 14, "The Defeat of Cabracan",                  94,   6, "B", "C", "II → III"),
    ("C→D", 20, "The Ascent of Lady Blood from Xibalba",   64,  36, "C", "D", "III → IV"),
    ("D→C", 24, "Hunahpu and Xbalanque in the Maizefield", 86,  14, "D", "C", "IV → V"),
    ("C→A", 39, "The Miraculous Maize of H&X",             15,  85, "C", "A", "V → VI"),
    ("A→E", 47, "The Creation of the Mothers",             55,  45, "A", "E", "VI → VII"),
    ("E→F", 64, "The Deaths of the Four Progenitors",      93,   7, "E", "F", "VII → VIII"),
]

rows = []
for trans, chap, title, pct_before, pct_after, cl_before, cl_after, boundary in play_data if play_data else transitions:
    user_div = next((r for r, l, c in USER_DIVISIONS[1:] if c == chap + 1), None)
    model_div = next((r for r, l, c in MODEL_DIVISIONS[1:] if c == chap), None)
    majority = cl_before if pct_before >= 50 else cl_after
    side = f"keep in ←{cl_before}" if pct_before >= 50 else f"move to →{cl_after}"
    rows.append({
        "Transition": trans,
        "Ch.": chap,
        "Title": title,
        f"% {cl_before}": f"{pct_before}%",
        f"% {cl_after}": f"{pct_after}%",
        "Majority": majority,
        "Boundary": boundary,
        "Ambiguous": "⚠" if abs(pct_before - pct_after) < 30 else "",
    })

st.dataframe(pd.DataFrame(rows).set_index("Transition"), width="stretch")

with st.expander("Interpretation", expanded=False):
    st.markdown("""
Each row is a chapter that straddles a cluster transition — it has tokens in two adjacent
cluster zones.  **% before / % after** show how the chapter's text is distributed across
the boundary.  A high majority (> 80%) means the placement is essentially forced; the only
genuinely ambiguous case is ch. 47 (*The Creation of the Mothers*, 55 / 45).

The **play** — the gap between dashed and solid division lines on the chart — is the set of
chapters where your interpretive judgement overrides the strict majority vote.  These are
the chapters that sit structurally at the seam between two paradigmatic elements, and their
placement articulates your reading of where one episode ends and another begins.
""")
