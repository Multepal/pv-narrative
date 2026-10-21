"""
Narrative Play — Chapter / Cluster Boundary Alignment

Shows the mismatch between HAC cluster boundaries (computed at chunk level)
and chapter boundaries, revealing where an analyst has interpretive "play"
when mapping clusters to narrative divisions.

Two horizontal bands share a proportional x-axis (token position):
  Top:    n_chunks chunks (config reporting setting) coloured by cluster A–F
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
from utils import find_token_file, load_tokens
from grid_search_boundary_core import cluster_labels
from divisions import author_divisions, chapter_titles, DIV_CFG, DOC_TABLES

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

# Cluster colours (assigned by first appearance in narrative)
CLUSTER_COLORS = ["#4babc9", "#e06c4b", "#8e6bbf", "#6b9bd2", "#5aab6b", "#c9a84c"]

# User-proposed divisions (Roman numeral, label, first chapter of this division), from config
USER_DIVISIONS = [(rom, label, chap) for rom, label, chap, _ in author_divisions()]

def chunk_transitions(TOKEN, chunk_df, chap_titles):
    """Every chunk-level cluster transition, located in the chapter that contains it: how the
    chapter's tokens split across the boundary, which side the majority vote puts the chapter
    on, and the division start (chapter) that vote implies."""
    rows = []
    prev = None
    for chunk_id, row in chunk_df.iterrows():
        if prev is not None and row["cluster"] != prev["cluster"]:
            first_tok = int(TOKEN.index[TOKEN["chunk_num"] == chunk_id].min())
            chap = int(TOKEN.loc[first_tok, "chap_num"])
            in_chap = TOKEN[TOKEN["chap_num"] == chap]
            before = int((in_chap.index < first_tok).sum())
            pct_before = round(100 * before / len(in_chap))
            model_start = chap + 1 if pct_before >= 50 else chap
            rows.append({
                "transition": f"{prev['cluster']}→{row['cluster']}", "chapter": chap,
                "title": chap_titles.get(chap, ""), "pct_before": pct_before,
                "pct_after": 100 - pct_before, "before": prev["cluster"], "after": row["cluster"],
                "model_start": model_start,
            })
        prev = row
    return rows


SRC_ID = DIV_CFG["chapter_editions"][0]      # Christenson's K'iche' text (Christenson chapters)


@st.cache_data(show_spinner="Computing cluster assignments…")
def compute_data(n_chunks: int, k: int, min_df: int, max_df: float):
    """Chunk clusters from the app's pipeline (TF-IDF → Ward, cut at k, as on Clustering
    (TF-IDF)), mapped onto Christenson's chapters."""
    tp = find_token_file(SRC_ID)
    TOKEN = load_tokens(SRC_ID, tp).reset_index()
    DOCMAP = pd.read_csv(os.path.join(DOC_TABLES, f"{SRC_ID}-DOCMAP.csv"))[["doc_id", "chap_num"]]
    TOKEN = TOKEN.merge(DOCMAP, on="doc_id", how="left")
    CHAP = pd.DataFrame({"chap_num": sorted(TOKEN["chap_num"].dropna().unique().astype(int))})
    CHAP["chap_title"] = CHAP["chap_num"].map(chapter_titles())

    TOKEN["chunk_num"] = pd.cut(TOKEN.index, n_chunks, labels=range(n_chunks)).astype(int)
    raw_labels = np.asarray(cluster_labels("TF-IDF", SRC_ID, tp, n_chunks, min_df, max_df, k, (1, 1)))

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
    _ctl     = cfg["controls"]
    n_chunks = st.number_input("Chunks", 10, 60, _ctl["n_chunks"]["default"], 1)
    k        = st.number_input("k clusters", 2, 12, _ctl["n_clusters"]["default"], 1)
    min_df   = st.number_input("min_df", 1, 20, _ctl["min_df"]["default"], 1)
    max_df   = st.number_input("max_df", 0.1, 1.0, _ctl["max_df"]["default"], 0.05)
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
TRANSITIONS = chunk_transitions(TOKEN, chunk_df, chap_titles)
# Majority-vote model divisions: each straddling chapter goes to the side holding most of it
MODEL_DIVISIONS = [(None, None, 1)] + [(None, None, t["model_start"]) for t in TRANSITIONS]
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
    f"Top band = {n_chunks}-chunk HAC cluster sequence · "
    "Bottom band = per-chapter cluster composition (split bars = chapters straddling a boundary) · "
    "**Dashed** = your proposed division start · **Solid** = majority-vote model boundary · "
    "Hover over bottom band for chapter details."
)

# ── Play detail table ─────────────────────────────────────────────────────────
st.subheader("The Play at Each Transition", anchor=False)

def _division_label(chap):
    """User-division transition nearest a model division start, e.g. "III → IV"."""
    starts = USER_DIVISIONS[1:]
    i = min(range(len(starts)), key=lambda j: abs(starts[j][2] - chap))
    return f"{USER_DIVISIONS[i][0]} → {starts[i][0]}"


rows = []
for t in TRANSITIONS:
    rows.append({
        "Transition": t["transition"],
        "Ch.": t["chapter"],
        "Title": t["title"],
        "% before": f"{t['pct_before']}% {t['before']}",
        "% after": f"{t['pct_after']}% {t['after']}",
        "Majority": t["before"] if t["pct_before"] >= 50 else t["after"],
        "Model division starts": t["model_start"],
        "Nearest author boundary": _division_label(t["model_start"]),
        "Ambiguous": "⚠" if abs(t["pct_before"] - t["pct_after"]) < 30 else "",
    })

st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
_ambiguous = [f"ch. {t['chapter']} (*{t['title']}*, {t['pct_before']} / {t['pct_after']})"
              for t in TRANSITIONS if abs(t["pct_before"] - t["pct_after"]) < 30]

with st.expander("Interpretation", expanded=False):
    st.markdown(f"""
Each row is a chapter that straddles a cluster transition — it has tokens in two adjacent
cluster zones.  **% before / % after** show how the chapter's text is distributed across
the boundary.  A high majority (> 80%) means the placement is essentially forced; a split
within 30 points of even is ambiguous{(': ' + '; '.join(_ambiguous)) if _ambiguous else ' (none at this setting)'}.

The **play** — the gap between dashed and solid division lines on the chart — is the set of
chapters where your interpretive judgement overrides the strict majority vote.  These are
the chapters that sit structurally at the seam between two paradigmatic elements, and their
placement articulates your reading of where one episode ends and another begins.
""")
