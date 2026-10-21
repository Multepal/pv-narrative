"""
Division Validation — do the model's narrative boundaries fall where editors divide the text?

Scores the boundaries of the chosen segmentation (cluster label changes between chunks)
against independent editorial part divisions (Brasseur, Recinos, Edmonson, Tedlock, Colop,
Bazzett, Sotelo), the author's 8 interpretive divisions, and Christenson's chapters, each
against a cyclic-shift chance baseline. Also places the model among the editors: how well
it agrees with them compared with how well they agree with each other.
"""

import os
import yaml
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from toc import render_toc
from utils import find_token_file, cluster_string, tfidf_settings
from grid_search_boundary_core import cluster_labels, get_boundaries, boundary_tol, N_COMPONENTS
from divisions import (reference_sets, reference_starts, score_against, pairwise_f1,
                       consensus_boundaries, chapter_titles, DIV_CFG)

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
LANG_LABELS  = cfg["languages"]
_c           = cfg["controls"]
_v           = cfg["visualization"]
EDITORS      = list(DIV_CFG["editors"])
AUTHOR       = "Author's 8 divisions"
CHAPTERS     = "Christenson's chapters"

METHOD_LABELS = {
    "TF-IDF":     "TF-IDF → Ward HAC",
    "LSA":        f"TF-IDF → PCA/LSA (centered, n={N_COMPONENTS}) → Ward HAC",
    "Cosine-Sim": "TF-IDF → cosine-similarity matrix → Ward HAC",
    "NMF":        "TF-IDF → NMF → argmax",
}

st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)

METHOD_TEXT = """
**Placing the references.** Every reference is a list of Christenson chapter numbers where a
division starts. In Christenson's K'iche' text and his English prose, each chapter's start is
an exact token position. Other editions don't carry his chapter numbers, so his K'iche'
positions stand in by relative position; this assumes editions stay roughly proportional,
the same assumption the chunking makes.

**Recall, precision, F1.** A reference boundary is recovered if a model boundary lies within
the tolerance. Model boundaries sit on chunk edges, reference starts anywhere in the text: with
tolerance ± t chunks, a reference start counts if its *nearest chunk edge* is within t chunks of
a model boundary, i.e. within t + ½ chunks of raw distance. *Recall* = share of reference boundaries recovered; *precision* = share of model
boundaries that fall near some reference boundary; **F1 balances the two and is the test.**
Recall alone is easy to get by chance here: about 7 model boundaries, each with a window of
± (t + ½) chunks, cover much of the text, so a rotated sequence often recovers all 3–4 of an
editor's part starts too. F1 also charges the model for boundaries the editor doesn't have,
and that is what separates real agreement from chance. Recall still says *which* of an
editor's divisions the model finds.

**Chance.** The model's label sequence is rotated by every offset beyond the tolerance and
rescored against the fixed reference. Excess = observed − 95th percentile of the rotated
scores; p = share of rotations scoring at least as well. With ~30 rotations, the smallest
attainable p is about 0.03.

**Independence.** The editors' parts were made without these models, so they are the external
test. The author's divisions may have been informed by the models; report them separately.

**Chapters.** Christenson's 87 chapter starts are closer together than the tolerance window,
so any boundary lands near one and chance precision is already 1. Chapter-level validation
needs a finer resolution than equal chunks (e.g. sentence-level boundaries).
"""


def boundary_letters(labels):
    """Run segments of a label sequence as (start_idx, end_idx_exclusive, letter)."""
    s = cluster_string(labels)
    runs, start = [], 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[start]:
            runs.append((start, i, s[start]))
            start = i
    return runs


def render_timeline(labels, refs, chap_pos, tol, k):
    """Model segments on top, reference division starts below, model boundaries as dotted lines."""
    n = len(labels)
    model_row = f"Model (k = {k})"
    rows = [model_row, AUTHOR, *EDITORS, CHAPTERS]
    pos_to_chap = {round(v, 6): c for c, v in chap_pos.items()}
    titles = chapter_titles()
    palette = px.colors.qualitative.Plotly
    letter_color = {}

    fig = go.Figure()
    for b in get_boundaries(labels):
        fig.add_vrect(x0=b - tol, x1=b + tol, fillcolor="gray", opacity=0.08, line_width=0)
        fig.add_vline(x=b, line_dash="dot", line_color="gray", line_width=1)
    for a, z, letter in boundary_letters(labels):
        color = letter_color.setdefault(letter, palette[len(letter_color) % len(palette)])
        fig.add_trace(go.Scatter(
            x=[a / n, z / n], y=[model_row, model_row], mode="lines",
            line=dict(color=color, width=22), showlegend=False,
            hovertemplate=f"Cluster {letter}<br>chunks {a}–{z - 1}<extra></extra>",
        ))
        fig.add_annotation(x=(a + z) / (2 * n), y=model_row, text=f"<b>{letter}</b>",
                           showarrow=False, font=dict(color="white", size=12))
    for row in rows[1:]:
        xs = refs[row]
        chaps = [pos_to_chap.get(round(x, 6), "?") for x in xs]
        fig.add_trace(go.Scatter(
            x=xs, y=[row] * len(xs), mode="markers", showlegend=False,
            marker=dict(symbol="line-ns-open", size=8 if row == CHAPTERS else 18,
                        line=dict(width=1 if row == CHAPTERS else 3,
                                  color="#BBBBBB" if row == CHAPTERS else
                                  ("#EF553B" if row == AUTHOR else "#1f77b4"))),
            customdata=[[c, titles.get(c, "")] for c in chaps],
            hovertemplate=f"{row}<br>chapter %{{customdata[0]}}: %{{customdata[1]}}"
                          "<br>position %{x:.3f}<extra></extra>",
        ))
    fig.update_layout(
        height=60 + 34 * len(rows), margin=dict(l=10, r=20, t=10, b=40), plot_bgcolor="white",
        xaxis=dict(title="Position in text (share of tokens)", range=[0, 1], showgrid=False,
                   zeroline=False),
        yaxis=dict(categoryorder="array", categoryarray=rows, autorange="reversed",
                   showgrid=False, zeroline=False),
    )
    st.plotly_chart(fig, width="stretch")


def score_table(labels, refs):
    rows = []
    for name, ref in refs.items():
        s = score_against(labels, ref, tol)
        rows.append({
            "Reference": name, "Independent": "—" if name == AUTHOR else "yes",
            "Ref. boundaries": s["n_ref"], "Model boundaries": s["n_pred"],
            "Recall": s["recall"], "Recall chance 95th": s["recall_chance_p95"], "Recall p": s["recall_p"],
            "Precision": s["precision"],
            "F1": s["f1"], "F1 chance 95th": s["f1_chance_p95"], "F1 excess": s["f1_excess"],
            "F1 p": s["f1_p"],
        })
    return pd.DataFrame(rows)


# ── Controls ──────────────────────────────────────────────────────────────────
st.title("Division Validation — Popol Wuj")
render_toc([
    ("Timeline",                "timeline"),
    ("Scores",                  "scores"),
    ("Editorial Consensus",     "consensus"),
    ("The Model Among the Editors", "among-editors"),
    ("All Editions",            "all-editions"),
])
st.caption(
    "Do the boundaries of the chosen segmentation fall where editors divide the text? "
    "Editors' part divisions are the independent test; the author's divisions are reported "
    "separately."
)

src_ids = list(SOURCES_META.keys())
_default_src = DIV_CFG["chapter_editions"][0]
cols = st.columns([3, 3, 2])
src_id = cols[0].selectbox(
    "Edition", src_ids, index=src_ids.index(_default_src),
    format_func=lambda x: f"{SOURCES_META[x]['label']} ({LANG_LABELS[SOURCES_META[x]['lang']]})")
method = cols[1].selectbox("Method", list(METHOD_LABELS), format_func=METHOD_LABELS.get)
_tc = _c["boundary_tol_chunks"]
tol_chunks = cols[2].slider("Tolerance (± chunks)", _tc["min"], _tc["max"], _tc["default"], _tc["step"])

_set, _from_choose_k = tfidf_settings(_c)
n_chunks, k = int(_set["n_chunks"]), int(_set["k"])
ngram = (int(_set["ngram_min"]), int(_set["ngram_max"]))
tol = boundary_tol(n_chunks, tol_chunks)
st.caption(
    f"Setting: n_chunks = {n_chunks}, min_df = {_set['min_df']}, max_df = {float(_set['max_df']):.2f}, "
    f"k = {k} · {'from Choose k (this session)' if _from_choose_k else 'reporting setting from config'}"
    f" · tolerance ± {tol_chunks} chunk{'s' if tol_chunks != 1 else ''} (nearest chunk edge; "
    f"± {tol:.3f} of the text)."
)
st.divider()


def labels_for(src):
    return cluster_labels(method, src, find_token_file(src), n_chunks, int(_set["min_df"]),
                          float(_set["max_df"]), k, ngram)


labels = labels_for(src_id)
if labels is None:
    st.warning("The model couldn't run on this edition with the current setting.")
    st.stop()
labels = np.asarray(labels)
refs, exact = reference_sets(src_id)
chap_pos, _ = reference_starts(src_id)

# ── Timeline ──────────────────────────────────────────────────────────────────
st.subheader("Timeline", anchor="timeline")
st.caption(
    f"Top: the model's cluster sequence ({cluster_string(labels)}); dotted lines are its "
    "boundaries, shaded by the tolerance. Below: where each reference starts a division "
    "(blue = editors' parts, red = author's divisions, gray = Christenson's chapters). "
    "Hover for chapter titles."
    + ("" if exact else " **Positions approximate:** this edition lacks Christenson's chapter "
       "numbers, so his K'iche' positions are used by relative position.")
)
render_timeline(labels, refs, chap_pos, tol, k)

# ── Scores ────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Scores", anchor="scores")
st.caption(
    "**F1 is the test**: it rewards recovering the reference's division starts (recall) and "
    "penalizes boundaries the reference doesn't have (precision). Recall alone is often 1.0 by "
    "chance at this tolerance, so read it as *which* divisions are found. Chance = 95th "
    "percentile of scores after rotating the model's sequence."
)
tbl = score_table(labels, refs)
st.dataframe(tbl.round(3), width="stretch", hide_index=True)
_ed = tbl[tbl["Reference"].isin(EDITORS)]
_c1, _c2, _c3 = st.columns(3)
_c1.metric("Editors above chance (F1)", f"{int((_ed['F1 excess'] > 0).sum())} of {len(_ed)}",
           help="Editors whose part divisions the model matches better than the 95th percentile "
                "of rotated sequences.")
_c2.metric("Mean recall of editors' parts", f"{_ed['Recall'].mean():.2f}",
           help=f"Descriptive. Chance 95th percentile, averaged: {_ed['Recall chance 95th'].mean():.2f}")
_c3.metric(f"F1 vs {AUTHOR.lower()}", f"{tbl.loc[tbl['Reference'] == AUTHOR, 'F1'].iloc[0]:.2f}",
           help="Not independent of the models")
with st.expander("How the references are placed and scored"):
    st.markdown(METHOD_TEXT)

# ── Editorial consensus ───────────────────────────────────────────────────────
st.divider()
st.subheader("Editorial Consensus", anchor="consensus")
st.caption(
    "Every chapter where an editor starts a part, or where the author's divisions start. "
    "**Model** = distance from the nearest model boundary, in chunks; ✓ if within the tolerance."
)
cons = consensus_boundaries(chap_pos)
_mb = get_boundaries(labels)
_dist = [float(np.min(np.abs(_mb - p))) * n_chunks if len(_mb) else np.nan for p in cons["position"]]
cons["Model"] = [("✓ " if d <= tol * n_chunks else "✗ ") + f"{d:.1f}" for d in _dist]
cons["author"] = cons["author"].map({True: "✓", False: ""})
st.dataframe(
    cons.rename(columns={"chapter": "Chapter", "title": "Title", "n_editors": "Editors (n)",
                         "editors": "Editors", "author": "Author", "position": "Position"})
        .round({"Position": 3}),
    width="stretch", hide_index=True,
)

# ── The model among the editors ───────────────────────────────────────────────
st.divider()
st.subheader("The Model Among the Editors", anchor="among-editors")
model_name = f"Model (k = {k})"
sets = {model_name: get_boundaries(labels), AUTHOR: refs[AUTHOR], **{e: refs[e] for e in EDITORS}}
pf = pairwise_f1(sets, tol)
_ed_pairs = [pf.loc[a, b] for i, a in enumerate(EDITORS) for b in EDITORS[i + 1:]]
_model_ed = pf.loc[model_name, EDITORS]
st.caption(
    "Boundary F1 between every pair of segmentations, with the model treated as one more "
    f"editor. Editors agree with each other at mean F1 **{np.mean(_ed_pairs):.2f}** "
    f"(range {np.min(_ed_pairs):.2f}–{np.max(_ed_pairs):.2f}); the model agrees with them at "
    f"**{_model_ed.mean():.2f}**. F1 here penalizes the model's extra boundaries, so recall "
    "(above) is the fairer view of whether it finds the editors' divisions."
)
fig_pf = px.imshow(pf.round(2), text_auto=".2f", aspect="auto", zmin=0, zmax=1,
                   color_continuous_scale=_v["colorscale_boundary"])
fig_pf.update_layout(height=60 + 40 * len(sets), margin=dict(l=20, r=20, t=20, b=20),
                     coloraxis_colorbar=dict(title="F1"))
st.plotly_chart(fig_pf, width="stretch")

# ── All editions ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("All Editions", anchor="all-editions")
st.caption(
    "The same scores for every edition's own segmentation. Positions are exact for editions "
    "with Christenson's chapter numbers and approximate (relative position) for the rest."
)
rows = []
for s in src_ids:
    la = labels_for(s)
    if la is None:
        continue
    la = np.asarray(la)
    r_s, ex_s = reference_sets(s)
    ed_scores = [score_against(la, r_s[e], tol) for e in EDITORS]
    au = score_against(la, r_s[AUTHOR], tol)
    rows.append({
        "Edition": SOURCES_META[s]["label"], "Language": LANG_LABELS[SOURCES_META[s]["lang"]],
        "Positions": "exact" if ex_s else "relative", "Sequence": cluster_string(la),
        "Editors above chance (F1)": int(sum(x["f1_excess"] > 0 for x in ed_scores)),
        "Mean F1 excess (editors)": np.mean([x["f1_excess"] for x in ed_scores]),
        "Mean recall (editors)": np.mean([x["recall"] for x in ed_scores]),
        "F1 (author)": au["f1"], "F1 (author) chance 95th": au["f1_chance_p95"],
    })
st.dataframe(pd.DataFrame(rows).round(3), width="stretch", hide_index=True)
