"""
Editors' Divisions — do the text's narrative boundaries fall where editors divide it?

Two independent boundary detectors are scored against independent editorial part divisions
(Brasseur, Recinos, Edmonson, Tedlock, Colop, Bazzett, Sotelo), the author's 8 interpretive
divisions, and Christenson's chapters, each against a cyclic-shift chance baseline:

  Class changes   — where the chosen segmentation's cluster label changes (the syntagm as a
                    sequence of paradigmatic classes)
  Resonance peaks — the strongest novelty-minus-transience peaks (Barron, Huang, Spang &
                    DeDeo 2018): breaks in lexical continuity, with no classes at all

The page also scores how far the two detectors converge, and places both among the editors:
how well they agree with them compared with how well the editors agree with each other.
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
from grid_search_boundary_core import (cluster_labels, get_boundaries, boundary_tol, N_COMPONENTS,
                                       run_resonance_boundaries)
from divisions import (reference_sets, reference_starts, score_against, score_positions,
                       resonance_peaks, pairwise_f1, consensus_boundaries, chapter_titles, DIV_CFG,
                       chapter_labels, chapter_grid_positions)

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
RESONANCE    = "Resonance peaks"

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

THEORY_TEXT = """
A narrative boundary can mean three different things, and this page puts them side by side.

1. **A change of paradigmatic class.** The story is read as the sequence of its classes
   (Lévi-Strauss's score read along the rows), so an episode is a run of one class and a
   boundary is where the class changes. These are the **class changes** of the chosen
   segmentation.
2. **A break in the text's lexical continuity**, whatever the classes: a passage that departs
   from what came before and whose departure persists (high novelty, low transience; Barron,
   Huang, Spang & DeDeo 2018, in the tradition of lexical-cohesion segmentation such as
   Hearst's TextTiling). These are the **resonance peaks**.
3. **An editor's division.** The Ximénez manuscript is undivided; divisions enter with
   Brasseur (1861), and every edition since makes its own. They are readings, not ground
   truth.

The test is whether (1) and (2), computed independently of each other and of the editors,
converge on (3). If they do, the divisions the editorial tradition recognizes are
*overdetermined* by the language of the text. The model is then one more reader in that
tradition, judged by how it stands among the other readers.
"""

METHOD_TEXT = """
**Placing the references.** Every reference is a list of Christenson chapter numbers where a
division starts. In Christenson's K'iche' text and his English prose, each chapter's start is
an exact token position. Other editions don't carry his chapter numbers, so his K'iche'
positions stand in by relative position; this assumes editions stay roughly proportional,
the same assumption the chunking makes.

**Detectors.** *Class changes*: chunk edges where the cluster label changes (about 7 at
k = 6). *Resonance peaks*: the same number of highest local maxima of resonance, so both
detectors get the same boundary budget and their scores are comparable.

**Recall, precision, F1.** Detector boundaries sit on chunk edges, reference starts anywhere:
with tolerance ± t chunks, a reference start counts if its *nearest chunk edge* is within t
chunks of a detector boundary (t + ½ chunks of raw distance). *Recall* = share of reference
boundaries recovered; *precision* = share of detector boundaries near some reference
boundary; **F1 balances the two and is the test.** Recall alone is easy by chance here: ~7
boundaries with windows of ± (t + ½) chunks cover much of the text, so a shifted detector often
recovers all 3–4 of an editor's part starts too. F1 also charges for boundaries the editor
doesn't have. Recall still says *which* divisions are found.

**Chance.** Class changes: the label sequence is rotated by every offset beyond the tolerance.
Resonance peaks: the positions are rotated the same way. Excess = observed − 95th percentile
of the rotated scores; p = share of rotations scoring at least as well (smallest attainable
p ≈ 0.03).

**Independence.** The editors' parts were made without these models: the external test. The
author's divisions may have been informed by the models; report them separately.

**Chapters.** Christenson's 87 chapter starts are closer together than the chunk tolerance
window, so chance precision against them is already 1. The *Chapter Resolution* section
instead predicts a class for every chapter (nearest class centroid of the chunk model) and
scores the predicted divisions in whole chapters.
"""


def boundary_runs(labels):
    """Run segments of a label sequence as (start_idx, end_idx_exclusive, letter)."""
    s = cluster_string(labels)
    runs, start = [], 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[start]:
            runs.append((start, i, s[start]))
            start = i
    return runs


def render_timeline(labels, res_peaks, refs, chap_pos, tol, model_row):
    """Model segments and resonance peaks on top, reference division starts below; the model's
    boundaries as dotted lines shaded by the tolerance."""
    n = len(labels)
    rows = [model_row, RESONANCE, AUTHOR, *EDITORS, CHAPTERS]
    pos_to_chap = {round(v, 6): c for c, v in chap_pos.items()}
    titles = chapter_titles()
    palette = px.colors.qualitative.Plotly
    letter_color = {}

    fig = go.Figure()
    for b in get_boundaries(labels):
        fig.add_vrect(x0=b - tol, x1=b + tol, fillcolor="gray", opacity=0.08, line_width=0)
        fig.add_vline(x=b, line_dash="dot", line_color="gray", line_width=1)
    for a, z, letter in boundary_runs(labels):
        color = letter_color.setdefault(letter, palette[len(letter_color) % len(palette)])
        fig.add_trace(go.Scatter(
            x=[a / n, z / n], y=[model_row, model_row], mode="lines",
            line=dict(color=color, width=22), showlegend=False,
            hovertemplate=f"Cluster {letter}<br>chunks {a}–{z - 1}<extra></extra>",
        ))
        fig.add_annotation(x=(a + z) / (2 * n), y=model_row, text=f"<b>{letter}</b>",
                           showarrow=False, font=dict(color="white", size=12))
    fig.add_trace(go.Scatter(
        x=res_peaks, y=[RESONANCE] * len(res_peaks), mode="markers", showlegend=False,
        marker=dict(symbol="line-ns-open", size=18, line=dict(width=3, color="#00CC96")),
        hovertemplate="Resonance peak<br>position %{x:.3f}<extra></extra>",
    ))
    for row in rows[2:]:
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


def score_row(detector, name, s):
    return {
        "Detector": detector, "Reference": name, "Independent": "—" if name == AUTHOR else "yes",
        "Ref. boundaries": s["n_ref"], "Detector boundaries": s["n_pred"],
        "F1": s["f1"], "F1 chance 95th": s["f1_chance_p95"], "F1 excess": s["f1_excess"],
        "F1 p": s["f1_p"], "Recall": s["recall"], "Recall chance 95th": s["recall_chance_p95"],
        "Precision": s["precision"],
    }


# ── Controls ──────────────────────────────────────────────────────────────────
st.title("Editors' Divisions — Popol Wuj")
render_toc([
    ("Timeline",                    "timeline"),
    ("Scores",                      "scores"),
    ("Chapter Resolution",          "chapter-resolution"),
    ("Editorial Consensus",         "consensus"),
    ("The Detectors Among the Editors", "among-editors"),
    ("All Editions",                "all-editions"),
])
st.caption(
    "Do class changes and lexical breaks — two independent ways of finding narrative boundaries "
    "— fall where editors divide the text? Editors' part divisions are the independent test; "
    "the author's divisions are reported separately."
)
with st.expander("Three meanings of a narrative boundary"):
    st.markdown(THEORY_TEXT)

src_ids = list(SOURCES_META.keys())
_default_src = DIV_CFG["chapter_editions"][0]
cols = st.columns([3, 3, 2, 2])
src_id = cols[0].selectbox(
    "Edition", src_ids, index=src_ids.index(_default_src),
    format_func=lambda x: f"{SOURCES_META[x]['label']} ({LANG_LABELS[SOURCES_META[x]['lang']]})")
method = cols[1].selectbox("Method (class changes)", list(METHOD_LABELS), format_func=METHOD_LABELS.get)
_tc = _c["boundary_tol_chunks"]
tol_chunks = cols[2].slider("Tolerance (± chunks)", _tc["min"], _tc["max"], _tc["default"], _tc["step"])
window = cols[3].number_input("Resonance window (w)", 1, 10, _c["resonance_window"]["default"], 1,
                              help="Chunks on each side for novelty and transience.")

_set, _from_choose_k = tfidf_settings(_c)
n_chunks, k = int(_set["n_chunks"]), int(_set["k"])
min_df, max_df = int(_set["min_df"]), float(_set["max_df"])
ngram = (int(_set["ngram_min"]), int(_set["ngram_max"]))
tol = boundary_tol(n_chunks, tol_chunks)
MODEL = f"Class changes (k = {k})"
st.caption(
    f"Setting: n_chunks = {n_chunks}, min_df = {min_df}, max_df = {max_df:.2f}, k = {k} · "
    f"{'from Choose k (this session)' if _from_choose_k else 'reporting setting from config'}"
    f" · tolerance ± {tol_chunks} chunk{'s' if tol_chunks != 1 else ''} (nearest chunk edge; "
    f"± {tol:.3f} of the text)."
)
st.divider()


def detectors_for(src):
    """(labels, class-change positions, resonance-peak positions) for an edition, or None."""
    tp = find_token_file(src)
    la = cluster_labels(method, src, tp, n_chunks, min_df, max_df, k, ngram)
    res = run_resonance_boundaries(src, tp, n_chunks, min_df, max_df, int(window))
    if la is None or res is None:
        return None
    la = np.asarray(la)
    mb = get_boundaries(la)
    return la, mb, resonance_peaks(res["resonance"], len(mb))


det = detectors_for(src_id)
if det is None:
    st.warning("The detectors couldn't run on this edition with the current setting.")
    st.stop()
labels, model_b, res_b = det
refs, exact = reference_sets(src_id)
chap_pos, _ = reference_starts(src_id)

# ── Timeline ──────────────────────────────────────────────────────────────────
st.subheader("Timeline", anchor="timeline")
st.caption(
    f"Top: the cluster sequence ({cluster_string(labels)}); dotted lines are its class changes, "
    f"shaded by the tolerance. Green: the {len(res_b)} strongest resonance peaks. Below: where "
    "each reference starts a division (blue = editors' parts, red = author's divisions, gray = "
    "Christenson's chapters). Hover for chapter titles."
    + ("" if exact else " **Positions approximate:** this edition lacks Christenson's chapter "
       "numbers, so his K'iche' positions are used by relative position.")
)
render_timeline(labels, res_b, refs, chap_pos, tol, MODEL)

# ── Scores ────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Scores", anchor="scores")
st.caption(
    "**F1 is the test**: it rewards recovering the reference's division starts (recall) and "
    "penalizes boundaries the reference doesn't have (precision). Recall alone is often 1.0 by "
    "chance at this tolerance, so read it as *which* divisions are found. Chance = 95th "
    "percentile of scores after rotating the detector."
)
tbl = pd.DataFrame(
    [score_row(MODEL, name, score_against(labels, ref, tol)) for name, ref in refs.items()]
    + [score_row(RESONANCE, name, score_positions(res_b, ref, tol, n_chunks)) for name, ref in refs.items()]
)
conv = score_positions(res_b, model_b, tol, n_chunks)

_c1, _c2, _c3, _c4 = st.columns(4)
for col, d in [(_c1, MODEL), (_c2, RESONANCE)]:
    _ed = tbl[(tbl["Detector"] == d) & tbl["Reference"].isin(EDITORS)]
    col.metric(f"{d}: editors above chance", f"{int((_ed['F1 excess'] > 0).sum())} of {len(_ed)}",
               help=f"Mean recall of editors' parts: {_ed['Recall'].mean():.2f}")
_c3.metric("Convergence (F1, detectors)", f"{conv['f1']:.2f}",
           help=f"Chance 95th percentile {conv['f1_chance_p95']:.2f}, p = {conv['f1_p']:.3f}")
_c4.metric(f"{MODEL}: F1 vs author", f"{tbl.loc[(tbl['Detector'] == MODEL) & (tbl['Reference'] == AUTHOR), 'F1'].iloc[0]:.2f}",
           help="Not independent of the models")
st.dataframe(tbl.round(3), width="stretch", hide_index=True)
with st.expander("How the references are placed and scored"):
    st.markdown(METHOD_TEXT)

# ── Chapter resolution ────────────────────────────────────────────────────────
st.divider()
st.subheader("Chapter Resolution: Predicting the Divisions", anchor="chapter-resolution")
if src_id not in DIV_CFG["chapter_editions"]:
    st.info("Chapter resolution needs an edition with Christenson's chapter numbers: "
            + ", ".join(SOURCES_META[s]["label"] for s in DIV_CFG["chapter_editions"]) + ".")
else:
    st.caption(
        "Each of Christenson's chapters, which the model never saw, is assigned to the nearest "
        "class centroid of the chunk model. A change of class between consecutive chapters is a "
        "**predicted division**, scored in whole chapters against the editors' part starts. "
        "Chance = rotating the chapter sequence."
    )
    _ct = _c["chapter_tol"]
    chap_tol = st.slider("Chapter tolerance (± chapters)", _ct["min"], _ct["max"], _ct["default"],
                         _ct["step"], help="0 = the predicted division must fall on the editor's chapter.")
    chapters, ch_labels = chapter_labels(method, src_id, n_chunks, min_df, max_df, k, ngram)
    n_ch = len(chapters)
    ch_tol = (chap_tol + 0.5) / n_ch
    ch_bounds = get_boundaries(ch_labels)
    pred_chaps = [int(chapters[int(round(b * n_ch))]) for b in ch_bounds]
    ch_refs = {AUTHOR: chapter_grid_positions([d["chap"] for d in DIV_CFG["author"]], chapters),
               **{e: chapter_grid_positions(c, chapters) for e, c in DIV_CFG["editors"].items()}}
    ch_tbl = pd.DataFrame([score_row(f"Chapter classes (k = {k})", name, score_against(ch_labels, ref, ch_tol))
                           for name, ref in ch_refs.items()])
    _ced = ch_tbl[ch_tbl["Reference"].isin(EDITORS)]
    _m1, _m2, _m3 = st.columns(3)
    _m1.metric("Editors above chance", f"{int((_ced['F1 excess'] > 0).sum())} of {len(_ced)}")
    _m2.metric(f"Editors fully recovered (± {chap_tol})", f"{int((_ced['Recall'] == 1).sum())} of {len(_ced)}")
    _m3.metric("Predicted divisions", len(pred_chaps))
    st.markdown("**Predicted division starts (chapters):** " + ", ".join(map(str, pred_chaps)))

    # Chapter strip: class per chapter on top, division starts below
    palette = px.colors.qualitative.Plotly
    seq = cluster_string(ch_labels)
    letter_color = {}
    rows_ch = [f"Chapter classes (k = {k})", AUTHOR, *EDITORS]
    titles = chapter_titles()
    fig_ch = go.Figure()
    for b in ch_bounds:
        x = chapters[int(round(b * n_ch))] - 0.5
        fig_ch.add_vline(x=x, line_dash="dot", line_color="gray", line_width=1)
    for i, (ch, letter) in enumerate(zip(chapters, seq)):
        color = letter_color.setdefault(letter, palette[len(letter_color) % len(palette)])
        fig_ch.add_trace(go.Scatter(
            x=[ch - 0.5, ch + 0.5], y=[rows_ch[0]] * 2, mode="lines", line=dict(color=color, width=22),
            showlegend=False, hovertemplate=f"Chapter {ch}: {titles.get(ch, '')}<br>class {letter}<extra></extra>",
        ))
    for row, chaps in [(AUTHOR, [d["chap"] for d in DIV_CFG["author"]]), *DIV_CFG["editors"].items()]:
        cs = [c for c in chaps if c > 1]
        fig_ch.add_trace(go.Scatter(
            x=[c - 0.5 for c in cs], y=[row] * len(cs), mode="markers", showlegend=False,
            marker=dict(symbol="line-ns-open", size=18,
                        line=dict(width=3, color="#EF553B" if row == AUTHOR else "#1f77b4")),
            customdata=[[c, titles.get(c, "")] for c in cs],
            hovertemplate=f"{row}<br>chapter %{{customdata[0]}}: %{{customdata[1]}}<extra></extra>",
        ))
    fig_ch.update_layout(
        height=60 + 34 * len(rows_ch), margin=dict(l=10, r=20, t=10, b=40), plot_bgcolor="white",
        xaxis=dict(title="Christenson chapter", range=[0.5, chapters[-1] + 0.5], dtick=5,
                   showgrid=False, zeroline=False),
        yaxis=dict(categoryorder="array", categoryarray=rows_ch, autorange="reversed",
                   showgrid=False, zeroline=False),
    )
    st.plotly_chart(fig_ch, width="stretch")
    st.dataframe(ch_tbl.drop(columns=["Detector"]).round(3), width="stretch", hide_index=True)
    st.caption(
        "Single-chapter excursions (one chapter assigned to a different class than its "
        "neighbors) add two predicted divisions each and lower precision; they mark chapters "
        "whose vocabulary sits between classes."
    )

# ── Editorial consensus ───────────────────────────────────────────────────────
st.divider()
st.subheader("Editorial Consensus", anchor="consensus")
st.caption(
    "Every chapter where an editor starts a part, or where the author's divisions start, with "
    "the distance to the nearest boundary of each detector, in chunks (✓ = within the tolerance)."
)
cons = consensus_boundaries(chap_pos)
for name, b in [("Class changes", model_b), ("Resonance", res_b)]:
    d = [float(np.min(np.abs(b - p))) * n_chunks if len(b) else np.nan for p in cons["position"]]
    cons[name] = [("✓ " if x <= tol * n_chunks else "✗ ") + f"{x:.1f}" for x in d]
cons["author"] = cons["author"].map({True: "✓", False: ""})
st.dataframe(
    cons.rename(columns={"chapter": "Chapter", "title": "Title", "n_editors": "Editors (n)",
                         "editors": "Editors", "author": "Author", "position": "Position"})
        .round({"Position": 3}),
    width="stretch", hide_index=True,
)

# ── The detectors among the editors ───────────────────────────────────────────
st.divider()
st.subheader("The Detectors Among the Editors", anchor="among-editors")
sets = {MODEL: model_b, RESONANCE: res_b, AUTHOR: refs[AUTHOR], **{e: refs[e] for e in EDITORS}}
pf = pairwise_f1(sets, tol)
_ed_pairs = [pf.loc[a, b] for i, a in enumerate(EDITORS) for b in EDITORS[i + 1:]]
st.caption(
    "Boundary F1 between every pair of segmentations, with each detector treated as one more "
    f"reader. Editors agree with each other at mean F1 **{np.mean(_ed_pairs):.2f}** (range "
    f"{np.min(_ed_pairs):.2f}–{np.max(_ed_pairs):.2f}); class changes agree with them at "
    f"**{pf.loc[MODEL, EDITORS].mean():.2f}**, resonance peaks at "
    f"**{pf.loc[RESONANCE, EDITORS].mean():.2f}**. F1 penalizes the detectors' extra boundaries "
    "relative to the editors' 3–4 parts."
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
    "The same scores for every edition's own detectors. Positions are exact for editions with "
    "Christenson's chapter numbers and approximate (relative position) for the rest."
)
rows = []
for s in src_ids:
    d = detectors_for(s)
    if d is None:
        continue
    la, mb, rb = d
    r_s, ex_s = reference_sets(s)
    cc = [score_against(la, r_s[e], tol) for e in EDITORS]
    rr = [score_positions(rb, r_s[e], tol, n_chunks) for e in EDITORS]
    cv = score_positions(rb, mb, tol, n_chunks)
    rows.append({
        "Edition": SOURCES_META[s]["label"], "Language": LANG_LABELS[SOURCES_META[s]["lang"]],
        "Positions": "exact" if ex_s else "relative", "Sequence": cluster_string(la),
        "Class changes: editors above chance": int(sum(x["f1_excess"] > 0 for x in cc)),
        "Resonance: editors above chance": int(sum(x["f1_excess"] > 0 for x in rr)),
        "Convergence F1": cv["f1"], "Convergence chance 95th": cv["f1_chance_p95"],
        "Class changes: F1 (author)": score_against(la, r_s[AUTHOR], tol)["f1"],
    })
st.dataframe(pd.DataFrame(rows).round(3), width="stretch", hide_index=True)
