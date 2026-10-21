"""
Agreement with editorial divisions (Table 3) for the Popol Wuj narrative-structure essay.

For Christenson's K'iche' text (exact chapter positions) at the reporting setting, scores
three boundary predictors against seven editors' part divisions and the author's divisions,
each against a cyclic-shift chance baseline:

  chunk class changes   where the k-class sequence of chunks changes class
  resonance peaks       strongest novelty-minus-transience peaks (Barron et al. 2018)
  chapter classes       each chapter assigned out of sample to the nearest class centroid;
                        divisions where the chapter class changes, scored in whole chapters

Uses the Streamlit app's own pipeline (streamlit/pages), so the numbers match the
Editors' Divisions page.

    python notebooks/ensemble/division_table.py

Writes writing/tables/editors_divisions.{csv,md}.
"""

import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "streamlit", "pages"))
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)      # silence Streamlit's no-runtime cache warnings

from grid_search_boundary_core import (cluster_labels, get_boundaries, boundary_tol,  # noqa: E402
                                       run_resonance_boundaries)
from divisions import (reference_sets, reference_starts, score_against, score_positions,  # noqa: E402
                       resonance_peaks, pairwise_f1, consensus_boundaries, chapter_labels,
                       chapter_grid_positions, DIV_CFG)
from utils import find_token_file                                                      # noqa: E402

with open(os.path.join(ROOT, "streamlit", "config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

_c = cfg["controls"]
SRC      = DIV_CFG["chapter_editions"][0]
METHOD   = "TF-IDF"
N_CHUNKS = _c["n_chunks"]["default"]
MIN_DF   = _c["min_df"]["default"]
MAX_DF   = _c["max_df"]["default"]
NGRAM    = (_c["ngram_range"]["default_min"], _c["ngram_range"]["default_max"])
K        = _c["n_clusters"]["default"]
TOL      = _c["boundary_tol_chunks"]["default"]
CH_TOL   = _c["chapter_tol"]["default"]
WINDOW   = _c["resonance_window"]["default"]
AUTHOR   = "Author's 8 divisions"
EDITORS  = list(DIV_CFG["editors"])
OUT_DIR  = os.path.join(ROOT, "writing", "tables")


def main():
    tp = find_token_file(SRC)
    labels = np.asarray(cluster_labels(METHOD, SRC, tp, N_CHUNKS, MIN_DF, MAX_DF, K, NGRAM))
    model_b = get_boundaries(labels)
    res = run_resonance_boundaries(SRC, tp, N_CHUNKS, MIN_DF, MAX_DF, WINDOW)["resonance"]
    res_b = resonance_peaks(res, len(model_b))
    tol = boundary_tol(N_CHUNKS, TOL)
    refs, _ = reference_sets(SRC)

    chapters, ch_labels = chapter_labels(METHOD, SRC, N_CHUNKS, MIN_DF, MAX_DF, K, NGRAM)
    n_ch = len(chapters)
    ch_tol = (CH_TOL + 0.5) / n_ch
    ch_refs = {AUTHOR: chapter_grid_positions([d["chap"] for d in DIV_CFG["author"]], chapters),
               **{e: chapter_grid_positions(c, chapters) for e, c in DIV_CFG["editors"].items()}}
    ch_b = get_boundaries(ch_labels)
    pred_chaps = [int(chapters[int(round(b * n_ch))]) for b in ch_b]

    rows = []
    for name in [*EDITORS, AUTHOR]:
        for detector, s in [("chunk class changes", score_against(labels, refs[name], tol)),
                            ("resonance peaks", score_positions(res_b, refs[name], tol, N_CHUNKS)),
                            ("chapter classes", score_against(ch_labels, ch_refs[name], ch_tol))]:
            rows.append({"reference": name, "detector": detector,
                         **{k: s[k] for k in ["n_ref", "n_pred", "f1", "f1_chance_p95", "f1_excess",
                                              "f1_p", "recall", "precision"]}})
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "editors_divisions.csv"), index=False)

    # Table 3: one row per reference; F1 (chance 95th) for each detector
    det_order = ["chunk class changes", "resonance peaks", "chapter classes"]
    head = ["Reference (part starts)", "Chunk class changes", "Resonance peaks",
            f"Chapter classes (±{CH_TOL} ch)", "Chapter recall"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name in [*EDITORS, AUTHOR]:
        d = df[df["reference"] == name].set_index("detector")
        chaps = [c for c in (DIV_CFG["editors"].get(name) or [x["chap"] for x in DIV_CFG["author"]]) if c > 1]
        label = f"{name} ({', '.join(map(str, chaps))})" + (" †" if name == AUTHOR else "")
        cells = [label]
        for det in det_order:
            r = d.loc[det]
            mark = "**" if r["f1_excess"] > 0 else ""
            cells.append(f"{mark}{r['f1']:.2f}{mark} ({r['f1_chance_p95']:.2f})")
        cells.append(f"{d.loc['chapter classes', 'recall']:.2f}")
        lines.append("| " + " | ".join(cells) + " |")

    above = {det: int((df[(df["detector"] == det) & df["reference"].isin(EDITORS)]["f1_excess"] > 0).sum())
             for det in det_order}
    pf = pairwise_f1({"model": model_b, **{e: refs[e] for e in EDITORS}}, tol)
    ed_pairs = [pf.loc[a, b] for i, a in enumerate(EDITORS) for b in EDITORS[i + 1:]]
    caption = (
        f"Boundary F1 (chance 95th percentile) against each edition's part starts, given as "
        f"Christenson chapters, in Christenson's K'iche' text. {METHOD} → Ward, k = {K}, "
        f"{N_CHUNKS} chunks, min_df = {MIN_DF}, max_df = {MAX_DF}. Chunk-level detectors match "
        f"within ±{TOL} chunk (nearest chunk edge); chapter classes are each chapter's nearest "
        f"class centroid (out of sample), matched within ±{CH_TOL} chapters. Resonance peaks "
        f"use window {WINDOW} and the same boundary budget as the class changes. Bold = above "
        f"chance. Editors above chance: chunk class changes {above['chunk class changes']} of 7, "
        f"resonance {above['resonance peaks']} of 7, chapter classes {above['chapter classes']} "
        f"of 7. Predicted chapter divisions: {', '.join(map(str, pred_chaps))}. Editors agree "
        f"with each other at mean F1 {np.mean(ed_pairs):.2f}; chunk class changes agree with "
        f"them at {pf.loc['model', EDITORS].mean():.2f}. † Not independent of the models."
    )

    # Table 3a: consensus boundaries
    starts, _ = reference_starts(SRC)
    cons = consensus_boundaries(starts)
    pred_set = set(pred_chaps)
    hit = lambda b, p: "✓" if len(b) and np.min(np.abs(b - p)) <= tol else ""
    lines_a = ["| Chapter | Title | Editors | Author | Chunk class change | Resonance | Chapter classes (exact) |",
               "|---|---|---|---|---|---|---|"]
    for _, r in cons.iterrows():
        lines_a.append(f"| {r['chapter']} | {r['title']} | {r['n_editors']}"
                       f"{' (' + r['editors'] + ')' if r['editors'] else ''} | {'✓' if r['author'] else ''} | "
                       f"{hit(model_b, r['position'])} | {hit(res_b, r['position'])} | "
                       f"{'✓' if r['chapter'] in pred_set else ''} |")

    md = [f"**Table 3.** Agreement with editorial divisions (Christenson's K'iche' text, k = {K})", "",
          *lines, "", f"*{caption}*", "",
          "**Table 3a.** Division starts recognized by editors or the author, and which predictors mark them", "",
          *lines_a, ""]
    with open(os.path.join(OUT_DIR, "editors_divisions.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
