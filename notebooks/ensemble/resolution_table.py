"""
Sensitivity of the main results to n_chunks (Table S1) for the Popol Wuj essay.

For each resolution n: the stability of the k* partition across nearby resolutions (the rule
that chooses n), k* by cross-edition ARI, held-out agreement, the k = 6 sequence, and the
agreement of Christenson's K'iche' segmentation with the seven editors' divisions at chunk
and chapter resolution. Uses the Streamlit app's own pipeline (streamlit/pages).

    python notebooks/ensemble/resolution_table.py

Writes writing/tables/resolution_sensitivity.{csv,md}.
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

from grid_search_boundary_core import (agreement_curve, cluster_labels, get_boundaries,  # noqa: E402
                                       boundary_tol)
from grid_search_ui import K_VALS                                                        # noqa: E402
from divisions import (reference_sets, score_against, chapter_labels,                   # noqa: E402
                       chapter_grid_positions, DIV_CFG)
from resolution import resolution_stability, stability_plateau                          # noqa: E402
from utils import find_token_file, load_tokens, cluster_string                          # noqa: E402

with open(os.path.join(ROOT, "streamlit", "config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

_c       = cfg["controls"]
K        = _c["n_clusters"]["default"]
MIN_DF   = _c["min_df"]["default"]
MAX_DF   = _c["max_df"]["default"]
NGRAM    = (_c["ngram_range"]["default_min"], _c["ngram_range"]["default_max"])
N_REP    = _c["n_chunks"]["default"]
TOL      = _c["boundary_tol_chunks"]["default"]
MAX_DFS  = [round(0.30 + 0.05 * i, 2) for i in range(9)]       # the grid's max_df range
NS       = [18, 20, 25, 30, 32, 33, 35, 38, 40, 45, 50]
SRC      = DIV_CFG["chapter_editions"][0]
EDITORS  = list(DIV_CFG["editors"])
SHARED   = [9, 15, 41, 65]                                     # part starts shared across editors
OUT_DIR  = os.path.join(ROOT, "writing", "tables")


def collapse(seq):
    return " ".join(c for i, c in enumerate(seq) if i == 0 or c != seq[i - 1])


def main():
    stab = resolution_stability("TF-IDF", K, MIN_DF, MAX_DF, NGRAM).set_index("n_chunks")
    onset, end = stability_plateau(stab.reset_index())
    tp = find_token_file(SRC)
    n_tok = len(load_tokens(SRC, tp))
    rows = []
    for n in NS:
        grid = pd.DataFrame([{"max_df": m, **r} for m in MAX_DFS
                             for r in agreement_curve("TF-IDF", n, m, "ARI", 0, K_VALS)])
        by_k = grid[grid["k"] >= _c["k_selection_min"]].groupby("k")["within_excess"].mean()
        at_k = grid[grid["k"] == K]
        la = np.asarray(cluster_labels("TF-IDF", SRC, tp, n, MIN_DF, MAX_DF, K, NGRAM))
        refs, _ = reference_sets(SRC)
        tol = boundary_tol(n, TOL)
        chapters, ch = chapter_labels("TF-IDF", SRC, n, MIN_DF, MAX_DF, K, NGRAM)
        n_ch = len(chapters)
        pred = {int(chapters[int(round(b * n_ch))]) for b in get_boundaries(ch)}
        rows.append({
            "n_chunks": n, "tokens_per_chunk": int(n_tok / n),
            "stability": stab.loc[n, "smoothed"], "on_plateau": onset <= n <= end,
            "k_star_ari": int(by_k.idxmax()),
            "excess_k": at_k["within_excess"].mean(), "heldout_excess_k": at_k["cross_excess"].mean(),
            "sequence": collapse(cluster_string(la)),
            "editors_chunk": int(sum(score_against(la, refs[e], tol)["f1_excess"] > 0 for e in EDITORS)),
            "editors_chapter": int(sum(score_against(ch, chapter_grid_positions(c, chapters), 0.5 / n_ch)["f1_excess"] > 0
                                       for c in DIV_CFG["editors"].values())),
            "exact_shared": "".join("✓" if c in pred else "·" for c in SHARED),
        })
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "resolution_sensitivity.csv"), index=False)

    head = ["n_chunks", "Tokens/chunk", "Stability", f"k* (ARI)", f"Held-out excess (k = {K})",
            f"Sequence (k = {K})", "Editors beaten (chunks)", "Editors beaten (chapters)",
            "Exact at ch. " + "/".join(map(str, SHARED))]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in df.iterrows():
        n = f"**{r['n_chunks']}**" if r["n_chunks"] in (onset, N_REP) else str(r["n_chunks"])
        stab_cell = f"{r['stability']:.3f}" + (" ▮" if r["on_plateau"] else "")
        lines.append("| " + " | ".join([
            n, f"{r['tokens_per_chunk']:,}", stab_cell, str(r["k_star_ari"]), f"{r['heldout_excess_k']:+.2f}",
            r["sequence"], f"{r['editors_chunk']} of 7", f"{r['editors_chapter']} of 7", r["exact_shared"],
        ]) + " |")
    caption = (
        f"TF-IDF → Ward, min_df = {MIN_DF}; k* and held-out excess averaged over max_df "
        f"{MAX_DFS[0]:.2f}–{MAX_DFS[-1]:.2f}; sequence and editorial agreement for Christenson's "
        f"K'iche' text at max_df = {MAX_DF}. Stability = token-level ARI between the k = {K} "
        f"partition at n and at n ± 10%, mean over the 8 editions, smoothed; ▮ = on the stability "
        f"plateau ({onset}–{end}). The rule sets n_chunks at the plateau onset (**{onset}**); results "
        f"are reported at **{N_REP}**. Editors beaten = editors whose part divisions the "
        f"segmentation matches above chance (boundary F1; ±{TOL} chunk, or exact chapters)."
    )
    md = ["**Table S1.** Sensitivity of the main results to resolution (n_chunks)", "",
          *lines, "", f"*{caption}*", ""]
    with open(os.path.join(OUT_DIR, "resolution_sensitivity.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
