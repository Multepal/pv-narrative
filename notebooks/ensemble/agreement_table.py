"""
Cross-edition agreement table (Table 1) for the Popol Wuj narrative-structure essay.

Clusters every edition with each of the app's four methods at one fixed setting and
reports pairwise ARI and boundary F1 by pair group (K'iche' ↔ K'iche', K'iche' ↔
translation, all pairs) against the cyclic-shift chance baseline. A sensitivity
variant drops Christenson's Ximénez, which is close to the same text as Ximénez.

Uses the Streamlit app's own pipeline (streamlit/pages), so the numbers match the app.

    python notebooks/ensemble/agreement_table.py

Writes writing/tables/cross_edition_agreement.{csv,md}.
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

from grid_search_boundary_core import cluster_labels, agreement_by_group  # noqa: E402
from utils import find_token_file                                       # noqa: E402

with open(os.path.join(ROOT, "streamlit", "config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

_c = cfg["controls"]
SETTING = {
    "n_chunks":    _c["n_chunks"]["default"],
    "min_df":      _c["min_df"]["default"],
    "max_df":      _c["max_df"]["default"],
    "ngram_range": (_c["ngram_range"]["default_min"], _c["ngram_range"]["default_max"]),
    "k":           _c["n_clusters"]["default"],
    "tol_chunks":  _c["boundary_tol_chunks"]["default"],
}
METHODS  = {"TF-IDF": "TF-IDF → Ward", "LSA": "LSA → Ward",
            "Cosine-Sim": "Cosine-Sim → Ward", "NMF": "NMF → argmax"}
VARIANTS = {"All editions": [], "Without Christenson's Ximénez": ["christenson_ximenez"]}
METRICS  = {"ARI": "ARI", "F1": "F1"}
GROUPS   = {"within": "K'iche' ↔ K'iche'", "cross": "K'iche' ↔ translation", "all": "All pairs"}
OUT_DIR  = os.path.join(ROOT, "writing", "tables")


def edition_labels(method, exclude):
    labels, langs = [], []
    for src_id, meta in cfg["sources"].items():
        if src_id in exclude:
            continue
        la = cluster_labels(method, src_id, find_token_file(src_id), SETTING["n_chunks"],
                            SETTING["min_df"], SETTING["max_df"], SETTING["k"], SETTING["ngram_range"])
        if la is not None:
            labels.append(np.asarray(la))
            langs.append(meta["lang"])
    return labels, langs


def long_table():
    rows = []
    for variant, exclude in VARIANTS.items():
        for method in METHODS:
            labels, langs = edition_labels(method, exclude)
            for metric in METRICS:
                stats = agreement_by_group(labels, langs, metric, SETTING["tol_chunks"])
                for g in GROUPS:
                    rows.append({"variant": variant, "method": method, "metric": metric, "pairs": g,
                                 **{k: stats[f"{g}_{k}"] for k in
                                    ["n", "obs", "min", "max", "null_mean", "p95", "excess", "p"]}})
    return pd.DataFrame(rows)


def markdown_table(df, variant):
    """One row per method; for each metric and pair group: mean (range) and chance 95th pct."""
    d = df[(df["variant"] == variant) & (df["pairs"] != "all")]
    n = {g: int(d[d["pairs"] == g]["n"].iloc[0]) for g in ["within", "cross"]}
    head = ["Method"]
    for metric in METRICS:
        for g in ["within", "cross"]:
            head += [f"{METRICS[metric]}: {GROUPS[g]} ({n[g]} pairs)", "chance"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for method, label in METHODS.items():
        cells = [label]
        for metric in METRICS:
            for g in ["within", "cross"]:
                r = d[(d["method"] == method) & (d["metric"] == metric) & (d["pairs"] == g)].iloc[0]
                cells += [f"{r['obs']:.2f} ({r['min']:.2f}–{r['max']:.2f})", f"{r['p95']:.2f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    df = long_table()
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "cross_edition_agreement.csv"), index=False)

    s = SETTING
    main_rows = df[(df["variant"] == "All editions") & (df["pairs"] != "all")]
    p_max     = main_rows["p"].max()
    n_below   = int((main_rows["excess"] <= 0).sum())
    verdict   = ("Every mean exceeds its chance value" if n_below == 0 else
                 f"{n_below} of {len(main_rows)} means do not exceed their chance value")
    caption = (
        f"Mean pairwise agreement (range across pairs) between editions' segmentations at k = {s['k']}, "
        f"{s['n_chunks']} equal-token chunks, min_df = {s['min_df']}, max_df = {s['max_df']}, "
        f"n-grams {s['ngram_range'][0]}–{s['ngram_range'][1]}. ARI = adjusted Rand index; "
        f"F1 = boundary F1 with ±{s['tol_chunks']}-chunk tolerance. Chance = 95th percentile of "
        "scores after cyclically shifting one sequence of each pair, pooled within each pair group. "
        f"{verdict}; empirical p ≤ {p_max:.3f} throughout (pairs share editions, so p is "
        "descriptive)."
    )
    md = [f"**Table 1.** Cross-edition agreement of narrative segmentations (k = {s['k']})", "",
          markdown_table(df, "All editions"), "", f"*{caption}*", "",
          "**Table 1a.** Sensitivity: without Christenson's Ximénez", "",
          markdown_table(df, "Without Christenson's Ximénez"), "",
          "*Same setting. Christenson's Ximénez is close to the same text as Ximénez; dropping it "
          "tests whether near-duplicate transcriptions inflate K'iche' ↔ K'iche' agreement.*", ""]
    with open(os.path.join(OUT_DIR, "cross_edition_agreement.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
