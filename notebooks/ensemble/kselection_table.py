"""
Choosing k (Table 2) for the Popol Wuj narrative-structure essay.

For each metric (cross-edition ARI; boundary F1) and each of the app's four methods, runs the
grid search over n_chunks × max_df and chooses k* by the mean excess over chance across the
grid, on K'iche' ↔ K'iche' pairs only (translations held out) and on all pairs. Reports how
many combinations choose k* independently and the held-out K'iche' ↔ translation excess at
the reporting setting. Uses the Streamlit app's own pipeline (streamlit/pages), so the numbers
match the Choose k and Boundary Agreement pages.

    python notebooks/ensemble/kselection_table.py

Writes writing/tables/k_selection.{csv,md}. NMF needs a full fit per (combination, k): the
first run takes several minutes.
"""

import itertools
import logging
import os
import sys
import warnings

import pandas as pd
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "streamlit", "pages"))
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)      # silence Streamlit's no-runtime cache warnings

from grid_search_boundary_core import agreement_curve                    # noqa: E402
from grid_search_ui import K_VALS, K_MIN, choose_k, reporting_setting    # noqa: E402

with open(os.path.join(ROOT, "streamlit", "config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

_c      = cfg["controls"]
METHODS = ["TF-IDF", "LSA", "Cosine-Sim", "NMF"]
MAX_DFS = [round(0.30 + 0.05 * i, 2) for i in range(9)]        # the grid pages' default range
METRICS = {   # metric: (n_chunks range on its grid page, tolerance in chunks)
    "ARI": (list(range(10, 41, 5)), 0),
    "F1":  (list(range(15, 41, 5)), _c["boundary_tol_chunks"]["default"]),
}
OUT_DIR = os.path.join(ROOT, "writing", "tables")


def grid(method, metric):
    ncs, tol = METRICS[metric]
    rows = [{"combo_label": f"nc={nc} · max_df={m:.2f}", "n_chunks": nc, "max_df": m, **r}
            for nc, m in itertools.product(ncs, MAX_DFS)
            for r in agreement_curve(method, nc, m, metric, tol, K_VALS)]
    return pd.DataFrame(rows)


def main():
    rep = reporting_setting()
    rows = []
    for metric, method in itertools.product(METRICS, METHODS):
        print(f"{metric} · {method} …", file=sys.stderr, flush=True)
        df = grid(method, metric)
        rec = {"metric": metric, "method": method}
        for sel in ["within", "all"]:
            k_star, _ = choose_k(df, sel)
            ex = f"{sel}_excess"
            elig = df[df["k"] >= K_MIN]
            per_combo = elig.loc[elig.groupby("combo_label")[ex].idxmax()]
            rec.update({f"k_star_{sel}": k_star, f"mean_excess_{sel}": df[df["k"] == k_star][ex].mean(),
                        f"combos_{sel}": int((per_combo["k"] == k_star).sum()), f"n_combos_{sel}": len(per_combo)})
            if sel == "within":
                at_rep = next((r for r in agreement_curve(method, rep["n_chunks"], rep["max_df"], metric,
                                                          METRICS[metric][1], K_VALS, min_df=rep["min_df"],
                                                          ngram_range=(rep["ngram_min"], rep["ngram_max"]))
                               if r["k"] == k_star), None)
                rec["heldout_excess_reporting"] = at_rep["cross_excess"] if at_rep else float("nan")
                rec["heldout_excess_grid"] = df[df["k"] == k_star]["cross_excess"].mean()
        rows.append(rec)
    out = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(OUT_DIR, "k_selection.csv"), index=False)

    head = ["Metric", "Method", "k* (K'iche' pairs)", "Combinations choosing it", "Mean excess at k*",
            f"Held-out excess (n = {rep['n_chunks']})", "Held-out excess (grid mean)", "k* (all pairs)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in out.iterrows():
        lines.append("| " + " | ".join([
            "ARI" if r["metric"] == "ARI" else f"Boundary F1 (±{METRICS['F1'][1]})", r["method"],
            f"**{r['k_star_within']}**", f"{r['combos_within']} of {r['n_combos_within']}",
            f"{r['mean_excess_within']:+.2f}", f"{r['heldout_excess_reporting']:+.2f}",
            f"{r['heldout_excess_grid']:+.2f}", str(r["k_star_all"]),
        ]) + " |")
    ari = out[out["metric"] == "ARI"]
    caption = (
        f"k\\* = the k ≥ {K_MIN} with the highest mean excess over chance (observed − 95th percentile of "
        f"cyclic-shift scores) across the grid: n_chunks {METRICS['ARI'][0][0]}–{METRICS['ARI'][0][-1]} "
        f"(ARI) or {METRICS['F1'][0][0]}–{METRICS['F1'][0][-1]} (boundary F1), max_df "
        f"{MAX_DFS[0]:.2f}–{MAX_DFS[-1]:.2f}, min_df = {rep['min_df']}, k ≤ n_chunks / "
        f"{_c['min_chunks_per_cluster']}. Chosen on K'iche' ↔ K'iche' pairs; held-out excess is "
        f"K'iche' ↔ translation agreement at k\\*, at the reporting setting (n_chunks = "
        f"{rep['n_chunks']}, max_df = {rep['max_df']}) and averaged over the grid. ARI, which "
        f"measures agreement on the classes, chooses k; boundary F1 measures agreement on the "
        f"breaks. ARI k\\* is {', '.join(str(x) for x in ari['k_star_within'])} for "
        f"{', '.join(ari['method'])}."
    )
    md = ["**Table 2.** Choosing k by cross-edition agreement, translations held out", "",
          *lines, "", f"*{caption}*", ""]
    with open(os.path.join(OUT_DIR, "k_selection.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
