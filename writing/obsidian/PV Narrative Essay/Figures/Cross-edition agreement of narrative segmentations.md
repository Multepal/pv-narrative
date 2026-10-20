---
created: 2026-10-20
status: draft
source: "Streamlit app → Cross-Edition Comparison; Topics (NMF/LDA)"
---

# Cross-edition agreement of narrative segmentations

Plan for reporting the cross-edition comparison results as a publishable table. The aim is to answer three questions: does agreement beat chance, does it survive translation, and does it hold across methods? Use one main table at a single headline setting, plus a short robustness table or figure. Don't publish the full parameter sweep.

## Main table: one setting, rows = methods

Fix the setting and state it in the caption. Use the app defaults (33 chunks, min_df = 5, max_df = 0.5, unigrams) with k = 6 and ±1 chunk tolerance.

**Table 1.** Cross-edition agreement of narrative segmentations (k = 6)

| Method | ARI: K'iche' ↔ K'iche' (10 pairs) | ARI: K'iche' ↔ translation (15 pairs) | ARI: chance 95th pct | F1: K'iche' ↔ K'iche' | F1: K'iche' ↔ translation | F1: chance 95th pct |
|---|---|---|---|---|---|---|
| TF-IDF → Ward | 0.82 | 0.76 | 0.42 | 0.97 | 0.95 | 0.80 |
| LSA → Ward | — | 0.75 | 0.44 | — | — | — |
| Cosine-Sim → Ward | — | 0.78 | 0.45 | — | — | — |
| NMF → argmax | — | — | — | — | — | — |

*ARI = adjusted Rand index; F1 = boundary F1 with ±1-chunk tolerance. Chance = 95th percentile of scores after cyclically shifting one sequence of each pair, pooled over all 28 pairs. 33 equal-token chunks; min_df = 5, max_df = 0.5.*

> [!warning] Provisional numbers
> Values come from headless test runs of the app on 2026-10-20. Dashes are values those runs didn't print, or that came from a ±0-tolerance run. NMF wasn't run. Regenerate every cell from one scripted run at fixed settings before publishing.

## Choices that make it publishable

1. **Pair counts in the column headers.** 5 K'iche' editions give 10 pairs. K'iche' × 3 translations (Recinos, Tedlock, Christenson's prose) gives 15. All pairs: 28.
2. **Give each mean a spread.** Add the range or SD across pairs, e.g. `0.76 (0.61–0.89)`. A mean over 15 pairs with no spread hides a lot.
3. **Use empirical p-values, not just the 95th percentile.** Report the share of chance scores at or above the observed mean. Footnote: pairs aren't independent, because each edition appears in several pairs, so treat these as descriptive, not formal tests.
4. **Ximénez sensitivity row.** Ximénez and Christenson's Ximénez are nearly the same text. Add a footnote or row giving the K'iche' ↔ K'iche' scores with one of them dropped.
5. **Put the k-choice in a second table or a figure.** k = 6 standing out is a separate claim. Show ARI excess over chance (observed − 95th percentile) at k = 4, 6, 8, 12, or use the sweep chart itself. A figure works better for a sweep.
6. **The NMF ↔ HAC match goes elsewhere.** ARI 0.92 against a chance 95th percentile of 0.41 (single edition, k = 6) is a within-edition result. It belongs in the topic-model section, not this table.
7. **Keep Hamming and high-k F1 out of the main table.**
   - Hamming depends on letter order: letters are assigned by first appearance, so one early difference relabels every later chunk.
   - Boundary F1 reaches chance from about k = 9 up, because boundaries fall every chunk or two and a ±1 window matches almost anything.

## Supporting results (TF-IDF, sweep over k)

| k | ARI K'iche' ↔ translation | ARI chance 95th pct | F1 (±1) K'iche' ↔ translation | F1 chance 95th pct |
|---|---|---|---|---|
| 4 | 0.71 | 0.49 | 0.84 | 0.67 |
| **6** | **0.76** | **0.42** | **0.95** | **0.80** |
| 8 | 0.61 | 0.38 | 0.90 | 0.87 |
| 12 | 0.59 | 0.31 | 0.93 | 0.97 |
| 20 | 0.47 | 0.26 | 1.00 | 1.00 |

- **The usual "ARI ≈ 0 = chance" rule doesn't apply here.** Misaligned block sequences score about 0.13–0.29 on average.
- **Agreement survives translation.** K'iche' ↔ translation ARI stays above the chance band from k = 3 to 20.
- **k = 6 stands furthest above chance.** That holds on both metrics, and it matches the NMF ↔ HAC agreement on the Topics page. Collapsed, the k = 6 sequences read A B C D C A E F.

## Method notes

- **Chance baseline:** for each pair of editions, one cluster sequence is rotated by every offset and the score recomputed. This keeps each edition's block structure and label proportions and breaks only the alignment.
  - For ARI, all offsets from 1 to n−1 are used.
  - For boundary F1, offsets within the tolerance are skipped, since F1 counts those as matches.
- **Boundary tolerance is fixed in chunks.** It isn't scaled by k as on Grid Search (Boundary). That page's τ = α/2(k−1) assumes k−1 boundaries, but recurring clusters produce more (about 7 at k = 6, 15 at k = 12). The k-scaled windows then overlap, and even misaligned sequences score near 1: chance-level F1 is about 0.94 at k = 6 and 0.97 at k = 12. **Open issue:** this may distort the "optimal k*" chosen by Grid Search (Boundary).
- **Alignment caveat:** chunks are aligned by token position, and translations expand and contract unevenly. Boundary F1 absorbs drift within the tolerance. ARI doesn't, so its cross-language scores are conservative.

## Relation to the grid search pages

All three pages compute the same statistic: cluster every edition with the same preprocessing, then average agreement over the 28 edition pairs. The **grid searches choose** k and the parameters. The **comparison page checks** the choice.

| | Cross-Edition Comparison | Grid Search (ARI) | Grid Search (Boundary) |
|---|---|---|---|
| Job | Check one setting in detail | Choose parameters and k | Choose parameters and k (used in the plan) |
| Search | Vary one parameter, hold the rest fixed | n_chunks × max_df grid, k = 2–20 | Same grid, plus Ensemble mode across 4 methods |
| Metric | ARI and boundary F1 (±chunks) | ARI | Boundary F1, τ = α/2(k−1) |
| Chance baseline | yes | no | no |
| Language split | yes | no | no |

### Problems on the selection side

1. **Choosing by raw scores favors small k.** k\* is the k with the highest raw mean agreement, but chance agreement isn't flat across k.
   - At the default setting, raw ARI at k = 2 is 1.000 for every pair, against a chance 95th percentile of 0.77.
   - Boundary F1 chance sits at about 0.87–1.0 across most of the range.

   **Fix:** choose k by excess over chance (observed − chance 95th percentile, or a z-score against the rotation baseline). At the default setting this measure finds k = 6 on both metrics.
2. **Choosing and checking with the same statistic is circular.** Reporting agreement at the k\* it was chosen to maximize inflates the result.

   **Fix:** a held-out design. Choose k and the parameters on **K'iche' ↔ K'iche'** pairs only. Check them on **K'iche' ↔ translation** pairs, which played no part in the choice. The translations share almost no vocabulary with the original, so agreement above chance there is the strongest evidence the structure belongs to the narrative. It also answers the "tuned until it agreed" objection.
3. **Duplicated code.** Each page has its own copy of the clustering code, so improvements (chance baseline, tolerance, language split) don't carry across pages.

### Plan

1. Move the clustering code into one shared module (`grid_search_boundary_core`).
2. Change the grid searches to choose k by excess over chance, with an option to choose on K'iche' ↔ K'iche' pairs only. Switch Grid Search (Boundary) to the chunk-based tolerance.
3. Have the comparison page report the held-out K'iche' ↔ translation scores at the chosen k; these become Table 1.

> [!important] Re-check k = 6
> Step 2 changes how the main k is chosen. Confirm k\* still comes out at 6 before relying on it in the essay.

## Next step

Write a script (e.g. `notebooks/ensemble/agreement_table.py`) that reuses the app's functions. It should run all four methods at the fixed setting and compute means, ranges, empirical p-values and the Ximénez-dropped variant. It should write a CSV plus a Markdown/LaTeX table into `writing/`, so the table regenerates when the pipeline changes.
