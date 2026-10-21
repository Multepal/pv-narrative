---
created: 2026-10-20
updated: 2026-10-20
status: tables generated
source: "notebooks/ensemble/agreement_table.py → writing/tables/cross_edition_agreement.{csv,md}"
---

# Cross-edition agreement of narrative segmentations

Do independent editions and translations of the Popol Wuj segment into the same narrative structure? The tables answer three questions: does agreement beat chance, does it survive translation, and does it hold across methods?

> [!info] Regenerating
> Tables 1 and 1a are written by `python notebooks/ensemble/agreement_table.py` (project env), which uses the Streamlit app's own pipeline. Rerun it after any pipeline change and paste the new `writing/tables/cross_edition_agreement.md` here. Regenerated 2026-10-21 (LSA now uses centered PCA, as on the PCA / LSA page).

## Table 1

**Table 1.** Cross-edition agreement of narrative segmentations (k = 6)

| Method | ARI: K'iche' ↔ K'iche' (10 pairs) | chance | ARI: K'iche' ↔ translation (15 pairs) | chance | F1: K'iche' ↔ K'iche' (10 pairs) | chance | F1: K'iche' ↔ translation (15 pairs) | chance |
|---|---|---|---|---|---|---|---|---|
| TF-IDF → Ward | 0.82 (0.73–1.00) | 0.42 | 0.76 (0.62–1.00) | 0.41 | 0.97 (0.87–1.00) | 0.75 | 0.95 (0.87–1.00) | 0.80 |
| LSA → Ward | 0.89 (0.78–1.00) | 0.49 | 0.79 (0.66–0.88) | 0.45 | 1.00 (1.00–1.00) | 0.80 | 0.92 (0.71–1.00) | 0.76 |
| Cosine-Sim → Ward | 0.79 (0.66–0.92) | 0.45 | 0.78 (0.56–1.00) | 0.45 | 0.91 (0.86–1.00) | 0.79 | 0.94 (0.86–1.00) | 0.80 |
| NMF → argmax | 0.92 (0.84–1.00) | 0.48 | 0.80 (0.70–0.87) | 0.48 | 1.00 (1.00–1.00) | 0.80 | 1.00 (1.00–1.00) | 0.80 |

*Mean pairwise agreement (range across pairs) between editions' segmentations at k = 6, 33 equal-token chunks, min_df = 5, max_df = 0.5, n-grams 1–1. ARI = adjusted Rand index; F1 = boundary F1 with ±1-chunk tolerance. Chance = 95th percentile of scores after cyclically shifting one sequence of each pair, pooled within each pair group. Every mean exceeds its chance value; empirical p ≤ 0.003 throughout (pairs share editions, so p is descriptive).*

**Table 1a.** Sensitivity: without Christenson's Ximénez

| Method | ARI: K'iche' ↔ K'iche' (6 pairs) | chance | ARI: K'iche' ↔ translation (12 pairs) | chance | F1: K'iche' ↔ K'iche' (6 pairs) | chance | F1: K'iche' ↔ translation (12 pairs) | chance |
|---|---|---|---|---|---|---|---|---|
| TF-IDF → Ward | 0.82 (0.77–1.00) | 0.41 | 0.78 (0.66–1.00) | 0.42 | 0.98 (0.87–1.00) | 0.75 | 0.97 (0.87–1.00) | 0.80 |
| LSA → Ward | 0.92 (0.84–1.00) | 0.49 | 0.79 (0.66–0.88) | 0.45 | 1.00 (1.00–1.00) | 0.80 | 0.95 (0.86–1.00) | 0.73 |
| Cosine-Sim → Ward | 0.84 (0.72–0.92) | 0.44 | 0.82 (0.61–1.00) | 0.45 | 0.93 (0.86–1.00) | 0.78 | 0.96 (0.86–1.00) | 0.80 |
| NMF → argmax | 0.91 (0.84–1.00) | 0.46 | 0.79 (0.70–0.87) | 0.48 | 1.00 (1.00–1.00) | 0.80 | 1.00 (1.00–1.00) | 0.80 |

*Same setting. Christenson's Ximénez is close to the same text as Ximénez; dropping it tests whether near-duplicate transcriptions inflate K'iche' ↔ K'iche' agreement.*

### Reading the tables

- **Every method beats chance on both metrics, within K'iche' and across translation.** ARI is about 0.3–0.45 above chance; boundary F1 about 0.1–0.2 above (its chance level is high because F1 with tolerance is forgiving).
- **The near-duplicate transcriptions don't drive the result.** Dropping Christenson's Ximénez changes no mean by more than 0.05.
- **NMF boundaries agree perfectly within ±1 chunk.** All 8 editions place their seven boundaries near chunks 2, 6, 8–9, 13–14, 18–19, 20 and 28–29, and every edition, Spanish and English included, reads **A B C D C A E F**.

## Choosing k (Table 2, to draft)

k = 6 comes from the grid search pages, which choose k by **excess over chance** (observed − chance 95th percentile) on **K'iche' ↔ K'iche' pairs only**, holding the translations out. Settings: n_chunks 10–40 (ARI) or 15–40 (boundary F1), max_df 0.30–0.70, min_df 5, k ≥ 3, k ≤ n_chunks / 3.

| Grid search | Methods | k\* | Held-out K'iche' ↔ translation excess at k\* |
|---|---|---|---|
| ARI | TF-IDF, LSA, Cosine-Sim | 6, 6, 6 | +0.48, +0.47, +0.51 |
| Boundary F1 (±1), single method | TF-IDF, LSA, Cosine-Sim | 6, 6, 6 | +0.26, +0.27, +0.18 |
| Boundary F1 (±1), Ensemble (recommended: 40 chunks, max_df 0.50) | TF-IDF, LSA, Cosine-Sim, NMF | 6, 6, 6, 6 | +0.26, +0.25, +0.22, +0.28 |

k\* is also 6 when chosen on all pairs. These values were read from the app on 2026-10-20, LSA and Ensemble rows updated 2026-10-21 after switching the grid searches to centered PCA (not yet scripted). If this becomes a published table, extend `agreement_table.py` to produce it.

- **Two peaks in boundary F1: k = 2 and k = 6.** At k = 2 all editions agree on where the single main division falls (F1 ≈ 1.0 against chance ≈ 0.67). That is a real result about the text's primary break, worth a sentence in the essay, but it is a different question from the episode structure, so k\* is chosen among k ≥ 3.
- **From about k = 8 up, boundary agreement is no better than chance.**

## Method notes

- **Chance baseline:** for each pair of editions, one cluster sequence is rotated by every offset and the score recomputed. This keeps each edition's block structure and label proportions and breaks only the alignment. Each pair group is compared with its own pairs' rotated scores.
  - For ARI, all offsets from 1 to n−1 are used.
  - For boundary F1, offsets within the tolerance are skipped, since F1 counts those as matches.
- **Why not ARI ≈ 0 as chance:** misaligned block sequences score about 0.13–0.29 on average, and up to 0.77 at k = 2.
- **Boundary tolerance is fixed in chunks**, not scaled by k. The old τ = α/2(k−1) assumed k−1 boundaries, but recurring clusters produce more (about 7 at k = 6, 15 at k = 12), so k-scaled windows overlapped and even misaligned sequences scored near 1.
- **Why k ≤ n_chunks / 3:** near k = n_chunks, partitions approach one chunk per cluster and agreement becomes trivial. k\* = 6 is the same for caps of 2, 3 or 4 chunks per cluster.
- **Reproducibility:** NMF is seeded (`model.random_state` in config). Before 2026-10-20 it wasn't, and NMF labels at k ≥ 9 varied between runs.
- **Alignment caveat:** chunks are aligned by token position, and translations expand and contract unevenly. Boundary F1 absorbs drift within the tolerance; ARI doesn't, so its cross-language scores are conservative.
- **The NMF ↔ HAC match belongs elsewhere.** ARI 0.92 against a chance 95th percentile of 0.41 (single edition, k = 6) is a within-edition result for the topic-model section.
- **Kept out of the tables:** Hamming distance (depends on letter order) and boundary F1 at high k (at chance).

## Status

- [x] Shared pipeline module (`grid_search_boundary_core`)
- [x] Grid searches choose k by excess over chance, K'iche'-only selection with held-out translations
- [x] Comparison page reports per-group agreement, range, chance and p at the chosen setting
- [x] Table 1 and 1a generated by script
- [ ] Script Table 2 (grid k\* and held-out scores) if it is published
- [ ] Decide whether to discuss the k = 2 primary division in the essay
