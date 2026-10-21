---
created: 2026-10-20
updated: 2026-10-21
status: tables generated
source: "notebooks/ensemble/agreement_table.py → writing/tables/cross_edition_agreement.{csv,md}"
---

# Cross-edition agreement of narrative segmentations

Do independent editions and translations of the Popol Wuj segment into the same narrative structure? The tables answer three questions: does agreement beat chance, does it survive translation, and does it hold across methods?

> [!info] Regenerating
> Tables 1 and 1a are written by `python notebooks/ensemble/agreement_table.py` (project env), which uses the Streamlit app's own pipeline. Rerun it after any pipeline change and paste the new `writing/tables/cross_edition_agreement.md` here. Regenerated 2026-10-21 after fixing the tokenizer (whole K'iche' words; see Method notes).

## Table 1

**Table 1.** Cross-edition agreement of narrative segmentations (k = 6)

| Method | ARI: K'iche' ↔ K'iche' (10 pairs) | chance | ARI: K'iche' ↔ translation (15 pairs) | chance | F1: K'iche' ↔ K'iche' (10 pairs) | chance | F1: K'iche' ↔ translation (15 pairs) | chance |
|---|---|---|---|---|---|---|---|---|
| TF-IDF → Ward | 0.81 (0.71–0.94) | 0.42 | 0.78 (0.62–1.00) | 0.42 | 0.94 (0.86–1.00) | 0.75 | 0.94 (0.86–1.00) | 0.80 |
| LSA → Ward | 0.83 (0.71–0.95) | 0.48 | 0.80 (0.69–1.00) | 0.46 | 0.97 (0.86–1.00) | 0.86 | 0.94 (0.86–1.00) | 0.80 |
| Cosine-Sim → Ward | 0.80 (0.66–0.92) | 0.44 | 0.70 (0.49–0.88) | 0.43 | 0.97 (0.86–1.00) | 0.80 | 0.94 (0.83–1.00) | 0.79 |
| NMF → argmax | 0.94 (0.84–1.00) | 0.44 | 0.80 (0.69–0.87) | 0.47 | 1.00 (1.00–1.00) | 0.80 | 1.00 (1.00–1.00) | 0.80 |

*Mean pairwise agreement (range across pairs) between editions' segmentations at k = 6, 33 equal-token chunks, min_df = 5, max_df = 0.5, n-grams 1–1. ARI = adjusted Rand index; F1 = boundary F1 with ±1-chunk tolerance. Chance = 95th percentile of scores after cyclically shifting one sequence of each pair, pooled within each pair group. Every mean exceeds its chance value; empirical p ≤ 0.006 throughout (pairs share editions, so p is descriptive).*

**Table 1a.** Sensitivity: without Christenson's Ximénez

| Method | ARI: K'iche' ↔ K'iche' (6 pairs) | chance | ARI: K'iche' ↔ translation (12 pairs) | chance | F1: K'iche' ↔ K'iche' (6 pairs) | chance | F1: K'iche' ↔ translation (12 pairs) | chance |
|---|---|---|---|---|---|---|---|---|
| TF-IDF → Ward | 0.81 (0.71–0.92) | 0.42 | 0.81 (0.66–1.00) | 0.44 | 0.95 (0.86–1.00) | 0.79 | 0.96 (0.86–1.00) | 0.80 |
| LSA → Ward | 0.83 (0.71–0.95) | 0.47 | 0.80 (0.69–1.00) | 0.45 | 0.98 (0.86–1.00) | 0.81 | 0.96 (0.86–1.00) | 0.80 |
| Cosine-Sim → Ward | 0.83 (0.77–0.92) | 0.42 | 0.73 (0.56–0.88) | 0.42 | 0.98 (0.87–1.00) | 0.80 | 0.96 (0.87–1.00) | 0.79 |
| NMF → argmax | 0.92 (0.84–1.00) | 0.44 | 0.80 (0.69–0.87) | 0.47 | 1.00 (1.00–1.00) | 0.80 | 1.00 (1.00–1.00) | 0.80 |

*Same setting. Christenson's Ximénez is close to the same text as Ximénez; dropping it tests whether near-duplicate transcriptions inflate K'iche' ↔ K'iche' agreement.*

### Reading the tables

- **Every method beats chance on both metrics, within K'iche' and across translation.** ARI is about 0.27–0.50 above chance; boundary F1 about 0.11–0.20 above (its chance level is high because F1 with tolerance is forgiving).
- **The near-duplicate transcriptions don't drive the result.** Dropping Christenson's Ximénez changes no mean by more than 0.05.
- **NMF boundaries agree perfectly within ±1 chunk.** All 8 editions place their seven boundaries near chunks 2, 6, 8–9, 13–14, 18–19, 20 and 28–29, and every edition, Spanish and English included, reads **A B C D C A E F**.

## Choosing k (Table 2, to draft)

k = 6 comes from the grid search pages, which choose k by **excess over chance** (observed − chance 95th percentile) on **K'iche' ↔ K'iche' pairs only**, holding the translations out. Settings: n_chunks 10–40 (ARI) or 15–40 (boundary F1), max_df 0.30–0.70, min_df 5, k ≥ 3, k ≤ n_chunks / 3.

| Grid search | Methods | k\* | Held-out K'iche' ↔ translation excess at k\* |
|---|---|---|---|
| ARI | TF-IDF, LSA, Cosine-Sim, NMF | 6, 6, 6, 6 | +0.43, +0.36, +0.38, +0.49 |
| Boundary F1 (±1), single method | TF-IDF, LSA, Cosine-Sim | 6, 6, **3** | +0.28, +0.24, +0.46 |
| Boundary F1 (±1), Ensemble (recommended: 40 chunks, max_df 0.55) | TF-IDF, LSA, Cosine-Sim, NMF | 6, 6, **3**, **3** | +0.26, +0.24, +0.55, +0.25 |

Chosen on all pairs, k\* is 6 for TF-IDF, LSA and Cosine-Sim on both metrics. Values read from the app on 2026-10-21 after the tokenizer fix (not yet scripted). If this becomes a published table, extend `agreement_table.py` to produce it.

- **ARI chooses k; boundary F1 measures the divisions.** ARI (agreement on the *classes*) gives k\* = 6 for all four methods. Boundary F1 (agreement on the *breaks*) gives 6 for TF-IDF and LSA but 3 for Cosine-Sim and NMF: with whole-word tokens there is a strong coarse three-part division, and for TF-IDF and LSA k = 6 beats k = 3 only narrowly (mean excess +0.13 vs +0.12). Since k counts paradigmatic classes, ARI is the right selector; the k = 3 boundary peak is a finding about the text's coarse divisions.
- **Boundary F1 also peaks at the coarsest levels (k = 2 or 3).** Editions agree strongly on where the main division falls. That is a real result about the text's primary breaks, worth a sentence in the essay, but a different question from the episode structure, so k\* is chosen among k ≥ 3.
- **At high k, boundary excess falls to about zero** (e.g. −0.00 at k = 11 for TF-IDF).

## Choosing n_chunks (Table S1)

n_chunks is a choice of resolution. Cross-edition agreement can't choose it: excess over chance isn't comparable across n_chunks, because the chance baseline falls as chunks get fewer. It is chosen instead by **stability**: at each n, every edition's k = 6 partition is compared token by token with its partitions at n ± 10%. Stability rises until about 32 chunks and then plateaus. The rule takes the **plateau onset**, the coarsest stable resolution (stability-based model selection: Ben-Hur, Elisseeff & Guyon 2002; Lange, Roth, Braun & Buhmann 2004). It never uses the editors, so the editorial test stays independent. k is chosen first; that order is harmless because k\* = 6 at every n. The original 33 was chosen by eye from the scree and dendrogram; the rule puts the onset at 32, so the visual choice sat at the start of the plateau.

Regenerate with `python notebooks/ensemble/resolution_table.py`.

**Table S1.** Sensitivity of the main results to resolution (n_chunks)

| n_chunks | Tokens/chunk | Stability | k* (ARI) | Held-out excess (k = 6) | Sequence (k = 6) | Editors beaten (chunks) | Editors beaten (chapters) | Exact at ch. 9/15/41/65 |
|---|---|---|---|---|---|---|---|---|
| 18 | 1,628 | 0.694 | 6 | +0.38 | A B C D C A E F | 2 of 7 | 3 of 7 | ✓✓·✓ |
| 20 | 1,466 | 0.707 | 6 | +0.37 | A B C D C A E F | 4 of 7 | 0 of 7 | ✓··· |
| 25 | 1,172 | 0.719 | 6 | +0.30 | A B C D C A E F | 4 of 7 | 3 of 7 | ✓·✓· |
| 30 | 977 | 0.753 | 6 | +0.29 | A B C D C A E F | 4 of 7 | 6 of 7 | ✓✓✓· |
| **32** | 916 | 0.757 ▮ | 6 | +0.28 | A B C D C A E F | 4 of 7 | 4 of 7 | ✓·✓✓ |
| **33** | 888 | 0.757 ▮ | 6 | +0.30 | A B C D C A E F | 4 of 7 | 6 of 7 | ✓✓✓✓ |
| 35 | 837 | 0.767 ▮ | 6 | +0.25 | A B C D C D A E F | 4 of 7 | 6 of 7 | ✓✓✓· |
| 38 | 771 | 0.776 ▮ | 6 | +0.29 | A B C D C A E F | 4 of 7 | 4 of 7 | ✓·✓✓ |
| 40 | 733 | 0.773 ▮ | 6 | +0.29 | A B C D C A E F | 4 of 7 | 5 of 7 | ✓✓·✓ |
| 45 | 651 | 0.759 ▮ | 6 | +0.26 | A B C D C A E F E F | 2 of 7 | 3 of 7 | ✓·✓· |
| 50 | 586 | 0.761 ▮ | 6 | +0.27 | A B C D C D A E F | 6 of 7 | 3 of 7 | ✓✓·· |

*TF-IDF → Ward, min_df = 5; k* and held-out excess averaged over max_df 0.30–0.70; sequence and editorial agreement for Christenson's K'iche' text at max_df = 0.5. Stability = token-level ARI between the k = 6 partition at n and at n ± 10%, mean over the 8 editions, smoothed; ▮ = on the stability plateau (32–50). The rule sets n_chunks at the plateau onset (**32**); results are reported at **33**. Editors beaten = editors whose part divisions the segmentation matches above chance (boundary F1; ±1 chunk, or exact chapters).*

- **Invariant across the plateau:** k\* = 6, held-out agreement above chance, 4 of 7 editors beaten at chunk level (32–40), and the sequence A B C D C A E F (an extra D at 35 and 50; a repeated E F at 45).
- **Sensitive:** the chapter-level prediction. 4–6 of 7 editors across 32–40; 6 of 7 with all four shared part starts exact only at 33. Report the band, not the best point. At the rule's own value (32) it is 4 of 7.

## Method notes

- **Chance baseline:** for each pair of editions, one cluster sequence is rotated by every offset and the score recomputed. This keeps each edition's block structure and label proportions and breaks only the alignment. Each pair group is compared with its own pairs' rotated scores.
  - For ARI, all offsets from 1 to n−1 are used.
  - For boundary F1, offsets within the tolerance are skipped, since F1 counts those as matches.
- **Why not ARI ≈ 0 as chance:** misaligned block sequences score about 0.13–0.29 on average, and up to 0.77 at k = 2.
- **Boundary tolerance is fixed in chunks**, not scaled by k. The old τ = α/2(k−1) assumed k−1 boundaries, but recurring clusters produce more (about 7 at k = 6, 15 at k = 12), so k-scaled windows overlapped and even misaligned sequences scored near 1.
- **Why k ≤ n_chunks / 3:** near k = n_chunks, partitions approach one chunk per cluster and agreement becomes trivial. k\* = 6 is the same for caps of 2, 3 or 4 chunks per cluster.
- **Reproducibility:** NMF is seeded (`model.random_state` in config). Before 2026-10-20 it wasn't, and NMF labels at k ≥ 9 varied between runs.
- **Tokenization:** until 2026-10-21 the app's vectorizers used scikit-learn's default token pattern, which split K'iche' words at the glottal-stop apostrophe (k'ajolom → "ajolom") and dropped one-letter words; about 40% of tokens in Christenson's and Colop's orthography were affected. All vectorizers now keep the parser's tokens whole (`model.token_pattern: '\S+'`), as `local_lib`'s NarrativeParser does. The tables above are from the fixed pipeline.
- **Alignment caveat:** chunks are aligned by token position, and translations expand and contract unevenly. Boundary F1 absorbs drift within the tolerance; ARI doesn't, so its cross-language scores are conservative.
- **The NMF ↔ HAC match belongs elsewhere.** ARI 0.92 against a chance 95th percentile of 0.41 (single edition, k = 6) is a within-edition result for the topic-model section.
- **Kept out of the tables:** Hamming distance (depends on letter order) and boundary F1 at high k (at chance).

## Status

- [x] Shared pipeline module (`grid_search_boundary_core`)
- [x] Grid searches choose k by excess over chance, K'iche'-only selection with held-out translations
- [x] Comparison page reports per-group agreement, range, chance and p at the chosen setting
- [x] Table 1 and 1a generated by script
- [ ] Script Table 2 (grid k\* and held-out scores) if it is published
- [ ] Decide whether to discuss the coarse (k = 2–3) primary divisions in the essay
- [x] Tokenizer fixed and tables regenerated (2026-10-21)
- [x] n_chunks chosen by stability (plateau onset 32; Table S1)
- [ ] Decide the reporting n_chunks: 33 (current) or the rule's 32
