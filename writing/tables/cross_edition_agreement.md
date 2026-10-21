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
