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
