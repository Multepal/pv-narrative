**Table 1.** Cross-edition agreement of narrative segmentations (k = 6)

| Method | ARI: K'iche' ↔ K'iche' (10 pairs) | chance | ARI: K'iche' ↔ translation (15 pairs) | chance | F1: K'iche' ↔ K'iche' (10 pairs) | chance | F1: K'iche' ↔ translation (15 pairs) | chance |
|---|---|---|---|---|---|---|---|---|
| TF-IDF → Ward | 0.86 (0.73–1.00) | 0.43 | 0.78 (0.55–0.93) | 0.44 | 0.96 (0.89–1.00) | 0.82 | 0.95 (0.86–1.00) | 0.83 |
| LSA → Ward | 0.88 (0.77–1.00) | 0.46 | 0.74 (0.64–0.85) | 0.47 | 1.00 (1.00–1.00) | 0.80 | 1.00 (1.00–1.00) | 0.78 |
| Cosine-Sim → Ward | 0.86 (0.73–1.00) | 0.43 | 0.76 (0.64–0.86) | 0.46 | 0.96 (0.89–1.00) | 0.82 | 0.97 (0.89–1.00) | 0.80 |
| NMF → argmax | 0.88 (0.77–1.00) | 0.46 | 0.75 (0.61–0.92) | 0.43 | 1.00 (1.00–1.00) | 0.80 | 0.95 (0.80–1.00) | 0.82 |

*Mean pairwise agreement (range across pairs) between editions' segmentations at k = 6, 32 equal-token chunks, min_df = 5, max_df = 0.5, n-grams 1–1. ARI = adjusted Rand index; F1 = boundary F1 with ±1-chunk tolerance. Chance = 95th percentile of scores after cyclically shifting one sequence of each pair, pooled within each pair group. Every mean exceeds its chance value; empirical p ≤ 0.003 throughout (pairs share editions, so p is descriptive).*

**Table 1a.** Sensitivity: without Christenson's Ximénez

| Method | ARI: K'iche' ↔ K'iche' (6 pairs) | chance | ARI: K'iche' ↔ translation (12 pairs) | chance | F1: K'iche' ↔ K'iche' (6 pairs) | chance | F1: K'iche' ↔ translation (12 pairs) | chance |
|---|---|---|---|---|---|---|---|---|
| TF-IDF → Ward | 0.82 (0.73–0.92) | 0.42 | 0.77 (0.55–0.93) | 0.44 | 0.95 (0.89–1.00) | 0.82 | 0.95 (0.86–1.00) | 0.84 |
| LSA → Ward | 0.84 (0.77–0.92) | 0.42 | 0.75 (0.64–0.85) | 0.47 | 1.00 (1.00–1.00) | 0.80 | 1.00 (1.00–1.00) | 0.78 |
| Cosine-Sim → Ward | 0.82 (0.73–0.92) | 0.42 | 0.75 (0.64–0.86) | 0.44 | 0.95 (0.89–1.00) | 0.82 | 0.97 (0.89–1.00) | 0.82 |
| NMF → argmax | 0.88 (0.77–1.00) | 0.40 | 0.74 (0.61–0.85) | 0.43 | 1.00 (1.00–1.00) | 0.80 | 0.94 (0.80–1.00) | 0.82 |

*Same setting. Christenson's Ximénez is close to the same text as Ximénez; dropping it tests whether near-duplicate transcriptions inflate K'iche' ↔ K'iche' agreement.*
