**Table 2.** Choosing k by cross-edition agreement, translations held out

| Metric | Method | k* (K'iche' pairs) | Combinations choosing it | Mean excess at k* | Held-out excess (n = 32) | Held-out excess (grid mean) | k* (all pairs) |
|---|---|---|---|---|---|---|---|
| ARI | TF-IDF | **6** | 28 of 58 | +0.37 | +0.33 | +0.30 | 6 |
| ARI | LSA | **6** | 29 of 58 | +0.39 | +0.27 | +0.32 | 6 |
| ARI | Cosine-Sim | **6** | 20 of 58 | +0.38 | +0.30 | +0.31 | 6 |
| ARI | NMF | **6** | 32 of 58 | +0.42 | +0.32 | +0.33 | 6 |
| Boundary F1 (±1) | TF-IDF | **6** | 12 of 53 | +0.13 | +0.12 | +0.10 | 6 |
| Boundary F1 (±1) | LSA | **6** | 8 of 53 | +0.14 | +0.22 | +0.12 | 6 |
| Boundary F1 (±1) | Cosine-Sim | **3** | 24 of 53 | +0.14 | +0.07 | +0.03 | 6 |
| Boundary F1 (±1) | NMF | **3** | 24 of 53 | +0.15 | +0.03 | +0.07 | 6 |

*k\* = the k ≥ 3 with the highest mean excess over chance (observed − 95th percentile of cyclic-shift scores) across the grid: n_chunks 10–40 (ARI) or 15–40 (boundary F1), max_df 0.30–0.70, min_df = 5, k ≤ n_chunks / 3. Chosen on K'iche' ↔ K'iche' pairs; held-out excess is K'iche' ↔ translation agreement at k\*, at the reporting setting (n_chunks = 32, max_df = 0.5) and averaged over the grid. ARI, which measures agreement on the classes, chooses k; boundary F1 measures agreement on the breaks. ARI k\* is 6, 6, 6, 6 for TF-IDF, LSA, Cosine-Sim, NMF.*
