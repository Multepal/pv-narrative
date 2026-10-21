**Table S1.** Sensitivity of the main results to resolution (n_chunks)

| n_chunks | Tokens/chunk | Stability | k* (ARI) | Held-out excess (k = 6) | Sequence (k = 6) | Editors beaten (chunks) | Editors beaten (chapters) | Exact at ch. 9/15/41/65 |
|---|---|---|---|---|---|---|---|---|
| 18 | 1,628 | 0.694 | 6 | +0.38 | A B C D C A E F | 2 of 7 | 3 of 7 | ✓✓·✓ |
| 20 | 1,466 | 0.707 | 6 | +0.37 | A B C D C A E F | 4 of 7 | 0 of 7 | ✓··· |
| 25 | 1,172 | 0.719 | 6 | +0.30 | A B C D C A E F | 4 of 7 | 3 of 7 | ✓·✓· |
| 30 | 977 | 0.753 | 6 | +0.29 | A B C D C A E F | 4 of 7 | 6 of 7 | ✓✓✓· |
| **32** | 916 | 0.757 ▮ | 6 | +0.28 | A B C D C A E F | 4 of 7 | 4 of 7 | ✓·✓✓ |
| 33 | 888 | 0.757 ▮ | 6 | +0.30 | A B C D C A E F | 4 of 7 | 6 of 7 | ✓✓✓✓ |
| 35 | 837 | 0.767 ▮ | 6 | +0.25 | A B C D C D A E F | 4 of 7 | 6 of 7 | ✓✓✓· |
| 38 | 771 | 0.776 ▮ | 6 | +0.29 | A B C D C A E F | 4 of 7 | 4 of 7 | ✓·✓✓ |
| 40 | 733 | 0.773 ▮ | 6 | +0.29 | A B C D C A E F | 4 of 7 | 5 of 7 | ✓✓·✓ |
| 45 | 651 | 0.759 ▮ | 6 | +0.26 | A B C D C A E F E F | 2 of 7 | 3 of 7 | ✓·✓· |
| 50 | 586 | 0.761 ▮ | 6 | +0.27 | A B C D C D A E F | 6 of 7 | 3 of 7 | ✓✓·· |

*TF-IDF → Ward, min_df = 5; k* and held-out excess averaged over max_df 0.30–0.70; sequence and editorial agreement for Christenson's K'iche' text at max_df = 0.5. Stability = token-level ARI between the k = 6 partition at n and at n ± 10%, mean over the 8 editions, smoothed; ▮ = on the stability plateau (32–50). The rule sets n_chunks at the plateau onset (**32**); results are reported at **32**. Editors beaten = editors whose part divisions the segmentation matches above chance (boundary F1; ±1 chunk, or exact chapters).*
