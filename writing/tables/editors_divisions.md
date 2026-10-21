**Table 3.** Agreement with editorial divisions (Christenson's K'iche' text, k = 6)

| Reference (part starts) | Chunk class changes | Resonance peaks | Chapter classes (±0 ch) | Chapter recall |
|---|---|---|---|---|
| Brasseur 1861 (15, 41, 57) | 0.52 (0.58) | 0.40 (0.60) | **0.27** (0.13) | 0.67 |
| Recinos 1947 (2, 15, 41, 57) | 0.46 (0.57) | 0.36 (0.65) | **0.25** (0.24) | 0.50 |
| Edmonson 1971 (9, 15, 41) | **0.73** (0.55) | **0.60** (0.52) | **0.40** (0.12) | 1.00 |
| Tedlock 1985 (10, 15, 41, 65) | **0.83** (0.67) | **0.73** (0.65) | **0.38** (0.24) | 0.75 |
| Colop 2011 (9, 15, 41, 65) | **0.83** (0.66) | **0.73** (0.65) | **0.50** (0.12) | 1.00 |
| Bazzett 2018 (9, 15, 41) | **0.73** (0.55) | **0.60** (0.52) | **0.40** (0.12) | 1.00 |
| Sotelo 2018 (22, 41, 57) | 0.52 (0.61) | 0.40 (0.60) | 0.13 (0.13) | 0.33 |
| Author's 8 divisions (9, 15, 19, 22, 41, 48, 65) † | **0.92** (0.73) | **0.78** (0.75) | **0.42** (0.21) | 0.57 |

*Boundary F1 (chance 95th percentile) against each edition's part starts, given as Christenson chapters, in Christenson's K'iche' text. TF-IDF → Ward, k = 6, 33 chunks, min_df = 5, max_df = 0.5. Chunk-level detectors match within ±1 chunk (nearest chunk edge); chapter classes are each chapter's nearest class centroid (out of sample), matched within ±0 chapters. Resonance peaks use window 3 and the same boundary budget as the class changes. Bold = above chance. Editors above chance: chunk class changes 4 of 7, resonance 4 of 7, chapter classes 6 of 7. Predicted chapter divisions: 9, 15, 16, 21, 26, 41, 43, 44, 47, 53, 54, 65. Editors agree with each other at mean F1 0.64; chunk class changes agree with them at 0.66. † Not independent of the models.*

**Table 3a.** Division starts recognized by editors or the author, and which predictors mark them

| Chapter | Title | Editors | Author | Chunk class change | Resonance | Chapter classes (exact) |
|---|---|---|---|---|---|---|
| 2 | The Primordial World | 1 (Recinos) |  |  |  |  |
| 9 | The Pride of Seven Macaw Before the Dawn | 3 (Edmonson, Colop, Bazzett) | ✓ | ✓ | ✓ | ✓ |
| 10 | The Fall of Seven Macaw and His Sons | 1 (Tedlock) |  | ✓ | ✓ |  |
| 15 | The Tale of the Father of Hunahpu and Xbalanque | 6 (Brasseur, Recinos, Edmonson, Tedlock, Colop, Bazzett) | ✓ | ✓ | ✓ | ✓ |
| 19 | The Maiden Lady Blood and the Tree of One Hunahpu | 0 | ✓ | ✓ |  |  |
| 22 | Hunahpu and Xbalanque in the House of the Grandmother | 1 (Sotelo) | ✓ | ✓ | ✓ |  |
| 41 | The Creation of Humanity | 7 (Brasseur, Recinos, Edmonson, Tedlock, Colop, Bazzett, Sotelo) | ✓ | ✓ | ✓ | ✓ |
| 48 | The Arrival at Tulan | 0 | ✓ | ✓ | ✓ |  |
| 57 | Offerings Are Made to the Gods | 3 (Brasseur, Recinos, Sotelo) |  |  |  |  |
| 65 | The Sons of the Progenitors Journey to Tulan | 2 (Tedlock, Colop) | ✓ | ✓ | ✓ | ✓ |
