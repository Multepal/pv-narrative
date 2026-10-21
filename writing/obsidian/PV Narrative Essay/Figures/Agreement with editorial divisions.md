---
created: 2026-10-21
status: tables generated (fixed tokenizer, 2026-10-21)
source: "notebooks/ensemble/division_table.py → writing/tables/editors_divisions.{csv,md}"
---

# Agreement with editorial divisions

Do the text's narrative boundaries fall where editors divide it? Two independent detectors, class changes (the syntagm as a sequence of paradigmatic classes) and resonance peaks (breaks in lexical continuity), plus an out-of-sample chapter-level prediction, are scored against seven editors' part divisions. The editors' divisions are independent of the models; the author's divisions are not.

> [!info] Regenerating
> Tables 3 and 3a are written by `python notebooks/ensemble/division_table.py` (project env), using the Streamlit app's own pipeline. Rerun after any pipeline change and paste `writing/tables/editors_divisions.md` here. Regenerated 2026-10-21 after the tokenizer fix (whole K'iche' words).

**Table 3.** Agreement with editorial divisions (Christenson's K'iche' text, k = 6)

| Reference (part starts) | Chunk class changes | Resonance peaks | Chapter classes (±0 ch) | Chapter recall |
|---|---|---|---|---|
| Brasseur 1861 (15, 41, 57) | 0.52 (0.58) | 0.40 (0.60) | **0.27** (0.13) | 0.67 |
| Recinos 1947 (2, 15, 41, 57) | 0.46 (0.57) | 0.36 (0.65) | **0.25** (0.24) | 0.50 |
| Edmonson 1971 (9, 15, 41) | **0.73** (0.55) | **0.60** (0.52) | **0.40** (0.13) | 1.00 |
| Tedlock 1985 (10, 15, 41, 65) | **0.83** (0.67) | **0.73** (0.65) | **0.38** (0.24) | 0.75 |
| Colop 2011 (9, 15, 41, 65) | **0.83** (0.66) | **0.73** (0.65) | **0.50** (0.21) | 1.00 |
| Bazzett 2018 (9, 15, 41) | **0.73** (0.55) | **0.60** (0.52) | **0.40** (0.13) | 1.00 |
| Sotelo 2018 (22, 41, 57) | 0.52 (0.61) | 0.40 (0.60) | 0.13 (0.13) | 0.33 |
| Author's 8 divisions (9, 15, 19, 22, 41, 48, 65) † | **0.92** (0.73) | **0.78** (0.75) | **0.42** (0.28) | 0.57 |

*Boundary F1 (chance 95th percentile) against each edition's part starts, given as Christenson chapters, in Christenson's K'iche' text. TF-IDF → Ward, k = 6, 33 chunks, min_df = 5, max_df = 0.5. Chunk-level detectors match within ±1 chunk (nearest chunk edge); chapter classes are each chapter's nearest class centroid (out of sample), matched within ±0 chapters. Resonance peaks use window 3 and the same boundary budget as the class changes. Bold = above chance. Editors above chance: chunk class changes 4 of 7, resonance 4 of 7, chapter classes 6 of 7. Predicted chapter divisions: 9, 15, 21, 26, 40, 41, 43, 44, 47, 53, 54, 65. Editors agree with each other at mean F1 0.64; chunk class changes agree with them at 0.66. † Not independent of the models.*

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

## Reading the tables

- **At chapter resolution the model predicts editorial divisions to the exact chapter.** Among the out-of-sample chapter-class changes are chapters 9, 15, 41 and 65: every part start of Edmonson, Colop and Bazzett, and all but one of Tedlock's (10). They beat chance against 6 of 7 editors.
- **Two independent detectors find the same divisions at chunk level.** Class changes and resonance peaks each beat chance against the same 4 editors. Class changes are the more robust: the same 4 in every edition, while resonance varies from 0 to 4.
- **No detector marks chapter 57** ("Offerings Are Made to the Gods"), where Brasseur, Recinos and Sotelo start Part IV. The language of the text supports the division at 65 ("The Sons of the Progenitors Journey to Tulan"), as in Tedlock and Colop. The editors who divide at 57 are dividing on other grounds.
- **The model stands among the editors** (mean F1 0.66) about as well as they stand among each other (0.64).

## Theory: three meanings of a boundary

1. A change of paradigmatic class: an episode is a run of one class (Lévi-Strauss's score read along the rows; Propp; Barthes 1966).
2. A break in lexical continuity: high novelty, low transience (Barron, Huang, Spang & DeDeo 2018; lexical-cohesion segmentation, Hearst 1997).
3. An editor's division: the Ximénez manuscript is undivided; divisions enter with Brasseur (1861). They are readings, not ground truth.

The test is whether 1 and 2, computed independently of each other and of the editors, converge on 3. Where they do, the editorial tradition's divisions are overdetermined by the language of the text, and the model is one more reader in that tradition.
