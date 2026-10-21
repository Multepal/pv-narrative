Your two directions fit Lévi-Strauss's own two readings of a myth: down the columns (paradigm) and along the rows (syntagm). They also give the app a symmetry it doesn't yet show: **each axis has an exploration and an external check**. Below: (1) the overall shape, (2) theoretical grounding for the syntagmatic axis, (3) what Division Validation actually does, and (4) where every remaining page goes.

## 1. Overall shape

After Model → Choose k → Test, the argument forks:

| | **Paradigmatic axis (structure)** | **Syntagmatic axis (event)** |
|---|---|---|
| Question | What are the classes, and how do they relate? | How is the story cut up and sequenced? |
| Explore | Relations (PCA/LSA), Content (NMF Topics) | Boundaries, sequence and recurrence |
| External check | **Temas**: do the classes match independently annotated entities and themes? | **Editors' divisions**: do the boundaries match the editorial tradition? |

Then the two axes come together in **prediction**: applying the model to units it never saw (chapters), to infer the text's divisions and themes.

## 2. Theoretical grounding for the syntagmatic axis

Lévi-Strauss himself is thin here: he treats the syntagm as surface and moves quickly to the columns. So the syntagmatic axis needs other sources, and it needs a clear statement of what a "boundary" means. There are three distinct notions, and the app uses all three:

1. **A boundary is a change of paradigmatic class.** The story is the sequence of mythemes, and an episode is a run of one class. This is Lévi-Strauss's implicit model, and it's what the k = 6 boundaries are. *Theory:* Lévi-Strauss (1955) on reading the score horizontally; Propp's functions as a sequence; Barthes' "Introduction to the Structural Analysis of Narratives" (1966), where units of content are distributed along the sequence.
2. **A boundary is a discontinuity in the flow of the text**, a break in lexical cohesion, regardless of any classes. This is what Episode Boundaries computes. *Theory:* text segmentation by lexical cohesion (Hearst's TextTiling, 1997); novelty, transience and resonance. *Check the citation:* the page credits "Jurafsky et al. (2018) PNAS 115(10)". I believe that framework is Barron, Huang, Spang & DeDeo (2018), PNAS 115(18).
3. **A boundary is an editor's division.** This is interpretive and philological, and it carries a key point you already note: **the Ximénez manuscript is undivided**. Your notes record that Brasseur (1861) "introduces divisions". So there's no authorial ground truth; every division is a reading. *Theory:* the history of editing; Barthes' kernels (cardinal functions) vs. satellites (catalyses), and Chatman's version of the same distinction, as the narratological account of where the joints are.

This gives the syntagmatic argument a clean shape. **Do notions 1 and 2, derived independently, converge on notion 3?** If both class changes and cohesion breaks fall where the editors divide, the narrative's divisions are *overdetermined* by its language. That's a strong claim. It also fits computer-aided hermeneutics: the model is **one more reader in the editorial tradition**, judged by how it stands among the other readers, not against a ground truth that doesn't exist.

## 3. What "Division Validation" does

In plain terms:

1. Take the k = 6 sequence (A B C D C A E F) and mark its **7 boundaries**, where the class changes. That's notion 1.
2. Place each editor's part starts, and your 8 divisions, in the same text. They're recorded as Christenson chapters and converted to token positions.
3. Ask whether the model's boundaries land near the editors' boundaries **more often than the same sequence would if it were shifted out of alignment**. That's F1 against the rotation baseline.
4. Show **which** editorial breaks the model recovers (the consensus table) and **how the model stands among the editors** (the pairwise F1 matrix).

What it tests, theoretically: **the hypothesis that the editorial tradition divides the text where the paradigmatic classes change.** It's a genuine test because the parameters were fixed beforehand, by cross-edition agreement, and never tuned to the editors.

What it doesn't do:
- It doesn't use notion 2. Episode Boundaries could be scored the same way, which would complete the convergence argument.
- It can't score chapters, because chunks are too coarse.

The name undersells it. I'd rename it **"Editors' Divisions"** or **"The Model Among the Editors"**.

## 4. Where everything goes

| Section | Pages | Role |
|---|---|---|
| **1 · Model** | Clustering (TF-IDF) | The method |
| **2 · Choose k** | Choose k (ARI Grid Search) | Parameters chosen by cross-edition agreement |
| **3 · Test** | Cross-Edition Comparison, Robustness (Cosine-Sim) | Held-out translations; does the result depend on the method? |
| **4 · Paradigmatic Axis** | Relations (PCA/LSA), Content (NMF Topics), **Tema Distribution** | Oppositions, content, and the **thematic check** against annotated temas |
| **5 · Syntagmatic Axis** | *Segmentation:* Boundary Agreement, Episode Boundaries, Editors' Divisions, **Narrative Play** · *Sequence:* **Narrative Transitions**, **Narrative DCT** | Where the cuts fall, checked against editors; what follows what |
| **6 · Prediction** | **Chapter Topics**, **Christenson Text Reader** | The model applied to chapters; the text read through the model |
| **Resources** | Annotation Coverage, Entity Index | Data quality and cross-edition concordances behind the temas |

**Why each remaining page goes where it does:**
- **Narrative Play** is the syntagmatic axis's hermeneutic coda, so yes, it belongs under divisions. It shows the gap between where the model cuts (chunks) and where a reader cuts (chapters). That gap is interpretive *play*: what the model leaves undetermined and a reader has to settle.
- **Narrative Transitions** (the topic sequence and transition graph) is the syntagm's *grammar*, Propp-style: which class follows which.
- **Narrative DCT** describes how each class recurs over the narrative, from long arcs to rapid oscillation. Recurrence is how the syntagm loops back on the paradigm (the returning C and A). It's also where your chiasmus claims would get their evidence.
- **Tema Distribution** is the paradigmatic counterpart of Editors' Divisions: an independent annotation the classes can be checked against. Your CLAUDE.md gives it two roles: motivation in step 1, and the payoff in step 6. As a page, it fits best as the thematic check. The essay can still open with its strip plot.
- **Chapter Topics** is prediction, as you say: it projects the trained topics onto Christenson's 87 chapters (step 7 of your plan). It also solves a problem from the last session. It produces **chapter-level labels**, so Editors' Divisions could then score at chapter resolution, which chunks can't support.
- **Christenson Text Reader** is where the argument arrives: the text, divided and labeled, to be read.
- **Annotation Coverage** and **Entity Index** support the tema analysis rather than make arguments, so they go under resources.

**Suggested next steps:**
1. Reorganize the sidebar as above, and rename Division Validation.
2. Add Episode Boundaries' boundaries to the Editors' Divisions page as a second detector.
3. Feed Chapter Topics' chapter labels into the same page for chapter-level scoring.

The first is a config change. The other two are substantive. Shall I start with the reorganization? `Essay.md` still has uncommitted edits, which I'd commit first.

🍺