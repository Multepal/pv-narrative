Your instinct is right. You don't need four *clustering* methods. You need **one partition and several lenses on it**, each answering a different question. At the moment the app presents them as four rival clusterers, and that framing makes the extra ones look redundant.

## Suggested framing: one representation, one partition, three lenses

Everything starts from the same object: L2-normalized TF-IDF vectors for the chunks.

| Role | Method | Question it answers | Lévi-Strauss analogue |
|---|---|---|---|
| **Partition** (primary) | TF-IDF → Ward HAC | What are the paradigmatic classes, and where do they occur in the narrative? | The columns of his chart (bundles of relations), read against the rows (narrative sequence) |
| **Relations** | PCA/LSA components | How do the classes stand toward each other: along which axes, and at which poles? | Binary oppositions between bundles |
| **Content** | NMF topics | What is each class *about*, lexically? | What each bundle says |
| **Robustness** | Cosine-Sim (second-order) | Does the partition depend on raw word overlap, or does it hold when chunks are defined by their relations? | Meaning as relational (see below) |

So in the paper, TF-IDF HAC carries the main claim. LSA and NMF *interpret* that partition, and Cosine-Sim *checks* it. LSA and NMF also produce partitions of their own (in the grid searches, and NMF's argmax). Use those as **corroboration** that the same structure appears under a different model, not as competitors.

## Each method, more precisely

**TF-IDF → Ward.** Euclidean distance between L2-normalized vectors is a monotone function of cosine distance (‖a − b‖² = 2 − 2 cos θ). So this is plain "chunks are alike if they use the same words," and Ward builds the hierarchy. It's the simplest and most transparent choice, which is why it should be the primary one.

**LSA/PCA.** Its value is the **components, not the clusters**. Clustering in a 10-dimensional LSA space mostly reproduces the TF-IDF partition with less noise, which is why its k\* and agreement scores look so much like TF-IDF's. What only LSA gives you:
- **Signed axes.** Terms load at opposite poles of a component, and the clusters' average positions (centroids) fall on one side or the other. That's where the opposition argument lives.

Two cautions:
1. **The math doesn't make the oppositions; you have to argue them.** Components are orthogonal and signed by construction, so every one has two poles whether or not they're opposed in meaning. The interpretive claim is that the terms at the poles are semantically opposed, for example Xib'alb'a vs. lineage and rule. Make that case from the loadings, one component at a time.
2. **"LSA" means two different things in the app.** The LSA page uses *centered* PCA, which the page itself notes avoids a meaningless first component. The grid searches and the comparison page use *uncentered* TruncatedSVD. Pick one. Centered PCA is better for the opposition reading.

**NMF.** It gives the classes their **vocabulary** (the topic word lists), and its soft mixtures show what the hard clusters hide: chunks where two classes overlap, and gradual transitions. It's also the most independent of the four, because it factorizes the matrix rather than clustering it. That makes its ARI of 0.92 with HAC, and its perfect boundary agreement across editions, the strongest corroboration you have. I'd drop LDA from the main argument: it fits raw counts with a different model, and it agreed much less (ARI 0.54).

## Justifying the second-order clustering (Cosine-Sim)

Here each chunk is represented by its **row of the chunk-to-chunk similarity matrix**: its similarity to every other chunk. Two chunks are then close if they *resemble the same passages*, even when they share few words with each other. You can justify it on three grounds:

1. **Relational meaning.** In Saussure and Lévi-Strauss, a unit's value comes from its relations to the other units, not from its content. Second-order similarity puts that into practice: a chunk is defined by its place among the other chunks.
2. **Structural equivalence.** It's the network-analysis idea that nodes are equivalent when they have the same pattern of ties. It's also the intuition behind spectral clustering.
3. **Recurrence.** A chunk's profile records *which other passages it echoes*. That should sharpen the returning classes (the C and A that come back in A B C D C A E F), which is exactly the paradigmatic recurrence you're claiming.

Be honest about its limit, though. It's a transformation of the same matrix (it computes X·Xᵀ), not new evidence. The fair claim is that **the partition holds up when chunks are defined relationally rather than by their words**. That's a real robustness result, but a modest one. It belongs in a robustness paragraph or an appendix, not alongside TF-IDF as a co-equal method. Your results support that ranking: Cosine-Sim mostly matches TF-IDF and adds nothing it doesn't already show.

## What this means for the app and the paper

- **Paper:** present TF-IDF HAC as the method. LSA is how you read the relations between classes, and NMF is how you read their content. The agreement of NMF (and, more weakly, Cosine-Sim and LSA) shows the partition isn't an artifact of one model. The four-way table can stay, reframed as corroboration.
- **App:**
  - Clustering (TF-IDF) is the primary page.
  - The LSA page leads with components and centroids, and its clustering becomes secondary.
  - The Topics page leads with what each cluster contains, which is where the Topics vs. HAC Clusters section you just added fits.
  - Cosine-Sim moves to a robustness position, or into the grid's Ensemble mode only.
- **Validation step (from before):** run it on the TF-IDF partition. Report the other methods' boundary scores as corroboration.

One open question: should Ensemble mode keep giving all four methods equal weight when choosing parameters? Under this framing, TF-IDF should choose and the others should confirm. In practice it makes no difference now, since every method gives k\* = 6, but the logic should match how you describe it in the paper.

🍺