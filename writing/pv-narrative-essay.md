# The Narrative Structure of the Popol Wuj

## Abstract

This essay proposes a novel division of the Popol Wuj into parts and chapters based on the application of structuralist narratology supported by classical text analytic methods. It is proposed that the text comprises eight chapters belonging to two parts: part one contains five chapters oriented toward events before the first dawn, while part two contains three chapters concerning events following the dawn. These divisions form an integrated mythohistoric sequence that connects the actions of metahuman agents, or gods, to those of divine kings up to the time of the Spanish conquest. The connection between myth and history is represented in temporal and causal terms, in the sense of myth preceding and producing history, and structurally, in the sense that the mythic and historic chapters exhibit a parallel structure centered on the pivotal act of sacrifice. It is argued that although the text may have been composed in direct response to Vico’s *Theologia Indorum*, the thematic unity of the text exhibits a deep patterning reflective of an indigenous Maya historicity.

## Introduction

The K’iche’ Maya Popol Wuj[^1] is arguably the most significant surviving indigenous written text from the Americas. Its combination of word length, narrative sweep, and thematic unity have no counterpart in the Western hemisphere among extant literary artifacts. From a comparative perspective the Popol Wuj belongs to a mythohistoric genre that includes the Babylonian *Epic of Gilgamesh*, the Indian *Ramayana*, the Persian *Shahnameh*, and other foundational texts that recount the exploits of metahuman beings and their connections to living kings. Since coming into the hands of European scholars in the mid-nineteenth century, the text has been translated into at least twenty-five languages, and new editions, both scholarly and popular, continue to appear.

[^1]: This essay adopts this spelling over the more common *Popol Vuh*. By PW, we refer to the text contained in the manuscript composed in the 16^th^ century.

Because of its unique status, the Popol Wuj has long been regarded by scholars as a “window into the mind of the Ancient Maya” (Christenson and Sachse 2021: 3), providing a view relatively untainted by the beliefs and practices brutally imposed by their Spanish Catholic conquerors. Following the lead of Michael Coe (1973), the text has served as a key to the interpretation of numerous scenes depicted on carved and painted artifacts from the lowland preclassic and classic periods. Specific characters and episodes from the story have been identified on items dated to 300 BCE, two millennia before the composition of the current text.

Recently this view has been challenged.

Corroborated by the arbitrary nature of its assigned part and chapter divisions in its various editions.

Also argued that it is a western imposition, epistemic imperialism.

Unity on the division at Ch 41.

Continue to state hypothesis ...

## Mythohistory

## Structuralist narratology

## Existing divisions

# Method 

Standard TF-IDF down-weights terms that appear in many documents, using document frequency (DF) — the count of documents containing a term — as a proxy for informativeness. This works well when documents are topically diverse and term distributions are roughly uniform within documents. In highly formulaic corpora, however, DF can underestimate the pervasiveness of certain terms: a term that appears dozens of times within a single chunk contributes no more to its DF than one that appears once. To address this, we substitute corpus frequency (CF) — the total count of a term across all corpus segments — for DF in the IDF calculation, yielding an Inverse Corpus Frequency (ICF) weighting. CF-IDF penalizes terms more aggressively in proportion to their raw frequency across the corpus, making it better suited to oral-traditional and ritual texts characterized by dense, structured repetition. The resulting weights are L2-normalized per document, preserving the angular geometry required for cosine-based similarity measures and hierarchical clustering with Ward linkage.


Here's the full formulation, following your code precisely.

**Corpus Frequency (CF)**

$$\text{CF}(t) = \sum_{d \in D} \text{count}(t, d)$$

This is just the total number of times term $t$ appears across all documents (chunks) in corpus $D$ — i.e., `TF.sum()`.

**Inverse Corpus Frequency (ICF)**

$$\text{ICF}(t) = \log_2\left(\frac{|D| + 1}{\text{CF}(t) + 1} + 1\right)$$

where $|D|$ is the number of chunks (`n_chunks`). The $+1$ inside the fraction is additive smoothing; the outer $+1$ ensures the log is always positive.

**TF-ICF**

$$\text{TF-ICF}(t, d) = \text{count}(t, d) \cdot \text{ICF}(t)$$

**L2-normalized TF-ICF**

$$\widehat{\text{TF-ICF}}(t, d) = \frac{\text{TF-ICF}(t, d)}{\sqrt{\sum_{t' \in d} \text{TF-ICF}(t', d)^2}}$$

This is the final `self.TFIDF`, normalized column-wise (per document) via `div(l2_norm, axis=0)`.

---

Note that the key distinction from standard TF-IDF worth flagging in a methods note is that canonical IDF uses $\text{DF}(t) = |\{d \in D : t \in d\}|$ (a document count), whereas your ICF uses the raw corpus frequency $\text{CF}(t)$, making it sensitive to intensity of use rather than breadth of use.