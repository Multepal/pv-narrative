# About this App

## The Popol Wuj

The *Popol Wuj* (also spelled *Popol Vuh*) is the foundational mythological and historical narrative of the K'iche' Maya people of the Guatemalan highlands. It recounts the creation of the world, the exploits of the Hero Twins and a host of other metahuman beings, as well as the origins of the K'iche' nobility. Originally composed in the 1550s, possibly in response to Vico's *Theologia Indorum*, it was transcribed into K'iche' and Spanish using the Latin alphabet in the mid-sixteenth century. Today the text is valued both as a profound symbol of indigenous Maya identity as well as a window into the history and beliefs of the Maya as they existed before arrival of the Spanish.

## Narrative Structure

The *Popol Wuj* has been the subject of continuous academic study since it was rediscovered by European scholars in the nineteenth century. For the most part, this scholarship has focused on the linguistic, orthographic, and poetic features of the text, along with the analysis of its rich historical and religious content. This has resulted in a number of reliable and insightful editions in a variety of languages.

Surprisingly, little attention has been paid to the overall narrative structure of the text beyond positing somewhat arbitrary divisions of parts and chapters. This neglect owes to the widely held view that this level is unimportant to an understanding of the story contained in the text. At best, the reasoning goes, such divisions merely help make the text more readable to a non-Maya audience; at worst they constitute a form of epistemic imperialism, since they impose a western framework that does not align with indigenous understandings of story telling.

The study of narrative structure, however, is a fundamental practice in the study of culture. An understanding of how stories are structured—their sequential and paradigmatic ordering of events and topics—potentially provides valuable insights into distinctive modes of apprehending and acting in the world. Beyond an understanding of what a story is about and in what manner it was composed, the study of narrative structure seeks to reveal the culturally specific logic of *how* things happen, the specific understandings of causality and order that govern practices ranging from cooking to the waging of war.

The research presented in this app is meant as a contribution to the study of the narrative structure of the *Popol Wuh* by means of what we may call computational criticism—the use of statistical models and computational procedures to uncover meanings and structures in texts that may evade close reading, especially in cases where the cultural and historical distances between author and reader is large. In this case, the research has focused on operationalizing the structural study of myth associated with the anthropologist Claude Lévi-Strauss. This idea is simple: take the fundamental concepts of syntagm and paradigm used to analyze myths and map them onto topic modeling, a widely used method in digital humanities to surface the latent semantic structures of texts and text collections. In this manner, the goal is to produce an objective—in the sense of shareable and reproducible—picture of the topical sequence of the text, and to see if this picture can shed light on the text's narrative structure.

One goal of this research is to explore what this method actually reveals, and how what it reveals varies with our choice of parameters, i.e. the many choices that are made in preparing and modeling the text quantitatively. This app is an essential part of this process: it allows us to vary such parameters as which words to include and exclude, the way in which we divide the text for its representation as a vector space, and so on. The effects of these changes are immediately visible.

## What This App Does

This app applies **topic modeling** to sliding windows of text from multiple editions and translations of the *Popol Wuj* to reveal its underlying narrative structure.

The two main visualizations are:

-   **Topic Heatmap** — rows are latent topics (paradigmatic structure), columns are sequential text chunks (syntagmatic flow). Color intensity shows how strongly each topic is present in each chunk.
-   **Cosine Distance Bar Chart** — measures how much the topic mixture shifts between adjacent chunks, highlighting narrative transitions.

## Controls

| Control        | Description                                         |
|----------------|-----------------------------------------------------|
| **Source**     | The edition or translation to analyze               |
| **Chunk size** | Number of tokens per text window                    |
| **Overlap**    | Fraction of tokens shared between adjacent chunks   |![alt text](image.png)
| **min_df**     | Minimum document frequency for vocabulary inclusion |
| **max_df**     | Maximum document frequency (filters stop-words)     |
| **Topics**     | Number of NMF topics to extract                     |
| **Top words**  | Number of top terms shown per topic                 |

## Sources

The app supports seven editions spanning K'iche', Spanish, and English:

-   **Ajtzibab 2025** — modern K'iche' edition
-   **Christenson 2007** — English translation with K'iche' interlinear
-   **Colop 2012** — Spanish translation from K'iche'
-   **Christenson's Ximénez** — Christenson's transcription of the Ximénez manuscript
-   **Ximénez** — the earliest known manuscript (c. 1701–1703)
-   **Recinos 1947** — influential Spanish translation
-   **Tedlock 1983** — English translation with performance notes

## Project

This tool is part of the [**Multepal** project](https://multepal.github.io/), a digital humanities initiative studying the *Popol Wuj* through computational text analysis.

------------------------------------------------------------------------

\(c\) 2026 Rafael C. Alvarado