# About This App

## The Popol Wuj

The *Popol Wuj* (also spelled *Popol Vuh*) is the foundational mythological and historical narrative of the K'iche' Maya people of the Guatemalan highlands. It recounts the creation of the world, the exploits of the Hero Twins, and the origins of the K'iche' nobility. Originally composed in the 1550s, possibly in response to Vico's *Theologia Indorum*, it was transcribed into K'iche' and Spanish using the Latin alphabet in the mid-sixteenth century.

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
| **Overlap**    | Fraction of tokens shared between adjacent chunks   |
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

This tool is part of the **Multepal** project, a digital humanities initiative studying the *Popol Wuj* through computational text analysis.