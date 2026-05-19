# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research project intended to model the narrative structure of the Popol Wuj. It is also an experiment in computer-aided hermeneutics: I want to know which models and patterns can be used to do interpretive work in the context of understanding the K'iche' Maya creation narrative the *Popol Wuj*,

In this project, I want to operationalize structuralist narratology, especially as articulated by the anthropologist Claude Lévi-Strauss in "The Structural Study of Myth" (1955). The basic idea is captured in the concept of structure vs event, where structure means paradigmatic ordering of cultural concepts and event means narrative ordering of events and episodes in a story. We operationalize by treating clusters, principal components, and topics as proxies for cultural concepts.

The project uses Python, Pandas, SKLearn, Plotly, Streamlit and various other tools to create an interactive app to explore the various models illuminate the text. The app is meant to drive research, to present results – yet.

My strategy with regard to the application of computational methods and statistical models is as follows:

1.  Use the temas data in `notebooks/ximenez` to create a strip plot showing the distribution of named entities (aka temas or topics) over narrative time. This provides motivation and scaffolding for framing an approach to the interpretive question of how the text is organized thematically. The temas data was created by students and members of the NSF funded Multepal Project.
2.  For each of the editions, listed in `config.yaml` and located under `notebooks`, create TOKEN tables. Assign a chunk feature to this table to generate strings (docs) that can be fed to a document-term matrix and then, after filtering by min and max df, an L2-normalized weighted version of this. This representation is the foundation for all the other models.
3.  Use hierarchical agglomerative clustering (HAC) with euclidean distance and ward linkage to generate a dendrogram. Pick optimal cut-off, yielding k clusters, using grid search and then assign these clusters to chunks based on clades. Represent the distribution of clusters over time in a heatmap to show the paradigmatic and syntagmatic structure of the text. Note that the grid search uses a boundary concordance with an ensemble of models (not just HAC).
4.  Use non-negative matrix factorization topic modeling (NMF) to extract rich topics and a more noisy but interpretively rich topic/narrative heatmap. Choose k based on HAC. Associate NFM topics with HAC clusters.
5.  Use latent semantic analysis (LSA) to generate a latent semantic subspace of components to explore structural relationships among the clusters. These provide insights into the oppositional relationship between topics in the domain of synchronic semantics vs diachronic narrative (syuzhet).
6.  Apply the results of 3, 4, and 5 back to the temas artifact from 1 to create a rich interactive concordance of temas over time groups by clusters.
7.  Apply NMF as a model to predict Christenson in order to infer textual divisions.

## Some Rules

Always commit any existing uncommitted changes before making new code changes.

Always be ready to roll back to the state of the code before the plan was executed.

Always show a mug of beer 🍺 when a task is completed and you are awaiting further instructions.

## Running the Streamlit App

Build and run all code in Python using the Conda environment `pv-narrative`.

Run Streamlit from the root of the project like so:

``` bash
streamlit run notebooks/streamlit_app.py
```

Install dependencies first if needed (but the environment should already have them):

``` bash
pip install -r notebooks/requirements.txt
```

## Data Pipeline

Each source edition has its own subdirectory under `notebooks/` (e.g., `notebooks/colop/`, `notebooks/ajtzibab/`). Within each, a two-notebook pipeline runs sequentially to generate the tables used by this project and app:

1.  **`01-*-parse*.ipynb`** — imports raw source text from `textos/`, parses it into structured tables, and writes CSVs (DOC, TOKEN, VOCAB) into the source directory.
2.  **`02-*-model*.ipynb`** — reads TOKEN.csv, applies TF-IDF + NMF/HAC/PCA, and writes model output CSVs (THETA, PHI, CHUNK, TFIDF, CLUSTER, etc.) plus figures.

The `notebooks/ensemble/` directory aggregates TOKEN files across all sources. It also does some other explorations.

## Data Model

The project uses a relational data model to represent the corpus.

The reference model consists of the following tables:

1.  LIB
2.  DOC
3.  TOKEM
4.  VOCAB
5.  CHUNK

## Local Library (`local_lib/`)

Here are some shared Python modules used by the research notebooks, but not necessarily the Streamlit app:

-   `narrative_model.py` / `narrative_parser.py` — `NarrativeModel` and `NarrativeParser` classes encapsulating the full modeling pipeline (TF-IDF, HAC, NMF, PCA)
-   `hac.py` / `hac2.py` — hierarchical agglomerative clustering utilities
-   `langmod_class.py` / `langmod_funcs.py` — language modeling helpers
-   `textimporter.py` / `textparser.py` — text ingestion utilities
-   `eta/`, `mazo/` — additional utility subpackages

## TOKEN.csv Schema

TOKEN files are the central data format. Columns up to (not including) `token_str` form the OHCO index (Ordered Hierarchy of Content Objects — e.g., book, chapter, paragraph, token). `term_str` holds the normalized term used for modeling.