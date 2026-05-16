# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research project intended to model the narrative structure of the Popol Wuj.

It seeks to operationalize structuralist narratology, especially as articulated by the anthropologist Claude Lévi-Strauss in "The Structural Study of Myth" (1955).

It uses Python, Pandas, Plotyl, Streamlit and other tools to create an interactive app.

## Rules

ALWAYS commit any existing uncommitted changes before making new code changes.

ALWAYS be ready to roll back to the state of the code before the plan was executed.

## Running the Streamlit App

Build and run ALL code in Python using the conda environment `pv-narrative`.

``` bash
cd notebooks
streamlit run streamlit_app.py
```

Install dependencies first if needed:

``` bash
pip install -r notebooks/requirements.txt
```

## Architecture

### Data Pipeline

Each source edition has its own subdirectory under `notebooks/` (e.g., `notebooks/colop/`, `notebooks/ajtzibab/`). Within each, a two-notebook pipeline runs sequentially:

1.  **`01-*-parse*.ipynb`** — imports raw source text from `textos/`, parses it into structured tables, and writes CSVs (DOC, TOKEN, VOCAB) into the source directory.
2.  **`02-*-model*.ipynb`** — reads TOKEN.csv, applies TF-IDF + NMF/HAC/PCA, and writes model output CSVs (THETA, PHI, CHUNK, TFIDF, CLUSTER, etc.) plus figures.

The `notebooks/ensemble/` directory aggregates TOKEN files across all sources.

### Local Library (`local_lib/`)

Shared Python modules used by the research notebooks (not the Streamlit app):

-   `narrative_model.py` / `narrative_parser.py` — `NarrativeModel` and `NarrativeParser` classes encapsulating the full modeling pipeline (TF-IDF, HAC, NMF, PCA)
-   `hac.py` / `hac2.py` — hierarchical agglomerative clustering utilities
-   `langmod_class.py` / `langmod_funcs.py` — language modeling helpers
-   `textimporter.py` / `textparser.py` — text ingestion utilities
-   `eta/`, `mazo/` — additional utility subpackages

### TOKEN.csv Schema

TOKEN files are the central data format. Columns up to (not including) `token_str` form the OHCO index (Ordered Hierarchy of Content Objects — e.g., book, chapter, paragraph, token). `term_str` holds the normalized term used for modeling.