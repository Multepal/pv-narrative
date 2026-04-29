# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Streamlit App

```bash
cd notebooks
streamlit run streamlit_app.py
```

Install dependencies first if needed:

```bash
pip install -r notebooks/requirements.txt
```

## Architecture

This is a **MyST Markdown / Jupyter Book** project (`myst.yml`) combining research notebooks with an interactive Streamlit app.

### Data Pipeline

Each source edition has its own subdirectory under `notebooks/` (e.g., `notebooks/colop/`, `notebooks/ajtzibab/`). Within each, a two-notebook pipeline runs sequentially:

1. **`01-*-parse*.ipynb`** — imports raw source text from `textos/`, parses it into structured tables, and writes CSVs (DOC, TOKEN, VOCAB) into the source directory.
2. **`02-*-model*.ipynb`** — reads TOKEN.csv, applies TF-IDF + NMF/HAC/PCA, and writes model output CSVs (THETA, PHI, CHUNK, TFIDF, CLUSTER, etc.) plus figures.

The `notebooks/ensemble/` directory aggregates TOKEN files across all sources.

### Streamlit App (`notebooks/streamlit_app.py`)

The app is self-contained: it reads a `*-TOKEN.csv` file at runtime and runs the full NMF pipeline (TF-IDF → NMF → cosine distance) interactively. It does **not** depend on the pre-computed model CSVs from the notebooks — it re-runs the model on every parameter change.

Token file resolution order (for a given `src_id`):
1. `notebooks/<src_id>/<src_id>-TOKEN.csv`
2. `notebooks/ensemble/<src_id>-TOKEN.csv`
3. `notebooks/<src_id>-TOKEN.csv`
4. `../<src_id>/<src_id>-TOKEN.csv`

**`notebooks/about.md`** — sidebar content rendered in the app; edit this file to update explanatory text without touching Python code.

### Local Library (`local_lib/`)

Shared Python modules used by the research notebooks (not the Streamlit app):

- `narrative_model.py` / `narrative_parser.py` — `NarrativeModel` and `NarrativeParser` classes encapsulating the full modeling pipeline (TF-IDF, HAC, NMF, PCA)
- `hac.py` / `hac2.py` — hierarchical agglomerative clustering utilities
- `langmod_class.py` / `langmod_funcs.py` — language modeling helpers
- `textimporter.py` / `textparser.py` — text ingestion utilities
- `eta/`, `mazo/` — additional utility subpackages

### TOKEN.csv Schema

TOKEN files are the central data format. Columns up to (not including) `token_str` form the OHCO index (Ordered Hierarchy of Content Objects — e.g., book, chapter, paragraph, token). `term_str` holds the normalized term used for modeling.

### Published Site

The MyST site is configured to publish to `https://multepal.github.io/pv-narrative`. Build with:

```bash
myst build --html
```
