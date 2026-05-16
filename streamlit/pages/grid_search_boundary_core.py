"""
Shared pipeline functions for all boundary concordance grid search pages.

All @st.cache_data functions are defined here so that Streamlit's process-level
cache is shared: visiting any individual method page warms the cache for the
ensemble page and vice versa.
"""

import os
import yaml
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

N_COMPONENTS = 10
FIXED_MIN_DF = 5
FIXED_NGRAM  = (1, 1)
NMF_MAX_ITER = _cfg["model"]["nmf_max_iter"]


def find_token_file(src_id: str) -> str | None:
    p = os.path.normpath(
        os.path.join(APP_DIR, f"../../notebooks/{src_id}/{src_id}-TOKEN.csv")
    )
    return p if os.path.exists(p) else None


@st.cache_data(show_spinner=False)
def load_tokens(src_id: str, token_path: str) -> pd.DataFrame:
    TOKEN = pd.read_csv(token_path)
    idx_offset = TOKEN.columns.to_list().index("token_str")
    ohco = TOKEN.columns.to_list()[:idx_offset]
    return TOKEN.set_index(ohco)


def _vectorize(src_id, token_path, n_chunks, min_df, max_df, ngram_range):
    """Chunk text and TF-IDF vectorize. Returns (X_sparse, chunks_list) or (None, None)."""
    TOKEN = load_tokens(src_id, token_path)
    token_reset = TOKEN.reset_index()
    token_reset["chunk_num"] = pd.cut(
        token_reset.index, n_chunks, labels=list(range(n_chunks))
    )
    chunks_s = (
        token_reset.groupby("chunk_num", observed=True)["term_str"]
        .apply(lambda x: " ".join(x.dropna()))
    )
    chunks_list = chunks_s.tolist()
    if len(chunks_list) < 3:
        return None, None
    try:
        vec = TfidfVectorizer(
            lowercase=True, max_df=max_df, min_df=min_df,
            strip_accents=None, norm="l2", ngram_range=ngram_range,
        )
        X = vec.fit_transform(chunks_list)
    except ValueError:
        return None, None
    return X, chunks_list


@st.cache_data(show_spinner=False)
def run_linkage_lsa(src_id, token_path, n_chunks, min_df, max_df, ngram_range=(1, 1)):
    X, chunks_list = _vectorize(src_id, token_path, n_chunks, min_df, max_df, ngram_range)
    if X is None:
        return None
    n_comp = min(N_COMPONENTS, X.shape[0] - 1, X.shape[1] - 1)
    if n_comp < 2:
        return None
    X_svd = TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(X)
    Z = linkage(pdist(X_svd, metric="euclidean"), method="ward")
    return {"Z": Z, "n_chunks": len(chunks_list)}


@st.cache_data(show_spinner=False)
def run_linkage_tfidf(src_id, token_path, n_chunks, min_df, max_df, ngram_range=(1, 1)):
    X, chunks_list = _vectorize(src_id, token_path, n_chunks, min_df, max_df, ngram_range)
    if X is None:
        return None
    Z = linkage(pdist(X.toarray(), metric="euclidean"), method="ward")
    return {"Z": Z, "n_chunks": len(chunks_list)}


@st.cache_data(show_spinner=False)
def run_linkage_sim(src_id, token_path, n_chunks, min_df, max_df, ngram_range=(1, 1)):
    X, chunks_list = _vectorize(src_id, token_path, n_chunks, min_df, max_df, ngram_range)
    if X is None:
        return None
    SIM = (X @ X.T).toarray()
    Z = linkage(pdist(SIM, metric="euclidean"), method="ward")
    return {"Z": Z, "n_chunks": len(chunks_list)}


@st.cache_data(show_spinner=False)
def run_nmf_labels(src_id, token_path, n_chunks, min_df, max_df, k, ngram_range=(1, 1)):
    X, chunks_list = _vectorize(src_id, token_path, n_chunks, min_df, max_df, ngram_range)
    if X is None:
        return None
    if len(chunks_list) < k or X.shape[1] < k:
        return None
    try:
        THETA = NMF(n_components=k, init="nndsvda", max_iter=NMF_MAX_ITER).fit_transform(X)
    except Exception:
        return None
    return np.argmax(THETA, axis=1)


def threshold_for_k(Z, k, n):
    k = max(2, min(k, n - 1))
    return float((Z[n - k - 1, 2] + Z[n - k, 2]) / 2)


def get_boundaries(labels: np.ndarray) -> np.ndarray:
    """Normalized [0, 1] positions where consecutive cluster labels differ."""
    n = len(labels)
    return np.array([i / n for i in range(1, n) if labels[i] != labels[i - 1]])


def boundary_f1(b1: np.ndarray, b2: np.ndarray, tol: float) -> float:
    """F1 between two boundary sets within positional tolerance tol."""
    if len(b1) == 0 and len(b2) == 0:
        return 1.0
    if len(b1) == 0 or len(b2) == 0:
        return 0.0
    matched_1 = sum(any(abs(b - c) <= tol for c in b2) for b in b1)
    matched_2 = sum(any(abs(c - b) <= tol for b in b1) for c in b2)
    precision = matched_1 / len(b1)
    recall    = matched_2 / len(b2)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mean_pairwise_boundary_f1(label_arrays, tol: float) -> float:
    n = len(label_arrays)
    if n < 2:
        return float("nan")
    bs = [get_boundaries(la) for la in label_arrays]
    scores = [
        boundary_f1(bs[i], bs[j], tol)
        for i in range(n) for j in range(i + 1, n)
    ]
    return float(np.mean(scores))
