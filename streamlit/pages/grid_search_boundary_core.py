"""
Shared pipeline and agreement functions for the cross-edition pages: the grid
searches (ARI, boundary concordance) and Cross-Edition Comparison.

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
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score
from utils import find_token_file, load_tokens, threshold_for_k, cyclic_shift_ari

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

N_COMPONENTS = 10
FIXED_MIN_DF = 5
FIXED_NGRAM  = (1, 1)
NMF_MAX_ITER = _cfg["model"]["nmf_max_iter"]
SOURCE_LANG  = _cfg["source_lang"]


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
        THETA = NMF(n_components=k, init="nndsvda", max_iter=NMF_MAX_ITER, random_state=_cfg["model"]["random_state"]).fit_transform(X)
    except Exception:
        return None
    return np.argmax(THETA, axis=1)


HAC_RUN_FNS = {
    "TF-IDF":     run_linkage_tfidf,
    "LSA":        run_linkage_lsa,
    "Cosine-Sim": run_linkage_sim,
}


def cluster_labels(method, src_id, token_path, n_chunks, min_df, max_df, k, ngram_range=(1, 1)):
    """Per-chunk labels for one edition: a HAC method cut at k clusters, or NMF argmax over k topics.
    Returns None if the edition can't be modeled with these parameters."""
    if method == "NMF":
        return run_nmf_labels(src_id, token_path, n_chunks, min_df, max_df, k, ngram_range)
    result = HAC_RUN_FNS[method](src_id, token_path, n_chunks, min_df, max_df, ngram_range)
    if result is None:
        return None
    Z = result["Z"]
    return fcluster(Z, threshold_for_k(Z, k, result["n_chunks"]), criterion="distance")


@st.cache_data(show_spinner=False)
def run_resonance_boundaries(src_id, token_path, n_chunks, min_df, max_df, window):
    """Compute KLD resonance curve for one edition. Returns dict with 'resonance' and 'n_chunks'."""
    TOKEN = load_tokens(src_id, token_path)
    token_reset = TOKEN.reset_index()
    token_reset["chunk_num"] = pd.cut(
        token_reset.index, n_chunks, labels=list(range(n_chunks))
    )
    chunks_list = (
        token_reset.groupby("chunk_num", observed=True)["term_str"]
        .apply(lambda x: " ".join(x.dropna()))
        .tolist()
    )
    if len(chunks_list) < 3:
        return None
    try:
        vec = TfidfVectorizer(lowercase=True, max_df=max_df, min_df=min_df,
                              strip_accents=None, norm=None)
        X = vec.fit_transform(chunks_list).toarray().astype(float)
    except ValueError:
        return None

    X += 1e-9
    P = X / X.sum(axis=1, keepdims=True)
    n = len(P)

    def kld(p, q):
        return float(np.sum(p * np.log2(p / q)))

    novelty    = np.full(n, np.nan)
    transience = np.full(n, np.nan)
    for j in range(n):
        back = [kld(P[j], P[j - d]) for d in range(1, window + 1) if j - d >= 0]
        fwd  = [kld(P[j], P[j + d]) for d in range(1, window + 1) if j + d < n]
        if back:
            novelty[j]    = float(np.mean(back))
        if fwd:
            transience[j] = float(np.mean(fwd))

    return {"resonance": novelty - transience, "n_chunks": n}


def resonance_boundaries_at_pct(resonance: np.ndarray, n_chunks: int, threshold_pct: int) -> np.ndarray:
    """Normalized [0, 1] boundary positions where resonance ≥ threshold_pct percentile."""
    valid = resonance[~np.isnan(resonance)]
    if len(valid) == 0:
        return np.array([])
    thresh = float(np.nanpercentile(valid, threshold_pct))
    return np.array([
        j / n_chunks for j in range(n_chunks)
        if not np.isnan(resonance[j]) and resonance[j] >= thresh
    ])


def mean_pairwise_f1_from_positions(position_arrays: list, tol: float) -> float:
    """Mean pairwise boundary F1 from pre-normalized position arrays."""
    n = len(position_arrays)
    if n < 2:
        return float("nan")
    scores = [
        boundary_f1(position_arrays[i], position_arrays[j], tol)
        for i in range(n) for j in range(i + 1, n)
    ]
    return float(np.mean(scores))


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


def cyclic_shift_boundary_f1(a, b, tol: float) -> np.ndarray:
    """Boundary F1 between `a` and cyclic shifts of `b` that move it beyond the tolerance.

    Chance baseline matching utils.cyclic_shift_ari. Shifts within ±tol are skipped:
    by the F1's own definition they are not misalignments.
    """
    a, b = np.asarray(a), np.asarray(b)
    n = len(b)
    min_shift = int(np.floor(tol * n)) + 1
    ba = get_boundaries(a)
    return np.array([boundary_f1(ba, get_boundaries(np.roll(b, s)), tol)
                     for s in range(min_shift, n - min_shift + 1)])


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


# ── Cross-edition agreement ───────────────────────────────────────────────────

def pairwise_summary(label_arrays, langs, score, null):
    """Mean pairwise `score` overall, within the source language, and source ↔ translation,
    plus a chance baseline from `null` (an array of scores per pair) pooled over all pairs."""
    within, cross, every, pooled = [], [], [], []
    for i in range(len(label_arrays)):
        for j in range(i + 1, len(label_arrays)):
            a, b = label_arrays[i], label_arrays[j]
            s = score(a, b)
            every.append(s)
            n_src = (langs[i] == SOURCE_LANG) + (langs[j] == SOURCE_LANG)
            if n_src == 2:
                within.append(s)
            elif n_src == 1:
                cross.append(s)
            pooled.extend(null(a, b))
    _mean = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")
    return {
        "all":       _mean(every),
        "within":    _mean(within),
        "cross":     _mean(cross),
        "null_mean": _mean(pooled),
        "null_p95":  float(np.percentile(pooled, 95)) if len(pooled) else float("nan"),
    }


def ari_summary(label_arrays, langs):
    return pairwise_summary(label_arrays, langs, adjusted_rand_score, cyclic_shift_ari)


def f1_summary(label_arrays, langs, tol):
    return pairwise_summary(
        label_arrays, langs,
        lambda a, b: boundary_f1(get_boundaries(a), get_boundaries(b), tol),
        lambda a, b: cyclic_shift_boundary_f1(a, b, tol),
    )


def boundary_tol(n_chunks, tol_chunks):
    """Tolerance in normalized position for matching within ±tol_chunks chunks.

    Fixed in chunks rather than scaled by k (as on Grid Search (Boundary)) because
    recurring clusters yield more than k−1 boundaries, so k-scaled windows overlap
    and F1 saturates even for misaligned sequences. The half-chunk margin keeps the
    float comparison in boundary_f1 exact (0 = boundaries must coincide).
    """
    return (tol_chunks + 0.5) / n_chunks


def pairwise_f1_matrix(label_arrays, tol):
    bs  = [get_boundaries(np.asarray(la)) for la in label_arrays]
    mat = np.ones((len(bs), len(bs)))
    for i in range(len(bs)):
        for j in range(i + 1, len(bs)):
            mat[i, j] = mat[j, i] = boundary_f1(bs[i], bs[j], tol)
    return mat
