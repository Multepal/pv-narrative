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
    close     = np.abs(np.subtract.outer(np.asarray(b1), np.asarray(b2))) <= tol
    precision = close.any(axis=1).mean()
    recall    = close.any(axis=0).mean()
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


def metric_fns(metric, tol=None):
    """(score, null) functions on two label arrays for metric 'ARI' or 'F1' (boundary F1 at tol)."""
    if metric == "ARI":
        return adjusted_rand_score, cyclic_shift_ari
    return (lambda a, b: boundary_f1(get_boundaries(a), get_boundaries(b), tol),
            lambda a, b: cyclic_shift_boundary_f1(a, b, tol))


def ari_summary(label_arrays, langs):
    return pairwise_summary(label_arrays, langs, *metric_fns("ARI"))


def f1_summary(label_arrays, langs, tol):
    return pairwise_summary(label_arrays, langs, *metric_fns("F1", tol))


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


# ── Selection by excess over chance (grid searches) ───────────────────────────

PAIR_GROUPS = ("within", "cross", "all")


def pair_group(lang_a, lang_b):
    """'within' (both in the source language), 'cross' (source ↔ translation), or None."""
    return {2: "within", 1: "cross"}.get((lang_a == SOURCE_LANG) + (lang_b == SOURCE_LANG))


def agreement_by_group(label_arrays, langs, metric, tol_chunks=0):
    """Observed mean, chance 95th percentile and excess (observed − chance) per pair group.

    Each group's chance baseline pools the cyclic-shift scores of that group's own pairs,
    so a group is judged against what its own sequences would score when misaligned.
    """
    score, null = metric_fns(metric, boundary_tol(len(label_arrays[0]), tol_chunks))
    obs  = {g: [] for g in PAIR_GROUPS}
    pool = {g: [] for g in PAIR_GROUPS}
    for i in range(len(label_arrays)):
        for j in range(i + 1, len(label_arrays)):
            a, b = label_arrays[i], label_arrays[j]
            s, nl = score(a, b), null(a, b)
            for g in ("all", pair_group(langs[i], langs[j])):
                if g:
                    obs[g].append(s)
                    pool[g].extend(nl)
    out = {}
    for g in PAIR_GROUPS:
        o = float(np.mean(obs[g])) if obs[g] else float("nan")
        p = float(np.percentile(pool[g], 95)) if pool[g] else float("nan")
        out.update({f"{g}_obs": o, f"{g}_p95": p, f"{g}_excess": o - p})
    return out


@st.cache_data(show_spinner=False)
def agreement_curve(method, n_chunks, max_df, metric, tol_chunks, k_vals,
                    min_df=FIXED_MIN_DF, ngram_range=FIXED_NGRAM):
    """agreement_by_group at each k in k_vals (a tuple) for one parameter combination.
    Returns a list of row dicts. Skips k values with fewer than two modeled editions, and k
    above n_chunks / min_chunks_per_cluster: there partitions approach one chunk per cluster
    (threshold_for_k also caps k at n_chunks − 1), and agreement becomes trivial."""
    rows = []
    max_k = n_chunks // _cfg["controls"]["min_chunks_per_cluster"]
    for k in (k for k in k_vals if k <= max_k):
        labels, langs = [], []
        for src_id, meta in _cfg["sources"].items():
            tp = find_token_file(src_id)
            if tp is None:
                continue
            la = cluster_labels(method, src_id, tp, n_chunks, min_df, max_df, k, ngram_range)
            if la is not None:
                labels.append(np.asarray(la))
                langs.append(meta["lang"])
        if len(labels) >= 2:
            rows.append({"k": k, "n_editions": len(labels),
                         **agreement_by_group(labels, langs, metric, tol_chunks)})
    return rows
