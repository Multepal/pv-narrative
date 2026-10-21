"""
Scoring predicted narrative boundaries against editorial divisions.

References are division starts given as Christenson chapter numbers (config `divisions`).
They are placed in an edition's text by the normalized token position of each chapter's
first token: exactly for editions whose DOCMAP carries Christenson's chapter numbers, and
by relative position (Christenson's K'iche' text) for the rest.

Predicted boundaries are where an edition's chunk cluster label changes. Each boundary set
is scored with precision, recall and F1 within a tolerance, against a cyclic-shift chance
baseline that rotates the predicted label sequence and keeps the reference fixed.
"""

import os
import yaml
import numpy as np
import pandas as pd
import streamlit as st
from utils import tfidf_vectorizer
from sklearn.decomposition import NMF
from grid_search_boundary_core import get_boundaries, cluster_labels
from utils import find_token_file, load_tokens, make_chunks

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    _cfg = yaml.safe_load(_f)

DIV_CFG          = _cfg["divisions"]
CHAPTER_EDITIONS = set(DIV_CFG["chapter_editions"])
PROJECTION_SRC   = "christenson"
ROMAN            = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
DOC_TABLES       = os.path.normpath(os.path.join(APP_DIR, "../../notebooks/doc_tables"))
CHAP_MOD_PATH    = os.path.normpath(os.path.join(APP_DIR, "../../notebooks/christenson/christenson-CHAP_MOD.csv"))


def author_divisions() -> list[tuple]:
    """The author's divisions from config as (roman numeral, short label, first chapter, title)."""
    return [(ROMAN[i], d["label"], d["chap"], d.get("title", d["label"]))
            for i, d in enumerate(DIV_CFG["author"])]


@st.cache_data(show_spinner=False)
def chapter_starts(src_id: str) -> dict:
    """Normalized token position (0–1) of the first token of each Christenson chapter in src_id."""
    docmap = pd.read_csv(os.path.join(DOC_TABLES, f"{src_id}-DOCMAP.csv"))
    ntok = pd.read_csv(os.path.join(DOC_TABLES, f"{src_id}-TOKEN.csv"), usecols=["doc_id"]) \
             .groupby("doc_id").size()
    docmap["ntok"] = docmap["doc_id"].map(ntok).fillna(0)
    docmap["pos"] = docmap["ntok"].cumsum().shift(fill_value=0) / docmap["ntok"].sum()
    return docmap.groupby("chap_num")["pos"].min().to_dict()


@st.cache_data(show_spinner=False)
def chapter_titles() -> dict:
    return pd.read_csv(CHAP_MOD_PATH).set_index("chap_num")["chap_title"].to_dict()


def reference_starts(src_id: str) -> tuple[dict, bool]:
    """(chapter → position, exact) for src_id: its own chapters if it has Christenson's
    numbering, otherwise Christenson's K'iche' chapter positions as a relative-position proxy."""
    if src_id in CHAPTER_EDITIONS:
        return chapter_starts(src_id), True
    return chapter_starts(PROJECTION_SRC), False


def chapters_to_positions(chaps, starts: dict) -> np.ndarray:
    """Boundary positions for division-start chapters (chapter 1, the text start, is not a
    boundary). A chapter missing from the edition falls back to the next one present."""
    present = sorted(starts)
    out = []
    for c in chaps:
        if c <= 1:
            continue
        nxt = next((p for p in present if p >= c), None)
        if nxt is not None:
            out.append(starts[nxt])
    return np.array(sorted(set(out)))


def reference_sets(src_id: str) -> tuple[dict, bool]:
    """Named reference boundary sets for src_id: the author's divisions, each editor's parts,
    and Christenson's chapters. Returns ({name: positions}, exact)."""
    starts, exact = reference_starts(src_id)
    refs = {"Author's 8 divisions": chapters_to_positions([d["chap"] for d in DIV_CFG["author"]], starts)}
    for editor, chaps in DIV_CFG["editors"].items():
        refs[editor] = chapters_to_positions(chaps, starts)
    refs["Christenson's chapters"] = chapters_to_positions(sorted(starts), starts)
    return refs, exact


def prf(pred, ref, tol):
    """Precision (share of predicted boundaries near a reference boundary), recall (share of
    reference boundaries near a predicted one) and F1, within tolerance tol."""
    pred, ref = np.asarray(pred), np.asarray(ref)
    if len(pred) == 0 or len(ref) == 0:
        return 0.0, 0.0, 0.0
    close = np.abs(np.subtract.outer(pred, ref)) <= tol
    p, r = close.any(axis=1).mean(), close.any(axis=0).mean()
    return float(p), float(r), float(0.0 if p + r == 0 else 2 * p * r / (p + r))


def score_against(labels, ref, tol) -> dict:
    """Observed P/R/F1 of labels' boundaries against ref, with a cyclic-shift chance baseline.
    Shifts within ±tol are skipped (they would count as matches by definition)."""
    labels = np.asarray(labels)
    n = len(labels)
    p, r, f = prf(get_boundaries(labels), ref, tol)
    min_shift = int(np.floor(tol * n)) + 1
    null = np.array([prf(get_boundaries(np.roll(labels, s)), ref, tol)
                     for s in range(min_shift, n - min_shift + 1)])
    if not len(null):
        null = np.full((1, 3), np.nan)
    out = {"n_pred": len(get_boundaries(labels)), "n_ref": len(ref), "precision": p}
    # Recall is the headline against coarse references (an edition's 3–5 parts), F1 overall.
    for key, obs, col in [("recall", r, 1), ("f1", f, 2)]:
        nl = null[:, col]
        p95 = float(np.nanpercentile(nl, 95))
        out.update({key: obs, f"{key}_chance_mean": float(np.nanmean(nl)), f"{key}_chance_p95": p95,
                    f"{key}_excess": obs - p95,
                    f"{key}_p": float((np.sum(nl >= obs) + 1) / (np.sum(~np.isnan(nl)) + 1))})
    return out


def resonance_peaks(resonance, m: int) -> np.ndarray:
    """Positions (j / n) of the m highest local maxima of a resonance curve (Barron et al.
    2018). Taking local maxima keeps a run of high chunks from counting as several boundaries;
    taking m of them gives this detector the same boundary budget as the model."""
    r = np.where(np.isnan(resonance), -np.inf, np.asarray(resonance, dtype=float))
    n = len(r)
    peaks = [j for j in range(n)
             if np.isfinite(r[j]) and (j == 0 or r[j] >= r[j - 1]) and (j == n - 1 or r[j] >= r[j + 1])]
    top = sorted(sorted(peaks, key=lambda j: r[j], reverse=True)[:m])
    return np.array([j / n for j in top])


def score_positions(pred, ref, tol, n: int) -> dict:
    """score_against for a detector that outputs boundary positions on the chunk grid rather
    than a label sequence: the chance baseline rotates the positions by whole chunks."""
    pred = np.asarray(pred)
    p, r, f = prf(pred, ref, tol)
    min_shift = int(np.floor(tol * n)) + 1
    null = np.array([prf(np.sort((pred + s / n) % 1.0), ref, tol)
                     for s in range(min_shift, n - min_shift + 1)])
    if not len(null):
        null = np.full((1, 3), np.nan)
    out = {"n_pred": len(pred), "n_ref": len(ref), "precision": p}
    for key, obs, col in [("recall", r, 1), ("f1", f, 2)]:
        nl = null[:, col]
        p95 = float(np.nanpercentile(nl, 95))
        out.update({key: obs, f"{key}_chance_mean": float(np.nanmean(nl)), f"{key}_chance_p95": p95,
                    f"{key}_excess": obs - p95,
                    f"{key}_p": float((np.sum(nl >= obs) + 1) / (np.sum(~np.isnan(nl)) + 1))})
    return out


def pairwise_f1(sets: dict, tol) -> pd.DataFrame:
    """Symmetric boundary-F1 matrix between named boundary sets (editors, author, model)."""
    names = list(sets)
    mat = np.ones((len(names), len(names)))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            mat[i, j] = mat[j, i] = prf(sets[names[i]], sets[names[j]], tol)[2]
    return pd.DataFrame(mat, index=names, columns=names)


def consensus_boundaries(starts: dict) -> pd.DataFrame:
    """Every chapter where at least one editor starts a part: which editors, how many, whether
    the author's divisions start there, and the chapter's position."""
    rows = {d["chap"]: [] for d in DIV_CFG["author"] if d["chap"] > 1}
    for editor, chaps in DIV_CFG["editors"].items():
        for c in chaps:
            if c > 1:
                rows.setdefault(c, []).append(editor.split()[0])
    author = {d["chap"] for d in DIV_CFG["author"]}
    titles = chapter_titles()
    return pd.DataFrame([{
        "chapter": c, "title": titles.get(c, ""), "n_editors": len(eds), "editors": ", ".join(eds),
        "author": c in author, "position": chapters_to_positions([c], starts)[0],
    } for c, eds in sorted(rows.items())])


# ── Chapter resolution: predicting divisions for units the model never saw ────

def _chunk_and_chapter_tfidf(src_id, n_chunks, min_df, max_df, ngram_range):
    """TF-IDF of the edition's chunks (fit) and of its Christenson chapters (transform with
    the chunk vocabulary and IDF). Returns (vectorizer, X_chunks, X_chapters, chapter numbers)."""
    tp = find_token_file(src_id)
    vec = tfidf_vectorizer(lowercase=True, min_df=min_df, max_df=max_df, strip_accents=None,
                          norm="l2", ngram_range=ngram_range)
    X = vec.fit_transform(make_chunks(src_id, tp, n_chunks))
    docmap = pd.read_csv(os.path.join(DOC_TABLES, f"{src_id}-DOCMAP.csv"))[["doc_id", "chap_num"]]
    tokens = load_tokens(src_id, tp).reset_index().merge(docmap, on="doc_id")
    docs = tokens.groupby("chap_num")["term_str"].apply(lambda x: " ".join(x.dropna()))
    return vec, X, vec.transform(docs.tolist()), docs.index.to_numpy()


@st.cache_data(show_spinner="Projecting chapters onto the topic model…")
def chapter_topic_shares(src_id, n_chunks, min_df, max_df, k, ngram_range=(1, 1)):
    """The app's NMF (same vectorizer, k and seed as cluster_labels("NMF")) fit on the
    edition's chunks, with every Christenson chapter projected onto it (nmf.transform).
    Returns (chapter numbers, shares as topics × chapters with columns summing to 1,
    top terms per topic)."""
    vec, X, C, chapters = _chunk_and_chapter_tfidf(src_id, n_chunks, min_df, max_df, ngram_range)
    nmf = NMF(n_components=k, init="nndsvda", max_iter=_cfg["model"]["nmf_max_iter"],
              random_state=_cfg["model"]["random_state"]).fit(X)
    W = nmf.transform(C).T                                    # topics × chapters
    shares = W / np.where(W.sum(axis=0) == 0, 1, W.sum(axis=0))
    words = vec.get_feature_names_out()
    top_terms = [[words[j] for j in row.argsort()[::-1][:5]] for row in nmf.components_]
    return chapters, shares, top_terms


@st.cache_data(show_spinner="Assigning chapters to classes…")
def chapter_labels(method, src_id, n_chunks, min_df, max_df, k, ngram_range=(1, 1)):
    """Predict a class for every Christenson chapter of src_id. The model is fit on the
    edition's chunks as usual and chapters are never part of the fit, so this is an
    out-of-sample prediction. HAC methods: each chapter goes to the nearest class centroid
    (cosine) in the chunk model's TF-IDF space. NMF: each chapter's dominant topic after
    projection (the same projection as the Chapter Topics page).
    Returns (chapter numbers, labels), or None."""
    if method == "NMF":
        chapters, shares, _ = chapter_topic_shares(src_id, n_chunks, min_df, max_df, k, ngram_range)
        return chapters, shares.argmax(axis=0)
    tp = find_token_file(src_id)
    la = cluster_labels(method, src_id, tp, n_chunks, min_df, max_df, k, ngram_range)
    if la is None:
        return None
    la = np.asarray(la)
    _, X, C, chapters = _chunk_and_chapter_tfidf(src_id, n_chunks, min_df, max_df, ngram_range)
    X, C = X.toarray(), C.toarray()
    classes = np.unique(la)
    cents = np.array([X[la == c].mean(axis=0) for c in classes])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    return chapters, classes[np.argmax(C @ cents.T, axis=1)]


def chapter_grid_positions(chaps, chapters) -> np.ndarray:
    """Division starts (Christenson chapter numbers) as positions on the chapter grid
    (index / number of chapters), matching get_boundaries on a per-chapter label sequence.
    A chapter missing from the edition falls back to the next one present."""
    chapters = list(chapters)
    out = set()
    for c in chaps:
        if c <= 1:
            continue
        idx = next((i for i, ch in enumerate(chapters) if ch >= c), None)
        if idx is not None:
            out.add(idx / len(chapters))
    return np.array(sorted(out))
