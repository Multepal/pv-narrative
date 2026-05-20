"""
Shared utilities for all Streamlit page scripts.
"""

import os
import re
import yaml
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
from scipy.spatial.distance import hamming
from sklearn.metrics import adjusted_rand_score

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def find_token_file(src_id: str) -> str | None:
    for template in [
        f"../../notebooks/doc_tables/{src_id}-TOKEN.csv",
        f"../../notebooks/{src_id}/{src_id}-TOKEN.csv",
    ]:
        p = os.path.normpath(os.path.join(APP_DIR, template))
        if os.path.exists(p):
            return p
    return None


@st.cache_data(show_spinner=False)
def load_tokens(src_id: str, token_path: str) -> pd.DataFrame:
    TOKEN = pd.read_csv(token_path)
    idx_offset = TOKEN.columns.to_list().index("token_str")
    ohco = TOKEN.columns.to_list()[:idx_offset]
    return TOKEN.set_index(ohco)


def make_chunks(src_id: str, token_path: str, n_chunks: int) -> list[str]:
    TOKEN = load_tokens(src_id, token_path)
    token_reset = TOKEN.reset_index()
    token_reset["chunk_num"] = pd.cut(
        token_reset.index, n_chunks, labels=list(range(n_chunks))
    )
    return (
        token_reset.groupby("chunk_num", observed=True)["term_str"]
        .apply(lambda x: " ".join(x.dropna()))
        .tolist()
    )


@st.cache_data(show_spinner=False)
def load_chunk_doc_texts(src_id: str, n_chunks: int) -> list[str]:
    """Readable DOC prose per chunk (length = n_chunks).

    Groups tokens by their first OHCO level (paragraph / chapter), assigns
    each group to a chunk, then looks up the corresponding DOC text.
    Editions whose DOC files have sub-unit rows (e.g. ajtzibab, where each
    paragraph spans many line rows) are aggregated before lookup.
    Returns empty strings for chunks or editions with no data.
    """
    token_path = find_token_file(src_id)
    if token_path is None:
        return [""] * n_chunks

    doc_path = os.path.normpath(
        os.path.join(os.path.dirname(token_path), f"{src_id}-DOC.csv")
    )
    if not os.path.exists(doc_path):
        return [""] * n_chunks

    TOKEN = pd.read_csv(token_path)
    idx_end = TOKEN.columns.tolist().index("token_str")
    doc_col = TOKEN.columns.tolist()[0]          # coarsest OHCO level

    TOKEN["chunk_num"] = pd.cut(TOKEN.index, n_chunks, labels=range(n_chunks)).astype(int)
    # Collect all doc-unit IDs whose tokens appear in each chunk.  Using all
    # IDs (not just first) ensures chunks that fall inside a long paragraph
    # still receive text.
    chunk_to_ids = (
        TOKEN.groupby("chunk_num")[doc_col]
        .apply(lambda s: list(dict.fromkeys(s)))   # unique, insertion-ordered
    )

    doc = pd.read_csv(doc_path)
    doc_idx = doc.columns[0]
    # Aggregate sub-unit rows to doc_idx level (handles ajtzibab, tedlock, etc.)
    texts = (
        doc.groupby(doc_idx)["doc_str"]
        .apply(lambda x: " ".join(
            v for v in x.dropna().astype(str) if v.lower() != "nan"
        ))
    )

    _nan_prefix = re.compile(r"^\s*nan\s+", re.IGNORECASE)

    result = []
    for c in range(n_chunks):
        ids = chunk_to_ids.get(c, [])
        parts = [_nan_prefix.sub("", str(texts.get(i, ""))) for i in ids]
        parts = [p for p in parts if p.strip() and p.strip().lower() != "nan"]
        result.append("\n\n".join(parts))
    return result


def threshold_for_k(Z, k: int, n: int) -> float:
    k = max(2, min(k, n - 1))
    return float((Z[n - k - 1, 2] + Z[n - k, 2]) / 2)


def cluster_string(labels) -> str:
    mapping: dict = {}
    result = []
    for lbl in labels:
        if lbl not in mapping:
            mapping[lbl] = chr(65 + len(mapping))
        result.append(mapping[lbl])
    return "".join(result)


def make_cluster_table(tfidf_dense, words, labels, n_top_words: int) -> pd.DataFrame:
    tfidf_df = pd.DataFrame(tfidf_dense, columns=words)
    cluster_s = pd.Series(labels, name="cluster")
    cluster_tfidf = tfidf_df.groupby(cluster_s).mean()
    top_terms = cluster_tfidf.apply(
        lambda row: ", ".join(row.sort_values(ascending=False).head(n_top_words).index),
        axis=1,
    )
    n_per = cluster_s.value_counts().sort_index().rename("n_chunks")
    return pd.DataFrame({"n_chunks": n_per, "top_terms": top_terms})


def build_dendrogram_figure(Z, labels, n: int, cluster_to_color: dict, chunk_labels: list):
    gray = "#CCCCCC"
    node_clusters: dict = {}

    def leaf_clusters(node):
        if node in node_clusters:
            return node_clusters[node]
        if node < n:
            result = frozenset([labels[node]])
        else:
            i = int(node - n)
            result = leaf_clusters(int(Z[i, 0])) | leaf_clusters(int(Z[i, 1]))
        node_clusters[node] = result
        return result

    for i in range(len(Z)):
        leaf_clusters(n + i)

    def link_color_func(k):
        cs = leaf_clusters(k)
        return cluster_to_color[next(iter(cs))] if len(cs) == 1 else gray

    dend = scipy_dendrogram(Z, no_plot=True, link_color_func=link_color_func)

    _height_to_z = {float(Z[i, 2]): i for i in range(len(Z))}
    cluster_root_height: dict = {}
    for i in range(len(Z)):
        cs = node_clusters[n + i]
        if len(cs) == 1:
            c = next(iter(cs))
            h = float(Z[i, 2])
            if c not in cluster_root_height or h > cluster_root_height[c]:
                cluster_root_height[c] = h

    cluster_root_x: dict = {}
    for xs, ys in zip(dend["icoord"], dend["dcoord"]):
        h = float(ys[1])
        z_idx = _height_to_z.get(h)
        if z_idx is None:
            continue
        cs = node_clusters.get(n + z_idx, frozenset())
        if len(cs) == 1:
            c = next(iter(cs))
            if cluster_root_height.get(c) == h:
                cluster_root_x[c] = (float(xs[1]) + float(xs[2])) / 2

    fig = go.Figure()
    for xs, ys, color in zip(dend["icoord"], dend["dcoord"], dend["color_list"]):
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=color, width=2.5),
            showlegend=False, hoverinfo="none",
        ))

    leaf_x = [10 * i + 5 for i in range(n)]
    fig.update_layout(
        xaxis=dict(
            tickvals=leaf_x, ticktext=[chunk_labels[o] for o in dend["leaves"]],
            range=[-5, 10 * n + 5],
            showline=False, showgrid=False, zeroline=False,
        ),
        yaxis=dict(showline=False, showgrid=False, zeroline=False),
        plot_bgcolor="white",
    )
    return fig, dend["leaves"], cluster_root_x



def wrap_text(text: str, width: int = 60) -> str:
    words, lines, line = text.split(), [], []
    for word in words:
        if sum(len(w) for w in line) + len(line) + len(word) > width:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    return "<br>".join(lines)


def mean_pairwise_hamming(strings: list[str]) -> float:
    n = len(strings)
    if n < 2:
        return float("nan")
    return float(np.mean([
        hamming(list(strings[i]), list(strings[j]))
        for i in range(n) for j in range(i + 1, n)
    ]))


def mean_pairwise_ari(label_arrays: list) -> float:
    n = len(label_arrays)
    if n < 2:
        return float("nan")
    return float(np.mean([
        adjusted_rand_score(label_arrays[i], label_arrays[j])
        for i in range(n) for j in range(i + 1, n)
    ]))


def pairwise_hamming_matrix(strings: list[str]) -> np.ndarray:
    n = len(strings)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming(list(strings[i]), list(strings[j]))
            mat[i, j] = mat[j, i] = d
    return mat


def pairwise_ari_matrix(label_arrays: list) -> np.ndarray:
    n = len(label_arrays)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            score = adjusted_rand_score(label_arrays[i], label_arrays[j])
            mat[i, j] = mat[j, i] = score
    return mat
