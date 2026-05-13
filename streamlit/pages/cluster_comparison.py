"""
Cross-Edition Cluster Comparison — run HAC across all editions with fixed
parameters and compare cluster sequences using pairwise Hamming distance.

Hold out one parameter to sweep its full valid range and observe how
cross-edition agreement varies.
"""

import os
import yaml
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import pdist, hamming
from scipy.cluster.hierarchy import linkage, fcluster

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
LANG_LABELS  = cfg["languages"]

st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)

# Full valid sweep range for each hold-out parameter
HOLDOUT_DEFS = {
    "k":         {"label": "k (clusters)", "dtype": int,   "vals": list(range(2, 21))},
    "n_chunks":  {"label": "n_chunks",     "dtype": int,   "vals": list(range(10, 55, 5))},
    "min_df":    {"label": "min_df",       "dtype": int,   "vals": list(range(1, 11))},
    "max_df":    {"label": "max_df",       "dtype": float, "vals": [round(v, 2) for v in np.arange(0.20, 0.95, 0.10)]},
    "ngram_max": {"label": "ngram max",    "dtype": int,   "vals": list(range(1, 5))},
}


def find_token_file(src_id: str) -> str | None:
    candidates = [
        os.path.join(APP_DIR, f"../../notebooks/{src_id}/{src_id}-TOKEN.csv"),
    ]
    for p in candidates:
        norm = os.path.normpath(p)
        if os.path.exists(norm):
            return norm
    return None


@st.cache_data(show_spinner=False)
def load_tokens(src_id: str, token_path: str) -> pd.DataFrame:
    TOKEN = pd.read_csv(token_path)
    idx_offset = TOKEN.columns.to_list().index("token_str")
    ohco = TOKEN.columns.to_list()[:idx_offset]
    return TOKEN.set_index(ohco)


@st.cache_data(show_spinner=False)
def run_linkage(src_id, token_path, n_chunks, min_df, max_df, ngram_range=(1, 1)):
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
        return None
    try:
        vec = TfidfVectorizer(
            lowercase=True, max_df=max_df, min_df=min_df,
            strip_accents=None, norm="l2", ngram_range=ngram_range,
        )
        X = vec.fit_transform(chunks_list)
    except ValueError:
        return None
    tfidf_dense = X.toarray()
    dist_condensed = pdist(tfidf_dense, metric="euclidean")
    Z = linkage(dist_condensed, method="ward")
    return {"Z": Z, "n_chunks": len(chunks_list)}


def threshold_for_k(Z, k, n):
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


def mean_pairwise_hamming(strings: list[str]) -> float:
    n = len(strings)
    if n < 2:
        return float("nan")
    dists = [
        hamming(list(strings[i]), list(strings[j]))
        for i in range(n) for j in range(i + 1, n)
    ]
    return float(np.mean(dists))


def pairwise_hamming_matrix(strings: list[str]) -> np.ndarray:
    n = len(strings)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming(list(strings[i]), list(strings[j]))
            mat[i, j] = mat[j, i] = d
    return mat


# ── Controls ──────────────────────────────────────────────────────────────────
st.title("Cross-Edition Cluster Comparison")
st.caption(
    "Runs HAC across all editions. Each edition is divided into exactly `n_chunks` equal bins "
    "via `pd.cut` (no overlap). Select a parameter to hold out; it will be swept over its full "
    "valid range while all other parameters stay fixed."
)

_c = cfg["controls"]
_col_ratios = cfg["layout"]["column_ratios"]

cols = st.columns(_col_ratios[:5])

holdout_param = cols[0].selectbox(
    "Hold-out parameter", list(HOLDOUT_DEFS.keys()),
    format_func=lambda x: HOLDOUT_DEFS[x]["label"],
)

n_chunks  = cols[1].number_input("n_chunks",  min_value=5, max_value=200, value=20, step=1,
                                  disabled=(holdout_param == "n_chunks"))
min_df    = cols[2].number_input("min_df",    _c["min_df"]["min"], _c["min_df"]["max"],
                                  _c["min_df"]["default"], step=_c["min_df"]["step"],
                                  disabled=(holdout_param == "min_df"))
max_df    = cols[2].number_input("max_df",    _c["max_df"]["min"], _c["max_df"]["max"],
                                  _c["max_df"]["default"], step=_c["max_df"]["step"],
                                  format="%.2f", disabled=(holdout_param == "max_df"))
_ng       = _c["ngram_range"]
ngram_min = cols[3].number_input("ngram min", _ng["min_n"], _ng["max_n"], _ng["default_min"], step=1)
ngram_max = cols[3].number_input("ngram max", _ng["min_n"], _ng["max_n"], _ng["default_max"], step=1,
                                  disabled=(holdout_param == "ngram_max"))
ngram_max = max(ngram_max, ngram_min)
k         = cols[4].number_input("k (clusters)", _c["n_clusters"]["min"], _c["n_clusters"]["max"],
                                  _c["n_clusters"]["default"], step=1,
                                  disabled=(holdout_param == "k"))

st.divider()

# ── Sweep ─────────────────────────────────────────────────────────────────────
sweep_vals = HOLDOUT_DEFS[holdout_param]["vals"]

all_rows     = []
summary_rows = []
progress     = st.progress(0, text="Running…")

for step_i, val in enumerate(sweep_vals):
    progress.progress((step_i + 1) / len(sweep_vals),
                      text=f"{HOLDOUT_DEFS[holdout_param]['label']} = {val}  ({step_i + 1}/{len(sweep_vals)})")

    p_n_chunks  = int(val)   if holdout_param == "n_chunks"  else int(n_chunks)
    p_min_df    = int(val)   if holdout_param == "min_df"    else int(min_df)
    p_max_df    = float(val) if holdout_param == "max_df"    else float(max_df)
    p_ngram_max = max(int(ngram_min), int(val)) if holdout_param == "ngram_max" else int(ngram_max)
    p_k         = int(val)   if holdout_param == "k"         else int(k)

    edition_strings = []
    for src_id, meta in SOURCES_META.items():
        token_path = find_token_file(src_id)
        if token_path is None:
            continue
        TOKEN = load_tokens(src_id, token_path)
        result = run_linkage(src_id, token_path, p_n_chunks, p_min_df, p_max_df,
                             (int(ngram_min), p_ngram_max))
        if result is None:
            continue
        Z       = result["Z"]
        n_act   = result["n_chunks"]
        labels  = fcluster(Z, threshold_for_k(Z, p_k, n_act), criterion="distance")
        cstr    = cluster_string(labels)
        edition_strings.append(cstr)
        all_rows.append({
            holdout_param:   val,
            "edition":       meta["label"],
            "lang":          LANG_LABELS[meta["lang"]],
            "n_tokens":      len(TOKEN),
            "k_actual":      int(len(np.unique(labels))),
            "cluster_string": cstr,
        })

    summary_rows.append({
        holdout_param:  val,
        "mean_hamming": mean_pairwise_hamming(edition_strings),
    })

progress.empty()

if not all_rows:
    st.warning("No editions could be clustered with the current parameters.")
    st.stop()

df_all     = pd.DataFrame(all_rows)
df_summary = pd.DataFrame(summary_rows)

# ── Line chart ────────────────────────────────────────────────────────────────
st.subheader(f"Mean Pairwise Hamming Distance vs. {HOLDOUT_DEFS[holdout_param]['label']}")
fig_line = px.line(
    df_summary, x=holdout_param, y="mean_hamming",
    markers=True,
    labels={holdout_param: HOLDOUT_DEFS[holdout_param]["label"], "mean_hamming": "Mean Hamming"},
)
fig_line.update_layout(
    height=320,
    margin=dict(l=60, r=30, t=20, b=50),
    plot_bgcolor="white",
    yaxis=dict(range=[0, 1], showgrid=True, gridcolor="#EEEEEE", zeroline=False),
    xaxis=dict(showgrid=False, zeroline=False),
)
st.plotly_chart(fig_line, width="stretch")

st.dataframe(
    df_summary.rename(columns={
        holdout_param:  HOLDOUT_DEFS[holdout_param]["label"],
        "mean_hamming": "Mean Hamming",
    }),
    use_container_width=True,
    hide_index=True,
)

# ── Pairwise matrix for a selected sweep value ────────────────────────────────
st.divider()
st.subheader("Pairwise Hamming Matrix")
sel_val = st.selectbox(
    HOLDOUT_DEFS[holdout_param]["label"],
    sweep_vals,
    index=len(sweep_vals) - 1,
)
if HOLDOUT_DEFS[holdout_param]["dtype"] == float:
    df_sel = df_all[np.isclose(df_all[holdout_param].astype(float), float(sel_val))]
else:
    df_sel = df_all[df_all[holdout_param] == sel_val]

if len(df_sel) >= 2:
    edition_labels = df_sel["edition"].tolist()
    mat = pairwise_hamming_matrix(df_sel["cluster_string"].tolist())
    dist_df = pd.DataFrame(mat, index=edition_labels, columns=edition_labels).round(4)
    fig_hm = px.imshow(
        dist_df, color_continuous_scale="Blues", zmin=0, zmax=1,
        text_auto=".3f", aspect="auto",
    )
    fig_hm.update_layout(
        height=max(300, len(edition_labels) * 50 + 100),
        margin=dict(l=20, r=20, t=20, b=20),
        coloraxis_colorbar=dict(title="Hamming"),
    )
    fig_hm.update_traces(textfont_size=11)
    st.plotly_chart(fig_hm, width="stretch")

# ── Detailed cluster strings ───────────────────────────────────────────────────
st.divider()
with st.expander("Cluster sequences by edition and parameter value"):
    st.dataframe(
        df_all[[holdout_param, "edition", "lang", "n_tokens", "k_actual", "cluster_string"]],
        use_container_width=True,
        hide_index=True,
    )
