"""
Parameter Grid Search — NMF (ARI) — sweep combinations of non-k parameters and
report how cross-edition ARI varies with k (number of topics) for each combination.

Unlike HAC, NMF cannot sweep k cheaply post-fit: each (combo, k) requires a full
NMF fit. run_nmf_labels is cached per (src_id, n_chunks, min_df, max_df, k), so
the full grid is expensive on first load but instant on subsequent reruns.
"""

import os
import yaml
import itertools
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score
from toc import render_toc

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META  = cfg["sources"]
LANG_LABELS   = cfg["languages"]
K_VALS        = list(range(2, 21))
NMF_MAX_ITER  = cfg["model"]["nmf_max_iter"]
FIXED_MIN_DF  = 5
FIXED_NGRAM   = (1, 1)


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
def run_nmf_labels(src_id, token_path, n_chunks, min_df, max_df, k, ngram_range=(1, 1)):
    """Return dominant-topic label array (argmax of THETA) for one edition at one k."""
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
    if len(chunks_list) < max(3, k):
        return None
    try:
        vec = TfidfVectorizer(
            lowercase=True, max_df=max_df, min_df=min_df,
            strip_accents=None, norm="l2", ngram_range=ngram_range,
        )
        X = vec.fit_transform(chunks_list)
    except ValueError:
        return None
    if X.shape[1] < k:
        return None
    try:
        model = NMF(n_components=k, init="nndsvda", max_iter=NMF_MAX_ITER)
        THETA = model.fit_transform(X)
    except Exception:
        return None
    return np.argmax(THETA, axis=1)


def mean_pairwise_ari(label_arrays: list[np.ndarray]) -> float:
    n = len(label_arrays)
    if n < 2:
        return float("nan")
    scores = [
        adjusted_rand_score(label_arrays[i], label_arrays[j])
        for i in range(n) for j in range(i + 1, n)
    ]
    return float(np.mean(scores))


# ── Controls ──────────────────────────────────────────────────────────────────
st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)

st.title("Parameter Grid Search — NMF (ARI)")
render_toc([
    ("ARI vs. k",              "ari-chart"),
    ("Optimal k Distribution", "k-distribution"),
    ("Summary",                "summary"),
])
st.caption(
    "Sweeps combinations of `n_chunks` and `max_df`, running NMF for each k from 2–20 "
    "per edition. Dominant topic per chunk = argmax of the THETA matrix. "
    "ARI is permutation-invariant: structurally identical assignments always score 1.0. "
    "**Note:** each (combo, k) requires a full NMF fit — first load is slow; "
    "results are cached for instant reruns."
)

col1, col2 = st.columns(2)

nc_range = col1.slider("n_chunks range", min_value=5, max_value=50, value=(10, 40), step=5)
nc_vals  = list(range(nc_range[0], nc_range[1] + 1, 5))

maxdf_range = col2.slider("max_df range", min_value=0.20, max_value=0.95, value=(0.30, 0.70), step=0.05, format="%.2f")
_n_maxdf    = round((maxdf_range[1] - maxdf_range[0]) / 0.05) + 1
maxdf_vals  = [round(maxdf_range[0] + i * 0.05, 2) for i in range(_n_maxdf)]

n_combos = len(nc_vals) * len(maxdf_vals)
n_fits   = n_combos * len(K_VALS) * len(SOURCES_META)
st.caption(
    f"**{n_combos}** combinations × **{len(K_VALS)}** k values × **{len(SOURCES_META)}** editions "
    f"= **{n_fits}** NMF fits (est.)  ·  min_df={FIXED_MIN_DF} · ngram=(1,1) fixed"
)

# ── Grid computation ───────────────────────────────────────────────────────────
combos      = list(itertools.product(nc_vals, maxdf_vals))
token_files = {src_id: find_token_file(src_id) for src_id in SOURCES_META}

curve_rows   = []
summary_rows = []
total_steps  = len(combos) * len(K_VALS)
progress     = st.progress(0, text="Running grid…")
step         = 0

for nc, mxdf in combos:
    combo_label  = f"nc={nc} · max_df={mxdf:.2f}"
    combo_k_rows = []

    for k in K_VALS:
        step += 1
        progress.progress(
            step / total_steps,
            text=f"{combo_label}  ·  k={k}  ({step}/{total_steps})",
        )

        label_arrays = []
        for src_id in SOURCES_META:
            tp = token_files[src_id]
            if tp is None:
                continue
            labels = run_nmf_labels(src_id, tp, nc, FIXED_MIN_DF, mxdf, k, FIXED_NGRAM)
            if labels is not None:
                label_arrays.append(labels)

        if len(label_arrays) < 2:
            continue

        ari = mean_pairwise_ari(label_arrays)
        combo_k_rows.append({"k": k, "mean_ari": ari})
        curve_rows.append({
            "combo_label": combo_label,
            "n_chunks": nc, "max_df": mxdf,
            "k": k, "mean_ari": ari,
        })

    if combo_k_rows:
        best = max(combo_k_rows, key=lambda r: r["mean_ari"])
        summary_rows.append({
            "combo_label": combo_label,
            "n_chunks": nc, "max_df": mxdf,
            "k*": int(best["k"]),
            "max_ari": round(float(best["mean_ari"]), 4),
        })

progress.empty()

if not curve_rows:
    st.warning("No combinations produced valid results. Try adjusting the grid.")
    st.stop()

df_curves  = pd.DataFrame(curve_rows)
df_summary = pd.DataFrame(summary_rows).sort_values("max_ari", ascending=False).reset_index(drop=True)

# ── Spaghetti plot ─────────────────────────────────────────────────────────────
st.subheader("ARI vs. k — All Combinations", anchor="ari-chart")
st.caption(
    "Each gray curve = one parameter combination. "
    "Bold blue = mean across all combinations. "
    "A consistent peak indicates a robust optimal k."
)

fig = go.Figure()

for label, gdf in df_curves.groupby("combo_label", sort=False):
    fig.add_trace(go.Scatter(
        x=gdf["k"], y=gdf["mean_ari"],
        mode="lines",
        line=dict(color="#CCCCCC", width=1),
        showlegend=False,
        hovertemplate=f"{label}<br>k=%{{x}}<br>ARI=%{{y:.3f}}<extra></extra>",
    ))

mean_curve = df_curves.groupby("k")["mean_ari"].mean().reset_index()
fig.add_trace(go.Scatter(
    x=mean_curve["k"], y=mean_curve["mean_ari"],
    mode="lines+markers",
    line=dict(color="#1f77b4", width=3),
    marker=dict(size=6),
    name="Mean across combos",
    hovertemplate="mean · k=%{x}<br>ARI=%{y:.3f}<extra></extra>",
))

fig.update_layout(
    height=400,
    margin=dict(l=60, r=30, t=10, b=50),
    plot_bgcolor="white",
    xaxis=dict(title="k (topics)", dtick=1, showgrid=False, zeroline=False),
    yaxis=dict(title="Mean ARI", range=[-0.05, 1.05],
               showgrid=True, gridcolor="#EEEEEE", zeroline=False),
    legend=dict(x=0.02, y=0.02),
)
st.plotly_chart(fig, width="stretch")

# ── k* distribution ────────────────────────────────────────────────────────────
st.divider()
st.subheader("Distribution of Optimal k*", anchor="k-distribution")
st.caption("How many parameter combinations achieve their maximum ARI at each k.")

k_star_counts = (
    df_summary["k*"].value_counts()
    .reindex(range(2, 21), fill_value=0)
    .reset_index()
)
k_star_counts.columns = ["k*", "count"]

fig_bar = px.bar(k_star_counts, x="k*", y="count",
                 labels={"k*": "k (topics)", "count": "# combinations"})
fig_bar.update_layout(
    height=280,
    margin=dict(l=60, r=30, t=10, b=50),
    plot_bgcolor="white",
    xaxis=dict(dtick=1, showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False),
)
st.plotly_chart(fig_bar, width="stretch")

# ── Summary table ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("Summary — Sorted by Maximum ARI", anchor="summary")
st.caption("Most coherent parameter combinations first (higher ARI = more agreement across editions).")
st.dataframe(df_summary, use_container_width=True, hide_index=True)
