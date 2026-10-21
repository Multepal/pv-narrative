"""
Parameter Grid Search — Cross-Edition ARI (method selector).

Sweeps combinations of n_chunks and max_df; for each k, scores cross-edition ARI as
excess over a cyclic-shift chance baseline and chooses k* by that excess, optionally
on K'iche' pairs only with translations held out for validation.

  TF-IDF / LSA / Cosine-Sim — k swept cheaply post-linkage (fcluster only)
  NMF                        — full fit per (combo, k); cached but slow first run
"""

import os
import yaml
import itertools
import streamlit as st
from toc import render_toc
from grid_search_boundary_core import FIXED_MIN_DF, N_COMPONENTS
from grid_search_ui import K_VALS, run_grid, selection_control, render_selection_results

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]

METHOD_LABELS = {
    "TF-IDF":     "TF-IDF → Ward HAC",
    "LSA":        f"TF-IDF → PCA/LSA (centered, n={N_COMPONENTS}) → Ward HAC",
    "Cosine-Sim": "TF-IDF → cosine-similarity matrix → Ward HAC",
    "NMF":        "TF-IDF → NMF (full fit per k)",
}

METHOD_K_LABEL = {
    "TF-IDF": "k (clusters)", "LSA": "k (clusters)",
    "Cosine-Sim": "k (clusters)", "NMF": "k (topics)",
}


# ── Controls ──────────────────────────────────────────────────────────────────
st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)

st.title("Parameter Grid Search — ARI")
render_toc([
    ("ARI Excess vs. k",       "agreement-chart"),
    ("Optimal k Distribution", "k-distribution"),
    ("k* and Robustness",      "best-params"),
    ("Resolution (n_chunks)",  "resolution"),
    ("Summary",                "summary"),
])
st.caption(
    "Sweeps `n_chunks` × `max_df` combinations; reports cross-edition ARI in excess of chance vs. k. "
    "For HAC methods k is swept cheaply post-linkage. "
    "For NMF each (combo, k) requires a full fit — first load is slow; results are cached."
)

col_m, col1, col2 = st.columns(3)
method = col_m.selectbox("Method", list(METHOD_LABELS.keys()),
                          format_func=lambda x: METHOD_LABELS[x])
k_label = METHOD_K_LABEL[method]

nc_range    = col1.slider("n_chunks range", min_value=5, max_value=50, value=(10, 40), step=5)
nc_vals     = list(range(nc_range[0], nc_range[1] + 1, 5))
maxdf_range = col2.slider("max_df range", min_value=0.20, max_value=0.95, value=(0.30, 0.70),
                           step=0.05, format="%.2f")
_n_maxdf    = round((maxdf_range[1] - maxdf_range[0]) / 0.05) + 1
maxdf_vals  = [round(maxdf_range[0] + i * 0.05, 2) for i in range(_n_maxdf)]

sel = selection_control(st, key="gs_ari_sel")

n_combos = len(nc_vals) * len(maxdf_vals)
if method == "NMF":
    n_fits = n_combos * len(K_VALS) * len(SOURCES_META)
    st.caption(
        f"**{n_combos}** combinations × **{len(K_VALS)}** k values × **{len(SOURCES_META)}** editions "
        f"= **{n_fits}** NMF fits (est.)  ·  min_df={FIXED_MIN_DF} · ngram=(1,1) fixed"
    )
else:
    st.caption(
        f"**{n_combos}** combinations · **{n_combos * len(SOURCES_META)}** linkage runs (est.)  "
        f"·  min_df={FIXED_MIN_DF} · ngram=(1,1) fixed"
    )

# ── Grid computation ───────────────────────────────────────────────────────────
df_curves = run_grid(method, list(itertools.product(nc_vals, maxdf_vals)), "ARI")

if df_curves.empty:
    st.warning("No combinations produced valid results. Try adjusting the grid.")
    st.stop()

render_selection_results(df_curves, sel, "ARI", k_label, method, handoff=True, resolution=True)
