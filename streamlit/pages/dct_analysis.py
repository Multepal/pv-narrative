"""
Narrative DCT — Discrete Cosine Transform of PCA/HAC cluster presence signals.

For each cluster, the binary presence signal (1 where active, 0 elsewhere) is
decomposed into frequency components via DCT-II. Low-frequency coefficients
capture long narrative arcs; high-frequency coefficients capture rapid
alternation. A low-pass reconstruction (IDCT of the first N coefficients)
gives the structural envelope of each cluster's narrative presence.

Cross-edition section compares low-pass envelopes across all available editions
to show where editions agree or diverge on the narrative rhythm of each cluster.
"""

import os
import yaml
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.fft import dct, idct
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from toc import render_toc

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
LANG_LABELS  = cfg["languages"]

st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)


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
def run_lsa(src_id, token_path, n_chunks, min_df, max_df, n_components, ngram_range=(1, 1)):
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
        vec = TfidfVectorizer(lowercase=True, max_df=max_df, min_df=min_df,
                              strip_accents=None, norm="l2", ngram_range=ngram_range)
        X = vec.fit_transform(chunks_list)
    except ValueError:
        return None
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1] - 1)
    if n_comp < 2:
        return None
    THETA = TruncatedSVD(n_components=n_comp, random_state=42).fit_transform(X)
    return {"THETA": THETA, "n_chunks": len(chunks_list)}


def threshold_for_k(Z, k, n):
    k = max(2, min(k, n - 1))
    return float((Z[n - k - 1, 2] + Z[n - k, 2]) / 2)


def get_labels(THETA, k):
    """Ward HAC on PCA scores → normalized cluster labels (0..k-1 by first appearance)."""
    Z = linkage(pdist(THETA, metric="euclidean"), method="ward")
    n = THETA.shape[0]
    raw = fcluster(Z, threshold_for_k(Z, k, n), criterion="distance")
    mapping = {}
    out = np.zeros(len(raw), dtype=int)
    for i, lbl in enumerate(raw):
        if lbl not in mapping:
            mapping[lbl] = len(mapping)
        out[i] = mapping[lbl]
    return out


def binary_signals(labels, k):
    """(k × n_chunks) binary matrix — row i is 1 where cluster i is active."""
    n = len(labels)
    mat = np.zeros((k, n))
    for t, c in enumerate(labels):
        if c < k:
            mat[c, t] = 1.0
    return mat


def dct_matrix(signals):
    """DCT-II of each row. Returns (k × n_chunks) coefficient matrix."""
    return np.array([dct(row, type=2, norm="ortho") for row in signals])


def low_pass(signal, n_keep):
    """Reconstruct signal keeping only the first n_keep DCT coefficients."""
    coeffs = dct(signal, type=2, norm="ortho")
    filtered = np.zeros_like(coeffs)
    filtered[:n_keep] = coeffs[:n_keep]
    return idct(filtered, type=2, norm="ortho")


# ── Controls ──────────────────────────────────────────────────────────────────
st.title("Narrative DCT Analysis — Popol Wuj")
render_toc([
    ("DCT Spectra",             "dct-spectra"),
    ("Low-pass Reconstruction", "low-pass"),
    ("Cross-edition Envelopes", "cross-edition"),
])
st.caption(
    "Applies the Discrete Cosine Transform to each cluster's binary presence signal "
    "across narrative time. **Low-frequency coefficients** = long narrative arcs; "
    "**high-frequency** = rapid oscillation. The low-pass reconstruction is the "
    "structural envelope of each cluster — the narrative curve stripped of noise."
)

src_ids = list(SOURCES_META.keys())
_c      = cfg["controls"]
_col_r  = cfg["layout"]["column_ratios"]
cols    = st.columns(_col_r[:5])

src_id    = cols[0].selectbox(
    "Source", src_ids, index=src_ids.index(_c["default_source"]),
    format_func=lambda x: f"{SOURCES_META[x]['label']} ({LANG_LABELS[SOURCES_META[x]['lang']]})")
n_chunks  = cols[1].number_input("n_chunks",  min_value=5,  max_value=200, value=20,  step=1)
min_df    = cols[2].number_input("min_df",    _c["min_df"]["min"], _c["min_df"]["max"],
                                  _c["min_df"]["default"], step=_c["min_df"]["step"])
max_df    = cols[2].number_input("max_df",    _c["max_df"]["min"], _c["max_df"]["max"],
                                  _c["max_df"]["default"], step=_c["max_df"]["step"], format="%.2f")
k         = cols[3].number_input("k (clusters)", _c["n_clusters"]["min"], _c["n_clusters"]["max"],
                                  _c["n_clusters"]["default"], step=1)
n_components = cols[4].number_input("PCA components", 2, _c["n_topics"]["max"], 10, step=1)

n_keep_max = max(2, min(12, int(n_chunks) // 2))
n_keep = st.slider("Low-pass cutoff (DCT coefficients to keep)", 1, n_keep_max,
                   min(4, n_keep_max), step=1,
                   help="Keep the first N DCT coefficients; reconstruct with IDCT. Lower = smoother.")

st.divider()

# ── Run LSA + HAC for the selected edition ────────────────────────────────────
token_path = find_token_file(src_id)
if token_path is None:
    st.warning(f"Token file not found for `{src_id}`.")
    st.stop()

result = run_lsa(src_id, token_path, int(n_chunks), int(min_df), float(max_df),
                 int(n_components))
if result is None:
    st.warning("LSA couldn't run — try adjusting parameters.")
    st.stop()

THETA   = result["THETA"]
n_act   = result["n_chunks"]
labels  = get_labels(THETA, int(k))
k_act   = int(labels.max()) + 1  # actual distinct clusters (may be < k)

signals = binary_signals(labels, k_act)
coeffs  = dct_matrix(signals)

_palette     = px.colors.qualitative.Plotly
cluster_names = [f"Cluster {chr(65 + i)}" for i in range(k_act)]
colors        = [_palette[i % len(_palette)] for i in range(k_act)]
positions     = np.arange(n_act)
freq_bins     = np.arange(n_act)

# ── Section 1: DCT Spectra ─────────────────────────────────────────────────────
st.subheader("DCT Spectra", anchor="dct-spectra")
st.caption(
    "Magnitude of each DCT coefficient for each cluster's binary presence signal. "
    "Bin 0 = DC (mean presence); bins 1–N = increasing frequency. "
    "The dashed line marks the low-pass cutoff — coefficients to the right are zeroed "
    "in the reconstruction below."
)

_show_bins = min(n_act, 16)  # only show the informative low-frequency portion

fig_spec = go.Figure()
for i in range(k_act):
    fig_spec.add_trace(go.Scatter(
        x=freq_bins[:_show_bins],
        y=np.abs(coeffs[i, :_show_bins]),
        mode="lines+markers",
        name=cluster_names[i],
        line=dict(color=colors[i], width=2),
        marker=dict(size=6),
        hovertemplate=f"{cluster_names[i]}<br>bin %{{x}}<br>|coeff| = %{{y:.4f}}<extra></extra>",
    ))
fig_spec.add_vline(x=n_keep - 0.5, line_dash="dash", line_color="gray", line_width=1.5,
                   annotation_text=f"cutoff = {n_keep}", annotation_position="top right")
fig_spec.update_layout(
    height=340,
    margin=dict(l=60, r=30, t=20, b=50),
    plot_bgcolor="white",
    xaxis=dict(title="DCT frequency bin", dtick=1, showgrid=False, zeroline=False,
               range=[-0.5, _show_bins - 0.5]),
    yaxis=dict(title="|DCT coefficient|", showgrid=True, gridcolor="#EEEEEE", zeroline=False),
    legend=dict(x=0.75, y=0.98),
)
st.plotly_chart(fig_spec, width="stretch")

# ── Section 2: Low-pass Reconstruction ────────────────────────────────────────
st.divider()
st.subheader("Low-pass Reconstruction", anchor="low-pass")
st.caption(
    "IDCT of the first **{n}** coefficients — the structural envelope of each cluster's "
    "narrative presence. The dashed line is the raw binary signal; the solid line is the "
    "smooth reconstruction. Adjust the **Low-pass cutoff** slider above to control smoothness."
    .format(n=n_keep)
)

fig_lp = go.Figure()
x_pos = np.linspace(1, 100, n_act)

for i in range(k_act):
    recon = low_pass(signals[i], n_keep)
    # raw signal as faint dashed background
    fig_lp.add_trace(go.Scatter(
        x=x_pos, y=signals[i],
        mode="lines",
        line=dict(color=colors[i], width=1, dash="dot"),
        opacity=0.3,
        showlegend=False,
        hoverinfo="skip",
    ))
    # low-pass reconstruction as bold solid line
    fig_lp.add_trace(go.Scatter(
        x=x_pos, y=recon,
        mode="lines",
        name=cluster_names[i],
        line=dict(color=colors[i], width=2.5),
        hovertemplate=f"{cluster_names[i]}<br>pos %{{x:.0f}}<br>envelope = %{{y:.3f}}<extra></extra>",
    ))

fig_lp.update_layout(
    height=380,
    margin=dict(l=60, r=30, t=20, b=50),
    plot_bgcolor="white",
    xaxis=dict(title="Narrative position (1–100)", showgrid=False, zeroline=False),
    yaxis=dict(title="Presence (0–1)", range=[-0.15, 1.15],
               showgrid=True, gridcolor="#EEEEEE", zeroline=True, zerolinecolor="#CCCCCC"),
    legend=dict(x=0.01, y=0.98),
)
st.plotly_chart(fig_lp, width="stretch")

# ── Section 3: Cross-edition Envelopes ────────────────────────────────────────
st.divider()
st.subheader("Cross-edition Envelopes", anchor="cross-edition")
st.caption(
    "Low-pass reconstruction for every available edition using the same parameters. "
    "Clusters are aligned by **order of first appearance** (Cluster A = first-appearing "
    "cluster in narrative order). Convergence of lines = editions agree on the narrative "
    "rhythm of that cluster; divergence = structural disagreement."
)

all_edition_data = {}
missing = []
progress = st.progress(0, text="Computing cross-edition LSA…")

src_list = list(SOURCES_META.keys())
for ei, sid in enumerate(src_list):
    progress.progress((ei + 1) / len(src_list), text=f"{SOURCES_META[sid]['label']}")
    tp = find_token_file(sid)
    if tp is None:
        missing.append(sid)
        continue
    res = run_lsa(sid, tp, int(n_chunks), int(min_df), float(max_df), int(n_components))
    if res is None:
        missing.append(sid)
        continue
    lbl = get_labels(res["THETA"], int(k))
    k_ed = int(lbl.max()) + 1
    sig  = binary_signals(lbl, k_ed)
    recons = np.array([low_pass(sig[i], n_keep) if i < k_ed else np.zeros(n_act)
                       for i in range(k_act)])
    all_edition_data[sid] = recons

progress.empty()

if missing:
    st.caption(f"Editions skipped (no token file): {', '.join(missing)}")

if len(all_edition_data) >= 2:
    _edition_colors = px.colors.qualitative.Dark24
    cluster_cols = st.columns(min(k_act, 3))

    for ci in range(k_act):
        col = cluster_cols[ci % len(cluster_cols)]
        fig_ed = go.Figure()
        for ei, (sid, recons) in enumerate(all_edition_data.items()):
            meta = SOURCES_META[sid]
            ed_color = _edition_colors[ei % len(_edition_colors)]
            fig_ed.add_trace(go.Scatter(
                x=x_pos, y=recons[ci],
                mode="lines",
                name=f"{meta['label']}",
                line=dict(color=ed_color, width=1.8),
                hovertemplate=(
                    f"{meta['label']}<br>pos %{{x:.0f}}<br>envelope = %{{y:.3f}}<extra></extra>"
                ),
            ))
        fig_ed.update_layout(
            title=dict(text=cluster_names[ci], font=dict(size=13), x=0.05),
            height=260,
            margin=dict(l=40, r=10, t=36, b=40),
            plot_bgcolor="white",
            xaxis=dict(title="Narrative position", showgrid=False, zeroline=False),
            yaxis=dict(range=[-0.2, 1.2], showgrid=True, gridcolor="#EEEEEE",
                       zeroline=True, zerolinecolor="#CCCCCC"),
            showlegend=(ci == 0),
            legend=dict(font=dict(size=9), x=0, y=1),
        )
        col.plotly_chart(fig_ed, width="stretch", key=f"dct_ed_{ci}")
else:
    st.info("Need at least 2 editions with token files for cross-edition comparison.")
