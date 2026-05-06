"""
Model Diagnostics for Narrative Structure of the Popol Wuj
"""

import os
import yaml
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage as ward_linkage

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
LANG_LABELS  = cfg["languages"]


def find_token_file(src_id: str) -> str | None:
    candidates = [
        # os.path.join(APP_DIR, src_id, f"{src_id}-TOKEN.csv"),
        # os.path.join(APP_DIR, "ensemble", f"{src_id}-TOKEN.csv"),
        # os.path.join(APP_DIR, f"{src_id}-TOKEN.csv"),
        # os.path.join(APP_DIR, "..", src_id, f"{src_id}-TOKEN.csv"),
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
def compute_vocab_stats(src_id, token_path, chunk_size, overlap_int):
    """Return (n_chunks, suggested_min_df, suggested_max_df) from a vocab frequency scan."""
    TOKEN = load_tokens(src_id, token_path)
    tokens = TOKEN["term_str"].dropna().to_list()
    token_arr = np.array(tokens)
    step = max(1, chunk_size - overlap_int)
    if len(token_arr) < chunk_size:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(token_arr, chunk_size)[::step]
    chunks = [" ".join(row) for row in windows]
    n_chunks = len(chunks)
    if n_chunks < 2:
        return None
    vec = TfidfVectorizer(lowercase=True, max_df=1.0, min_df=1, strip_accents=None)
    try:
        X = vec.fit_transform(chunks)
    except ValueError:
        return None
    doc_freqs = np.asarray((X > 0).sum(axis=0)).flatten() / n_chunks
    min_df_sug = max(2, round(0.02 * n_chunks))
    # Restrict knee detection to words appearing in ≥5% of chunks — the range
    # where function words and common content words live. The full distribution's
    # long rare-word tail would pull the knee to a meaninglessly low value.
    high = np.sort(doc_freqs[doc_freqs >= 0.05])[::-1]
    if len(high) >= 3:
        x = np.linspace(0, 1, len(high))
        y_range = high[0] - high[-1]
        y_norm = (high - high[-1]) / y_range if y_range > 1e-10 else np.ones(len(high))
        line = np.array([1.0, -1.0])
        vecs = np.column_stack([x, y_norm]) - np.array([0.0, 1.0])
        line3 = np.append(line / np.linalg.norm(line), 0)
        vecs3 = np.hstack([vecs, np.zeros((len(vecs), 1))])
        dists = np.abs(np.cross(line3, vecs3)[:, 2])
        max_df_sug = float(np.clip(round(high[np.argmax(dists)], 2), 0.05, 0.95))
    else:
        max_df_sug = 0.35
    return n_chunks, min_df_sug, max_df_sug


def _umass_coherence(X_bin, feature_names, top_words):
    """UMass coherence for one topic (Mimno et al. 2011)."""
    name_idx = {w: i for i, w in enumerate(feature_names)}
    indices = [name_idx[w] for w in top_words if w in name_idx]
    if len(indices) < 2:
        return 0.0
    score, count = 0.0, 0
    for i in range(1, len(indices)):
        for j in range(i):
            co   = X_bin[:, indices[i]].multiply(X_bin[:, indices[j]]).sum()
            df_j = X_bin[:, indices[j]].sum()
            if df_j > 0:
                score += np.log((co + 1.0) / df_j)
                count += 1
    return score / count if count > 0 else 0.0


@st.cache_data(show_spinner="Running elbow analysis…")
def run_elbow(src_id, token_path, chunk_size, overlap_int, min_df, max_df, max_topics, model_type):
    TOKEN = load_tokens(src_id, token_path)
    tokens = TOKEN["term_str"].dropna().to_list()
    step = max(1, chunk_size - overlap_int)
    token_arr = np.array(tokens)
    if len(token_arr) < chunk_size:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(token_arr, chunk_size)[::step]
    chunks_s = pd.Series(
        np.apply_along_axis(lambda row: " ".join(row), axis=1, arr=windows)
    ).loc[lambda s: s.str.split().str.len() >= 50]
    chunks_list = chunks_s.tolist()
    if len(chunks_list) < 2:
        return None
    try:
        if model_type == "NMF":
            vec = TfidfVectorizer(lowercase=True, max_df=max_df, min_df=min_df,
                                  strip_accents=None, norm="l2")
        else:
            vec = CountVectorizer(lowercase=True, max_df=max_df, min_df=min_df,
                                  strip_accents=None)
        X = vec.fit_transform(chunks_list)
    except ValueError:
        return None

    X_bin = (X > 0).tocsc()
    words = vec.get_feature_names_out()
    n_top = 10

    rows = []
    for n in range(2, max_topics + 1):
        if model_type == "NMF":
            m = NMF(n_components=n, init="nndsvda", max_iter=cfg["model"]["nmf_max_iter"])
            m.fit(X)
        else:
            m = LatentDirichletAllocation(n_components=n, random_state=42, max_iter=20)
            m.fit(X)
        phi = [
            {words[j]: m.components_[i][j] for j in m.components_[i].argsort()[::-1][:n_top]}
            for i in range(n)
        ]
        coherence = float(np.mean([_umass_coherence(X_bin, words, list(p.keys())) for p in phi]))
        dists = cosine_distances(m.components_)
        idx = np.triu_indices(n, k=1)
        independence = float(np.mean(dists[idx])) if n > 1 else 0.0
        rows.append({"n_topics": n, "coherence": coherence, "independence": independence})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Running topic model…")
def run_model(src_id, token_path, chunk_size, overlap_int, min_df, max_df, n_topics, model_type, n_top_words):
    TOKEN = load_tokens(src_id, token_path)
    tokens = TOKEN["term_str"].dropna().to_list()
    step = max(1, chunk_size - overlap_int)
    token_arr = np.array(tokens)
    if len(token_arr) < chunk_size:
        return None, None, None
    windows = np.lib.stride_tricks.sliding_window_view(token_arr, chunk_size)[::step]
    chunks_s = pd.Series(
        np.apply_along_axis(lambda row: " ".join(row), axis=1, arr=windows)
    ).loc[lambda s: s.str.split().str.len() >= 50]
    chunks_list = chunks_s.tolist()
    if len(chunks_list) < 2:
        return None, None, None
    try:
        if model_type == "NMF":
            vec = TfidfVectorizer(lowercase=True, max_df=max_df, min_df=min_df,
                                  strip_accents=None, norm="l2")
            X = vec.fit_transform(chunks_list)
            model = NMF(n_components=n_topics, init="nndsvda", max_iter=cfg["model"]["nmf_max_iter"])
        else:
            vec = CountVectorizer(lowercase=True, max_df=max_df, min_df=min_df,
                                  strip_accents=None)
            X = vec.fit_transform(chunks_list)
            model = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
    except ValueError:
        return None, None, None
    THETA = pd.DataFrame(model.fit_transform(X))
    THETA.index.name, THETA.columns.name = "chunk_id", "topic_id"
    words = vec.get_feature_names_out()
    PHI = [
        {words[j]: model.components_[i][j] for j in model.components_[i].argsort()[::-1][:n_top_words]}
        for i in range(n_topics)
    ]
    return THETA, PHI, chunks_list


def render_heatmap(THETA, PHI, chunks_list, key="heatmap"):
    """Render a topic heatmap into the current Streamlit container."""
    _v = cfg["visualization"]
    n_chunks = len(chunks_list)
    n_topics = THETA.shape[1]
    _row_h = cfg["layout"]["heatmap_row_height_px"]
    height = n_topics * _row_h + 60  # 60px = top+bottom margin overhead
    _preview_len = _v["preview_len"]

    def _wrap(text, width=_v["wrap_width"]):
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

    _chunk_previews = [
        _wrap((c[:_preview_len] + "…") if len(c) > _preview_len else c)
        for c in chunks_list
    ]
    _topic_seq = THETA.idxmax(axis=1).tolist()
    _topic_order = []
    for t in _topic_seq:
        if t not in _topic_order:
            _topic_order.append(t)
    for t in range(THETA.shape[1]):
        if t not in _topic_order:
            _topic_order.append(t)
    _topic_order_plot = list(reversed(_topic_order))
    _topic_labels = [f"Topic {i}" for i in _topic_order_plot]

    _x_scaled = np.round(np.linspace(1, 100, n_chunks)).astype(int)
    fig = px.imshow(
        THETA.T.loc[_topic_order_plot].values,
        y=_topic_labels,
        x=_x_scaled,
        aspect="auto",
        color_continuous_scale=_v["heatmap_color_scale"],
        labels=dict(x="Position (1–100)", y="Topic"),
    )
    _topic_words = [", ".join(list(PHI[t].keys())[:_v["hover_top_words"]]) for t in _topic_order_plot]
    _customdata = np.empty((len(_topic_order_plot), n_chunks, 2), dtype=object)
    _customdata[:, :, 0] = np.tile(_chunk_previews, (len(_topic_order_plot), 1))
    _customdata[:, :, 1] = np.array(_topic_words)[:, np.newaxis]
    fig.update_traces(
        customdata=_customdata,
        hovertemplate=(
            "<b>Topic %{y} · Position %{x}</b><br>"
            "Weight: %{z:.3f}<br>"
            "<i>%{customdata[1]}</i><br><br>"
            "%{customdata[0]}<extra></extra>"
        ),
    )
    fig.update_layout(height=height, margin=dict(l=60, r=20, t=20, b=40),
                      coloraxis_showscale=False)
    st.plotly_chart(fig, width='stretch', key=key)


st.title("Model Diagnostics — Popol Wuj")

# ── Phase 1 controls (no n_topics — k is chosen by clicking the coherence plot) ──
src_ids = list(SOURCES_META.keys())
_col_ratios = cfg["layout"]["column_ratios"]
cols = st.columns(_col_ratios[:-1])  # drop last slot (n_topics not needed here)

_c = cfg["controls"]
src_id     = cols[0].selectbox(
    "Source", src_ids, index=src_ids.index(_c["default_source"]),
    format_func=lambda x: f"{SOURCES_META[x]['label']} ({LANG_LABELS[SOURCES_META[x]['lang']]})")

_early_token_path = find_token_file(src_id)
_n_tokens_early = len(load_tokens(src_id, _early_token_path)) if _early_token_path else None

_cp = _c["chunk_pct"]
chunk_pct  = cols[1].number_input("Chunk %", _cp["min"], _cp["max"], _cp["default"], step=_cp["step"], format="%.3f")
if _n_tokens_early:
    cols[1].caption(f"{int(chunk_pct * _n_tokens_early):,} tokens")
chunk_size = max(50, int(chunk_pct * _n_tokens_early)) if _n_tokens_early else 100
overlap    = cols[2].number_input("Overlap", _c["overlap"]["min"], _c["overlap"]["max"], _c["overlap"]["default"], step=_c["overlap"]["step"], format="%.2f")
if _n_tokens_early:
    cols[2].caption(f"{int(overlap * chunk_size):,} tokens")
overlap_int = int(overlap * chunk_size)

_vstats = compute_vocab_stats(src_id, _early_token_path, chunk_size, overlap_int) if _early_token_path else None

min_df     = cols[3].number_input("min_df", _c["min_df"]["min"], _c["min_df"]["max"], _c["min_df"]["default"], step=_c["min_df"]["step"])
if _vstats:
    cols[3].caption(f"→ {_vstats[1]} (2% of {_vstats[0]} chunks)")
max_df     = cols[4].number_input("max_df", _c["max_df"]["min"], _c["max_df"]["max"], _c["max_df"]["default"], step=_c["max_df"]["step"], format="%.2f")
if _vstats:
    cols[4].caption(f"→ {_vstats[2]:.2f} (vocab knee)")
model_type = cols[5].selectbox("Model", ["NMF", "LDA"])

st.divider()

# ── File resolution ───────────────────────────────────────────────────────────
token_path = find_token_file(src_id)
if token_path is None:
    st.warning(f"Token file not found for `{src_id}`.")
    st.stop()

# ── Session state: clear selected k when Phase 1 parameters change ────────────
_params_key = f"{src_id}_{chunk_size}_{overlap_int}_{min_df}_{max_df}_{model_type}"
if st.session_state.get("_diag_params_key") != _params_key:
    st.session_state["_diag_params_key"] = _params_key
    st.session_state["selected_k"] = None

# ── Phase 1: coherence curve + topic similarity dendrogram ───────────────────
elbow_df = run_elbow(src_id, token_path, chunk_size, overlap_int, min_df, max_df,
                     _c["n_topics"]["max"], model_type)

_m = dict(l=10, r=120, t=40, b=40)
_selected_k = st.session_state.get("selected_k")
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Coherence & Independence vs. Number of Topics")
    st.caption(
        "UMass coherence (blue, left axis): higher = more coherent. "
        "Mean pairwise cosine distance (red, right axis): higher = more independent. "
        "**Click a point to select k.**"
    )
    if elbow_df is not None:
        fig_coh = make_subplots(specs=[[{"secondary_y": True}]])
        fig_coh.add_trace(
            go.Scatter(x=elbow_df["n_topics"], y=elbow_df["coherence"],
                       mode="lines+markers", name="Coherence",
                       line=dict(color="#636EFA")),
            secondary_y=False,
        )
        fig_coh.add_trace(
            go.Scatter(x=elbow_df["n_topics"], y=elbow_df["independence"],
                       mode="lines+markers", name="Independence",
                       line=dict(color="#EF553B")),
            secondary_y=True,
        )
        if _selected_k is not None:
            fig_coh.add_vline(x=_selected_k, line_dash="dash", line_color="gray", line_width=2,
                              annotation_text=f"k = {_selected_k}", annotation_position="top right")
        fig_coh.update_yaxes(title_text="Mean UMass coherence", secondary_y=False)
        fig_coh.update_yaxes(title_text="Mean pairwise cosine distance", secondary_y=True)
        fig_coh.update_xaxes(title_text="Number of topics (k)")
        fig_coh.update_layout(height=380, margin=_m,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        _coh_event = st.plotly_chart(fig_coh, width='stretch',
                                     key=f"coh_{_params_key}", on_select="rerun")
        if _coh_event.selection.points:
            _clicked_k = int(_coh_event.selection.points[0]["x"])
            if _clicked_k != st.session_state.get("selected_k"):
                st.session_state["selected_k"] = _clicked_k
                st.rerun()
        if elbow_df is not None:
            _peak_k = int(elbow_df.loc[elbow_df["coherence"].idxmax(), "n_topics"])
            st.caption(f"→ k = {_peak_k} (coherence peak)")
        if _selected_k is not None:
            st.success(f"k = {_selected_k} selected — twin heatmaps below ↓")
        else:
            st.info("Click a point on the curve to select k.")
    else:
        st.warning("Elbow analysis could not run — try adjusting chunk size, min_df, or max_df.")

with col_right:
    st.subheader("Topic Similarity")
    if _selected_k is None:
        st.info("Select k on the coherence curve to show topic similarity.")
    elif _selected_k < 3:
        st.caption("Need at least 3 topics for a dendrogram.")
    else:
        _THETA_d, _PHI_d, _ = run_model(
            src_id, token_path, chunk_size, overlap_int,
            min_df, max_df, _selected_k, model_type, _c["n_top_words"]["default"]
        )
        if _PHI_d is not None:
            _all_words = sorted({w for p in _PHI_d for w in p})
            _phi_matrix = np.array([[p.get(w, 0.0) for w in _all_words] for p in _PHI_d])
            _phi_norm = normalize(_phi_matrix, norm='l2')
            _fig_dend = ff.create_dendrogram(
                _phi_norm,
                labels=[f"Topic {i}" for i in range(_selected_k)],
                distfun=lambda x: pdist(x, metric='euclidean'),
                linkagefun=lambda x: ward_linkage(x, method='ward'),
            )
            _fig_dend.update_layout(height=380, margin=_m)
            st.plotly_chart(_fig_dend, width='stretch', key=f"dend_{_params_key}_{_selected_k}")
        else:
            st.warning("Model couldn't run — try adjusting parameters.")

# ── Phase 2: twin heatmaps triggered by click ─────────────────────────────────
_selected_k = st.session_state.get("selected_k")
if _selected_k is not None:
    st.divider()
    st.subheader(f"Twin Heatmaps — k = {_selected_k}")
    st.caption(
        "Adjust chunk size, overlap, and vocabulary filters independently for each heatmap. "
        "Both use the same k and source."
    )

    _n_top_words = _c["n_top_words"]["default"]
    _heat_cols = st.columns(2)

    for _label, _pfx, _hcol in [("A", "ha", _heat_cols[0]), ("B", "hb", _heat_cols[1])]:
        with _hcol:
            st.markdown(f"**Heatmap {_label}**")
            _wc = st.columns(4)
            _h_chunk_pct = _wc[0].number_input("Chunk %", _cp["min"], _cp["max"],
                                                chunk_pct, step=_cp["step"], format="%.3f",
                                                key=f"{_pfx}_chunk")
            _h_chunk = max(50, int(_h_chunk_pct * _n_tokens_early))
            _wc[0].caption(f"{_h_chunk:,} tokens")
            _h_overlap = _wc[1].number_input("Overlap", _c["overlap"]["min"], _c["overlap"]["max"],
                                              overlap, step=_c["overlap"]["step"], format="%.2f",
                                              key=f"{_pfx}_overlap")
            _wc[1].caption(f"{int(_h_overlap * _h_chunk):,} tokens")
            _h_min_df  = _wc[2].number_input("min_df", _c["min_df"]["min"], _c["min_df"]["max"],
                                              min_df, step=_c["min_df"]["step"],
                                              key=f"{_pfx}_min_df")
            _h_max_df  = _wc[3].number_input("max_df", _c["max_df"]["min"], _c["max_df"]["max"],
                                              max_df, step=_c["max_df"]["step"], format="%.2f",
                                              key=f"{_pfx}_max_df")
            _h_overlap_int = int(_h_overlap * _h_chunk)

            THETA, PHI, chunks_list = run_model(
                src_id, token_path, _h_chunk, _h_overlap_int,
                _h_min_df, _h_max_df, _selected_k, model_type, _n_top_words
            )
            if THETA is None:
                st.warning("Model couldn't run — try adjusting parameters.")
            else:
                n_chunks = len(chunks_list)
                st.caption(f"{n_chunks} chunks · {THETA.shape[0]} × {THETA.shape[1]} · chunk = {_h_chunk_pct * 100:.1f}% ({_h_chunk:,} tokens)")
                render_heatmap(THETA, PHI, chunks_list, key=f"heat_{_pfx}_{_selected_k}")
