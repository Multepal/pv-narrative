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
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_distances

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
def compute_vocab_growth(src_id: str, token_path: str, n_points: int = 300):
    """Fit Heaps' Law; return n* (vocabulary saturation point)."""
    TOKEN = load_tokens(src_id, token_path)
    tokens = TOKEN["term_str"].dropna().to_list()
    n_total = len(tokens)

    ns = np.unique(np.geomspace(1, n_total, n_points).astype(int))
    seen, vs, prev = set(), [], 0
    for n in ns:
        for tok in tokens[prev:n]:
            seen.add(tok)
        vs.append(len(seen))
        prev = n
    ns, vs = np.array(ns), np.array(vs)

    beta, log_K = np.polyfit(np.log(ns), np.log(vs), 1)
    K = np.exp(log_K)

    if 0.0 < beta < 1.0:
        n_star = int(0.10 ** (1.0 / (beta - 1.0)))
        n_star = max(10, min(n_star, n_total // 2))
    else:
        n_star = None

    return ns, vs, K, float(beta), n_star, n_total


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
            m = NMF(n_components=n, init="nndsvd", max_iter=500)
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
            model = NMF(n_components=n_topics, init="nndsvd", max_iter=500)
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
min_df     = cols[3].number_input("min_df", _c["min_df"]["min"], _c["min_df"]["max"], _c["min_df"]["default"], step=_c["min_df"]["step"])
max_df     = cols[4].number_input("max_df", _c["max_df"]["min"], _c["max_df"]["max"], _c["max_df"]["default"], step=_c["max_df"]["step"], format="%.2f")
model_type = cols[5].selectbox("Model", ["NMF", "LDA"])

overlap_int = int(overlap * chunk_size)

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

# ── Phase 1: vocabulary saturation + coherence curve ─────────────────────────
_ns, _vs, _K, _beta, _n_star, _n_total = compute_vocab_growth(src_id, token_path)
elbow_df = run_elbow(src_id, token_path, chunk_size, overlap_int, min_df, max_df,
                     _c["n_topics"]["max"], model_type)

_m = dict(l=10, r=120, t=40, b=40)
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Vocabulary Saturation (Heaps' Law)")
    st.caption(
        f"V(n) = {_K:.1f} · n^{_beta:.3f} · "
        f"{_n_total:,} tokens · {_vs[-1]:,} types · "
        + (f"n* = {_n_star}" if _n_star is not None else "n* unavailable")
        + f" · chunk = {chunk_pct * 100:.1f}% ({chunk_size:,} tokens)"
    )
    _ns_fit = np.geomspace(1, _n_total, 300)
    _gain_df = pd.DataFrame({
        "Tokens (n)": _ns_fit,
        "Marginal vocabulary gain": _ns_fit ** (_beta - 1.0),
    })
    fig_gain = px.line(_gain_df, x="Tokens (n)", y="Marginal vocabulary gain")
    fig_gain.add_hline(y=0.10, line_dash="dash", line_color="green", line_width=1.5,
                       annotation_text="10% threshold", annotation_position="right")
    if _n_star is not None:
        fig_gain.add_vline(x=_n_star, line_dash="dash", line_color="green", line_width=1.5,
                           annotation_text=f"n* = {_n_star}", annotation_position="top left")
    fig_gain.add_vline(x=chunk_size, line_dash="dot", line_color="gray", line_width=1.5,
                       annotation_text=f"chunk = {chunk_size}", annotation_position="bottom right")
    fig_gain.update_layout(height=380, margin=_m)
    st.plotly_chart(fig_gain, width='stretch', key="vocab_gain")

with col_right:
    st.subheader("Coherence & Independence vs. Number of Topics")
    st.caption(
        "UMass coherence (blue, left axis): higher = more coherent. "
        "Mean pairwise cosine distance (red, right axis): higher = more independent. "
        "**Click a point to open the twin heatmap for that k.**"
    )
    if elbow_df is not None:
        _selected_k = st.session_state.get("selected_k")
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
        if _selected_k is not None:
            st.success(f"k = {_selected_k} selected — twin heatmaps below ↓")
        else:
            st.info("Click a point on the curve to fit the topic model.")
    else:
        st.warning("Elbow analysis could not run — try adjusting chunk size, min_df, or max_df.")

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
            _h_chunk = max(50, int(_h_chunk_pct * _n_total))
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
