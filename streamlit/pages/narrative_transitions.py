"""
Narrative Transitions — NMF topic sequence and topic-to-topic transition matrix.
"""

import os
import yaml
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import entropy as scipy_entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from toc import render_toc
from utils import find_token_file, load_tokens

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

SOURCES_META = cfg["sources"]
LANG_LABELS  = cfg["languages"]
NMF_MAX_ITER = cfg["model"]["nmf_max_iter"]

st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Running NMF…")
def run_nmf(src_id, token_path, n_chunks, min_df, max_df, k, ngram_range=(1, 1)):
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
    words = vec.get_feature_names_out()
    n_top = cfg["controls"]["n_top_words"]["default"]
    PHI = [
        [words[j] for j in model.components_[i].argsort()[::-1][:n_top]]
        for i in range(k)
    ]
    return {"THETA": THETA, "PHI": PHI, "n_chunks": len(chunks_list)}


def build_fsm_figure(K, T_prob, PHI, colors, labels, min_prob=0.1):
    """Directed graph: nodes ordered by first appearance along x-axis.
    Forward edges arc above; backward edges arc below."""
    # First-appearance order
    rank = {}
    for lbl in labels:
        if lbl not in rank:
            rank[lbl] = len(rank)
    for ti in range(K):
        if ti not in rank:
            rank[ti] = len(rank)

    spacing = 2.0
    node_x  = np.array([rank[ti] * spacing for ti in range(K)], dtype=float)
    node_y  = np.zeros(K, dtype=float)
    node_r  = 0.22  # arrow-tip offset from node centre

    fig = go.Figure()

    for i in range(K):
        for j in range(K):
            p = float(T_prob[i, j])
            if p < min_prob:
                continue
            color = colors[i]
            lw    = 1.0 + p * 5.0

            if i == j:
                # Self-loop: small circle above the node
                loop_r = 0.30
                cy     = node_y[i] + loop_r + 0.10
                t_loop = np.linspace(0, 2 * np.pi, 60)
                fig.add_trace(go.Scatter(
                    x=node_x[i] + loop_r * np.cos(t_loop),
                    y=cy        + loop_r * np.sin(t_loop),
                    mode="lines",
                    line=dict(color=color, width=lw),
                    showlegend=False, hoverinfo="skip",
                ))
                fig.add_annotation(
                    x=node_x[i], y=cy + loop_r + 0.10,
                    text=f"{p:.2f}", showarrow=False,
                    font=dict(size=9, color=color),
                    bgcolor="white", opacity=0.85,
                )
            else:
                x1, y1 = node_x[i], node_y[i]
                x2, y2 = node_x[j], node_y[j]
                span   = abs(rank[j] - rank[i])
                # Arc height scales with span; forward above, backward below
                h = 0.5 + span * 0.20
                if rank[j] > rank[i]:
                    cpy = h       # forward: above axis
                else:
                    cpy = -h      # backward: below axis
                cpx = (x1 + x2) / 2

                t  = np.linspace(0, 1, 60)
                bx = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * cpx + t ** 2 * x2
                by = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cpy + t ** 2 * y2

                ddx = x2 - bx[-10]
                ddy = y2 - by[-10]
                dd  = np.sqrt(ddx ** 2 + ddy ** 2) + 1e-9
                tip_x = x2 - node_r * ddx / dd
                tip_y = y2 - node_r * ddy / dd

                fig.add_trace(go.Scatter(
                    x=bx, y=by, mode="lines",
                    line=dict(color=color, width=lw),
                    showlegend=False, hoverinfo="skip",
                ))
                fig.add_annotation(
                    x=tip_x, y=tip_y,
                    ax=bx[-12], ay=by[-12],
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=2, arrowsize=1.5,
                    arrowwidth=max(1.0, lw * 0.7),
                    arrowcolor=color,
                )
                fig.add_annotation(
                    x=float(cpx), y=float(cpy) + (0.08 if cpy >= 0 else -0.14),
                    text=f"{p:.2f}", showarrow=False,
                    font=dict(size=9, color=color),
                    bgcolor="white", opacity=0.85,
                )

    # Nodes — sorted by rank for correct x placement
    ordered = sorted(range(K), key=lambda ti: rank[ti])
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=44, color=colors[:K], line=dict(color="white", width=2)),
        text=[f"T{i}" for i in range(K)],
        textposition="middle center",
        textfont=dict(color="white", size=12),
        showlegend=False,
        hovertext=[f"T{i}: {', '.join(PHI[i][:5])}" for i in range(K)],
        hoverinfo="text",
    ))

    x_lo = -1.0
    x_hi = (K - 1) * spacing + 1.0
    max_h = 0.5 + (K - 1) * 0.20 + 0.70  # tallest possible arc + self-loop
    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white",
        xaxis=dict(visible=False, range=[x_lo, x_hi]),
        yaxis=dict(visible=False, range=[-max_h, max_h]),
    )
    return fig


# ── Controls ──────────────────────────────────────────────────────────────────
st.title("Narrative Transitions")
render_toc([
    ("Topic Sequence",      "topic-sequence"),
    ("Transition Matrix",   "transition-matrix"),
    ("FSM Graph",           "fsm-graph"),
    ("Topic Summary",       "topic-summary"),
])
st.caption(
    "Divides each edition into `n_chunks` equal bins, fits NMF, and assigns each chunk "
    "its dominant topic (argmax of THETA). The transition matrix counts how often "
    "topic *i* is followed by topic *j* in the narrative sequence."
)

_c          = cfg["controls"]
_col_ratios = cfg["layout"]["column_ratios"]
src_ids     = list(SOURCES_META.keys())
cols        = st.columns(_col_ratios[:5])

src_id = cols[0].selectbox(
    "Source", src_ids, index=src_ids.index(_c["default_source"]),
    format_func=lambda x: f"{SOURCES_META[x]['label']} ({LANG_LABELS[SOURCES_META[x]['lang']]})")

token_path = find_token_file(src_id)
if token_path is None:
    st.error(f"No TOKEN file found for '{src_id}'.")
    st.stop()

n_chunks  = cols[1].number_input("n_chunks", min_value=_c["n_chunks"]["min"],  max_value=_c["n_chunks"]["max"],
                                  value=_c["n_chunks"]["default"], step=_c["n_chunks"]["step"])
min_df    = cols[2].number_input("min_df",   _c["min_df"]["min"], _c["min_df"]["max"],
                                  _c["min_df"]["default"], step=_c["min_df"]["step"])
max_df    = cols[2].number_input("max_df",   _c["max_df"]["min"], _c["max_df"]["max"],
                                  _c["max_df"]["default"], step=_c["max_df"]["step"], format="%.2f")
_ng       = _c["ngram_range"]
ngram_min = cols[3].number_input("ngram min", _ng["min_n"], _ng["max_n"], _ng["default_min"], step=1)
ngram_max = cols[3].number_input("ngram max", _ng["min_n"], _ng["max_n"], _ng["default_max"], step=1)
ngram_max = max(ngram_max, ngram_min)
k         = cols[4].number_input("k (topics)", _c["n_topics"]["min"], _c["n_topics"]["max"],
                                  _c["n_topics"]["default"], step=1)

result = run_nmf(src_id, token_path, int(n_chunks), int(min_df), float(max_df),
                 int(k), (int(ngram_min), int(ngram_max)))

if result is None:
    st.warning("NMF could not run — try adjusting n_chunks, min_df, max_df, or k.")
    st.stop()

THETA    = result["THETA"]
PHI      = result["PHI"]
n_actual = result["n_chunks"]
labels   = np.argmax(THETA, axis=1)
K        = int(k)

_topic_labels = [f"T{i}: {', '.join(PHI[i][:4])}" for i in range(K)]
_colors       = px.colors.qualitative.Plotly[:K]

st.divider()

# ── Topic Sequence ────────────────────────────────────────────────────────────
st.subheader("Topic Sequence", anchor="topic-sequence")
st.caption(
    "Each bar is one chunk; color = dominant topic. "
    "Read left to right as narrative progression."
)

_positions = list(range(n_actual))
fig_seq = go.Figure()
for ti in range(K):
    mask = labels == ti
    fig_seq.add_trace(go.Bar(
        x=[i for i, m in enumerate(mask) if m],
        y=[1] * int(mask.sum()),
        name=_topic_labels[ti],
        marker_color=_colors[ti],
        width=1.0,
        hovertemplate=f"{_topic_labels[ti]}<br>chunk %{{x}}<extra></extra>",
    ))
fig_seq.update_layout(
    barmode="stack",
    height=120,
    margin=dict(l=60, r=20, t=10, b=40),
    plot_bgcolor="white",
    showlegend=True,
    legend=dict(orientation="h", y=-0.5, x=0),
    xaxis=dict(title="Chunk position", showgrid=False, zeroline=False),
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    bargap=0,
)
st.plotly_chart(fig_seq, width="stretch")

# THETA heatmap
_v       = cfg["visualization"]
_row_h   = cfg["layout"]["heatmap_row_height_px"]
_x_pos   = np.round(np.linspace(1, 100, n_actual)).astype(int)

fig_theta = px.imshow(
    THETA.T,
    y=_topic_labels,
    x=_x_pos,
    aspect="auto",
    color_continuous_scale=_v["heatmap_color_scale"],
    labels=dict(x="Position (1–100)", y="Topic"),
)
fig_theta.update_layout(
    height=K * _row_h + 60,
    margin=dict(l=60, r=20, t=10, b=40),
    coloraxis_showscale=False,
)
st.plotly_chart(fig_theta, width="stretch")

st.divider()

# ── Transition Matrix ─────────────────────────────────────────────────────────
st.subheader("Transition Matrix", anchor="transition-matrix")
st.caption(
    "**Counts** (left): how many times topic *row* is followed by topic *column*. "
    "**Probabilities** (right): row-normalized — given topic *row*, how likely is topic *column* next? "
    "Diagonal = self-transition (theme persistence)."
)

T_count = np.zeros((K, K), dtype=int)
for i in range(len(labels) - 1):
    T_count[labels[i], labels[i + 1]] += 1

row_sums  = T_count.sum(axis=1, keepdims=True)
T_prob    = np.where(row_sums > 0, T_count / row_sums.astype(float), 0.0)

_short_labels = [f"T{i}" for i in range(K)]
_hover_labels = [f"T{i}: {', '.join(PHI[i][:3])}" for i in range(K)]

count_df = pd.DataFrame(T_count, index=_hover_labels, columns=_hover_labels)
prob_df  = pd.DataFrame(T_prob,  index=_hover_labels, columns=_hover_labels)

_mat_h = max(300, K * 50 + 100)
_mat_margin = dict(l=20, r=20, t=40, b=20)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Transition counts**")
    fig_cnt = px.imshow(
        count_df, text_auto=True, aspect="auto",
        color_continuous_scale=_v["colorscale_ari"],
        zmin=0,
    )
    fig_cnt.update_layout(height=_mat_h, margin=_mat_margin,
                          coloraxis_colorbar=dict(title="Count"))
    fig_cnt.update_traces(textfont_size=12)
    st.plotly_chart(fig_cnt, width="stretch")

with col_right:
    st.markdown("**Transition probabilities**")
    fig_prob = px.imshow(
        prob_df.round(2), text_auto=".2f", aspect="auto",
        color_continuous_scale=_v["colorscale_transitions"],
        zmin=0, zmax=1,
    )
    fig_prob.update_layout(height=_mat_h, margin=_mat_margin,
                           coloraxis_colorbar=dict(title="P"))
    fig_prob.update_traces(textfont_size=12)
    st.plotly_chart(fig_prob, width="stretch")

st.divider()

# ── FSM Graph ─────────────────────────────────────────────────────────────────
st.subheader("FSM Graph", anchor="fsm-graph")
st.caption(
    "Nodes = topics; directed edges = transition probabilities. "
    "Edge width and label show probability. "
    "Self-loops (theme persistence) appear as circles above each node. "
    "Adjust the threshold to hide weak transitions."
)

_min_prob = st.slider("Minimum probability to show", 0.0, 0.5, 0.1, 0.05, format="%.2f")
fig_fsm = build_fsm_figure(K, T_prob, PHI, _colors, labels, min_prob=_min_prob)
st.plotly_chart(fig_fsm, width="stretch")

st.divider()

# ── Topic Summary ─────────────────────────────────────────────────────────────
st.subheader("Topic Summary", anchor="topic-summary")
st.caption(
    "**Self-transition**: probability the theme persists to the next chunk. "
    "**Entropy**: uncertainty of outgoing transitions (higher = less predictable)."
)

summary_rows = []
for ti in range(K):
    dominant_count = int((labels == ti).sum())
    self_p         = float(T_prob[ti, ti])
    row_p          = T_prob[ti]
    ent            = float(scipy_entropy(row_p + 1e-12))
    summary_rows.append({
        "Topic":            f"T{ti}",
        "Top words":        ", ".join(PHI[ti]),
        "Dominant chunks":  dominant_count,
        "Self-transition":  round(self_p, 3),
        "Entropy":          round(ent, 3),
    })

st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
