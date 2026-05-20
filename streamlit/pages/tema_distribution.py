"""
Tema Distribution — proper nouns and named entities over narrative time.

Source: <rs ana="..."> annotations in the Ximenez XML manuscript, exported as
  notebooks/ximenez/ximenez-quc-TEMA_TOKEN.csv  (one row per annotation)
  notebooks/ximenez/ximenez-quc-TEMA.csv        (per-tema statistics)

TEMA_TOKEN is indexed by paragraph (para_num); TOKEN.csv carries both para_num
and token order, so chunk_num is derived by re-cutting tokens at the chosen
n_chunks and mapping each paragraph to the chunk containing its first token.
"""

import os
import re
import textwrap
import html as html_mod
import xml.etree.ElementTree as ET
import yaml
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, fcluster
from toc import render_toc
from utils import find_token_file, load_tokens, load_chunk_doc_texts

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

TEMA_TOKEN_PATH = os.path.normpath(
    os.path.join(APP_DIR, "../../notebooks/ximenez/ximenez-quc-TEMA_TOKEN.csv")
)
TEMA_PATH = os.path.normpath(
    os.path.join(APP_DIR, "../../notebooks/ximenez/ximenez-quc-TEMA.csv")
)
TOKEN_PATH = os.path.normpath(
    os.path.join(APP_DIR, "../../notebooks/ximenez/ximenez-TOKEN.csv")
)
DOCMAP_PATH = os.path.normpath(
    os.path.join(APP_DIR, "../../notebooks/ximenez/ximenez-DOCMAP.csv")
)
SURFACE_FORMS_PATH = os.path.normpath(
    os.path.join(APP_DIR, "../../notebooks/ximenez/ximenez-SURFACE_FORMS.csv")
)
LINE_PATH = os.path.normpath(
    os.path.join(APP_DIR, "../../notebooks/ximenez/ximenez-quc-LINE.csv")
)
CXIM_LINE_PATH = os.path.normpath(
    os.path.join(APP_DIR, "../../notebooks/christenson_ximenez/christenson_ximenez-LINE.csv")
)
COVERAGE_PATH = os.path.normpath(
    os.path.join(APP_DIR, "../../notebooks/ximenez/ximenez-COVERAGE.csv")
)
PHI_PATH = os.path.normpath(
    os.path.join(APP_DIR, "../../notebooks/ximenez/ximenez-PHI.csv")
)
THETA_PATH = os.path.normpath(
    os.path.join(APP_DIR, "../../notebooks/ximenez/ximenez-THETA.csv")
)
TOPIC_PATH = os.path.normpath(
    os.path.join(APP_DIR, "../../notebooks/ximenez/ximenez-TOPIC.csv")
)
PHI_N_CHUNKS = 60  # chunk count used when PHI/THETA were computed
TOPICS_XML_PATH = os.path.normpath(
    os.path.join(APP_DIR, "../../temas/topics.xml")
)
COLOP_GLOSSARY_PATH = os.path.normpath(
    os.path.join(APP_DIR, "../../textos/colop-glossary.txt")
)

st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_topic_dict() -> dict:
    """Parse topics.xml → {ana_id_key: {title, type, english_name, desc_html, desc_short}}."""
    tree = ET.parse(TOPICS_XML_PATH)
    out  = {}
    for t in tree.getroot().findall("topic"):
        key = (t.findtext("key") or "").strip()
        if not key:
            continue

        title  = (t.findtext("title") or "").strip()
        ttype  = (t.findtext("type")  or "").strip()

        # english_name is HTML-encoded text (<ul><li>…</li></ul> stored as entities)
        eng_el = t.find("english_name")
        eng    = (eng_el.text or "") if eng_el is not None else ""
        eng    = re.sub(r"<[^>]+>", "", html_mod.unescape(eng))
        eng    = re.sub(r"\s+", " ", eng).strip()

        # description is a tree of <p> / <em> / <a> child elements
        desc_el = t.find("description")
        if desc_el is not None:
            desc_html  = ET.tostring(desc_el, encoding="unicode")
            # Strip outer <description> tag to get just the inner HTML
            desc_html  = re.sub(r"^<description>|</description>$", "", desc_html).strip()
            desc_plain = " ".join(desc_el.itertext())
            desc_plain = re.sub(r"\s+", " ", desc_plain).strip()
        else:
            desc_html = desc_plain = ""

        first_sent = re.split(r"(?<=[.!?])\s+", desc_plain)[0] if desc_plain else ""
        if len(first_sent) > 220:
            first_sent = first_sent[:220] + "…"

        out[key] = {
            "title":       title,
            "type":        ttype,
            "english":     eng,
            "desc_html":   desc_html,
            "desc_short":  first_sent,
        }
    return out


@st.cache_data(show_spinner=False)
def load_colop_glossary() -> dict[str, str]:
    """Load colop-glossary.txt → {normalized_key: spanish_gloss}."""
    df = pd.read_csv(COLOP_GLOSSARY_PATH, sep="|", dtype=str).fillna("")
    result = {}
    for _, row in df.iterrows():
        key = row["term"].strip().upper().replace(" ", "_")
        result[key] = row["gloss"].strip()
    return result


def _lookup_colop_gloss(ana_id: str, glossary: dict[str, str]) -> str:
    """Return Colop Spanish gloss for ana_id, or empty string. Falls back to apostrophe-stripped match."""
    if ana_id in glossary:
        return glossary[ana_id]
    stripped = ana_id.replace("'", "")
    for k, v in glossary.items():
        if k.replace("'", "") == stripped:
            return v
    return ""


@st.cache_data(show_spinner=False)
def load_coverage_data():
    sf = pd.read_csv(SURFACE_FORMS_PATH)
    cv = pd.read_csv(COVERAGE_PATH)
    return sf, cv


@st.cache_data(show_spinner=False)
def _load_annotation_lines(n_chunks: int) -> pd.DataFrame:
    """One row per annotation with line text and chunk_num. Shared base for hover + click panel."""
    LINE = pd.read_csv(LINE_PATH, sep="|")
    TT   = pd.read_csv(TEMA_TOKEN_PATH, sep="|")
    joined = TT.merge(
        LINE[["folio", "side", "para_num", "lb", "lb_str_plain"]],
        on=["folio", "side", "para_num", "lb"], how="left",
    )

    # Join Christenson normalized K'iche' line text via folio/side/lb alignment.
    # Christenson uses 'recto'/'verso'; Ximenez uses 1/2 — map before joining.
    CXIM = pd.read_csv(CXIM_LINE_PATH)
    CXIM["side"] = CXIM["side"].map({"recto": 1, "verso": 2})
    joined = joined.merge(
        CXIM[["folio_num", "side", "lb_num", "line_str"]].rename(
            columns={"folio_num": "folio", "lb_num": "lb", "line_str": "cxim_str"}
        ),
        on=["folio", "side", "lb"], how="left",
    )

    TOKEN = pd.read_csv(TOKEN_PATH).merge(
        pd.read_csv(DOCMAP_PATH)[["doc_id", "para_num"]], on="doc_id", how="left"
    )
    TOKEN["chunk_num"] = pd.cut(TOKEN.index, n_chunks, labels=range(n_chunks)).astype(int)
    para_to_chunk = TOKEN.groupby("para_num")["chunk_num"].first()
    joined = joined.join(para_to_chunk, on="para_num", how="left").dropna(subset=["chunk_num"])
    joined["chunk_num"] = joined["chunk_num"].astype(int)
    return joined


@st.cache_data(show_spinner="Building concordance…")
def load_concordance(n_chunks: int) -> dict:
    """Return dict (ana_id, chunk_num) → HTML hover string for the heatmap."""
    joined = _load_annotation_lines(n_chunks)
    eng_texts = load_chunk_doc_texts("christenson_english_prose", n_chunks)

    conc = {}
    for (ana_id, chunk_num), grp in joined.groupby(["ana_id", "chunk_num"]):
        entries = []
        for _, row in grp.iterrows():
            side_str = "r" if int(row["side"]) == 1 else "v"
            ref  = f"f.{int(row['folio'])}{side_str} lb.{int(row['lb'])}"
            norm = str(row.get("cxim_str", "")).strip()
            raw  = str(row["lb_str_plain"]).strip()
            line = norm if norm and norm != "nan" else raw
            if len(line) > 72:
                line = line[:72] + "…"
            entries.append(f"<i>{ref}</i>  {line}")
        n = len(entries)
        header = f"<b>{ana_id}</b> · {n} line{'s' if n != 1 else ''} in chunk {chunk_num}<br>"
        body = "<br>".join(entries[:10])
        if n > 10:
            body += f"<br><i>… +{n - 10} more</i>"

        eng = eng_texts[chunk_num] if chunk_num < len(eng_texts) else ""
        if eng:
            flat = eng.replace("\n\n", " ¶ ")
            truncated = flat[:200] + ("…" if len(flat) > 200 else "")
            wrapped = "<br>".join(textwrap.wrap(truncated, width=55))
            gloss = "<br>─────────────────────────────<br><i>" + wrapped + "</i>"
        else:
            gloss = ""

        conc[(ana_id, chunk_num)] = header + body + gloss
    return conc


METRIC_LABELS = {
    "df":            "Document frequency — number of chunks a tema appears in",
    "n":             "Total count — raw number of annotations",
    "dh":            "Distributional entropy — evenness of spread across chunks",
    "range":         "Range — last chunk minus first chunk of appearance",
    "concentration": "Concentration — how episodically focused (inverse of spread)",
}

METRIC_HELP = {
    "df":            "How many distinct chunks contain at least one annotation of this tema. "
                     "High df = present across many episodes.",
    "n":             "Total number of times this tema is annotated in the K'iche' text. "
                     "High n = narratively prominent.",
    "dh":            "Shannon entropy of the tema's distribution across chunks (in bits). "
                     "High dh = evenly spread; low dh = concentrated in one episode.",
    "range":         "Chunk index of last appearance minus chunk index of first appearance. "
                     "High range = active across a large span of the narrative.",
    "concentration": "1 − dh/log₂(n_chunks). High concentration = episodically specific; "
                     "low concentration = ubiquitous across the whole narrative.",
}


@st.cache_data(show_spinner="Loading tema annotations…")
def load_tema_data(n_chunks: int):
    TOKEN     = pd.read_csv(TOKEN_PATH).merge(
        pd.read_csv(DOCMAP_PATH)[["doc_id", "para_num"]], on="doc_id", how="left"
    )
    TEMA_TOK  = pd.read_csv(TEMA_TOKEN_PATH, sep="|")
    TEMA_META = pd.read_csv(TEMA_PATH, sep="|")

    # Re-cut tokens at the chosen n_chunks to get chunk_num per token row
    TOKEN["chunk_num"] = pd.cut(
        TOKEN.index, n_chunks, labels=list(range(n_chunks))
    ).astype(int)

    # Map each para_num to the chunk_num of its first token
    para_to_chunk = (
        TOKEN.groupby("para_num")["chunk_num"].first().rename("chunk_num")
    )

    # Join annotations to chunk_num via para_num
    merged = TEMA_TOK.join(para_to_chunk, on="para_num", how="left")
    merged = merged.dropna(subset=["chunk_num"])
    merged["chunk_num"] = merged["chunk_num"].astype(int)

    # Build CAM: Chunk × Tema count matrix
    CAM = (
        merged.groupby(["chunk_num", "ana_id"])
        .size()
        .unstack(fill_value=0)
    )
    # Ensure all chunks represented
    CAM = CAM.reindex(range(n_chunks), fill_value=0)

    # Compute per-tema metrics
    df_metric = (CAM > 0).sum().rename("df")
    n_metric  = CAM.sum().rename("n")

    # dh from TEMA_META (computed at n_chunks=60 in notebook; recompute here)
    P  = CAM / CAM.sum().replace(0, 1)
    DI = np.log2(1 / P.replace(0, np.nan))
    dh_metric = (P * DI).sum().rename("dh")

    rng_metric = (CAM > 0).apply(
        lambda col: int(col[col > 0].index.max() - col[col > 0].index.min())
        if col.sum() > 0 else 0
    ).rename("range")

    max_dh = np.log2(max(n_chunks, 2))
    conc_metric = (1 - dh_metric / max_dh).rename("concentration")

    # Human-readable labels from TEMA_META
    label_map = TEMA_META.set_index("ana_id")["label"].to_dict()

    METRICS = pd.concat([df_metric, n_metric, dh_metric, rng_metric, conc_metric], axis=1)
    METRICS["label"] = METRICS.index.map(lambda x: label_map.get(x, x.replace("_", " ").title()))

    return merged, CAM, METRICS


# ── Hover text helpers ────────────────────────────────────────────────────────

def _hw(text: str, width: int = 60) -> str:
    """Word-wrap plain text for Plotly hover, joining lines with <br>."""
    if not text:
        return ""
    return "<br>".join(textwrap.wrap(text, width=width))


# ── Boundary helpers ──────────────────────────────────────────────────────────

def _chunk_texts(token_path, n_chunks):
    TOKEN = pd.read_csv(token_path)
    TOKEN["chunk_num"] = pd.cut(TOKEN.index, n_chunks, labels=range(n_chunks)).astype(int)
    return (
        TOKEN.groupby("chunk_num")["term_str"]
        .apply(lambda x: " ".join(x.dropna()))
        .reindex(range(n_chunks), fill_value="")
    )


@st.cache_data(show_spinner=False)
def _resonance_boundaries(token_path, n_chunks, min_df, max_df, window, pct):
    chunks = _chunk_texts(token_path, n_chunks)
    X = TfidfVectorizer(norm=None, min_df=min_df, max_df=max_df).fit_transform(chunks).toarray().astype(float)
    X += 1e-9
    P = X / X.sum(axis=1, keepdims=True)
    n = len(P)
    nov, tra = np.zeros(n), np.zeros(n)
    for j in range(n):
        back = [float(np.sum(P[j] * np.log2(P[j] / P[j-d]))) for d in range(1, window+1) if j-d >= 0]
        fwd  = [float(np.sum(P[j] * np.log2(P[j] / P[j+d]))) for d in range(1, window+1) if j+d < n]
        if back: nov[j] = np.mean(back)
        if fwd:  tra[j] = np.mean(fwd)
    res = nov - tra
    thresh = float(np.nanpercentile(res, pct))
    return [j for j in range(n) if res[j] >= thresh]


@st.cache_data(show_spinner=False)
def _hac_boundaries(token_path, n_chunks, min_df, max_df, n_clusters):
    chunks = _chunk_texts(token_path, n_chunks)
    X = TfidfVectorizer(norm="l2", min_df=min_df, max_df=max_df).fit_transform(chunks).toarray()
    labels = fcluster(linkage(X, method="ward"), n_clusters, criterion="maxclust")
    bounds = [j for j in range(1, n_chunks) if labels[j] != labels[j - 1]]
    return bounds, labels


@st.cache_data(show_spinner=False)
def _phi_topic_assignments():
    """Assign each ana_id to its dominant NMF topic via PHI vocabulary weights.

    For each tema, take the surface forms that appear in the PHI vocabulary,
    then find the topic with the highest weight across those forms.
    Returns (assignments dict, topic_labels dict).
    """
    phi  = pd.read_csv(PHI_PATH, index_col=0)          # (n_topics, vocab)
    sf   = pd.read_csv(SURFACE_FORMS_PATH)
    meta = pd.read_csv(TOPIC_PATH)

    vocab = set(phi.columns)
    assignments = {}
    for ana_id, grp in sf.groupby("ana_id"):
        forms = [f for f in grp["surface_form"].str.strip() if f in vocab]
        if not forms:
            assignments[ana_id] = -1
            continue
        weights = phi[forms].max(axis=1)   # max across matching forms per topic
        assignments[ana_id] = int(weights.idxmax())

    topic_labels = dict(zip(meta["topic_id"].astype(int), meta["label"]))
    return assignments, topic_labels


@st.cache_data(show_spinner=False)
def _hac_tema_assignments(token_path, tema_token_path, min_df, max_df, n_clusters):
    """Assign each ana_id to its dominant HAC cluster (annotation-weighted).

    HAC is run at PHI_N_CHUNKS (the same chunking used for PHI/THETA) so that
    vertical and horizontal groupings are computed on a consistent representation.
    Temas with no annotations at that granularity get label -1.
    """
    chunks = _chunk_texts(token_path, PHI_N_CHUNKS)
    X = TfidfVectorizer(norm="l2", min_df=min_df, max_df=max_df).fit_transform(chunks).toarray()
    chunk_labels = fcluster(linkage(X, method="ward"), n_clusters, criterion="maxclust")

    TOKEN = pd.read_csv(token_path).merge(
        pd.read_csv(DOCMAP_PATH)[["doc_id", "para_num"]], on="doc_id", how="left"
    )
    TOKEN["chunk_num"] = pd.cut(TOKEN.index, PHI_N_CHUNKS,
                                labels=range(PHI_N_CHUNKS)).astype(int)
    para_to_chunk = TOKEN.groupby("para_num")["chunk_num"].first()

    TEMA_TOK = pd.read_csv(tema_token_path, sep="|")
    merged = TEMA_TOK.join(para_to_chunk, on="para_num", how="left").dropna(subset=["chunk_num"])
    merged["chunk_num"] = merged["chunk_num"].astype(int)
    merged["hac_label"] = merged["chunk_num"].map(lambda c: int(chunk_labels[c]))

    assignments = {}
    for ana_id, grp in merged.groupby("ana_id"):
        assignments[ana_id] = int(grp["hac_label"].mode().iloc[0])
    return assignments, chunk_labels


# ── Controls ──────────────────────────────────────────────────────────────────
st.title("Tema Distribution — Popol Wuj (Ximenez)")
st.caption(
    "Named entities and themes (*temas*) from `<rs ana=\"...\">` annotations in "
    "the Ximenez manuscript (K'iche' column). Each point = one annotation. "
    "The Ximenez source in the app is derived from the same XML."
)
with st.expander("Role in the analysis", expanded=False):
    st.markdown("""
The `<rs>` annotations are **interpretive, not exhaustive**. Coverage analysis (see *Annotation
Coverage* page) shows that many named entities appear far more often in the raw text than they
are annotated — the markup records scholarly judgment about *when a character is narratively
active*, not a complete catalogue of every mention.

This makes the tema distribution unsuitable as a primary segmentation tool, but valuable as
**partial external validation** for the unsupervised models on the *Episode Boundaries* and
*Clustering* pages. Those models are computed from raw word frequencies with no knowledge of
the annotations. Where their segment boundaries coincide with shifts in named-entity distribution,
that convergence is evidence the models are recovering genuine narrative structure rather than
translation or editorial artifacts. Where they diverge, the discrepancy is itself analytically
interesting — pointing to passages where lexical register changes without a corresponding change
in which characters are foregrounded, or vice versa.
""")

_c = cfg["controls"]
_col_ratios = cfg["layout"]["column_ratios"]
cols = st.columns(_col_ratios[:4])

n_chunks = cols[0].number_input(
    "Chunks", min_value=_c["n_chunks"]["min"], max_value=_c["n_chunks"]["max"],
    value=_c["n_chunks"]["default"], step=_c["n_chunks"]["step"]
)
metric = cols[1].selectbox(
    "Filter metric", list(METRIC_LABELS.keys()),
    format_func=lambda k: k,
    help="\n\n".join(f"**{k}** — {v}" for k, v in METRIC_HELP.items())
)

merged, CAM, METRICS = load_tema_data(int(n_chunks))

_mvals = METRICS[metric]
_mmin, _mmax = float(_mvals.min()), float(_mvals.max())

if metric in ("df", "n", "range"):
    thresh = cols[2].slider(
        f"Min {metric}", min_value=int(_mmin), max_value=int(_mmax),
        value=max(int(_mmin) + 1, int(_mmax * 0.3)), step=1
    )
    selected_temas = METRICS[METRICS[metric] >= thresh].index
else:
    thresh = cols[2].slider(
        f"Min {metric}", min_value=round(_mmin, 2), max_value=round(_mmax, 2),
        value=round(_mmin + (_mmax - _mmin) * 0.5, 2), step=round((_mmax - _mmin) / 20, 2)
    )
    selected_temas = METRICS[METRICS[metric] >= thresh].index

sort_by = cols[3].selectbox("Sort temas by", ["first appearance", "df", "n", "dh", "label"])

n_selected = len(selected_temas)
st.caption(
    f"**{n_selected}** of {len(METRICS)} temas selected · "
    f"{metric} ≥ {thresh} · {int(n_chunks)} chunks"
)

if n_selected == 0:
    st.warning("No temas match this threshold — lower the filter value.")
    st.stop()

# Sort
_m = METRICS.loc[selected_temas].copy()
if sort_by == "first appearance":
    _first = merged[merged.ana_id.isin(selected_temas)].groupby("ana_id")["chunk_num"].min()
    _m["_first"] = _first
    _m = _m.sort_values("_first")
elif sort_by == "label":
    _m = _m.sort_values("label")
else:
    _m = _m.sort_values(sort_by, ascending=False)

ordered_temas = _m.index.tolist()
ordered_labels = _m["label"].tolist()
label_map = dict(zip(ordered_temas, ordered_labels))

_tab_strip, _tab_heat, _tab_stats, _tab_cov = st.tabs([
    "Strip Plot", "Presence Heatmap", "Statistics", "Coverage"
])

# ── Tab 1: Strip plot ─────────────────────────────────────────────────────────
with _tab_strip:
    st.caption(
        "Each dot = one annotation in the K'iche' text. "
        "Y-axis = tema; X-axis = chunk position in narrative order. "
        "Vertical jitter separates overlapping points within each row."
    )

    plot_data = merged[merged.ana_id.isin(selected_temas)].copy()

    _rng = np.random.default_rng(42)   # fixed seed so jitter is stable across rerenders
    _palette = px.colors.qualitative.Dark24
    _row_h = max(22, 600 // max(n_selected, 1))

    fig_strip = go.Figure()

    # Alternating row bands
    for i in range(n_selected):
        if i % 2 == 0:
            fig_strip.add_shape(
                type="rect",
                x0=-0.5, x1=n_chunks - 0.5,
                y0=i - 0.5, y1=i + 0.5,
                fillcolor="rgba(0,0,0,0.04)",
                line_width=0,
                layer="below",
            )

    for i, (tid, tlabel) in enumerate(zip(ordered_temas, ordered_labels)):
        pts = plot_data[plot_data.ana_id == tid]
        if pts.empty:
            continue
        n_pts = len(pts)
        jitter = _rng.uniform(-0.38, 0.38, n_pts)
        fig_strip.add_trace(go.Scatter(
            x=pts["chunk_num"].values,
            y=np.full(n_pts, i) + jitter,
            mode="markers",
            marker=dict(
                size=8,
                color=_palette[i % len(_palette)],
                opacity=0.65,
                line=dict(width=0.5, color="white"),
            ),
            name=tlabel,
            showlegend=False,
            hovertemplate=f"<b>{tlabel}</b><br>chunk %{{x}}<extra></extra>",
        ))

    fig_strip.update_layout(
        height=max(350, n_selected * _row_h + 80),
        margin=dict(l=200, r=20, t=10, b=60),
        plot_bgcolor="white",
        xaxis=dict(
            title="Chunk (narrative order)",
            showgrid=True, gridcolor="#E0E0E0", zeroline=False,
            range=[-0.5, n_chunks - 0.5],
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(n_selected)),
            ticktext=ordered_labels,
            showgrid=False, zeroline=False,
            range=[n_selected - 0.5, -0.5],
        ),
    )
    st.plotly_chart(fig_strip, width="stretch")

# ── Tab 2: Presence heatmap ───────────────────────────────────────────────────
with _tab_heat:
    st.caption(
        "Rows = temas, columns = chunks. Color = annotation count. "
        "Horizontal cuts re-sort temas by their assigned topic/cluster and draw group boundaries."
    )

    # ── Controls ─────────────────────────────────────────────────────────────
    _hc1 = st.columns([1, 1, 1])
    _n_clust    = _hc1[0].number_input("k clusters", 2, 20, 6, 1, key="h_k")
    _min_df_h   = _hc1[1].number_input("min_df",     1, 20,  3, 1, key="h_min_df")
    _max_df_h   = _hc1[2].number_input("max_df %",  10, 100, 50, 5, key="h_max_df") / 100.0
    _show_hac   = "HAC Clusters"

    _vert_bounds  = []
    _horiz_temas  = ordered_temas
    _horiz_labels = ordered_labels
    _horiz_bounds = []
    _horiz_legend = ""

    if _show_hac == "HAC Clusters":
        _vert_bounds, _ = _hac_boundaries(
            TOKEN_PATH, int(n_chunks), int(_min_df_h), _max_df_h, int(_n_clust)
        )
        _hac_assign, _tema_chunk_labels = _hac_tema_assignments(
            TOKEN_PATH, TEMA_TOKEN_PATH, int(_min_df_h), _max_df_h, int(_n_clust)
        )
        _horiz_temas = sorted(
            ordered_temas,
            key=lambda t: (_hac_assign.get(t, 999), ordered_temas.index(t))
        )
        _horiz_labels = [label_map[t] for t in _horiz_temas]
        _horiz_bounds = [
            i + 0.5 for i in range(len(_horiz_temas) - 1)
            if _hac_assign.get(_horiz_temas[i], -1) != _hac_assign.get(_horiz_temas[i+1], -1)
        ]
        _used_clusters = sorted({_hac_assign.get(t, -1) for t in _horiz_temas if _hac_assign.get(t, -1) >= 0})
        # Use the same chunk_labels that produced the tema assignments for first-seen ordering
        _first_seen = {c: int(np.argmax(_tema_chunk_labels == c)) for c in _used_clusters}
        _clusters_by_appearance = sorted(_used_clusters, key=lambda c: _first_seen[c])
        _letter_map_global = {c: chr(65 + i) for i, c in enumerate(_clusters_by_appearance)}
        _horiz_legend = (
            f"k={_n_clust} · HAC clusters: "
            + " · ".join(f"**{_letter_map_global[c]}**" for c in _clusters_by_appearance)
        )

    # ── Build heatmap ─────────────────────────────────────────────────────────
    conc = load_concordance(int(n_chunks))

    _chunks = list(range(int(n_chunks)))
    topic_dict    = load_topic_dict()
    colop_glossary = load_colop_glossary()

    # Build z (count) and customdata (hover text) matrices — rows = temas (top→bottom)
    _z, _hover = [], []
    for tid, tlabel in zip(_horiz_temas, _horiz_labels):
        # Topic header: shown at the top of every non-empty cell's hover
        info = topic_dict.get(tid, {})
        if info:
            eng_str  = f" — {info['english']}" if info["english"] else ""
            t_header = (
                f"<b>{info['title']}{eng_str}</b>  <i>({info['type']})</i><br>"
                + (_hw(info["desc_short"]) + "<br>" if info["desc_short"] else "")
                + "─────────────────────────────<br>"
            )
        else:
            t_header = f"<b>{tlabel}</b><br>─────────────────────────────<br>"

        row_z, row_h = [], []
        for c in _chunks:
            count = int(CAM.loc[c, tid]) if (c in CAM.index and tid in CAM.columns) else 0
            row_z.append(count)
            if count > 0 and (tid, c) in conc:
                row_h.append(t_header + conc[(tid, c)])
            elif count > 0:
                row_h.append(t_header + f"{count} annotation{'s' if count != 1 else ''}")
            else:
                row_h.append(t_header + "<i>(not present in this chunk)</i>")
        _z.append(row_z)
        _hover.append(row_h)

    _colorscale = cfg["visualization"]["heatmap_color_scale"]
    fig_heat = go.Figure(go.Heatmap(
        z=_z,
        x=_chunks,
        y=_horiz_labels,
        customdata=_hover,
        hovertemplate="%{customdata}<extra></extra>",
        colorscale=_colorscale,
        showscale=False,
        hoverongaps=False,
        xgap=1,
        ygap=1,
        hoverlabel=dict(align="left", font=dict(family="monospace", size=12)),
    ))

    for _b in _vert_bounds:
        fig_heat.add_shape(type="line",
            x0=_b - 0.5, x1=_b - 0.5, y0=-0.5, y1=len(_horiz_labels) - 0.5,
            line=dict(color="crimson", width=1.5))

    for _h in _horiz_bounds:
        fig_heat.add_shape(type="line",
            x0=-0.5, x1=n_chunks - 0.5, y0=_h, y1=_h,
            line=dict(color="steelblue", width=1.5, dash="dash"))

    if _horiz_bounds:
        _band_starts = [-0.5] + list(_horiz_bounds)
        _band_ends   = list(_horiz_bounds) + [len(_horiz_temas) - 0.5]
        for _bs, _be in zip(_band_starts, _band_ends):
            _row_idx = int(_bs + 0.5)
            _cid     = _hac_assign.get(_horiz_temas[_row_idx], -1)
            fig_heat.add_annotation(
                x=1.01, y=(_bs + _be) / 2,
                xref="paper", yref="y",
                text=f"<b>{_letter_map_global.get(_cid, '?')}</b>",
                showarrow=False, xanchor="left",
                font=dict(size=13, color="steelblue"),
            )

    _row_px = max(14, 400 // max(n_selected, 1))
    fig_heat.update_layout(
        height=max(300, n_selected * _row_px + 80),
        margin=dict(l=200, r=50, t=20, b=60),
        xaxis=dict(title="Chunk (narrative order)", tickmode="linear"),
        yaxis=dict(
            autorange="reversed",
            categoryorder="array",
            categoryarray=_horiz_labels,
            tickmode="array",
            tickvals=list(range(len(_horiz_labels))),
            ticktext=_horiz_labels,
            tickfont=dict(size=max(8, min(12, 300 // max(n_selected, 1)))),
        ),
    )
    st.plotly_chart(fig_heat, width="stretch")

    _caps = []
    if _vert_bounds:
        _caps.append(f"**Vertical (crimson):** HAC Clusters — boundaries at chunks {sorted(_vert_bounds)}")
    if _horiz_legend:
        _caps.append(_horiz_legend)
    if _caps:
        st.caption(" · ".join(_caps))

    # ── Chunk text browser ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Chunk Text Browser", anchor=False)
    st.caption(
        "Read the source text for any chunk across editions. "
        "K'iche' (normalized) on the left; English prose on the right."
    )
    _chunk_sel = st.selectbox(
        "Chunk", range(int(n_chunks)),
        format_func=lambda c: f"Chunk {c}",
        key="chunk_browser_sel",
    )
    _cxim_texts = load_chunk_doc_texts("christenson_ximenez", int(n_chunks))
    _eng_texts  = load_chunk_doc_texts("christenson_english_prose", int(n_chunks))
    _chr_texts  = load_chunk_doc_texts("christenson", int(n_chunks))

    _br_left, _br_right = st.columns(2)
    with _br_left:
        st.caption("**Christenson's Ximénez** — normalized K'iche'")
        _cxim_chunk = _cxim_texts[_chunk_sel] if _chunk_sel < len(_cxim_texts) else ""
        st.text_area(
            "cxim_text", value=_cxim_chunk or "No text available.",
            height=300, disabled=True, label_visibility="collapsed",
        )
        with st.expander("Christenson 2007 — modern K'iche'", expanded=False):
            _chr_chunk = _chr_texts[_chunk_sel] if _chunk_sel < len(_chr_texts) else ""
            st.text_area(
                "chr_text", value=_chr_chunk or "No text available.",
                height=250, disabled=True, label_visibility="collapsed",
            )
    with _br_right:
        st.caption("**Christenson — English prose**")
        _eng_chunk = _eng_texts[_chunk_sel] if _chunk_sel < len(_eng_texts) else ""
        st.text_area(
            "eng_text", value=_eng_chunk or "No text available.",
            height=300, disabled=True, label_visibility="collapsed",
        )

    # ── Glossary panel (selectbox) ────────────────────────────────────────────
    st.divider()
    _label_to_id = dict(zip(_horiz_labels, _horiz_temas))
    _sel_label = st.selectbox(
        "Select a tema for its full description:",
        ["— select —"] + _horiz_labels, key="tema_desc_sel"
    )
    if _sel_label and _sel_label != "— select —":
        _sel_id = _label_to_id.get(_sel_label, "")
        _info   = topic_dict.get(_sel_id, {})
        if _info:
            eng_str = f" — *{_info['english']}*" if _info["english"] else ""
            st.markdown(
                f"#### {_info['title']}{eng_str} "
                f"<span style='font-size:0.8em;color:gray'>({_info['type']})</span>",
                unsafe_allow_html=True,
            )
            if _info["desc_html"]:
                st.markdown(_info["desc_html"], unsafe_allow_html=True)
        else:
            st.info(f"No entry for `{_sel_id}` in topics.xml.")
        _colop_gloss = _lookup_colop_gloss(_sel_id, colop_glossary)
        if _colop_gloss:
            with st.expander("Colop definition (Spanish)"):
                st.markdown(_colop_gloss)

# ── Tab 3: Tema statistics ────────────────────────────────────────────────────
with _tab_stats:
    st.caption("Metrics for the currently selected temas, sorted as above.")

    _display = _m[["label", "df", "n", "dh", "range", "concentration"]].copy()
    _display.index = _display["label"]
    _display = _display.drop(columns="label")
    _display.index.name = "Tema"
    _display["dh"]            = _display["dh"].round(3)
    _display["concentration"] = _display["concentration"].round(3)
    st.dataframe(_display, width="stretch")

# ── Tab 4: Coverage ───────────────────────────────────────────────────────────
with _tab_cov:
    st.caption(
        "Annotation completeness for the selected temas. "
        "`coverage_ratio` = n_annotated / n_found_exact (exact surface-form matches in the K'iche' text). "
        "Values < 1 indicate more occurrences in the text than annotations; > 1 indicates forms split "
        "across line breaks or otherwise unmatched."
    )

    SF, CV = load_coverage_data()

    # Filter coverage table to selected temas
    cv_sel = CV[CV["ana_id"].isin(selected_temas)].copy()
    cv_sel["label"] = cv_sel["ana_id"].map(label_map)
    cv_sel = cv_sel.set_index("label").drop(columns=["ana_id"])
    cv_sel.index.name = "Tema"
    cv_sel["coverage_ratio"] = cv_sel["coverage_ratio"].round(3)
    st.subheader("Coverage Metrics")
    st.dataframe(cv_sel, width="stretch")

    st.subheader("Surface Form Inventory")
    st.caption("All spelling variants found in `<rs>` elements for the selected temas.")
    sf_sel = SF[SF["ana_id"].isin(selected_temas)].copy()
    sf_sel["label"] = sf_sel["ana_id"].map(label_map)
    sf_sel = sf_sel[["label", "surface_form", "form_length", "n_annotated"]].rename(
        columns={"label": "Tema", "surface_form": "Surface Form",
                 "form_length": "Tokens", "n_annotated": "Total Annotations"}
    )
    st.dataframe(sf_sel.sort_values(["Tema", "Tokens"], ascending=[True, False]),
                 width="stretch")
