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

_tab_strip, _tab_heat, _tab_stats, _tab_cov, _tab_outlier, _tab_cooc = st.tabs([
    "Strip Plot", "Presence Heatmap", "Statistics", "Coverage",
    "Outlier Analysis", "Co-occurrence Network"
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
        _vert_bounds, _chunk_labels_hac = _hac_boundaries(
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

# ── Tab 5: Outlier Analysis ───────────────────────────────────────────────────
with _tab_outlier:
    st.caption(
        "Standardized residuals for each (tema, cluster) pair. "
        "Red = tema appears more in this cluster than expected by chance; "
        "blue = less than expected. "
        "Uses the same k-cluster HAC parameters as the Presence Heatmap tab."
    )

    # Map each chunk (0-indexed) to its HAC cluster (1-indexed fcluster output)
    _chunk_to_clust = {c: int(_chunk_labels_hac[c]) for c in range(int(n_chunks))}

    # Filter CAM to temas that actually appear as columns
    _temas_in_cam = [t for t in selected_temas if t in CAM.columns]
    if not _temas_in_cam:
        st.warning("No annotations for the selected temas.")
        st.stop()

    # Aggregate annotation counts by cluster
    _cam_sel = CAM[_temas_in_cam].copy()
    _cam_sel.index = pd.Index([_chunk_to_clust.get(c, -1) for c in _cam_sel.index], name="cluster")
    _CT = _cam_sel[_cam_sel.index >= 0].groupby(level=0).sum()

    # Standardized residuals: (O − E) / sqrt(E)
    _O = _CT.values.astype(float)
    _row_sums = _O.sum(axis=1, keepdims=True)
    _col_sums = _O.sum(axis=0, keepdims=True)
    _grand = _O.sum()
    if _grand == 0:
        st.warning("No annotations in the selected temas.")
        st.stop()
    _E = (_row_sums @ _col_sums) / _grand
    _SR = np.where(_E > 0, (_O - _E) / np.sqrt(_E), 0.0)
    _SR_df = pd.DataFrame(_SR, index=_CT.index, columns=_CT.columns)
    _E_df  = pd.DataFrame(_E,  index=_CT.index, columns=_CT.columns)

    _threshold_ol = st.slider("Highlight threshold |r| ≥", 0.5, 5.0, 2.0, 0.5, key="ol_thresh")

    # Build letter map from _chunk_labels_hac first-appearance order so cluster
    # letters match the vertical boundary groupings in the Presence Heatmap tab.
    _clust_ids = sorted(_CT.index.tolist())
    _hac_arr   = np.array(_chunk_labels_hac)
    _first_seen_ol     = {c: int(np.argmax(_hac_arr == c)) for c in _clust_ids}
    _by_appearance_ol  = sorted(_clust_ids, key=lambda c: _first_seen_ol[c])
    _letter_map_ol     = {c: chr(65 + i) for i, c in enumerate(_by_appearance_ol)}
    _clust_lbls        = [_letter_map_ol[c] for c in _clust_ids]

    # Tema axis — same ordering as heatmap
    _ol_temas  = [t for t in _horiz_temas if t in _SR_df.columns]
    _ol_labels = [label_map[t] for t in _ol_temas]

    # Build z and hover matrices: rows = temas, cols = clusters
    _z_ol, _hover_ol = [], []
    for tid, tlabel in zip(_ol_temas, _ol_labels):
        row_z, row_h = [], []
        for cid, clbl in zip(_clust_ids, _clust_lbls):
            sr  = float(_SR_df.loc[cid, tid])
            obs = int(_CT.loc[cid, tid])
            exp = float(_E_df.loc[cid, tid])
            row_z.append(sr if abs(sr) >= _threshold_ol else 0.0)
            row_h.append(
                f"<b>{tlabel}</b> · Cluster {clbl}<br>"
                f"Observed: {obs} · Expected: {exp:.1f}<br>"
                f"Residual: {sr:+.2f}"
            )
        _z_ol.append(row_z)
        _hover_ol.append(row_h)

    fig_ol = go.Figure(go.Heatmap(
        z=_z_ol,
        x=_clust_lbls,
        y=_ol_labels,
        customdata=_hover_ol,
        hovertemplate="%{customdata}<extra></extra>",
        colorscale="RdBu",
        zmid=0,
        showscale=True,
        colorbar=dict(title="r"),
        hoverlabel=dict(align="left", font=dict(family="monospace", size=12)),
        xgap=3,
        ygap=1,
    ))
    _row_px_ol = max(14, 400 // max(n_selected, 1))
    fig_ol.update_layout(
        height=max(300, n_selected * _row_px_ol + 80),
        margin=dict(l=200, r=80, t=20, b=40),
        xaxis=dict(title="HAC Cluster", side="top"),
        yaxis=dict(
            autorange="reversed",
            categoryorder="array",
            categoryarray=_ol_labels,
            tickfont=dict(size=max(8, min(12, 300 // max(n_selected, 1)))),
        ),
    )
    st.plotly_chart(fig_ol, width="stretch")

    # Outlier table
    _SR_df.index.name = "cluster"
    _melt = _SR_df.reset_index().melt(id_vars="cluster", var_name="ana_id", value_name="residual")
    _melt["Cluster"]    = _melt["cluster"].map(_letter_map_ol)
    _melt["Tema"]       = _melt["ana_id"].map(label_map)
    _melt["|r|"]        = _melt["residual"].abs()
    _extreme = (
        _melt[_melt["|r|"] >= _threshold_ol]
        .sort_values("|r|", ascending=False)
        [["Tema", "Cluster", "residual", "|r|"]]
        .rename(columns={"residual": "Residual", "|r|": "|Residual|"})
        .assign(Residual=lambda d: d["Residual"].round(2))
        .assign(**{"|Residual|": lambda d: d["|Residual|"].round(2)})
        .reset_index(drop=True)
    )
    with st.expander(
        f"Outlier table — {len(_extreme)} (tema, cluster) pairs with |r| ≥ {_threshold_ol}",
        expanded=True
    ):
        if _extreme.empty:
            st.info("No pairs exceed the threshold — try lowering it.")
        else:
            st.dataframe(_extreme, width="stretch")

    # ── Tema profile ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Tema Profile", anchor=False)
    st.caption(
        "Bars = annotation count per chunk, colored by HAC cluster. "
        "Right panel = standardized residual per cluster. "
        "Dashed lines at r = ±2."
    )

    from plotly.subplots import make_subplots as _make_subplots

    _prof_temas   = _ol_temas if _ol_temas else list(CAM.columns)
    _prof_default = next(
        (i for i, t in enumerate(_prof_temas) if "XIB" in t.upper()), 0
    )
    _prof_sel = st.selectbox(
        "Tema",
        options=_prof_temas,
        index=_prof_default,
        format_func=lambda t: label_map.get(t, t),
        key="ol_prof_sel",
    )

    _prof_label  = label_map.get(_prof_sel, _prof_sel)
    _prof_counts = CAM[_prof_sel] if _prof_sel in CAM.columns else pd.Series(
        0, index=range(int(n_chunks))
    )

    _pal_bg    = px.colors.qualitative.Pastel
    _clust_col = {c: _pal_bg[i % len(_pal_bg)] for i, c in enumerate(_by_appearance_ol)}
    _bar_colors = [
        _clust_col.get(_chunk_to_clust.get(c, -1), "#cccccc")
        for c in range(int(n_chunks))
    ]
    _bar_hover = [
        f"Chunk {c} · Cluster {_letter_map_ol.get(_chunk_to_clust.get(c, -1), '?')}"
        f"<br>{int(_prof_counts.iloc[c])} annotations"
        for c in range(int(n_chunks))
    ]

    fig_prof = _make_subplots(
        rows=1, cols=2,
        column_widths=[0.72, 0.28],
        subplot_titles=[f"{_prof_label} — annotations per chunk", "Residual per cluster"],
    )

    fig_prof.add_trace(go.Bar(
        x=list(range(int(n_chunks))),
        y=_prof_counts.values,
        marker_color=_bar_colors,
        customdata=_bar_hover,
        hovertemplate="%{customdata}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    if _prof_sel in _SR_df.columns:
        _res_vals = [float(_SR_df.loc[c, _prof_sel]) for c in _clust_ids]
        fig_prof.add_trace(go.Bar(
            x=_clust_lbls,
            y=_res_vals,
            marker_color=["crimson" if r > 0 else "steelblue" for r in _res_vals],
            hovertemplate="Cluster %{x}<br>r = %{y:.2f}<extra></extra>",
            showlegend=False,
        ), row=1, col=2)
        for _thresh_y in [2.0, -2.0]:
            fig_prof.add_shape(
                type="line",
                x0=0, x1=1, xref="x2 domain",
                y0=_thresh_y, y1=_thresh_y, yref="y2",
                line=dict(dash="dot", color="#888", width=1),
            )
        fig_prof.add_shape(
            type="line",
            x0=0, x1=1, xref="x2 domain",
            y0=0, y1=0, yref="y2",
            line=dict(color="#333", width=0.8),
        )

    fig_prof.update_layout(
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor="white",
    )
    fig_prof.update_xaxes(title_text="Chunk", row=1, col=1, showgrid=False)
    fig_prof.update_xaxes(title_text="Cluster", row=1, col=2)
    fig_prof.update_yaxes(title_text="Annotations", row=1, col=1,
                          showgrid=True, gridcolor="#eee")
    fig_prof.update_yaxes(title_text="r", row=1, col=2, zeroline=False)
    st.plotly_chart(fig_prof, width="stretch")

    with st.expander("How outlier detection works", expanded=False):
        st.markdown(r"""
#### Detecting narrative outliers: temas in unexpected clusters

This analysis asks a structural question: does a named entity (*tema*) appear in a
narrative cluster **more or less often than we would expect** given its overall
frequency in the text?

---

**Step 1 — Cluster the text**

The K'iche' text is divided into *n* equal chunks in narrative order.
HAC (hierarchical agglomerative clustering) then groups those chunks into *k* clusters
based on lexical similarity: chunks that share vocabulary end up in the same cluster.
The cluster boundaries are the vertical crimson lines in the **Presence Heatmap** tab.

**Step 2 — Count observed annotations**

For each (tema *t*, cluster *c*) pair, count how many `<rs>` annotations fall there.
This is the observed count *O*:

$$O_{tc} = \text{annotations of tema } t \text{ in chunks belonging to cluster } c$$

**Step 3 — Compute expected counts**

If annotations were distributed proportionally to cluster size, we would predict:

$$E_{tc} = \frac{\text{(total annotations in cluster } c\text{)} \times \text{(total annotations of tema } t\text{)}}{\text{grand total annotations}}$$

A large cluster naturally attracts more annotations just by having more text,
so *E* adjusts for that — it is the null hypothesis of *no association* between
tema identity and cluster membership.

**Step 4 — Standardized residual**

The residual measures how far the observation deviates from the expectation, scaled
by the expected variability:

$$r_{tc} = \frac{O_{tc} - E_{tc}}{\sqrt{E_{tc}}}$$

This is the per-cell contribution to a Pearson chi-square test of independence.

---

#### Reading the residual heatmap

| Color | Residual | Interpretation |
|---|---|---|
| **Deep red** | r ≫ 0 | Tema is *over-represented* — it clusters here more than chance predicts |
| White | r ≈ 0 | Tema appears about as often as expected |
| **Deep blue** | r ≪ 0 | Tema is *under-represented* — conspicuously absent from this cluster |

Conventional thresholds: **\|r\| ≥ 2** is noteworthy; **\|r\| ≥ 3** is a strong outlier.

#### Reading the tema profile

The **Tema Profile** panel below shows one entity's annotations across all chunks
(bar height = count, bar color = cluster membership) alongside its per-cluster
residuals. Together they answer: *where in the narrative does this entity live, and
in which cluster does it appear more or less than its overall distribution predicts?*

---

#### Interpretive caution

Standardized residuals are sensitive to total count. A tema with only one or two
annotations can produce a large residual by chance — the formula amplifies small
deviations when *E* is near zero. Treat high-residual cells as **leads for closer
reading**, not as definitive structural claims. The **Chunk Text Browser** in the
Presence Heatmap tab lets you read the actual passage for any surprising (tema, chunk) pair.
""")

# ── Tab 6: Co-occurrence Network ──────────────────────────────────────────────
with _tab_cooc:
    import networkx as _nx

    st.caption(
        "Nodes = temas. An edge connects two temas that appear together in at least "
        "N chunks within the selected cluster. Node color = dominant cluster. "
        "Node size = total annotation count. Edge width = co-occurrence count."
    )

    _cc_cols = st.columns([1, 1, 1, 1])

    # Cluster filter
    _cc_clust_opts = ["All clusters"] + [
        f"Cluster {_letter_map_ol[c]}" for c in _by_appearance_ol
    ]
    _cc_clust_filter = _cc_cols[0].selectbox(
        "Cluster", _cc_clust_opts, key="cc_clust"
    )
    _cc_layout_choice = _cc_cols[3].selectbox(
        "Layout", ["Spring", "Kamada-Kawai", "Circular"], key="cc_layout"
    )

    # Resolve which chunks to include
    if _cc_clust_filter == "All clusters":
        _cc_chunks = list(range(int(n_chunks)))
    else:
        _cc_letter  = _cc_clust_filter.split()[-1]
        _cc_clust_id = next(c for c in _clust_ids if _letter_map_ol[c] == _cc_letter)
        _cc_chunks  = [c for c in range(int(n_chunks))
                       if _chunk_to_clust.get(c) == _cc_clust_id]

    # Co-occurrence matrix: chunk-level binary presence → dot product
    _cc_temas   = [t for t in _ol_temas if t in CAM.columns]
    _CAM_cc     = CAM.loc[_cc_chunks, _cc_temas] if _cc_chunks else pd.DataFrame()
    if _CAM_cc.empty or _CAM_cc.values.sum() == 0:
        st.info("No annotations in the selected cluster — try 'All clusters'.")
        st.stop()

    _CAM_bin    = (_CAM_cc > 0).astype(int)
    _CO         = _CAM_bin.T @ _CAM_bin   # temas × temas

    # Upper-triangle values (exclude diagonal) for slider range
    _co_vals = _CO.values[np.triu_indices(len(_cc_temas), k=1)]
    _co_max  = int(_co_vals.max()) if len(_co_vals) else 1
    _co_default = max(1, int(np.percentile(_co_vals[_co_vals > 0], 75))
                      if (_co_vals > 0).any() else 1)

    _cc_thresh = _cc_cols[1].slider(
        "Min co-occurrence", 1, max(_co_max, 2), _co_default, 1, key="cc_thresh"
    )
    _cc_show_isolated = _cc_cols[2].checkbox("Show isolated nodes", False, key="cc_iso")

    # Build graph
    G_cc = _nx.Graph()
    # Dominant cluster per tema (from _CT, consistent with _chunk_labels_hac)
    _tema_dom_clust = {
        t: int(_CT[t].idxmax()) if t in _CT.columns and _CT[t].sum() > 0 else -1
        for t in _cc_temas
    }
    _n_metric_map = METRICS["n"].to_dict()

    for t in _cc_temas:
        G_cc.add_node(t,
            label=label_map.get(t, t),
            cluster=_tema_dom_clust.get(t, -1),
            n=int(_n_metric_map.get(t, 1)),
        )
    for i, t1 in enumerate(_cc_temas):
        for t2 in _cc_temas[i + 1:]:
            w = int(_CO.loc[t1, t2])
            if w >= _cc_thresh:
                G_cc.add_edge(t1, t2, weight=w)

    if not _cc_show_isolated:
        G_cc.remove_nodes_from([n for n in list(G_cc.nodes()) if G_cc.degree(n) == 0])

    if G_cc.number_of_nodes() == 0:
        st.info("No co-occurrences at this threshold — lower the slider.")
        st.stop()

    # Layout
    _cc_k = 2.5 / max(G_cc.number_of_nodes() ** 0.5, 1)
    if _cc_layout_choice == "Kamada-Kawai":
        _pos = _nx.kamada_kawai_layout(G_cc, weight="weight")
    elif _cc_layout_choice == "Circular":
        _pos = _nx.circular_layout(G_cc)
    else:
        _pos = _nx.spring_layout(G_cc, seed=42, k=_cc_k, weight="weight", iterations=60)

    # Colors: one per cluster letter, matched to Pastel palette
    _pal_net  = px.colors.qualitative.Pastel
    _net_clust_col = {
        c: _pal_net[i % len(_pal_net)] for i, c in enumerate(_by_appearance_ol)
    }

    # Edge traces (one per edge for variable width)
    _cc_edge_traces = []
    _max_w = max((G_cc[u][v]["weight"] for u, v in G_cc.edges()), default=1)
    for u, v, data in G_cc.edges(data=True):
        x0, y0 = _pos[u]
        x1, y1 = _pos[v]
        w = data["weight"]
        _cc_edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=max(0.8, w / _max_w * 7), color="rgba(100,100,100,0.35)"),
            hoverinfo="none",
            showlegend=False,
        ))

    # Invisible midpoint scatter for edge hover
    _mid_x  = [((_pos[u][0] + _pos[v][0]) / 2) for u, v in G_cc.edges()]
    _mid_y  = [((_pos[u][1] + _pos[v][1]) / 2) for u, v in G_cc.edges()]
    _mid_hv = [
        f"<b>{label_map.get(u, u)}</b> × <b>{label_map.get(v, v)}</b>"
        f"<br>Co-occurrence: {G_cc[u][v]['weight']} chunks"
        for u, v in G_cc.edges()
    ]
    _cc_edge_hover = go.Scatter(
        x=_mid_x, y=_mid_y, mode="markers",
        marker=dict(size=10, color="rgba(0,0,0,0)"),
        customdata=_mid_hv,
        hovertemplate="%{customdata}<extra></extra>",
        showlegend=False,
    )

    # Node trace
    _cc_nodes = list(G_cc.nodes())
    _cc_nx    = [_pos[t][0] for t in _cc_nodes]
    _cc_ny    = [_pos[t][1] for t in _cc_nodes]
    _cc_sizes = [max(10, min(40, G_cc.nodes[t]["n"] ** 0.55 * 4)) for t in _cc_nodes]
    _cc_colors = [
        _net_clust_col.get(G_cc.nodes[t]["cluster"], "#bbbbbb") for t in _cc_nodes
    ]
    _cc_hover = [
        f"<b>{label_map.get(t, t)}</b><br>"
        f"Cluster {_letter_map_ol.get(G_cc.nodes[t]['cluster'], '?')}<br>"
        f"Total annotations: {G_cc.nodes[t]['n']}<br>"
        f"Connections: {G_cc.degree(t)}"
        for t in _cc_nodes
    ]
    _cc_node_trace = go.Scatter(
        x=_cc_nx, y=_cc_ny,
        mode="markers+text",
        marker=dict(
            size=_cc_sizes,
            color=_cc_colors,
            line=dict(width=1, color="white"),
        ),
        text=[label_map.get(t, t) for t in _cc_nodes],
        textposition="top center",
        textfont=dict(size=9),
        customdata=_cc_hover,
        hovertemplate="%{customdata}<extra></extra>",
        showlegend=False,
    )

    fig_net = go.Figure(
        data=_cc_edge_traces + [_cc_edge_hover, _cc_node_trace],
        layout=go.Layout(
            height=620,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="white",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode="closest",
        ),
    )
    st.plotly_chart(fig_net, width="stretch")
    st.caption(
        f"{G_cc.number_of_nodes()} nodes · {G_cc.number_of_edges()} edges · "
        f"co-occurrence ≥ {_cc_thresh} chunks · "
        f"{'all clusters' if _cc_clust_filter == 'All clusters' else _cc_clust_filter}"
    )
