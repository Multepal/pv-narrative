"""
Entity Index — cross-edition named entity concordance.

Source: notebooks/ner/ENTITY_CROSSMAP.csv and per-edition {name}-NER.csv files,
produced by notebooks/ner/01-ner-pipeline.ipynb (ner_pipeline.py).
"""

import os
import yaml
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

APP_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

with open(APP_DIR / "../config.yaml", encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

NER_DIR = APP_DIR / "../../notebooks/ner"

EDITION_LABELS = {
    name: info["label"]
    for name, info in cfg.get("sources", {}).items()
}
EDITION_ORDER = list(EDITION_LABELS.keys())

st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_crossmap() -> pd.DataFrame:
    return pd.read_csv(NER_DIR / "ENTITY_CROSSMAP.csv")


@st.cache_data(show_spinner=False)
def load_ner(edition: str) -> pd.DataFrame:
    path = NER_DIR / f"{edition}-NER.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=[
        "entity_id", "edition", "ohco_path", "ngram", "ngram_len", "match_type", "label", "type"
    ])


@st.cache_data(show_spinner=False)
def load_all_ner() -> pd.DataFrame:
    frames = [load_ner(ed) for ed in EDITION_ORDER]
    return pd.concat([f for f in frames if len(f)], ignore_index=True)


# ── Page header ───────────────────────────────────────────────────────────────

st.title("Entity Index — Cross-Edition")
st.caption(
    "Named entities identified by gazetteer n-gram matching across all eight editions. "
    "Gazetteer: `temas/topics.xml` + `textos/colop-glossary.txt` + Ximénez surface-form variants. "
    "Location: paragraph for K'iche' editions; sentence for Recinos/Tedlock."
)

with st.expander("Methodology", expanded=False):
    st.markdown("""
The NER pipeline (`notebooks/ner/ner_pipeline.py`) uses **greedy longest-match** n-gram
lookup (n = 1–4) against a unified gazetteer of K'iche', Spanish, and English surface forms
for each canonical entity. Matching is exact first, then **fuzzy** (apostrophes and
diacritics stripped) to bridge orthographic variants across editions and centuries.

Spanish and English editions use K'iche' transliterations for proper names, so their
lookup includes the full K'iche' surface-form inventory as a fallback. Common short
tokens (single K'iche' morphemes) may produce false positives — use the `min mentions`
filter to focus on high-confidence matches.

Each match is attributed to the **canonical entity_id** from `topics.xml`. Entities
with the same transliterated form (e.g., "Hunahpu" for both the hero and the mountain
*Junajpu Juyub'*) are disambiguated by the priority order in the gazetteer.
""")

xmap = load_crossmap()

# ── Controls ──────────────────────────────────────────────────────────────────

_col_ratios = cfg["layout"]["column_ratios"]
_c1, _c2, _c3, _c4 = st.columns(_col_ratios[:4])

_all_types = sorted(xmap["type"].dropna().unique())
sel_types = _c1.multiselect(
    "Entity type", _all_types, default=_all_types, key="ei_types"
)
min_mentions = _c2.slider(
    "Min total mentions", min_value=1, max_value=200, value=5, step=1
)
n_entities = _c3.slider(
    "Max entities shown", min_value=10, max_value=100, value=40, step=5
)
sort_by = _c4.selectbox(
    "Sort entities by", ["total mentions", "label A-Z", "type"], key="ei_sort"
)

# Filter and sort entity list
_total = xmap.groupby(["entity_id", "label", "type"])["mention_count"].sum().reset_index()
_total.columns = ["entity_id", "label", "type", "total"]

if sel_types:
    _total = _total[_total["type"].isin(sel_types)]
_total = _total[_total["total"] >= min_mentions]

if sort_by == "total mentions":
    _total = _total.sort_values("total", ascending=False)
elif sort_by == "label A-Z":
    _total = _total.sort_values("label")
else:
    _total = _total.sort_values(["type", "label"])

_total_top = _total.head(n_entities)
_label_to_eid  = dict(zip(_total_top["label"], _total_top["entity_id"]))
_label_to_type = dict(zip(_total_top["label"], _total_top["type"]))

n_shown = len(_total_top)
st.caption(
    f"**{n_shown}** entities shown · {len(_total)} pass filters"
)

if n_shown == 0:
    st.warning("No entities match the current filters.")
    st.stop()

# ── Heatmap ───────────────────────────────────────────────────────────────────

_eids_ordered   = _total_top["entity_id"].tolist()
_labels_ordered = _total_top["label"].tolist()
_ed_cols    = [e for e in EDITION_ORDER if e in xmap["edition"].unique()]
_ed_display = [EDITION_LABELS.get(e, e) for e in _ed_cols]

_pivot = (
    xmap[xmap["entity_id"].isin(_eids_ordered)]
    .pivot_table(index="entity_id", columns="edition",
                 values="mention_count", fill_value=0)
    .reindex(index=_eids_ordered, columns=_ed_cols, fill_value=0)
)

_z, _hover = [], []
for eid, lbl in zip(_eids_ordered, _labels_ordered):
    row_z, row_h = [], []
    for ed in _ed_cols:
        cnt = int(_pivot.loc[eid, ed]) if eid in _pivot.index else 0
        row_z.append(cnt)
        row_h.append(f"<b>{lbl}</b><br>{EDITION_LABELS.get(ed, ed)}: {cnt}")
    _z.append(row_z)
    _hover.append(row_h)

_fig_heat = go.Figure(go.Heatmap(
    z=_z,
    x=_ed_display,
    y=_labels_ordered,
    customdata=_hover,
    hovertemplate="%{customdata}<extra></extra>",
    colorscale="YlOrRd",
    showscale=True,
))
_fig_heat.update_layout(
    yaxis=dict(
        autorange="reversed",
        categoryorder="array",
        categoryarray=_labels_ordered,
        tickfont=dict(size=10),
    ),
    xaxis=dict(tickangle=-30),
    margin=dict(l=180, r=20, t=30, b=80),
    height=max(350, n_shown * 16 + 120),
)

st.plotly_chart(_fig_heat, use_container_width=True, key="ei_heatmap")

# ── Entity selector ───────────────────────────────────────────────────────────

st.divider()

_sel_label = st.selectbox(
    "Inspect entity:",
    ["— select —"] + _labels_ordered,
    key="ei_ent_sel",
)

if not _sel_label or _sel_label == "— select —":
    st.stop()

sel_eid  = _label_to_eid[_sel_label]
sel_type = _label_to_type.get(_sel_label, "")


_hcol1, _hcol2 = st.columns([3, 1])
_hcol1.markdown(
    f"#### {_sel_label} "
    f"<span style='font-size:0.8em;color:gray'>({sel_type})</span>",
    unsafe_allow_html=True,
)
_tot = int(_total[_total["entity_id"] == sel_eid]["total"].sum())
_hcol2.metric("Total mentions", _tot)

# Per-edition bar chart
_ent_xmap = xmap[xmap["entity_id"] == sel_eid].copy()
_ent_xmap["edition_label"] = _ent_xmap["edition"].map(EDITION_LABELS)
_ent_xmap = _ent_xmap.sort_values("mention_count", ascending=False)

_bar = go.Figure(go.Bar(
    x=_ent_xmap["edition_label"],
    y=_ent_xmap["mention_count"],
    marker_color="#e07020",
))
_bar.update_layout(
    margin=dict(l=10, r=10, t=20, b=40),
    height=220,
    yaxis_title="mentions",
    xaxis_tickangle=-30,
)
st.plotly_chart(_bar, use_container_width=True)

# Concordance table: ngram × edition → count + sample locations
_all_ner  = load_all_ner()
_ent_rows = _all_ner[_all_ner["entity_id"] == sel_eid].copy()
_ent_rows["edition_label"] = _ent_rows["edition"].map(EDITION_LABELS)

_summary = (
    _ent_rows.groupby(["edition_label", "ngram", "match_type"])
    .agg(
        count=("ohco_path", "count"),
        locations=("ohco_path", lambda x: ", ".join(str(v) for v in sorted(x.unique())[:5])),
    )
    .reset_index()
    .sort_values(["edition_label", "count"], ascending=[True, False])
)
_summary.columns = ["Edition", "Matched ngram", "Match type", "Count", "Locations (sample)"]

st.caption("Matched ngrams across editions — location is paragraph id for K'iche', sentence id for others.")
st.dataframe(_summary, use_container_width=True, hide_index=True)

with st.expander("All ngram variants (combined)"):
    _ngrams = (
        _ent_rows.groupby(["ngram", "match_type"])["edition"]
        .count()
        .rename("total_count")
        .reset_index()
        .sort_values("total_count", ascending=False)
    )
    _ngrams.columns = ["Ngram", "Match type", "Count"]
    st.dataframe(_ngrams, use_container_width=True, hide_index=True)
