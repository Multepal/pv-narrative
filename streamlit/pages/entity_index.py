"""
Entity Index — cross-edition named entity concordance.

Source: notebooks/ner/ENTITY_CROSSMAP.csv and per-edition {name}-NER.csv files,
produced by notebooks/ner/01-ner-pipeline.ipynb (ner_pipeline.py).
"""

import os
import json
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
    "Max entities shown in heatmap", min_value=10, max_value=100, value=40, step=5
)
sort_by = _c4.selectbox(
    "Sort entities by", ["total mentions", "label A-Z", "type"], key="ei_sort"
)

# Filter crossmap
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

n_shown = len(_total_top)
st.caption(
    f"**{n_shown}** entities shown · {len(_total)} pass filters "
    f"(type ∈ {{{', '.join(sel_types[:3])}{'…' if len(sel_types) > 3 else ''}}} · "
    f"total ≥ {min_mentions})"
)

if n_shown == 0:
    st.warning("No entities match the current filters.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────

_tab_heat, _tab_index = st.tabs(["Cross-Edition Heatmap", "Entity Detail"])

# ── Tab 1: Heatmap ────────────────────────────────────────────────────────────

with _tab_heat:
    st.caption(
        "Mention counts per entity (rows) × edition (columns). "
        "Sorted by total mentions descending. Hover for counts."
    )

    # Build heatmap matrix
    _eids_ordered = _total_top["entity_id"].tolist()
    _labels_ordered = _total_top["label"].tolist()
    _ed_cols = [e for e in EDITION_ORDER if e in xmap["edition"].unique()]
    _ed_display = [EDITION_LABELS.get(e, e) for e in _ed_cols]

    _pivot = (
        xmap[xmap["entity_id"].isin(_eids_ordered)]
        .pivot_table(index="entity_id", columns="edition",
                     values="mention_count", fill_value=0)
        .reindex(index=_eids_ordered, columns=_ed_cols, fill_value=0)
    )

    _z     = _pivot.values.tolist()
    _hover = []
    for i, (eid, lbl) in enumerate(zip(_eids_ordered, _labels_ordered)):
        row = []
        for j, ed in enumerate(_ed_cols):
            cnt = _pivot.loc[eid, ed] if eid in _pivot.index else 0
            row.append(f"<b>{lbl}</b><br>{EDITION_LABELS.get(ed, ed)}: {cnt}")
        _hover.append(row)

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
            categoryarray=list(reversed(_labels_ordered)),
            tickfont=dict(size=10),
        ),
        xaxis=dict(tickangle=-30),
        margin=dict(l=180, r=20, t=30, b=80),
        height=max(350, n_shown * 16 + 120),
    )
    st.plotly_chart(_fig_heat, use_container_width=True)

# ── Tab 2: Entity detail ──────────────────────────────────────────────────────

with _tab_index:
    _label_options = _total_top["label"].tolist()
    _label_to_eid  = dict(zip(_total_top["label"], _total_top["entity_id"]))
    _label_to_type = dict(zip(_total_top["label"], _total_top["type"]))

    sel_label = st.selectbox(
        "Select entity:", ["— select —"] + _label_options, key="ei_ent_sel"
    )

    if sel_label and sel_label != "— select —":
        sel_eid  = _label_to_eid[sel_label]
        sel_type = _label_to_type.get(sel_label, "")

        # Header row
        _hcol1, _hcol2 = st.columns([3, 1])
        _hcol1.markdown(
            f"#### {sel_label} "
            f"<span style='font-size:0.8em;color:gray'>({sel_type})</span>",
            unsafe_allow_html=True,
        )
        _tot = _total[_total["entity_id"] == sel_eid]["total"].sum()
        _hcol2.metric("Total mentions", int(_tot))

        # Edition breakdown
        _ent_xmap = xmap[xmap["entity_id"] == sel_eid].copy()
        _ent_xmap["edition_label"] = _ent_xmap["edition"].map(EDITION_LABELS)
        _ent_xmap = _ent_xmap.sort_values("mention_count", ascending=False)

        st.caption("Mentions per edition:")
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

        # Per-mention table with ngram variants
        st.divider()
        st.caption("All mentions across editions — ngram matched, location, and match type.")

        _all_ner  = load_all_ner()
        _ent_rows = _all_ner[_all_ner["entity_id"] == sel_eid].copy()
        _ent_rows["edition_label"] = _ent_rows["edition"].map(EDITION_LABELS)

        # Summarise: ngram × edition → count + sample locations
        _summary = (
            _ent_rows.groupby(["edition_label", "ngram", "match_type"])
            .agg(count=("ohco_path", "count"),
                 locations=("ohco_path", lambda x: ", ".join(str(v) for v in sorted(x.unique())[:5])))
            .reset_index()
            .sort_values(["edition_label", "count"], ascending=[True, False])
        )
        _summary.columns = ["Edition", "Matched ngram", "Match type", "Count", "Locations (sample)"]

        st.dataframe(_summary, use_container_width=True, hide_index=True)

        with st.expander("Raw n-gram inventory (all editions combined)"):
            _ngrams = (
                _ent_rows.groupby(["ngram", "match_type"])["edition"]
                .count()
                .rename("total_count")
                .reset_index()
                .sort_values("total_count", ascending=False)
            )
            _ngrams.columns = ["Ngram", "Match type", "Count"]
            st.dataframe(_ngrams, use_container_width=True, hide_index=True)
