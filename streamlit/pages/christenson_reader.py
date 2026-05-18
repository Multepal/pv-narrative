"""
Christenson's Popol Wuj — Text Reader

Shows all 87 chapters from Christenson (2007), organized by the 8 narrative
divisions derived from k=6 HAC clustering of the K'iche' text.

Division boundaries come from the contiguous runs of cluster_label in
christenson-CHAP_MOD.csv.  The single-chapter cluster-6 anomaly (chap 13,
"The Defeat of Zipacna") is merged into the preceding cluster-1 run, and the
five closing genealogy chapters (83–87, cluster NaN) are merged into the
cluster-5 run, yielding exactly 8 divisions.
"""

import os
import yaml
import streamlit as st
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "../config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

CHAP_MOD_PATH = os.path.normpath(
    os.path.join(APP_DIR, "../../notebooks/christenson/christenson-CHAP_MOD.csv")
)

# 8 narrative divisions (k=6 HAC, anomaly merged)
DIVISIONS = [
    {"div": 1, "cluster": 6, "gloss": "kaj",        "chap_start": 1,  "chap_end": 7},
    {"div": 2, "cluster": 1, "gloss": "kaqix",      "chap_start": 8,  "chap_end": 13},
    {"div": 3, "cluster": 3, "gloss": "xib'alb'a",  "chap_start": 14, "chap_end": 18},
    {"div": 4, "cluster": 2, "gloss": "q'apoj",     "chap_start": 19, "chap_end": 24},
    {"div": 5, "cluster": 3, "gloss": "xib'alb'a",  "chap_start": 25, "chap_end": 37},
    {"div": 6, "cluster": 6, "gloss": "kaj",        "chap_start": 38, "chap_end": 46},
    {"div": 7, "cluster": 4, "gloss": "b'alam",     "chap_start": 47, "chap_end": 64},
    {"div": 8, "cluster": 5, "gloss": "ajaw",       "chap_start": 65, "chap_end": 87},
]

CLUSTER_COLORS = {
    1: "#e06c4b",
    2: "#6b9bd2",
    3: "#8e6bbf",
    4: "#5aab6b",
    5: "#c9a84c",
    6: "#4babc9",
}


@st.cache_data(show_spinner=False)
def load_chapters() -> pd.DataFrame:
    df = pd.read_csv(CHAP_MOD_PATH)
    df = df.set_index("chap_num")
    return df


st.markdown(
    f"<style>.block-container{{max-width:{cfg['layout']['max_width_px']}px !important;}}</style>",
    unsafe_allow_html=True,
)

st.title("Christenson's Popol Wuj — Narrative Divisions")
st.caption(
    "All 87 chapters from Allen J. Christenson's 2007 literal translation, "
    "grouped into 8 narrative divisions derived from k=6 hierarchical agglomerative "
    "clustering (HAC) of the K'iche' text."
)

with st.expander("About the divisions", expanded=False):
    st.markdown("""
The 8 divisions are contiguous runs of the same HAC cluster label over the narrative.
With k=6 clusters the text splits into 9 raw runs; one single-chapter anomaly (ch. 13,
cluster 6 sandwiched inside the cluster-1 Sipakna arc) is merged into Division 2, and
the five closing genealogy chapters (83–87, unassigned) are absorbed into Division 8,
giving exactly 8 divisions.  Division and cluster labels come from
`christenson-CHAP_MOD.csv`; the cluster gloss is the highest-weight K'iche' term for
that cluster derived from TF-IDF.
""")

CHAPS = load_chapters()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    lang = st.radio("Display text", ["English", "K'iche'", "Both"], index=0)
    st.divider()
    st.markdown("**Divisions**")
    for d in DIVISIONS:
        n = d["chap_end"] - d["chap_start"] + 1
        st.markdown(
            f"[{d['div']}. *{d['gloss']}*  (ch. {d['chap_start']}–{d['chap_end']})](#div-{d['div']})"
        )

# ── Main content ──────────────────────────────────────────────────────────────
for d in DIVISIONS:
    c = d["cluster"]
    color = CLUSTER_COLORS.get(c, "#888")
    n_chaps = d["chap_end"] - d["chap_start"] + 1

    st.markdown(
        f"<h2 id='div-{d['div']}' style='color:{color}'>"
        f"Division {d['div']} &mdash; <em>{d['gloss']}</em>"
        f"<span style='font-size:0.6em; color:#888; font-weight:normal'>"
        f"  cluster {c} &nbsp;·&nbsp; "
        f"chapters {d['chap_start']}–{d['chap_end']} &nbsp;·&nbsp; "
        f"{n_chaps} chapter{'s' if n_chaps != 1 else ''}"
        f"</span></h2>",
        unsafe_allow_html=True,
    )

    div_chaps = CHAPS.loc[d["chap_start"]: d["chap_end"]]

    for chap_num, row in div_chaps.iterrows():
        title = row["chap_title"]
        page = int(row["page_num"]) if pd.notna(row.get("page_num")) else None
        eng = str(row["chap_eng_str"]) if pd.notna(row.get("chap_eng_str")) else ""
        quc = str(row["chap_quc_str"]) if pd.notna(row.get("chap_quc_str")) else ""

        page_tag = f" <span style='font-size:0.85em; color:#aaa'>(p. {page})</span>" if page else ""
        label = f"**{chap_num}.** {title}{page_tag}"

        with st.expander(label, expanded=False):
            if lang == "English":
                st.markdown(eng if eng else "*No English text available.*")
            elif lang == "K'iche'":
                st.markdown(quc if quc else "*No K'iche' text available.*")
            else:
                col_e, col_q = st.columns(2)
                with col_e:
                    st.caption("English (Christenson 2007, literal)")
                    st.markdown(eng if eng else "*No English text available.*")
                with col_q:
                    st.caption("K'iche' (Christenson 2007)")
                    st.markdown(quc if quc else "*No K'iche' text available.*")

    st.divider()
