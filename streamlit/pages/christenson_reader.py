"""
Christenson's Popol Wuj — Text Reader

Shows all 87 chapters from Christenson (2007), organized by 8 interpretive
narrative divisions with thematic labels.  Each chapter also shows its k=6
HAC cluster label (from christenson-CHAP_MOD.csv) as secondary analytical data.
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

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]

DIVISIONS = [
    {"label": "First Creation",                "chap_start": 1,  "chap_end": 8},
    {"label": "7 Macaw",                       "chap_start": 9,  "chap_end": 14},
    {"label": "About the Two Boys' Father",    "chap_start": 15, "chap_end": 18},
    {"label": "About the Two Boys' Mother",    "chap_start": 19, "chap_end": 21},
    {"label": "The Two Boys Defeat Death",     "chap_start": 22, "chap_end": 40},
    {"label": "The Second Creation",           "chap_start": 41, "chap_end": 47},
    {"label": "Religion",                      "chap_start": 48, "chap_end": 64},
    {"label": "Politics",                      "chap_start": 65, "chap_end": 87},
]

CLUSTER_COLORS = {
    1: "#e06c4b",
    2: "#6b9bd2",
    3: "#8e6bbf",
    4: "#5aab6b",
    5: "#c9a84c",
    6: "#4babc9",
}

DIV_COLORS = [
    "#4babc9",  # I
    "#e06c4b",  # II
    "#8e6bbf",  # III
    "#6b9bd2",  # IV
    "#8e6bbf",  # V
    "#4babc9",  # VI
    "#5aab6b",  # VII
    "#c9a84c",  # VIII
]


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
    "organized into 8 narrative divisions. "
    "Each chapter's k=6 HAC cluster label is shown as analytical context."
)

CHAPS = load_chapters()

# ── Sidebar: language toggle + TOC ───────────────────────────────────────────
with st.sidebar:
    lang = st.radio("Display text", ["English", "K'iche'", "Both"], index=0)
    st.divider()
    st.markdown("**Divisions**")
    for i, d in enumerate(DIVISIONS):
        rom = ROMAN[i]
        st.markdown(
            f"[{rom}. {d['label']}  "
            f"*(ch. {d['chap_start']}–{d['chap_end']})*](#div-{rom.lower()})"
        )

# ── Main content ──────────────────────────────────────────────────────────────
for i, d in enumerate(DIVISIONS):
    rom = ROMAN[i]
    color = DIV_COLORS[i]
    n_chaps = d["chap_end"] - d["chap_start"] + 1

    st.markdown(
        f"<h2 id='div-{rom.lower()}' style='color:{color}'>"
        f"{rom}. {d['label']}"
        f"<span style='font-size:0.55em; color:#888; font-weight:normal'>"
        f"  &nbsp;·&nbsp; chapters {d['chap_start']}–{d['chap_end']}"
        f"  &nbsp;·&nbsp; {n_chaps} chapter{'s' if n_chaps != 1 else ''}"
        f"</span></h2>",
        unsafe_allow_html=True,
    )

    div_chaps = CHAPS.loc[d["chap_start"]: d["chap_end"]]

    for chap_num, row in div_chaps.iterrows():
        title = row["chap_title"]
        page = int(row["page_num"]) if pd.notna(row.get("page_num")) else None
        eng = str(row["chap_eng_str"]) if pd.notna(row.get("chap_eng_str")) else ""
        quc = str(row["chap_quc_str"]) if pd.notna(row.get("chap_quc_str")) else ""

        cl = row.get("cluster_label")
        gloss = row.get("max_cluster_gloss", "")
        cl_str = (
            f" · cluster {int(cl)} *{gloss}*"
            if pd.notna(cl) and pd.notna(gloss) and str(gloss) not in ("", "nan")
            else ""
        )
        page_str = f" · p. {page}" if page else ""
        label = f"**{chap_num}.** {title}{page_str}{cl_str}"

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
