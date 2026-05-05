import os
import yaml
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(APP_DIR, "config.yaml"), encoding="utf-8") as _f:
    cfg = yaml.safe_load(_f)

st.set_page_config(
    page_title=cfg["app"]["page_title"],
    page_icon=cfg["app"]["page_icon"],
    layout=cfg["app"]["layout"],
)

st.markdown(f"""
<style>
.block-container {{ padding-top: 1rem; padding-bottom: 1rem; }}
h3 {{ margin-bottom: 1rem; }}
@media (min-width: 768px) {{
    section[data-testid="stSidebar"] {{
        min-width: {cfg['sidebar']['min_width_px']}px;
        width: {cfg['sidebar']['width_vw']}vw;
    }}
}}
</style>
""", unsafe_allow_html=True)

pg = st.navigation([
    st.Page("pages/explorer.py",               title="Structural Explorer", icon="📜"),
    st.Page("pages/1_ℹ️_About.py",             title="About",              icon="ℹ️"),
    st.Page("pages/2_⚙️_Model_Diagnostics.py", title="Model Diagnostics",  icon="⚙️"),
])
pg.run()
