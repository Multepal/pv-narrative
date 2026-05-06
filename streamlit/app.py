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

st.markdown(
    f"<style>:root {{ --sidebar-min-width: {cfg['sidebar']['min_width_px']}px; --sidebar-width: {cfg['sidebar']['width_vw']}vw; }}</style>",
    unsafe_allow_html=True,
)
with open(os.path.join(APP_DIR, "style.css")) as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    f'<div style="background:#111111;color:#ffffff;padding:0.5rem 1.25rem;'
    f'margin-bottom:1rem;font-size:1.1rem;font-weight:600;letter-spacing:0.03em;">'
    f'{cfg["app"]["title"]}</div>',
    unsafe_allow_html=True,
)

pg = st.navigation([
    st.Page(p["path"], title=p["title"], icon=p["icon"])
    for p in cfg["pages"]
])
pg.run()
