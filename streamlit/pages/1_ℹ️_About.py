import streamlit as st
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))

content = open(f"{APP_DIR}/about.md", "r", encoding="utf-8").read()
st.markdown(content)