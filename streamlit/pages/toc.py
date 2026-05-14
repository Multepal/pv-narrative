import streamlit as st


def render_toc(sections: list[tuple[str, str]]) -> None:
    """Render a page-level TOC in the sidebar.

    sections: list of (label, anchor_id) pairs matching st.subheader(anchor=...) calls.
    """
    with st.sidebar:
        st.markdown("**Contents**")
        st.markdown("\n".join(f"- [{label}](#{anchor})" for label, anchor in sections))
        st.divider()
