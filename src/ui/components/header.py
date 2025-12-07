"""
Komponent nagłówka - prosty i stabilny
"""

import streamlit as st
from ...config import APP


def render_header():
    """Prosty nagłówek"""

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image(APP.LOGO, use_container_width=True)
        except:
            st.title("☁️ Chmura+")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Klasy ASPRS", "45+")
    with col2:
        st.metric("⚡ Wersja", "2.0")
    with col3:
        st.metric("🏆 Hackathon", "2025")

    st.caption("Automatyczna klasyfikacja chmur punktów LiDAR dla infrastruktury CPK")
