"""
CPK - Klasyfikator Chmur Punktów v2.0
HackNation 2025
"""

import streamlit as st
import logging

from src.config import APP
from src.ui import (
    apply_styles,
    render_header,
    render_footer,
    render_file_loader,
    render_preview,
    render_classification,
    render_hackathon_classification,
    render_analysis,
    render_ml_classifier,
    render_railway_analyzer,
    render_bim_analyzer,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    st.set_page_config(
        page_title=APP.TITLE,
        page_icon=APP.FAVICON,
        layout="wide",
        initial_sidebar_state="collapsed"  # Ukryj sidebar
    )

    # Minimalne style
    apply_styles()

    # Nagłówek
    render_header()

    # Zakładki
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📁 Wczytaj plik",
        "👁️ Podgląd",
        "🚀 Hackathon",
        "📐 Analiza",
        "🤖 ML",
        "🚂 Railway",
        "🏗️ BIM",
        "🎯 Klasyfikacja"
    ])

    with tab1:
        render_file_loader()

    with tab2:
        render_preview()

    with tab3:
        render_hackathon_classification()

    with tab4:
        render_analysis()

    with tab5:
        render_ml_classifier()

    with tab6:
        render_railway_analyzer()

    with tab7:
        render_bim_analyzer()

    with tab8:
        render_classification()

    # Stopka
    render_footer()


if __name__ == "__main__":
    main()
