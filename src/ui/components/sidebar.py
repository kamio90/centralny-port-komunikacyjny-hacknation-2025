"""
Komponent sidebar - prosty i stabilny
"""

import streamlit as st
from ...v2 import ClassifierRegistry


def render_sidebar() -> int:
    """Prosty sidebar"""

    st.sidebar.title("☁️ Chmura+")
    st.sidebar.caption("CPK Point Cloud Classifier")

    st.sidebar.markdown("---")

    # Ustawienia
    st.sidebar.subheader("⚙️ Ustawienia")

    processing_mode = st.sidebar.selectbox(
        "Tryb przetwarzania",
        ["Standardowy", "Szybki"],
        help="Szybki = mniej dokładny ale szybszy"
    )

    n_threads = 1  # Single thread for stability

    st.sidebar.markdown("---")

    # Klasyfikatory
    st.sidebar.subheader("📊 Dostępne klasy")

    all_classifiers = ClassifierRegistry.get_all()
    st.sidebar.info(f"**{len(all_classifiers)}** klas ASPRS")

    with st.sidebar.expander("Pokaż klasy"):
        for class_id, classifier_class in sorted(all_classifiers.items()):
            classifier = classifier_class()
            st.caption(f"[{class_id}] {classifier.class_name}")

    st.sidebar.markdown("---")

    # Info
    st.sidebar.subheader("ℹ️ Info")
    st.sidebar.caption("**Formaty:** LAS, LAZ")
    st.sidebar.caption("**Algorytmy:** CSF, HAG, RANSAC")
    st.sidebar.caption("**Eksport:** LAS, JSON, TXT, IFC")

    st.sidebar.markdown("---")
    st.sidebar.caption("HackNation 2025 🏆")

    return n_threads
