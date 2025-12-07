"""
Minimalne style CSS
"""

import streamlit as st


def apply_styles():
    """Minimalne style"""

    st.markdown("""
    <style>
        /* Ukryj sidebar całkowicie */
        [data-testid="stSidebar"] {
            display: none;
        }

        /* Główny przycisk */
        .stButton > button[kind="primary"] {
            background-color: #0A1E42;
            border-color: #0A1E42;
        }

        /* Metryki - granatowy kolor */
        [data-testid="stMetricValue"] {
            color: #0A1E42;
            font-weight: 700;
        }
    </style>
    """, unsafe_allow_html=True)
