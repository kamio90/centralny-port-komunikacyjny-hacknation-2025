"""
Komponent nagłówka - prosty i stabilny
"""

import streamlit as st
from ...config import APP


def render_header():
    """Prosty naglowek z lepszym UX"""

    # Logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image(APP.LOGO, use_container_width=True)
        except:
            st.title("Chmura+")

    # Hero section
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
                padding: 25px 30px; border-radius: 12px; margin: 10px 0 20px 0;
                box-shadow: 0 4px 20px rgba(26, 35, 126, 0.3);'>
        <h2 style='color: white; margin: 0 0 10px 0; font-weight: 600;'>
            Automatyczna klasyfikacja chmur punktow LiDAR
        </h2>
        <p style='color: #90caf9; margin: 0; font-size: 16px;'>
            Profesjonalne narzedzie do analizy danych dla infrastruktury CPK
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Key features - compact
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 10px;'>
            <div style='font-size: 28px;'>🎯</div>
            <div style='font-weight: 600; color: #1976D2;'>19+ klas</div>
            <div style='font-size: 11px; color: #666;'>ASPRS + CPK</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 10px;'>
            <div style='font-size: 28px;'>🚄</div>
            <div style='font-weight: 600; color: #1976D2;'>Kolej</div>
            <div style='font-size: 11px; color: #666;'>Tory, slupy, perony</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 10px;'>
            <div style='font-size: 28px;'>🏢</div>
            <div style='font-weight: 600; color: #1976D2;'>Budynki</div>
            <div style='font-size: 11px; color: #666;'>Dachy, sciany</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style='text-align: center; padding: 10px;'>
            <div style='font-size: 28px;'>🌉</div>
            <div style='font-weight: 600; color: #1976D2;'>Mosty</div>
            <div style='font-size: 11px; color: #666;'>Woda, drogi</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown("""
        <div style='text-align: center; padding: 10px;'>
            <div style='font-size: 28px;'>📊</div>
            <div style='font-weight: 600; color: #1976D2;'>Export</div>
            <div style='font-size: 11px; color: #666;'>LAS, IFC, GeoJSON</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
