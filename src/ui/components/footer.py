"""
Komponent stopki aplikacji Chmura+
"""

import streamlit as st


def render_footer():
    """Renderuje stopkę aplikacji"""

    st.markdown("---")

    # Prostszy footer bez skomplikowanego HTML
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("🏛️ Centralny Port Komunikacyjny")

    with col2:
        st.caption("🏆 HackNation 2025")

    with col3:
        st.caption("🎯 45+ klas ASPRS")

    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; color: #64748B; font-size: 0.85rem;">
        <strong style="color: #0A1E42;">☁️ Chmura+</strong> | Klasyfikator Chmur Punktów v2.0<br>
        <small>Python • NumPy • SciPy • Open3D • Plotly • laspy</small>
    </div>
    """, unsafe_allow_html=True)
