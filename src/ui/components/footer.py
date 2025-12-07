"""
Komponent stopki aplikacji Chmura+
"""

import streamlit as st


def render_footer():
    """Renderuje stopke aplikacji"""

    st.markdown("---")

    # Hackathon branding
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
                padding: 20px 30px; border-radius: 10px; margin-top: 20px;'>
        <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;'>
            <div>
                <div style='color: white; font-size: 18px; font-weight: 600;'>Chmura+</div>
                <div style='color: #90caf9; font-size: 13px;'>Klasyfikator Chmur Punktow v2.0</div>
            </div>
            <div style='text-align: center;'>
                <div style='color: #ffd54f; font-size: 14px; font-weight: 600;'>HackNation 2025</div>
                <div style='color: #90caf9; font-size: 12px;'>Chmura pod Kontrola</div>
            </div>
            <div style='text-align: right;'>
                <div style='color: white; font-size: 14px;'>Centralny Port Komunikacyjny</div>
                <div style='color: #90caf9; font-size: 12px;'>19+ klas ASPRS</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tech stack - subtle
    st.markdown("""
    <div style='text-align: center; padding: 15px 0 5px 0; color: #9e9e9e; font-size: 11px;'>
        Python • NumPy • SciPy • Open3D • CSF • Plotly • laspy • Streamlit
    </div>
    """, unsafe_allow_html=True)
