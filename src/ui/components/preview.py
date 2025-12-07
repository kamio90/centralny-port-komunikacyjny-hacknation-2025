"""
Podgląd chmury punktów - prosty i stabilny
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any
import time
from pathlib import Path

from ...v2 import LASLoader
from ...v2.core import PointCloudSampler


DEFAULT_TARGET_POINTS = 50_000
MAX_DISPLAY_POINTS = 300_000


def render_preview() -> None:
    """Prosty podgląd"""

    st.subheader("👁️ Podgląd chmury punktów")

    if 'input_file' not in st.session_state:
        st.warning("⚠️ Najpierw wczytaj plik")
        return

    input_path = st.session_state['input_file']
    file_info = st.session_state['file_info']

    st.success(f"Plik: **{Path(input_path).name}** ({file_info['n_points']:,} punktów)")

    # Kontrolki
    col1, col2 = st.columns(2)

    with col1:
        target_points = st.slider(
            "Punkty w podglądzie",
            min_value=10_000,
            max_value=MAX_DISPLAY_POINTS,
            value=DEFAULT_TARGET_POINTS,
            step=10_000
        )

    with col2:
        color_mode = st.selectbox(
            "Kolorowanie",
            ["Wysokość Z", "RGB", "Intensywność"]
        )

    if st.button("🔄 Generuj podgląd", type="primary"):
        _generate_preview(input_path, target_points, color_mode)

    elif 'preview_coords' in st.session_state:
        _show_preview(color_mode)


def _generate_preview(input_path: str, target_points: int, color_mode: str):
    """Generuje podgląd"""

    with st.spinner("Wczytywanie..."):
        loader = LASLoader(input_path)
        data = loader.load()

    with st.spinner("Sampling..."):
        sampler = PointCloudSampler(
            coords=data['coords'],
            colors=data['colors'],
            intensity=data['intensity']
        )
        result = sampler.adaptive_sample(target_points=target_points)

    # Cache
    st.session_state['preview_coords'] = result.coords
    st.session_state['preview_colors'] = result.colors
    st.session_state['preview_intensity'] = result.intensity
    st.session_state['preview_stats'] = result.stats

    _show_preview(color_mode)


def _show_preview(color_mode: str):
    """Wyświetla podgląd"""

    coords = st.session_state['preview_coords']
    colors = st.session_state['preview_colors']
    intensity = st.session_state['preview_intensity']
    stats = st.session_state['preview_stats']

    # Kolory
    if color_mode == "RGB" and colors is not None:
        point_colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in colors]
        colorscale = None
    elif color_mode == "Intensywność" and intensity is not None:
        point_colors = intensity
        colorscale = 'Viridis'
    else:
        point_colors = coords[:, 2]
        colorscale = 'Turbo'

    # Wykres
    n_pts = len(coords)
    marker_size = 1.0 if n_pts > 100_000 else 1.5

    fig = go.Figure(data=[go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers',
        marker=dict(
            size=marker_size,
            color=point_colors,
            colorscale=colorscale,
            opacity=0.7
        ),
        hoverinfo='skip'  # Wyłączone dla wydajności
    )])

    fig.update_layout(
        scene=dict(aspectmode='data'),
        height=500,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Statystyki
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Punkty (oryg.)", f"{stats['original_points']:,}")
    with col2:
        st.metric("Punkty (podgląd)", f"{stats['sampled_points']:,}")
    with col3:
        st.metric("Gęstość", f"{stats['density_original']:.0f} pkt/m²")
    with col4:
        st.metric("Metoda", stats['method'])
