"""
Hackathon Klasyfikacja - prosty i stabilny
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import List
import time
from pathlib import Path

from ...v2 import LASLoader
from ...v2.core import GridManager
from ...v2.pipeline import ProfessionalPipeline, PipelineConfig


# Kolory klas
ASPRS_COLORS = {
    1: "#9E9E9E", 2: "#8D6E63", 3: "#AED581", 4: "#66BB6A", 5: "#2E7D32",
    6: "#D7CCC8", 7: "#F44336", 18: "#6D4C41", 19: "#FDD835", 20: "#546E7A",
    21: "#78909C", 30: "#455A64", 32: "#B0BEC5", 35: "#FF5722", 36: "#90A4AE",
    40: "#BCAAA4", 41: "#A1887F"
}


def render_hackathon_classification(n_threads: int = 1) -> None:
    """Hackathon - klasyfikacja fragmentów"""

    st.subheader("🚀 Hackathon - Klasyfikacja fragmentów")

    if 'input_file' not in st.session_state:
        st.warning("⚠️ Najpierw wczytaj plik")
        return

    input_path = st.session_state['input_file']
    file_info = st.session_state['file_info']

    st.success(f"Plik: **{Path(input_path).name}** ({file_info['n_points']:,} punktów)")

    st.markdown("---")

    # Krok 1: Generuj siatkę
    st.subheader("📐 Krok 1: Siatka")

    col1, col2 = st.columns(2)
    with col1:
        target_pts = st.slider("Punktów/kwadrat", 100_000, 2_000_000, 500_000, 50_000)
    with col2:
        if 'hack_grid' in st.session_state:
            stats = st.session_state['hack_grid'].get_statistics()
            st.metric("Siatka", stats['grid_dimensions'])

    if st.button("🔄 Generuj siatkę", type="primary"):
        with st.spinner("Generowanie..."):
            if 'hack_data' not in st.session_state:
                loader = LASLoader(input_path)
                st.session_state['hack_data'] = loader.load()

            data = st.session_state['hack_data']
            grid = GridManager(data['coords'], target_points_per_square=target_pts)
            grid.create_grid()
            st.session_state['hack_grid'] = grid
        st.rerun()

    if 'hack_grid' not in st.session_state:
        return

    # Krok 2: Wizualizacja
    st.markdown("---")
    st.subheader("🗺️ Krok 2: Wybierz kwadraty")

    grid = st.session_state['hack_grid']
    squares = grid.get_squares()

    # Mapa 2D
    fig = go.Figure()
    for sq in squares:
        fig.add_trace(go.Scatter(
            x=[sq.bounds[0], sq.bounds[2], sq.bounds[2], sq.bounds[0], sq.bounds[0]],
            y=[sq.bounds[1], sq.bounds[1], sq.bounds[3], sq.bounds[3], sq.bounds[1]],
            mode='lines+text',
            text=[str(sq.id), '', '', '', ''],
            textposition='middle center',
            line=dict(color='blue', width=1),
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.2)',
            name=f"Kwadrat {sq.id}"
        ))

    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="X (m)",
        yaxis_title="Y (m)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Wybór kwadratów
    all_ids = [sq.id for sq in squares]
    selected = st.multiselect("Wybierz kwadraty:", all_ids, default=all_ids[:min(3, len(all_ids))])

    if not selected:
        st.warning("Wybierz co najmniej 1 kwadrat")
        return

    # Krok 3: Klasyfikacja
    st.markdown("---")
    st.subheader("🎯 Krok 3: Klasyfikacja")

    est_time = len(selected) * target_pts * 0.00005
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Wybrane kwadraty", len(selected))
    with col2:
        st.metric("Est. czas", f"~{int(est_time)}s")

    if st.button("🚀 KLASYFIKUJ WYBRANE", type="primary", use_container_width=True):
        _run_hack_classification(grid, selected)

    # Wyniki
    if 'hack_results' in st.session_state:
        _show_hack_results()


def _run_hack_classification(grid, selected_ids: List[int]):
    """Klasyfikacja wybranych kwadratów"""

    progress = st.progress(0)
    status = st.empty()
    start = time.time()

    results = []
    data = st.session_state['hack_data']

    for i, sq_id in enumerate(selected_ids):
        status.info(f"🔄 Klasyfikacja kwadratu {sq_id}...")
        progress.progress((i + 1) / (len(selected_ids) + 1))

        # Pobierz punkty
        sq = grid.get_square(sq_id)
        mask = sq.get_point_mask(data['coords'])
        coords = data['coords'][mask]
        colors = data['colors'][mask] if data['colors'] is not None else None
        intensity = data['intensity'][mask] if data['intensity'] is not None else None

        # Klasyfikacja
        config = PipelineConfig(
            detect_noise=True,
            classify_ground=True,
            classify_vegetation=True,
            detect_buildings=True,
            detect_infrastructure=True
        )

        pipeline = ProfessionalPipeline(coords, colors, intensity, config)
        classification, stats = pipeline.run()

        results.append({
            'square_id': sq_id,
            'coords': coords,
            'classification': classification,
            'stats': stats,
            'n_points': len(coords)
        })

    progress.progress(100)
    elapsed = time.time() - start
    status.success(f"✅ Gotowe! Czas: {elapsed:.1f}s")

    st.session_state['hack_results'] = results
    st.session_state['hack_elapsed'] = elapsed
    st.rerun()


def _show_hack_results():
    """Wyświetla wyniki"""

    results = st.session_state['hack_results']
    elapsed = st.session_state['hack_elapsed']

    st.markdown("---")
    st.subheader("📊 Wyniki")

    total_pts = sum(r['n_points'] for r in results)
    st.metric("Przetworzonych punktów", f"{total_pts:,}")

    # Wizualizacja
    with st.expander("🎨 Wizualizacja 3D", expanded=True):
        max_pts = st.slider("Max punktów", 10_000, 100_000, 50_000, 10_000)

        if st.button("🔄 Generuj"):
            all_coords = []
            all_colors = []

            for r in results:
                coords = r['coords']
                classification = r['classification']

                # Sampling
                if len(coords) > max_pts // len(results):
                    idx = np.random.choice(len(coords), max_pts // len(results), replace=False)
                    coords = coords[idx]
                    classification = classification[idx]

                all_coords.append(coords)
                for cls in classification:
                    all_colors.append(ASPRS_COLORS.get(cls, "#888"))

            coords = np.vstack(all_coords)

            fig = go.Figure(data=[go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                mode='markers',
                marker=dict(size=1, color=all_colors, opacity=0.7),
                hoverinfo='skip'
            )])
            fig.update_layout(scene=dict(aspectmode='data'), height=500, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
