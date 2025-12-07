"""
Komponent analizy infrastruktury - HackNation 2025 CPK

Narzedzia do analizy:
- Skrajnia kolejowa
- Profile terenu (DTM/DSM)
- Przekroje poprzeczne
- Obliczenia objetosciowe
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict
import json

from ...v2 import LASLoader


def render_analysis():
    """Glowny komponent analizy"""

    st.markdown("""
    <div style='background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
                padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: white; margin: 0;'>Analiza Infrastruktury CPK</h2>
        <p style='color: #90caf9; margin: 5px 0 0 0;'>
            Narzedzia do planowania infrastruktury kolejowej i lotniczej
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sprawdz czy mamy dane
    if 'hack_full_results' not in st.session_state:
        st.warning("Najpierw wykonaj klasyfikacje w zakladce 'Hackathon'")
        _show_analysis_info()
        return

    results = st.session_state['hack_full_results']
    coords = results.get('coords')
    classification = results.get('classification')

    if coords is None or classification is None:
        st.warning("Brak danych do analizy (tryb batch nie zapisuje danych w pamieci)")
        return

    # Menu analiz
    analysis_type = st.selectbox(
        "Wybierz analize:",
        [
            "Skrajnia kolejowa",
            "Model terenu (DTM/DSM)",
            "Obliczenia objetosciowe",
            "Statystyki wysokosciowe"
        ]
    )

    st.markdown("---")

    if analysis_type == "Skrajnia kolejowa":
        _render_clearance_analysis(coords, classification)
    elif analysis_type == "Model terenu (DTM/DSM)":
        _render_terrain_analysis(coords, classification)
    elif analysis_type == "Obliczenia objetosciowe":
        _render_volume_analysis(coords, classification)
    else:
        _render_height_statistics(coords, classification)


def _show_analysis_info():
    """Informacje o dostepnych analizach"""
    st.markdown("### Dostepne analizy:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Skrajnia kolejowa**
        - Wykrywanie naruszen profilu skrajni
        - Analiza przeszkod wzdluz torow
        - Raport z lokalizacja naruszen

        **Model terenu (DTM/DSM)**
        - Generowanie DTM (teren)
        - Generowanie DSM (powierzchnia)
        - CHM (wysokosc roslinnosci)
        """)

    with col2:
        st.markdown("""
        **Obliczenia objetosciowe**
        - Wykopy i nasypy
        - Bilans mas ziemnych
        - Szacowanie kosztow

        **Statystyki wysokosciowe**
        - Rozklad wysokosci per klasa
        - Profile wysokosciowe
        - Analiza roslinnosci
        """)


def _render_clearance_analysis(coords: np.ndarray, classification: np.ndarray):
    """Analiza skrajni kolejowej"""
    st.subheader("Analiza skrajni kolejowej")

    # Sprawdz czy sa tory
    track_mask = classification == 18
    if not track_mask.any():
        st.warning("Nie wykryto torow kolejowych (klasa 18). Skrajnia wymaga punktow torow.")
        return

    st.info(f"Wykryto {track_mask.sum():,} punktow torow kolejowych")

    if st.button("Uruchom analize skrajni", type="primary"):
        with st.spinner("Analizuje skrajnie..."):
            from ...v2.analysis import RailwayClearanceAnalyzer

            analyzer = RailwayClearanceAnalyzer(coords, classification)
            violations = analyzer.analyze()
            report = analyzer.generate_report()

            # Wyswietl wyniki
            st.markdown("### Wyniki analizy")

            # Metryki
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                status_color = "green" if report['status'] == 'OK' else "red"
                st.markdown(f"**Status:** :{status_color}[{report['status']}]")
            with col2:
                st.metric("Naruszenia", report['summary']['total'])
            with col3:
                st.metric("Krytyczne", report['summary']['critical'])
            with col4:
                st.metric("Ostrzezenia", report['summary']['warning'])

            if violations:
                # Wizualizacja naruszen
                st.markdown("### Mapa naruszen")

                viol_coords, severities = analyzer.get_violation_points()

                # Plot
                fig = go.Figure()

                # Tory
                track_pts = coords[track_mask]
                if len(track_pts) > 5000:
                    idx = np.random.choice(len(track_pts), 5000, replace=False)
                    track_pts = track_pts[idx]

                fig.add_trace(go.Scatter3d(
                    x=track_pts[:, 0],
                    y=track_pts[:, 1],
                    z=track_pts[:, 2],
                    mode='markers',
                    marker=dict(size=2, color='#6D4C41'),
                    name='Tory'
                ))

                # Naruszenia
                colors = ['#FFC107', '#FF9800', '#F44336']  # minor, warning, critical
                severity_names = ['Drobne', 'Ostrzezenie', 'Krytyczne']

                for sev_val, sev_name, color in zip([1, 2, 3], severity_names, colors):
                    mask = severities == sev_val
                    if mask.any():
                        fig.add_trace(go.Scatter3d(
                            x=viol_coords[mask, 0],
                            y=viol_coords[mask, 1],
                            z=viol_coords[mask, 2],
                            mode='markers',
                            marker=dict(size=5, color=color),
                            name=sev_name
                        ))

                fig.update_layout(
                    scene=dict(aspectmode='data'),
                    height=500,
                    title="Naruszenia skrajni kolejowej"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Tabela najgorszych
                st.markdown("### Najgorsze naruszenia")
                worst = report.get('worst_violations', [])[:5]
                for i, v in enumerate(worst):
                    st.markdown(f"""
                    **#{i+1}** - {v['severity'].upper()}
                    - Pozycja: ({v['position'][0]:.1f}, {v['position'][1]:.1f}, {v['position'][2]:.1f})
                    - Naruszenie: **{v['violation_distance_m']*100:.0f} cm**
                    - Wysokosc nad szyna: {v['height_above_rail_m']:.2f} m
                    """)

                # Export
                st.download_button(
                    "Pobierz raport JSON",
                    json.dumps(report, indent=2, ensure_ascii=False),
                    "skrajnia_raport.json",
                    mime="application/json"
                )
            else:
                st.success("Brak naruszen skrajni!")


def _render_terrain_analysis(coords: np.ndarray, classification: np.ndarray):
    """Analiza terenu DTM/DSM"""
    st.subheader("Model terenu (DTM/DSM)")

    col1, col2 = st.columns(2)
    with col1:
        resolution = st.slider("Rozdzielczosc [m]", 0.5, 5.0, 1.0, 0.5)
    with col2:
        model_type = st.radio("Typ modelu", ["DTM", "DSM", "CHM"], horizontal=True)

    if st.button("Generuj model", type="primary"):
        with st.spinner(f"Generowanie {model_type}..."):
            from ...v2.analysis import TerrainAnalyzer

            analyzer = TerrainAnalyzer(coords, classification)

            if model_type == "DTM":
                model = analyzer.generate_dtm(resolution=resolution)
                title = "Digital Terrain Model (DTM)"
                colorscale = "Earth"
            elif model_type == "DSM":
                model = analyzer.generate_dsm(resolution=resolution)
                title = "Digital Surface Model (DSM)"
                colorscale = "Viridis"
            else:
                dtm = analyzer.generate_dtm(resolution=resolution)
                dsm = analyzer.generate_dsm(resolution=resolution)
                model = analyzer.calculate_chm(dtm, dsm)
                title = "Canopy Height Model (CHM)"
                colorscale = "Greens"

            # Wizualizacja
            fig = go.Figure(data=[go.Surface(
                x=model.grid_x,
                y=model.grid_y,
                z=model.heights,
                colorscale=colorscale,
                colorbar=dict(title="Wysokosc [m]")
            )])

            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="X [m]",
                    yaxis_title="Y [m]",
                    zaxis_title="Z [m]",
                    aspectmode='data'
                ),
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

            # Statystyki
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Min", f"{np.nanmin(model.heights):.1f} m")
            with col2:
                st.metric("Max", f"{np.nanmax(model.heights):.1f} m")
            with col3:
                st.metric("Srednia", f"{np.nanmean(model.heights):.1f} m")
            with col4:
                st.metric("Rozdzielczosc", f"{model.resolution} m")


def _render_volume_analysis(coords: np.ndarray, classification: np.ndarray):
    """Obliczenia objetosciowe"""
    st.subheader("Obliczenia objetosciowe")

    st.info("Podaj os trasy (punkty poczatkowy i koncowy) oraz wysokosc projektowa.")

    # Automatycznie wykryj granice
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    z_mean = coords[:, 2].mean()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Punkt poczatkowy:**")
        x1 = st.number_input("X1", value=float(x_min + (x_max-x_min)*0.1), key="x1")
        y1 = st.number_input("Y1", value=float((y_min+y_max)/2), key="y1")

    with col2:
        st.markdown("**Punkt koncowy:**")
        x2 = st.number_input("X2", value=float(x_max - (x_max-x_min)*0.1), key="x2")
        y2 = st.number_input("Y2", value=float((y_min+y_max)/2), key="y2")

    design_height = st.number_input(
        "Wysokosc projektowa [m]",
        value=float(z_mean),
        help="Docelowa wysokosc niwelety"
    )

    corridor_width = st.slider("Szerokosc korytarza [m]", 5, 50, 20)

    if st.button("Oblicz objetosc", type="primary"):
        with st.spinner("Obliczam..."):
            from ...v2.analysis import VolumeCalculator

            calc = VolumeCalculator(coords, classification)

            axis = np.array([[x1, y1], [x2, y2]])
            design_heights = np.array([design_height, design_height])

            result = calc.calculate_corridor_volume(axis, design_heights, corridor_width)
            report = calc.generate_report(result)

            # Wyniki
            st.markdown("### Wyniki")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Wykop", f"{report['summary']['cut_volume_m3']:,.0f} m³")
            with col2:
                st.metric("Nasyp", f"{report['summary']['fill_volume_m3']:,.0f} m³")
            with col3:
                balance_color = "green" if report['summary']['net_volume_m3'] >= 0 else "red"
                st.metric(
                    f"Bilans ({report['summary']['balance']})",
                    f"{report['summary']['balance_volume_m3']:,.0f} m³"
                )

            # Dodatkowe info
            st.markdown("### Szczegoly")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **Roboty ziemne:**
                - Sredni wykop: {report['depths']['avg_cut_m']:.2f} m
                - Max wykop: {report['depths']['max_cut_m']:.2f} m
                - Sredni nasyp: {report['depths']['avg_fill_m']:.2f} m
                - Max nasyp: {report['depths']['max_fill_m']:.2f} m
                """)

            with col2:
                st.markdown(f"""
                **Roslinnosc:**
                - Do usuniecia: {report['vegetation']['volume_m3']:,.0f} m³

                **Powierzchnia:**
                - {report['area']['total_m2']:,.0f} m² ({report['area']['total_ha']:.2f} ha)
                """)

            # Koszty
            st.markdown("### Szacunkowe koszty")
            costs = report['costs_estimate']
            st.markdown(f"""
            | Pozycja | Koszt |
            |---------|-------|
            | Wykopy | {costs['cut_cost_pln']:,.0f} PLN |
            | Nasypy | {costs['fill_cost_pln']:,.0f} PLN |
            | Usuniecie roslinnosci | {costs['veg_removal_pln']:,.0f} PLN |
            | **RAZEM** | **{costs['total_estimate_pln']:,.0f} PLN** |
            """)

            st.caption("*Koszty orientacyjne, wymagaja weryfikacji kosztorysowej*")

            # Export
            st.download_button(
                "Pobierz raport JSON",
                json.dumps(report, indent=2, ensure_ascii=False),
                "objetosc_raport.json",
                mime="application/json"
            )


def _render_height_statistics(coords: np.ndarray, classification: np.ndarray):
    """Statystyki wysokosciowe"""
    st.subheader("Statystyki wysokosciowe")

    # Rozklad wysokosci
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Rozklad wysokosci", "Wysokosc per klasa"])

    # Histogram
    fig.add_trace(go.Histogram(
        x=coords[:, 2],
        nbinsx=50,
        marker_color='#1976D2',
        name="Wszystkie"
    ), row=1, col=1)

    # Box plot per klasa
    CLASS_NAMES = {
        2: "Grunt", 3: "Rosl. niska", 4: "Rosl. srednia", 5: "Rosl. wysoka",
        6: "Budynek", 18: "Tory", 19: "Linie", 20: "Slupy"
    }

    unique_classes = sorted(np.unique(classification))
    for cls in unique_classes:
        if cls in CLASS_NAMES:
            mask = classification == cls
            fig.add_trace(go.Box(
                y=coords[mask, 2],
                name=CLASS_NAMES[cls],
                boxmean=True
            ), row=1, col=2)

    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Tabela statystyk
    st.markdown("### Statystyki per klasa")

    stats_data = []
    for cls in unique_classes:
        mask = classification == cls
        z = coords[mask, 2]
        if len(z) > 0:
            stats_data.append({
                'Klasa': cls,
                'Nazwa': CLASS_NAMES.get(cls, f'Klasa {cls}'),
                'Punktow': f"{len(z):,}",
                'Min [m]': f"{z.min():.2f}",
                'Max [m]': f"{z.max():.2f}",
                'Srednia [m]': f"{z.mean():.2f}",
                'Std [m]': f"{z.std():.2f}"
            })

    import pandas as pd
    df = pd.DataFrame(stats_data)
    st.dataframe(df, hide_index=True, use_container_width=True)
