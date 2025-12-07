"""
BIM Analyzer UI - Interfejs do analizy BIM

Funkcjonalnosci:
- Ekstrakcja budynkow
- Analiza geometrii
- Klasyfikacja LOD
- Wykrywanie kolizji
- Eksport IFC

HackNation 2025 - CPK Chmura+
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict
from pathlib import Path
import time

from ...config import PATHS


def render_bim_analyzer():
    """Glowny komponent BIM Analyzer"""

    st.markdown("""
    <div style='background: linear-gradient(135deg, #00695c 0%, #004d40 100%);
                padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: white; margin: 0;'>BIM Analyzer</h2>
        <p style='color: #b2dfdb; margin: 5px 0 0 0;'>
            Building Information Modeling - analiza i eksport chmur punktow
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Menu
    mode = st.radio(
        "Tryb:",
        ["Budynki", "Geometria 3D", "Klasyfikacja LOD", "Kolizje", "Eksport BIM"],
        horizontal=True
    )

    st.markdown("---")

    if mode == "Budynki":
        _render_building_mode()
    elif mode == "Geometria 3D":
        _render_geometry_mode()
    elif mode == "Klasyfikacja LOD":
        _render_lod_mode()
    elif mode == "Kolizje":
        _render_clash_mode()
    else:
        _render_export_mode()


def _get_data():
    """Pobierz dane z session state"""
    if 'hack_full_results' in st.session_state:
        results = st.session_state['hack_full_results']
        return {
            'coords': results.get('coords'),
            'classification': results.get('classification'),
            'colors': results.get('colors'),
            'intensity': results.get('intensity')
        }
    if 'ml_coords' in st.session_state:
        return {
            'coords': st.session_state.get('ml_coords'),
            'classification': st.session_state.get('ml_classification'),
            'colors': None,
            'intensity': None
        }
    return None


def _render_building_mode():
    """Tryb ekstrakcji budynkow"""
    st.subheader("Ekstrakcja budynkow")

    st.info("""
    **Ekstrakcja budynkow** z chmury punktow obejmuje:
    - Segmentacje punktow klasy 6 (budynki)
    - Ekstrakcje obrysow (footprint)
    - Analize dachow (typ, nachylenie)
    - Obliczanie wysokosci i objetosci
    """)

    data = _get_data()
    if data is None or data['coords'] is None:
        st.warning("Wczytaj i sklasyfikuj dane")
        return

    coords = data['coords']
    classification = data['classification']

    # Statystyki
    building_mask = classification == 6
    building_count = building_mask.sum()

    st.markdown(f"**Punkty budynkow (klasa 6):** {building_count:,}")

    if building_count < 100:
        st.warning("Za malo punktow budynkow. Wykonaj klasyfikacje.")
        return

    # Konfiguracja
    col1, col2 = st.columns(2)
    with col1:
        min_area = st.slider("Min powierzchnia [m2]", 10, 100, 20, 5)
        min_height = st.slider("Min wysokosc [m]", 1.0, 5.0, 2.5, 0.5)
    with col2:
        cluster_eps = st.slider("Odleglosc klastrowania [m]", 1.0, 5.0, 2.0, 0.5)

    if st.button("EKSTRAHUJ BUDYNKI", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        try:
            from ...v2.bim import BuildingExtractor

            status.info("Inicjalizacja ekstraktora...")
            progress.progress(10)

            extractor = BuildingExtractor(
                coords, classification,
                min_building_area=min_area,
                min_building_height=min_height
            )

            status.info("Ekstrakcja budynkow...")
            progress.progress(30)

            buildings = extractor.extract()
            progress.progress(80)

            stats = extractor.get_statistics(buildings)
            progress.progress(100)
            status.success(f"Wykryto {len(buildings)} budynkow!")

            # Wyniki
            _display_building_results(buildings, stats)

            st.session_state['bim_buildings'] = buildings

        except Exception as e:
            status.error(f"Blad: {e}")
            import traceback
            st.code(traceback.format_exc())


def _display_building_results(buildings, stats):
    """Wyswietl wyniki ekstrakcji budynkow"""
    st.markdown("### Wyniki")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Budynkow", stats['total_count'])
    with col2:
        st.metric("Laczna pow.", f"{stats['total_area_m2']:.0f} m2")
    with col3:
        st.metric("Srednia wys.", f"{stats['avg_height_m']:.1f} m")
    with col4:
        st.metric("Laczna obj.", f"{stats['total_volume_m3']:.0f} m3")

    # Typy dachow
    if stats.get('roof_types'):
        st.markdown("### Typy dachow")
        fig = go.Figure(data=[go.Pie(
            labels=list(stats['roof_types'].keys()),
            values=list(stats['roof_types'].values()),
            hole=0.4
        )])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Histogram wysokosci
    if buildings:
        heights = [b.height for b in buildings]
        fig2 = go.Figure(data=[go.Histogram(
            x=heights,
            marker_color='#00695c',
            nbinsx=20
        )])
        fig2.update_layout(
            height=300,
            title="Rozklad wysokosci budynkow",
            xaxis_title="Wysokość [m]",
            yaxis_title="Liczba"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Tabela budynkow
    with st.expander(f"Lista budynkow ({len(buildings)})"):
        for b in buildings[:20]:
            st.markdown(f"""
            **Budynek #{b.id}:** {b.footprint.area_m2:.0f} m2,
            wys: {b.height:.1f}m,
            dach: {b.roof_type.value},
            obj: {b.volume_m3:.0f} m3,
            kondygnacji: ~{b.floor_count_estimate}
            """)


def _render_geometry_mode():
    """Tryb analizy geometrii"""
    st.subheader("Analiza geometrii 3D")

    st.info("""
    **Analiza geometrii** oblicza:
    - Bounding box (AABB i OBB)
    - Convex hull 3D
    - Glowne kierunki (PCA)
    - Metryki ksztaltu (zwartość, wydłużenie, planarność)
    """)

    data = _get_data()
    if data is None or data['coords'] is None:
        st.warning("Wczytaj dane")
        return

    coords = data['coords']
    classification = data['classification']

    # Wybor elementu do analizy
    analysis_target = st.selectbox(
        "Analizuj:",
        ["Cala chmura", "Budynki (klasa 6)", "Infrastruktura (18-20)", "Wybrany budynek"]
    )

    # Filtruj punkty
    if analysis_target == "Cala chmura":
        target_coords = coords
    elif analysis_target == "Budynki (klasa 6)":
        mask = classification == 6
        target_coords = coords[mask]
    elif analysis_target == "Infrastruktura (18-20)":
        mask = np.isin(classification, [18, 19, 20])
        target_coords = coords[mask]
    else:
        if 'bim_buildings' not in st.session_state:
            st.warning("Najpierw ekstrahuj budynki")
            return
        buildings = st.session_state['bim_buildings']
        if not buildings:
            st.warning("Brak budynkow")
            return
        building_idx = st.selectbox(
            "Wybierz budynek:",
            range(len(buildings)),
            format_func=lambda i: f"Budynek #{buildings[i].id} ({buildings[i].footprint.area_m2:.0f} m2)"
        )
        target_coords = buildings[building_idx].points

    st.markdown(f"**Punktow do analizy:** {len(target_coords):,}")

    if len(target_coords) < 10:
        st.warning("Za malo punktow")
        return

    # Ogranicz dla wydajnosci
    if len(target_coords) > 50000:
        sample_idx = np.random.choice(len(target_coords), 50000, replace=False)
        target_coords = target_coords[sample_idx]
        st.info(f"Probkowanie do 50,000 punktow dla wydajnosci")

    if st.button("ANALIZUJ GEOMETRIE", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        try:
            from ...v2.bim import GeometryAnalyzer

            status.info("Analiza geometrii...")
            progress.progress(30)

            analyzer = GeometryAnalyzer(target_coords)
            metrics = analyzer.analyze()
            progress.progress(100)
            status.success("Analiza zakonczona!")

            # Wyniki
            _display_geometry_results(metrics, target_coords)

            st.session_state['bim_geometry'] = metrics

        except Exception as e:
            status.error(f"Blad: {e}")
            import traceback
            st.code(traceback.format_exc())


def _display_geometry_results(metrics, coords):
    """Wyswietl wyniki analizy geometrii"""
    st.markdown("### Metryki geometryczne")

    # AABB
    st.markdown("#### Bounding Box (AABB)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Szerokosc (X)", f"{metrics.aabb.dimensions[0]:.1f} m")
    with col2:
        st.metric("Glebokosc (Y)", f"{metrics.aabb.dimensions[1]:.1f} m")
    with col3:
        st.metric("Wysokosc (Z)", f"{metrics.aabb.dimensions[2]:.1f} m")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Objetosc AABB", f"{metrics.aabb.volume:.0f} m3")
    with col2:
        st.metric("Powierzchnia", f"{metrics.aabb.surface_area:.0f} m2")

    # Metryki ksztaltu
    st.markdown("#### Metryki ksztaltu")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Zwartość", f"{metrics.compactness:.2f}")
    with col2:
        st.metric("Wydłużenie", f"{metrics.elongation:.2f}")
    with col3:
        st.metric("Planarność", f"{metrics.planarity:.2f}")
    with col4:
        st.metric("Sferyczność", f"{metrics.sphericity:.2f}")

    # Convex Hull
    if metrics.convex_hull:
        st.markdown("#### Convex Hull")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Objetosc Hull", f"{metrics.convex_hull.volume:.0f} m3")
        with col2:
            st.metric("Powierzchnia Hull", f"{metrics.convex_hull.surface_area:.0f} m2")

    # Wizualizacja
    st.markdown("### Wizualizacja")

    # Subsample dla wizualizacji
    sample_size = min(5000, len(coords))
    sample_idx = np.random.choice(len(coords), sample_size, replace=False)
    vis_coords = coords[sample_idx]

    fig = go.Figure(data=[go.Scatter3d(
        x=vis_coords[:, 0],
        y=vis_coords[:, 1],
        z=vis_coords[:, 2],
        mode='markers',
        marker=dict(size=1, color=vis_coords[:, 2], colorscale='Viridis')
    )])

    # Dodaj bounding box
    bb = metrics.aabb
    edges = [
        [bb.min_point, [bb.max_point[0], bb.min_point[1], bb.min_point[2]]],
        [bb.min_point, [bb.min_point[0], bb.max_point[1], bb.min_point[2]]],
        [bb.min_point, [bb.min_point[0], bb.min_point[1], bb.max_point[2]]],
        [[bb.max_point[0], bb.min_point[1], bb.min_point[2]], [bb.max_point[0], bb.max_point[1], bb.min_point[2]]],
        [[bb.max_point[0], bb.min_point[1], bb.min_point[2]], [bb.max_point[0], bb.min_point[1], bb.max_point[2]]],
        [[bb.min_point[0], bb.max_point[1], bb.min_point[2]], [bb.max_point[0], bb.max_point[1], bb.min_point[2]]],
        [[bb.min_point[0], bb.max_point[1], bb.min_point[2]], [bb.min_point[0], bb.max_point[1], bb.max_point[2]]],
        [[bb.min_point[0], bb.min_point[1], bb.max_point[2]], [bb.max_point[0], bb.min_point[1], bb.max_point[2]]],
        [[bb.min_point[0], bb.min_point[1], bb.max_point[2]], [bb.min_point[0], bb.max_point[1], bb.max_point[2]]],
        [bb.max_point, [bb.max_point[0], bb.max_point[1], bb.min_point[2]]],
        [bb.max_point, [bb.max_point[0], bb.min_point[1], bb.max_point[2]]],
        [bb.max_point, [bb.min_point[0], bb.max_point[1], bb.max_point[2]]],
    ]

    for edge in edges:
        fig.add_trace(go.Scatter3d(
            x=[edge[0][0], edge[1][0]],
            y=[edge[0][1], edge[1][1]],
            z=[edge[0][2], edge[1][2]],
            mode='lines',
            line=dict(color='red', width=2),
            showlegend=False
        ))

    fig.update_layout(height=500, scene=dict(aspectmode='data'))
    st.plotly_chart(fig, use_container_width=True)


def _render_lod_mode():
    """Tryb klasyfikacji LOD"""
    st.subheader("Klasyfikacja LOD")

    st.info("""
    **Level of Detail (LOD)** w BIM:
    - **LOD 100**: Koncepcyjny (masy, objetosci)
    - **LOD 200**: Przyblizona geometria
    - **LOD 300**: Dokladna geometria
    - **LOD 350**: Koordynacja z interfejsami
    - **LOD 400**: Fabrication
    - **LOD 500**: As-built
    """)

    data = _get_data()
    if data is None or data['coords'] is None:
        st.warning("Wczytaj dane")
        return

    coords = data['coords']
    classification = data['classification']

    st.markdown(f"**Punktow:** {len(coords):,}")

    if st.button("KLASYFIKUJ LOD", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        try:
            from ...v2.bim import LODClassifier

            status.info("Klasyfikacja LOD...")
            progress.progress(50)

            classifier = LODClassifier(coords, classification)
            result = classifier.classify()
            progress.progress(100)
            status.success("Klasyfikacja zakonczona!")

            # Wyniki
            st.markdown("### Wynik")

            # LOD Level - duzy
            lod_color = {
                'LOD_100': '#f44336',
                'LOD_200': '#ff9800',
                'LOD_300': '#ffeb3b',
                'LOD_350': '#8bc34a',
                'LOD_400': '#4caf50',
                'LOD_500': '#2196f3'
            }

            color = lod_color.get(result.level.name, '#9e9e9e')

            st.markdown(f"""
            <div style='background: {color}; padding: 30px; border-radius: 10px;
                        text-align: center; margin-bottom: 20px;'>
                <h1 style='color: white; margin: 0; font-size: 3em;'>{result.level.name}</h1>
                <p style='color: white; margin: 10px 0 0 0;'>
                    Confidence: {result.confidence:.0%}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Metryki
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Gestosc punktow", f"{result.point_density:.1f} pts/m2")
            with col2:
                st.metric("Kompletnosc", f"{result.geometry_completeness:.0%}")
            with col3:
                st.metric("Szczegolowość", f"{result.detail_score:.0%}")

            # Szczegolowe metryki
            with st.expander("Szczegolowe metryki"):
                st.json(result.metrics)

            # Rekomendacje
            st.markdown("### Rekomendacje")
            for rec in result.recommendations:
                st.markdown(f"- {rec}")

            # Wymagania dla wyższego LOD
            if result.level.value < 500:
                next_lod = {100: 200, 200: 300, 300: 350, 350: 400, 400: 500}
                target = next_lod.get(result.level.value, 500)

                from ...v2.bim import LODLevel
                target_lod = LODLevel(target)
                requirements = classifier.get_lod_requirements(target_lod)

                st.markdown(f"### Wymagania dla {target_lod.name}")
                st.markdown(f"- Min gestosc: **{requirements['min_density']} pts/m2**")
                st.markdown(f"- Dokladnosc: **{requirements['accuracy_m']} m**")
                st.markdown(f"- Zastosowania: {', '.join(requirements['use_cases'])}")

            st.session_state['bim_lod'] = result

        except Exception as e:
            status.error(f"Blad: {e}")
            import traceback
            st.code(traceback.format_exc())


def _render_clash_mode():
    """Tryb wykrywania kolizji"""
    st.subheader("Wykrywanie kolizji (Clash Detection)")

    st.info("""
    **Clash Detection** wykrywa konflikty przestrzenne:
    - **Hard clash**: Fizyczne przeciecie obiektow
    - **Soft clash**: Naruszenie minimalnego odstępu
    - **Clearance clash**: Naruszenie strefy ochronnej

    Sprawdzane pary: budynki-przewody, roslinnosc-przewody, budynki-tory, itp.
    """)

    data = _get_data()
    if data is None or data['coords'] is None:
        st.warning("Wczytaj dane")
        return

    coords = data['coords']
    classification = data['classification']

    # Konfiguracja
    col1, col2 = st.columns(2)
    with col1:
        default_clearance = st.slider("Domyslny odstęp [m]", 1.0, 10.0, 3.0, 0.5)
    with col2:
        max_clashes = st.slider("Max kolizji", 100, 1000, 500, 100)

    include_soft = st.checkbox("Uwzglednij soft clashes", value=True)

    if st.button("WYKRYJ KOLIZJE", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        try:
            from ...v2.bim import ClashDetector

            status.info("Inicjalizacja detektora...")
            progress.progress(10)

            detector = ClashDetector(coords, classification)

            status.info("Wykrywanie kolizji...")
            progress.progress(30)

            report = detector.detect_all_clashes(
                default_clearance=default_clearance,
                include_soft=include_soft,
                max_clashes=max_clashes
            )

            progress.progress(100)
            status.success(f"Wykryto {report.total_clashes} kolizji!")

            # Wyniki
            _display_clash_results(report)

            st.session_state['bim_clashes'] = report

        except Exception as e:
            status.error(f"Blad: {e}")
            import traceback
            st.code(traceback.format_exc())


def _display_clash_results(report):
    """Wyswietl wyniki detekcji kolizji"""
    st.markdown("### Podsumowanie")

    # Metryki
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Razem", report.total_clashes)
    with col2:
        st.metric("Krytyczne", report.critical_count, delta=None if report.critical_count == 0 else "!")
    with col3:
        st.metric("Powazne", report.major_count)
    with col4:
        st.metric("Drobne", report.minor_count)

    # Wykres typow
    if report.total_clashes > 0:
        st.markdown("### Kolizje wg typu")

        fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'bar'}]])

        # Pie - severity
        severity_counts = {
            'Critical': report.critical_count,
            'Major': report.major_count,
            'Minor': report.minor_count
        }
        fig.add_trace(
            go.Pie(
                labels=list(severity_counts.keys()),
                values=list(severity_counts.values()),
                marker_colors=['#c62828', '#f57c00', '#ffc107'],
                hole=0.4
            ),
            row=1, col=1
        )

        # Bar - by type
        fig.add_trace(
            go.Bar(
                x=list(report.clashes_by_type.keys()),
                y=list(report.clashes_by_type.values()),
                marker_color='#00695c'
            ),
            row=1, col=2
        )

        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        # Kolizje wg elementow
        if report.clashes_by_elements:
            st.markdown("### Kolizje wg par elementow")
            fig2 = go.Figure(data=[go.Bar(
                x=list(report.clashes_by_elements.keys()),
                y=list(report.clashes_by_elements.values()),
                marker_color='#c62828'
            )])
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)

        # Lista kolizji
        with st.expander(f"Lista kolizji ({report.total_clashes})"):
            for clash in report.clashes[:30]:
                severity_color = {
                    'critical': '🔴',
                    'major': '🟠',
                    'minor': '🟡',
                    'info': '🔵'
                }
                icon = severity_color.get(clash.severity.value, '⚪')

                st.markdown(f"""
                {icon} **#{clash.id}** [{clash.clash_type.value}] {clash.element_a} vs {clash.element_b}
                - Odleglosc: {clash.distance:.2f}m
                - Lokalizacja: ({clash.location[0]:.1f}, {clash.location[1]:.1f}, {clash.location[2]:.1f})
                """)
    else:
        st.success("Brak wykrytych kolizji!")


def _render_export_mode():
    """Tryb eksportu BIM"""
    st.subheader("Eksport BIM")

    st.info("""
    **Eksport do formatow BIM:**
    - **IFC**: Industry Foundation Classes - standard BIM
    - **JSON**: Struktura BIM w JSON
    - **XML**: Struktura BIM w XML

    Eksportowane elementy: budynki, teren, infrastruktura
    """)

    data = _get_data()
    if data is None or data['coords'] is None:
        st.warning("Wczytaj dane")
        return

    coords = data['coords']
    classification = data['classification']

    # Sprawdz dostepne dane
    has_buildings = 'bim_buildings' in st.session_state
    has_terrain = True  # Mozemy wygenerowac
    has_lod = 'bim_lod' in st.session_state

    st.markdown("### Dostepne dane")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Budynki" if has_buildings else "~~Budynki~~ (ekstrahuj najpierw)")
    with col2:
        st.write("Teren ✓")
    with col3:
        st.write("LOD" if has_lod else "~~LOD~~ (klasyfikuj najpierw)")

    # Konfiguracja
    st.markdown("### Konfiguracja eksportu")

    project_name = st.text_input("Nazwa projektu", value="CPK_BIM_Export")

    col1, col2 = st.columns(2)
    with col1:
        export_format = st.selectbox("Format:", ["IFC", "JSON", "XML"])
    with col2:
        include_terrain = st.checkbox("Dolacz teren", value=True)

    include_buildings = st.checkbox("Dolacz budynki", value=has_buildings, disabled=not has_buildings)
    include_pointcloud = st.checkbox("Dolacz chmure punktow (probka)", value=False)

    if include_pointcloud:
        sample_rate = st.slider("Procent punktow", 1, 20, 5) / 100
    else:
        sample_rate = 0.01

    if st.button("EKSPORTUJ", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        try:
            from ...v2.bim import IFCExporter
            from ...v2.analysis import TerrainAnalyzer

            status.info("Przygotowanie eksportu...")
            progress.progress(10)

            exporter = IFCExporter(
                project_name=project_name,
                author="CPK Chmura+",
                organization="HackNation 2025"
            )

            # Budynki
            if include_buildings and has_buildings:
                status.info("Dodawanie budynkow...")
                progress.progress(20)

                buildings = st.session_state['bim_buildings']
                for i, b in enumerate(buildings):
                    exporter.add_building(
                        name=f"Building_{i+1}",
                        footprint_vertices=b.footprint.vertices,
                        height=b.height,
                        ground_elevation=b.height_min,
                        properties={
                            'roof_type': b.roof_type.value,
                            'area_m2': b.footprint.area_m2,
                            'volume_m3': b.volume_m3
                        }
                    )

            # Teren
            if include_terrain:
                status.info("Generowanie terenu...")
                progress.progress(40)

                terrain_analyzer = TerrainAnalyzer(coords, classification)
                dtm = terrain_analyzer.generate_dtm(resolution=5.0)

                exporter.add_terrain(
                    grid_x=dtm.grid_x,
                    grid_y=dtm.grid_y,
                    heights=dtm.heights,
                    name="DTM"
                )

            # Chmura punktow
            if include_pointcloud:
                status.info("Dodawanie chmury punktow...")
                progress.progress(60)

                exporter.add_point_cloud(
                    coords=coords,
                    classification=classification,
                    name="PointCloud",
                    sample_rate=sample_rate
                )

            # Eksport
            status.info("Eksportowanie...")
            progress.progress(80)

            output_dir = PATHS.OUTPUT_DIR / "bim"
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            ext = export_format.lower()
            if ext == "ifc":
                ext = "ifc"
            output_path = output_dir / f"{project_name}_{timestamp}.{ext}"

            success = exporter.export(str(output_path), format=export_format.lower())

            progress.progress(100)

            if success:
                status.success("Eksport zakonczony!")

                # Podsumowanie
                summary = exporter.get_summary()
                st.markdown("### Podsumowanie")
                st.json(summary)

                # Download
                with open(output_path, 'rb') as f:
                    content = f.read()

                mime_types = {
                    'ifc': 'application/x-step',
                    'json': 'application/json',
                    'xml': 'application/xml'
                }

                st.download_button(
                    label=f"Pobierz {output_path.name}",
                    data=content,
                    file_name=output_path.name,
                    mime=mime_types.get(ext, 'application/octet-stream')
                )

                st.success(f"Plik zapisany: {output_path}")
            else:
                status.error("Eksport nie powiodl sie")

        except Exception as e:
            status.error(f"Blad: {e}")
            import traceback
            st.code(traceback.format_exc())
