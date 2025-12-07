"""
Railway Analyzer UI - Analiza infrastruktury kolejowej

Funkcjonalnosci:
- Detekcja sieci trakcyjnej (catenary)
- Ekstrakcja geometrii torow
- Detekcja slupow i masztow
- Detekcja sygnalizacji
- Generowanie raportow

HackNation 2025 - CPK Chmura+
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict
from pathlib import Path
import time
import tempfile
import os

from ...config import PATHS


def render_railway_analyzer():
    """Glowny komponent Railway Analyzer"""

    st.markdown("""
    <div style='background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
                padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: white; margin: 0;'>Railway Infrastructure Analyzer</h2>
        <p style='color: #bbdefb; margin: 5px 0 0 0;'>
            Zaawansowana analiza infrastruktury kolejowej z chmur punktow LiDAR
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Menu
    mode = st.radio(
        "Tryb:",
        ["Analiza kompletna", "Siec trakcyjna", "Geometria torow", "Slupy i maszty", "Sygnalizacja", "Raporty"],
        horizontal=True
    )

    st.markdown("---")

    if mode == "Analiza kompletna":
        _render_full_analysis()
    elif mode == "Siec trakcyjna":
        _render_catenary_mode()
    elif mode == "Geometria torow":
        _render_track_geometry_mode()
    elif mode == "Slupy i maszty":
        _render_poles_mode()
    elif mode == "Sygnalizacja":
        _render_signals_mode()
    else:
        _render_reports_mode()


def _get_data():
    """Pobierz dane z session state"""
    # Sprawdz rozne zrodla danych
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


def _render_full_analysis():
    """Kompletna analiza infrastruktury"""
    st.subheader("Kompletna analiza infrastruktury kolejowej")

    st.info("""
    **Analiza kompletna** wykrywa:
    - Siec trakcyjna (przewody jezdne, nośne, powrotne)
    - Geometrie torow (osie, rozstaw, przechyłka, krzywizny)
    - Slupy trakcyjne i oświetleniowe
    - Sygnalizacje kolejowa

    Wymaga sklasyfikowanej chmury punktow z klasami:
    - 18: Tory kolejowe
    - 19: Przewody/linie
    - 20: Slupy/maszty
    """)

    data = _get_data()

    if data is None or data['coords'] is None:
        st.warning("Wczytaj i sklasyfikuj dane w zakladkach 'Wczytaj plik' lub 'Hackathon'")
        return

    coords = data['coords']
    classification = data['classification']

    # Statystyki danych
    st.markdown("### Dane wejściowe")
    unique_classes, counts = np.unique(classification, return_counts=True)

    CLASS_NAMES = {
        2: "Grunt", 3: "Rosl. niska", 4: "Rosl. srednia", 5: "Rosl. wysoka",
        6: "Budynek", 7: "Szum", 18: "Tory", 19: "Linie", 20: "Slupy"
    }

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Punktow", f"{len(coords):,}")
    with col2:
        track_count = counts[unique_classes == 18].sum() if 18 in unique_classes else 0
        st.metric("Punkty torow", f"{track_count:,}")
    with col3:
        wire_count = counts[unique_classes == 19].sum() if 19 in unique_classes else 0
        st.metric("Punkty linii", f"{wire_count:,}")
    with col4:
        pole_count = counts[unique_classes == 20].sum() if 20 in unique_classes else 0
        st.metric("Punkty slupow", f"{pole_count:,}")

    # Sprawdz wymagane klasy
    required_classes = [18, 19, 20]
    missing = [c for c in required_classes if c not in unique_classes]

    if missing:
        st.warning(f"Brak punktow w klasach: {missing}. Niektore analizy moga byc ograniczone.")

    # Konfiguracja
    st.markdown("### Konfiguracja")

    col1, col2 = st.columns(2)
    with col1:
        detect_catenary = st.checkbox("Wykryj siec trakcyjna", value=True)
        detect_tracks = st.checkbox("Analizuj geometrie torow", value=True)

    with col2:
        detect_poles = st.checkbox("Wykryj slupy", value=True)
        detect_signals = st.checkbox("Wykryj sygnalizacje", value=True)

    generate_report = st.checkbox("Generuj raport HTML", value=True)

    # Uruchom analize
    if st.button("URUCHOM ANALIZE", type="primary", use_container_width=True):
        _run_full_analysis(
            coords, classification,
            data.get('colors'), data.get('intensity'),
            detect_catenary, detect_tracks, detect_poles, detect_signals,
            generate_report
        )


def _run_full_analysis(coords, classification, colors, intensity,
                       detect_catenary, detect_tracks, detect_poles, detect_signals,
                       generate_report):
    """Uruchom kompletna analize"""

    progress = st.progress(0)
    status = st.empty()
    start_time = time.time()

    results = {}

    try:
        step = 0
        total_steps = sum([detect_catenary, detect_tracks, detect_poles, detect_signals]) + 1

        # 1. Catenary
        if detect_catenary:
            status.info("Wykrywanie sieci trakcyjnej...")
            step += 1
            progress.progress(int(step / total_steps * 100))

            from ...v2.railway import CatenaryDetector
            detector = CatenaryDetector(coords, classification)
            catenary = detector.detect()

            results['catenary'] = catenary
            results['catenary_stats'] = detector.get_statistics(catenary)

        # 2. Tracks
        if detect_tracks:
            status.info("Analiza geometrii torow...")
            step += 1
            progress.progress(int(step / total_steps * 100))

            from ...v2.railway import TrackExtractor
            extractor = TrackExtractor(coords, classification)
            tracks = extractor.extract_tracks()

            results['tracks'] = tracks
            results['track_geometry'] = extractor.detect_geometry()

        # 3. Poles
        if detect_poles:
            status.info("Wykrywanie slupow...")
            step += 1
            progress.progress(int(step / total_steps * 100))

            from ...v2.railway import PoleDetector
            detector = PoleDetector(coords, classification)
            poles = detector.detect()

            results['poles'] = poles
            results['pole_stats'] = detector.get_statistics(poles)

        # 4. Signals
        if detect_signals:
            status.info("Wykrywanie sygnalizacji...")
            step += 1
            progress.progress(int(step / total_steps * 100))

            from ...v2.railway import SignalDetector
            detector = SignalDetector(coords, classification, colors)
            signals = detector.detect()

            results['signals'] = signals
            results['signal_stats'] = detector.get_statistics(signals)

        # Podsumowanie
        elapsed = time.time() - start_time
        progress.progress(100)
        status.success(f"Analiza zakonczona! ({elapsed:.1f}s)")

        # Wyswietl wyniki
        _display_full_results(results)

        # Generuj raport
        if generate_report:
            _generate_html_report(results, coords, classification)

        # Zapisz w session state
        st.session_state['railway_results'] = results

    except Exception as e:
        status.error(f"Blad analizy: {e}")
        import traceback
        st.code(traceback.format_exc())


def _display_full_results(results):
    """Wyswietl wyniki kompletnej analizy"""

    st.markdown("---")
    st.markdown("## Wyniki analizy")

    # Metryki glowne
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if 'catenary' in results:
            catenary = results['catenary']
            st.metric("Przewody", len(catenary.contact_wires) + len(catenary.messenger_wires))

    with col2:
        if 'tracks' in results:
            st.metric("Segmenty torow", len(results['tracks']))

    with col3:
        if 'poles' in results:
            st.metric("Slupy", len(results['poles']))

    with col4:
        if 'signals' in results:
            st.metric("Sygnaly", len(results['signals']))

    # Szczegoly - Catenary
    if 'catenary_stats' in results:
        with st.expander("Siec trakcyjna - szczegoly", expanded=True):
            stats = results['catenary_stats']

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Przewody jezdne**")
                st.write(f"Liczba: {stats.get('contact_wire_count', 0)}")
                st.write(f"Długość: {stats.get('total_contact_length_m', 0):.1f} m")
                st.write(f"Śr. wysokość: {stats.get('avg_contact_height_m', 0):.2f} m")

            with col2:
                st.markdown("**Liny nośne**")
                st.write(f"Liczba: {stats.get('messenger_wire_count', 0)}")
                st.write(f"Długość: {stats.get('total_messenger_length_m', 0):.1f} m")
                st.write(f"Śr. wysokość: {stats.get('avg_messenger_height_m', 0):.2f} m")

            with col3:
                st.markdown("**Wieszaki**")
                st.write(f"Liczba: {stats.get('dropper_count', 0)}")

    # Szczegoly - Tracks
    if 'tracks' in results and results['tracks']:
        with st.expander("Geometria torow - szczegoly", expanded=True):
            tracks = results['tracks']

            total_length = sum(t.length for t in tracks)
            avg_gauge = np.mean([t.gauge for t in tracks]) if tracks else 0
            max_cant = max(abs(t.cant) for t in tracks) if tracks else 0

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Calkowita dlugosc", f"{total_length:.1f} m")
            with col2:
                st.metric("Sredni rozstaw", f"{avg_gauge:.3f} m")
            with col3:
                odchylenie = (avg_gauge - 1.435) * 1000
                st.metric("Odchylenie rozstawu", f"{odchylenie:+.1f} mm")
            with col4:
                st.metric("Max przechyłka", f"{max_cant:.1f} mm")

            # Wykres typow geometrii
            geometry_types = [t.geometry_type for t in tracks]
            type_counts = {t: geometry_types.count(t) for t in set(geometry_types)}

            if type_counts:
                fig = go.Figure(data=[go.Pie(
                    labels=list(type_counts.keys()),
                    values=list(type_counts.values()),
                    hole=0.4
                )])
                fig.update_layout(height=300, title="Typy geometrii")
                st.plotly_chart(fig, use_container_width=True)

    # Szczegoly - Poles
    if 'pole_stats' in results:
        with st.expander("Slupy - szczegoly", expanded=True):
            stats = results['pole_stats']

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Slupy trakcyjne", stats.get('catenary_count', 0))
            with col2:
                st.metric("Slupy oswietleniowe", stats.get('lighting_count', 0))
            with col3:
                st.metric("Slupy sygnalizacyjne", stats.get('signal_count', 0))

            if stats.get('total_count', 0) > 0:
                st.write(f"Średnia wysokość: {stats.get('avg_height_m', 0):.1f} m")
                st.write(f"Średni odstęp: {stats.get('avg_spacing_m', 0):.1f} m")

    # Szczegoly - Signals
    if 'signal_stats' in results:
        with st.expander("Sygnalizacja - szczegoly", expanded=True):
            stats = results['signal_stats']

            by_type = stats.get('by_type', {})

            if by_type:
                fig = go.Figure(data=[go.Bar(
                    x=list(by_type.keys()),
                    y=list(by_type.values()),
                    marker_color='#1565c0'
                )])
                fig.update_layout(height=300, title="Typy sygnałów")
                st.plotly_chart(fig, use_container_width=True)


def _generate_html_report(results, coords, classification):
    """Generuj raport HTML"""

    from ...v2.railway import InfrastructureReporter

    st.markdown("### Raport")

    try:
        reporter = InfrastructureReporter(
            coords=coords,
            classification=classification,
            catenary=results.get('catenary'),
            tracks=results.get('tracks', []),
            poles=results.get('poles', []),
            signals=results.get('signals', [])
        )

        # Generuj do pliku tymczasowego
        output_dir = PATHS.OUTPUT_DIR / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"railway_report_{timestamp}.html"

        reporter.to_html(str(report_path))

        st.success(f"Raport zapisany: {report_path}")

        # Download button
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        st.download_button(
            label="Pobierz raport HTML",
            data=html_content,
            file_name=f"railway_report_{timestamp}.html",
            mime="text/html"
        )

    except Exception as e:
        st.error(f"Blad generowania raportu: {e}")


def _render_catenary_mode():
    """Tryb analizy sieci trakcyjnej"""
    st.subheader("Siec trakcyjna")

    st.info("""
    **Siec trakcyjna** (catenary system) składa się z:
    - **Przewód jezdny** (contact wire) - 5.0-5.5m nad torem
    - **Lina nośna** (messenger wire) - 6.0-7.5m nad torem
    - **Wieszaki** (droppers) - pionowe połączenia
    - **Przewód powrotny** (return wire) - opcjonalnie

    Algorytm wykrywa przewody używając analizy wysokości i RANSAC.
    """)

    data = _get_data()

    if data is None or data['coords'] is None:
        st.warning("Wczytaj dane")
        return

    coords = data['coords']
    classification = data['classification']

    # Sprawdz klase 19
    wire_mask = classification == 19
    wire_count = wire_mask.sum()

    st.markdown(f"**Punkty klasy 19 (linie):** {wire_count:,}")

    if wire_count < 100:
        st.warning("Za malo punktow linii. Najpierw wykonaj klasyfikacje.")
        return

    # Konfiguracja
    col1, col2 = st.columns(2)
    with col1:
        contact_min = st.slider("Min wys. przew. jezdnego [m]", 4.0, 5.5, 5.0, 0.1)
        contact_max = st.slider("Max wys. przew. jezdnego [m]", 5.0, 6.0, 5.5, 0.1)

    with col2:
        messenger_min = st.slider("Min wys. liny nośnej [m]", 5.5, 7.0, 6.0, 0.1)
        messenger_max = st.slider("Max wys. liny nośnej [m]", 6.5, 8.0, 7.5, 0.1)

    if st.button("WYKRYJ SIEC TRAKCYJNA", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        try:
            from ...v2.railway import CatenaryDetector

            status.info("Inicjalizacja detektora...")
            progress.progress(10)

            # Modyfikuj zakresy wysokosci
            CatenaryDetector.CONTACT_WIRE_HEIGHT = (contact_min, contact_max)
            CatenaryDetector.MESSENGER_WIRE_HEIGHT = (messenger_min, messenger_max)

            detector = CatenaryDetector(coords, classification)

            status.info("Wykrywanie przewodow...")
            progress.progress(30)

            catenary = detector.detect()

            progress.progress(90)

            stats = detector.get_statistics(catenary)

            progress.progress(100)
            status.success("Detekcja zakonczona!")

            # Wyniki
            st.markdown("### Wyniki")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Przewody jezdne", stats.get('contact_wire_count', 0))
            with col2:
                st.metric("Liny nosne", stats.get('messenger_wire_count', 0))
            with col3:
                st.metric("Wieszaki", stats.get('dropper_count', 0))

            # Wizualizacja 3D (uproszczona)
            if len(catenary.contact_wires) > 0 or len(catenary.messenger_wires) > 0:
                st.markdown("### Wizualizacja")

                fig = go.Figure()

                # Contact wires
                for i, wire in enumerate(catenary.contact_wires[:10]):  # Limit dla wydajnosci
                    fig.add_trace(go.Scatter3d(
                        x=wire.points[:, 0],
                        y=wire.points[:, 1],
                        z=wire.points[:, 2],
                        mode='markers',
                        marker=dict(size=2, color='red'),
                        name=f'Contact {i+1}'
                    ))

                # Messenger wires
                for i, wire in enumerate(catenary.messenger_wires[:10]):
                    fig.add_trace(go.Scatter3d(
                        x=wire.points[:, 0],
                        y=wire.points[:, 1],
                        z=wire.points[:, 2],
                        mode='markers',
                        marker=dict(size=2, color='blue'),
                        name=f'Messenger {i+1}'
                    ))

                fig.update_layout(
                    height=500,
                    title="Siec trakcyjna 3D",
                    scene=dict(
                        aspectmode='data',
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z'
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

            st.session_state['catenary_results'] = catenary

        except Exception as e:
            status.error(f"Blad: {e}")
            import traceback
            st.code(traceback.format_exc())


def _render_track_geometry_mode():
    """Tryb analizy geometrii torow"""
    st.subheader("Geometria torow")

    st.info("""
    **Analiza geometrii torow** obejmuje:
    - **Ekstrakcja osi** lewej i prawej szyny
    - **Rozstaw torow** (gauge) - normalny 1435mm
    - **Przechyłka** (cant) - roznica wysokosci szyn w lukach
    - **Krzywizna** (curvature) - promienie lukow
    - **Gradient** - nachylenie podluzne

    Wymaga klasy 18 (tory).
    """)

    data = _get_data()

    if data is None or data['coords'] is None:
        st.warning("Wczytaj dane")
        return

    coords = data['coords']
    classification = data['classification']

    track_mask = classification == 18
    track_count = track_mask.sum()

    st.markdown(f"**Punkty klasy 18 (tory):** {track_count:,}")

    if track_count < 100:
        st.warning("Za malo punktow torow")
        return

    # Konfiguracja
    col1, col2 = st.columns(2)
    with col1:
        expected_gauge = st.number_input("Oczekiwany rozstaw [m]", 1.400, 1.500, 1.435, 0.001)
    with col2:
        min_segment = st.slider("Min dlugosc segmentu [m]", 5, 50, 10, 5)

    if st.button("ANALIZUJ GEOMETRIE TOROW", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        try:
            from ...v2.railway import TrackExtractor

            status.info("Ekstrakcja torow...")
            progress.progress(20)

            extractor = TrackExtractor(
                coords, classification,
                expected_gauge=expected_gauge,
                min_segment_length=min_segment
            )

            tracks = extractor.extract_tracks()
            progress.progress(60)

            status.info("Analiza geometrii...")
            geometries = extractor.detect_geometry()
            progress.progress(90)

            progress.progress(100)
            status.success(f"Wykryto {len(tracks)} segmentow")

            if not tracks:
                st.warning("Nie wykryto segmentow torow")
                return

            # Wyniki
            st.markdown("### Wyniki")

            total_length = sum(t.length for t in tracks)
            avg_gauge = np.mean([t.gauge for t in tracks])
            gauges = [t.gauge for t in tracks]
            cants = [t.cant for t in tracks]
            curvatures = [t.curvature for t in tracks]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Calkowita dlugosc", f"{total_length:.1f} m")
            with col2:
                st.metric("Sredni rozstaw", f"{avg_gauge*1000:.1f} mm")
            with col3:
                deviation = (avg_gauge - expected_gauge) * 1000
                st.metric("Odchylenie", f"{deviation:+.1f} mm")
            with col4:
                st.metric("Segmentow", len(tracks))

            # Wykresy
            st.markdown("### Wykresy")

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=["Rozstaw szyn", "Przechyłka", "Krzywizna", "Typ geometrii"]
            )

            # Gauge
            fig.add_trace(
                go.Histogram(x=[g*1000 for g in gauges], marker_color='#1565c0', name='Rozstaw'),
                row=1, col=1
            )
            fig.add_vline(x=expected_gauge*1000, line_dash="dash", line_color="red",
                         annotation_text="Norma", row=1, col=1)

            # Cant
            fig.add_trace(
                go.Histogram(x=cants, marker_color='#43a047', name='Przechyłka'),
                row=1, col=2
            )

            # Curvature
            radii = [1/c if c > 0.0001 else 10000 for c in curvatures]
            fig.add_trace(
                go.Histogram(x=radii, marker_color='#f57c00', name='Promień'),
                row=2, col=1
            )

            # Geometry types
            types = [t.geometry_type for t in tracks]
            type_counts = {t: types.count(t) for t in set(types)}
            fig.add_trace(
                go.Bar(x=list(type_counts.keys()), y=list(type_counts.values()),
                      marker_color='#7b1fa2'),
                row=2, col=2
            )

            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Tabela segmentow
            with st.expander("Szczegoly segmentow"):
                for i, track in enumerate(tracks[:20]):  # Limit
                    radius = 1/track.curvature if track.curvature > 0.0001 else float('inf')
                    st.markdown(f"""
                    **Segment {i+1}:** {track.length:.1f}m,
                    rozstaw: {track.gauge*1000:.1f}mm,
                    przechyłka: {track.cant:.1f}mm,
                    typ: {track.geometry_type},
                    promień: {radius:.0f}m
                    """)

            st.session_state['track_results'] = tracks

        except Exception as e:
            status.error(f"Blad: {e}")
            import traceback
            st.code(traceback.format_exc())


def _render_poles_mode():
    """Tryb detekcji slupow"""
    st.subheader("Slupy i maszty")

    st.info("""
    **Detekcja slupow** rozpoznaje:
    - **Slupy trakcyjne** - 6-12m, podtrzymuja siec trakcyjna
    - **Slupy oswietleniowe** - >12m, oświetlenie peronu/stacji
    - **Slupy sygnalizacyjne** - 3-8m, sygnalizacja
    - **Bramownice** - szerokie konstrukcje nad torami

    Wymaga klasy 20 (slupy).
    """)

    data = _get_data()

    if data is None or data['coords'] is None:
        st.warning("Wczytaj dane")
        return

    coords = data['coords']
    classification = data['classification']

    pole_mask = classification == 20
    pole_count = pole_mask.sum()

    st.markdown(f"**Punkty klasy 20 (slupy):** {pole_count:,}")

    if pole_count < 50:
        st.warning("Za malo punktow slupow")
        return

    # Konfiguracja
    col1, col2 = st.columns(2)
    with col1:
        min_height = st.slider("Min wysokosc slupa [m]", 2.0, 5.0, 3.0, 0.5)
    with col2:
        max_width = st.slider("Max szerokosc slupa [m]", 1.0, 3.0, 2.0, 0.5)

    if st.button("WYKRYJ SLUPY", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        try:
            from ...v2.railway import PoleDetector

            status.info("Inicjalizacja detektora...")
            progress.progress(10)

            detector = PoleDetector(
                coords, classification,
                min_pole_height=min_height,
                max_pole_width=max_width
            )

            status.info("Wykrywanie slupow...")
            progress.progress(30)

            poles = detector.detect()
            progress.progress(80)

            stats = detector.get_statistics(poles)

            progress.progress(100)
            status.success(f"Wykryto {len(poles)} slupow")

            # Wyniki
            st.markdown("### Wyniki")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Slupy razem", len(poles))
            with col2:
                st.metric("Trakcyjne", stats.get('catenary_count', 0))
            with col3:
                st.metric("Oswietleniowe", stats.get('lighting_count', 0))
            with col4:
                st.metric("Sygnalizacyjne", stats.get('signal_count', 0))

            if poles:
                # Statystyki
                st.markdown("### Statystyki")

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Średnia wysokość: {stats.get('avg_height_m', 0):.1f} m")
                    st.write(f"Min wysokość: {stats.get('min_height_m', 0):.1f} m")
                    st.write(f"Max wysokość: {stats.get('max_height_m', 0):.1f} m")
                with col2:
                    st.write(f"Średni odstęp: {stats.get('avg_spacing_m', 0):.1f} m")

                # Wykres typow
                by_type = stats.get('by_type', {})
                if by_type:
                    fig = go.Figure(data=[go.Pie(
                        labels=list(by_type.keys()),
                        values=list(by_type.values()),
                        hole=0.4
                    )])
                    fig.update_layout(height=350, title="Typy slupow")
                    st.plotly_chart(fig, use_container_width=True)

                # Wysokosci
                heights = [p.height for p in poles]
                fig2 = go.Figure(data=[go.Histogram(
                    x=heights,
                    marker_color='#1565c0',
                    nbinsx=20
                )])
                fig2.update_layout(
                    height=300,
                    title="Rozklad wysokosci slupow",
                    xaxis_title="Wysokość [m]",
                    yaxis_title="Liczba"
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.session_state['pole_results'] = poles

        except Exception as e:
            status.error(f"Blad: {e}")
            import traceback
            st.code(traceback.format_exc())


def _render_signals_mode():
    """Tryb detekcji sygnalizacji"""
    st.subheader("Sygnalizacja kolejowa")

    st.info("""
    **Detekcja sygnalizacji** rozpoznaje:
    - **Semafory glowne** - 3.5-6m nad torem
    - **Semafory przejsciowe** - 2.5-4.5m nad torem
    - **Ograniczenia predkosci** - 1.5-3m
    - **Slupki kilometrazowe** - 0.5-1.5m
    - **Wskazniki W** - sygnaly dzwiekowe

    Sygnaly sa wykrywane jako male obiekty na slupach lub wolnostojace.
    """)

    data = _get_data()

    if data is None or data['coords'] is None:
        st.warning("Wczytaj dane")
        return

    coords = data['coords']
    classification = data['classification']
    colors = data.get('colors')

    if st.button("WYKRYJ SYGNALIZACJE", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        try:
            from ...v2.railway import SignalDetector

            status.info("Inicjalizacja detektora...")
            progress.progress(10)

            detector = SignalDetector(coords, classification, colors)

            status.info("Wykrywanie sygnalow...")
            progress.progress(30)

            signals = detector.detect()
            progress.progress(80)

            stats = detector.get_statistics(signals)

            progress.progress(100)
            status.success(f"Wykryto {len(signals)} sygnalow")

            # Wyniki
            st.markdown("### Wyniki")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sygnaly razem", len(signals))
            with col2:
                st.metric("Semafory glowne", stats.get('main_signals', 0))
            with col3:
                st.metric("Ograniczenia", stats.get('speed_limits', 0))
            with col4:
                st.metric("Slupki km", stats.get('km_posts', 0))

            if signals:
                # Wykres typow
                by_type = stats.get('by_type', {})
                if by_type:
                    fig = go.Figure(data=[go.Bar(
                        x=list(by_type.keys()),
                        y=list(by_type.values()),
                        marker_color='#d32f2f'
                    )])
                    fig.update_layout(height=350, title="Typy sygnałów")
                    st.plotly_chart(fig, use_container_width=True)

                # Tabela
                with st.expander("Lista sygnalow"):
                    for i, sig in enumerate(signals[:30]):  # Limit
                        st.markdown(f"""
                        **Sygnal {i+1}:** {sig.signal_type.value},
                        wys: {sig.height:.1f}m,
                        szer: {sig.width:.2f}m,
                        confidence: {sig.confidence:.0%}
                        """)

            st.session_state['signal_results'] = signals

        except Exception as e:
            status.error(f"Blad: {e}")
            import traceback
            st.code(traceback.format_exc())


def _render_reports_mode():
    """Tryb raportow"""
    st.subheader("Raporty infrastruktury")

    st.info("""
    **Generowanie raportow** w formatach:
    - **HTML** - interaktywny raport z wizualizacjami
    - **CSV** - dane tabelaryczne do dalszej analizy
    - **GeoJSON** - dane geograficzne dla GIS
    - **KML** - dla Google Earth
    """)

    # Sprawdz dostepne wyniki
    has_catenary = 'catenary_results' in st.session_state or 'railway_results' in st.session_state
    has_tracks = 'track_results' in st.session_state or 'railway_results' in st.session_state
    has_poles = 'pole_results' in st.session_state or 'railway_results' in st.session_state
    has_signals = 'signal_results' in st.session_state or 'railway_results' in st.session_state

    if not any([has_catenary, has_tracks, has_poles, has_signals]):
        st.warning("Najpierw wykonaj analize w jednym z trybów")
        return

    st.markdown("### Dostepne dane")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write("Siec trakcyjna" if has_catenary else "~~Siec trakcyjna~~")
    with col2:
        st.write("Geometria torow" if has_tracks else "~~Geometria torow~~")
    with col3:
        st.write("Slupy" if has_poles else "~~Slupy~~")
    with col4:
        st.write("Sygnalizacja" if has_signals else "~~Sygnalizacja~~")

    # Formaty
    st.markdown("### Format raportu")

    format_options = st.multiselect(
        "Wybierz formaty:",
        ["HTML", "CSV", "GeoJSON", "KML"],
        default=["HTML"]
    )

    project_name = st.text_input("Nazwa projektu", value="CPK_Railway_Analysis")

    if st.button("GENERUJ RAPORTY", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        try:
            # Zbierz dane
            data = _get_data()
            coords = data['coords'] if data else None
            classification = data['classification'] if data else None

            # Pobierz wyniki
            if 'railway_results' in st.session_state:
                rr = st.session_state['railway_results']
                catenary = rr.get('catenary')
                tracks = rr.get('tracks', [])
                poles = rr.get('poles', [])
                signals = rr.get('signals', [])
            else:
                catenary = st.session_state.get('catenary_results')
                tracks = st.session_state.get('track_results', [])
                poles = st.session_state.get('pole_results', [])
                signals = st.session_state.get('signal_results', [])

            output_dir = PATHS.OUTPUT_DIR / "reports"
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            generated_files = []

            from ...v2.railway import InfrastructureReporter

            reporter = InfrastructureReporter(
                coords=coords,
                classification=classification,
                catenary=catenary,
                tracks=tracks if tracks else [],
                poles=poles if poles else [],
                signals=signals if signals else []
            )

            # HTML
            if "HTML" in format_options:
                status.info("Generowanie raportu HTML...")
                progress.progress(25)

                html_path = output_dir / f"{project_name}_{timestamp}.html"
                reporter.to_html(str(html_path))
                generated_files.append(('HTML', html_path))

            # CSV
            if "CSV" in format_options:
                status.info("Generowanie CSV...")
                progress.progress(50)

                csv_dir = output_dir / f"{project_name}_{timestamp}_csv"
                csv_dir.mkdir(exist_ok=True)
                reporter.to_csv(str(csv_dir))
                generated_files.append(('CSV', csv_dir))

            # GeoJSON
            if "GeoJSON" in format_options:
                status.info("Generowanie GeoJSON...")
                progress.progress(75)

                from ...v2.railway import export_to_geojson
                geojson_path = output_dir / f"{project_name}_{timestamp}.geojson"
                export_to_geojson(poles if poles else [], signals if signals else [], str(geojson_path))
                generated_files.append(('GeoJSON', geojson_path))

            # KML
            if "KML" in format_options:
                status.info("Generowanie KML...")
                progress.progress(90)

                from ...v2.railway import export_to_kml
                kml_path = output_dir / f"{project_name}_{timestamp}.kml"
                export_to_kml(poles if poles else [], signals if signals else [], str(kml_path))
                generated_files.append(('KML', kml_path))

            progress.progress(100)
            status.success("Raporty wygenerowane!")

            # Lista plikow
            st.markdown("### Wygenerowane pliki")

            for fmt, path in generated_files:
                if path.is_file():
                    with open(path, 'rb') as f:
                        content = f.read()

                    mime_types = {
                        'HTML': 'text/html',
                        'GeoJSON': 'application/json',
                        'KML': 'application/vnd.google-earth.kml+xml'
                    }

                    st.download_button(
                        label=f"Pobierz {fmt}: {path.name}",
                        data=content,
                        file_name=path.name,
                        mime=mime_types.get(fmt, 'application/octet-stream')
                    )
                elif path.is_dir():
                    st.write(f"{fmt}: {path}")

        except Exception as e:
            status.error(f"Blad: {e}")
            import traceback
            st.code(traceback.format_exc())
