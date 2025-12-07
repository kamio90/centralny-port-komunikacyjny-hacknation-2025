"""
Hackathon Klasyfikacja - HackNation 2025 CPK
Kompletny modul z eksportem LAS/LAZ, raportami i IFC

Dwa tryby:
1. PELNA KLASYFIKACJA - caly plik jednym kliknieciem
2. TRYB KWADRATOW - wybierz fragmenty do klasyfikacji
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
import time
import json
from pathlib import Path
import tempfile
import gc
import laspy
import os

from ...v2 import LASLoader
from ...v2.core import GridManager
from ...v2.pipeline import ProfessionalPipeline, PipelineConfig, BatchClassifier, BatchConfig
from ...config import PATHS


# Progi dla roznych trybów
BATCH_THRESHOLD = 50_000_000  # 50M punktów - użyj batch mode
WARN_THRESHOLD = 100_000_000  # 100M - ostrzeżenie o czasie


# Pelna mapa klas ASPRS + CPK
ASPRS_CLASSES = {
    1: {"name": "Nieklasyfikowane", "color": "#9E9E9E", "description": "Punkty bez przypisanej klasy"},
    2: {"name": "Grunt", "color": "#8D6E63", "description": "Powierzchnia gruntu"},
    3: {"name": "Roslinnosc niska", "color": "#AED581", "description": "Trawy <0.5m"},
    4: {"name": "Roslinnosc srednia", "color": "#66BB6A", "description": "Krzewy 0.5-2m"},
    5: {"name": "Roslinnosc wysoka", "color": "#2E7D32", "description": "Drzewa >2m"},
    6: {"name": "Budynek", "color": "#D7CCC8", "description": "Obiekty kubaturowe"},
    7: {"name": "Szum", "color": "#F44336", "description": "Punkty bledne/zaklocenia"},
    9: {"name": "Woda", "color": "#29B6F6", "description": "Zbiorniki wodne"},
    17: {"name": "Most", "color": "#795548", "description": "Obiekty mostowe"},
    18: {"name": "Tory kolejowe", "color": "#6D4C41", "description": "Szyny i podklady"},
    19: {"name": "Linie energetyczne", "color": "#FDD835", "description": "Przewody nad torami"},
    20: {"name": "Slupy trakcyjne", "color": "#546E7A", "description": "Maszty i wsporniki"},
    21: {"name": "Peron kolejowy", "color": "#78909C", "description": "Krawedz peronowa"},
    22: {"name": "Znaki kolejowe", "color": "#FF7043", "description": "Semafory i tablice"},
    23: {"name": "Ogrodzenie kolejowe", "color": "#8D6E63", "description": "Siatki wzdluz linii"},
    30: {"name": "Jezdnia", "color": "#455A64", "description": "Nawierzchnia drogowa"},
    31: {"name": "Chodnik", "color": "#607D8B", "description": "Nawierzchnia piesza"},
    32: {"name": "Kraweznik", "color": "#B0BEC5", "description": "Granica jezdnia/chodnik"},
    33: {"name": "Pas zieleni", "color": "#81C784", "description": "Zielen przydrozna"},
    34: {"name": "Oswietlenie", "color": "#FFD54F", "description": "Latarnie"},
    35: {"name": "Znak drogowy", "color": "#FF5722", "description": "Slupki i tablice"},
    36: {"name": "Bariera drogowa", "color": "#90A4AE", "description": "Barierki ochronne"},
    40: {"name": "Sciany budynkow", "color": "#BCAAA4", "description": "Fasady zewnetrzne"},
    41: {"name": "Dachy budynkow", "color": "#A1887F", "description": "Pokrycie dachowe"},
    64: {"name": "Terminal lotniska", "color": "#7986CB", "description": "Budynki lotniskowe CPK"},
    65: {"name": "Pas startowy", "color": "#424242", "description": "Nawierzchnia lotniskowa"},
    66: {"name": "Wezzel kolejowy", "color": "#5D4037", "description": "Skrzyzowania torow"},
    67: {"name": "Parking", "color": "#757575", "description": "Miejsca postojowe"},
}


def render_hackathon_classification(n_threads: int = 1) -> None:
    """Hackathon - kompletny modul klasyfikacji"""

    st.markdown("""
    <div style='background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
                padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: white; margin: 0;'>HackNation 2025 - CPK Chmura+</h2>
        <p style='color: #90caf9; margin: 5px 0 0 0;'>Automatyczna klasyfikacja chmur punktow LiDAR</p>
    </div>
    """, unsafe_allow_html=True)

    if 'input_file' not in st.session_state:
        st.warning("Najpierw wczytaj plik LAS/LAZ w zakladce 'Wczytaj plik'")
        _show_supported_classes()
        return

    input_path = st.session_state['input_file']
    file_info = st.session_state['file_info']

    # Info o pliku
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Plik", Path(input_path).name)
    with col2:
        st.metric("Punktow", f"{file_info['n_points']:,}")
    with col3:
        size_mb = os.path.getsize(input_path) / (1024*1024)
        st.metric("Rozmiar", f"{size_mb:.1f} MB")

    st.markdown("---")

    # Wybor trybu
    mode = st.radio(
        "Wybierz tryb klasyfikacji:",
        ["Pelna klasyfikacja", "Tryb kwadratow (zaawansowany)"],
        horizontal=True,
        help="Pelna klasyfikacja - caly plik. Tryb kwadratow - wybierz fragmenty."
    )

    if mode == "Pelna klasyfikacja":
        _render_full_classification_mode()
    else:
        _render_grid_mode()


def _show_supported_classes():
    """Pokazuje wspierane klasy"""
    with st.expander("Wspierane klasy klasyfikacji", expanded=True):
        cols = st.columns(3)
        classes_list = list(ASPRS_CLASSES.items())
        per_col = len(classes_list) // 3 + 1

        for i, col in enumerate(cols):
            with col:
                for cls_id, info in classes_list[i*per_col:(i+1)*per_col]:
                    st.markdown(f"""
                    <div style='display: flex; align-items: center; margin: 5px 0;'>
                        <div style='width: 20px; height: 20px; background: {info["color"]};
                                    border-radius: 3px; margin-right: 10px;'></div>
                        <span><b>[{cls_id}]</b> {info["name"]}</span>
                    </div>
                    """, unsafe_allow_html=True)


def _render_full_classification_mode():
    """Tryb pelnej klasyfikacji - caly plik"""

    st.subheader("Pelna klasyfikacja")

    input_path = st.session_state['input_file']
    file_info = st.session_state['file_info']
    n_points = file_info['n_points']

    # Informacja o duzym pliku
    if n_points > WARN_THRESHOLD:
        st.warning(f"""
        **Bardzo duzy plik ({n_points/1_000_000:.0f}M punktow)**

        Przetwarzanie moze zajac 30+ minut. Zalecamy:
        - Tryb "Szybki" dla wstepnej analizy
        - Batch mode automatycznie obsluguje ogromne pliki
        - Upewnij sie, ze masz wystarczajaco pamieci RAM (min. 16GB)
        """)
    elif n_points > BATCH_THRESHOLD:
        st.info(f"Duzy plik ({n_points/1_000_000:.0f}M) - automatycznie uzyty zostanie batch mode")

    # Konfiguracja
    col1, col2 = st.columns(2)

    with col1:
        quality = st.selectbox(
            "Jakosc klasyfikacji",
            ["Standardowa", "Wysoka (wolniejsza)", "Szybka (mniej dokladna)"],
            help="Wysza jakosc = lepsze wyniki, ale dluzszy czas"
        )

        include_buildings = st.checkbox("Wykrywaj budynki (RANSAC)", value=True,
                                       help="Wymaga wiecej czasu, ale wykrywa dachy i sciany")

    with col2:
        output_format = st.selectbox(
            "Format wyjsciowy",
            ["LAS", "LAZ (skompresowany)"],
            help="LAZ jest mniejszy, ale wymaga wiecej czasu"
        )

        generate_ifc = st.checkbox("Generuj IFC (BIM)", value=False,
                                   help="Eksport do formatu BIM dla integracji z systemami projektowymi")

    # Dodatkowe opcje dla duzych plikow
    use_batch = False
    chunk_size = 10_000_000
    if n_points > BATCH_THRESHOLD:
        with st.expander("Zaawansowane opcje (batch mode)", expanded=False):
            use_batch = st.checkbox("Uzyj batch mode", value=True,
                                   help="Zalecane dla plikow > 50M punktow - mniejsze zuzycie pamieci")
            chunk_size = st.slider("Rozmiar chunka (mln)", 5, 20, 10) * 1_000_000

    # Estymacja czasu
    if quality == "Szybka (mniej dokladna)":
        est_speed = 400_000
    elif quality == "Wysoka (wolniejsza)":
        est_speed = 80_000
    else:
        est_speed = 150_000

    est_time = n_points / est_speed

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Estymowany czas", f"{int(est_time//60)} min {int(est_time%60)} s")
    with col2:
        st.metric("Predkosc", f"~{est_speed:,} pkt/s")

    # Przycisk START
    if st.button("ROZPOCZNIJ KLASYFIKACJE", type="primary", use_container_width=True):
        if use_batch and n_points > BATCH_THRESHOLD:
            _run_batch_classification(
                quality=quality,
                include_buildings=include_buildings,
                output_format=output_format,
                generate_ifc=generate_ifc,
                chunk_size=chunk_size
            )
        else:
            _run_full_classification(
                quality=quality,
                include_buildings=include_buildings,
                output_format=output_format,
                generate_ifc=generate_ifc
            )

    # Wyniki
    if 'hack_full_results' in st.session_state:
        _show_full_results()


def _run_full_classification(quality: str, include_buildings: bool,
                             output_format: str, generate_ifc: bool):
    """Uruchamia pelna klasyfikacje"""

    progress = st.progress(0)
    status = st.empty()
    start_time = time.time()

    input_path = st.session_state['input_file']
    file_info = st.session_state['file_info']

    try:
        # 1. Wczytaj dane
        status.info("Wczytywanie danych...")
        progress.progress(5)

        loader = LASLoader(input_path)
        data = loader.load()
        coords = data['coords']
        colors = data['colors']
        intensity = data['intensity']

        n_points = len(coords)
        status.info(f"Wczytano {n_points:,} punktow")
        progress.progress(15)

        # 2. Konfiguracja pipeline
        if quality == "Szybka (mniej dokladna)":
            config = PipelineConfig(
                detect_noise=True,
                classify_ground=True,
                classify_vegetation=True,
                detect_buildings=False,
                detect_infrastructure=True,
                use_fast_noise_detection=True,
                noise_voxel_size=1.0,
                noise_k_neighbors=15,
                hag_grid_resolution=2.0
            )
        elif quality == "Wysoka (wolniejsza)":
            config = PipelineConfig(
                detect_noise=True,
                classify_ground=True,
                classify_vegetation=True,
                detect_buildings=include_buildings,
                detect_infrastructure=True,
                use_fast_noise_detection=False,
                noise_k_neighbors=50,
                hag_grid_resolution=0.5
            )
        else:
            config = PipelineConfig(
                detect_noise=True,
                classify_ground=True,
                classify_vegetation=True,
                detect_buildings=include_buildings,
                detect_infrastructure=True,
                use_fast_noise_detection=(n_points > 5_000_000)
            )

        # 3. Klasyfikacja
        status.info("Klasyfikacja w toku...")
        progress.progress(20)

        pipeline = ProfessionalPipeline(coords, colors, intensity, config)

        def progress_cb(step, pct, msg):
            overall = 20 + int(pct * 0.6)
            progress.progress(min(overall, 80))
            status.info(f"{step}: {msg}")

        classification, stats = pipeline.run(progress_callback=progress_cb)

        progress.progress(85)

        # 4. Zapisz wyniki
        status.info("Zapisywanie wynikow...")

        # Przygotuj LAS
        with laspy.open(input_path) as src:
            las_orig = src.read()

            max_class = int(classification.max())

            if max_class > 31:
                has_rgb = hasattr(las_orig, 'red') and las_orig.red is not None
                point_format = 7 if has_rgb else 6
                header = laspy.LasHeader(point_format=point_format, version="1.4")
                header.scales = las_orig.header.scales
                header.offsets = las_orig.header.offsets
                las = laspy.LasData(header)
                las.x = las_orig.x
                las.y = las_orig.y
                las.z = las_orig.z
                las.intensity = las_orig.intensity
                if has_rgb:
                    las.red = las_orig.red
                    las.green = las_orig.green
                    las.blue = las_orig.blue
                las.classification = classification.astype(np.uint8)
            else:
                las = las_orig
                las.classification = classification.astype(np.uint8)

            # Zapisz do bufora
            suffix = '.laz' if 'LAZ' in output_format else '.las'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                if suffix == '.laz':
                    las.write(tmp.name, laz_backend=laspy.compression.LazBackend.Laszip)
                else:
                    las.write(tmp.name)

                with open(tmp.name, 'rb') as f:
                    las_bytes = f.read()

        progress.progress(95)

        # 5. Generuj IFC jesli wymagane
        ifc_content = None
        if generate_ifc:
            status.info("Generowanie IFC...")
            try:
                from ...v2.core import IFCExporter
                exporter = IFCExporter(
                    coords=coords,
                    classification=classification,
                    project_name=f"CPK - {Path(input_path).stem}",
                    project_description="HackNation 2025 - Automatic LiDAR Classification",
                    author="CPK Chmura+ Classifier v2.0"
                )
                with tempfile.NamedTemporaryFile(suffix='.ifc', delete=False, mode='w') as tmp:
                    exporter.export(tmp.name)
                    with open(tmp.name, 'r') as f:
                        ifc_content = f.read()
            except Exception as e:
                st.warning(f"Nie udalo sie wygenerowac IFC: {e}")

        progress.progress(100)
        elapsed = time.time() - start_time

        status.success(f"Gotowe! Czas: {elapsed:.1f}s ({n_points/elapsed:,.0f} pkt/s)")

        # Zapisz wyniki do session state
        st.session_state['hack_full_results'] = {
            'classification': classification,
            'stats': stats,
            'elapsed': elapsed,
            'n_points': n_points,
            'las_bytes': las_bytes,
            'ifc_content': ifc_content,
            'output_format': output_format,
            'coords': coords
        }

        # Cleanup
        del pipeline, classification
        gc.collect()

        st.rerun()

    except Exception as e:
        status.error(f"Blad: {e}")
        import traceback
        st.code(traceback.format_exc())


def _run_batch_classification(quality: str, include_buildings: bool,
                              output_format: str, generate_ifc: bool,
                              chunk_size: int):
    """Uruchamia batch klasyfikacje dla ogromnych plikow"""

    progress = st.progress(0)
    status = st.empty()
    chunk_info = st.empty()
    start_time = time.time()

    input_path = st.session_state['input_file']
    file_info = st.session_state['file_info']
    n_points = file_info['n_points']

    try:
        # Konfiguracja pipeline
        if quality == "Szybka (mniej dokladna)":
            config = PipelineConfig(
                detect_noise=True,
                classify_ground=True,
                classify_vegetation=True,
                detect_buildings=False,
                detect_infrastructure=True,
                use_fast_noise_detection=True,
                noise_voxel_size=1.0,
                noise_k_neighbors=15,
                hag_grid_resolution=2.0
            )
        elif quality == "Wysoka (wolniejsza)":
            config = PipelineConfig(
                detect_noise=True,
                classify_ground=True,
                classify_vegetation=True,
                detect_buildings=include_buildings,
                detect_infrastructure=True,
                use_fast_noise_detection=False,
                noise_k_neighbors=50,
                hag_grid_resolution=0.5
            )
        else:
            config = PipelineConfig(
                detect_noise=True,
                classify_ground=True,
                classify_vegetation=True,
                detect_buildings=include_buildings,
                detect_infrastructure=True,
                use_fast_noise_detection=True
            )

        # Output path
        suffix = '.laz' if 'LAZ' in output_format else '.las'
        output_path = PATHS.OUTPUT_DIR / f"{Path(input_path).stem}_classified{suffix}"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Batch config
        batch_config = BatchConfig(chunk_size=chunk_size)

        # Uruchom batch classifier
        status.info("Inicjalizacja batch classifier...")
        classifier = BatchClassifier(
            str(input_path),
            str(output_path),
            pipeline_config=config,
            batch_config=batch_config
        )

        def batch_progress_cb(chunk_idx, n_chunks, pct, msg):
            progress.progress(int(pct))
            status.info(msg)
            chunk_info.markdown(f"**Chunk {chunk_idx+1}/{n_chunks}** - {pct:.1f}%")

        stats = classifier.run(progress_callback=batch_progress_cb)

        progress.progress(100)
        elapsed = time.time() - start_time

        status.success(f"Gotowe! Czas: {elapsed:.1f}s ({n_points/elapsed:,.0f} pkt/s)")
        chunk_info.empty()

        # Wczytaj wynikowy plik do pobrania
        with open(output_path, 'rb') as f:
            las_bytes = f.read()

        # IFC (dla batch mode generujemy z probki)
        ifc_content = None
        if generate_ifc:
            status.info("Generowanie IFC (z probki)...")
            try:
                from ...v2.core import IFCExporter
                # Wczytaj probke
                loader = LASLoader(str(output_path))
                data = loader.load(sample_size=500_000)

                exporter = IFCExporter(
                    coords=data['coords'],
                    classification=data['classification'],
                    project_name=f"CPK - {Path(input_path).stem}",
                    project_description="HackNation 2025 - Batch Classification",
                    author="CPK Chmura+ Classifier v2.0"
                )
                with tempfile.NamedTemporaryFile(suffix='.ifc', delete=False, mode='w') as tmp:
                    exporter.export(tmp.name)
                    with open(tmp.name, 'r') as f:
                        ifc_content = f.read()
            except Exception as e:
                st.warning(f"Nie udalo sie wygenerowac IFC: {e}")

        # Zapisz wyniki
        st.session_state['hack_full_results'] = {
            'classification': None,  # Nie trzymamy w pamieci
            'stats': stats,
            'elapsed': elapsed,
            'n_points': n_points,
            'las_bytes': las_bytes,
            'ifc_content': ifc_content,
            'output_format': output_format,
            'coords': None,  # Nie trzymamy w pamieci
            'batch_mode': True,
            'output_path': str(output_path)
        }

        gc.collect()
        st.rerun()

    except Exception as e:
        status.error(f"Blad batch: {e}")
        import traceback
        st.code(traceback.format_exc())


def _show_full_results():
    """Pokazuje wyniki pelnej klasyfikacji"""

    results = st.session_state['hack_full_results']
    stats = results['stats']
    elapsed = results['elapsed']
    n_points = results['n_points']

    st.markdown("---")
    st.subheader("Wyniki klasyfikacji")

    # Metryki
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Czas", f"{elapsed:.1f} s")
    with col2:
        st.metric("Predkosc", f"{n_points/elapsed:,.0f} pkt/s")
    with col3:
        classified_pct = stats.get('summary', {}).get('classified_percentage', 0)
        st.metric("Sklasyfikowane", f"{classified_pct:.1f}%")
    with col4:
        n_classes = len([k for k in stats.get('classification', {}).keys() if k != 1])
        st.metric("Wykrytych klas", n_classes)

    # Wykres rozkladu klas
    st.subheader("Rozklad klas")
    class_stats = stats.get('classification', {})

    if class_stats:
        # Sortuj po liczbie punktow
        sorted_classes = sorted(class_stats.items(), key=lambda x: x[1]['count'], reverse=True)

        data_rows = []
        colors_list = []
        for cls_id, info in sorted_classes:
            cls_info = ASPRS_CLASSES.get(cls_id, {"name": f"Klasa {cls_id}", "color": "#888"})
            data_rows.append({
                'id': cls_id,
                'name': cls_info['name'],
                'count': info['count'],
                'pct': info['percentage']
            })
            colors_list.append(cls_info['color'])

        # Wykres slupkowy
        fig = go.Figure(data=[go.Bar(
            x=[f"[{d['id']}] {d['name']}" for d in data_rows],
            y=[d['count'] for d in data_rows],
            marker_color=colors_list,
            text=[f"{d['pct']:.1f}%" for d in data_rows],
            textposition='auto'
        )])
        fig.update_layout(
            height=400,
            xaxis_tickangle=-45,
            margin=dict(b=150),
            yaxis_title="Liczba punktow"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tabela ze szczegolami
        with st.expander("Szczegolowa tabela klas"):
            import pandas as pd
            df = pd.DataFrame(data_rows)
            df.columns = ['ID', 'Nazwa klasy', 'Punktow', 'Procent']
            df['Procent'] = df['Procent'].apply(lambda x: f"{x:.2f}%")
            df['Punktow'] = df['Punktow'].apply(lambda x: f"{x:,}")
            st.dataframe(df, hide_index=True, use_container_width=True)

    # Eksport
    st.markdown("---")
    st.subheader("Pobierz wyniki")

    input_path = st.session_state['input_file']
    input_name = Path(input_path).stem

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        suffix = '.laz' if 'LAZ' in results['output_format'] else '.las'
        st.download_button(
            f"Pobierz {suffix.upper()}",
            results['las_bytes'],
            f"{input_name}_classified{suffix}",
            mime="application/octet-stream",
            use_container_width=True,
            type="primary"
        )

    with col2:
        json_report = _generate_json_report_full(results)
        st.download_button(
            "Pobierz JSON",
            json_report,
            f"{input_name}_raport.json",
            mime="application/json",
            use_container_width=True
        )

    with col3:
        txt_report = _generate_txt_report_full(results)
        st.download_button(
            "Pobierz TXT",
            txt_report,
            f"{input_name}_raport.txt",
            mime="text/plain",
            use_container_width=True
        )

    with col4:
        if results['ifc_content']:
            st.download_button(
                "Pobierz IFC",
                results['ifc_content'],
                f"{input_name}.ifc",
                mime="application/x-step",
                use_container_width=True
            )
        else:
            st.button("IFC niedostepny", disabled=True, use_container_width=True)

    with col5:
        # GeoJSON export
        geojson_content = _generate_geojson_export(results)
        if geojson_content:
            st.download_button(
                "Pobierz GeoJSON",
                geojson_content,
                f"{input_name}_classification.geojson",
                mime="application/geo+json",
                use_container_width=True
            )
        else:
            st.button("GeoJSON niedostepny", disabled=True, use_container_width=True)

    # Drugi wiersz eksportow
    st.markdown("")
    col_html, col_empty1, col_empty2, col_empty3, col_empty4 = st.columns(5)

    with col_html:
        # HTML Viewer export
        html_content = _generate_html_viewer_export(results, input_name)
        if html_content:
            st.download_button(
                "Pobierz HTML Viewer",
                html_content,
                f"{input_name}_viewer.html",
                mime="text/html",
                use_container_width=True,
                help="Interaktywna wizualizacja 3D w przegladarce"
            )
        else:
            st.button("HTML niedostepny", disabled=True, use_container_width=True)

    # Wizualizacja 3D
    st.markdown("---")
    st.subheader("Wizualizacja 3D")

    with st.expander("Podglad sklasyfikowanej chmury"):
        max_pts = st.slider("Liczba punktow", 20_000, 200_000, 100_000, 10_000)
        if st.button("Generuj wizualizacje", key="viz_full"):
            # Dla batch mode wczytaj dane z pliku
            if results.get('batch_mode') or results.get('coords') is None:
                output_path = results.get('output_path')
                if output_path and Path(output_path).exists():
                    with st.spinner("Wczytywanie danych do wizualizacji..."):
                        loader = LASLoader(output_path)
                        data = loader.load(sample_size=max_pts)
                        _visualize_classification(data['coords'], data['classification'], max_pts)
                else:
                    st.error("Plik wyjsciowy nie istnieje")
            else:
                _visualize_classification(results['coords'], results['classification'], max_pts)


def _generate_json_report_full(results: Dict) -> str:
    """Generuje raport JSON dla pelnej klasyfikacji z rozszerzonymi metrykami"""

    input_path = st.session_state['input_file']
    file_info = st.session_state['file_info']
    stats = results['stats']

    class_stats = {}
    for cls_id, info in stats.get('classification', {}).items():
        cls_info = ASPRS_CLASSES.get(cls_id, {"name": f"Klasa {cls_id}", "description": ""})
        class_stats[str(cls_id)] = {
            "name": cls_info['name'],
            "description": cls_info.get('description', ''),
            "count": int(info['count']),
            "percentage": round(info['percentage'], 4),
            "asprs_standard": cls_id <= 31
        }

    # Oblicz metryki jakosci
    classified_pct = stats.get('summary', {}).get('classified_percentage', 0)
    n_classes = len([k for k in class_stats.keys() if k != '1'])

    # Metryki jakosci (scoring)
    quality_score = min(100, (
        classified_pct * 0.4 +  # 40% za % sklasyfikowanych
        min(n_classes / 10, 1) * 30 +  # 30% za roznorodnosc klas (max 10)
        min(results['n_points'] / results['elapsed'] / 200000, 1) * 30  # 30% za predkosc
    ))

    report = {
        "metadata": {
            "tool": "CPK Chmura+ Classifier v2.0",
            "hackathon": "HackNation 2025",
            "challenge": "Chmura pod Kontrola",
            "version": "2.0.0",
            "input_file": Path(input_path).name,
            "input_format": file_info.get('version', 'LAS'),
            "batch_mode": results.get('batch_mode', False),
            "processing_time_seconds": round(results['elapsed'], 2),
            "points_per_second": int(results['n_points'] / results['elapsed'])
        },
        "quality_metrics": {
            "overall_score": round(quality_score, 1),
            "classification_coverage": round(classified_pct, 2),
            "class_diversity": n_classes,
            "processing_efficiency": round(results['n_points'] / results['elapsed'], 0),
            "recommendations": _get_quality_recommendations(classified_pct, n_classes)
        },
        "statistics": {
            "total_points": int(results['n_points']),
            "classified_points": int(results['n_points'] * classified_pct / 100),
            "unclassified_points": int(results['n_points'] * (100 - classified_pct) / 100),
            "classified_percentage": round(classified_pct, 2),
            "number_of_classes": n_classes
        },
        "classification": class_stats,
        "pipeline_steps": stats.get('steps', {}),
        "algorithms_used": {
            "noise_detection": "Statistical Outlier Removal (SOR)",
            "ground_classification": "Cloth Simulation Filter (CSF)",
            "height_computation": "Height Above Ground (HAG) grid interpolation",
            "vegetation": "HAG-based thresholding with NDVI support",
            "buildings": "RANSAC plane detection",
            "infrastructure": "Geometric feature analysis"
        }
    }

    return json.dumps(report, ensure_ascii=False, indent=2)


def _get_quality_recommendations(classified_pct: float, n_classes: int) -> list:
    """Generuje rekomendacje na podstawie wynikow"""
    recommendations = []

    if classified_pct < 50:
        recommendations.append("Niski % klasyfikacji - rozważ użycie trybu 'Wysoka jakość'")
    if classified_pct < 80:
        recommendations.append("Część punktów nieklasyfikowana - sprawdź dane wejściowe")

    if n_classes < 5:
        recommendations.append("Mało klas - dane mogą wymagać dodatkowych algorytmów")
    if n_classes >= 10:
        recommendations.append("Bogata klasyfikacja - wykryto wiele typów obiektów")

    if not recommendations:
        recommendations.append("Klasyfikacja zakończona pomyślnie - brak uwag")

    return recommendations


def _generate_geojson_export(results: Dict) -> Optional[str]:
    """Generuje eksport GeoJSON z klasyfikacji"""
    try:
        coords = results.get('coords')
        classification = results.get('classification')

        if coords is None or classification is None:
            # Dla batch mode nie mamy danych w pamieci
            return None

        from ...v2.exporters import GeoJSONExporter

        exporter = GeoJSONExporter(coords, classification)
        return exporter.to_string()
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"GeoJSON export failed: {e}")
        return None


def _generate_html_viewer_export(results: Dict, title: str) -> Optional[str]:
    """Generuje eksport HTML viewer z klasyfikacji"""
    try:
        coords = results.get('coords')
        classification = results.get('classification')

        if coords is None or classification is None:
            # Dla batch mode nie mamy danych w pamieci
            return None

        from ...v2.exporters import export_to_html_viewer
        import tempfile

        # Zapisz do tymczasowego pliku i wczytaj zawartosc
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w') as tmp:
            export_to_html_viewer(
                coords,
                classification,
                tmp.name,
                title=f"CPK Chmura+ - {title}",
                max_points=300_000  # Limit dla przegladarki
            )

        with open(tmp.name, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Usun plik tymczasowy
        Path(tmp.name).unlink(missing_ok=True)

        return html_content
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"HTML viewer export failed: {e}")
        return None


def _generate_txt_report_full(results: Dict) -> str:
    """Generuje raport TXT dla pelnej klasyfikacji z rozszerzonymi metrykami"""

    input_path = st.session_state['input_file']
    file_info = st.session_state['file_info']
    stats = results['stats']

    classified_pct = stats.get('summary', {}).get('classified_percentage', 0)
    n_classes = len([k for k in stats.get('classification', {}).keys() if k != 1])

    # Oblicz score
    quality_score = min(100, (
        classified_pct * 0.4 +
        min(n_classes / 10, 1) * 30 +
        min(results['n_points'] / results['elapsed'] / 200000, 1) * 30
    ))

    lines = [
        "=" * 70,
        "          RAPORT KLASYFIKACJI CHMURY PUNKTOW",
        "          CPK Chmura+ Classifier v2.0",
        "          HackNation 2025 - Chmura pod Kontrola",
        "=" * 70,
        "",
        "INFORMACJE O PLIKU",
        "-" * 70,
        f"  Plik wejsciowy:      {Path(input_path).name}",
        f"  Format:              LAS {file_info.get('version', '1.2')}",
        f"  Rozmiar:             {file_info.get('file_size_mb', 0):.1f} MB",
        f"  Liczba punktow:      {results['n_points']:,}",
        "",
        "WYNIKI PRZETWARZANIA",
        "-" * 70,
        f"  Czas przetwarzania:  {results['elapsed']:.1f} s ({results['elapsed']/60:.1f} min)",
        f"  Predkosc:            {results['n_points']/results['elapsed']:,.0f} pkt/s",
        f"  Sklasyfikowano:      {classified_pct:.1f}%",
        f"  Wykrytych klas:      {n_classes}",
        f"  Tryb batch:          {'Tak' if results.get('batch_mode') else 'Nie'}",
        "",
        "METRYKI JAKOSCI",
        "-" * 70,
        f"  Ogolna ocena:        {quality_score:.0f}/100",
        f"  Pokrycie:            {classified_pct:.1f}%",
        f"  Roznorodnosc:        {n_classes} klas",
        f"  Wydajnosc:           {results['n_points']/results['elapsed']:,.0f} pkt/s",
        "",
        "ROZKLAD KLASYFIKACJI",
        "-" * 70,
        "",
        "  ID   Klasa                        Punkty        Procent",
        "  " + "-" * 60,
    ]

    for cls_id, info in sorted(stats.get('classification', {}).items(),
                                key=lambda x: x[1]['count'], reverse=True):
        cls_info = ASPRS_CLASSES.get(cls_id, {"name": f"Klasa {cls_id}"})
        lines.append(f"  [{cls_id:2d}] {cls_info['name']:25s} {info['count']:>12,} {info['percentage']:>10.2f}%")

    lines.extend([
        "",
        "UZYTE ALGORYTMY",
        "-" * 70,
        "  - Detekcja szumu:     Statistical Outlier Removal (SOR)",
        "  - Klasyfikacja gruntu: Cloth Simulation Filter (CSF)",
        "  - Wysokosc nad gruntem: HAG grid interpolation",
        "  - Roslinnosc:         HAG thresholding + NDVI",
        "  - Budynki:            RANSAC plane detection",
        "  - Infrastruktura:     Geometric feature analysis",
        "",
        "OPIS WYKRYTYCH KLAS",
        "-" * 70,
        ""
    ])

    used_classes = list(stats.get('classification', {}).keys())
    for cls_id in sorted(used_classes):
        if cls_id in ASPRS_CLASSES:
            info = ASPRS_CLASSES[cls_id]
            lines.append(f"  [{cls_id:2d}] {info['name']}")
            lines.append(f"       {info.get('description', '')}")
            lines.append("")

    recommendations = _get_quality_recommendations(classified_pct, n_classes)
    lines.extend([
        "REKOMENDACJE",
        "-" * 70
    ])
    for rec in recommendations:
        lines.append(f"  * {rec}")

    lines.extend([
        "",
        "=" * 70,
        "Wygenerowano przez CPK Chmura+ Classifier v2.0",
        "HackNation 2025 - Centralny Port Komunikacyjny",
        "=" * 70
    ])

    return "\n".join(lines)


def _render_grid_mode():
    """Tryb kwadratow - zaawansowany"""

    st.subheader("Tryb kwadratow (zaawansowany)")

    st.info("Ten tryb pozwala wybrac konkretne fragmenty chmury do klasyfikacji. "
            "Przydatny dla bardzo duzych plikow lub gdy chcesz przetestowac klasyfikacje na wybranym obszarze.")

    input_path = st.session_state['input_file']

    # Krok 1: Generuj siatke
    col1, col2 = st.columns(2)
    with col1:
        target_pts = st.slider(
            "Punktow na kwadrat",
            100_000, 2_000_000, 500_000, 50_000,
            help="Wiecej punktow = dokladniejsza klasyfikacja"
        )
    with col2:
        if 'hack_grid' in st.session_state:
            stats = st.session_state['hack_grid'].get_statistics()
            st.metric("Siatka", stats['grid_dimensions'])

    if st.button("Generuj siatke", type="secondary"):
        with st.spinner("Generowanie siatki..."):
            if 'hack_data' not in st.session_state:
                loader = LASLoader(input_path)
                st.session_state['hack_data'] = loader.load()

            data = st.session_state['hack_data']
            grid = GridManager(data['coords'], target_points_per_square=target_pts)
            grid.create_grid()
            st.session_state['hack_grid'] = grid

            if 'hack_results' in st.session_state:
                del st.session_state['hack_results']
        st.rerun()

    if 'hack_grid' not in st.session_state:
        return

    # Krok 2: Mapa i wybor
    grid = st.session_state['hack_grid']
    squares = grid.get_squares()
    non_empty = [sq for sq in squares if sq.point_count > 0]

    if not non_empty:
        st.error("Wszystkie kwadraty sa puste")
        return

    # Mapa 2D
    fig = go.Figure()
    for sq in non_empty:
        x_min, x_max = sq.bounds['x']
        y_min, y_max = sq.bounds['y']
        fig.add_trace(go.Scatter(
            x=[x_min, x_max, x_max, x_min, x_min],
            y=[y_min, y_min, y_max, y_max, y_min],
            mode='lines+text',
            text=[str(sq.square_id), '', '', '', ''],
            textposition='middle center',
            line=dict(color='#1976D2', width=2),
            fill='toself',
            fillcolor='rgba(25, 118, 210, 0.2)',
            name=f"Kw. {sq.square_id}",
            hovertemplate=f"<b>Kwadrat {sq.square_id}</b><br>{sq.point_count:,} pkt<extra></extra>"
        ))

    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="X [m]",
        yaxis_title="Y [m]",
        xaxis=dict(scaleanchor="y", scaleratio=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Wybor kwadratow
    all_ids = [sq.square_id for sq in non_empty]
    selected = st.multiselect(
        "Wybierz kwadraty:",
        all_ids,
        default=all_ids[:min(3, len(all_ids))]
    )

    if not selected:
        st.warning("Wybierz co najmniej 1 kwadrat")
        return

    selected_pts = sum(sq.point_count for sq in non_empty if sq.square_id in selected)
    st.metric("Punktow do przetworzenia", f"{selected_pts:,}")

    if st.button("KLASYFIKUJ WYBRANE", type="primary", use_container_width=True):
        _run_grid_classification(grid, selected)

    if 'hack_results' in st.session_state:
        _show_grid_results()


def _run_grid_classification(grid, selected_ids: List[int]):
    """Klasyfikacja wybranych kwadratow"""

    progress = st.progress(0)
    status = st.empty()
    start = time.time()

    results = []
    all_indices = []
    all_classification = []

    data = st.session_state['hack_data']
    total = len(selected_ids)

    for i, sq_id in enumerate(selected_ids):
        status.info(f"Kwadrat {sq_id} ({i+1}/{total})...")
        progress.progress(i / total)

        sq = grid.get_square_by_id(sq_id)
        if sq is None:
            continue

        indices = grid.get_square_indices(sq)
        if len(indices) == 0:
            continue

        coords = data['coords'][indices]
        colors = data['colors'][indices] if data['colors'] is not None else None
        intensity = data['intensity'][indices] if data['intensity'] is not None else None

        config = PipelineConfig(
            detect_noise=True,
            classify_ground=True,
            classify_vegetation=True,
            detect_buildings=True,
            detect_infrastructure=True,
            use_fast_noise_detection=(len(coords) > 500_000)
        )

        try:
            pipeline = ProfessionalPipeline(coords, colors, intensity, config)
            classification, stats = pipeline.run()

            results.append({
                'square_id': sq_id,
                'indices': indices,
                'classification': classification,
                'stats': stats,
                'n_points': len(coords)
            })

            all_indices.append(indices)
            all_classification.append(classification)

        except Exception as e:
            st.warning(f"Blad w kwadracie {sq_id}: {e}")

        del pipeline
        gc.collect()

    progress.progress(100)
    elapsed = time.time() - start
    total_pts = sum(r['n_points'] for r in results)

    status.success(f"Gotowe! {total_pts:,} punktow w {elapsed:.1f}s")

    st.session_state['hack_results'] = results
    st.session_state['hack_elapsed'] = elapsed
    st.session_state['hack_all_indices'] = all_indices
    st.session_state['hack_all_classification'] = all_classification

    st.rerun()


def _show_grid_results():
    """Wyswietla wyniki trybu kwadratow"""

    results = st.session_state['hack_results']
    elapsed = st.session_state['hack_elapsed']

    if not results:
        return

    st.markdown("---")
    st.subheader("Wyniki")

    total_pts = sum(r['n_points'] for r in results)

    # Agreguj statystyki
    class_counts = {}
    for r in results:
        for cls_id, info in r['stats'].get('classification', {}).items():
            class_counts[cls_id] = class_counts.get(cls_id, 0) + info['count']

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Czas", f"{elapsed:.1f} s")
    with col2:
        st.metric("Punktow", f"{total_pts:,}")
    with col3:
        st.metric("Klas", len([k for k in class_counts if k != 1]))

    # Eksport
    st.subheader("Eksport")

    input_path = st.session_state['input_file']
    input_name = Path(input_path).stem

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Generuj LAS", type="primary"):
            _generate_las_export_grid()

    if 'hack_las_bytes' in st.session_state:
        with col1:
            st.download_button(
                "Pobierz LAS",
                st.session_state['hack_las_bytes'],
                f"{input_name}_classified.las",
                use_container_width=True
            )

    with col2:
        json_report = json.dumps({
            "squares": len(results),
            "points": total_pts,
            "time": elapsed,
            "classes": {str(k): v for k, v in class_counts.items()}
        }, indent=2)
        st.download_button("JSON", json_report, f"{input_name}_raport.json")

    with col3:
        txt_lines = [f"Kwadraty: {len(results)}", f"Punkty: {total_pts:,}", f"Czas: {elapsed:.1f}s"]
        for cls_id, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            name = ASPRS_CLASSES.get(cls_id, {"name": f"Klasa {cls_id}"})['name']
            txt_lines.append(f"[{cls_id}] {name}: {count:,}")
        st.download_button("TXT", "\n".join(txt_lines), f"{input_name}_raport.txt")


def _generate_las_export_grid():
    """Eksport LAS dla trybu kwadratow"""

    with st.spinner("Generowanie LAS..."):
        try:
            input_path = st.session_state['input_file']
            all_indices = st.session_state['hack_all_indices']
            all_classification = st.session_state['hack_all_classification']

            with laspy.open(input_path) as src:
                las_orig = src.read()
                full_classification = np.ones(len(las_orig.x), dtype=np.uint8)

                for indices, classification in zip(all_indices, all_classification):
                    full_classification[indices] = classification.astype(np.uint8)

                max_class = int(full_classification.max())

                if max_class > 31:
                    has_rgb = hasattr(las_orig, 'red') and las_orig.red is not None
                    point_format = 7 if has_rgb else 6
                    header = laspy.LasHeader(point_format=point_format, version="1.4")
                    header.scales = las_orig.header.scales
                    header.offsets = las_orig.header.offsets
                    las = laspy.LasData(header)
                    las.x = las_orig.x
                    las.y = las_orig.y
                    las.z = las_orig.z
                    las.intensity = las_orig.intensity
                    if has_rgb:
                        las.red = las_orig.red
                        las.green = las_orig.green
                        las.blue = las_orig.blue
                    las.classification = full_classification
                else:
                    las = las_orig
                    las.classification = full_classification

                with tempfile.NamedTemporaryFile(suffix='.las', delete=False) as tmp:
                    las.write(tmp.name)
                    with open(tmp.name, 'rb') as f:
                        st.session_state['hack_las_bytes'] = f.read()

            st.success("LAS wygenerowany!")
            st.rerun()

        except Exception as e:
            st.error(f"Blad: {e}")


def _visualize_classification(coords: np.ndarray, classification: np.ndarray, max_pts: int):
    """Wizualizacja 3D sklasyfikowanej chmury"""

    try:
        # Sampling
        if len(coords) > max_pts:
            idx = np.random.choice(len(coords), max_pts, replace=False)
            coords = coords[idx]
            classification = classification[idx]

        # Kolory
        colors = [ASPRS_CLASSES.get(int(c), {"color": "#888"})['color'] for c in classification]

        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(size=1, color=colors, opacity=0.8),
            hoverinfo='skip'
        )])

        fig.update_layout(
            scene=dict(
                aspectmode='data',
                xaxis_title='X [m]',
                yaxis_title='Y [m]',
                zaxis_title='Z [m]'
            ),
            height=600,
            margin=dict(l=0, r=0, t=30, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.success(f"Wyswietlono {len(coords):,} punktow")

        # Legenda
        unique_classes = np.unique(classification)
        st.markdown("**Legenda:**")
        cols = st.columns(min(len(unique_classes), 6))
        for i, cls_id in enumerate(sorted(unique_classes)):
            info = ASPRS_CLASSES.get(int(cls_id), {"name": f"Klasa {cls_id}", "color": "#888"})
            with cols[i % len(cols)]:
                st.markdown(f"""
                <div style='display: flex; align-items: center;'>
                    <div style='width: 15px; height: 15px; background: {info["color"]};
                                border-radius: 3px; margin-right: 5px;'></div>
                    <span style='font-size: 12px;'>[{cls_id}] {info["name"]}</span>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Blad wizualizacji: {e}")
