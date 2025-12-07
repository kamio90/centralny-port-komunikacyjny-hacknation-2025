"""
Pełna Klasyfikacja - prosty i stabilny
"""

import streamlit as st
from pathlib import Path
import time
import json
from typing import Dict, Any
import logging
import gc

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import laspy

from ...v2 import LASLoader
from ...v2.pipeline import ProfessionalPipeline, PipelineConfig
from ...config import PATHS

logger = logging.getLogger(__name__)

# Kolory klas ASPRS
ASPRS_CLASSES = {
    1: {"name": "Nieklasyfikowane", "color": "#9E9E9E"},
    2: {"name": "Grunt", "color": "#8D6E63"},
    3: {"name": "Roślinność niska", "color": "#AED581"},
    4: {"name": "Roślinność średnia", "color": "#66BB6A"},
    5: {"name": "Roślinność wysoka", "color": "#2E7D32"},
    6: {"name": "Budynek", "color": "#D7CCC8"},
    7: {"name": "Szum", "color": "#F44336"},
    9: {"name": "Woda", "color": "#29B6F6"},
    18: {"name": "Tory kolejowe", "color": "#6D4C41"},
    19: {"name": "Linie energetyczne", "color": "#FDD835"},
    20: {"name": "Słupy trakcyjne", "color": "#546E7A"},
    21: {"name": "Peron kolejowy", "color": "#78909C"},
    30: {"name": "Droga", "color": "#455A64"},
    32: {"name": "Krawężnik", "color": "#B0BEC5"},
    35: {"name": "Znak drogowy", "color": "#FF5722"},
    36: {"name": "Bariera drogowa", "color": "#90A4AE"},
    40: {"name": "Ściany budynków", "color": "#BCAAA4"},
    41: {"name": "Dachy budynków", "color": "#A1887F"},
}


def render_classification(n_threads: int = 1):
    """Główna funkcja klasyfikacji"""

    st.subheader("🎯 Pełna Klasyfikacja")

    if 'input_file' not in st.session_state:
        st.warning("⚠️ Najpierw wczytaj plik")
        return

    input_path = st.session_state['input_file']
    file_info = st.session_state['file_info']
    n_points = file_info['n_points']

    st.success(f"✅ Gotowy: **{Path(input_path).name}** ({n_points:,} punktów)")

    st.markdown("---")

    # Konfiguracja
    st.subheader("⚙️ Konfiguracja")

    col1, col2 = st.columns(2)

    with col1:
        default_name = f"{Path(input_path).stem}_classified.las"
        output_name = st.text_input("Plik wyjściowy", value=default_name)
        output_path = PATHS.OUTPUT_DIR / output_name

        mode = st.selectbox(
            "Tryb",
            ["Pełna jakość", "Szybki (bez budynków)", "Ultra-szybki"]
        )

    with col2:
        # Estymacja czasu
        if mode == "Ultra-szybki":
            est_time = n_points / 400_000
        elif mode == "Szybki (bez budynków)":
            est_time = n_points / 200_000
        else:
            est_time = n_points / 100_000

        st.metric("⏱️ Estymowany czas", f"{int(est_time // 60)} min {int(est_time % 60)} s")
        st.metric("🚀 Prędkość", f"~{int(n_points/est_time):,} pkt/s")

    st.markdown("---")

    # Przycisk START
    if st.button("🚀 ROZPOCZNIJ KLASYFIKACJĘ", type="primary", use_container_width=True):
        _run_classification(input_path, str(output_path), mode, n_points)

    # Wyniki
    if 'classification_results' in st.session_state:
        _render_results()


def _run_classification(input_path: str, output_path: str, mode: str, n_points: int):
    """Uruchamia klasyfikację"""

    progress = st.progress(0)
    status = st.empty()
    start_time = time.time()

    try:
        # 1. Wczytaj
        status.info("📥 Wczytywanie danych...")
        progress.progress(10)

        loader = LASLoader(input_path)
        data = loader.load()

        # 2. Konfiguracja
        config = PipelineConfig(
            detect_noise=True,
            classify_ground=True,
            classify_vegetation=True,
            detect_buildings=(mode == "Pełna jakość"),
            detect_infrastructure=True,
            use_fast_noise_detection=(mode != "Pełna jakość")
        )

        # 3. Pipeline
        status.info("🔄 Klasyfikacja w toku...")
        progress.progress(30)

        pipeline = ProfessionalPipeline(
            coords=data['coords'],
            colors=data['colors'],
            intensity=data['intensity'],
            config=config
        )

        def progress_cb(step, pct, msg):
            progress.progress(min(30 + int(pct * 0.6), 90))
            status.info(f"🔄 {step}: {msg}")

        classification, stats = pipeline.run(progress_callback=progress_cb)

        # 4. Zapisz
        status.info("💾 Zapisywanie...")
        progress.progress(95)

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

            if output_path.endswith('.laz'):
                las.write(output_path, laz_backend=laspy.compression.LazBackend.Laszip)
            else:
                las.write(output_path)

        progress.progress(100)
        elapsed = time.time() - start_time

        status.success(f"✅ Gotowe! Czas: {elapsed:.1f}s ({n_points/elapsed:,.0f} pkt/s)")
        st.balloons()

        # Zapisz wyniki
        st.session_state['classification_results'] = {
            'stats': stats,
            'elapsed': elapsed,
            'output_path': output_path,
            'input_path': input_path,
            'n_points': n_points
        }

        del classification, pipeline
        gc.collect()

        st.rerun()

    except Exception as e:
        status.error(f"❌ Błąd: {e}")
        logger.exception("Classification error")


def _render_results():
    """Wyświetla wyniki"""

    results = st.session_state['classification_results']
    stats = results['stats']
    elapsed = results['elapsed']
    n_points = results['n_points']
    output_path = results['output_path']
    input_path = results['input_path']

    st.markdown("---")
    st.subheader("📊 Wyniki")

    # Metryki
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⏱️ Czas", f"{elapsed:.1f} s")
    with col2:
        st.metric("🚀 Prędkość", f"{n_points/elapsed:,.0f} pkt/s")
    with col3:
        pct = stats.get('summary', {}).get('classified_percentage', 0)
        st.metric("✅ Sklasyfikowane", f"{pct:.1f}%")
    with col4:
        n_cls = len([k for k in stats.get('classification', {}).keys() if k != 1])
        st.metric("🎯 Klasy", n_cls)

    # Rozkład klas
    st.subheader("📈 Rozkład klas")

    classification_stats = stats.get('classification', {})

    if classification_stats:
        data = []
        for class_id, info in sorted(classification_stats.items(), key=lambda x: x[1]['count'], reverse=True):
            cls_info = ASPRS_CLASSES.get(class_id, {"name": f"Klasa {class_id}", "color": "#888"})
            data.append({
                'Klasa': f"{class_id}: {cls_info['name']}",
                'Punkty': info['count'],
                'Procent': info['percentage'],
                'Kolor': cls_info['color']
            })

        df = pd.DataFrame(data)

        # Wykres
        fig = go.Figure(data=[go.Bar(
            x=df['Klasa'],
            y=df['Procent'],
            marker_color=df['Kolor'],
            text=df['Procent'].apply(lambda x: f"{x:.1f}%"),
            textposition='auto'
        )])
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        # Tabela
        with st.expander("📋 Szczegóły"):
            st.dataframe(df[['Klasa', 'Punkty', 'Procent']], hide_index=True)

    # Wizualizacja 3D
    st.subheader("🎨 Wizualizacja 3D")

    with st.expander("Podgląd sklasyfikowanej chmury"):
        max_pts = st.slider("Max punktów", 50_000, 200_000, 100_000, 10_000)

        if st.button("🔄 Generuj wizualizację"):
            _visualize_results(output_path, max_pts)

    # Eksport
    st.markdown("---")
    st.subheader("💾 Pobierz")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if Path(output_path).exists():
            with open(output_path, "rb") as f:
                st.download_button("📥 LAS", f, Path(output_path).name, use_container_width=True)

    with col2:
        txt = _gen_txt_report(stats, n_points, elapsed, input_path)
        st.download_button("📄 TXT", txt, f"{Path(input_path).stem}_raport.txt", use_container_width=True)

    with col3:
        json_str = _gen_json_report(stats, n_points, elapsed, input_path, output_path)
        st.download_button("📊 JSON", json_str, f"{Path(input_path).stem}_raport.json", use_container_width=True)

    with col4:
        if st.button("🏗️ IFC", use_container_width=True):
            _gen_ifc(output_path, input_path)


def _visualize_results(output_path: str, max_pts: int):
    """Wizualizacja 3D"""

    try:
        with laspy.open(output_path) as f:
            las = f.read()
            coords = np.vstack([las.x, las.y, las.z]).T
            classification = np.array(las.classification)

        # Sampling
        if len(coords) > max_pts:
            idx = np.random.choice(len(coords), max_pts, replace=False)
            coords = coords[idx]
            classification = classification[idx]

        # Kolory
        colors = []
        for cls in classification:
            c = ASPRS_CLASSES.get(cls, {"color": "#888"})['color']
            colors.append(c)

        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(size=1, color=colors, opacity=0.7),
            hoverinfo='skip'
        )])
        fig.update_layout(scene=dict(aspectmode='data'), height=600, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"Wyświetlono {len(coords):,} punktów")

    except Exception as e:
        st.error(f"Błąd: {e}")


def _gen_txt_report(stats, n_points, elapsed, input_path):
    """Generuje raport TXT"""
    lines = [
        "=" * 50,
        "RAPORT KLASYFIKACJI - CPK Chmura+",
        "=" * 50,
        f"Plik: {Path(input_path).name}",
        f"Punkty: {n_points:,}",
        f"Czas: {elapsed:.1f}s",
        f"Prędkość: {n_points/elapsed:,.0f} pkt/s",
        "",
        "Rozkład klas:",
        "-" * 50
    ]
    for cls_id, info in sorted(stats.get('classification', {}).items()):
        name = ASPRS_CLASSES.get(cls_id, {"name": f"Klasa {cls_id}"})['name']
        lines.append(f"  [{cls_id:2d}] {name:20s}: {info['count']:>10,} ({info['percentage']:>5.2f}%)")
    return "\n".join(lines)


def _gen_json_report(stats, n_points, elapsed, input_path, output_path):
    """Generuje raport JSON"""
    report = {
        "plik": Path(input_path).name,
        "punkty": n_points,
        "czas_s": round(elapsed, 2),
        "klasy": {
            str(k): {"nazwa": ASPRS_CLASSES.get(k, {"name": f"Klasa {k}"})['name'], **v}
            for k, v in stats.get('classification', {}).items()
        }
    }
    return json.dumps(report, ensure_ascii=False, indent=2)


def _gen_ifc(output_path, input_path):
    """Generuje IFC"""
    try:
        from ...v2.core import IFCExporter

        with st.spinner("Generowanie IFC..."):
            with laspy.open(output_path) as f:
                las = f.read()
                coords = np.vstack([las.x, las.y, las.z]).T
                classification = np.array(las.classification)

            exporter = IFCExporter(coords, classification, f"CPK - {Path(input_path).stem}")
            ifc_path = Path(output_path).parent / f"{Path(input_path).stem}.ifc"
            exporter.export(str(ifc_path))

            with open(ifc_path, 'r') as f:
                st.download_button("📥 Pobierz IFC", f.read(), ifc_path.name)

        st.success("✅ IFC wygenerowany")

    except Exception as e:
        st.error(f"Błąd IFC: {e}")
