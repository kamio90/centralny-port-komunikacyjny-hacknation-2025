"""
Komponent ładowania plików LAS/LAZ
Z trybem DEMO dla szybkiego pokazu
"""

import streamlit as st
from pathlib import Path
import tempfile
import glob
from typing import Optional, Dict, Any
import numpy as np

from ...v2 import LASLoader
from ...config import PATHS


# Demo data cache
_DEMO_CACHE = {}


def _run_demo_mode():
    """Uruchamia tryb DEMO z automatyczna klasyfikacja"""
    import time

    # Znajdź plik testowy
    data_files = find_las_files()
    if not data_files:
        st.error("Brak plików w folderze data/ do DEMO")
        return

    demo_file = data_files[0]  # Użyj pierwszego dostępnego

    # Progress
    progress = st.progress(0)
    status = st.empty()

    try:
        # 1. Wczytaj próbkę
        status.info("🔄 Wczytywanie danych demo...")
        progress.progress(10)

        loader = LASLoader(str(demo_file))
        data = loader.load(sample_size=100_000)  # 100k punktów

        progress.progress(30)
        status.info(f"✅ Wczytano {data['n_points']:,} punktów")

        # 2. Klasyfikacja
        status.info("🔄 Klasyfikacja w toku...")
        progress.progress(40)

        from ...v2.pipeline import ProfessionalPipeline, PipelineConfig

        config = PipelineConfig(
            detect_noise=True,
            classify_ground=True,
            classify_vegetation=True,
            detect_buildings=True,
            detect_infrastructure=True,
            use_fast_noise_detection=True
        )

        pipeline = ProfessionalPipeline(
            data['coords'],
            data['colors'],
            data['intensity'],
            config
        )

        classification, stats = pipeline.run()

        progress.progress(90)

        # 3. Zapisz do session state
        st.session_state['input_file'] = str(demo_file)
        st.session_state['file_info'] = {
            'n_points': data['n_points'],
            'file_size_mb': demo_file.stat().st_size / (1024*1024),
            'has_rgb': data['colors'] is not None,
            'version': '1.2',
            'bounds': data['bounds']
        }

        # Zapisz wyniki demo
        st.session_state['demo_results'] = {
            'coords': data['coords'],
            'classification': classification,
            'stats': stats,
            'n_points': data['n_points']
        }

        progress.progress(100)
        status.success(f"✅ DEMO gotowe! Wykryto {len(np.unique(classification))} klas")

        # Auto-redirect info
        time.sleep(1)

    except Exception as e:
        status.error(f"❌ Błąd DEMO: {e}")
        import traceback
        st.code(traceback.format_exc())


def find_las_files() -> list:
    """Znajduje pliki LAS/LAZ w katalogu data/"""
    las_files = []
    data_dir = PATHS.DATA_DIR

    if data_dir.exists():
        for ext in ['*.las', '*.laz', '*.LAS', '*.LAZ']:
            las_files.extend(data_dir.glob(ext))

    return sorted(las_files, key=lambda x: x.name)


def render_file_loader() -> Optional[Dict[str, Any]]:
    """File loader - 4 metody z trybem DEMO"""

    # Znajdź pliki w data/
    available_files = find_las_files()

    # Welcome experience dla nowych użytkowników
    if 'input_file' not in st.session_state:
        # Quick start section
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 25px 30px; border-radius: 12px; margin-bottom: 25px;
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);'>
            <h2 style='color: white; margin: 0 0 10px 0;'>Szybki start</h2>
            <p style='color: #e0e0e0; margin: 0 0 15px 0; font-size: 15px;'>
                Wyprobuj klasyfikacje na danych testowych lub wczytaj wlasny plik
            </p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("DEMO", type="primary", use_container_width=True, help="100k punktow - ok. 10 sekund"):
                _run_demo_mode()
                st.rerun()
            st.caption("Szybki pokaz")

        with col2:
            if available_files:
                if st.button("Pierwszy plik", use_container_width=True, help=f"Uzyj {available_files[0].name}"):
                    st.session_state['input_file'] = str(available_files[0])
                    st.session_state['file_info'] = LASLoader.get_file_info(str(available_files[0]))
                    st.rerun()
                st.caption(f"{len(available_files)} plikow w data/")

        # Workflow guide
        st.markdown("---")
        st.markdown("**Jak to dziala:**")

        wf1, wf2, wf3, wf4 = st.columns(4)
        with wf1:
            st.markdown("""
            <div style='text-align: center; padding: 15px; background: #f5f5f5; border-radius: 8px;'>
                <div style='font-size: 32px; margin-bottom: 8px;'>1️⃣</div>
                <div style='font-weight: 600;'>Wczytaj</div>
                <div style='font-size: 12px; color: #666;'>LAS/LAZ plik</div>
            </div>
            """, unsafe_allow_html=True)
        with wf2:
            st.markdown("""
            <div style='text-align: center; padding: 15px; background: #f5f5f5; border-radius: 8px;'>
                <div style='font-size: 32px; margin-bottom: 8px;'>2️⃣</div>
                <div style='font-weight: 600;'>Podglad</div>
                <div style='font-size: 12px; color: #666;'>Wizualizacja 3D</div>
            </div>
            """, unsafe_allow_html=True)
        with wf3:
            st.markdown("""
            <div style='text-align: center; padding: 15px; background: #f5f5f5; border-radius: 8px;'>
                <div style='font-size: 32px; margin-bottom: 8px;'>3️⃣</div>
                <div style='font-weight: 600;'>Klasyfikuj</div>
                <div style='font-size: 12px; color: #666;'>Automatycznie</div>
            </div>
            """, unsafe_allow_html=True)
        with wf4:
            st.markdown("""
            <div style='text-align: center; padding: 15px; background: #f5f5f5; border-radius: 8px;'>
                <div style='font-size: 32px; margin-bottom: 8px;'>4️⃣</div>
                <div style='font-weight: 600;'>Eksportuj</div>
                <div style='font-size: 12px; color: #666;'>LAS, IFC, GeoJSON</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

    st.subheader("Wczytaj chmure punktow")

    # 4 metody
    method = st.radio(
        "Wybierz metodę:",
        [
            f"📂 Z folderu data/ ({len(available_files)} plików)",
            "📤 Upload pliku",
            "✏️ Podaj ścieżkę",
            "🎮 Tryb DEMO"
        ],
        horizontal=True
    )

    # === METODA 4: DEMO ===
    if "DEMO" in method:
        st.info("Tryb DEMO wczytuje 100k punktów z pliku testowego i klasyfikuje je automatycznie.")
        if st.button("▶️ Uruchom DEMO", type="primary"):
            _run_demo_mode()
            st.rerun()
        return None

    temp_path = None

    # === METODA 1: Z folderu data/ ===
    if "data/" in method:
        if available_files:
            # Lista plików z rozmiarem
            file_options = []
            for f in available_files:
                size = f.stat().st_size
                if size > 1024**3:
                    size_str = f"{size / 1024**3:.2f} GB"
                else:
                    size_str = f"{size / 1024**2:.0f} MB"
                file_options.append(f"{f.name} ({size_str})")

            selected_idx = st.selectbox(
                "Wybierz plik:",
                range(len(available_files)),
                format_func=lambda i: file_options[i]
            )

            if selected_idx is not None:
                temp_path = available_files[selected_idx]
        else:
            st.warning("⚠️ Brak plików w folderze `data/`")
            st.caption("Skopiuj pliki LAS/LAZ do folderu `data/` lub użyj innej metody")

    # === METODA 2: Upload ===
    elif "Upload" in method:
        st.caption("⚠️ Limit przeglądarki: ~500MB. Dla większych plików użyj innej metody.")

        uploaded = st.file_uploader(
            "Przeciągnij plik lub kliknij",
            type=['las', 'laz'],
            help="Drag & drop lub wybierz z dysku"
        )

        if uploaded:
            with st.spinner("Zapisywanie..."):
                suffix = Path(uploaded.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getbuffer())
                    temp_path = Path(tmp.name)
                st.session_state['_uploaded_name'] = uploaded.name

        elif '_uploaded_temp' in st.session_state:
            p = Path(st.session_state['_uploaded_temp'])
            if p.exists():
                temp_path = p

    # === METODA 3: Ścieżka ręczna ===
    else:
        st.caption("💡 Dla plików poza folderem data/ lub na innych dyskach")

        path_input = st.text_input(
            "Pełna ścieżka:",
            placeholder="/Users/.../chmura.las"
        )

        if path_input:
            p = Path(path_input)
            if p.exists() and p.suffix.lower() in ['.las', '.laz']:
                temp_path = p
            else:
                st.error("❌ Plik nie istnieje lub zły format")

    # === WYŚWIETL INFO O PLIKU ===
    if temp_path and temp_path.exists():
        # Nazwa pliku
        display_name = st.session_state.get('_uploaded_name', temp_path.name)
        st.success(f"✅ **{display_name}**")

        # Cache - sprawdź czy ten sam plik
        cache_key = str(temp_path)
        if st.session_state.get('_file_cache_key') != cache_key:
            with st.spinner("Analizuję plik..."):
                try:
                    file_info = LASLoader.get_file_info(str(temp_path))
                    st.session_state['_file_cache_key'] = cache_key
                    st.session_state['_file_cache_info'] = file_info
                except Exception as e:
                    st.error(f"❌ Błąd: {e}")
                    return None
        else:
            file_info = st.session_state['_file_cache_info']

        # Statystyki
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            n = file_info['n_points']
            if n > 1_000_000_000:
                st.metric("📊 Punkty", f"{n/1e9:.2f}B")
            elif n > 1_000_000:
                st.metric("📊 Punkty", f"{n/1e6:.1f}M")
            else:
                st.metric("📊 Punkty", f"{n:,}")

        with col2:
            mb = file_info['file_size_mb']
            if mb > 1000:
                st.metric("💾 Rozmiar", f"{mb/1024:.2f} GB")
            else:
                st.metric("💾 Rozmiar", f"{mb:.0f} MB")

        with col3:
            st.metric("🎨 RGB", "✅" if file_info['has_rgb'] else "❌")

        with col4:
            st.metric("📄 LAS", file_info.get('las_version', '1.4'))

        # Granice
        with st.expander("📏 Granice"):
            bounds = file_info['bounds']
            c1, c2, c3 = st.columns(3)
            with c1:
                st.caption(f"**X:** {bounds['x'][0]:.0f} → {bounds['x'][1]:.0f} m")
            with c2:
                st.caption(f"**Y:** {bounds['y'][0]:.0f} → {bounds['y'][1]:.0f} m")
            with c3:
                st.caption(f"**Z:** {bounds['z'][0]:.0f} → {bounds['z'][1]:.0f} m")

        # Zapisz do session
        st.session_state['input_file'] = str(temp_path)
        st.session_state['file_info'] = file_info

        st.info("👉 Przejdź do **Podgląd** lub **Klasyfikacja**")

        return {'path': str(temp_path), 'info': file_info}

    return None
