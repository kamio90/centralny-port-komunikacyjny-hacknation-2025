"""
Komponent ładowania plików LAS/LAZ
"""

import streamlit as st
from pathlib import Path
import tempfile
import glob
from typing import Optional, Dict, Any

from ...v2 import LASLoader
from ...config import PATHS


def find_las_files() -> list:
    """Znajduje pliki LAS/LAZ w katalogu data/"""
    las_files = []
    data_dir = PATHS.DATA_DIR

    if data_dir.exists():
        for ext in ['*.las', '*.laz', '*.LAS', '*.LAZ']:
            las_files.extend(data_dir.glob(ext))

    return sorted(las_files, key=lambda x: x.name)


def render_file_loader() -> Optional[Dict[str, Any]]:
    """File loader - 3 metody"""

    st.subheader("📁 Wczytaj chmurę punktów")

    # Znajdź pliki w data/
    available_files = find_las_files()

    # 3 metody
    method = st.radio(
        "Wybierz metodę:",
        [
            f"📂 Z folderu data/ ({len(available_files)} plików)",
            "📤 Upload pliku",
            "✏️ Podaj ścieżkę"
        ],
        horizontal=True
    )

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
