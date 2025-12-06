"""
CPK - Klasyfikator Chmur Punktów v2.0

Aplikacja Streamlit do automatycznej klasyfikacji elementów infrastruktury
na podstawie chmur punktów LAS/LAZ.

Centralny Port Komunikacyjny - HackNation2025
"""

import streamlit as st
import logging
from pathlib import Path
import time
import glob

# Nowe v2 API
from src.v2 import (
    ClassificationPipeline,
    LASLoader,
    ClassifierRegistry
)

# Konfiguracja loggingu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_las_files(directory: str = "data") -> list:
    """Znajduje pliki LAS/LAZ w katalogu"""
    las_files = []
    if Path(directory).exists():
        las_files.extend(glob.glob(f"{directory}/*.las"))
        las_files.extend(glob.glob(f"{directory}/*.laz"))
    return sorted([Path(f).absolute() for f in las_files])

# Konfiguracja strony
st.set_page_config(
    page_title="Chmura+ | Klasyfikator Chmur Punktów",
    page_icon="assets/favicon-chmura.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STYLE CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0A1E42;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0A1E42;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e6f0ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0A1E42;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .logo-container {
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# NAGŁÓWEK
# ============================================================================

# Logo Chmura+
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("assets/logo_chmura.png", use_container_width=True)

st.markdown("""
<div class="info-box">
    <b>Automatyczna klasyfikacja elementów infrastruktury</b><br>
    Wersja 2.0 - Czysta, modularna architektura | 45 klas infrastruktury | Thread-safe
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - Ustawienia
# ============================================================================

st.sidebar.header("⚙️ Ustawienia")

# Tryb pracy
demo_mode = st.sidebar.checkbox(
    "🚀 Tryb DEMO (szybszy, mniej dokładny)",
    value=False,
    help="Tryb demonstracyjny: mniejsza próbka PCA, większe kafelki"
)

# Liczba wątków
n_threads = st.sidebar.slider(
    "🔧 Liczba wątków",
    min_value=1,
    max_value=8,
    value=4,
    help="Liczba równoległych wątków do przetwarzania kafelków"
)

# Informacje o klasyfikatorach
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Zarejestrowane klasy")
all_classifiers = ClassifierRegistry.get_all()
st.sidebar.info(f"**{len(all_classifiers)} klas** gotowych do klasyfikacji")

with st.sidebar.expander("📋 Lista klas"):
    for class_id, classifier_class in sorted(all_classifiers.items()):
        classifier = classifier_class()
        st.write(f"**[{class_id:2d}]** {classifier.class_name}")

# ============================================================================
# GŁÓWNA TREŚĆ
# ============================================================================

# Tabs
tab1, tab2 = st.tabs(["📁 Wczytaj plik", "🎯 Klasyfikacja"])

# ============================================================================
# TAB 1: Wczytywanie pliku
# ============================================================================

with tab1:
    st.markdown('<div class="sub-header">📁 Wczytaj chmurę punktów</div>', unsafe_allow_html=True)

    st.info("💡 **Obsługuje pliki 10GB+** - wybierz z listy lub podaj własną ścieżkę")

    # Znajdź pliki w katalogu data/
    available_files = find_las_files("data")

    # Tryb wyboru
    selection_mode = st.radio(
        "Wybierz tryb:",
        ["📂 Z listy dostępnych plików", "✏️ Własna ścieżka"],
        horizontal=True
    )

    temp_path = None

    if selection_mode == "📂 Z listy dostępnych plików":
        if available_files:
            # Pokaż listę plików
            file_options = [f"{f.name} ({f.stat().st_size / (1024**3):.2f} GB)" for f in available_files]
            selected_idx = st.selectbox(
                "Wybierz plik:",
                range(len(available_files)),
                format_func=lambda i: file_options[i]
            )

            if selected_idx is not None:
                temp_path = available_files[selected_idx]
                st.success(f"✅ Wybrany plik: **{temp_path.name}**")
        else:
            st.warning("⚠️ Brak plików LAS/LAZ w katalogu `data/`")
            st.info("Przenieś pliki do folderu `data/` lub użyj trybu 'Własna ścieżka'")

    else:  # Własna ścieżka
        custom_path = st.text_input(
            "Podaj pełną ścieżkę do pliku:",
            placeholder="/Users/user/Downloads/moja_chmura.las"
        )

        if custom_path:
            if Path(custom_path).exists():
                temp_path = Path(custom_path)
                st.success(f"✅ Plik znaleziony: **{temp_path.name}**")
            else:
                st.error(f"❌ Plik nie istnieje: {custom_path}")

    # Jeśli plik został wybrany
    if temp_path and temp_path.exists():
        pass  # Kontynuuj dalej

        # Pobierz informacje o pliku
        with st.spinner("Analizuję plik..."):
            file_info = LASLoader.get_file_info(str(temp_path))

        # Wyświetl informacje
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📊 Liczba punktów", f"{file_info['n_points']:,}")

        with col2:
            st.metric("💾 Rozmiar pliku", f"{file_info['file_size_mb']:.1f} MB")

        with col3:
            rgb_status = "✅ Tak" if file_info['has_rgb'] else "❌ Nie"
            st.metric("🎨 Kolory RGB", rgb_status)

        # Granice
        with st.expander("📏 Granice chmury punktów"):
            bounds = file_info['bounds']
            st.write(f"**X:** {bounds['x'][0]:.2f} → {bounds['x'][1]:.2f} m "
                    f"(zakres: {bounds['x'][1]-bounds['x'][0]:.2f} m)")
            st.write(f"**Y:** {bounds['y'][0]:.2f} → {bounds['y'][1]:.2f} m "
                    f"(zakres: {bounds['y'][1]-bounds['y'][0]:.2f} m)")
            st.write(f"**Z:** {bounds['z'][0]:.2f} → {bounds['z'][1]:.2f} m "
                    f"(zakres: {bounds['z'][1]-bounds['z'][0]:.2f} m)")

        # Zapisz w session_state
        st.session_state['input_file'] = str(temp_path)
        st.session_state['file_info'] = file_info

        st.info("👉 Przejdź do zakładki **Klasyfikacja** aby rozpocząć przetwarzanie")

    else:
        st.markdown("""
        <div class="info-box">
            ℹ️ <b>Instrukcja:</b><br>
            <b>Tryb 1: Z listy</b><br>
            • Przenieś pliki LAS/LAZ do folderu <code>data/</code><br>
            • Wybierz z listy rozwijanej<br><br>
            <b>Tryb 2: Własna ścieżka</b><br>
            • Przełącz na "Własna ścieżka"<br>
            • Podaj pełną ścieżkę do pliku (obsługuje 10GB+)
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# TAB 2: Klasyfikacja
# ============================================================================

with tab2:
    st.markdown('<div class="sub-header">🎯 Klasyfikacja chmury punktów</div>', unsafe_allow_html=True)

    if 'input_file' not in st.session_state:
        st.warning("⚠️ Najpierw wczytaj plik w zakładce **Wczytaj plik**")
    else:
        input_path = st.session_state['input_file']
        file_info = st.session_state['file_info']

        st.success(f"✅ Gotowy do klasyfikacji: **{Path(input_path).name}**")

        # Nazwa pliku wyjściowego
        output_name = st.text_input(
            "Nazwa pliku wyjściowego",
            value=f"{Path(input_path).stem}_classified.las"
        )

        output_path = Path("output") / output_name
        output_path.parent.mkdir(exist_ok=True)

        # Estymacja czasu (realistyczna dla hackatonu!)
        n_points = file_info['n_points']
        if demo_mode:
            # DEMO: ~28 kafelków × 20s/kafelek = ~9 minut
            est_time = (n_points / 10_000_000) * 20  # 10M punktów na kafelek, 20s/kafelek
        else:
            # NORMAL: wolniejsze, dokładniejsze
            est_time = n_points / 100_000  # ~100k pkt/s

        est_min = int(est_time // 60)
        est_sec = int(est_time % 60)

        mode_label = "DEMO (szybkie)" if demo_mode else "Normalne (dokładne)"
        st.info(f"⏱️ **Estymowany czas ({mode_label}):** {est_min} min {est_sec} sek")

        # Przycisk START
        if st.button("🚀 ROZPOCZNIJ KLASYFIKACJĘ", type="primary", use_container_width=True):
            # Progress placeholders
            progress_bar = st.progress(0)
            status_text = st.empty()
            stats_placeholder = st.empty()

            # Uruchom pipeline
            try:
                # Pokazuj status inicjalizacji
                with st.spinner("⏳ Inicjalizacja pipeline..."):
                    pipeline = ClassificationPipeline(
                        input_path=input_path,
                        output_path=str(output_path),
                        n_threads=n_threads,
                        demo_mode=demo_mode
                    )

                def progress_callback(progress_dict):
                    """Callback do aktualizacji postępu"""
                    pct = progress_dict['progress_pct']
                    completed = progress_dict['completed_tiles']
                    total = progress_dict['total_tiles']
                    eta_sec = progress_dict['eta_seconds']

                    # Aktualizuj progress bar
                    progress_bar.progress(int(pct))

                    # Aktualizuj status
                    eta_min = int(eta_sec // 60)
                    eta_sec_rem = int(eta_sec % 60)
                    status_text.text(
                        f"⏳ Przetwarzanie: {completed}/{total} kafelków ({pct:.1f}%) | "
                        f"Pozostało: {eta_min}m {eta_sec_rem}s"
                    )

                # Uruchom z informacją o postępie
                status_text.info("📥 KROK 1/5: Wczytywanie chmury punktów...")
                start_time = time.time()
                stats = pipeline.run(progress_callback=progress_callback)
                elapsed = time.time() - start_time

                # Sukces!
                progress_bar.progress(100)
                status_text.empty()

                st.balloons()
                st.success(f"✅ **Klasyfikacja zakończona!** ({elapsed:.1f}s)")

                # Statystyki
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("⏱️ Czas przetwarzania", f"{elapsed:.1f} s")

                with col2:
                    st.metric("🚀 Prędkość", f"{stats['points_per_second']:,.0f} pkt/s")

                with col3:
                    st.metric("📦 Kafelków", stats['n_tiles'])

                # ========== RAPORT JAKOŚCI (30% OCENY!) ==========
                st.markdown("---")
                st.markdown("### 📊 RAPORT JAKOŚCI KLASYFIKACJI")

                classification_stats = stats['classification_stats']

                # Metryki główne
                col1, col2, col3 = st.columns(3)

                classified_count = sum(count for cls, count in classification_stats.items() if cls != 1)
                unclassified_count = classification_stats.get(1, 0)
                n_classes = len([cls for cls in classification_stats.keys() if cls != 1])

                with col1:
                    st.metric("✅ Sklasyfikowane", f"{classified_count:,}",
                             f"{classified_count/n_points*100:.1f}%")

                with col2:
                    st.metric("❓ Nieklasyfikowane", f"{unclassified_count:,}",
                             f"{unclassified_count/n_points*100:.1f}%")

                with col3:
                    st.metric("🎯 Wykryte klasy", n_classes)

                # Rozkład klasyfikacji z wykresem
                with st.expander("📊 Szczegółowy rozkład klasyfikacji", expanded=True):
                    import pandas as pd

                    # Przygotuj dane dla tabeli
                    report_data = []
                    for class_id, count in sorted(classification_stats.items()):
                        if class_id in all_classifiers:
                            classifier = all_classifiers[class_id]()
                            name = classifier.class_name
                        else:
                            name = f"Klasa {class_id}"

                        pct = count / n_points * 100
                        report_data.append({
                            'ID': class_id,
                            'Nazwa klasy': name,
                            'Liczba punktów': f"{count:,}",
                            'Procent': f"{pct:.2f}%",
                            'Procent_raw': pct  # Dla sortowania
                        })

                    # Stwórz DataFrame i wyświetl
                    df = pd.DataFrame(report_data)

                    # Sortuj po procentach malejąco (pomijając raw column w wyświetlaniu)
                    df_sorted = df.sort_values('Procent_raw', ascending=False)
                    st.dataframe(
                        df_sorted[['ID', 'Nazwa klasy', 'Liczba punktów', 'Procent']],
                        use_container_width=True,
                        hide_index=True
                    )

                    # Wykres słupkowy (top 10 klas)
                    st.markdown("#### 📈 Top 10 klas (wykres)")
                    top10 = df_sorted.head(10)

                    import plotly.graph_objects as go

                    fig = go.Figure(data=[
                        go.Bar(
                            x=top10['Nazwa klasy'],
                            y=top10['Procent_raw'],
                            text=top10['Procent'],
                            textposition='auto',
                        )
                    ])
                    fig.update_layout(
                        xaxis_title="Klasa",
                        yaxis_title="Procent punktów (%)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Pobierz plik
                st.markdown("---")
                st.markdown("### 💾 Pobierz wyniki")

                col1, col2, col3 = st.columns(3)

                # 1. Sklasyfikowany plik LAS
                with col1:
                    if output_path.exists():
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="📥 Sklasyfikowany LAS",
                                data=f,
                                file_name=output_name,
                                mime="application/octet-stream",
                                use_container_width=True
                            )

                # 2. Raport TXT
                with col2:
                    # Generuj raport tekstowy
                    from src.v2.core import LASWriter
                    class_names = {cid: all_classifiers[cid]().class_name for cid in all_classifiers}
                    class_names[1] = "Nieklasyfikowane"  # Dodaj Unclassified

                    # Pobierz klasyfikację z zapisanego pliku
                    import laspy
                    with laspy.open(output_path) as f:
                        las = f.read()
                        saved_classification = las.classification

                    txt_report = LASWriter.create_classification_report(
                        classification=saved_classification,
                        class_names=class_names
                    )

                    st.download_button(
                        label="📄 Raport TXT",
                        data=txt_report,
                        file_name=f"{Path(input_path).stem}_raport.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                # 3. Raport JSON
                with col3:
                    import json

                    json_report = {
                        "metadata": {
                            "plik_wejsciowy": Path(input_path).name,
                            "plik_wyjsciowy": output_name,
                            "czas_przetwarzania_s": round(elapsed, 2),
                            "predkosc_pkt_s": int(stats['points_per_second']),
                            "liczba_kafelkow": stats['n_tiles'],
                            "tryb_demo": demo_mode,
                            "liczba_watkow": n_threads
                        },
                        "statystyki": {
                            "calkowita_liczba_punktow": n_points,
                            "sklasyfikowane": classified_count,
                            "nieklasyfikowane": unclassified_count,
                            "wykryte_klasy": n_classes
                        },
                        "rozklad_klas": [
                            {
                                "id": int(cls),
                                "nazwa": class_names.get(cls, f"Klasa {cls}"),
                                "liczba": int(count),
                                "procent": round(count / n_points * 100, 2)
                            }
                            for cls, count in sorted(classification_stats.items(), key=lambda x: x[1], reverse=True)
                        ]
                    }

                    st.download_button(
                        label="📊 Raport JSON",
                        data=json.dumps(json_report, ensure_ascii=False, indent=2),
                        file_name=f"{Path(input_path).stem}_raport.json",
                        mime="application/json",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"❌ **Błąd podczas klasyfikacji:** {str(e)}")
                logger.exception("Classification error")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #0A1E42; font-size: 0.9rem;">
    <b>Chmura+ v2.0</b><br>
    Centralny Port Komunikacyjny | HackNation2025<br>
    Automatyczna klasyfikacja 45 klas infrastruktury
</div>
""", unsafe_allow_html=True)
