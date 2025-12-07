"""
ML Classifier UI - Interfejs do trenowania i uzycia modeli ML

HackNation 2025 - CPK Chmura+
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict
import json
from pathlib import Path
import time

from ...config import PATHS


def render_ml_classifier():
    """Glowny komponent ML"""

    st.markdown("""
    <div style='background: linear-gradient(135deg, #7b1fa2 0%, #512da8 100%);
                padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: white; margin: 0;'>Machine Learning Classifier</h2>
        <p style='color: #e1bee7; margin: 5px 0 0 0;'>
            Trenuj i urzywaj modeli ML do klasyfikacji chmur punktow
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Menu
    mode = st.radio(
        "Tryb:",
        ["Trening modelu", "Klasyfikacja ML", "PointNet", "Ensemble", "Auto-Tuning", "Post-processing", "Porownanie modeli"],
        horizontal=True
    )

    st.markdown("---")

    if mode == "Trening modelu":
        _render_training_mode()
    elif mode == "Klasyfikacja ML":
        _render_inference_mode()
    elif mode == "PointNet":
        _render_pointnet_mode()
    elif mode == "Ensemble":
        _render_ensemble_mode()
    elif mode == "Auto-Tuning":
        _render_auto_tuning_mode()
    elif mode == "Post-processing":
        _render_post_processing_mode()
    else:
        _render_model_comparison_mode()


def _render_training_mode():
    """Tryb trenowania modelu"""
    st.subheader("Trening modelu ML")

    st.info("""
    **Jak to dziala:**
    1. Wczytaj plik LAS/LAZ z juz sklasyfikowanymi punktami (etykiety)
    2. Model nauczy sie rozpoznawac klasy na podstawie cech geometrycznych
    3. Zapisany model mozna uzyc do klasyfikacji nowych chmur
    """)

    # Sprawdz czy mamy dane z klasyfikacja
    if 'hack_full_results' not in st.session_state:
        st.warning("Najpierw wykonaj klasyfikacje w zakladce 'Hackathon' aby wygenerowac dane treningowe")

        # Alternatywa - wczytaj plik z etykietami
        st.markdown("---")
        st.markdown("**Lub wczytaj plik z istniejaca klasyfikacja:**")

        uploaded = st.file_uploader("Plik LAS/LAZ z etykietami", type=['las', 'laz'])
        if uploaded:
            _train_from_file(uploaded)
        return

    results = st.session_state['hack_full_results']
    coords = results.get('coords')
    classification = results.get('classification')

    if coords is None or classification is None:
        st.error("Brak danych w pamieci (tryb batch)")
        return

    # Statystyki danych
    st.markdown("### Dane treningowe")
    unique_classes, counts = np.unique(classification, return_counts=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Punktow", f"{len(coords):,}")
    with col2:
        st.metric("Klas", len(unique_classes))
    with col3:
        st.metric("Min per klasa", f"{counts.min():,}")

    # Rozklad klas
    fig = go.Figure(data=[go.Bar(
        x=[f"Klasa {c}" for c in unique_classes],
        y=counts,
        marker_color='#7b1fa2'
    )])
    fig.update_layout(height=300, title="Rozklad klas w danych treningowych")
    st.plotly_chart(fig, use_container_width=True)

    # Konfiguracja
    st.markdown("### Konfiguracja treningu")

    col1, col2 = st.columns(2)
    with col1:
        n_estimators = st.slider("Liczba drzew (n_estimators)", 50, 300, 100, 25)
        max_depth = st.slider("Max glebokosc drzewa", 5, 30, 15)

    with col2:
        max_samples = st.slider("Max probek per klasa", 10000, 100000, 30000, 5000)
        k_neighbors = st.slider("Sasiedzi do cech", 10, 50, 30, 5)

    model_name = st.text_input("Nazwa modelu", value="cpk_classifier_v1")
    model_path = PATHS.OUTPUT_DIR / "models" / f"{model_name}.pkl"

    # Trening
    if st.button("TRENUJ MODEL", type="primary", use_container_width=True):
        _run_training(
            coords, classification,
            results.get('colors'), results.get('intensity'),
            n_estimators, max_depth, max_samples, k_neighbors,
            str(model_path)
        )


def _run_training(coords, classification, colors, intensity,
                  n_estimators, max_depth, max_samples, k_neighbors, model_path):
    """Uruchamia trening"""
    from ...v2.ml import TrainingPipeline, TrainingConfig

    progress = st.progress(0)
    status = st.empty()

    config = TrainingConfig(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_samples_per_class=max_samples,
        k_neighbors=k_neighbors,
        model_path=model_path
    )

    def progress_cb(step, pct, msg):
        progress.progress(pct / 100)
        status.info(f"{step}: {msg}")

    try:
        pipeline = TrainingPipeline(config)
        result = pipeline.train(
            coords, classification,
            colors, intensity,
            progress_callback=progress_cb
        )

        progress.progress(100)
        status.success("Trening zakonczony!")

        # Wyniki
        st.markdown("### Wyniki treningu")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train Accuracy", f"{result.train_accuracy:.1%}")
        with col2:
            st.metric("Val Accuracy", f"{result.val_accuracy:.1%}")
        with col3:
            st.metric("Test Accuracy", f"{result.test_accuracy:.1%}")
        with col4:
            st.metric("Czas", f"{result.training_time:.1f}s")

        # Feature importance
        st.markdown("### Waznosc cech")
        importance = result.metrics.feature_importance
        if importance:
            sorted_imp = sorted(importance.items(), key=lambda x: -x[1])

            fig = go.Figure(data=[go.Bar(
                x=[name for name, _ in sorted_imp],
                y=[imp for _, imp in sorted_imp],
                marker_color='#7b1fa2'
            )])
            fig.update_layout(height=300, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        # Classification report
        with st.expander("Raport klasyfikacji"):
            st.code(result.metrics.classification_report)

        st.success(f"Model zapisany: {model_path}")

        # Zapisz do session state
        st.session_state['ml_model_path'] = model_path

    except Exception as e:
        status.error(f"Blad treningu: {e}")
        import traceback
        st.code(traceback.format_exc())


def _render_inference_mode():
    """Tryb klasyfikacji ML"""
    st.subheader("Klasyfikacja ML")

    # Sprawdz dostepne modele
    models_dir = PATHS.OUTPUT_DIR / "models"
    available_models = list(models_dir.glob("*.pkl")) if models_dir.exists() else []

    if not available_models:
        st.warning("Brak wytrenowanych modeli. Najpierw wytrenuj model.")
        return

    # Wybor modelu
    model_options = [m.stem for m in available_models]
    selected_model = st.selectbox("Wybierz model:", model_options)
    model_path = models_dir / f"{selected_model}.pkl"

    # Sprawdz dane
    if 'input_file' not in st.session_state:
        st.warning("Wczytaj plik w zakladce 'Wczytaj plik'")
        return

    input_path = st.session_state['input_file']
    file_info = st.session_state['file_info']

    st.markdown(f"**Plik:** {Path(input_path).name} ({file_info['n_points']:,} punktow)")

    # Opcje
    col1, col2 = st.columns(2)
    with col1:
        sample_size = st.slider(
            "Probka (0 = wszystko)",
            0, min(file_info['n_points'], 1000000), 100000, 10000
        )
    with col2:
        k_neighbors = st.slider("Sasiedzi do cech", 10, 50, 30, 5)

    if st.button("KLASYFIKUJ ML", type="primary", use_container_width=True):
        _run_ml_inference(str(model_path), input_path, sample_size, k_neighbors)


def _run_ml_inference(model_path, input_path, sample_size, k_neighbors):
    """Uruchamia inference ML"""
    from ...v2 import LASLoader
    from ...v2.ml import MLInference

    progress = st.progress(0)
    status = st.empty()
    start_time = time.time()

    try:
        # Wczytaj dane
        status.info("Wczytywanie danych...")
        progress.progress(10)

        loader = LASLoader(input_path)
        data = loader.load(sample_size=sample_size if sample_size > 0 else None)

        # Inference
        status.info("Ekstrakcja cech i klasyfikacja ML...")
        progress.progress(30)

        inference = MLInference(model_path, k_neighbors=k_neighbors)

        def progress_cb(step, pct, msg):
            overall = 30 + int(pct * 0.6)
            progress.progress(overall)
            status.info(f"{step}: {msg}")

        classification, confidence = inference.predict(
            data['coords'],
            data['colors'],
            data['intensity'],
            return_confidence=True,
            progress_callback=progress_cb
        )

        progress.progress(95)
        elapsed = time.time() - start_time

        # Wyniki
        progress.progress(100)
        status.success(f"Klasyfikacja ML zakonczona! ({elapsed:.1f}s)")

        # Metryki
        st.markdown("### Wyniki")

        unique_classes, counts = np.unique(classification, return_counts=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Punktow", f"{len(classification):,}")
        with col2:
            st.metric("Klas", len(unique_classes))
        with col3:
            st.metric("Sredni confidence", f"{confidence.mean():.1%}")
        with col4:
            st.metric("Predkosc", f"{len(classification)/elapsed:,.0f} pkt/s")

        # Rozklad klas
        CLASS_NAMES = {
            2: "Grunt", 3: "Rosl. niska", 4: "Rosl. srednia", 5: "Rosl. wysoka",
            6: "Budynek", 7: "Szum", 18: "Tory", 19: "Linie", 20: "Slupy"
        }

        fig = go.Figure(data=[go.Pie(
            labels=[CLASS_NAMES.get(c, f"Klasa {c}") for c in unique_classes],
            values=counts,
            hole=0.4
        )])
        fig.update_layout(height=400, title="Rozklad klas (ML)")
        st.plotly_chart(fig, use_container_width=True)

        # Confidence histogram
        fig2 = go.Figure(data=[go.Histogram(
            x=confidence,
            nbinsx=50,
            marker_color='#7b1fa2'
        )])
        fig2.update_layout(
            height=300,
            title="Rozklad pewnosci (confidence)",
            xaxis_title="Confidence",
            yaxis_title="Liczba punktow"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Low confidence warning
        low_conf_pct = (confidence < 0.5).sum() / len(confidence) * 100
        if low_conf_pct > 20:
            st.warning(f"{low_conf_pct:.1f}% punktow ma niska pewnosc (<50%). "
                      "Rozważ dotrenowanie modelu na wiekszej ilosci danych.")

        # Zapisz wyniki
        st.session_state['ml_classification'] = classification
        st.session_state['ml_confidence'] = confidence
        st.session_state['ml_coords'] = data['coords']

    except Exception as e:
        status.error(f"Blad: {e}")
        import traceback
        st.code(traceback.format_exc())


def _render_model_info():
    """Informacje o modelu"""
    st.subheader("Informacje o modelach")

    models_dir = PATHS.OUTPUT_DIR / "models"
    available_models = list(models_dir.glob("*.pkl")) if models_dir.exists() else []

    if not available_models:
        st.info("Brak zapisanych modeli")
        return

    for model_path in available_models:
        with st.expander(f"Model: {model_path.stem}"):
            try:
                from ...v2.ml import MLInference
                inference = MLInference(str(model_path))
                info = inference.get_model_info()

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Liczba cech:** {info['n_features']}")
                    st.markdown(f"**Liczba klas:** {info['n_classes']}")
                with col2:
                    st.markdown(f"**Klasy:** {info['classes']}")

                st.markdown("**Cechy:**")
                st.code(", ".join(info['feature_names']))

                # Rozmiar pliku
                size_mb = model_path.stat().st_size / (1024 * 1024)
                st.caption(f"Rozmiar pliku: {size_mb:.2f} MB")

            except Exception as e:
                st.error(f"Blad wczytywania: {e}")


def _train_from_file(uploaded):
    """Trenuj z uploadowanego pliku"""
    st.info("Funkcja w przygotowaniu - uzyj danych z klasyfikacji")


def _render_pointnet_mode():
    """Tryb PointNet - Deep Learning"""
    from ...v2.ml import is_torch_available, get_device_info

    st.subheader("PointNet - Deep Learning")

    # Sprawdz PyTorch
    if not is_torch_available():
        st.error("""
        **PyTorch nie jest zainstalowany!**

        Aby uzyc PointNet, zainstaluj PyTorch:
        ```bash
        pip install torch
        ```

        Lub z obsluga CUDA:
        ```bash
        pip install torch --index-url https://download.pytorch.org/whl/cu118
        ```
        """)
        return

    # Info o urzadzeniu
    device_info = get_device_info()
    if device_info['cuda_available']:
        st.success(f"GPU dostepne: {device_info['device_name']}")
    else:
        st.info("Trening na CPU (GPU niedostepne)")

    st.markdown("""
    **PointNet** to zaawansowana architektura deep learning zaprojektowana specjalnie
    do pracy z chmurami punktow. W przeciwienstwie do Random Forest, PointNet:

    - Uczy sie reprezentacji bezposrednio z surowych danych XYZ
    - Jest niezmienczy na permutacje punktow
    - Wykrywa lokalne i globalne wzorce geometryczne
    - Daje lepsze wyniki dla zlozonych scen 3D
    """)

    # Tabs
    pn_mode = st.radio("Operacja:", ["Trening PointNet", "Klasyfikacja PointNet"], horizontal=True)

    st.markdown("---")

    if pn_mode == "Trening PointNet":
        _render_pointnet_training()
    else:
        _render_pointnet_inference()


def _render_pointnet_training():
    """Trening modelu PointNet"""
    from ...v2.ml import PointNetConfig, PointNetTrainer

    # Sprawdz dane
    if 'hack_full_results' not in st.session_state:
        st.warning("Najpierw wykonaj klasyfikacje w zakladce 'Hackathon' aby wygenerowac dane treningowe")
        return

    results = st.session_state['hack_full_results']
    coords = results.get('coords')
    classification = results.get('classification')
    colors = results.get('colors')
    intensity = results.get('intensity')

    if coords is None or classification is None:
        st.error("Brak danych w pamieci")
        return

    # Statystyki
    unique_classes, counts = np.unique(classification, return_counts=True)
    n_classes = len(unique_classes)

    st.markdown("### Dane treningowe")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Punktow", f"{len(coords):,}")
    with col2:
        st.metric("Klas", n_classes)
    with col3:
        st.metric("Min per klasa", f"{counts.min():,}")

    # Konfiguracja
    st.markdown("### Konfiguracja PointNet")

    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider("Liczba epok", 10, 200, 50, 10)
        batch_size = st.slider("Batch size", 4, 32, 8, 4)
        learning_rate = st.select_slider(
            "Learning rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001
        )

    with col2:
        hidden_dim = st.slider("Wymiar ukryty", 128, 512, 256, 64)
        dropout = st.slider("Dropout", 0.0, 0.5, 0.3, 0.1)
        use_tnet = st.checkbox("Uzyj T-Net (Spatial Transformer)", value=True)

    model_name = st.text_input("Nazwa modelu", value="pointnet_cpk_v1")
    model_path = PATHS.OUTPUT_DIR / "models" / f"{model_name}.pth"

    # Oblicz input channels
    input_channels = 3  # xyz
    if colors is not None:
        input_channels += 3  # rgb
    if intensity is not None:
        input_channels += 1

    st.caption(f"Input channels: {input_channels} (xyz" +
               ("+rgb" if colors is not None else "") +
               ("+intensity" if intensity is not None else "") + ")")

    # Trening
    if st.button("TRENUJ POINTNET", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        try:
            config = PointNetConfig(
                n_classes=n_classes,
                input_channels=input_channels,
                hidden_dims=[64, 128, hidden_dim],
                dropout=dropout,
                use_tnet=use_tnet
            )

            trainer = PointNetTrainer(config, learning_rate=learning_rate)

            def progress_cb(step, pct, msg):
                progress.progress(pct / 100)
                status.info(f"{step}: {msg}")

            result = trainer.train(
                coords, classification,
                colors, intensity,
                epochs=epochs,
                batch_size=batch_size,
                progress_callback=progress_cb
            )

            progress.progress(100)
            status.success("Trening PointNet zakonczony!")

            # Wyniki
            st.markdown("### Wyniki")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Val Accuracy", f"{result['best_val_acc']:.1%}")
            with col2:
                st.metric("Czas treningu", f"{result['training_time']:.1f}s")
            with col3:
                st.metric("Liczba klas", result['n_classes'])

            # Historia treningu
            history = result['history']
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Loss", "Accuracy"])

            fig.add_trace(
                go.Scatter(y=history['train_loss'], name="Train Loss", line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(y=history['val_loss'], name="Val Loss", line=dict(color='orange')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(y=history['train_acc'], name="Train Acc", line=dict(color='green')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(y=history['val_acc'], name="Val Acc", line=dict(color='red')),
                row=1, col=2
            )

            fig.update_layout(height=400, title="Historia treningu")
            st.plotly_chart(fig, use_container_width=True)

            # Zapisz model
            trainer.save(str(model_path))
            st.success(f"Model zapisany: {model_path}")

            st.session_state['pointnet_trainer'] = trainer

        except Exception as e:
            status.error(f"Blad: {e}")
            import traceback
            st.code(traceback.format_exc())


def _render_pointnet_inference():
    """Klasyfikacja PointNet"""
    from ...v2.ml import PointNetTrainer

    st.markdown("### Klasyfikacja PointNet")

    # Dostepne modele
    models_dir = PATHS.OUTPUT_DIR / "models"
    pth_models = list(models_dir.glob("*.pth")) if models_dir.exists() else []

    if not pth_models and 'pointnet_trainer' not in st.session_state:
        st.warning("Brak wytrenowanych modeli PointNet. Najpierw wytrenuj model.")
        return

    # Wybor modelu
    if pth_models:
        model_options = ["-- Z pamieci --"] + [m.stem for m in pth_models]
        selected = st.selectbox("Wybierz model:", model_options)

        if selected == "-- Z pamieci --":
            if 'pointnet_trainer' not in st.session_state:
                st.warning("Brak modelu w pamieci")
                return
            trainer = st.session_state['pointnet_trainer']
        else:
            model_path = models_dir / f"{selected}.pth"
            trainer = PointNetTrainer.load(str(model_path))
    else:
        trainer = st.session_state['pointnet_trainer']

    # Dane do klasyfikacji
    if 'input_file' not in st.session_state:
        st.warning("Wczytaj plik w zakladce 'Wczytaj plik'")
        return

    input_path = st.session_state['input_file']
    file_info = st.session_state['file_info']

    st.markdown(f"**Plik:** {Path(input_path).name} ({file_info['n_points']:,} punktow)")

    sample_size = st.slider(
        "Probka (0 = wszystko)",
        0, min(file_info['n_points'], 500000), 100000, 10000
    )

    if st.button("KLASYFIKUJ POINTNET", type="primary", use_container_width=True):
        from ...v2 import LASLoader

        progress = st.progress(0)
        status = st.empty()
        start_time = time.time()

        try:
            # Wczytaj dane
            status.info("Wczytywanie danych...")
            progress.progress(10)

            loader = LASLoader(input_path)
            data = loader.load(sample_size=sample_size if sample_size > 0 else None)

            # Inference
            status.info("Klasyfikacja PointNet...")
            progress.progress(20)

            def progress_cb(step, pct, msg):
                overall = 20 + int(pct * 0.7)
                progress.progress(overall)
                status.info(f"{step}: {msg}")

            classification, confidence = trainer.predict(
                data['coords'],
                data['colors'],
                data['intensity'],
                return_confidence=True,
                progress_callback=progress_cb
            )

            elapsed = time.time() - start_time
            progress.progress(100)
            status.success(f"Klasyfikacja zakonczona! ({elapsed:.1f}s)")

            # Wyniki
            st.markdown("### Wyniki")
            unique_classes, counts = np.unique(classification, return_counts=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Punktow", f"{len(classification):,}")
            with col2:
                st.metric("Klas", len(unique_classes))
            with col3:
                st.metric("Sredni confidence", f"{confidence.mean():.1%}")
            with col4:
                st.metric("Predkosc", f"{len(classification)/elapsed:,.0f} pkt/s")

            # Rozklad klas
            CLASS_NAMES = {
                2: "Grunt", 3: "Rosl. niska", 4: "Rosl. srednia", 5: "Rosl. wysoka",
                6: "Budynek", 7: "Szum", 18: "Tory", 19: "Linie", 20: "Slupy"
            }

            fig = go.Figure(data=[go.Pie(
                labels=[CLASS_NAMES.get(c, f"Klasa {c}") for c in unique_classes],
                values=counts,
                hole=0.4
            )])
            fig.update_layout(height=400, title="Rozklad klas (PointNet)")
            st.plotly_chart(fig, use_container_width=True)

            # Confidence
            fig2 = go.Figure(data=[go.Histogram(
                x=confidence,
                nbinsx=50,
                marker_color='#512da8'
            )])
            fig2.update_layout(
                height=300,
                title="Rozklad pewnosci (confidence)",
                xaxis_title="Confidence",
                yaxis_title="Punkty"
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Zapisz wyniki
            st.session_state['pointnet_classification'] = classification
            st.session_state['pointnet_confidence'] = confidence
            st.session_state['pointnet_coords'] = data['coords']

        except Exception as e:
            status.error(f"Blad: {e}")
            import traceback
            st.code(traceback.format_exc())


def _render_ensemble_mode():
    """Tryb Ensemble - laczenie wielu modeli"""
    from ...v2.ml import EnsembleClassifier, EnsembleConfig, is_torch_available

    st.subheader("Ensemble Classifier")

    st.markdown("""
    **Ensemble** laczy predykcje z wielu modeli (Random Forest + PointNet) dla lepszej dokladnosci.

    **Strategie votingu:**
    - **Hard** - wybierz klase z najwieksza liczba glosow
    - **Soft** - srednia prawdopodobienstw, wybierz najwyzsza
    - **Weighted** - wazony voting z preferencja dla lepszego modelu
    """)

    pn_available = is_torch_available()
    if not pn_available:
        st.warning("PyTorch niedostepny - Ensemble bedzie uzywac tylko Random Forest")

    # Tabs
    ens_mode = st.radio("Operacja:", ["Trening Ensemble", "Klasyfikacja Ensemble"], horizontal=True)

    st.markdown("---")

    if ens_mode == "Trening Ensemble":
        _render_ensemble_training(pn_available)
    else:
        _render_ensemble_inference()


def _render_ensemble_training(pn_available: bool):
    """Trening ensemble"""
    from ...v2.ml import EnsembleClassifier, EnsembleConfig

    # Sprawdz dane
    if 'hack_full_results' not in st.session_state:
        st.warning("Najpierw wykonaj klasyfikacje w zakladce 'Hackathon' aby wygenerowac dane treningowe")
        return

    results = st.session_state['hack_full_results']
    coords = results.get('coords')
    classification = results.get('classification')
    colors = results.get('colors')
    intensity = results.get('intensity')

    if coords is None or classification is None:
        st.error("Brak danych w pamieci")
        return

    # Statystyki
    unique_classes, counts = np.unique(classification, return_counts=True)

    st.markdown("### Dane treningowe")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Punktow", f"{len(coords):,}")
    with col2:
        st.metric("Klas", len(unique_classes))
    with col3:
        st.metric("Min per klasa", f"{counts.min():,}")

    # Konfiguracja
    st.markdown("### Konfiguracja Ensemble")

    col1, col2 = st.columns(2)
    with col1:
        voting = st.selectbox("Strategia votingu:", ["soft", "hard", "weighted"])
        use_rf = st.checkbox("Uzyj Random Forest", value=True)
        use_pn = st.checkbox("Uzyj PointNet", value=pn_available, disabled=not pn_available)

    with col2:
        rf_estimators = st.slider("RF: Liczba drzew", 50, 200, 100, 25)
        rf_depth = st.slider("RF: Max glebokosc", 10, 30, 20, 5)
        if use_pn:
            pn_epochs = st.slider("PointNet: Epoki", 10, 100, 30, 10)
        else:
            pn_epochs = 30

    model_name = st.text_input("Nazwa modelu", value="ensemble_cpk_v1")
    model_path = PATHS.OUTPUT_DIR / "models" / f"{model_name}.pkl"

    # Trening
    if st.button("TRENUJ ENSEMBLE", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        try:
            config = EnsembleConfig(
                voting=voting,
                use_rf=use_rf,
                use_pointnet=use_pn,
                rf_n_estimators=rf_estimators,
                rf_max_depth=rf_depth,
                pointnet_epochs=pn_epochs
            )

            ensemble = EnsembleClassifier(config)

            def progress_cb(step, pct, msg):
                progress.progress(pct / 100)
                status.info(f"{step}: {msg}")

            result = ensemble.train(
                coords, classification,
                colors, intensity,
                progress_callback=progress_cb
            )

            progress.progress(100)
            status.success("Ensemble wytrenowany!")

            # Wyniki
            st.markdown("### Wyniki")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Czas treningu", f"{result['total_time']:.1f}s")
            with col2:
                st.metric("Liczba modeli", result['n_models'])
            with col3:
                st.metric("Modele", ", ".join(result['models']))

            if 'rf_val_accuracy' in result:
                st.metric("RF Val Accuracy", f"{result['rf_val_accuracy']:.1%}")
            if 'pn_val_accuracy' in result:
                st.metric("PointNet Val Accuracy", f"{result['pn_val_accuracy']:.1%}")

            # Zapisz model
            ensemble.save(str(model_path))
            st.success(f"Ensemble zapisany: {model_path}")

            st.session_state['ensemble_classifier'] = ensemble

        except Exception as e:
            status.error(f"Blad: {e}")
            import traceback
            st.code(traceback.format_exc())


def _render_ensemble_inference():
    """Klasyfikacja ensemble"""
    from ...v2.ml import EnsembleClassifier

    st.markdown("### Klasyfikacja Ensemble")

    # Dostepne modele
    models_dir = PATHS.OUTPUT_DIR / "models"
    pkl_models = [m for m in models_dir.glob("*.pkl") if 'ensemble' in m.stem.lower()] if models_dir.exists() else []

    if not pkl_models and 'ensemble_classifier' not in st.session_state:
        st.warning("Brak wytrenowanych modeli Ensemble. Najpierw wytrenuj model.")
        return

    # Wybor modelu
    if pkl_models:
        model_options = ["-- Z pamieci --"] + [m.stem for m in pkl_models]
        selected = st.selectbox("Wybierz model:", model_options)

        if selected == "-- Z pamieci --":
            if 'ensemble_classifier' not in st.session_state:
                st.warning("Brak modelu w pamieci")
                return
            ensemble = st.session_state['ensemble_classifier']
        else:
            model_path = models_dir / f"{selected}.pkl"
            ensemble = EnsembleClassifier.load(str(model_path))
    else:
        ensemble = st.session_state['ensemble_classifier']

    # Info o ensemble
    info = ensemble.get_info()
    st.info(f"Ensemble: {info['n_models']} modeli ({', '.join(info['models'])}), voting: {info['voting']}")

    # Dane do klasyfikacji
    if 'input_file' not in st.session_state:
        st.warning("Wczytaj plik w zakladce 'Wczytaj plik'")
        return

    input_path = st.session_state['input_file']
    file_info = st.session_state['file_info']

    st.markdown(f"**Plik:** {Path(input_path).name} ({file_info['n_points']:,} punktow)")

    sample_size = st.slider(
        "Probka (0 = wszystko)",
        0, min(file_info['n_points'], 500000), 100000, 10000
    )

    if st.button("KLASYFIKUJ ENSEMBLE", type="primary", use_container_width=True):
        from ...v2 import LASLoader

        progress = st.progress(0)
        status = st.empty()
        start_time = time.time()

        try:
            # Wczytaj dane
            status.info("Wczytywanie danych...")
            progress.progress(10)

            loader = LASLoader(input_path)
            data = loader.load(sample_size=sample_size if sample_size > 0 else None)

            # Inference
            status.info("Klasyfikacja Ensemble...")
            progress.progress(20)

            def progress_cb(step, pct, msg):
                overall = 20 + int(pct * 0.7)
                progress.progress(overall)
                status.info(f"{step}: {msg}")

            result = ensemble.predict(
                data['coords'],
                data['colors'],
                data['intensity'],
                return_details=True,
                progress_callback=progress_cb
            )

            elapsed = time.time() - start_time
            progress.progress(100)
            status.success(f"Klasyfikacja zakonczona! ({elapsed:.1f}s)")

            # Wyniki
            st.markdown("### Wyniki")
            unique_classes, counts = np.unique(result.classification, return_counts=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Punktow", f"{len(result.classification):,}")
            with col2:
                st.metric("Klas", len(unique_classes))
            with col3:
                st.metric("Sredni confidence", f"{result.confidence.mean():.1%}")
            with col4:
                st.metric("Agreement", f"{result.agreement_ratio:.1%}")

            # Rozklad klas
            CLASS_NAMES = {
                2: "Grunt", 3: "Rosl. niska", 4: "Rosl. srednia", 5: "Rosl. wysoka",
                6: "Budynek", 7: "Szum", 18: "Tory", 19: "Linie", 20: "Slupy"
            }

            fig = go.Figure(data=[go.Pie(
                labels=[CLASS_NAMES.get(c, f"Klasa {c}") for c in unique_classes],
                values=counts,
                hole=0.4
            )])
            fig.update_layout(height=400, title="Rozklad klas (Ensemble)")
            st.plotly_chart(fig, use_container_width=True)

            # Porownanie modeli
            if len(result.model_predictions) > 1:
                st.markdown("### Porownanie modeli")

                for model_name, preds in result.model_predictions.items():
                    unique_m, counts_m = np.unique(preds, return_counts=True)
                    conf_m = result.model_confidences.get(model_name, np.array([]))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**{model_name}**")
                    with col2:
                        if len(conf_m) > 0:
                            st.metric(f"Sredni conf", f"{conf_m.mean():.1%}")

            # Zapisz wyniki
            st.session_state['ensemble_classification'] = result.classification
            st.session_state['ensemble_confidence'] = result.confidence
            st.session_state['ensemble_coords'] = data['coords']

        except Exception as e:
            status.error(f"Blad: {e}")
            import traceback
            st.code(traceback.format_exc())


def _render_auto_tuning_mode():
    """Auto-tuning hiperparametrow"""
    from ...v2.ml import AutoMLPipeline, is_optuna_available

    st.subheader("Auto-Tuning Hiperparametrów")

    st.markdown("""
    **Automatyczna optymalizacja** znajduje najlepsze hiperparametry dla modelu.

    **Metody:**
    - **Random Search** - losowe probkowanie przestrzeni parametrow
    - **Grid Search** - przeszukanie wszystkich kombinacji
    - **Bayesian** - inteligentne przeszukiwanie (wymaga optuna)
    """)

    optuna_ok = is_optuna_available()
    if not optuna_ok:
        st.info("Optuna niedostepna. Zainstaluj: `pip install optuna` dla Bayesian optimization")

    # Sprawdz dane
    if 'hack_full_results' not in st.session_state:
        st.warning("Najpierw wykonaj klasyfikacje w zakladce 'Hackathon'")
        return

    results = st.session_state['hack_full_results']
    coords = results.get('coords')
    classification = results.get('classification')

    if coords is None:
        st.error("Brak danych")
        return

    st.markdown(f"**Dane:** {len(coords):,} punktów")

    # Konfiguracja
    col1, col2 = st.columns(2)
    with col1:
        methods = ["random", "grid"]
        if optuna_ok:
            methods.append("bayesian")
        search_method = st.selectbox("Metoda:", methods)
        n_trials = st.slider("Liczba prób", 10, 100, 30, 10)

    with col2:
        cv_folds = st.slider("K-Fold CV", 2, 5, 3)
        max_samples = st.slider("Max próbek", 10000, 100000, 30000, 5000)

    if st.button("URUCHOM AUTO-TUNING", type="primary", use_container_width=True):
        from ...v2.ml import FeatureExtractor

        progress = st.progress(0)
        status = st.empty()

        try:
            # Subsample
            if len(coords) > max_samples:
                indices = np.random.choice(len(coords), max_samples, replace=False)
                coords_sub = coords[indices]
                labels_sub = classification[indices]
                colors_sub = results.get('colors')[indices] if results.get('colors') is not None else None
                intensity_sub = results.get('intensity')[indices] if results.get('intensity') is not None else None
            else:
                coords_sub = coords
                labels_sub = classification
                colors_sub = results.get('colors')
                intensity_sub = results.get('intensity')

            status.info("Ekstrakcja cech...")
            progress.progress(10)

            extractor = FeatureExtractor(coords_sub, colors_sub, intensity_sub)
            features = extractor.extract_all()
            X = features.to_array()

            # Usun NaN
            nan_mask = np.isnan(X)
            if nan_mask.any():
                col_means = np.nanmean(X, axis=0)
                for i in range(X.shape[1]):
                    X[nan_mask[:, i], i] = col_means[i]

            status.info("Auto-tuning...")
            progress.progress(20)

            pipeline = AutoMLPipeline(
                search_method=search_method,
                n_trials=n_trials,
                cv_folds=cv_folds
            )

            def progress_cb(step, pct, msg):
                overall = 20 + int(pct * 0.7)
                progress.progress(overall)
                status.info(f"{step}: {msg}")

            result = pipeline.auto_tune_rf(
                X, labels_sub,
                features.feature_names(),
                progress_callback=progress_cb
            )

            progress.progress(100)
            status.success("Auto-tuning zakończony!")

            # Wyniki
            st.markdown("### Najlepsze parametry")
            st.json(result.best_params)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best CV Score", f"{result.best_score:.1%}")
            with col2:
                st.metric("Czas", f"{result.search_time:.1f}s")
            with col3:
                st.metric("Prób", result.n_trials)

            # Historia
            if len(result.all_results) > 0:
                scores = [r['score'] for r in result.all_results if r['score'] is not None]
                fig = go.Figure(data=[go.Scatter(
                    y=scores,
                    mode='lines+markers',
                    marker_color='#7b1fa2'
                )])
                fig.update_layout(
                    height=300,
                    title="Historia prób",
                    xaxis_title="Próba",
                    yaxis_title="CV Score"
                )
                st.plotly_chart(fig, use_container_width=True)

            st.session_state['best_rf_params'] = result.best_params

        except Exception as e:
            status.error(f"Błąd: {e}")
            import traceback
            st.code(traceback.format_exc())


def _render_post_processing_mode():
    """Post-processing klasyfikacji"""
    from ...v2.ml import SpatialSmoother, OutlierRemover, PostProcessingPipeline

    st.subheader("Post-processing")

    st.markdown("""
    **Post-processing** poprawia jakość klasyfikacji przez:
    - **Spatial smoothing** - wygładzanie przez głosowanie sąsiadów
    - **Outlier removal** - usuwanie izolowanych punktów
    - **Morfologiczne operacje** - opening/closing dla regionów
    """)

    # Sprawdz czy mamy klasyfikacje
    classification_keys = ['ml_classification', 'pointnet_classification', 'ensemble_classification']
    available = [k for k in classification_keys if k in st.session_state]

    if not available:
        st.warning("Najpierw wykonaj klasyfikację ML")
        return

    # Wybor zrodla
    source = st.selectbox("Źródło klasyfikacji:", available)

    classification = st.session_state[source]
    coords_key = source.replace('classification', 'coords')
    confidence_key = source.replace('classification', 'confidence')

    coords = st.session_state.get(coords_key)
    confidence = st.session_state.get(confidence_key)

    if coords is None:
        st.error("Brak współrzędnych")
        return

    st.markdown(f"**Punktów:** {len(classification):,}")

    # Operacje
    st.markdown("### Operacje")

    col1, col2 = st.columns(2)

    with col1:
        do_smoothing = st.checkbox("Spatial Smoothing", value=True)
        if do_smoothing:
            smooth_k = st.slider("K sąsiadów", 5, 30, 15, 5)
            smooth_agreement = st.slider("Min zgodność", 0.5, 0.9, 0.6, 0.1)

    with col2:
        do_outlier = st.checkbox("Usuwanie outlierów", value=False)
        if do_outlier:
            outlier_class = st.number_input("Klasa do czyszczenia", min_value=0, max_value=255, value=6)
            min_cluster = st.slider("Min rozmiar klastra", 5, 50, 10, 5)

    if st.button("WYKONAJ POST-PROCESSING", type="primary", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()

        try:
            pipeline = PostProcessingPipeline()

            if do_smoothing:
                smoother = SpatialSmoother(k_neighbors=smooth_k, min_agreement=smooth_agreement)
                pipeline.steps.append(('smoothing', lambda c, cls, conf: smoother.smooth(c, cls, conf)))

            if do_outlier:
                remover = OutlierRemover(min_cluster_size=min_cluster)
                pipeline.steps.append((
                    'outlier_removal',
                    lambda c, cls, conf, tc=outlier_class: remover.remove_isolated_points(c, cls, tc)
                ))

            def progress_cb(step, pct, msg):
                progress.progress(pct / 100)
                status.info(f"{step}: {msg}")

            result = pipeline.process(coords, classification, confidence, progress_callback=progress_cb)

            progress.progress(100)
            status.success("Post-processing zakończony!")

            # Wyniki
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Zmieniono punktów", f"{result.changes_count:,}")
            with col2:
                st.metric("Procent zmian", f"{result.changes_ratio:.1%}")
            with col3:
                st.metric("Czas", f"{result.processing_time:.2f}s")

            # Porownanie
            st.markdown("### Porównanie przed/po")

            unique_before, counts_before = np.unique(classification, return_counts=True)
            unique_after, counts_after = np.unique(result.classification, return_counts=True)

            CLASS_NAMES = {
                2: "Grunt", 3: "Rosl. niska", 4: "Rosl. srednia", 5: "Rosl. wysoka",
                6: "Budynek", 7: "Szum", 18: "Tory", 19: "Linie", 20: "Slupy"
            }

            fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'pie'}, {'type': 'pie'}]],
                               subplot_titles=["Przed", "Po"])

            fig.add_trace(
                go.Pie(labels=[CLASS_NAMES.get(c, f"Klasa {c}") for c in unique_before],
                      values=counts_before, hole=0.4),
                row=1, col=1
            )
            fig.add_trace(
                go.Pie(labels=[CLASS_NAMES.get(c, f"Klasa {c}") for c in unique_after],
                      values=counts_after, hole=0.4),
                row=1, col=2
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Zapisz wyniki
            st.session_state['postprocessed_classification'] = result.classification
            st.session_state['postprocessed_confidence'] = result.confidence

        except Exception as e:
            status.error(f"Błąd: {e}")
            import traceback
            st.code(traceback.format_exc())


def _render_model_comparison_mode():
    """Porównanie modeli"""
    from ...v2.ml import ModelComparator, ErrorAnalyzer

    st.subheader("Porównanie Modeli")

    st.markdown("""
    **Porównanie modeli** pozwala ocenić różne podejścia ML na tych samych danych.

    Porównywane metryki:
    - Accuracy, Precision, Recall, F1-score
    - Per-class performance
    - Confusion matrix
    - Analiza błędów
    """)

    # Sprawdz dostepne klasyfikacje
    classification_sources = {
        'ml_classification': 'Random Forest',
        'pointnet_classification': 'PointNet',
        'ensemble_classification': 'Ensemble',
        'postprocessed_classification': 'Post-processed'
    }

    available = {k: v for k, v in classification_sources.items() if k in st.session_state}

    if len(available) < 1:
        st.warning("Wykonaj klasyfikację aby móc porównywać modele")
        return

    # Ground truth - jesli dostepne
    if 'hack_full_results' in st.session_state:
        gt = st.session_state['hack_full_results'].get('original_classification')
        if gt is not None:
            st.success("Ground truth dostępne z oryginalnych etykiet")
        else:
            gt = None
            st.info("Brak ground truth - porównanie będzie relatywne")
    else:
        gt = None

    # Wybor modeli do porownania
    selected = st.multiselect(
        "Wybierz modele do porównania:",
        list(available.values()),
        default=list(available.values())[:2]
    )

    if len(selected) < 1:
        st.warning("Wybierz przynajmniej 1 model")
        return

    # Mapuj nazwy z powrotem na klucze
    selected_keys = [k for k, v in available.items() if v in selected]

    if st.button("PORÓWNAJ MODELE", type="primary", use_container_width=True):
        try:
            CLASS_NAMES = {
                2: "Grunt", 3: "Rosl. niska", 4: "Rosl. srednia", 5: "Rosl. wysoka",
                6: "Budynek", 7: "Szum", 18: "Tory", 19: "Linie", 20: "Slupy"
            }

            comparator = ModelComparator(CLASS_NAMES)

            # Jesli mamy GT
            if gt is not None:
                for key in selected_keys:
                    preds = st.session_state[key]
                    name = available[key]
                    # Upewnij sie ze rozmiary sie zgadzaja
                    min_len = min(len(gt), len(preds))
                    comparator.add_model(name, gt[:min_len], preds[:min_len])

                result = comparator.compare()

                # Tabela podsumowujaca
                st.markdown("### Podsumowanie")
                summary = comparator.get_summary_table()

                # Stworz DataFrame-like display
                cols = st.columns(len(summary[0]))
                for i, key in enumerate(summary[0].keys()):
                    cols[i].markdown(f"**{key}**")

                for row in summary:
                    cols = st.columns(len(row))
                    for i, (key, val) in enumerate(row.items()):
                        cols[i].write(val)

                # Best model
                st.success(f"Najlepszy model: **{result.best_model}** (accuracy: {result.best_accuracy:.1%})")

                # Per-class comparison
                st.markdown("### Per-class F1 Score")
                per_class = comparator.get_per_class_comparison()

                for cls_id, data in per_class.items():
                    cls_name = data.get('class_name', f'Klasa {cls_id}')
                    cols = st.columns([1] + [1] * len(selected))
                    cols[0].write(cls_name)
                    for i, model_name in enumerate(selected):
                        score = data.get(model_name, 0)
                        cols[i+1].metric(model_name, f"{score:.1%}")

                # Agreement matrix
                if len(selected) > 1:
                    st.markdown("### Zgodność między modelami")
                    fig = go.Figure(data=go.Heatmap(
                        z=result.agreement_matrix,
                        x=selected,
                        y=selected,
                        colorscale='Viridis',
                        text=[[f"{v:.1%}" for v in row] for row in result.agreement_matrix],
                        texttemplate="%{text}",
                        textfont={"size": 14}
                    ))
                    fig.update_layout(height=400, title="Macierz zgodności")
                    st.plotly_chart(fig, use_container_width=True)

            else:
                # Bez GT - porownaj tylko zgodnosc
                st.markdown("### Porównanie bez ground truth")

                preds = {available[k]: st.session_state[k] for k in selected_keys}

                # Agreement
                if len(preds) >= 2:
                    names = list(preds.keys())
                    p1, p2 = preds[names[0]], preds[names[1]]
                    min_len = min(len(p1), len(p2))
                    agreement = np.mean(p1[:min_len] == p2[:min_len])

                    st.metric(f"Zgodność {names[0]} vs {names[1]}", f"{agreement:.1%}")

                # Rozklad klas
                for name, pred in preds.items():
                    unique, counts = np.unique(pred, return_counts=True)
                    st.markdown(f"**{name}:** {len(unique)} klas")

        except Exception as e:
            st.error(f"Błąd: {e}")
            import traceback
            st.code(traceback.format_exc())
