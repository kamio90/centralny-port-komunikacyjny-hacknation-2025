"""
Interaktywna wizualizacja 3D z filtrowaniem klas
HackNation 2025 - CPK Chmura+
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Optional
import pandas as pd


# Mapa klas ASPRS z kolorami
CLASS_INFO = {
    1: {"name": "Nieklasyfikowane", "color": "#9E9E9E", "icon": "❓"},
    2: {"name": "Grunt", "color": "#8D6E63", "icon": "🌍"},
    3: {"name": "Roslinnosc niska", "color": "#AED581", "icon": "🌱"},
    4: {"name": "Roslinnosc srednia", "color": "#66BB6A", "icon": "🌿"},
    5: {"name": "Roslinnosc wysoka", "color": "#2E7D32", "icon": "🌳"},
    6: {"name": "Budynek", "color": "#D7CCC8", "icon": "🏢"},
    7: {"name": "Szum", "color": "#F44336", "icon": "⚡"},
    9: {"name": "Woda", "color": "#29B6F6", "icon": "💧"},
    17: {"name": "Most", "color": "#795548", "icon": "🌉"},
    18: {"name": "Tory kolejowe", "color": "#6D4C41", "icon": "🛤️"},
    19: {"name": "Linie energetyczne", "color": "#FDD835", "icon": "⚡"},
    20: {"name": "Slupy trakcyjne", "color": "#546E7A", "icon": "🗼"},
    21: {"name": "Peron", "color": "#78909C", "icon": "🚉"},
    30: {"name": "Jezdnia", "color": "#455A64", "icon": "🛣️"},
    32: {"name": "Kraweznik", "color": "#B0BEC5", "icon": "⬜"},
    35: {"name": "Znak drogowy", "color": "#FF5722", "icon": "🚧"},
    36: {"name": "Bariera", "color": "#90A4AE", "icon": "🚧"},
    40: {"name": "Sciana", "color": "#BCAAA4", "icon": "🧱"},
    41: {"name": "Dach", "color": "#A1887F", "icon": "🏠"},
}


def render_interactive_viz(coords: np.ndarray, classification: np.ndarray,
                           colors: Optional[np.ndarray] = None,
                           title: str = "Wizualizacja 3D"):
    """
    Renderuje interaktywna wizualizacje 3D z filtrowaniem klas

    Args:
        coords: (N, 3) wspolrzedne XYZ
        classification: (N,) klasy punktow
        colors: (N, 3) opcjonalne kolory RGB
        title: tytul wizualizacji
    """

    st.subheader(f"🎨 {title}")

    # Pobierz unikalne klasy
    unique_classes = sorted(np.unique(classification))

    # Panel kontrolny
    with st.expander("⚙️ Ustawienia wizualizacji", expanded=True):
        col1, col2 = st.columns([2, 1])

        with col1:
            # Wybor klas do wyswietlenia
            st.markdown("**Filtruj klasy:**")

            # Quick actions
            qcol1, qcol2, qcol3 = st.columns(3)
            with qcol1:
                if st.button("✅ Wszystkie", key="viz_all"):
                    st.session_state['viz_selected_classes'] = list(unique_classes)
            with qcol2:
                if st.button("❌ Zadne", key="viz_none"):
                    st.session_state['viz_selected_classes'] = []
            with qcol3:
                if st.button("🌍 Tylko teren", key="viz_terrain"):
                    st.session_state['viz_selected_classes'] = [2, 3, 4, 5]

            # Multiselect z ikonami
            class_options = []
            for cls_id in unique_classes:
                info = CLASS_INFO.get(cls_id, {"name": f"Klasa {cls_id}", "icon": "📦"})
                count = np.sum(classification == cls_id)
                pct = count / len(classification) * 100
                class_options.append(f"{info['icon']} [{cls_id}] {info['name']} ({pct:.1f}%)")

            default_selected = st.session_state.get('viz_selected_classes', list(unique_classes))

            selected_display = st.multiselect(
                "Wybrane klasy:",
                class_options,
                default=[class_options[i] for i, c in enumerate(unique_classes) if c in default_selected],
                key="viz_class_select"
            )

            # Parsuj wybrane klasy
            selected_classes = []
            for s in selected_display:
                # Extract class ID from string like "🌍 [2] Grunt (48.5%)"
                try:
                    cls_id = int(s.split('[')[1].split(']')[0])
                    selected_classes.append(cls_id)
                except:
                    pass

            st.session_state['viz_selected_classes'] = selected_classes

        with col2:
            # Ustawienia wyswietlania
            max_pts = st.slider("Max punktow", 10_000, 500_000, 100_000, 10_000, key="viz_max")
            point_size = st.slider("Rozmiar punktu", 1, 5, 2, key="viz_size")
            color_mode = st.radio("Kolorowanie", ["Klasy", "RGB", "Wysokosc"], key="viz_color")

    # Filtruj punkty
    if not selected_classes:
        st.warning("Wybierz co najmniej jedna klase")
        return

    mask = np.isin(classification, selected_classes)
    filtered_coords = coords[mask]
    filtered_class = classification[mask]
    filtered_colors = colors[mask] if colors is not None else None

    # Sampling jesli za duzo
    if len(filtered_coords) > max_pts:
        idx = np.random.choice(len(filtered_coords), max_pts, replace=False)
        filtered_coords = filtered_coords[idx]
        filtered_class = filtered_class[idx]
        if filtered_colors is not None:
            filtered_colors = filtered_colors[idx]

    # Przygotuj kolory
    if color_mode == "Klasy":
        point_colors = [CLASS_INFO.get(int(c), {"color": "#888"})['color'] for c in filtered_class]
    elif color_mode == "RGB" and filtered_colors is not None:
        # Konwertuj RGB [0-1] na hex
        point_colors = ['rgb({},{},{})'.format(
            int(c[0]*255), int(c[1]*255), int(c[2]*255)
        ) for c in filtered_colors]
    else:
        # Wysokosc
        z_norm = (filtered_coords[:, 2] - filtered_coords[:, 2].min()) / \
                 (filtered_coords[:, 2].max() - filtered_coords[:, 2].min() + 0.001)
        point_colors = z_norm

    # Plotly 3D
    if color_mode == "Wysokosc":
        fig = go.Figure(data=[go.Scatter3d(
            x=filtered_coords[:, 0],
            y=filtered_coords[:, 1],
            z=filtered_coords[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=point_colors,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Wysokosc")
            ),
            hoverinfo='skip'
        )])
    else:
        fig = go.Figure(data=[go.Scatter3d(
            x=filtered_coords[:, 0],
            y=filtered_coords[:, 1],
            z=filtered_coords[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=point_colors,
                opacity=0.8
            ),
            hoverinfo='skip'
        )])

    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title='X [m]',
            yaxis_title='Y [m]',
            zaxis_title='Z [m]',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                eye=dict(x=1.5, y=1.5, z=1.0)
            )
        ),
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Statystyki pod wizualizacja
    st.caption(f"Wyswietlono: {len(filtered_coords):,} / {len(coords):,} punktow "
               f"({len(selected_classes)} klas)")

    # Legenda
    _render_legend(unique_classes, classification)


def _render_legend(classes: List[int], classification: np.ndarray):
    """Renderuje legende klas"""

    st.markdown("**Legenda:**")

    # Podziel na kolumny
    n_cols = min(len(classes), 6)
    cols = st.columns(n_cols)

    for i, cls_id in enumerate(sorted(classes)):
        info = CLASS_INFO.get(cls_id, {"name": f"Klasa {cls_id}", "color": "#888", "icon": "📦"})
        count = np.sum(classification == cls_id)
        pct = count / len(classification) * 100

        with cols[i % n_cols]:
            st.markdown(f"""
            <div style='display: flex; align-items: center; margin: 3px 0;'>
                <div style='width: 15px; height: 15px; background: {info["color"]};
                            border-radius: 3px; margin-right: 5px;'></div>
                <span style='font-size: 11px;'>{info["icon"]} {info["name"]}<br/>
                <small style='color: #888;'>{pct:.1f}%</small></span>
            </div>
            """, unsafe_allow_html=True)


def render_comparison_view(coords: np.ndarray,
                           original_class: np.ndarray,
                           new_class: np.ndarray,
                           max_pts: int = 50_000):
    """
    Renderuje widok porownawczy przed/po klasyfikacji

    Args:
        coords: wspolrzedne
        original_class: oryginalna klasyfikacja (lub None dla nieklasyfikowanych)
        new_class: nowa klasyfikacja
        max_pts: max punktow do wyswietlenia
    """

    st.subheader("📊 Porownanie przed/po")

    # Sampling
    if len(coords) > max_pts:
        idx = np.random.choice(len(coords), max_pts, replace=False)
        coords = coords[idx]
        original_class = original_class[idx] if original_class is not None else np.ones(max_pts)
        new_class = new_class[idx]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Przed klasyfikacja:**")
        colors_before = [CLASS_INFO.get(int(c), {"color": "#888"})['color'] for c in original_class]

        fig1 = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode='markers',
            marker=dict(size=1, color=colors_before, opacity=0.7),
            hoverinfo='skip'
        )])
        fig1.update_layout(
            scene=dict(aspectmode='data'),
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("**Po klasyfikacji:**")
        colors_after = [CLASS_INFO.get(int(c), {"color": "#888"})['color'] for c in new_class]

        fig2 = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode='markers',
            marker=dict(size=1, color=colors_after, opacity=0.7),
            hoverinfo='skip'
        )])
        fig2.update_layout(
            scene=dict(aspectmode='data'),
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Statystyki zmian
    st.markdown("**Zmiany:**")
    changes = original_class != new_class
    st.metric("Punktow zmienionych", f"{np.sum(changes):,} ({np.sum(changes)/len(changes)*100:.1f}%)")


def render_statistics_dashboard(stats: Dict, n_points: int, elapsed: float):
    """
    Renderuje dashboard ze statystykami

    Args:
        stats: slownik ze statystykami z pipeline
        n_points: liczba punktow
        elapsed: czas przetwarzania
    """

    st.subheader("📊 Dashboard statystyk")

    # Glowne metryki
    col1, col2, col3, col4 = st.columns(4)

    classified_pct = stats.get('summary', {}).get('classified_percentage', 0)
    n_classes = len([k for k in stats.get('classification', {}).keys() if k != 1])

    with col1:
        st.metric(
            "Punkty",
            f"{n_points/1e6:.1f}M" if n_points > 1e6 else f"{n_points:,}",
            help="Calkowita liczba punktow"
        )
    with col2:
        st.metric(
            "Sklasyfikowane",
            f"{classified_pct:.1f}%",
            delta=f"+{classified_pct:.0f}%" if classified_pct > 80 else None,
            delta_color="normal"
        )
    with col3:
        st.metric(
            "Wykryte klasy",
            n_classes,
            delta="+5 wymagane" if n_classes >= 5 else f"{5-n_classes} brakuje",
            delta_color="normal" if n_classes >= 5 else "inverse"
        )
    with col4:
        speed = n_points / elapsed if elapsed > 0 else 0
        st.metric(
            "Predkosc",
            f"{speed/1000:.0f}k/s",
            help="Punktow na sekunde"
        )

    # Wykres kolowy
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Rozklad klas**")

        class_stats = stats.get('classification', {})
        if class_stats:
            labels = []
            values = []
            colors = []

            for cls_id, info in sorted(class_stats.items(), key=lambda x: x[1]['count'], reverse=True):
                cls_info = CLASS_INFO.get(cls_id, {"name": f"Klasa {cls_id}", "color": "#888"})
                labels.append(f"[{cls_id}] {cls_info['name']}")
                values.append(info['count'])
                colors.append(cls_info['color'])

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker_colors=colors,
                hole=0.4,
                textinfo='percent+label',
                textposition='outside'
            )])
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Szczegoly klas**")

        # Tabela z klasami
        data = []
        for cls_id, info in sorted(class_stats.items(), key=lambda x: x[1]['count'], reverse=True):
            cls_info = CLASS_INFO.get(cls_id, {"name": f"Klasa {cls_id}", "icon": "📦"})
            data.append({
                'Klasa': f"{cls_info['icon']} {cls_info['name']}",
                'Punkty': f"{info['count']:,}",
                '%': f"{info['percentage']:.1f}%"
            })

        df = pd.DataFrame(data)
        st.dataframe(df, hide_index=True, use_container_width=True)

    # Wydajnosc pipeline
    st.markdown("---")
    st.markdown("**Wydajnosc pipeline**")

    steps = stats.get('steps', {})
    if steps:
        step_data = []
        for step_name, step_info in steps.items():
            if 'time' in step_info:
                step_data.append({
                    'Krok': step_name.capitalize(),
                    'Czas': f"{step_info['time']:.2f}s",
                    'Czas_val': step_info['time']
                })

        if step_data:
            df_steps = pd.DataFrame(step_data)

            fig = go.Figure(data=[go.Bar(
                x=[d['Krok'] for d in step_data],
                y=[d['Czas_val'] for d in step_data],
                marker_color='#1976D2',
                text=[d['Czas'] for d in step_data],
                textposition='auto'
            )])
            fig.update_layout(
                height=250,
                xaxis_title="Krok pipeline",
                yaxis_title="Czas [s]",
                margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
