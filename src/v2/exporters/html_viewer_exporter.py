"""
HTML Viewer Exporter - eksport do interaktywnego viewera webowego

Generuje samodzielny plik HTML z wizualizacja 3D chmury punktow
uzywajac Three.js. Plik mozna otworzyc w przegladarce bez serwera.

Features:
- Interaktywna wizualizacja 3D (obrot, zoom, pan)
- Kolorowanie wg klas ASPRS
- Filtrowanie klas
- Statystyki
- Eksport do PNG
"""

import json
import base64
import gzip
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# Kolory klas ASPRS (hex)
CLASS_COLORS = {
    1: "#9E9E9E",   # Nieklasyfikowane
    2: "#8D6E63",   # Grunt
    3: "#AED581",   # Roslinnosc niska
    4: "#66BB6A",   # Roslinnosc srednia
    5: "#2E7D32",   # Roslinnosc wysoka
    6: "#D7CCC8",   # Budynek
    7: "#F44336",   # Szum
    9: "#29B6F6",   # Woda
    17: "#795548",  # Most
    18: "#6D4C41",  # Tory kolejowe
    19: "#FDD835",  # Linie energetyczne
    20: "#546E7A",  # Slupy trakcyjne
    21: "#78909C",  # Peron
    30: "#455A64",  # Jezdnia
    32: "#B0BEC5",  # Kraweznik
    35: "#FF5722",  # Znak drogowy
    36: "#90A4AE",  # Bariera
    40: "#BCAAA4",  # Sciana
    41: "#A1887F",  # Dach
}

CLASS_NAMES = {
    1: "Nieklasyfikowane",
    2: "Grunt",
    3: "Roslinnosc niska",
    4: "Roslinnosc srednia",
    5: "Roslinnosc wysoka",
    6: "Budynek",
    7: "Szum",
    9: "Woda",
    17: "Most",
    18: "Tory kolejowe",
    19: "Linie energetyczne",
    20: "Slupy trakcyjne",
    21: "Peron",
    30: "Jezdnia",
    32: "Kraweznik",
    35: "Znak drogowy",
    36: "Bariera",
    40: "Sciana",
    41: "Dach",
}


def export_to_html_viewer(
    coords: np.ndarray,
    classification: np.ndarray,
    output_path: str,
    title: str = "CPK Chmura+ Viewer",
    max_points: int = 500_000,
    point_size: float = 2.0
) -> str:
    """
    Eksportuje chmure punktow do interaktywnego viewera HTML

    Args:
        coords: (N, 3) wspolrzedne XYZ
        classification: (N,) klasy punktow
        output_path: sciezka do pliku HTML
        title: tytul strony
        max_points: max punktow (sampling dla duzych chmur)
        point_size: rozmiar punktow

    Returns:
        sciezka do zapisanego pliku
    """
    logger.info(f"Exporting HTML viewer: {len(coords):,} points")

    # Sampling jesli za duzo punktow
    if len(coords) > max_points:
        idx = np.random.choice(len(coords), max_points, replace=False)
        coords = coords[idx]
        classification = classification[idx]
        logger.info(f"Sampled to {max_points:,} points")

    # Normalizuj wspolrzedne (centruj i skaluj)
    center = coords.mean(axis=0)
    coords_centered = coords - center
    scale = np.abs(coords_centered).max()
    coords_normalized = coords_centered / scale

    # Przygotuj dane dla Three.js
    positions = coords_normalized.flatten().astype(np.float32)
    classes = classification.astype(np.uint8)

    # Zakoduj do base64
    positions_b64 = base64.b64encode(positions.tobytes()).decode('ascii')
    classes_b64 = base64.b64encode(classes.tobytes()).decode('ascii')

    # Statystyki
    unique_classes, counts = np.unique(classification, return_counts=True)
    class_stats = {}
    for cls_id, count in zip(unique_classes, counts):
        class_stats[int(cls_id)] = {
            "name": CLASS_NAMES.get(cls_id, f"Klasa {cls_id}"),
            "color": CLASS_COLORS.get(cls_id, "#888888"),
            "count": int(count),
            "percentage": round(count / len(classification) * 100, 2)
        }

    # Generuj HTML
    html_content = _generate_html(
        title=title,
        positions_b64=positions_b64,
        classes_b64=classes_b64,
        n_points=len(coords),
        class_stats=class_stats,
        center=center.tolist(),
        scale=float(scale),
        point_size=point_size
    )

    # Zapisz
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"Saved HTML viewer: {output_path}")
    return str(output_path)


def _generate_html(
    title: str,
    positions_b64: str,
    classes_b64: str,
    n_points: int,
    class_stats: Dict,
    center: list,
    scale: float,
    point_size: float
) -> str:
    """Generuje kompletny HTML z viewerem"""

    class_colors_js = json.dumps({str(k): v["color"] for k, v in class_stats.items()})
    class_stats_js = json.dumps(class_stats, ensure_ascii=False)

    return f'''<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: white; overflow: hidden; }}
        #container {{ width: 100vw; height: 100vh; }}
        #info {{ position: absolute; top: 10px; left: 10px; background: rgba(26, 35, 126, 0.9); padding: 15px 20px; border-radius: 10px; max-width: 300px; z-index: 100; }}
        #info h1 {{ font-size: 18px; margin-bottom: 10px; color: #90caf9; }}
        #info p {{ font-size: 13px; color: #e0e0e0; margin: 5px 0; }}
        #controls {{ position: absolute; top: 10px; right: 10px; background: rgba(26, 35, 126, 0.9); padding: 15px; border-radius: 10px; z-index: 100; max-height: 80vh; overflow-y: auto; }}
        #controls h3 {{ font-size: 14px; margin-bottom: 10px; color: #90caf9; }}
        .class-toggle {{ display: flex; align-items: center; margin: 5px 0; font-size: 12px; cursor: pointer; }}
        .class-toggle input {{ margin-right: 8px; }}
        .class-color {{ width: 14px; height: 14px; border-radius: 3px; margin-right: 8px; }}
        #stats {{ position: absolute; bottom: 10px; left: 10px; background: rgba(26, 35, 126, 0.9); padding: 10px 15px; border-radius: 10px; font-size: 12px; z-index: 100; }}
        #loading {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; z-index: 200; }}
        #loading .spinner {{ width: 50px; height: 50px; border: 4px solid #1976D2; border-top: 4px solid #90caf9; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 15px; }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
        .hidden {{ display: none !important; }}
    </style>
</head>
<body>
    <div id="loading">
        <div class="spinner"></div>
        <p>Ladowanie chmury punktow...</p>
    </div>

    <div id="info" class="hidden">
        <h1>{title}</h1>
        <p><strong>Punkty:</strong> {n_points:,}</p>
        <p><strong>Skala:</strong> {scale:.2f}m</p>
        <p id="fps"></p>
    </div>

    <div id="controls" class="hidden">
        <h3>Filtruj klasy</h3>
        <div id="class-toggles"></div>
        <hr style="margin: 10px 0; border-color: #444;">
        <button onclick="toggleAll(true)" style="margin: 5px 2px; padding: 5px 10px; cursor: pointer;">Wszystkie</button>
        <button onclick="toggleAll(false)" style="margin: 5px 2px; padding: 5px 10px; cursor: pointer;">Zadne</button>
    </div>

    <div id="stats" class="hidden">
        <span id="visible-count"></span> / {n_points:,} punktow
    </div>

    <div id="container"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>

    <script>
        // Dane
        const positionsB64 = "{positions_b64}";
        const classesB64 = "{classes_b64}";
        const classColors = {class_colors_js};
        const classStats = {class_stats_js};
        const nPoints = {n_points};
        const pointSize = {point_size};

        // Three.js setup
        let scene, camera, renderer, controls, points;
        let visibleClasses = new Set(Object.keys(classStats).map(Number));
        let originalColors;

        function init() {{
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);

            // Camera
            camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 100);
            camera.position.set(2, 2, 2);

            // Renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.getElementById('container').appendChild(renderer.domElement);

            // Controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.1;

            // Decode data
            const positionsArray = new Float32Array(atob(positionsB64).split('').map(c => c.charCodeAt(0)).reduce((acc, byte, i, arr) => {{
                if (i % 4 === 0) acc.push(0);
                acc[acc.length - 1] |= byte << (8 * (i % 4));
                return acc;
            }}, []).map(v => {{
                const buffer = new ArrayBuffer(4);
                new Uint32Array(buffer)[0] = v;
                return new Float32Array(buffer)[0];
            }}));

            const classesArray = new Uint8Array(atob(classesB64).split('').map(c => c.charCodeAt(0)));

            // Create geometry
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positionsArray, 3));

            // Colors based on class
            const colors = new Float32Array(nPoints * 3);
            originalColors = new Float32Array(nPoints * 3);

            for (let i = 0; i < nPoints; i++) {{
                const cls = classesArray[i];
                const colorHex = classColors[cls] || '#888888';
                const r = parseInt(colorHex.slice(1, 3), 16) / 255;
                const g = parseInt(colorHex.slice(3, 5), 16) / 255;
                const b = parseInt(colorHex.slice(5, 7), 16) / 255;
                colors[i * 3] = r;
                colors[i * 3 + 1] = g;
                colors[i * 3 + 2] = b;
                originalColors[i * 3] = r;
                originalColors[i * 3 + 1] = g;
                originalColors[i * 3 + 2] = b;
            }}

            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

            // Store classes for filtering
            geometry.userData.classes = classesArray;

            // Material
            const material = new THREE.PointsMaterial({{
                size: pointSize * 0.01,
                vertexColors: true,
                sizeAttenuation: true
            }});

            // Points
            points = new THREE.Points(geometry, material);
            scene.add(points);

            // Grid helper
            const gridHelper = new THREE.GridHelper(4, 20, 0x444444, 0x222222);
            scene.add(gridHelper);

            // Axes helper
            const axesHelper = new THREE.AxesHelper(1);
            scene.add(axesHelper);

            // UI
            createClassToggles();
            updateVisibleCount();

            // Hide loading, show UI
            document.getElementById('loading').classList.add('hidden');
            document.getElementById('info').classList.remove('hidden');
            document.getElementById('controls').classList.remove('hidden');
            document.getElementById('stats').classList.remove('hidden');

            // Events
            window.addEventListener('resize', onWindowResize);

            // Start animation
            animate();
        }}

        function createClassToggles() {{
            const container = document.getElementById('class-toggles');
            const sortedClasses = Object.keys(classStats).sort((a, b) => classStats[b].count - classStats[a].count);

            for (const cls of sortedClasses) {{
                const info = classStats[cls];
                const div = document.createElement('label');
                div.className = 'class-toggle';
                div.innerHTML = `
                    <input type="checkbox" checked data-class="${{cls}}">
                    <span class="class-color" style="background: ${{info.color}}"></span>
                    <span>${{info.name}} (${{info.percentage}}%)</span>
                `;
                div.querySelector('input').addEventListener('change', (e) => {{
                    const clsId = parseInt(e.target.dataset.class);
                    if (e.target.checked) {{
                        visibleClasses.add(clsId);
                    }} else {{
                        visibleClasses.delete(clsId);
                    }}
                    updatePointColors();
                }});
                container.appendChild(div);
            }}
        }}

        function toggleAll(visible) {{
            document.querySelectorAll('#class-toggles input').forEach(cb => {{
                cb.checked = visible;
                const cls = parseInt(cb.dataset.class);
                if (visible) visibleClasses.add(cls);
                else visibleClasses.delete(cls);
            }});
            updatePointColors();
        }}

        function updatePointColors() {{
            const colors = points.geometry.attributes.color.array;
            const classes = points.geometry.userData.classes;

            let visibleCount = 0;

            for (let i = 0; i < nPoints; i++) {{
                const cls = classes[i];
                if (visibleClasses.has(cls)) {{
                    colors[i * 3] = originalColors[i * 3];
                    colors[i * 3 + 1] = originalColors[i * 3 + 1];
                    colors[i * 3 + 2] = originalColors[i * 3 + 2];
                    visibleCount++;
                }} else {{
                    colors[i * 3] = 0;
                    colors[i * 3 + 1] = 0;
                    colors[i * 3 + 2] = 0;
                }}
            }}

            points.geometry.attributes.color.needsUpdate = true;
            document.getElementById('visible-count').textContent = visibleCount.toLocaleString();
        }}

        function updateVisibleCount() {{
            document.getElementById('visible-count').textContent = nPoints.toLocaleString();
        }}

        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}

        let frameCount = 0;
        let lastTime = performance.now();

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);

            // FPS counter
            frameCount++;
            const now = performance.now();
            if (now - lastTime >= 1000) {{
                document.getElementById('fps').textContent = `FPS: ${{frameCount}}`;
                frameCount = 0;
                lastTime = now;
            }}
        }}

        // Initialize when DOM is ready
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', init);
        }} else {{
            init();
        }}
    </script>
</body>
</html>
'''


class HTMLViewerExporter:
    """
    Klasa eksportera do formatu HTML viewer

    Usage:
        exporter = HTMLViewerExporter(coords, classification)
        exporter.save("output.html")
    """

    def __init__(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        title: str = "CPK Chmura+ Viewer",
        max_points: int = 500_000,
        point_size: float = 2.0
    ):
        self.coords = coords
        self.classification = classification
        self.title = title
        self.max_points = max_points
        self.point_size = point_size

    def save(self, output_path: str) -> str:
        """Zapisuje viewer do pliku HTML"""
        return export_to_html_viewer(
            self.coords,
            self.classification,
            output_path,
            self.title,
            self.max_points,
            self.point_size
        )
