"""
Infrastructure Report Generator - Generowanie raportów infrastruktury

Formaty wyjściowe:
- HTML (interaktywny)
- PDF (do druku)
- GeoJSON (dla GIS)
- KML (Google Earth)
- CSV (dane tabelaryczne)

HackNation 2025 - CPK Chmura+
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json
import logging

from .catenary import CatenarySystem, WireSegment
from .track_extraction import TrackSegment, TrackGeometry
from .pole_detection import Pole, PoleType
from .signal_detection import RailwaySignal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class InfrastructureInventory:
    """Kompletny inwentarz infrastruktury"""
    catenary: Optional[CatenarySystem] = None
    tracks: Optional[List[TrackSegment]] = None
    poles: Optional[List[Pole]] = None
    signals: Optional[List[RailwaySignal]] = None
    scan_date: str = ""
    scan_area_km2: float = 0
    total_points: int = 0
    file_name: str = ""


class InfrastructureReporter:
    """
    Generator raportów infrastruktury kolejowej

    Generuje kompleksowe raporty z:
    - Statystykami elementów
    - Mapami lokalizacji
    - Wykresami
    - Eksportem do różnych formatów
    """

    def __init__(self, inventory: InfrastructureInventory):
        self.inventory = inventory

    def generate_summary(self) -> Dict:
        """Generuj podsumowanie"""
        summary = {
            'scan_info': {
                'file': self.inventory.file_name,
                'date': self.inventory.scan_date or datetime.now().isoformat(),
                'area_km2': self.inventory.scan_area_km2,
                'total_points': self.inventory.total_points
            },
            'catenary': self._catenary_summary(),
            'tracks': self._tracks_summary(),
            'poles': self._poles_summary(),
            'signals': self._signals_summary()
        }
        return summary

    def _catenary_summary(self) -> Dict:
        if not self.inventory.catenary:
            return {'detected': False}

        cat = self.inventory.catenary
        return {
            'detected': True,
            'contact_wires': len(cat.contact_wires),
            'messenger_wires': len(cat.messenger_wires),
            'return_wires': len(cat.return_wires),
            'droppers': len(cat.droppers),
            'total_length_m': cat.total_length,
            'span_length_m': cat.span_length,
            'system_height_m': cat.system_height
        }

    def _tracks_summary(self) -> Dict:
        if not self.inventory.tracks:
            return {'detected': False}

        tracks = self.inventory.tracks
        total_length = sum(t.length for t in tracks)
        avg_gauge = np.mean([t.gauge for t in tracks])

        return {
            'detected': True,
            'segments': len(tracks),
            'total_length_m': total_length,
            'avg_gauge_m': avg_gauge,
            'gauge_deviation_mm': (avg_gauge - 1.435) * 1000,
            'straight_segments': sum(1 for t in tracks if t.geometry_type == 'straight'),
            'curved_segments': sum(1 for t in tracks if t.geometry_type == 'curve')
        }

    def _poles_summary(self) -> Dict:
        if not self.inventory.poles:
            return {'detected': False}

        poles = self.inventory.poles
        by_type = {}
        for p in poles:
            t = p.pole_type.value
            by_type[t] = by_type.get(t, 0) + 1

        return {
            'detected': True,
            'total_count': len(poles),
            'by_type': by_type,
            'avg_height_m': np.mean([p.height for p in poles]),
            'avg_spacing_m': self._calc_avg_spacing(poles)
        }

    def _signals_summary(self) -> Dict:
        if not self.inventory.signals:
            return {'detected': False}

        signals = self.inventory.signals
        by_type = {}
        for s in signals:
            t = s.signal_type.value
            by_type[t] = by_type.get(t, 0) + 1

        return {
            'detected': True,
            'total_count': len(signals),
            'by_type': by_type
        }

    def _calc_avg_spacing(self, poles: List[Pole]) -> float:
        if len(poles) < 2:
            return 0
        positions = np.array([p.position for p in poles])
        # Sortuj według X
        sorted_pos = positions[np.argsort(positions[:, 0])]
        diffs = np.diff(sorted_pos, axis=0)
        distances = np.linalg.norm(diffs[:, :2], axis=1)
        return np.mean(distances)

    def to_html(self, output_path: str):
        """Generuj raport HTML"""
        summary = self.generate_summary()

        html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raport Infrastruktury - CPK Chmura+</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #1a237e;
            border-bottom: 2px solid #1a237e;
            padding-bottom: 10px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .metric {{
            background: #e8eaf6;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #1a237e;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #1a237e;
            color: white;
        }}
        .status-ok {{ color: #2e7d32; }}
        .status-warning {{ color: #f57c00; }}
        .status-error {{ color: #c62828; }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚄 Raport Infrastruktury Kolejowej</h1>
        <p>CPK Chmura+ - HackNation 2025</p>
        <p>Data: {summary['scan_info']['date'][:10]} | Plik: {summary['scan_info']['file']}</p>
    </div>

    <div class="section">
        <h2>📊 Podsumowanie skanu</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{summary['scan_info']['total_points']:,}</div>
                <div class="metric-label">Punktów</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['scan_info']['area_km2']:.2f}</div>
                <div class="metric-label">Powierzchnia [km²]</div>
            </div>
        </div>
    </div>

    {self._catenary_html(summary['catenary'])}
    {self._tracks_html(summary['tracks'])}
    {self._poles_html(summary['poles'])}
    {self._signals_html(summary['signals'])}

    <div class="footer">
        <p>Wygenerowano przez CPK Chmura+ | HackNation 2025</p>
    </div>
</body>
</html>"""

        Path(output_path).write_text(html, encoding='utf-8')
        logger.info(f"HTML report saved to {output_path}")

    def _catenary_html(self, data: Dict) -> str:
        if not data.get('detected'):
            return ""
        return f"""
    <div class="section">
        <h2>⚡ Sieć trakcyjna</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{data['contact_wires']}</div>
                <div class="metric-label">Przewody jezdne</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data['messenger_wires']}</div>
                <div class="metric-label">Liny nośne</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data['total_length_m']:.0f}m</div>
                <div class="metric-label">Długość całkowita</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data['span_length_m']:.0f}m</div>
                <div class="metric-label">Rozstaw słupów</div>
            </div>
        </div>
    </div>"""

    def _tracks_html(self, data: Dict) -> str:
        if not data.get('detected'):
            return ""

        gauge_status = "status-ok" if abs(data['gauge_deviation_mm']) < 5 else "status-warning"

        return f"""
    <div class="section">
        <h2>🛤️ Tory</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{data['total_length_m']:.0f}m</div>
                <div class="metric-label">Długość torów</div>
            </div>
            <div class="metric">
                <div class="metric-value {gauge_status}">{data['avg_gauge_m']*1000:.0f}mm</div>
                <div class="metric-label">Rozstaw szyn</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data['straight_segments']}</div>
                <div class="metric-label">Odcinki proste</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data['curved_segments']}</div>
                <div class="metric-label">Łuki</div>
            </div>
        </div>
    </div>"""

    def _poles_html(self, data: Dict) -> str:
        if not data.get('detected'):
            return ""

        type_rows = "".join([
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in data['by_type'].items()
        ])

        return f"""
    <div class="section">
        <h2>🏗️ Słupy</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{data['total_count']}</div>
                <div class="metric-label">Liczba słupów</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data['avg_height_m']:.1f}m</div>
                <div class="metric-label">Średnia wysokość</div>
            </div>
            <div class="metric">
                <div class="metric-value">{data['avg_spacing_m']:.0f}m</div>
                <div class="metric-label">Średni rozstaw</div>
            </div>
        </div>
        <h3>Typy słupów</h3>
        <table>
            <tr><th>Typ</th><th>Liczba</th></tr>
            {type_rows}
        </table>
    </div>"""

    def _signals_html(self, data: Dict) -> str:
        if not data.get('detected'):
            return ""

        type_rows = "".join([
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in data['by_type'].items()
        ])

        return f"""
    <div class="section">
        <h2>🚦 Sygnalizacja</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{data['total_count']}</div>
                <div class="metric-label">Liczba sygnałów</div>
            </div>
        </div>
        <h3>Typy sygnałów</h3>
        <table>
            <tr><th>Typ</th><th>Liczba</th></tr>
            {type_rows}
        </table>
    </div>"""

    def to_csv(self, output_dir: str):
        """Eksportuj do plików CSV"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Słupy
        if self.inventory.poles:
            poles_data = []
            for p in self.inventory.poles:
                poles_data.append({
                    'x': p.position[0],
                    'y': p.position[1],
                    'z': p.position[2],
                    'height': p.height,
                    'type': p.pole_type.value,
                    'km': p.km_position or 0,
                    'side': p.side or ''
                })
            self._write_csv(output_path / 'poles.csv', poles_data)

        # Sygnały
        if self.inventory.signals:
            signals_data = []
            for s in self.inventory.signals:
                signals_data.append({
                    'x': s.position[0],
                    'y': s.position[1],
                    'z': s.position[2],
                    'height': s.height,
                    'type': s.signal_type.value,
                    'km': s.km_position or 0,
                    'side': s.side or ''
                })
            self._write_csv(output_path / 'signals.csv', signals_data)

        logger.info(f"CSV files saved to {output_dir}")

    def _write_csv(self, path: Path, data: List[Dict]):
        if not data:
            return
        import csv
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)


def generate_infrastructure_report(
    coords: np.ndarray,
    classification: np.ndarray,
    output_path: str,
    file_name: str = "scan"
) -> Dict:
    """
    Generuj kompletny raport infrastruktury

    Args:
        coords: współrzędne punktów
        classification: klasyfikacja
        output_path: ścieżka wyjściowa
        file_name: nazwa pliku źródłowego

    Returns:
        Dict z podsumowaniem
    """
    from .catenary import CatenaryDetector
    from .track_extraction import TrackExtractor
    from .pole_detection import PoleDetector
    from .signal_detection import SignalDetector

    logger.info("Generating infrastructure report...")

    # Wykryj elementy
    catenary_detector = CatenaryDetector(coords, classification)
    catenary = catenary_detector.detect()

    track_extractor = TrackExtractor(coords, classification)
    tracks = track_extractor.extract_tracks()

    pole_detector = PoleDetector(coords, classification)
    poles = pole_detector.detect()

    signal_detector = SignalDetector(coords, classification)
    signals = signal_detector.detect()

    # Oblicz powierzchnię
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    area_km2 = (x_range * y_range) / 1_000_000

    # Utwórz inwentarz
    inventory = InfrastructureInventory(
        catenary=catenary,
        tracks=tracks,
        poles=poles,
        signals=signals,
        scan_date=datetime.now().isoformat(),
        scan_area_km2=area_km2,
        total_points=len(coords),
        file_name=file_name
    )

    # Generuj raport
    reporter = InfrastructureReporter(inventory)
    reporter.to_html(output_path)

    return reporter.generate_summary()


def export_to_geojson(
    poles: List[Pole],
    signals: List[RailwaySignal],
    output_path: str
):
    """Eksportuj do GeoJSON"""
    features = []

    # Słupy
    for pole in poles:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [pole.position[0], pole.position[1]]
            },
            "properties": {
                "type": "pole",
                "pole_type": pole.pole_type.value,
                "height": pole.height,
                "km": pole.km_position
            }
        })

    # Sygnały
    for signal in signals:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [signal.position[0], signal.position[1]]
            },
            "properties": {
                "type": "signal",
                "signal_type": signal.signal_type.value,
                "height": signal.height
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2)

    logger.info(f"GeoJSON exported to {output_path}")


def export_to_kml(
    poles: List[Pole],
    signals: List[RailwaySignal],
    output_path: str
):
    """Eksportuj do KML (Google Earth)"""
    kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <name>CPK Railway Infrastructure</name>
    <description>Railway infrastructure detected from LiDAR</description>

    <Style id="poleStyle">
        <IconStyle>
            <color>ff0000ff</color>
            <scale>1.0</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/square.png</href></Icon>
        </IconStyle>
    </Style>

    <Style id="signalStyle">
        <IconStyle>
            <color>ff00ff00</color>
            <scale>0.8</scale>
            <Icon><href>http://maps.google.com/mapfiles/kml/shapes/triangle.png</href></Icon>
        </IconStyle>
    </Style>
"""

    kml_footer = """</Document>
</kml>"""

    placemarks = []

    for pole in poles:
        placemarks.append(f"""
    <Placemark>
        <name>Pole: {pole.pole_type.value}</name>
        <description>Height: {pole.height:.1f}m</description>
        <styleUrl>#poleStyle</styleUrl>
        <Point>
            <coordinates>{pole.position[0]},{pole.position[1]},{pole.position[2]}</coordinates>
        </Point>
    </Placemark>""")

    for signal in signals:
        placemarks.append(f"""
    <Placemark>
        <name>Signal: {signal.signal_type.value}</name>
        <description>Height: {signal.height:.1f}m</description>
        <styleUrl>#signalStyle</styleUrl>
        <Point>
            <coordinates>{signal.position[0]},{signal.position[1]},{signal.position[2]}</coordinates>
        </Point>
    </Placemark>""")

    kml_content = kml_header + "\n".join(placemarks) + kml_footer

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(kml_content)

    logger.info(f"KML exported to {output_path}")
