"""
GeoJSON Exporter - eksport klasyfikacji do formatu GeoJSON

Format GeoJSON jest standardem dla danych geograficznych i moze byc
importowany do:
- QGIS, ArcGIS
- Leaflet, Mapbox
- PostGIS
- Google Earth

Eksportuje:
1. Granice obszaru jako Polygon
2. Statystyki klas jako Properties
3. Centroidy klas jako Points (opcjonalnie)
"""

import json
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Mapa klas ASPRS
CLASS_NAMES = {
    1: "Nieklasyfikowane",
    2: "Grunt",
    3: "Roslinnosc_niska",
    4: "Roslinnosc_srednia",
    5: "Roslinnosc_wysoka",
    6: "Budynek",
    7: "Szum",
    9: "Woda",
    17: "Most",
    18: "Tory_kolejowe",
    19: "Linie_energetyczne",
    20: "Slupy_trakcyjne",
    21: "Peron",
    30: "Jezdnia",
    32: "Kraweznik",
    35: "Znak_drogowy",
    36: "Bariera",
    40: "Sciana",
    41: "Dach",
}


@dataclass
class GeoJSONConfig:
    """Konfiguracja eksportu GeoJSON"""
    include_class_centroids: bool = True  # Dodaj centroidy jako punkty
    include_class_polygons: bool = False  # Konweksowe otoczki per klasa (wolne)
    include_statistics: bool = True  # Statystyki w properties
    sample_centroids: int = 1000  # Max punktow do obliczenia centroidow
    precision: int = 6  # Precyzja wspolrzednych


class GeoJSONExporter:
    """
    Eksporter do formatu GeoJSON

    Usage:
        exporter = GeoJSONExporter(coords, classification)
        geojson = exporter.export()
        exporter.save("output.geojson")
    """

    def __init__(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        config: Optional[GeoJSONConfig] = None,
        crs: str = "EPSG:2180"  # Domyslny uklad PL-2000
    ):
        """
        Args:
            coords: (N, 3) wspolrzedne XYZ
            classification: (N,) klasy punktow
            config: konfiguracja eksportu
            crs: uklad wspolrzednych (informacyjnie)
        """
        self.coords = coords
        self.classification = classification
        self.config = config or GeoJSONConfig()
        self.crs = crs
        self.n_points = len(coords)

        logger.info(f"GeoJSONExporter: {self.n_points:,} points")

    def export(self) -> Dict:
        """
        Eksportuje dane do formatu GeoJSON

        Returns:
            Dict - GeoJSON FeatureCollection
        """
        logger.info("Exporting to GeoJSON...")

        features = []

        # 1. Granice obszaru jako Polygon
        bounds_feature = self._create_bounds_feature()
        features.append(bounds_feature)

        # 2. Centroidy klas jako Points
        if self.config.include_class_centroids:
            class_features = self._create_class_features()
            features.extend(class_features)

        # 3. Statystyki ogolne
        stats = self._compute_statistics()

        geojson = {
            "type": "FeatureCollection",
            "name": "CPK_Classification",
            "crs": {
                "type": "name",
                "properties": {"name": self.crs}
            },
            "properties": {
                "generator": "CPK Chmura+ Classifier v2.0",
                "total_points": self.n_points,
                **stats
            },
            "features": features
        }

        logger.info(f"Exported {len(features)} features")
        return geojson

    def _create_bounds_feature(self) -> Dict:
        """Tworzy feature z granicami obszaru"""
        x_min, x_max = float(self.coords[:, 0].min()), float(self.coords[:, 0].max())
        y_min, y_max = float(self.coords[:, 1].min()), float(self.coords[:, 1].max())
        z_min, z_max = float(self.coords[:, 2].min()), float(self.coords[:, 2].max())

        # Polygon (bounding box)
        coords = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
            [x_min, y_min]  # Zamkniecie
        ]

        # Zaokraglij
        p = self.config.precision
        coords = [[round(c[0], p), round(c[1], p)] for c in coords]

        return {
            "type": "Feature",
            "id": "bounds",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            },
            "properties": {
                "feature_type": "bounding_box",
                "x_min": round(x_min, p),
                "x_max": round(x_max, p),
                "y_min": round(y_min, p),
                "y_max": round(y_max, p),
                "z_min": round(z_min, p),
                "z_max": round(z_max, p),
                "width_m": round(x_max - x_min, 2),
                "height_m": round(y_max - y_min, 2),
                "elevation_range_m": round(z_max - z_min, 2)
            }
        }

    def _create_class_features(self) -> List[Dict]:
        """Tworzy features dla kazdej klasy (centroid + stats)"""
        features = []
        unique_classes = np.unique(self.classification)

        for cls_id in unique_classes:
            mask = self.classification == cls_id
            count = int(mask.sum())

            if count == 0:
                continue

            # Centroid klasy
            class_coords = self.coords[mask]

            # Sampling dla duzych klas
            if len(class_coords) > self.config.sample_centroids:
                idx = np.random.choice(len(class_coords), self.config.sample_centroids, replace=False)
                class_coords = class_coords[idx]

            centroid_x = float(class_coords[:, 0].mean())
            centroid_y = float(class_coords[:, 1].mean())
            centroid_z = float(class_coords[:, 2].mean())

            # Nazwa klasy
            class_name = CLASS_NAMES.get(cls_id, f"Klasa_{cls_id}")

            p = self.config.precision

            feature = {
                "type": "Feature",
                "id": f"class_{cls_id}",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(centroid_x, p), round(centroid_y, p)]
                },
                "properties": {
                    "feature_type": "class_centroid",
                    "class_id": int(cls_id),
                    "class_name": class_name,
                    "point_count": count,
                    "percentage": round(count / self.n_points * 100, 2),
                    "centroid_z": round(centroid_z, 2),
                    "z_min": round(float(self.coords[mask, 2].min()), 2),
                    "z_max": round(float(self.coords[mask, 2].max()), 2)
                }
            }

            features.append(feature)

        return features

    def _compute_statistics(self) -> Dict:
        """Oblicza statystyki klasyfikacji"""
        unique, counts = np.unique(self.classification, return_counts=True)

        class_stats = {}
        for cls_id, count in zip(unique, counts):
            class_name = CLASS_NAMES.get(cls_id, f"Klasa_{cls_id}")
            class_stats[f"class_{cls_id}_{class_name}"] = {
                "count": int(count),
                "percentage": round(float(count / self.n_points * 100), 2)
            }

        classified = self.classification != 1
        return {
            "classified_points": int(classified.sum()),
            "classified_percentage": round(float(classified.sum() / self.n_points * 100), 2),
            "n_classes": len(unique),
            "classes": class_stats
        }

    def save(self, output_path: str) -> str:
        """
        Zapisuje GeoJSON do pliku

        Args:
            output_path: sciezka do pliku wyjsciowego

        Returns:
            sciezka do zapisanego pliku
        """
        geojson = self.export()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved GeoJSON: {output_path}")
        return str(output_path)

    def to_string(self) -> str:
        """Eksportuje do stringa JSON"""
        geojson = self.export()
        return json.dumps(geojson, ensure_ascii=False, indent=2)


def export_to_geojson(
    coords: np.ndarray,
    classification: np.ndarray,
    output_path: Optional[str] = None,
    crs: str = "EPSG:2180"
) -> str:
    """
    Convenience function do eksportu GeoJSON

    Args:
        coords: wspolrzedne
        classification: klasyfikacja
        output_path: sciezka wyjsciowa (opcjonalnie)
        crs: uklad wspolrzednych

    Returns:
        JSON string lub sciezka do pliku
    """
    exporter = GeoJSONExporter(coords, classification, crs=crs)

    if output_path:
        return exporter.save(output_path)
    else:
        return exporter.to_string()
