"""
LOD Classifier - Klasyfikacja poziomu szczegolowosci (Level of Detail/Development)

Poziomy LOD w BIM:
- LOD 100: Koncepcyjny (masa, objetosc)
- LOD 200: Przyblizona geometria
- LOD 300: Dokladna geometria
- LOD 350: Koordynacja (LOD 300 + interfejsy)
- LOD 400: Fabrication (gotowe do produkcji)
- LOD 500: As-built (stan faktyczny)

HackNation 2025 - CPK Chmura+
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LODLevel(Enum):
    """Poziomy LOD"""
    LOD_100 = 100  # Koncepcyjny
    LOD_200 = 200  # Przyblizona geometria
    LOD_300 = 300  # Dokladna geometria
    LOD_350 = 350  # Koordynacja
    LOD_400 = 400  # Fabrication
    LOD_500 = 500  # As-built


@dataclass
class LODResult:
    """Wynik klasyfikacji LOD"""
    level: LODLevel
    confidence: float
    point_density: float  # punkty/m2
    geometry_completeness: float  # 0-1
    detail_score: float  # 0-1
    recommendations: List[str]
    metrics: Dict


class LODClassifier:
    """
    Klasyfikator poziomu szczegolowosci dla chmur punktow

    Ocenia jakosc danych LiDAR i okresla odpowiedni poziom LOD
    dla modelowania BIM.

    Kryteria:
    - Gestosc punktow
    - Kompletnosc geometrii
    - Dokladnosc krawedzi
    - Poziom szumu

    Usage:
        classifier = LODClassifier(coords, classification)
        result = classifier.classify()
    """

    # Progi gestosci dla LOD [punkty/m2]
    DENSITY_THRESHOLDS = {
        LODLevel.LOD_100: 1,
        LODLevel.LOD_200: 5,
        LODLevel.LOD_300: 20,
        LODLevel.LOD_350: 50,
        LODLevel.LOD_400: 100,
        LODLevel.LOD_500: 200
    }

    def __init__(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        bounds: Optional[Dict] = None
    ):
        """
        Args:
            coords: wspolrzedne punktow
            classification: klasyfikacja
            bounds: granice obszaru (opcjonalnie)
        """
        self.coords = coords
        self.classification = classification

        if bounds:
            self.bounds = bounds
        else:
            self.bounds = {
                'x': (coords[:, 0].min(), coords[:, 0].max()),
                'y': (coords[:, 1].min(), coords[:, 1].max()),
                'z': (coords[:, 2].min(), coords[:, 2].max())
            }

        # Oblicz podstawowe metryki
        self._compute_basic_metrics()

    def _compute_basic_metrics(self):
        """Oblicz podstawowe metryki"""
        # Powierzchnia
        x_range = self.bounds['x'][1] - self.bounds['x'][0]
        y_range = self.bounds['y'][1] - self.bounds['y'][0]
        self.area_m2 = x_range * y_range

        # Gestosc
        self.point_density = len(self.coords) / self.area_m2 if self.area_m2 > 0 else 0

        # Unikalne klasy
        self.unique_classes = np.unique(self.classification)

        logger.info(f"Area: {self.area_m2:.0f} m2, Density: {self.point_density:.1f} pts/m2")

    def classify(self) -> LODResult:
        """
        Klasyfikuj poziom LOD

        Returns:
            LODResult z wynikiem klasyfikacji
        """
        # Metryki
        density_score = self._evaluate_density()
        completeness_score = self._evaluate_completeness()
        detail_score = self._evaluate_detail()
        noise_score = self._evaluate_noise()

        # Srednia wazona
        overall_score = (
            density_score * 0.4 +
            completeness_score * 0.3 +
            detail_score * 0.2 +
            noise_score * 0.1
        )

        # Okresl LOD
        lod_level = self._score_to_lod(overall_score)

        # Confidence
        confidence = min(1.0, overall_score / 100)

        # Rekomendacje
        recommendations = self._generate_recommendations(
            density_score, completeness_score, detail_score, noise_score
        )

        return LODResult(
            level=lod_level,
            confidence=confidence,
            point_density=self.point_density,
            geometry_completeness=completeness_score / 100,
            detail_score=detail_score / 100,
            recommendations=recommendations,
            metrics={
                'density_score': density_score,
                'completeness_score': completeness_score,
                'detail_score': detail_score,
                'noise_score': noise_score,
                'overall_score': overall_score,
                'area_m2': self.area_m2,
                'point_count': len(self.coords),
                'class_count': len(self.unique_classes)
            }
        )

    def _evaluate_density(self) -> float:
        """Ocena gestosci punktow (0-100)"""
        # Logarytmiczna skala
        if self.point_density < 1:
            return 10
        elif self.point_density < 5:
            return 30
        elif self.point_density < 20:
            return 50
        elif self.point_density < 50:
            return 70
        elif self.point_density < 100:
            return 85
        else:
            return min(100, 85 + (self.point_density - 100) / 10)

    def _evaluate_completeness(self) -> float:
        """Ocena kompletnosci geometrii (0-100)"""
        # Sprawdz pokrycie przestrzenne
        x_range = self.bounds['x'][1] - self.bounds['x'][0]
        y_range = self.bounds['y'][1] - self.bounds['y'][0]

        if x_range == 0 or y_range == 0:
            return 0

        # Podziel na siatke i sprawdz pokrycie
        grid_size = 10  # 10m x 10m
        nx = max(1, int(x_range / grid_size))
        ny = max(1, int(y_range / grid_size))

        # Stworz siatke
        grid = np.zeros((ny, nx), dtype=bool)

        # Przypisz punkty do komorek
        ix = ((self.coords[:, 0] - self.bounds['x'][0]) / grid_size).astype(int)
        iy = ((self.coords[:, 1] - self.bounds['y'][0]) / grid_size).astype(int)

        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)

        grid[iy, ix] = True

        # Procent pokrycia
        coverage = grid.sum() / grid.size
        return coverage * 100

    def _evaluate_detail(self) -> float:
        """Ocena poziomu szczegolowosci (0-100)"""
        score = 0

        # Bonus za rozne klasy
        class_score = min(30, len(self.unique_classes) * 3)
        score += class_score

        # Bonus za budynki
        if 6 in self.unique_classes:
            building_mask = self.classification == 6
            building_count = building_mask.sum()
            if building_count > 100:
                score += 20

        # Bonus za infrastrukture
        infra_classes = [18, 19, 20]
        for cls in infra_classes:
            if cls in self.unique_classes:
                score += 10

        # Bonus za roslinnosc (wiele warstw)
        veg_classes = [3, 4, 5]
        veg_count = sum(1 for cls in veg_classes if cls in self.unique_classes)
        score += veg_count * 5

        return min(100, score)

    def _evaluate_noise(self) -> float:
        """Ocena poziomu szumu (0-100, wyzszy = lepszy)"""
        # Sprawdz klase 7 (noise)
        if 7 in self.unique_classes:
            noise_mask = self.classification == 7
            noise_ratio = noise_mask.sum() / len(self.coords)
            # Im mniej szumu, tym lepiej
            return max(0, 100 - noise_ratio * 500)
        return 100  # Brak zidentyfikowanego szumu

    def _score_to_lod(self, score: float) -> LODLevel:
        """Konwertuj score na poziom LOD"""
        if score >= 90:
            return LODLevel.LOD_500
        elif score >= 75:
            return LODLevel.LOD_400
        elif score >= 60:
            return LODLevel.LOD_350
        elif score >= 45:
            return LODLevel.LOD_300
        elif score >= 30:
            return LODLevel.LOD_200
        else:
            return LODLevel.LOD_100

    def _generate_recommendations(
        self,
        density: float,
        completeness: float,
        detail: float,
        noise: float
    ) -> List[str]:
        """Generuj rekomendacje"""
        recommendations = []

        if density < 50:
            recommendations.append(
                f"Niska gestosc punktow ({self.point_density:.1f} pts/m2). "
                "Rozważ dodatkowy skan lub zmniejszenie odleglosci skanowania."
            )

        if completeness < 70:
            recommendations.append(
                "Niepelne pokrycie przestrzenne. "
                "Sprawdz czy nie ma luk w danych."
            )

        if detail < 50:
            recommendations.append(
                "Ograniczone rozpoznanie elementow. "
                "Rozważ dodatkowa klasyfikacje lub filtrowanie."
            )

        if noise < 80:
            recommendations.append(
                "Wykryto znaczny poziom szumu. "
                "Rozważ filtrowanie outlierow."
            )

        if not recommendations:
            recommendations.append(
                "Dane dobrej jakosci, odpowiednie do modelowania BIM."
            )

        return recommendations

    def classify_element(
        self,
        element_coords: np.ndarray,
        element_type: str = "building"
    ) -> LODResult:
        """
        Klasyfikuj LOD dla pojedynczego elementu

        Args:
            element_coords: punkty elementu
            element_type: typ elementu

        Returns:
            LODResult
        """
        # Lokalna gestosc
        if len(element_coords) < 10:
            return LODResult(
                level=LODLevel.LOD_100,
                confidence=0.3,
                point_density=0,
                geometry_completeness=0,
                detail_score=0,
                recommendations=["Za malo punktow dla elementu"],
                metrics={}
            )

        # Oblicz lokalne metryki
        x_range = element_coords[:, 0].max() - element_coords[:, 0].min()
        y_range = element_coords[:, 1].max() - element_coords[:, 1].min()
        z_range = element_coords[:, 2].max() - element_coords[:, 2].min()

        # Powierzchnia (przyblizona)
        footprint_area = max(x_range * y_range, 1)
        local_density = len(element_coords) / footprint_area

        # Ocena krawedzi
        edge_quality = self._evaluate_edge_quality(element_coords)

        # Score
        density_score = min(100, local_density * 2)
        edge_score = edge_quality * 100

        overall = (density_score * 0.5 + edge_score * 0.5)
        lod = self._score_to_lod(overall)

        return LODResult(
            level=lod,
            confidence=min(1.0, len(element_coords) / 200),
            point_density=local_density,
            geometry_completeness=edge_score,
            detail_score=overall / 100,
            recommendations=[],
            metrics={
                'point_count': len(element_coords),
                'footprint_area': footprint_area,
                'height': z_range,
                'edge_quality': edge_quality
            }
        )

    def _evaluate_edge_quality(self, coords: np.ndarray) -> float:
        """Ocena jakosci krawedzi (0-1)"""
        if len(coords) < 20:
            return 0.3

        # Sprawdz rozklad punktow na granicy
        # Uzyj convex hull
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords[:, :2])

            # Stosunek punktow na brzegu do wszystkich
            boundary_ratio = len(hull.vertices) / len(coords)

            # Optymalnie: 10-30% punktow na brzegu
            if 0.1 <= boundary_ratio <= 0.3:
                return 0.9
            elif 0.05 <= boundary_ratio <= 0.4:
                return 0.7
            else:
                return 0.5

        except Exception:
            return 0.5

    def get_lod_requirements(self, target_lod: LODLevel) -> Dict:
        """
        Zwroc wymagania dla danego poziomu LOD

        Args:
            target_lod: docelowy LOD

        Returns:
            Dict z wymaganiami
        """
        requirements = {
            LODLevel.LOD_100: {
                "min_density": 1,
                "description": "Masy i objetosci",
                "geometry": "Bounding box lub prosta bryła",
                "accuracy_m": 1.0,
                "use_cases": ["Studia koncepcyjne", "Analizy masy"]
            },
            LODLevel.LOD_200: {
                "min_density": 5,
                "description": "Przyblizona geometria",
                "geometry": "Uproszczone ksztalty z orientacja",
                "accuracy_m": 0.5,
                "use_cases": ["Planowanie przestrzenne", "Wstepne koordynacja"]
            },
            LODLevel.LOD_300: {
                "min_density": 20,
                "description": "Dokladna geometria",
                "geometry": "Rzeczywiste ksztalty i wymiary",
                "accuracy_m": 0.15,
                "use_cases": ["Dokumentacja projektowa", "Koordynacja 3D"]
            },
            LODLevel.LOD_350: {
                "min_density": 50,
                "description": "Koordynacja z interfejsami",
                "geometry": "Szczegolowa geometria + polaczenia",
                "accuracy_m": 0.05,
                "use_cases": ["Koordynacja miedzybranzowa", "Clash detection"]
            },
            LODLevel.LOD_400: {
                "min_density": 100,
                "description": "Fabrication",
                "geometry": "Pelna szczegolowość produkcyjna",
                "accuracy_m": 0.02,
                "use_cases": ["Prefabrykacja", "Produkcja"]
            },
            LODLevel.LOD_500: {
                "min_density": 200,
                "description": "As-built",
                "geometry": "Weryfikacja wykonania",
                "accuracy_m": 0.01,
                "use_cases": ["Odbiory", "Dokumentacja powykonawcza", "FM"]
            }
        }

        return requirements.get(target_lod, requirements[LODLevel.LOD_300])
