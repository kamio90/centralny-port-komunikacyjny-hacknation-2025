"""
Building Extraction - Ekstrakcja budynkow z chmury punktow

Algorytmy:
- Region growing dla segmentacji budynkow
- RANSAC dla ekstrakcji plaszczyzn dachu
- Alpha shape dla obrysow budynkow
- Klasyfikacja typow dachow

HackNation 2025 - CPK Chmura+
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
from scipy.spatial import cKDTree, ConvexHull, Delaunay
from scipy.ndimage import label
import logging

logger = logging.getLogger(__name__)


class RoofType(Enum):
    """Typy dachow"""
    FLAT = "flat"  # Plaski
    GABLE = "gable"  # Dwuspadowy
    HIP = "hip"  # Czterospadowy
    SHED = "shed"  # Jednospadowy
    COMPLEX = "complex"  # Zlozony
    UNKNOWN = "unknown"


@dataclass
class BuildingFootprint:
    """Obrys budynku"""
    vertices: np.ndarray  # (N, 2) wierzcholki obrysu
    area_m2: float
    perimeter_m: float
    centroid: np.ndarray  # (2,) srodek
    is_rectangular: bool
    orientation_deg: float  # Kat glownej osi


@dataclass
class Building:
    """Wykryty budynek"""
    id: int
    points: np.ndarray  # (N, 3) punkty budynku
    footprint: BuildingFootprint
    height_min: float  # Wysokosc podstawy [m]
    height_max: float  # Wysokosc dachu [m]
    height: float  # Wysokosc budynku [m]
    roof_type: RoofType
    roof_slope_deg: float  # Nachylenie dachu [deg]
    volume_m3: float  # Objetosc
    floor_count_estimate: int  # Szacunkowa liczba kondygnacji
    confidence: float
    classification: str = "building"  # BIM classification


class BuildingExtractor:
    """
    Ekstraktor budynkow z chmury punktow LiDAR

    Workflow:
    1. Filtruj punkty budynkow (klasa 6)
    2. Segmentuj na pojedyncze budynki (DBSCAN)
    3. Ekstrahuj obrys (alpha shape / convex hull)
    4. Analizuj dach (RANSAC)
    5. Oblicz metryki

    Usage:
        extractor = BuildingExtractor(coords, classification)
        buildings = extractor.extract()
    """

    BUILDING_CLASS = 6
    FLOOR_HEIGHT = 3.0  # Typowa wysokosc kondygnacji [m]

    def __init__(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        ground_height: Optional[float] = None,
        min_building_area: float = 20.0,
        min_building_height: float = 2.5
    ):
        """
        Args:
            coords: wspolrzedne punktow
            classification: klasyfikacja
            ground_height: wysokosc gruntu (opcjonalnie)
            min_building_area: minimalna powierzchnia budynku [m2]
            min_building_height: minimalna wysokosc budynku [m]
        """
        self.coords = coords
        self.classification = classification
        self.min_building_area = min_building_area
        self.min_building_height = min_building_height

        # Wysokosc gruntu
        if ground_height is None:
            ground_mask = classification == 2
            if ground_mask.any():
                self.ground_height = coords[ground_mask, 2].mean()
            else:
                self.ground_height = coords[:, 2].min()
        else:
            self.ground_height = ground_height

        # Punkty budynkow
        self.building_mask = classification == self.BUILDING_CLASS
        self.building_coords = coords[self.building_mask]

        logger.info(f"Building points: {len(self.building_coords):,}")

    def extract(self) -> List[Building]:
        """
        Ekstrahuj wszystkie budynki

        Returns:
            Lista wykrytych budynkow
        """
        if len(self.building_coords) < 100:
            logger.warning("Too few building points")
            return []

        # Segmentuj budynki
        clusters = self._segment_buildings()
        logger.info(f"Found {len(clusters)} building clusters")

        buildings = []
        for i, cluster_coords in enumerate(clusters):
            building = self._analyze_building(i, cluster_coords)
            if building is not None:
                buildings.append(building)

        # Sortuj po powierzchni
        buildings.sort(key=lambda b: b.footprint.area_m2, reverse=True)

        logger.info(f"Extracted {len(buildings)} buildings")
        return buildings

    def _segment_buildings(self, eps: float = 2.0, min_samples: int = 20) -> List[np.ndarray]:
        """Segmentuj punkty na poszczegolne budynki"""
        from sklearn.cluster import DBSCAN

        # Klastruj w XY (budynki sa oddzielone poziomo)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(
            self.building_coords[:, :2]
        )
        labels = clustering.labels_

        clusters = []
        for label_id in set(labels):
            if label_id == -1:
                continue
            mask = labels == label_id
            cluster = self.building_coords[mask]
            if len(cluster) >= min_samples:
                clusters.append(cluster)

        return clusters

    def _analyze_building(self, building_id: int, coords: np.ndarray) -> Optional[Building]:
        """Analizuj pojedynczy budynek"""
        try:
            # Obrys
            footprint = self._extract_footprint(coords)

            if footprint.area_m2 < self.min_building_area:
                return None

            # Wysokosci
            z_min = coords[:, 2].min()
            z_max = coords[:, 2].max()
            height = z_max - self.ground_height

            if height < self.min_building_height:
                return None

            # Typ i nachylenie dachu
            roof_type, roof_slope = self._analyze_roof(coords)

            # Objetosc (uproszczona)
            volume = footprint.area_m2 * height

            # Liczba kondygnacji
            floor_count = max(1, int(height / self.FLOOR_HEIGHT + 0.5))

            # Confidence
            confidence = min(1.0, len(coords) / 500)

            return Building(
                id=building_id,
                points=coords,
                footprint=footprint,
                height_min=z_min,
                height_max=z_max,
                height=height,
                roof_type=roof_type,
                roof_slope_deg=roof_slope,
                volume_m3=volume,
                floor_count_estimate=floor_count,
                confidence=confidence
            )

        except Exception as e:
            logger.warning(f"Building analysis failed: {e}")
            return None

    def _extract_footprint(self, coords: np.ndarray) -> BuildingFootprint:
        """Ekstrahuj obrys budynku"""
        points_2d = coords[:, :2]

        # Convex hull jako podstawa
        try:
            hull = ConvexHull(points_2d)
            vertices = points_2d[hull.vertices]
        except Exception:
            # Fallback - bounding box
            x_min, y_min = points_2d.min(axis=0)
            x_max, y_max = points_2d.max(axis=0)
            vertices = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ])

        # Powierzchnia
        area = self._polygon_area(vertices)

        # Obwod
        perimeter = np.sum(np.linalg.norm(np.diff(
            np.vstack([vertices, vertices[0]]), axis=0
        ), axis=1))

        # Centroid
        centroid = vertices.mean(axis=0)

        # Sprawdz czy prostokatny
        is_rectangular = self._is_rectangular(vertices)

        # Orientacja (glowna os)
        orientation = self._get_orientation(vertices)

        return BuildingFootprint(
            vertices=vertices,
            area_m2=area,
            perimeter_m=perimeter,
            centroid=centroid,
            is_rectangular=is_rectangular,
            orientation_deg=orientation
        )

    def _polygon_area(self, vertices: np.ndarray) -> float:
        """Oblicz powierzchnie wielokata (Shoelace formula)"""
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i, 0] * vertices[j, 1]
            area -= vertices[j, 0] * vertices[i, 1]
        return abs(area) / 2.0

    def _is_rectangular(self, vertices: np.ndarray, tolerance: float = 15.0) -> bool:
        """Sprawdz czy obrys jest prostokatny"""
        if len(vertices) != 4:
            return False

        # Katy miedzy krawedzami
        angles = []
        n = len(vertices)
        for i in range(n):
            v1 = vertices[(i + 1) % n] - vertices[i]
            v2 = vertices[(i + 2) % n] - vertices[(i + 1) % n]

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            angles.append(angle)

        # Sprawdz czy katy sa bliskie 90 stopni
        return all(abs(a - 90) < tolerance for a in angles)

    def _get_orientation(self, vertices: np.ndarray) -> float:
        """Znajdz orientacje budynku (kat glownej osi)"""
        # Uzyj PCA
        centered = vertices - vertices.mean(axis=0)
        cov = np.cov(centered.T)

        if cov.shape == (2, 2):
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            main_axis = eigenvectors[:, -1]
            angle = np.degrees(np.arctan2(main_axis[1], main_axis[0]))
            return angle % 180
        return 0.0

    def _analyze_roof(self, coords: np.ndarray) -> Tuple[RoofType, float]:
        """Analizuj typ i nachylenie dachu"""
        # Punkty dachu (gorna 30% wysokosci)
        z_range = coords[:, 2].max() - coords[:, 2].min()
        z_threshold = coords[:, 2].max() - 0.3 * z_range
        roof_mask = coords[:, 2] >= z_threshold
        roof_points = coords[roof_mask]

        if len(roof_points) < 20:
            return RoofType.UNKNOWN, 0.0

        # Dopasuj plaszczyzne RANSAC
        try:
            normal, inliers_ratio = self._fit_plane_ransac(roof_points)

            # Nachylenie wzgledem pionu
            vertical = np.array([0, 0, 1])
            cos_angle = abs(np.dot(normal, vertical))
            slope_rad = np.arccos(np.clip(cos_angle, 0, 1))
            slope_deg = np.degrees(slope_rad)

            # Klasyfikuj typ dachu
            if slope_deg < 5:
                roof_type = RoofType.FLAT
            elif slope_deg < 25:
                # Sprawdz symetrie
                if self._is_symmetric_roof(roof_points):
                    roof_type = RoofType.GABLE
                else:
                    roof_type = RoofType.SHED
            elif slope_deg < 45:
                roof_type = RoofType.HIP
            else:
                roof_type = RoofType.COMPLEX

            return roof_type, slope_deg

        except Exception:
            return RoofType.UNKNOWN, 0.0

    def _fit_plane_ransac(
        self,
        points: np.ndarray,
        n_iterations: int = 100,
        threshold: float = 0.1
    ) -> Tuple[np.ndarray, float]:
        """Dopasuj plaszczyzne RANSAC"""
        best_normal = np.array([0, 0, 1])
        best_inliers = 0

        for _ in range(n_iterations):
            # Losuj 3 punkty
            idx = np.random.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[idx]

            # Oblicz normalna
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-8:
                continue
            normal = normal / norm

            # Policz inliers
            distances = np.abs(np.dot(points - p1, normal))
            inliers = (distances < threshold).sum()

            if inliers > best_inliers:
                best_inliers = inliers
                best_normal = normal

        inliers_ratio = best_inliers / len(points)
        return best_normal, inliers_ratio

    def _is_symmetric_roof(self, roof_points: np.ndarray) -> bool:
        """Sprawdz czy dach jest symetryczny (dwuspadowy)"""
        # Uproszczona heurystyka: sprawdz rozklad wysokosci
        center_x = roof_points[:, 0].mean()
        left_mask = roof_points[:, 0] < center_x
        right_mask = ~left_mask

        if left_mask.sum() < 10 or right_mask.sum() < 10:
            return False

        left_z_mean = roof_points[left_mask, 2].mean()
        right_z_mean = roof_points[right_mask, 2].mean()

        # Symetryczny jesli srednie wysokosci podobne
        return abs(left_z_mean - right_z_mean) < 0.5

    def get_statistics(self, buildings: List[Building]) -> Dict:
        """Statystyki wykrytych budynkow"""
        if not buildings:
            return {'total_count': 0}

        areas = [b.footprint.area_m2 for b in buildings]
        heights = [b.height for b in buildings]
        volumes = [b.volume_m3 for b in buildings]

        roof_types = {}
        for b in buildings:
            t = b.roof_type.value
            roof_types[t] = roof_types.get(t, 0) + 1

        return {
            'total_count': len(buildings),
            'total_area_m2': sum(areas),
            'total_volume_m3': sum(volumes),
            'avg_area_m2': np.mean(areas),
            'avg_height_m': np.mean(heights),
            'max_height_m': max(heights),
            'min_height_m': min(heights),
            'roof_types': roof_types,
            'rectangular_count': sum(1 for b in buildings if b.footprint.is_rectangular),
            'total_floors_estimate': sum(b.floor_count_estimate for b in buildings)
        }


def extract_buildings(
    coords: np.ndarray,
    classification: np.ndarray
) -> List[Building]:
    """Convenience function dla ekstrakcji budynkow"""
    extractor = BuildingExtractor(coords, classification)
    return extractor.extract()
