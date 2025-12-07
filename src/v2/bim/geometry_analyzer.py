"""
Geometry Analyzer - Analiza geometrii 3D chmur punktow

Funkcjonalnosci:
- Bounding box (AABB, OBB)
- Convex hull 3D
- Metryki geometryczne
- Analiza glownych kierunkow (PCA)
- Obliczenia objetosci i powierzchni

HackNation 2025 - CPK Chmura+
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from scipy.spatial import ConvexHull
import logging

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box (AABB - Axis Aligned)"""
    min_point: np.ndarray  # (3,)
    max_point: np.ndarray  # (3,)
    center: np.ndarray  # (3,)
    dimensions: np.ndarray  # (3,) szerokosc, glebokosc, wysokosc
    volume: float
    surface_area: float


@dataclass
class OrientedBoundingBox:
    """Oriented Bounding Box (OBB)"""
    center: np.ndarray  # (3,)
    axes: np.ndarray  # (3, 3) glowne osie
    half_extents: np.ndarray  # (3,) polowy wymiarow
    rotation_matrix: np.ndarray  # (3, 3)
    volume: float
    orientation_deg: float  # Kat obrotu wzgledem osi X


@dataclass
class ConvexHull3D:
    """Convex hull w 3D"""
    vertices: np.ndarray  # (N, 3) wierzcholki
    faces: np.ndarray  # (M, 3) trojkaty (indeksy)
    volume: float
    surface_area: float
    centroid: np.ndarray


@dataclass
class GeometryMetrics:
    """Metryki geometryczne obiektu"""
    point_count: int
    aabb: BoundingBox
    obb: Optional[OrientedBoundingBox]
    convex_hull: Optional[ConvexHull3D]
    principal_directions: np.ndarray  # (3, 3) glowne kierunki
    eigenvalues: np.ndarray  # (3,) wartosci wlasne
    compactness: float  # 0-1, jak "zwarty" jest obiekt
    elongation: float  # Stosunek najdluzszego do najkrotszego wymiaru
    planarity: float  # Jak plaski jest obiekt
    sphericity: float  # Jak kulisty jest obiekt
    density_uniformity: float  # Jednorodnosc gestosci


class GeometryAnalyzer:
    """
    Analizator geometrii 3D

    Oblicza rozne metryki geometryczne dla chmury punktow lub jej fragmentu.

    Usage:
        analyzer = GeometryAnalyzer(coords)
        metrics = analyzer.analyze()
        obb = analyzer.compute_obb()
    """

    def __init__(self, coords: np.ndarray):
        """
        Args:
            coords: (N, 3) wspolrzedne punktow
        """
        if len(coords) < 4:
            raise ValueError("Need at least 4 points for 3D analysis")

        self.coords = coords
        self.n_points = len(coords)
        self.centroid = coords.mean(axis=0)

        logger.debug(f"GeometryAnalyzer initialized with {self.n_points} points")

    def analyze(self) -> GeometryMetrics:
        """
        Wykonaj pelna analize geometrii

        Returns:
            GeometryMetrics z wszystkimi metrykami
        """
        # AABB
        aabb = self.compute_aabb()

        # OBB (jesli wystarczajaco punktow)
        obb = self.compute_obb() if self.n_points >= 10 else None

        # Convex Hull (jesli wystarczajaco punktow)
        convex_hull = self.compute_convex_hull() if self.n_points >= 10 else None

        # PCA
        principal_dirs, eigenvalues = self.compute_pca()

        # Metryki ksztaltu
        compactness = self._compute_compactness(aabb, convex_hull)
        elongation = self._compute_elongation(eigenvalues)
        planarity = self._compute_planarity(eigenvalues)
        sphericity = self._compute_sphericity(eigenvalues)

        # Jednorodnosc gestosci
        density_uniformity = self._compute_density_uniformity()

        return GeometryMetrics(
            point_count=self.n_points,
            aabb=aabb,
            obb=obb,
            convex_hull=convex_hull,
            principal_directions=principal_dirs,
            eigenvalues=eigenvalues,
            compactness=compactness,
            elongation=elongation,
            planarity=planarity,
            sphericity=sphericity,
            density_uniformity=density_uniformity
        )

    def compute_aabb(self) -> BoundingBox:
        """Oblicz Axis-Aligned Bounding Box"""
        min_point = self.coords.min(axis=0)
        max_point = self.coords.max(axis=0)
        center = (min_point + max_point) / 2
        dimensions = max_point - min_point

        volume = np.prod(dimensions)
        surface_area = 2 * (
            dimensions[0] * dimensions[1] +
            dimensions[1] * dimensions[2] +
            dimensions[2] * dimensions[0]
        )

        return BoundingBox(
            min_point=min_point,
            max_point=max_point,
            center=center,
            dimensions=dimensions,
            volume=volume,
            surface_area=surface_area
        )

    def compute_obb(self) -> OrientedBoundingBox:
        """Oblicz Oriented Bounding Box (minimalny obrotowy BB)"""
        # Uzyj PCA do znalezienia glownych osi
        centered = self.coords - self.centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sortuj malejaco
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Przeksztalc punkty do ukladu glownych osi
        rotated = centered @ eigenvectors

        # Wymiary w nowym ukladzie
        min_rot = rotated.min(axis=0)
        max_rot = rotated.max(axis=0)
        half_extents = (max_rot - min_rot) / 2

        # Srodek w oryginalnym ukladzie
        center_rot = (min_rot + max_rot) / 2
        center = self.centroid + eigenvectors @ center_rot

        volume = np.prod(2 * half_extents)

        # Kat obrotu (wzgledem osi X)
        main_axis = eigenvectors[:, 0]
        orientation_deg = np.degrees(np.arctan2(main_axis[1], main_axis[0]))

        return OrientedBoundingBox(
            center=center,
            axes=eigenvectors,
            half_extents=half_extents,
            rotation_matrix=eigenvectors,
            volume=volume,
            orientation_deg=orientation_deg
        )

    def compute_convex_hull(self) -> ConvexHull3D:
        """Oblicz Convex Hull 3D"""
        try:
            hull = ConvexHull(self.coords)

            vertices = self.coords[hull.vertices]
            faces = hull.simplices
            volume = hull.volume
            surface_area = hull.area

            # Centroid
            centroid = vertices.mean(axis=0)

            return ConvexHull3D(
                vertices=vertices,
                faces=faces,
                volume=volume,
                surface_area=surface_area,
                centroid=centroid
            )

        except Exception as e:
            logger.warning(f"Convex hull computation failed: {e}")
            # Fallback
            return ConvexHull3D(
                vertices=self.coords,
                faces=np.array([]),
                volume=0,
                surface_area=0,
                centroid=self.centroid
            )

    def compute_pca(self) -> Tuple[np.ndarray, np.ndarray]:
        """Oblicz glowne kierunki (PCA)"""
        centered = self.coords - self.centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sortuj malejaco
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvectors, eigenvalues

    def _compute_compactness(
        self,
        aabb: BoundingBox,
        hull: Optional[ConvexHull3D]
    ) -> float:
        """Oblicz zwartość (compactness)"""
        if hull is None or hull.volume == 0:
            return 0.5

        # Stosunek objetosci convex hull do AABB
        compactness = hull.volume / aabb.volume if aabb.volume > 0 else 0
        return min(1.0, compactness)

    def _compute_elongation(self, eigenvalues: np.ndarray) -> float:
        """Oblicz wydluzenie"""
        if eigenvalues[-1] == 0:
            return float('inf')
        return np.sqrt(eigenvalues[0] / eigenvalues[-1])

    def _compute_planarity(self, eigenvalues: np.ndarray) -> float:
        """Oblicz planarność (jak plaski jest obiekt)"""
        if eigenvalues[1] == 0:
            return 0
        # Plaski obiekt: lambda_2 >> lambda_3
        return (eigenvalues[1] - eigenvalues[2]) / eigenvalues[1]

    def _compute_sphericity(self, eigenvalues: np.ndarray) -> float:
        """Oblicz sferyczność"""
        if eigenvalues[0] == 0:
            return 0
        # Kulisty obiekt: wszystkie lambda podobne
        return eigenvalues[2] / eigenvalues[0]

    def _compute_density_uniformity(self) -> float:
        """Oblicz jednorodność gęstości"""
        # Podziel na oktanty i sprawdz rozklad
        centered = self.coords - self.centroid

        octant_counts = np.zeros(8)
        for i, point in enumerate(centered):
            octant = (
                (1 if point[0] >= 0 else 0) +
                (2 if point[1] >= 0 else 0) +
                (4 if point[2] >= 0 else 0)
            )
            octant_counts[octant] += 1

        # Jednorodnosc - odchylenie od sredniej
        if octant_counts.sum() == 0:
            return 0

        expected = octant_counts.sum() / 8
        deviation = np.std(octant_counts) / expected if expected > 0 else 0

        # Im mniejsze odchylenie, tym lepsza jednorodność
        uniformity = max(0, 1 - deviation)
        return uniformity

    def compute_cross_section(
        self,
        plane_point: np.ndarray,
        plane_normal: np.ndarray,
        tolerance: float = 0.1
    ) -> np.ndarray:
        """
        Oblicz przekroj plaszczyzna

        Args:
            plane_point: punkt na plaszczyznie
            plane_normal: normalna plaszczyzny (znormalizowana)
            tolerance: tolerancja odleglosci [m]

        Returns:
            Punkty blisko plaszczyzny
        """
        # Odleglosc od plaszczyzny
        distances = np.abs(np.dot(self.coords - plane_point, plane_normal))

        # Filtruj punkty blisko plaszczyzny
        mask = distances < tolerance
        section_points = self.coords[mask]

        return section_points

    def compute_horizontal_slice(
        self,
        z_level: float,
        tolerance: float = 0.1
    ) -> np.ndarray:
        """Oblicz przekroj poziomy na danej wysokosci"""
        return self.compute_cross_section(
            plane_point=np.array([0, 0, z_level]),
            plane_normal=np.array([0, 0, 1]),
            tolerance=tolerance
        )

    def compute_projection(self, axis: str = 'z') -> np.ndarray:
        """
        Oblicz projekcje na plaszczyzne

        Args:
            axis: os prostopadła ('x', 'y', 'z')

        Returns:
            (N, 2) punkty rzutowane
        """
        if axis == 'x':
            return self.coords[:, [1, 2]]
        elif axis == 'y':
            return self.coords[:, [0, 2]]
        else:  # z
            return self.coords[:, [0, 1]]

    def compute_footprint_area(self) -> float:
        """Oblicz powierzchnie rzutu na XY"""
        projection = self.compute_projection('z')

        if len(projection) < 3:
            return 0

        try:
            hull = ConvexHull(projection)
            return hull.volume  # W 2D volume = area
        except Exception:
            # Fallback - bounding box
            ranges = projection.max(axis=0) - projection.min(axis=0)
            return ranges[0] * ranges[1]

    def compute_surface_roughness(self, k_neighbors: int = 10) -> float:
        """
        Oblicz chropowatosc powierzchni

        Uzywa lokalnej wariancji normalnych jako miary chropowatosci

        Args:
            k_neighbors: liczba sasiadow do analizy

        Returns:
            Srednia chropowatosc (0 = gladka, wyzej = chropowata)
        """
        from scipy.spatial import cKDTree

        if self.n_points < k_neighbors + 1:
            return 0

        tree = cKDTree(self.coords)

        roughness_values = []

        # Probkuj punkty (dla wydajnosci)
        sample_size = min(1000, self.n_points)
        sample_idx = np.random.choice(self.n_points, sample_size, replace=False)

        for idx in sample_idx:
            # Znajdz sasiadow
            _, neighbor_idx = tree.query(self.coords[idx], k=k_neighbors + 1)
            neighbors = self.coords[neighbor_idx[1:]]  # Pomijamy sam punkt

            # Oblicz lokalna normalna przez PCA
            centered = neighbors - neighbors.mean(axis=0)
            try:
                _, _, vh = np.linalg.svd(centered)
                normal = vh[-1]  # Najmniejszy wektor wlasny

                # Oblicz wariancje odleglosci od plaszczyzny
                distances = np.abs(np.dot(centered, normal))
                roughness_values.append(np.std(distances))
            except Exception:
                pass

        if roughness_values:
            return np.mean(roughness_values)
        return 0

    @staticmethod
    def merge_bounding_boxes(boxes: List[BoundingBox]) -> BoundingBox:
        """Polacz wiele bounding boxow"""
        if not boxes:
            return BoundingBox(
                min_point=np.zeros(3),
                max_point=np.zeros(3),
                center=np.zeros(3),
                dimensions=np.zeros(3),
                volume=0,
                surface_area=0
            )

        all_min = np.min([b.min_point for b in boxes], axis=0)
        all_max = np.max([b.max_point for b in boxes], axis=0)
        center = (all_min + all_max) / 2
        dimensions = all_max - all_min

        volume = np.prod(dimensions)
        surface_area = 2 * (
            dimensions[0] * dimensions[1] +
            dimensions[1] * dimensions[2] +
            dimensions[2] * dimensions[0]
        )

        return BoundingBox(
            min_point=all_min,
            max_point=all_max,
            center=center,
            dimensions=dimensions,
            volume=volume,
            surface_area=surface_area
        )


def analyze_geometry(coords: np.ndarray) -> GeometryMetrics:
    """Convenience function dla analizy geometrii"""
    analyzer = GeometryAnalyzer(coords)
    return analyzer.analyze()
