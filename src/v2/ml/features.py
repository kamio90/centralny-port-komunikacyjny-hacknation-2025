"""
Feature Extraction - Ekstrakcja cech geometrycznych z chmury punktow

Cechy per punkt:
- Geometryczne: wysokosc, nachylenie, krzywizna
- Lokalne: planarnosc, linearnosc, sferycznosc
- Kontekstowe: gestosc, wariancja wysokosci
- Kolorowe: RGB, intensywnosc

Te cechy sa uzywane przez klasyfikatory ML.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from scipy.spatial import cKDTree
from scipy.linalg import eigh
import logging

logger = logging.getLogger(__name__)


@dataclass
class GeometricFeatures:
    """Kontener na cechy geometryczne"""
    # Podstawowe
    height_above_ground: np.ndarray  # Wysokosc nad gruntem
    z_normalized: np.ndarray  # Znormalizowana wysokosc

    # Lokalne (z PCA na sasiedztwie)
    linearity: np.ndarray  # Liniowosc (krawedzie, linie)
    planarity: np.ndarray  # Planarnosc (plaskie powierzchnie)
    sphericity: np.ndarray  # Sferycznosc (punkty izolowane)
    verticality: np.ndarray  # Wertykalnosc (sciany, slupy)

    # Normalne
    normal_z: np.ndarray  # Skladowa Z normalnej (0=pionowa, 1=pozioma)

    # Gestosc
    density: np.ndarray  # Lokalna gestosc punktow
    height_variance: np.ndarray  # Wariancja wysokosci w sasiedztwie

    # Opcjonalne
    intensity: Optional[np.ndarray] = None
    red: Optional[np.ndarray] = None
    green: Optional[np.ndarray] = None
    blue: Optional[np.ndarray] = None

    def to_array(self) -> np.ndarray:
        """Konwertuje cechy do macierzy (N, F)"""
        features = [
            self.height_above_ground,
            self.z_normalized,
            self.linearity,
            self.planarity,
            self.sphericity,
            self.verticality,
            self.normal_z,
            self.density,
            self.height_variance
        ]

        if self.intensity is not None:
            features.append(self.intensity)
        if self.red is not None:
            features.extend([self.red, self.green, self.blue])

        return np.column_stack(features)

    @property
    def feature_names(self) -> List[str]:
        """Nazwy cech"""
        names = [
            'height_above_ground',
            'z_normalized',
            'linearity',
            'planarity',
            'sphericity',
            'verticality',
            'normal_z',
            'density',
            'height_variance'
        ]
        if self.intensity is not None:
            names.append('intensity')
        if self.red is not None:
            names.extend(['red', 'green', 'blue'])
        return names


class FeatureExtractor:
    """
    Ekstraktor cech geometrycznych z chmury punktow

    Oblicza cechy dla kazdego punktu na podstawie jego sasiedztwa.

    Usage:
        extractor = FeatureExtractor(coords, k_neighbors=30)
        features = extractor.extract_all()
        X = features.to_array()
    """

    def __init__(
        self,
        coords: np.ndarray,
        colors: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
        ground_class: Optional[np.ndarray] = None,
        k_neighbors: int = 30,
        radius: Optional[float] = None
    ):
        """
        Args:
            coords: (N, 3) wspolrzedne XYZ
            colors: (N, 3) kolory RGB [0-1]
            intensity: (N,) intensywnosc
            ground_class: (N,) maska gruntu (True dla punktow gruntu)
            k_neighbors: liczba sasiadow do analizy
            radius: promien sasiedztwa (alternatywa dla k)
        """
        self.coords = coords
        self.colors = colors
        self.intensity = intensity
        self.ground_class = ground_class
        self.k_neighbors = k_neighbors
        self.radius = radius
        self.n_points = len(coords)

        # Buduj KD-Tree
        logger.info(f"Building KD-Tree for {self.n_points:,} points...")
        self.tree = cKDTree(coords)

        # Cache dla sasiedztwa
        self._neighbors_cache = None

    def _get_neighbors(self) -> np.ndarray:
        """Znajduje sasiadow dla wszystkich punktow"""
        if self._neighbors_cache is not None:
            return self._neighbors_cache

        logger.info(f"Finding {self.k_neighbors} neighbors per point...")

        if self.radius:
            # Radius-based
            neighbors = self.tree.query_ball_tree(self.tree, self.radius)
            # Pad to fixed size
            max_len = max(len(n) for n in neighbors)
            padded = np.zeros((self.n_points, min(max_len, self.k_neighbors)), dtype=np.int32)
            for i, n in enumerate(neighbors):
                n = n[:self.k_neighbors]
                padded[i, :len(n)] = n
            self._neighbors_cache = padded
        else:
            # K-nearest
            _, indices = self.tree.query(self.coords, k=self.k_neighbors + 1)
            self._neighbors_cache = indices[:, 1:]  # Exclude self

        return self._neighbors_cache

    def extract_all(
        self,
        compute_colors: bool = True,
        progress_callback=None
    ) -> GeometricFeatures:
        """
        Ekstrahuje wszystkie cechy

        Args:
            compute_colors: czy wlaczac cechy kolorowe
            progress_callback: callback(step, pct, msg)

        Returns:
            GeometricFeatures z wszystkimi cechami
        """
        logger.info("Extracting geometric features...")

        if progress_callback:
            progress_callback("Features", 0, "Inicjalizacja...")

        # 1. Wysokosc nad gruntem
        if progress_callback:
            progress_callback("Features", 10, "Obliczanie wysokosci...")
        hag = self._compute_height_above_ground()
        z_norm = (self.coords[:, 2] - self.coords[:, 2].min()) / \
                 (self.coords[:, 2].max() - self.coords[:, 2].min() + 1e-6)

        # 2. Cechy lokalne (PCA)
        if progress_callback:
            progress_callback("Features", 30, "Analiza PCA sasiedztwa...")
        linearity, planarity, sphericity, verticality, normal_z = \
            self._compute_local_pca_features()

        # 3. Gestosc i wariancja
        if progress_callback:
            progress_callback("Features", 70, "Obliczanie gestosci...")
        density = self._compute_density()
        height_var = self._compute_height_variance()

        if progress_callback:
            progress_callback("Features", 90, "Finalizacja...")

        # Buduj obiekt
        features = GeometricFeatures(
            height_above_ground=hag,
            z_normalized=z_norm,
            linearity=linearity,
            planarity=planarity,
            sphericity=sphericity,
            verticality=verticality,
            normal_z=normal_z,
            density=density,
            height_variance=height_var
        )

        # Dodaj kolory i intensywnosc
        if compute_colors and self.intensity is not None:
            features.intensity = self.intensity

        if compute_colors and self.colors is not None:
            features.red = self.colors[:, 0]
            features.green = self.colors[:, 1]
            features.blue = self.colors[:, 2]

        if progress_callback:
            progress_callback("Features", 100, "Gotowe!")

        logger.info(f"Extracted {len(features.feature_names)} features")
        return features

    def _compute_height_above_ground(self) -> np.ndarray:
        """Oblicza wysokosc nad gruntem"""
        if self.ground_class is not None and self.ground_class.any():
            # Mamy punkty gruntu
            ground_points = self.coords[self.ground_class]
            ground_tree = cKDTree(ground_points[:, :2])

            # Dla kazdego punktu znajdz najblizszy punkt gruntu
            _, indices = ground_tree.query(self.coords[:, :2], k=1)
            ground_z = ground_points[indices, 2]
            hag = self.coords[:, 2] - ground_z
        else:
            # Brak informacji o gruncie - uzyj percentyla
            z_ground = np.percentile(self.coords[:, 2], 5)
            hag = self.coords[:, 2] - z_ground

        return np.maximum(hag, 0)

    def _compute_local_pca_features(self) -> Tuple[np.ndarray, ...]:
        """
        Oblicza cechy z PCA na sasiedztwie

        Returns:
            (linearity, planarity, sphericity, verticality, normal_z)
        """
        neighbors = self._get_neighbors()

        linearity = np.zeros(self.n_points)
        planarity = np.zeros(self.n_points)
        sphericity = np.zeros(self.n_points)
        verticality = np.zeros(self.n_points)
        normal_z = np.zeros(self.n_points)

        # Batch processing dla wydajnosci
        batch_size = 10000
        n_batches = (self.n_points + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, self.n_points)

            for i in range(start, end):
                neighbor_idx = neighbors[i]
                neighbor_coords = self.coords[neighbor_idx]

                if len(neighbor_coords) < 3:
                    continue

                # PCA
                centered = neighbor_coords - neighbor_coords.mean(axis=0)
                cov = np.cov(centered.T)

                try:
                    eigenvalues, eigenvectors = eigh(cov)
                    # Sortuj malejaco
                    idx = np.argsort(eigenvalues)[::-1]
                    eigenvalues = eigenvalues[idx]
                    eigenvectors = eigenvectors[:, idx]

                    # Normalizuj
                    eigenvalues = np.maximum(eigenvalues, 1e-10)
                    total = eigenvalues.sum()

                    l1, l2, l3 = eigenvalues / total

                    # Cechy
                    linearity[i] = (l1 - l2) / l1
                    planarity[i] = (l2 - l3) / l1
                    sphericity[i] = l3 / l1

                    # Normalna (najslabszy kierunek)
                    normal = eigenvectors[:, 2]
                    normal_z[i] = abs(normal[2])
                    verticality[i] = 1.0 - normal_z[i]

                except Exception:
                    pass

        return linearity, planarity, sphericity, verticality, normal_z

    def _compute_density(self) -> np.ndarray:
        """Oblicza lokalna gestosc punktow"""
        # Gestosc = liczba sasiadow w stalym promieniu
        radius = 1.0  # 1m
        counts = self.tree.query_ball_point(self.coords, radius, return_length=True)
        # Normalizuj do [0, 1]
        density = counts / counts.max()
        return density

    def _compute_height_variance(self) -> np.ndarray:
        """Oblicza wariancje wysokosci w sasiedztwie"""
        neighbors = self._get_neighbors()
        height_var = np.zeros(self.n_points)

        for i in range(self.n_points):
            neighbor_z = self.coords[neighbors[i], 2]
            height_var[i] = neighbor_z.var() if len(neighbor_z) > 1 else 0

        # Normalizuj
        height_var = height_var / (height_var.max() + 1e-6)
        return height_var


def extract_point_features(
    coords: np.ndarray,
    colors: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None,
    ground_mask: Optional[np.ndarray] = None,
    k_neighbors: int = 30
) -> np.ndarray:
    """
    Convenience function do ekstrakcji cech

    Args:
        coords: wspolrzedne
        colors: kolory RGB
        intensity: intensywnosc
        ground_mask: maska gruntu
        k_neighbors: liczba sasiadow

    Returns:
        (N, F) macierz cech
    """
    extractor = FeatureExtractor(
        coords, colors, intensity, ground_mask, k_neighbors
    )
    features = extractor.extract_all()
    return features.to_array()
