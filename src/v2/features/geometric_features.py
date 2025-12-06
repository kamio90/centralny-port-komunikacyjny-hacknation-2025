"""
Moduł do ekstrakcji cech geometrycznych z chmur punktów

Wykorzystuje PCA (Principal Component Analysis) do obliczania:
- Planarity (płaskość)
- Linearity (liniowość)
- Sphericity (kulistość)
- Normals (wektory normalne)
- Verticality/Horizontality
- Roughness (chropowatość)

OPTYMALIZACJA: Sampling PCA (0.5-1%) dla przyspieszenia
"""

import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class GeometricFeatureExtractor:
    """
    Ekstrakcja cech geometrycznych z chmur punktów

    OPTYMALIZACJA:
    - PCA tylko dla próbki punktów (0.5-1%)
    - NumPy zamiast sklearn (szybsze)
    - Vectorized operations gdzie możliwe
    """

    def __init__(self,
                 coords: np.ndarray,
                 sample_rate: float = 0.005,
                 search_radius: float = 1.5):
        """
        Args:
            coords: (N, 3) Współrzędne XYZ
            sample_rate: Procent punktów do pełnego PCA (0.005 = 0.5%)
            search_radius: Promień wyszukiwania sąsiadów w metrach
        """
        self.coords = coords
        self.n_points = len(coords)
        self.sample_rate = sample_rate
        self.search_radius = search_radius

        logger.info(f"GeometricFeatureExtractor: {self.n_points:,} punktów, "
                   f"sampling={sample_rate*100:.1f}%, radius={search_radius}m")

        # Buduj KD-tree (LOCAL - thread-safe!)
        self.kdtree = cKDTree(coords)

    def extract(self) -> Dict[str, np.ndarray]:
        """
        Ekstraktuje pełny zestaw cech geometrycznych

        Returns:
            Dict z cechami:
                - planarity: (N,) Płaskość [0-1]
                - linearity: (N,) Liniowość [0-1]
                - sphericity: (N,) Kulistość [0-1]
                - normals: (N, 3) Wektory normalne
                - verticality: (N,) Wertykalność [0-1]
                - horizontality: (N,) Horyzontalność [0-1]
                - roughness: (N,) Chropowatość
                - height_variance: (N,) Wariancja wysokości
                - density: (N,) Gęstość punktów
        """
        logger.info("Rozpoczynam ekstrakcję cech geometrycznych...")

        # Inicjalizuj tablice cech
        features = {
            'planarity': np.zeros(self.n_points),
            'linearity': np.zeros(self.n_points),
            'sphericity': np.zeros(self.n_points),
            'normals': np.zeros((self.n_points, 3)),
            'verticality': np.zeros(self.n_points),
            'horizontality': np.zeros(self.n_points),
            'roughness': np.zeros(self.n_points),
            'height_variance': np.zeros(self.n_points),
            'density': np.zeros(self.n_points)
        }

        # Wybierz próbkę punktów do pełnego PCA
        sample_size = min(5000, max(500, int(self.n_points * self.sample_rate)))
        sample_indices = np.random.choice(self.n_points, sample_size, replace=False)

        logger.info(f"Próbka PCA: {sample_size:,} punktów ({sample_size/self.n_points*100:.2f}%)")

        # Przetwórz próbkę (pełne PCA)
        for i, idx in enumerate(sample_indices):
            if i % 1000 == 0 and i > 0:
                logger.info(f"  Przetworzono {i:,}/{sample_size:,} punktów próbki...")

            # Znajdź sąsiadów
            neighbors_idx = self.kdtree.query_ball_point(self.coords[idx], self.search_radius)

            if len(neighbors_idx) < 10:
                continue

            neighbors = self.coords[neighbors_idx]

            # Oblicz cechy PCA
            pca_features = self._compute_pca_features(neighbors)

            # Przypisz do tablicy wynikowej
            features['planarity'][idx] = pca_features['planarity']
            features['linearity'][idx] = pca_features['linearity']
            features['sphericity'][idx] = pca_features['sphericity']
            features['normals'][idx] = pca_features['normal']
            features['verticality'][idx] = pca_features['verticality']
            features['horizontality'][idx] = pca_features['horizontality']
            features['roughness'][idx] = pca_features['roughness']
            features['height_variance'][idx] = pca_features['height_variance']
            features['density'][idx] = pca_features['density']

        # Dla pozostałych punktów: użyj globalnych przybliżeń
        non_sampled = np.setdiff1d(np.arange(self.n_points), sample_indices)

        if len(non_sampled) > 0:
            logger.info(f"Przybliżone cechy dla {len(non_sampled):,} pozostałych punktów...")

            # Globalna statystyka wysokości
            z_std = self.coords[:, 2].std()

            if z_std > 5.0:  # Prawdopodobnie struktury wertykalne
                features['verticality'][non_sampled] = 0.5
                features['linearity'][non_sampled] = 0.3
            else:  # Prawdopodobnie płaskie
                features['horizontality'][non_sampled] = 0.5
                features['planarity'][non_sampled] = 0.3

        logger.info("Ekstrakcja cech zakończona!")
        return features

    def _compute_pca_features(self, neighbors: np.ndarray) -> Dict:
        """
        Oblicza cechy PCA dla grupy punktów

        Args:
            neighbors: (M, 3) Punkty sąsiadujące

        Returns:
            Dict z cechami geometrycznymi
        """
        # Wyśrodkuj punkty
        centroid = neighbors.mean(axis=0)
        centered = neighbors - centroid

        # Macierz kowariancji
        cov_matrix = (centered.T @ centered) / len(neighbors)

        # Wartości i wektory własne (NumPy - szybsze niż sklearn!)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sortuj malejąco
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx].T

        λ1, λ2, λ3 = eigenvalues[0], eigenvalues[1], eigenvalues[2]

        # Cechy geometryczne (Weinmann et al.)
        epsilon = 1e-8
        planarity = (λ2 - λ3) / (λ1 + epsilon)
        linearity = (λ1 - λ2) / (λ1 + epsilon)
        sphericity = λ3 / (λ1 + epsilon)

        # Wektor normalny (3. składowa PCA)
        normal = eigenvectors[2]

        # Wertykalność i horyzontalność
        verticality = 1.0 - abs(normal[2])
        horizontality = abs(normal[2])

        # Chropowatość (odległość od płaszczyzny)
        distances = np.abs(centered @ normal)
        roughness = distances.std()

        # Wariancja wysokości
        height_variance = neighbors[:, 2].std()

        # Gęstość
        volume = (4/3) * np.pi * (self.search_radius ** 3)
        density = len(neighbors) / volume

        return {
            'planarity': planarity,
            'linearity': linearity,
            'sphericity': sphericity,
            'normal': normal,
            'verticality': verticality,
            'horizontality': horizontality,
            'roughness': roughness,
            'height_variance': height_variance,
            'density': density
        }

    @staticmethod
    def compute_ndvi(colors: np.ndarray) -> np.ndarray:
        """
        Oblicza NDVI (Normalized Difference Vegetation Index) z kolorów RGB

        Args:
            colors: (N, 3) Kolory RGB [0-1]

        Returns:
            (N,) NDVI [-1, 1]
        """
        if colors is None:
            return None

        # NDVI = (Green - Red) / (Green + Red)
        ndvi = (colors[:, 1] - colors[:, 0]) / (colors[:, 1] + colors[:, 0] + 1e-8)
        return ndvi

    @staticmethod
    def compute_brightness(colors: np.ndarray) -> np.ndarray:
        """
        Oblicza jasność (brightness) z kolorów RGB

        Args:
            colors: (N, 3) Kolory RGB [0-1]

        Returns:
            (N,) Brightness [0-1]
        """
        if colors is None:
            return None

        # Średnia z RGB
        brightness = colors.mean(axis=1)
        return brightness

    @staticmethod
    def compute_saturation(colors: np.ndarray) -> np.ndarray:
        """
        Oblicza nasycenie (saturation) z kolorów RGB

        Args:
            colors: (N, 3) Kolory RGB [0-1]

        Returns:
            (N,) Saturation [0-1]
        """
        if colors is None:
            return None

        # Odchylenie standardowe RGB
        saturation = colors.std(axis=1)
        return saturation
