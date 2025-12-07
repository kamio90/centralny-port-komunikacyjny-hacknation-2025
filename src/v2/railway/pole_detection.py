"""
Pole Detection - Detekcja słupów i masztów trakcyjnych

Typy słupów:
- Słupy trakcyjne (catenary poles)
- Słupy oświetleniowe
- Słupy sygnalizacyjne
- Bramownice

Algorytmy:
- Detekcja pionowych struktur
- DBSCAN clustering
- Analiza kształtu i wysokości

HackNation 2025 - CPK Chmura+
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy.spatial import cKDTree
import logging

logger = logging.getLogger(__name__)


class PoleType(Enum):
    """Typy słupów"""
    CATENARY = "catenary"  # Słup trakcyjny
    LIGHTING = "lighting"  # Słup oświetleniowy
    SIGNAL = "signal"  # Słup sygnalizacyjny
    GANTRY = "gantry"  # Bramownica
    UNKNOWN = "unknown"


@dataclass
class Pole:
    """Wykryty słup"""
    position: np.ndarray  # (3,) pozycja podstawy
    top_position: np.ndarray  # (3,) pozycja szczytu
    height: float  # wysokość [m]
    width: float  # szerokość [m]
    pole_type: PoleType
    points: np.ndarray  # punkty słupa
    confidence: float
    km_position: Optional[float] = None  # pozycja kilometrażowa
    side: Optional[str] = None  # 'left', 'right', 'center'


class PoleDetector:
    """
    Detektor słupów trakcyjnych i innych

    Workflow:
    1. Filtruj punkty słupów (klasa 20)
    2. Klastruj w pionie
    3. Analizuj kształt klastra
    4. Klasyfikuj typ słupa
    """

    POLE_CLASS = 20  # Słupy w LAS

    # Typowe wymiary słupów [m]
    CATENARY_HEIGHT_RANGE = (6.0, 12.0)
    LIGHTING_HEIGHT_RANGE = (8.0, 15.0)
    SIGNAL_HEIGHT_RANGE = (3.0, 8.0)

    def __init__(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        ground_height: Optional[float] = None,
        min_pole_height: float = 3.0,
        max_pole_width: float = 2.0
    ):
        """
        Args:
            coords: współrzędne punktów
            classification: klasyfikacja
            ground_height: wysokość gruntu (opcjonalnie)
            min_pole_height: minimalna wysokość słupa [m]
            max_pole_width: maksymalna szerokość słupa [m]
        """
        self.coords = coords
        self.classification = classification
        self.min_pole_height = min_pole_height
        self.max_pole_width = max_pole_width

        # Wysokość gruntu
        if ground_height is None:
            ground_mask = classification == 2
            if ground_mask.any():
                self.ground_height = coords[ground_mask, 2].mean()
            else:
                self.ground_height = coords[:, 2].min()
        else:
            self.ground_height = ground_height

        # Punkty słupów
        self.pole_mask = classification == self.POLE_CLASS
        self.pole_coords = coords[self.pole_mask]

        logger.info(f"Pole points: {len(self.pole_coords)}")

    def detect(self) -> List[Pole]:
        """
        Wykryj wszystkie słupy

        Returns:
            Lista wykrytych słupów
        """
        if len(self.pole_coords) < 10:
            logger.warning("Too few pole points")
            return []

        # Klastruj punkty w XY
        clusters = self._cluster_poles()

        poles = []
        for cluster_coords in clusters:
            pole = self._analyze_cluster(cluster_coords)
            if pole is not None:
                poles.append(pole)

        # Sortuj według pozycji X
        poles.sort(key=lambda p: p.position[0])

        logger.info(f"Detected {len(poles)} poles")
        return poles

    def _cluster_poles(self, eps: float = 1.5) -> List[np.ndarray]:
        """Klastruj punkty słupów używając DBSCAN"""
        from sklearn.cluster import DBSCAN

        if len(self.pole_coords) < 3:
            return []

        # Klastruj w XY (słupy są pionowe)
        clustering = DBSCAN(eps=eps, min_samples=5).fit(self.pole_coords[:, :2])
        labels = clustering.labels_

        clusters = []
        for label in set(labels):
            if label == -1:
                continue
            mask = labels == label
            clusters.append(self.pole_coords[mask])

        return clusters

    def _analyze_cluster(self, coords: np.ndarray) -> Optional[Pole]:
        """Analizuj klaster i utwórz obiekt Pole"""
        if len(coords) < 5:
            return None

        # Wymiary
        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        z_min = coords[:, 2].min()
        z_max = coords[:, 2].max()
        height = z_max - z_min

        # Szerokość jako większy z wymiarów XY
        width = max(x_range, y_range)

        # Sprawdź czy to słup (wysoki i wąski)
        if height < self.min_pole_height:
            return None
        if width > self.max_pole_width:
            return None

        # Stosunek wysokość/szerokość
        aspect_ratio = height / width if width > 0 else 0
        if aspect_ratio < 2:  # Słup powinien być co najmniej 2x wyższy niż szerszy
            return None

        # Pozycje
        base_position = np.array([
            coords[:, 0].mean(),
            coords[:, 1].mean(),
            z_min
        ])

        top_position = np.array([
            coords[:, 0].mean(),
            coords[:, 1].mean(),
            z_max
        ])

        # Wysokość nad gruntem
        height_above_ground = z_max - self.ground_height

        # Klasyfikuj typ słupa
        pole_type = self._classify_pole_type(height_above_ground, width, coords)

        # Confidence
        confidence = min(1.0, len(coords) / 50)

        return Pole(
            position=base_position,
            top_position=top_position,
            height=height,
            width=width,
            pole_type=pole_type,
            points=coords,
            confidence=confidence
        )

    def _classify_pole_type(
        self,
        height: float,
        width: float,
        coords: np.ndarray
    ) -> PoleType:
        """Klasyfikuj typ słupa na podstawie wymiarów"""
        # Słupy trakcyjne - 6-12m
        if self.CATENARY_HEIGHT_RANGE[0] <= height <= self.CATENARY_HEIGHT_RANGE[1]:
            return PoleType.CATENARY

        # Słupy oświetleniowe - wyższe
        if height > self.LIGHTING_HEIGHT_RANGE[0]:
            return PoleType.LIGHTING

        # Słupy sygnalizacyjne - niższe
        if self.SIGNAL_HEIGHT_RANGE[0] <= height <= self.SIGNAL_HEIGHT_RANGE[1]:
            return PoleType.SIGNAL

        # Bramownica - szeroka
        if width > 1.0 and height > 5.0:
            return PoleType.GANTRY

        return PoleType.UNKNOWN

    def assign_positions(
        self,
        poles: List[Pole],
        track_axis: Optional[np.ndarray] = None
    ) -> List[Pole]:
        """
        Przypisz pozycje kilometrażowe i strony do słupów

        Args:
            poles: lista słupów
            track_axis: oś toru (opcjonalnie)
        """
        if track_axis is None or len(track_axis) < 2:
            return poles

        # Posortuj oś toru
        sorted_indices = np.lexsort((track_axis[:, 1], track_axis[:, 0]))
        sorted_axis = track_axis[sorted_indices]

        # Oblicz kilometraż wzdłuż osi
        diffs = np.diff(sorted_axis, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        cumulative_km = np.concatenate([[0], np.cumsum(distances)]) / 1000

        # Główny kierunek toru
        main_dir = sorted_axis[-1, :2] - sorted_axis[0, :2]
        main_dir = main_dir / np.linalg.norm(main_dir)
        perp_dir = np.array([-main_dir[1], main_dir[0]])

        # KD-Tree dla osi
        tree = cKDTree(sorted_axis[:, :2])

        for pole in poles:
            # Znajdź najbliższy punkt na osi
            dist, idx = tree.query(pole.position[:2])
            pole.km_position = cumulative_km[idx]

            # Określ stronę
            vec_to_pole = pole.position[:2] - sorted_axis[idx, :2]
            side_proj = np.dot(vec_to_pole, perp_dir)

            if side_proj < -1:
                pole.side = 'left'
            elif side_proj > 1:
                pole.side = 'right'
            else:
                pole.side = 'center'

        return poles

    def get_statistics(self, poles: List[Pole]) -> Dict:
        """Statystyki wykrytych słupów"""
        if not poles:
            return {
                'total_count': 0,
                'by_type': {}
            }

        by_type = {}
        for pole in poles:
            type_name = pole.pole_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

        heights = [p.height for p in poles]

        # Odstępy między słupami
        if len(poles) > 1:
            positions = np.array([p.position for p in poles])
            diffs = np.diff(positions, axis=0)
            distances = np.linalg.norm(diffs[:, :2], axis=1)
            avg_spacing = np.mean(distances)
        else:
            avg_spacing = 0

        return {
            'total_count': len(poles),
            'by_type': by_type,
            'avg_height_m': np.mean(heights),
            'min_height_m': np.min(heights),
            'max_height_m': np.max(heights),
            'avg_spacing_m': avg_spacing,
            'catenary_count': by_type.get('catenary', 0),
            'lighting_count': by_type.get('lighting', 0),
            'signal_count': by_type.get('signal', 0)
        }


def detect_poles(
    coords: np.ndarray,
    classification: np.ndarray
) -> List[Pole]:
    """Convenience function dla detekcji słupów"""
    detector = PoleDetector(coords, classification)
    return detector.detect()


def classify_pole_type(
    pole_coords: np.ndarray,
    ground_height: float
) -> PoleType:
    """Klasyfikuj typ pojedynczego słupa"""
    height = pole_coords[:, 2].max() - ground_height
    width = max(
        pole_coords[:, 0].max() - pole_coords[:, 0].min(),
        pole_coords[:, 1].max() - pole_coords[:, 1].min()
    )

    if 6.0 <= height <= 12.0:
        return PoleType.CATENARY
    elif height > 12.0:
        return PoleType.LIGHTING
    elif 3.0 <= height <= 8.0:
        return PoleType.SIGNAL
    elif width > 1.0:
        return PoleType.GANTRY

    return PoleType.UNKNOWN
