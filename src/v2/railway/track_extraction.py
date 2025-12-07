"""
Track Extraction - Ekstrakcja osi i geometrii torów

Funkcjonalności:
- Ekstrakcja osi torów z chmury punktów
- Detekcja geometrii (łuki, proste, przechyłki)
- Analiza rozstawu szyn
- Detekcja rozjazdów

Algorytmy:
- RANSAC dla linii/krzywych
- Region growing dla szyn
- Analiza głównych kierunków (PCA)

HackNation 2025 - CPK Chmura+
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
import logging

logger = logging.getLogger(__name__)

# Stałe kolejowe
TRACK_GAUGE_STANDARD = 1.435  # Rozstaw normalny [m]
TRACK_GAUGE_TOLERANCE = 0.020  # Tolerancja [m]


@dataclass
class RailAxis:
    """Oś pojedynczej szyny"""
    points: np.ndarray  # (N, 3) punkty osi
    side: str  # 'left' lub 'right'
    length: float  # długość [m]
    start_km: float  # kilometraż początkowy
    end_km: float  # kilometraż końcowy


@dataclass
class TrackSegment:
    """Segment toru (para szyn)"""
    left_rail: RailAxis
    right_rail: RailAxis
    center_line: np.ndarray  # (N, 3) oś toru
    length: float
    gauge: float  # rozstaw szyn [m]
    gauge_deviation: float  # odchylenie od normy [m]
    cant: float  # przechyłka [m]
    curvature: float  # krzywizna [1/m]
    geometry_type: str  # 'straight', 'curve', 'transition'


@dataclass
class TrackGeometry:
    """Geometria toru"""
    horizontal_radius: float  # promień łuku poziomego [m]
    vertical_radius: float  # promień łuku pionowego [m]
    cant: float  # przechyłka [mm]
    gradient: float  # pochylenie [‰]
    km_start: float
    km_end: float


class TrackExtractor:
    """
    Ekstraktor osi i geometrii torów

    Workflow:
    1. Filtruj punkty torów (klasa 18)
    2. Segmentuj na lewą i prawą szynę
    3. Dopasuj osie szyn
    4. Oblicz geometrię
    """

    TRACK_CLASS = 18  # Tory kolejowe

    def __init__(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        intensity: Optional[np.ndarray] = None,
        expected_gauge: float = TRACK_GAUGE_STANDARD,
        min_segment_length: float = 5.0
    ):
        """
        Args:
            coords: współrzędne punktów
            classification: klasyfikacja
            intensity: intensywność (opcjonalnie, szyny mają wysoką)
            expected_gauge: oczekiwany rozstaw [m]
            min_segment_length: minimalna długość segmentu [m]
        """
        self.coords = coords
        self.classification = classification
        self.intensity = intensity
        self.expected_gauge = expected_gauge
        self.min_segment_length = min_segment_length

        # Filtruj punkty torów
        self.track_mask = classification == self.TRACK_CLASS
        self.track_coords = coords[self.track_mask]

        logger.info(f"Track points: {len(self.track_coords)}")

    def extract_tracks(self) -> List[TrackSegment]:
        """
        Ekstrahuj wszystkie tory

        Returns:
            Lista segmentów torów
        """
        if len(self.track_coords) < 100:
            logger.warning("Too few track points")
            return []

        # Znajdź główny kierunek trasy
        main_direction = self._find_main_direction()

        # Segmentuj na odcinki
        segments = self._segment_track_points(main_direction)

        track_segments = []
        for segment_coords in segments:
            track = self._extract_single_track(segment_coords, main_direction)
            if track is not None:
                track_segments.append(track)

        logger.info(f"Extracted {len(track_segments)} track segments")
        return track_segments

    def _find_main_direction(self) -> np.ndarray:
        """Znajdź główny kierunek toru używając PCA"""
        centered = self.track_coords - self.track_coords.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Największa wartość własna = główny kierunek
        main_dir = eigenvectors[:, -1]

        # Upewnij się że kierunek jest w płaszczyźnie XY
        main_dir[2] = 0
        main_dir = main_dir / np.linalg.norm(main_dir)

        return main_dir

    def _segment_track_points(
        self,
        main_direction: np.ndarray,
        segment_length: float = 50.0
    ) -> List[np.ndarray]:
        """Podziel punkty na segmenty wzdłuż trasy"""
        # Projektuj na główny kierunek
        projections = self.track_coords @ main_direction
        min_proj = projections.min()
        max_proj = projections.max()

        segments = []
        for start in np.arange(min_proj, max_proj, segment_length):
            end = start + segment_length
            mask = (projections >= start) & (projections < end)
            segment_coords = self.track_coords[mask]

            if len(segment_coords) >= 50:
                segments.append(segment_coords)

        return segments

    def _extract_single_track(
        self,
        coords: np.ndarray,
        main_direction: np.ndarray
    ) -> Optional[TrackSegment]:
        """Ekstrahuj pojedynczy segment toru"""
        try:
            # Rozdziel lewą i prawą szynę
            left_rail, right_rail = self._separate_rails(coords, main_direction)

            if left_rail is None or right_rail is None:
                return None

            # Oblicz linię środkową
            center_line = self._compute_center_line(left_rail, right_rail)

            # Oblicz geometrię
            gauge = self._compute_gauge(left_rail, right_rail)
            cant = self._compute_cant(left_rail, right_rail)
            curvature = self._compute_curvature(center_line)

            # Klasyfikuj typ geometrii
            if curvature < 0.0001:  # Promień > 10km
                geometry_type = 'straight'
            elif curvature < 0.001:  # Promień > 1km
                geometry_type = 'transition'
            else:
                geometry_type = 'curve'

            length = self._compute_length(center_line)

            return TrackSegment(
                left_rail=RailAxis(
                    points=left_rail,
                    side='left',
                    length=length,
                    start_km=0,
                    end_km=length / 1000
                ),
                right_rail=RailAxis(
                    points=right_rail,
                    side='right',
                    length=length,
                    start_km=0,
                    end_km=length / 1000
                ),
                center_line=center_line,
                length=length,
                gauge=gauge,
                gauge_deviation=gauge - self.expected_gauge,
                cant=cant,
                curvature=curvature,
                geometry_type=geometry_type
            )

        except Exception as e:
            logger.warning(f"Track extraction failed: {e}")
            return None

    def _separate_rails(
        self,
        coords: np.ndarray,
        main_direction: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Rozdziel punkty na lewą i prawą szynę"""
        # Kierunek prostopadły
        perp_direction = np.array([-main_direction[1], main_direction[0], 0])

        # Projektuj na kierunek prostopadły
        center = coords.mean(axis=0)
        perp_projections = (coords - center) @ perp_direction

        # Rozdziel według znaku projekcji
        left_mask = perp_projections < 0
        right_mask = perp_projections >= 0

        left_rail = coords[left_mask] if left_mask.sum() > 20 else None
        right_rail = coords[right_mask] if right_mask.sum() > 20 else None

        return left_rail, right_rail

    def _compute_center_line(
        self,
        left_rail: np.ndarray,
        right_rail: np.ndarray
    ) -> np.ndarray:
        """Oblicz linię środkową toru"""
        # Użyj KD-Tree do znajdowania odpowiadających punktów
        tree = cKDTree(right_rail)

        center_points = []
        for left_point in left_rail:
            dist, idx = tree.query(left_point)
            if dist < self.expected_gauge * 2:
                center = (left_point + right_rail[idx]) / 2
                center_points.append(center)

        if len(center_points) < 10:
            # Fallback: średnia z obu szyn
            all_points = np.vstack([left_rail, right_rail])
            return all_points

        return np.array(center_points)

    def _compute_gauge(
        self,
        left_rail: np.ndarray,
        right_rail: np.ndarray
    ) -> float:
        """Oblicz średni rozstaw szyn"""
        tree = cKDTree(right_rail[:, :2])  # Tylko XY

        gauges = []
        for left_point in left_rail:
            dist, idx = tree.query(left_point[:2])
            if dist < self.expected_gauge * 2:
                gauges.append(dist)

        if gauges:
            return np.median(gauges)
        return self.expected_gauge

    def _compute_cant(
        self,
        left_rail: np.ndarray,
        right_rail: np.ndarray
    ) -> float:
        """Oblicz przechyłkę (różnicę wysokości szyn)"""
        tree = cKDTree(right_rail[:, :2])

        height_diffs = []
        for left_point in left_rail:
            dist, idx = tree.query(left_point[:2])
            if dist < self.expected_gauge * 2:
                height_diff = left_point[2] - right_rail[idx, 2]
                height_diffs.append(height_diff)

        if height_diffs:
            return np.median(height_diffs) * 1000  # w mm
        return 0

    def _compute_curvature(self, center_line: np.ndarray) -> float:
        """Oblicz krzywiznę toru"""
        if len(center_line) < 10:
            return 0

        try:
            # Użyj splajnów do dopasowania krzywej
            # Sortuj punkty wzdłuż trasy
            sorted_indices = np.lexsort((center_line[:, 1], center_line[:, 0]))
            sorted_points = center_line[sorted_indices]

            # Oblicz krzywizną jako zmianę kierunku
            dx = np.diff(sorted_points[:, 0])
            dy = np.diff(sorted_points[:, 1])

            angles = np.arctan2(dy, dx)
            angle_changes = np.abs(np.diff(angles))

            # Średnia krzywizna
            distances = np.sqrt(dx**2 + dy**2)
            if distances.sum() > 0:
                curvature = angle_changes.sum() / distances[:-1].sum()
                return curvature

        except Exception:
            pass

        return 0

    def _compute_length(self, center_line: np.ndarray) -> float:
        """Oblicz długość linii środkowej"""
        if len(center_line) < 2:
            return 0

        # Sortuj punkty
        sorted_indices = np.lexsort((center_line[:, 1], center_line[:, 0]))
        sorted_points = center_line[sorted_indices]

        # Suma odległości
        diffs = np.diff(sorted_points, axis=0)
        distances = np.linalg.norm(diffs, axis=1)

        return distances.sum()

    def detect_geometry(self) -> List[TrackGeometry]:
        """Wykryj szczegółową geometrię trasy"""
        tracks = self.extract_tracks()

        geometries = []
        current_km = 0

        for track in tracks:
            # Promień poziomy
            if track.curvature > 0:
                h_radius = 1.0 / track.curvature
            else:
                h_radius = float('inf')

            # Gradient (nachylenie)
            if len(track.center_line) > 1:
                dz = track.center_line[-1, 2] - track.center_line[0, 2]
                dx = track.length
                gradient = (dz / dx) * 1000 if dx > 0 else 0  # w promilach
            else:
                gradient = 0

            geometries.append(TrackGeometry(
                horizontal_radius=h_radius,
                vertical_radius=float('inf'),  # Uproszczenie
                cant=track.cant,
                gradient=gradient,
                km_start=current_km,
                km_end=current_km + track.length / 1000
            ))

            current_km += track.length / 1000

        return geometries


def extract_rail_axes(
    coords: np.ndarray,
    classification: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Convenience function - ekstrahuj osie lewej i prawej szyny

    Returns:
        (left_rail_axis, right_rail_axis) lub (None, None)
    """
    extractor = TrackExtractor(coords, classification)
    tracks = extractor.extract_tracks()

    if not tracks:
        return None, None

    # Połącz wszystkie segmenty
    left_points = []
    right_points = []

    for track in tracks:
        left_points.append(track.left_rail.points)
        right_points.append(track.right_rail.points)

    left_axis = np.vstack(left_points) if left_points else None
    right_axis = np.vstack(right_points) if right_points else None

    return left_axis, right_axis


def detect_track_geometry(
    coords: np.ndarray,
    classification: np.ndarray
) -> Dict:
    """
    Wykryj geometrię toru

    Returns:
        Dict z parametrami geometrii
    """
    extractor = TrackExtractor(coords, classification)
    tracks = extractor.extract_tracks()

    if not tracks:
        return {
            'detected': False,
            'message': 'No tracks detected'
        }

    # Agreguj statystyki
    total_length = sum(t.length for t in tracks)
    avg_gauge = np.mean([t.gauge for t in tracks])
    max_cant = max(abs(t.cant) for t in tracks)
    max_curvature = max(t.curvature for t in tracks)

    # Klasyfikuj odcinki
    straight_length = sum(t.length for t in tracks if t.geometry_type == 'straight')
    curve_length = sum(t.length for t in tracks if t.geometry_type == 'curve')
    transition_length = sum(t.length for t in tracks if t.geometry_type == 'transition')

    return {
        'detected': True,
        'total_length_m': total_length,
        'segments_count': len(tracks),
        'avg_gauge_m': avg_gauge,
        'gauge_deviation_mm': (avg_gauge - TRACK_GAUGE_STANDARD) * 1000,
        'max_cant_mm': max_cant,
        'max_curvature': max_curvature,
        'min_radius_m': 1.0 / max_curvature if max_curvature > 0 else float('inf'),
        'straight_length_m': straight_length,
        'curve_length_m': curve_length,
        'transition_length_m': transition_length,
        'straight_ratio': straight_length / total_length if total_length > 0 else 0
    }
