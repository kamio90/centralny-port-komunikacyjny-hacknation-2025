"""
Catenary Detection - Detekcja sieci trakcyjnej

Wykrywa elementy overhead contact system (OCS):
- Przewód jezdny (contact wire) - 5.0-5.5m nad szyną
- Lina nośna (messenger/catenary wire) - 6.0-7.0m nad szyną
- Wieszaki (droppers)
- Przewody powrotne (return conductors)

Algorytmy:
- RANSAC dla dopasowania paraboli/linii
- DBSCAN dla klastrowania
- Analiza wysokości względem torów

HackNation 2025 - CPK Chmura+
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
import logging

logger = logging.getLogger(__name__)


@dataclass
class WireSegment:
    """Segment przewodu"""
    points: np.ndarray  # (N, 3) punkty przewodu
    wire_type: str  # 'contact', 'messenger', 'return', 'dropper'
    start_point: np.ndarray  # początek segmentu
    end_point: np.ndarray  # koniec segmentu
    length: float  # długość [m]
    height_above_track: float  # średnia wysokość nad torem [m]
    sag: float  # zwis [m]
    confidence: float  # pewność detekcji


@dataclass
class CatenarySystem:
    """Kompletny system trakcyjny"""
    contact_wires: List[WireSegment]
    messenger_wires: List[WireSegment]
    return_wires: List[WireSegment]
    droppers: List[WireSegment]
    span_length: float  # rozstaw słupów [m]
    system_height: float  # wysokość systemu [m]
    total_length: float  # całkowita długość trasy [m]


class CatenaryDetector:
    """
    Detektor sieci trakcyjnej

    Wykorzystuje:
    - Filtrowanie wysokości (przewody 4.5-8m nad torem)
    - RANSAC dla dopasowania krzywych
    - DBSCAN dla segmentacji
    """

    # Standardowe wysokości OCS (wg norm UIC)
    CONTACT_WIRE_HEIGHT = (5.0, 5.5)  # przewód jezdny
    MESSENGER_WIRE_HEIGHT = (6.0, 7.5)  # lina nośna
    RETURN_WIRE_HEIGHT = (6.5, 8.0)  # przewód powrotny

    # Klasy LAS
    WIRE_CLASS = 19  # Linie energetyczne

    def __init__(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        track_height: Optional[np.ndarray] = None,
        min_wire_length: float = 10.0,
        ransac_threshold: float = 0.1
    ):
        """
        Args:
            coords: (N, 3) współrzędne punktów
            classification: (N,) klasyfikacja
            track_height: wysokość torów (z klasy 18)
            min_wire_length: minimalna długość przewodu [m]
            ransac_threshold: próg RANSAC [m]
        """
        self.coords = coords
        self.classification = classification
        self.min_wire_length = min_wire_length
        self.ransac_threshold = ransac_threshold

        # Oblicz wysokość torów jeśli nie podano
        if track_height is None:
            track_mask = classification == 18  # Tory
            if track_mask.any():
                self.track_height = coords[track_mask, 2].mean()
            else:
                # Użyj gruntu
                ground_mask = classification == 2
                if ground_mask.any():
                    self.track_height = coords[ground_mask, 2].mean()
                else:
                    self.track_height = coords[:, 2].min()
        else:
            self.track_height = track_height

        logger.info(f"Track height: {self.track_height:.2f}m")

    def detect(self) -> CatenarySystem:
        """
        Wykryj kompletny system trakcyjny

        Returns:
            CatenarySystem z wszystkimi elementami
        """
        logger.info("Detecting catenary system...")

        # Filtruj punkty przewodów
        wire_mask = self.classification == self.WIRE_CLASS
        wire_coords = self.coords[wire_mask]

        if len(wire_coords) == 0:
            logger.warning("No wire points found (class 19)")
            return self._empty_system()

        # Oblicz wysokość nad torem
        heights_above_track = wire_coords[:, 2] - self.track_height

        # Segmentuj według wysokości
        contact_wires = self._detect_wires_at_height(
            wire_coords, heights_above_track,
            self.CONTACT_WIRE_HEIGHT, 'contact'
        )

        messenger_wires = self._detect_wires_at_height(
            wire_coords, heights_above_track,
            self.MESSENGER_WIRE_HEIGHT, 'messenger'
        )

        return_wires = self._detect_wires_at_height(
            wire_coords, heights_above_track,
            self.RETURN_WIRE_HEIGHT, 'return'
        )

        # Wykryj wieszaki (pionowe elementy)
        droppers = self._detect_droppers(wire_coords, heights_above_track)

        # Oblicz statystyki
        all_wires = contact_wires + messenger_wires + return_wires
        total_length = sum(w.length for w in all_wires)

        if len(all_wires) > 1:
            # Oszacuj rozstaw słupów
            span_length = self._estimate_span_length(wire_coords)
        else:
            span_length = 0

        system_height = max(
            [w.height_above_track for w in all_wires],
            default=0
        )

        logger.info(
            f"Detected: {len(contact_wires)} contact, "
            f"{len(messenger_wires)} messenger, "
            f"{len(return_wires)} return wires, "
            f"{len(droppers)} droppers"
        )

        return CatenarySystem(
            contact_wires=contact_wires,
            messenger_wires=messenger_wires,
            return_wires=return_wires,
            droppers=droppers,
            span_length=span_length,
            system_height=system_height,
            total_length=total_length
        )

    def _detect_wires_at_height(
        self,
        coords: np.ndarray,
        heights: np.ndarray,
        height_range: Tuple[float, float],
        wire_type: str
    ) -> List[WireSegment]:
        """Wykryj przewody w danym zakresie wysokości"""
        min_h, max_h = height_range

        # Filtruj według wysokości
        mask = (heights >= min_h) & (heights <= max_h)
        filtered_coords = coords[mask]

        if len(filtered_coords) < 10:
            return []

        # Klastruj punkty
        clusters = self._cluster_points(filtered_coords)

        segments = []
        for cluster_coords in clusters:
            if len(cluster_coords) < 5:
                continue

            # Dopasuj linię/parabolę
            segment = self._fit_wire_segment(cluster_coords, wire_type)
            if segment is not None and segment.length >= self.min_wire_length:
                segments.append(segment)

        return segments

    def _cluster_points(self, coords: np.ndarray, eps: float = 2.0) -> List[np.ndarray]:
        """Klastruj punkty używając DBSCAN"""
        from sklearn.cluster import DBSCAN

        if len(coords) < 3:
            return []

        # Użyj tylko X, Y dla klastrowania (przewody są poziome)
        clustering = DBSCAN(eps=eps, min_samples=3).fit(coords[:, :2])
        labels = clustering.labels_

        clusters = []
        for label in set(labels):
            if label == -1:  # Szum
                continue
            mask = labels == label
            clusters.append(coords[mask])

        return clusters

    def _fit_wire_segment(
        self,
        coords: np.ndarray,
        wire_type: str
    ) -> Optional[WireSegment]:
        """Dopasuj segment przewodu używając RANSAC"""
        if len(coords) < 5:
            return None

        try:
            # Znajdź główny kierunek (PCA)
            centered = coords - coords.mean(axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            main_direction = eigenvectors[:, -1]

            # Projektuj na główny kierunek
            projections = coords @ main_direction
            sorted_indices = np.argsort(projections)
            sorted_coords = coords[sorted_indices]

            # Początek i koniec
            start_point = sorted_coords[0]
            end_point = sorted_coords[-1]
            length = np.linalg.norm(end_point - start_point)

            # Wysokość
            height = coords[:, 2].mean() - self.track_height

            # Zwis (sag) - różnica między max a min wysokości
            z_range = coords[:, 2].max() - coords[:, 2].min()
            sag = z_range

            # Confidence based on point density
            confidence = min(1.0, len(coords) / 100)

            return WireSegment(
                points=coords,
                wire_type=wire_type,
                start_point=start_point,
                end_point=end_point,
                length=length,
                height_above_track=height,
                sag=sag,
                confidence=confidence
            )

        except Exception as e:
            logger.warning(f"Wire fitting failed: {e}")
            return None

    def _detect_droppers(
        self,
        coords: np.ndarray,
        heights: np.ndarray
    ) -> List[WireSegment]:
        """Wykryj wieszaki (pionowe elementy)"""
        # Wieszaki są między przewodem jezdnym a liną nośną
        min_h = self.CONTACT_WIRE_HEIGHT[0]
        max_h = self.MESSENGER_WIRE_HEIGHT[1]

        mask = (heights >= min_h) & (heights <= max_h)
        filtered_coords = coords[mask]

        if len(filtered_coords) < 10:
            return []

        # Szukaj pionowych klastrów
        droppers = []

        # Grid search dla pionowych elementów
        xy_grid_size = 0.5  # 50cm grid
        x_min, y_min = filtered_coords[:, :2].min(axis=0)
        x_max, y_max = filtered_coords[:, :2].max(axis=0)

        for x in np.arange(x_min, x_max, xy_grid_size):
            for y in np.arange(y_min, y_max, xy_grid_size):
                # Punkty w tej komórce
                cell_mask = (
                    (filtered_coords[:, 0] >= x) &
                    (filtered_coords[:, 0] < x + xy_grid_size) &
                    (filtered_coords[:, 1] >= y) &
                    (filtered_coords[:, 1] < y + xy_grid_size)
                )
                cell_coords = filtered_coords[cell_mask]

                if len(cell_coords) < 3:
                    continue

                # Sprawdź czy rozciąga się pionowo
                z_range = cell_coords[:, 2].max() - cell_coords[:, 2].min()
                xy_range = np.linalg.norm(
                    cell_coords[:, :2].max(axis=0) - cell_coords[:, :2].min(axis=0)
                )

                # Wieszak: duży zakres Z, mały zakres XY
                if z_range > 0.5 and xy_range < 0.3:
                    droppers.append(WireSegment(
                        points=cell_coords,
                        wire_type='dropper',
                        start_point=cell_coords[cell_coords[:, 2].argmin()],
                        end_point=cell_coords[cell_coords[:, 2].argmax()],
                        length=z_range,
                        height_above_track=cell_coords[:, 2].mean() - self.track_height,
                        sag=0,
                        confidence=0.7
                    ))

        return droppers

    def _estimate_span_length(self, coords: np.ndarray) -> float:
        """Oszacuj typowy rozstaw słupów"""
        # Znajdź przerwy w danych (gdzie są słupy)
        sorted_by_x = coords[np.argsort(coords[:, 0])]

        gaps = []
        for i in range(1, len(sorted_by_x)):
            gap = sorted_by_x[i, 0] - sorted_by_x[i-1, 0]
            if gap > 5:  # Przerwa > 5m może być słupem
                gaps.append(gap)

        if gaps:
            # Typowy rozstaw to mediana dużych przerw
            return np.median(gaps)

        return 50.0  # Domyślny rozstaw

    def _empty_system(self) -> CatenarySystem:
        """Zwróć pusty system"""
        return CatenarySystem(
            contact_wires=[],
            messenger_wires=[],
            return_wires=[],
            droppers=[],
            span_length=0,
            system_height=0,
            total_length=0
        )

    def get_statistics(self, system: CatenarySystem) -> Dict:
        """Statystyki systemu trakcyjnego"""
        return {
            'contact_wires_count': len(system.contact_wires),
            'messenger_wires_count': len(system.messenger_wires),
            'return_wires_count': len(system.return_wires),
            'droppers_count': len(system.droppers),
            'total_wire_length_m': system.total_length,
            'avg_span_length_m': system.span_length,
            'system_height_m': system.system_height,
            'contact_wire_heights': [w.height_above_track for w in system.contact_wires],
            'messenger_wire_heights': [w.height_above_track for w in system.messenger_wires],
            'avg_sag_m': np.mean([w.sag for w in system.contact_wires]) if system.contact_wires else 0
        }


def detect_catenary_wires(
    coords: np.ndarray,
    classification: np.ndarray
) -> CatenarySystem:
    """Convenience function dla detekcji sieci trakcyjnej"""
    detector = CatenaryDetector(coords, classification)
    return detector.detect()


def detect_contact_wire(
    coords: np.ndarray,
    classification: np.ndarray
) -> List[WireSegment]:
    """Wykryj tylko przewód jezdny"""
    system = detect_catenary_wires(coords, classification)
    return system.contact_wires


def detect_messenger_wire(
    coords: np.ndarray,
    classification: np.ndarray
) -> List[WireSegment]:
    """Wykryj tylko linę nośną"""
    system = detect_catenary_wires(coords, classification)
    return system.messenger_wires
