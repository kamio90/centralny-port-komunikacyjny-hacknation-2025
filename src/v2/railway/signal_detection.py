"""
Signal Detection - Detekcja sygnalizacji kolejowej

Typy sygnałów:
- Semafory (sygnały świetlne)
- Tarcze ostrzegawcze
- Wskaźniki W
- Tablice kilometrażowe
- Znaki drogowe

HackNation 2025 - CPK Chmura+
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy.spatial import cKDTree
import logging

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Typy sygnałów kolejowych"""
    MAIN_SIGNAL = "main_signal"  # Semafor główny
    DISTANT_SIGNAL = "distant_signal"  # Semafor przejściowy
    SHUNTING_SIGNAL = "shunting_signal"  # Tarcza manewrowa
    SPEED_LIMIT = "speed_limit"  # Ograniczenie prędkości
    WHISTLE_BOARD = "whistle_board"  # Wskaźnik W
    KM_POST = "km_post"  # Słupek kilometrażowy
    WARNING_BOARD = "warning_board"  # Tarcza ostrzegawcza
    UNKNOWN = "unknown"


@dataclass
class RailwaySignal:
    """Wykryty sygnał kolejowy"""
    position: np.ndarray  # (3,) pozycja
    signal_type: SignalType
    height: float  # wysokość nad torem [m]
    width: float  # szerokość [m]
    depth: float  # głębokość [m]
    points: np.ndarray
    confidence: float
    km_position: Optional[float] = None
    side: Optional[str] = None  # 'left', 'right'
    orientation: Optional[float] = None  # kąt [deg]


class SignalDetector:
    """
    Detektor sygnalizacji kolejowej

    Sygnały są wykrywane jako:
    - Małe obiekty na słupach
    - Charakterystyczne kształty (prostokąt, koło)
    - Specyficzna wysokość montażu
    """

    # Typowe wysokości montażu [m]
    MAIN_SIGNAL_HEIGHT = (3.5, 6.0)
    DISTANT_SIGNAL_HEIGHT = (2.5, 4.5)
    SPEED_LIMIT_HEIGHT = (1.5, 3.0)
    KM_POST_HEIGHT = (0.5, 1.5)

    def __init__(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        colors: Optional[np.ndarray] = None,
        track_height: Optional[float] = None
    ):
        """
        Args:
            coords: współrzędne punktów
            classification: klasyfikacja
            colors: kolory RGB (opcjonalnie)
            track_height: wysokość torów
        """
        self.coords = coords
        self.classification = classification
        self.colors = colors

        # Wysokość torów
        if track_height is None:
            track_mask = classification == 18
            if track_mask.any():
                self.track_height = coords[track_mask, 2].mean()
            else:
                ground_mask = classification == 2
                if ground_mask.any():
                    self.track_height = coords[ground_mask, 2].mean()
                else:
                    self.track_height = coords[:, 2].min()
        else:
            self.track_height = track_height

        logger.info(f"Track height for signals: {self.track_height:.2f}m")

    def detect(self) -> List[RailwaySignal]:
        """
        Wykryj sygnały kolejowe

        Returns:
            Lista wykrytych sygnałów
        """
        signals = []

        # 1. Szukaj na słupach (klasa 20)
        pole_signals = self._detect_on_poles()
        signals.extend(pole_signals)

        # 2. Szukaj wolnostojących (niesklasyfikowane lub inne)
        standalone_signals = self._detect_standalone()
        signals.extend(standalone_signals)

        # 3. Deduplikacja
        signals = self._deduplicate(signals)

        logger.info(f"Detected {len(signals)} signals")
        return signals

    def _detect_on_poles(self) -> List[RailwaySignal]:
        """Wykryj sygnały montowane na słupach"""
        signals = []

        pole_mask = self.classification == 20
        pole_coords = self.coords[pole_mask]

        if len(pole_coords) < 10:
            return signals

        # Klastruj słupy
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=1.5, min_samples=5).fit(pole_coords[:, :2])
        labels = clustering.labels_

        for label in set(labels):
            if label == -1:
                continue

            mask = labels == label
            cluster_coords = pole_coords[mask]

            # Szukaj "wybrzuszeń" na słupie (sygnały)
            pole_signals = self._find_signals_on_pole(cluster_coords)
            signals.extend(pole_signals)

        return signals

    def _find_signals_on_pole(self, pole_coords: np.ndarray) -> List[RailwaySignal]:
        """Znajdź sygnały na pojedynczym słupie"""
        signals = []

        # Analizuj profil szerokości w funkcji wysokości
        height_bins = np.linspace(
            pole_coords[:, 2].min(),
            pole_coords[:, 2].max(),
            20
        )

        base_width = None

        for i in range(len(height_bins) - 1):
            z_mask = (pole_coords[:, 2] >= height_bins[i]) & \
                     (pole_coords[:, 2] < height_bins[i + 1])
            level_coords = pole_coords[z_mask]

            if len(level_coords) < 3:
                continue

            # Szerokość na tym poziomie
            width = max(
                level_coords[:, 0].max() - level_coords[:, 0].min(),
                level_coords[:, 1].max() - level_coords[:, 1].min()
            )

            # Zapamiętaj bazową szerokość (dół słupa)
            if base_width is None:
                base_width = width
                continue

            # Sygnał = znaczące poszerzenie
            if width > base_width * 1.5 and width > 0.3:
                height_above_track = (height_bins[i] + height_bins[i + 1]) / 2 - self.track_height

                signal_type = self._classify_by_height(height_above_track)

                signals.append(RailwaySignal(
                    position=np.array([
                        level_coords[:, 0].mean(),
                        level_coords[:, 1].mean(),
                        (height_bins[i] + height_bins[i + 1]) / 2
                    ]),
                    signal_type=signal_type,
                    height=height_above_track,
                    width=width,
                    depth=min(
                        level_coords[:, 0].max() - level_coords[:, 0].min(),
                        level_coords[:, 1].max() - level_coords[:, 1].min()
                    ),
                    points=level_coords,
                    confidence=0.7
                ))

        return signals

    def _detect_standalone(self) -> List[RailwaySignal]:
        """Wykryj wolnostojące sygnały"""
        signals = []

        # Szukaj małych obiektów w pobliżu torów
        # Wysokość 0.5-6m nad torem
        track_mask = self.classification == 18
        other_mask = ~track_mask & (self.classification != 2)  # Nie grunt, nie tor

        if not other_mask.any():
            return signals

        other_coords = self.coords[other_mask]
        heights = other_coords[:, 2] - self.track_height

        # Filtruj według wysokości
        height_mask = (heights >= 0.5) & (heights <= 6.0)
        filtered_coords = other_coords[height_mask]

        if len(filtered_coords) < 5:
            return signals

        # Klastruj
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(filtered_coords)
        labels = clustering.labels_

        for label in set(labels):
            if label == -1:
                continue

            mask = labels == label
            cluster = filtered_coords[mask]

            # Sprawdź wymiary - sygnał jest mały
            x_size = cluster[:, 0].max() - cluster[:, 0].min()
            y_size = cluster[:, 1].max() - cluster[:, 1].min()
            z_size = cluster[:, 2].max() - cluster[:, 2].min()

            # Słupek kilometrażowy lub mały sygnał
            if max(x_size, y_size) < 0.5 and z_size < 2.0:
                height = cluster[:, 2].mean() - self.track_height
                signal_type = self._classify_by_height(height)

                signals.append(RailwaySignal(
                    position=cluster.mean(axis=0),
                    signal_type=signal_type,
                    height=height,
                    width=max(x_size, y_size),
                    depth=min(x_size, y_size),
                    points=cluster,
                    confidence=0.5
                ))

        return signals

    def _classify_by_height(self, height: float) -> SignalType:
        """Klasyfikuj sygnał na podstawie wysokości"""
        if self.MAIN_SIGNAL_HEIGHT[0] <= height <= self.MAIN_SIGNAL_HEIGHT[1]:
            return SignalType.MAIN_SIGNAL
        elif self.DISTANT_SIGNAL_HEIGHT[0] <= height <= self.DISTANT_SIGNAL_HEIGHT[1]:
            return SignalType.DISTANT_SIGNAL
        elif self.SPEED_LIMIT_HEIGHT[0] <= height <= self.SPEED_LIMIT_HEIGHT[1]:
            return SignalType.SPEED_LIMIT
        elif self.KM_POST_HEIGHT[0] <= height <= self.KM_POST_HEIGHT[1]:
            return SignalType.KM_POST

        return SignalType.UNKNOWN

    def _classify_by_color(
        self,
        signal: RailwaySignal,
        point_indices: np.ndarray
    ) -> SignalType:
        """Klasyfikuj na podstawie koloru (jeśli dostępny)"""
        if self.colors is None:
            return signal.signal_type

        signal_colors = self.colors[point_indices]

        # Średni kolor
        avg_color = signal_colors.mean(axis=0)

        # Czerwony = sygnał główny/stop
        if avg_color[0] > 150 and avg_color[1] < 100 and avg_color[2] < 100:
            return SignalType.MAIN_SIGNAL

        # Żółty = ostrzeżenie
        if avg_color[0] > 150 and avg_color[1] > 150 and avg_color[2] < 100:
            return SignalType.WARNING_BOARD

        # Biały = wskaźnik
        if avg_color.mean() > 200:
            return SignalType.WHISTLE_BOARD

        return signal.signal_type

    def _deduplicate(
        self,
        signals: List[RailwaySignal],
        min_distance: float = 0.5
    ) -> List[RailwaySignal]:
        """Usuń duplikaty sygnałów"""
        if len(signals) <= 1:
            return signals

        positions = np.array([s.position for s in signals])
        keep = [True] * len(signals)

        for i in range(len(signals)):
            if not keep[i]:
                continue

            for j in range(i + 1, len(signals)):
                if not keep[j]:
                    continue

                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < min_distance:
                    # Zachowaj ten z wyższym confidence
                    if signals[i].confidence >= signals[j].confidence:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

        return [s for s, k in zip(signals, keep) if k]

    def get_statistics(self, signals: List[RailwaySignal]) -> Dict:
        """Statystyki wykrytych sygnałów"""
        if not signals:
            return {'total_count': 0, 'by_type': {}}

        by_type = {}
        for signal in signals:
            type_name = signal.signal_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

        heights = [s.height for s in signals]

        return {
            'total_count': len(signals),
            'by_type': by_type,
            'avg_height_m': np.mean(heights),
            'main_signals': by_type.get('main_signal', 0),
            'distant_signals': by_type.get('distant_signal', 0),
            'speed_limits': by_type.get('speed_limit', 0),
            'km_posts': by_type.get('km_post', 0)
        }


def detect_signals(
    coords: np.ndarray,
    classification: np.ndarray,
    colors: Optional[np.ndarray] = None
) -> List[RailwaySignal]:
    """Convenience function dla detekcji sygnałów"""
    detector = SignalDetector(coords, classification, colors)
    return detector.detect()
