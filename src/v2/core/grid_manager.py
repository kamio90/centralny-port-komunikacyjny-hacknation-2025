"""
Grid Manager - Podział chmury punktów na siatkę kwadratów

Optymalizowany dla hackathonu:
- Target: 500k-1M punktów na kwadrat (~30-60s przetwarzania)
- Automatyczne obliczanie optymalnego rozmiaru siatki
- Wizualizacja 2D z ponumerowanymi kwadratami
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GridSquare:
    """Reprezentacja pojedynczego kwadratu siatki"""
    square_id: int  # 1-indexed dla użytkownika
    row: int
    col: int
    bounds: Dict[str, Tuple[float, float]]  # {'x': (min, max), 'y': (min, max)}
    center: Tuple[float, float]
    indices: np.ndarray
    point_count: int
    area_m2: float
    is_classified: bool = False
    classification: Optional[np.ndarray] = None


class GridManager:
    """
    Manager siatki kwadratów dla chmury punktów

    Użycie:
        manager = GridManager(coords, colors, intensity, target_points_per_square=750_000)
        squares = manager.create_grid()
        viz_data = manager.get_visualization_data()
    """

    def __init__(
        self,
        coords: np.ndarray,
        colors: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
        target_points_per_square: int = 750_000
    ):
        self.coords = coords
        self.colors = colors
        self.intensity = intensity
        self.n_points = len(coords)
        self.target_points_per_square = target_points_per_square

        # Oblicz granice
        self.x_min = float(coords[:, 0].min())
        self.x_max = float(coords[:, 0].max())
        self.y_min = float(coords[:, 1].min())
        self.y_max = float(coords[:, 1].max())
        self.z_min = float(coords[:, 2].min())
        self.z_max = float(coords[:, 2].max())

        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min
        self.total_area = self.x_range * self.y_range

        # Oblicz gęstość
        self.density = self.n_points / self.total_area if self.total_area > 0 else 1

        # Oblicz optymalny rozmiar kwadratu
        target_area = self.target_points_per_square / self.density
        self.square_size = np.sqrt(target_area)

        # Zaokrąglij do ładnych wartości
        if self.square_size < 10:
            self.square_size = max(2, round(self.square_size))
        elif self.square_size < 50:
            self.square_size = round(self.square_size / 5) * 5
        else:
            self.square_size = round(self.square_size / 10) * 10

        # Oblicz liczbę wierszy i kolumn
        self.n_cols = max(1, int(np.ceil(self.x_range / self.square_size)))
        self.n_rows = max(1, int(np.ceil(self.y_range / self.square_size)))

        self.squares: List[GridSquare] = []

        logger.info(f"GridManager: {self.n_points:,} punktów")
        logger.info(f"  Obszar: {self.x_range:.1f} x {self.y_range:.1f} m")
        logger.info(f"  Gęstość: {self.density:.1f} pkt/m²")
        logger.info(f"  Rozmiar kwadratu: {self.square_size:.1f} m")
        logger.info(f"  Siatka: {self.n_cols} x {self.n_rows} = {self.n_cols * self.n_rows} kwadratów")

    def create_grid(self) -> List[GridSquare]:
        """
        Tworzy siatkę kwadratów - ULTRA ZOPTYMALIZOWANE dla 100M+ punktów

        Algorytm O(N) - BEZ SORTOWANIA:
        1. Oblicz grid_key dla każdego punktu (wektoryzowane)
        2. Użyj np.bincount do zliczenia punktów per kwadrat
        3. Zachowaj grid_keys dla lazy evaluation indeksów
        """
        logger.info("Tworzenie siatki kwadratów (ultra-zoptymalizowane)...")
        import time
        start = time.time()

        # KROK 1: Przypisz punkty do komórek (wektoryzowane) - O(N)
        x_idx = np.floor((self.coords[:, 0] - self.x_min) / self.square_size).astype(np.int32)
        y_idx = np.floor((self.coords[:, 1] - self.y_min) / self.square_size).astype(np.int32)

        # Ogranicz do zakresu
        x_idx = np.clip(x_idx, 0, self.n_cols - 1)
        y_idx = np.clip(y_idx, 0, self.n_rows - 1)

        # KROK 2: Oblicz unikalny klucz dla każdego punktu - O(N)
        # grid_key = row * n_cols + col (unikalny ID kwadratu 0-indexed)
        self._grid_keys = (y_idx * self.n_cols + x_idx).astype(np.int32)

        logger.info(f"  Przypisano punkty do siatki: {time.time() - start:.2f}s")

        # KROK 3: Zlicz punkty per kwadrat - O(N)
        counts = np.bincount(self._grid_keys, minlength=self.n_rows * self.n_cols)

        logger.info(f"  Zliczono punkty: {time.time() - start:.2f}s")

        # KROK 4: Utwórz obiekty GridSquare - O(K) - BEZ sortowania!
        self.squares = []

        square_id = 1
        for row in range(self.n_rows):
            for col in range(self.n_cols):
                grid_key = row * self.n_cols + col
                count = int(counts[grid_key])

                x_min_sq = self.x_min + col * self.square_size
                x_max_sq = x_min_sq + self.square_size
                y_min_sq = self.y_min + row * self.square_size
                y_max_sq = y_min_sq + self.square_size

                square = GridSquare(
                    square_id=square_id,
                    row=row,
                    col=col,
                    bounds={
                        'x': (x_min_sq, x_max_sq),
                        'y': (y_min_sq, y_max_sq)
                    },
                    center=(
                        (x_min_sq + x_max_sq) / 2,
                        (y_min_sq + y_max_sq) / 2
                    ),
                    indices=grid_key,  # Zachowaj klucz dla lazy lookup
                    point_count=count,
                    area_m2=self.square_size ** 2
                )

                self.squares.append(square)
                square_id += 1

        non_empty = len([s for s in self.squares if s.point_count > 0])
        elapsed = time.time() - start
        logger.info(f"  Utworzono {len(self.squares)} kwadratów ({non_empty} niepustych) w {elapsed:.2f}s")

        return self.squares

    def get_square_indices(self, square: GridSquare) -> np.ndarray:
        """
        Pobiera rzeczywiste indeksy punktów dla kwadratu

        Lazy evaluation - np.where jest wywoływane dopiero gdy potrzebne
        Dla pojedynczego kwadratu to O(N) ale tylko raz per kwadrat
        """
        if square.point_count == 0:
            return np.array([], dtype=np.int64)

        grid_key = square.indices  # To jest int (grid_key)
        return np.where(self._grid_keys == grid_key)[0]

    def get_square_by_id(self, square_id: int) -> Optional[GridSquare]:
        """Pobiera kwadrat po ID (1-indexed)"""
        for square in self.squares:
            if square.square_id == square_id:
                return square
        return None

    def get_squares_by_ids(self, square_ids: List[int]) -> List[GridSquare]:
        """Pobiera wiele kwadratów po ID"""
        return [s for s in self.squares if s.square_id in square_ids]

    def get_squares(self) -> List[GridSquare]:
        """Pobiera wszystkie kwadraty"""
        return self.squares

    def get_non_empty_squares(self) -> List[GridSquare]:
        """Pobiera tylko niepuste kwadraty"""
        return [s for s in self.squares if s.point_count > 0]

    def estimate_processing_time(
        self,
        square_ids: List[int],
        seconds_per_point: float = 0.00006
    ) -> float:
        """
        Estymuje czas przetwarzania dla wybranych kwadratów

        Args:
            square_ids: Lista ID kwadratów
            seconds_per_point: Czas na punkt (~60 mikrosekund)

        Returns:
            Estymowany czas w sekundach
        """
        total_points = sum(
            s.point_count for s in self.squares
            if s.square_id in square_ids
        )
        return total_points * seconds_per_point

    def get_visualization_data(self) -> Dict:
        """Przygotowuje dane do wizualizacji w Plotly"""
        squares_data = []

        for square in self.squares:
            x_min, x_max = square.bounds['x']
            y_min, y_max = square.bounds['y']

            # Rogi prostokąta dla Plotly
            corners_x = [x_min, x_max, x_max, x_min, x_min]
            corners_y = [y_min, y_min, y_max, y_max, y_min]

            density = square.point_count / square.area_m2 if square.area_m2 > 0 else 0

            squares_data.append({
                'id': square.square_id,
                'row': square.row,
                'col': square.col,
                'corners_x': corners_x,
                'corners_y': corners_y,
                'center': square.center,
                'point_count': square.point_count,
                'density': density,
                'is_empty': square.point_count == 0,
                'is_classified': square.is_classified
            })

        return {
            'squares': squares_data,
            'square_size': self.square_size,
            'bounds': {
                'x': (self.x_min, self.x_max),
                'y': (self.y_min, self.y_max),
                'z': (self.z_min, self.z_max)
            },
            'n_rows': self.n_rows,
            'n_cols': self.n_cols,
            'total_points': self.n_points,
            'density': self.density
        }

    def get_statistics(self) -> Dict:
        """Pobiera statystyki siatki"""
        non_empty = [s for s in self.squares if s.point_count > 0]
        point_counts = [s.point_count for s in non_empty]

        return {
            'total_squares': len(self.squares),
            'non_empty_squares': len(non_empty),
            'empty_squares': len(self.squares) - len(non_empty),
            'square_size_m': self.square_size,
            'grid_dimensions': f"{self.n_cols} x {self.n_rows}",
            'total_points': self.n_points,
            'avg_points_per_square': int(np.mean(point_counts)) if point_counts else 0,
            'min_points': min(point_counts) if point_counts else 0,
            'max_points': max(point_counts) if point_counts else 0,
            'density_pts_m2': self.density
        }
