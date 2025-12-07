"""
Spatial Cropper - Wybór reprezentatywnego fragmentu chmury

Inteligentnie wybiera fragment chmury punktów do trybu DEMO
z zachowaniem reprezentatywności i różnorodności klas.

SMART, CLEAN, REPRESENTATIVE
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class SpatialCropper:
    """
    Wybiera reprezentatywny fragment chmury punktów dla trybu DEMO

    Strategia:
    1. CENTRUM - domyślnie wybiera środek chmury (median X, Y)
    2. HIGH ENTROPY - opcjonalnie wybiera obszar z największą różnorodnością
    3. CUSTOM - możliwość manualnego wyboru centrum

    Dlaczego centrum?
    - Zazwyczaj najbardziej reprezentatywne (drogi, budynki, infrastruktura)
    - Unika brzegów i artefaktów
    - Stabilne dla różnych chmur
    """

    def __init__(self, coords: np.ndarray):
        """
        Args:
            coords: (N, 3) Współrzędne XYZ całej chmury
        """
        self.coords = coords
        self.n_points = len(coords)

        # Oblicz granice całej chmury
        self.full_bounds = {
            'x': (coords[:, 0].min(), coords[:, 0].max()),
            'y': (coords[:, 1].min(), coords[:, 1].max()),
            'z': (coords[:, 2].min(), coords[:, 2].max())
        }

        # Oblicz centrum (median - bardziej robust niż mean)
        self.center_x = np.median(coords[:, 0])
        self.center_y = np.median(coords[:, 1])

        logger.info(f"SpatialCropper zainicjalizowany:")
        logger.info(f"  Całkowita liczba punktów: {self.n_points:,}")
        logger.info(f"  Granice X: {self.full_bounds['x'][0]:.2f} → {self.full_bounds['x'][1]:.2f}m")
        logger.info(f"  Granice Y: {self.full_bounds['y'][0]:.2f} → {self.full_bounds['y'][1]:.2f}m")
        logger.info(f"  Centrum: ({self.center_x:.2f}, {self.center_y:.2f})")

    def select_area(self,
                    max_points: int,
                    strategy: str = 'center',
                    custom_center: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Wybiera fragment chmury z zadaną maksymalną liczbą punktów

        Args:
            max_points: Maksymalna liczba punktów w wybranym obszarze
            strategy: Strategia wyboru ('center', 'high_entropy', 'custom')
            custom_center: Opcjonalne centrum (x, y) dla strategy='custom'

        Returns:
            Tuple:
                - indices: (M,) Indeksy wybranych punktów
                - metadata: Dict z metadanymi obszaru
        """
        logger.info("=" * 60)
        logger.info(f"SPATIAL CROP - Strategia: {strategy}")
        logger.info("=" * 60)

        # 1. Wybierz centrum
        if strategy == 'center':
            center_x, center_y = self.center_x, self.center_y
            logger.info(f"  Używam centrum chmury: ({center_x:.2f}, {center_y:.2f})")

        elif strategy == 'high_entropy':
            center_x, center_y = self._find_high_entropy_area()
            logger.info(f"  Znaleziono obszar wysokiej entropii: ({center_x:.2f}, {center_y:.2f})")

        elif strategy == 'custom':
            if custom_center is None:
                raise ValueError("custom_center wymagane dla strategy='custom'")
            center_x, center_y = custom_center
            logger.info(f"  Używam custom centrum: ({center_x:.2f}, {center_y:.2f})")

        else:
            raise ValueError(f"Nieznana strategia: {strategy}")

        # 2. Iteracyjnie zwiększaj rozmiar kwadratu aż osiągniesz max_points
        area_side = self._find_optimal_square_size(center_x, center_y, max_points)

        # 3. Wytnij kwadrat
        indices, actual_points = self._crop_square(center_x, center_y, area_side)

        # 4. Przygotuj metadane
        demo_bounds = {
            'x': (center_x - area_side/2, center_x + area_side/2),
            'y': (center_y - area_side/2, center_y + area_side/2),
            'z': self.full_bounds['z']  # Pełny zakres Z
        }

        metadata = {
            'strategy': strategy,
            'center': (center_x, center_y),
            'area_side': area_side,
            'area_m2': area_side ** 2,
            'demo_bounds': demo_bounds,
            'full_bounds': self.full_bounds,
            'demo_points': actual_points,
            'total_points': self.n_points,
            'coverage_pct': (actual_points / self.n_points) * 100
        }

        logger.info(f"  ✓ Wybrano obszar: {area_side:.1f}m × {area_side:.1f}m")
        logger.info(f"  ✓ Liczba punktów: {actual_points:,} / {max_points:,}")
        logger.info(f"  ✓ Pokrycie: {metadata['coverage_pct']:.2f}%")
        logger.info("=" * 60)

        return indices, metadata

    def _find_optimal_square_size(self,
                                   center_x: float,
                                   center_y: float,
                                   max_points: int,
                                   tolerance: float = 0.05) -> float:
        """
        Znajduje optymalny rozmiar kwadratu metodą binary search

        Args:
            center_x, center_y: Centrum kwadratu
            max_points: Maksymalna liczba punktów
            tolerance: Tolerancja (domyślnie 5%)

        Returns:
            Wymiar kwadratu (metry)
        """
        # Oszacuj początkowy rozmiar bazując na gęstości
        x_range = self.full_bounds['x'][1] - self.full_bounds['x'][0]
        y_range = self.full_bounds['y'][1] - self.full_bounds['y'][0]
        area = x_range * y_range
        density = self.n_points / area

        estimated_side = np.sqrt(max_points / density)

        # Binary search dla dokładnego rozmiaru
        # WAŻNE: NIGDY nie przekraczamy max_points - better safe than sorry!
        min_side = estimated_side * 0.1  # Start bardzo konserwatywnie
        max_side = estimated_side * 1.2

        best_side = min_side  # Najlepszy rozmiar który NIE przekracza max_points

        for iteration in range(30):  # Więcej iteracji dla lepszej precyzji
            test_side = (min_side + max_side) / 2
            _, test_points = self._crop_square(center_x, center_y, test_side)

            if test_points <= max_points:
                # OK - nie przekracza max_points
                best_side = test_side  # Zapamiętaj

                # Sprawdź czy wystarczająco blisko
                if abs(test_points - max_points) / max_points < tolerance:
                    return test_side

                # Spróbuj większy rozmiar
                min_side = test_side
            else:
                # Przekracza max_points - zmniejsz
                max_side = test_side

        # Zwróć najlepszy rozmiar który NIE przekracza max_points
        logger.info(f"    Binary search: {best_side:.2f}m (po 30 iteracjach)")
        return best_side

    def _crop_square(self,
                     center_x: float,
                     center_y: float,
                     side: float) -> Tuple[np.ndarray, int]:
        """
        Wycina kwadrat wokół centrum

        Args:
            center_x, center_y: Centrum kwadratu
            side: Wymiar kwadratu (metry)

        Returns:
            Tuple:
                - indices: (M,) Indeksy punktów w kwadracie
                - count: Liczba punktów
        """
        half_side = side / 2

        mask = (
            (self.coords[:, 0] >= center_x - half_side) &
            (self.coords[:, 0] <= center_x + half_side) &
            (self.coords[:, 1] >= center_y - half_side) &
            (self.coords[:, 1] <= center_y + half_side)
        )

        indices = np.where(mask)[0]
        return indices, len(indices)

    def _find_high_entropy_area(self, grid_size: int = 10) -> Tuple[float, float]:
        """
        Znajduje obszar z największą entropią (różnorodnością wysokości)

        Strategia:
        1. Podziel chmurę na siatkę grid_size × grid_size
        2. Dla każdej komórki oblicz entropię wysokości (std dev Z)
        3. Wybierz komórkę z największą entropią

        Args:
            grid_size: Rozmiar siatki (domyślnie 10×10)

        Returns:
            Centrum obszaru o największej entropii (x, y)
        """
        x_min, x_max = self.full_bounds['x']
        y_min, y_max = self.full_bounds['y']

        x_step = (x_max - x_min) / grid_size
        y_step = (y_max - y_min) / grid_size

        max_entropy = -1
        best_center = (self.center_x, self.center_y)

        for i in range(grid_size):
            for j in range(grid_size):
                # Granice komórki
                cell_x_min = x_min + i * x_step
                cell_x_max = cell_x_min + x_step
                cell_y_min = y_min + j * y_step
                cell_y_max = cell_y_min + y_step

                # Punkty w komórce
                mask = (
                    (self.coords[:, 0] >= cell_x_min) &
                    (self.coords[:, 0] < cell_x_max) &
                    (self.coords[:, 1] >= cell_y_min) &
                    (self.coords[:, 1] < cell_y_max)
                )

                cell_points = self.coords[mask]

                if len(cell_points) < 1000:  # Zbyt mało punktów
                    continue

                # Oblicz entropię (std dev wysokości)
                z_std = cell_points[:, 2].std()

                if z_std > max_entropy:
                    max_entropy = z_std
                    best_center = (
                        (cell_x_min + cell_x_max) / 2,
                        (cell_y_min + cell_y_max) / 2
                    )

        logger.info(f"    Max entropia: {max_entropy:.2f} (std dev Z)")

        return best_center

    def visualize_selection(self, indices: np.ndarray) -> str:
        """
        Generuje ASCII art wizualizację wybranego obszaru

        Args:
            indices: Indeksy wybranych punktów

        Returns:
            String z wizualizacją
        """
        # Prosta wizualizacja - mapa gęstości 20×20
        grid_size = 20

        x_min, x_max = self.full_bounds['x']
        y_min, y_max = self.full_bounds['y']

        x_step = (x_max - x_min) / grid_size
        y_step = (y_max - y_min) / grid_size

        # Stwórz siatkę gęstości
        density_grid = np.zeros((grid_size, grid_size))
        selection_grid = np.zeros((grid_size, grid_size), dtype=bool)

        for idx in range(self.n_points):
            x, y = self.coords[idx, 0], self.coords[idx, 1]
            i = int((x - x_min) / x_step)
            j = int((y - y_min) / y_step)

            i = max(0, min(grid_size - 1, i))
            j = max(0, min(grid_size - 1, j))

            density_grid[j, i] += 1

            if idx in indices:
                selection_grid[j, i] = True

        # Renderuj ASCII
        lines = ["  " + "─" * (grid_size * 2)]

        for j in range(grid_size):
            line = " │"
            for i in range(grid_size):
                if selection_grid[j, i]:
                    line += "██"  # Wybrana komórka
                elif density_grid[j, i] > 0:
                    line += "░░"  # Pełna chmura
                else:
                    line += "  "  # Pusta
            line += "│"
            lines.append(line)

        lines.append("  " + "─" * (grid_size * 2))
        lines.append("  Legend: ██ = DEMO area, ░░ = Full cloud")

        return "\n".join(lines)
