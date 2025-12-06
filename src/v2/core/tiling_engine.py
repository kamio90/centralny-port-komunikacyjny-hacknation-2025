"""
Moduł do dzielenia chmur punktów na kafelki (tiles)

Stabilny, thread-safe system kafelkowania dla przetwarzania
dużych chmur punktów w sposób równoległy.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Tile:
    """Reprezentacja pojedynczego kafelka"""
    tile_id: int
    bounds: Dict[str, Tuple[float, float]]  # {'x': (min, max), 'y': (min, max)}
    center: Tuple[float, float]  # (x, y)
    indices: np.ndarray  # Indeksy punktów w tym kafelku
    point_count: int

    def __repr__(self):
        return f"Tile(id={self.tile_id}, points={self.point_count:,})"


class TilingEngine:
    """
    Silnik do dzielenia chmur punktów na kafelki

    Automatycznie oblicza optymalny rozmiar kafelków bazując na:
    - Liczbie punktów
    - Gęstości punktów
    - Docelowej liczbie punktów na kafelek (300k-800k)
    """

    def __init__(self,
                 coords: np.ndarray,
                 tile_size: Optional[int] = None,
                 target_points_per_tile: int = 500_000):
        """
        Args:
            coords: (N, 3) Współrzędne XYZ
            tile_size: Rozmiar kafelka w metrach (opcjonalne - jeśli None, auto-oblicza)
            target_points_per_tile: Docelowa liczba punktów na kafelek
        """
        self.coords = coords
        self.n_points = len(coords)
        self.target_points_per_tile = target_points_per_tile

        # Oblicz granice chmury
        self.bounds = {
            'x': (coords[:, 0].min(), coords[:, 0].max()),
            'y': (coords[:, 1].min(), coords[:, 1].max()),
            'z': (coords[:, 2].min(), coords[:, 2].max())
        }

        # Auto-oblicz rozmiar kafelka (jeśli nie podano)
        if tile_size is None:
            self.tile_size = self._calculate_optimal_tile_size()
        else:
            self.tile_size = tile_size

        logger.info(f"TilingEngine: {self.n_points:,} punktów, rozmiar kafelka={self.tile_size}m")

    def _calculate_optimal_tile_size(self) -> int:
        """
        Oblicza optymalny rozmiar kafelka bazując na gęstości punktów

        Cel: 300k-800k punktów na kafelek dla optymalnej wydajności
        """
        # Oblicz obszar XY
        x_range = self.bounds['x'][1] - self.bounds['x'][0]
        y_range = self.bounds['y'][1] - self.bounds['y'][0]
        area = x_range * y_range

        # Oblicz gęstość punktów (punkty/m²)
        density = self.n_points / area

        # Oblicz rozmiar kafelka dla docelowej liczby punktów
        tile_area = self.target_points_per_tile / density
        tile_size = int(np.sqrt(tile_area))

        # Zaokrąglij inteligentnie w zależności od rozmiaru
        if tile_size < 20:
            # Małe kafelki (wysoka gęstość): zaokrąglij do 5m
            tile_size = max(5, round(tile_size / 5) * 5)
        else:
            # Duże kafelki (niska gęstość): zaokrąglij do 10m
            tile_size = round(tile_size / 10) * 10

        # Ogranicz do rozsądnego zakresu (5m - 300m)
        tile_size = max(5, min(300, tile_size))

        logger.info(f"Auto-obliczony rozmiar kafelka: {tile_size}m "
                   f"(gęstość: {density:.1f} pkt/m²)")

        return tile_size

    def create_tiles(self) -> List[Tile]:
        """
        Dzieli chmurę punktów na kafelki (ZOPTYMALIZOWANA wersja wektoryzowana)

        Returns:
            Lista obiektów Tile
        """
        x_min, x_max = self.bounds['x']
        y_min, y_max = self.bounds['y']

        # Oblicz siatkę kafelków
        x_edges = np.arange(x_min, x_max + self.tile_size, self.tile_size)
        y_edges = np.arange(y_min, y_max + self.tile_size, self.tile_size)

        n_tiles_x = len(x_edges) - 1
        n_tiles_y = len(y_edges) - 1
        total_tiles = n_tiles_x * n_tiles_y

        logger.info(f"Tworzenie siatki: {n_tiles_x} x {n_tiles_y} = {total_tiles} kafelków")

        # WEKTORYZOWANA WERSJA - przypisz wszystkie punkty do kafelków w jednym przejściu
        logger.info(f"Przypisywanie {self.n_points:,} punktów do kafelków (wektoryzacja)...")

        # Oblicz indeksy kafelków dla każdego punktu
        x_idx = np.floor((self.coords[:, 0] - x_min) / self.tile_size).astype(int)
        y_idx = np.floor((self.coords[:, 1] - y_min) / self.tile_size).astype(int)

        # Ogranicz do poprawnych indeksów (edge cases)
        x_idx = np.clip(x_idx, 0, n_tiles_x - 1)
        y_idx = np.clip(y_idx, 0, n_tiles_y - 1)

        # Utwórz płaski indeks kafelka dla każdego punktu
        tile_indices = x_idx * n_tiles_y + y_idx

        tiles = []
        tile_id = 0

        # Dla każdego kafelka
        for i in range(n_tiles_x):
            for j in range(n_tiles_y):
                flat_idx = i * n_tiles_y + j

                # Znajdź punkty należące do tego kafelka (SZYBKIE - tylko porównanie indeksów)
                point_mask = (tile_indices == flat_idx)
                indices = np.where(point_mask)[0]
                point_count = len(indices)

                # Granice kafelka
                tile_bounds = {
                    'x': (x_edges[i], x_edges[i + 1]),
                    'y': (y_edges[j], y_edges[j + 1])
                }

                # Środek kafelka
                center = (
                    (tile_bounds['x'][0] + tile_bounds['x'][1]) / 2,
                    (tile_bounds['y'][0] + tile_bounds['y'][1]) / 2
                )

                # Utwórz kafelek
                tile = Tile(
                    tile_id=tile_id,
                    bounds=tile_bounds,
                    center=center,
                    indices=indices,
                    point_count=point_count
                )

                tiles.append(tile)
                tile_id += 1

        # Statystyki
        non_empty = [t for t in tiles if t.point_count > 0]
        logger.info(f"Utworzono {len(tiles)} kafelków ({len(non_empty)} niepustych)")

        if non_empty:
            avg_points = np.mean([t.point_count for t in non_empty])
            logger.info(f"Średnia liczba punktów na kafelek: {avg_points:,.0f}")

        return tiles

    def get_tile_data(self, tile: Tile) -> Dict[str, np.ndarray]:
        """
        Pobiera dane punktów dla danego kafelka

        Args:
            tile: Obiekt Tile

        Returns:
            Dict z danymi kafelka (coords, indices)
        """
        return {
            'coords': self.coords[tile.indices],
            'indices': tile.indices,  # Globalne indeksy
            'bounds': tile.bounds,
            'tile_id': tile.tile_id,
            'point_count': tile.point_count
        }

    def get_statistics(self) -> Dict:
        """
        Zwraca statystyki kafelkowania

        Returns:
            Dict ze statystykami
        """
        tiles = self.create_tiles()
        non_empty = [t for t in tiles if t.point_count > 0]

        stats = {
            'total_tiles': len(tiles),
            'non_empty_tiles': len(non_empty),
            'empty_tiles': len(tiles) - len(non_empty),
            'tile_size_m': self.tile_size,
            'total_points': self.n_points
        }

        if non_empty:
            point_counts = [t.point_count for t in non_empty]
            stats['avg_points_per_tile'] = np.mean(point_counts)
            stats['min_points_per_tile'] = np.min(point_counts)
            stats['max_points_per_tile'] = np.max(point_counts)

        return stats

    def get_visualization_data(self) -> Dict:
        """
        Przygotowuje dane do wizualizacji kafelków (dla Streamlit)

        Returns:
            Dict z danymi do plotly
        """
        tiles = self.create_tiles()

        rectangles = []
        for tile in tiles:
            # Narożniki prostokąta (dla plotly)
            x_min, x_max = tile.bounds['x']
            y_min, y_max = tile.bounds['y']

            corners = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
                [x_min, y_min]  # Zamknij prostokąt
            ])

            rectangles.append({
                'tile_id': tile.tile_id,
                'corners': corners,
                'center': tile.center,
                'point_count': tile.point_count
            })

        return {
            'rectangles': rectangles,
            'bounds': self.bounds
        }
