"""
Moduł do wczytywania chmur punktów LAS/LAZ

Używa laspy do odczytu plików LAS/LAZ z pełnym wsparciem
dla współrzędnych, kolorów RGB, intensywności i istniejących klasyfikacji.
"""

import laspy
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LASLoader:
    """Wczytywanie chmur punktów LAS/LAZ z optymalizacją pamięci"""

    def __init__(self, file_path: str):
        """
        Args:
            file_path: Ścieżka do pliku LAS/LAZ
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Plik nie istnieje: {file_path}")

        logger.info(f"Inicjalizacja loadera dla: {self.file_path.name}")

    def load(self, sample_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Wczytuje chmurę punktów z opcjonalnym próbkowaniem

        Args:
            sample_size: Jeśli podane, losowo wybiera N punktów (dla szybszych testów)

        Returns:
            Dict zawierający:
                - coords: (N, 3) XYZ współrzędne
                - colors: (N, 3) RGB [0-1] lub None
                - intensity: (N,) intensywność [0-1] lub None
                - classification: (N,) istniejące klasy lub None
                - header: laspy.LasHeader (metadane)
        """
        logger.info(f"Wczytywanie: {self.file_path.name}")

        with laspy.open(self.file_path) as f:
            las = f.read()

        # Współrzędne XYZ
        coords = np.vstack([las.x, las.y, las.z]).T
        n_points = len(coords)
        logger.info(f"Wczytano {n_points:,} punktów")

        # Próbkowanie (jeśli wymagane)
        sample_idx = None
        if sample_size and sample_size < n_points:
            sample_idx = np.random.choice(n_points, sample_size, replace=False)
            coords = coords[sample_idx]
            n_points = sample_size
            logger.info(f"Próbkowanie do {n_points:,} punktów")

        # Oblicz granice (bounds)
        bounds = {
            'x': (coords[:, 0].min(), coords[:, 0].max()),
            'y': (coords[:, 1].min(), coords[:, 1].max()),
            'z': (coords[:, 2].min(), coords[:, 2].max())
        }

        # Kolory RGB (jeśli dostępne)
        colors = None
        if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
            # Konwertuj z uint16 [0-65535] do float [0-1]
            colors = np.vstack([
                las.red / 65535.0,
                las.green / 65535.0,
                las.blue / 65535.0
            ]).T
            if sample_idx is not None:
                colors = colors[sample_idx]
            logger.info("Znaleziono kolory RGB")

        # Intensywność (jeśli dostępna)
        intensity = None
        if hasattr(las, 'intensity'):
            # Normalizuj do [0-1]
            max_intensity = las.intensity.max()
            if max_intensity > 0:
                intensity = las.intensity / max_intensity
            else:
                intensity = las.intensity.astype(np.float32)
            if sample_idx is not None:
                intensity = intensity[sample_idx]
            logger.info("Znaleziono intensywność")

        # Istniejąca klasyfikacja (jeśli dostępna)
        classification = None
        if hasattr(las, 'classification'):
            classification = np.array(las.classification)
            if sample_idx is not None:
                classification = classification[sample_idx]
            unique_classes = np.unique(classification)
            logger.info(f"Znaleziono klasyfikację: {len(unique_classes)} klas")

        return {
            'coords': coords,
            'colors': colors,
            'intensity': intensity,
            'classification': classification,
            'bounds': bounds,
            'header': las.header,
            'n_points': n_points
        }

    def load_subset(self, bounds: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
        """
        Wczytuje tylko punkty w określonych granicach (spatial filtering)

        Args:
            bounds: {'x': (min, max), 'y': (min, max), 'z': (min, max)}

        Returns:
            Dict z przefiltrowanymi danymi
        """
        data = self.load()
        coords = data['coords']

        # Filtruj punkty wewnątrz bounds
        mask = np.ones(len(coords), dtype=bool)

        if 'x' in bounds:
            mask &= (coords[:, 0] >= bounds['x'][0]) & (coords[:, 0] <= bounds['x'][1])
        if 'y' in bounds:
            mask &= (coords[:, 1] >= bounds['y'][0]) & (coords[:, 1] <= bounds['y'][1])
        if 'z' in bounds:
            mask &= (coords[:, 2] >= bounds['z'][0]) & (coords[:, 2] <= bounds['z'][1])

        # Zastosuj filtr
        filtered_data = {
            'coords': coords[mask],
            'colors': data['colors'][mask] if data['colors'] is not None else None,
            'intensity': data['intensity'][mask] if data['intensity'] is not None else None,
            'classification': data['classification'][mask] if data['classification'] is not None else None,
            'bounds': data['bounds'],
            'header': data['header'],
            'n_points': mask.sum()
        }

        logger.info(f"Przefiltrowano: {mask.sum():,} / {len(coords):,} punktów")
        return filtered_data

    def load_chunked(self, chunk_size: int = 10_000_000):
        """
        Generatorowa metoda do wczytywania dużych plików w kawałkach

        Używaj dla plików > 50M punktów gdy pamięć jest ograniczona.

        Args:
            chunk_size: Liczba punktów na chunk (domyślnie 10M)

        Yields:
            Dict z danymi dla każdego chunka + informacją o indeksach
        """
        logger.info(f"Wczytywanie chunkami: {self.file_path.name}")

        with laspy.open(self.file_path) as f:
            total_points = f.header.point_count
            logger.info(f"Łącznie {total_points:,} punktów, chunk_size={chunk_size:,}")

            chunk_idx = 0
            start_idx = 0

            for las_chunk in f.chunk_iterator(chunk_size):
                end_idx = start_idx + len(las_chunk.x)

                # Współrzędne
                coords = np.vstack([las_chunk.x, las_chunk.y, las_chunk.z]).T

                # Kolory
                colors = None
                if hasattr(las_chunk, 'red'):
                    colors = np.vstack([
                        las_chunk.red / 65535.0,
                        las_chunk.green / 65535.0,
                        las_chunk.blue / 65535.0
                    ]).T

                # Intensywność
                intensity = None
                if hasattr(las_chunk, 'intensity'):
                    max_int = las_chunk.intensity.max()
                    if max_int > 0:
                        intensity = las_chunk.intensity / max_int
                    else:
                        intensity = las_chunk.intensity.astype(np.float32)

                yield {
                    'coords': coords,
                    'colors': colors,
                    'intensity': intensity,
                    'chunk_idx': chunk_idx,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'n_points': len(coords),
                    'total_points': total_points,
                    'progress': end_idx / total_points * 100
                }

                start_idx = end_idx
                chunk_idx += 1

                logger.info(f"Chunk {chunk_idx}: {start_idx:,}/{total_points:,} ({start_idx/total_points*100:.1f}%)")

    @staticmethod
    def get_file_info(file_path: str) -> Dict:
        """
        Szybkie pobranie informacji o pliku bez pełnego wczytywania

        Args:
            file_path: Ścieżka do pliku LAS/LAZ

        Returns:
            Dict z metadanymi (liczba punktów, bounds, itp.)
        """
        with laspy.open(file_path) as f:
            header = f.header

            # Sprawdź czy format punktów ma kolory RGB
            # Formaty 2, 3, 5, 7, 8, 10 mają RGB
            has_rgb = header.point_format.id in [2, 3, 5, 7, 8, 10]

            return {
                'n_points': header.point_count,
                'bounds': {
                    'x': (header.x_min, header.x_max),
                    'y': (header.y_min, header.y_max),
                    'z': (header.z_min, header.z_max)
                },
                'version': f"{header.version.major}.{header.version.minor}",
                'point_format': header.point_format.id,
                'has_rgb': has_rgb,
                'file_size_mb': Path(file_path).stat().st_size / (1024 * 1024)
            }
