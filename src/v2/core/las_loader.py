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

    def load(self) -> Dict[str, np.ndarray]:
        """
        Wczytuje pełną chmurę punktów

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
            logger.info("Znaleziono kolory RGB")

        # Intensywność (jeśli dostępna)
        intensity = None
        if hasattr(las, 'intensity'):
            # Normalizuj do [0-1]
            intensity = las.intensity / las.intensity.max()
            logger.info("Znaleziono intensywność")

        # Istniejąca klasyfikacja (jeśli dostępna)
        classification = None
        if hasattr(las, 'classification'):
            classification = las.classification
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
