"""
Terrain Analysis - Analiza terenu dla planowania infrastruktury

Funkcje:
- Generowanie DTM (Digital Terrain Model) - model terenu
- Generowanie DSM (Digital Surface Model) - model powierzchni
- Ekstrakcja przekrojow poprzecznych
- Analiza profilu podluznego
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)


@dataclass
class CrossSection:
    """Przekroj poprzeczny terenu"""
    station: float  # Kilometraz / pozycja wzdluz trasy
    distances: np.ndarray  # Odleglosci od osi [m]
    terrain_heights: np.ndarray  # Wysokosci DTM
    surface_heights: np.ndarray  # Wysokosci DSM
    center_point: np.ndarray  # Punkt srodkowy [x, y, z]
    width: float  # Szerokosc przekroju


@dataclass
class TerrainModel:
    """Model terenu (DTM lub DSM)"""
    grid_x: np.ndarray  # Siatka X
    grid_y: np.ndarray  # Siatka Y
    heights: np.ndarray  # Wysokosci Z
    resolution: float  # Rozdzielczosc [m]
    bounds: Dict  # Granice
    model_type: str  # 'DTM' lub 'DSM'

    def get_height_at(self, x: float, y: float) -> Optional[float]:
        """Zwraca wysokosc w punkcie (interpolacja)"""
        if not (self.bounds['x'][0] <= x <= self.bounds['x'][1] and
                self.bounds['y'][0] <= y <= self.bounds['y'][1]):
            return None

        # Indeksy
        ix = int((x - self.bounds['x'][0]) / self.resolution)
        iy = int((y - self.bounds['y'][0]) / self.resolution)

        ix = max(0, min(ix, self.heights.shape[1] - 1))
        iy = max(0, min(iy, self.heights.shape[0] - 1))

        return float(self.heights[iy, ix])


class TerrainAnalyzer:
    """
    Analizator terenu

    Generuje modele terenu i ekstrahuje przekroje.

    Usage:
        analyzer = TerrainAnalyzer(coords, classification)
        dtm = analyzer.generate_dtm(resolution=1.0)
        dsm = analyzer.generate_dsm(resolution=1.0)
        sections = analyzer.extract_cross_sections(axis_points)
    """

    # Klasy gruntu
    GROUND_CLASSES = [2]

    # Klasy naziemne (do DSM)
    SURFACE_CLASSES = [2, 3, 4, 5, 6, 17, 18, 19, 20, 21, 30, 40, 41]

    def __init__(
        self,
        coords: np.ndarray,
        classification: np.ndarray
    ):
        self.coords = coords
        self.classification = classification
        self.bounds = {
            'x': (float(coords[:, 0].min()), float(coords[:, 0].max())),
            'y': (float(coords[:, 1].min()), float(coords[:, 1].max())),
            'z': (float(coords[:, 2].min()), float(coords[:, 2].max()))
        }

    def generate_dtm(
        self,
        resolution: float = 1.0,
        smooth: bool = True,
        smooth_sigma: float = 1.0
    ) -> TerrainModel:
        """
        Generuje DTM (Digital Terrain Model) - model terenu

        Args:
            resolution: rozdzielczosc siatki [m]
            smooth: czy wygladzic
            smooth_sigma: parametr wygladzania

        Returns:
            TerrainModel z DTM
        """
        logger.info(f"Generating DTM with resolution {resolution}m...")

        # Punkty gruntu
        ground_mask = np.isin(self.classification, self.GROUND_CLASSES)
        if not ground_mask.any():
            logger.warning("No ground points (class 2) found, using lowest points")
            # Fallback - uzyj najnizszych punktow
            z_threshold = np.percentile(self.coords[:, 2], 10)
            ground_mask = self.coords[:, 2] < z_threshold

        ground_points = self.coords[ground_mask]
        logger.info(f"Using {len(ground_points):,} ground points")

        # Siatka
        x_min, x_max = self.bounds['x']
        y_min, y_max = self.bounds['y']

        grid_x = np.arange(x_min, x_max + resolution, resolution)
        grid_y = np.arange(y_min, y_max + resolution, resolution)
        xx, yy = np.meshgrid(grid_x, grid_y)

        # Interpolacja
        heights = griddata(
            ground_points[:, :2],
            ground_points[:, 2],
            (xx, yy),
            method='linear',
            fill_value=np.nan
        )

        # Wypelnij NaN najblizszym sasiadem
        if np.isnan(heights).any():
            heights_nn = griddata(
                ground_points[:, :2],
                ground_points[:, 2],
                (xx, yy),
                method='nearest'
            )
            heights = np.where(np.isnan(heights), heights_nn, heights)

        # Wygladzanie
        if smooth:
            heights = gaussian_filter(heights, sigma=smooth_sigma)

        logger.info(f"DTM generated: {heights.shape}")

        return TerrainModel(
            grid_x=grid_x,
            grid_y=grid_y,
            heights=heights,
            resolution=resolution,
            bounds=self.bounds,
            model_type='DTM'
        )

    def generate_dsm(
        self,
        resolution: float = 1.0
    ) -> TerrainModel:
        """
        Generuje DSM (Digital Surface Model) - model powierzchni

        Uwzglednia wszystkie obiekty (budynki, roslinnosc, itp.)

        Args:
            resolution: rozdzielczosc siatki [m]

        Returns:
            TerrainModel z DSM
        """
        logger.info(f"Generating DSM with resolution {resolution}m...")

        # Wszystkie punkty powierzchniowe
        surface_mask = np.isin(self.classification, self.SURFACE_CLASSES)
        if not surface_mask.any():
            surface_mask = np.ones(len(self.coords), dtype=bool)

        surface_points = self.coords[surface_mask]

        # Siatka
        x_min, x_max = self.bounds['x']
        y_min, y_max = self.bounds['y']

        grid_x = np.arange(x_min, x_max + resolution, resolution)
        grid_y = np.arange(y_min, y_max + resolution, resolution)

        # Dla DSM bierzemy maksymalna wysokosc w komorce
        nx = len(grid_x)
        ny = len(grid_y)
        heights = np.full((ny, nx), np.nan)

        # Przypisz punkty do komorek
        ix = ((surface_points[:, 0] - x_min) / resolution).astype(int)
        iy = ((surface_points[:, 1] - y_min) / resolution).astype(int)

        # Ogranicz do granic
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)

        # Dla kazdej komorki - max wysokosc
        for i in range(len(surface_points)):
            current = heights[iy[i], ix[i]]
            new_val = surface_points[i, 2]
            if np.isnan(current) or new_val > current:
                heights[iy[i], ix[i]] = new_val

        # Wypelnij NaN interpolacja
        valid_mask = ~np.isnan(heights)
        if valid_mask.any() and (~valid_mask).any():
            valid_points = np.array(np.where(valid_mask)).T
            valid_values = heights[valid_mask]

            invalid_points = np.array(np.where(~valid_mask)).T
            if len(invalid_points) > 0:
                filled = griddata(valid_points, valid_values, invalid_points, method='nearest')
                heights[~valid_mask] = filled

        logger.info(f"DSM generated: {heights.shape}")

        return TerrainModel(
            grid_x=grid_x,
            grid_y=grid_y,
            heights=heights,
            resolution=resolution,
            bounds=self.bounds,
            model_type='DSM'
        )

    def extract_cross_sections(
        self,
        axis_points: np.ndarray,
        width: float = 50.0,
        interval: float = 10.0,
        resolution: float = 0.5
    ) -> List[CrossSection]:
        """
        Ekstrahuje przekroje poprzeczne wzdluz osi trasy

        Args:
            axis_points: (M, 2) lub (M, 3) punkty osi trasy
            width: szerokosc przekroju [m]
            interval: co ile metrow przekroj [m]
            resolution: rozdzielczosc wewnatrz przekroju [m]

        Returns:
            Lista przekrojow
        """
        logger.info(f"Extracting cross-sections: width={width}m, interval={interval}m")

        # Generuj modele
        dtm = self.generate_dtm(resolution=1.0)
        dsm = self.generate_dsm(resolution=1.0)

        sections = []
        cumulative_distance = 0.0

        for i in range(len(axis_points) - 1):
            p1 = axis_points[i][:2]
            p2 = axis_points[i + 1][:2]

            segment_length = np.linalg.norm(p2 - p1)
            direction = (p2 - p1) / segment_length
            perpendicular = np.array([-direction[1], direction[0]])

            # Ile przekrojow w tym segmencie
            n_sections = int(segment_length / interval)

            for j in range(n_sections):
                # Pozycja wzdluz segmentu
                t = (j * interval) / segment_length
                center = p1 + t * (p2 - p1)
                station = cumulative_distance + j * interval

                # Punkty przekroju
                distances = np.arange(-width/2, width/2 + resolution, resolution)
                terrain_heights = []
                surface_heights = []

                for d in distances:
                    point = center + d * perpendicular
                    th = dtm.get_height_at(point[0], point[1])
                    sh = dsm.get_height_at(point[0], point[1])
                    terrain_heights.append(th if th is not None else np.nan)
                    surface_heights.append(sh if sh is not None else np.nan)

                # Wysokosc srodka
                center_z = dtm.get_height_at(center[0], center[1]) or 0

                sections.append(CrossSection(
                    station=station,
                    distances=distances,
                    terrain_heights=np.array(terrain_heights),
                    surface_heights=np.array(surface_heights),
                    center_point=np.array([center[0], center[1], center_z]),
                    width=width
                ))

            cumulative_distance += segment_length

        logger.info(f"Extracted {len(sections)} cross-sections")
        return sections

    def extract_longitudinal_profile(
        self,
        axis_points: np.ndarray,
        interval: float = 1.0
    ) -> Dict:
        """
        Ekstrahuje profil podluzny wzdluz osi trasy

        Args:
            axis_points: punkty osi trasy
            interval: rozdzielczosc [m]

        Returns:
            Dict z profilem
        """
        logger.info("Extracting longitudinal profile...")

        dtm = self.generate_dtm(resolution=1.0)

        stations = []
        terrain_heights = []
        surface_heights = []

        dsm = self.generate_dsm(resolution=1.0)

        cumulative = 0.0
        for i in range(len(axis_points) - 1):
            p1 = axis_points[i][:2]
            p2 = axis_points[i + 1][:2]

            segment_length = np.linalg.norm(p2 - p1)
            n_points = int(segment_length / interval)

            for j in range(n_points):
                t = j / n_points
                point = p1 + t * (p2 - p1)

                th = dtm.get_height_at(point[0], point[1])
                sh = dsm.get_height_at(point[0], point[1])

                stations.append(cumulative + j * interval)
                terrain_heights.append(th if th is not None else np.nan)
                surface_heights.append(sh if sh is not None else np.nan)

            cumulative += segment_length

        return {
            'stations': np.array(stations),
            'terrain_heights': np.array(terrain_heights),
            'surface_heights': np.array(surface_heights),
            'total_length': cumulative,
            'min_height': float(np.nanmin(terrain_heights)),
            'max_height': float(np.nanmax(terrain_heights)),
            'elevation_change': float(np.nanmax(terrain_heights) - np.nanmin(terrain_heights))
        }

    def calculate_chm(self, dtm: TerrainModel, dsm: TerrainModel) -> TerrainModel:
        """
        Oblicza CHM (Canopy Height Model) - wysokosc roslinnosci

        CHM = DSM - DTM

        Args:
            dtm: model terenu
            dsm: model powierzchni

        Returns:
            TerrainModel z CHM
        """
        chm_heights = dsm.heights - dtm.heights
        chm_heights = np.maximum(chm_heights, 0)  # Nie moze byc ujemne

        return TerrainModel(
            grid_x=dtm.grid_x,
            grid_y=dtm.grid_y,
            heights=chm_heights,
            resolution=dtm.resolution,
            bounds=dtm.bounds,
            model_type='CHM'
        )
