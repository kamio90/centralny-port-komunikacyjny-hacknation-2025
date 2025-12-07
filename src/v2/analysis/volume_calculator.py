"""
Volume Calculator - Obliczanie objetosci dla robot ziemnych

Oblicza:
- Objetosc wykopu (cut)
- Objetosc nasypu (fill)
- Bilans mas ziemnych
- Objetosc roslinnosci do usuniecia
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .terrain_analysis import TerrainAnalyzer, TerrainModel, CrossSection

logger = logging.getLogger(__name__)


@dataclass
class VolumeResult:
    """Wynik obliczen objetosciowych"""
    cut_volume_m3: float  # Wykop [m3]
    fill_volume_m3: float  # Nasyp [m3]
    net_volume_m3: float  # Bilans (cut - fill)
    vegetation_volume_m3: float  # Roslinnosc do usuniecia [m3]
    area_m2: float  # Powierzchnia terenu [m2]
    avg_cut_depth_m: float  # Srednia glebokosc wykopu
    avg_fill_height_m: float  # Srednia wysokosc nasypu
    max_cut_depth_m: float  # Max glebokosc wykopu
    max_fill_height_m: float  # Max wysokosc nasypu


class VolumeCalculator:
    """
    Kalkulator objetosci dla robot ziemnych

    Oblicza objetosci wykopu i nasypu na podstawie:
    - Istniejacego terenu (DTM)
    - Projektowanej niwelety

    Usage:
        calc = VolumeCalculator(coords, classification)
        result = calc.calculate_corridor_volume(
            axis_points,
            design_heights,
            corridor_width=20.0
        )
    """

    def __init__(
        self,
        coords: np.ndarray,
        classification: np.ndarray
    ):
        self.coords = coords
        self.classification = classification
        self.terrain = TerrainAnalyzer(coords, classification)

    def calculate_corridor_volume(
        self,
        axis_points: np.ndarray,
        design_heights: np.ndarray,
        corridor_width: float = 20.0,
        side_slopes: float = 2.0  # 1:2 (pion:poziom)
    ) -> VolumeResult:
        """
        Oblicza objetosc wzdluz korytarza trasy

        Args:
            axis_points: (M, 2) punkty osi trasy
            design_heights: (M,) wysokosci projektowe na osi
            corridor_width: szerokosc korony [m]
            side_slopes: nachylenie skarpy (1:n)

        Returns:
            VolumeResult z objetosciami
        """
        logger.info(f"Calculating corridor volume: width={corridor_width}m")

        # Generuj DTM i DSM
        dtm = self.terrain.generate_dtm(resolution=1.0)
        dsm = self.terrain.generate_dsm(resolution=1.0)

        # Ekstrahuj przekroje
        sections = self.terrain.extract_cross_sections(
            axis_points,
            width=corridor_width + 20,  # Dodatkowy margines na skarpy
            interval=5.0,
            resolution=0.5
        )

        total_cut = 0.0
        total_fill = 0.0
        total_veg = 0.0
        total_area = 0.0

        cut_depths = []
        fill_heights = []

        # Interpoluj wysokosci projektowe
        if len(design_heights) != len(axis_points):
            design_heights = np.interp(
                np.linspace(0, 1, len(axis_points)),
                np.linspace(0, 1, len(design_heights)),
                design_heights
            )

        # Oblicz dla kazdego przekroju
        for i, section in enumerate(sections):
            # Interpoluj wysokosc projektowa dla tej stacji
            t = section.station / sections[-1].station if sections[-1].station > 0 else 0
            idx = min(int(t * (len(design_heights) - 1)), len(design_heights) - 1)
            design_h = design_heights[idx]

            # Oblicz objetosc przekroju
            section_result = self._calculate_section_volume(
                section,
                design_h,
                corridor_width,
                side_slopes
            )

            # Akumuluj (trapezowa metoda)
            if i > 0:
                dx = section.station - sections[i-1].station
                total_cut += section_result['cut_area'] * dx
                total_fill += section_result['fill_area'] * dx
                total_veg += section_result['veg_area'] * dx
                total_area += corridor_width * dx

            if section_result['cut_area'] > 0:
                cut_depths.append(section_result['max_cut'])
            if section_result['fill_area'] > 0:
                fill_heights.append(section_result['max_fill'])

        # Wyniki
        avg_cut = np.mean(cut_depths) if cut_depths else 0
        avg_fill = np.mean(fill_heights) if fill_heights else 0
        max_cut = max(cut_depths) if cut_depths else 0
        max_fill = max(fill_heights) if fill_heights else 0

        result = VolumeResult(
            cut_volume_m3=total_cut,
            fill_volume_m3=total_fill,
            net_volume_m3=total_cut - total_fill,
            vegetation_volume_m3=total_veg,
            area_m2=total_area,
            avg_cut_depth_m=avg_cut,
            avg_fill_height_m=avg_fill,
            max_cut_depth_m=max_cut,
            max_fill_height_m=max_fill
        )

        logger.info(f"Volume calculation complete: cut={total_cut:.0f}m3, fill={total_fill:.0f}m3")
        return result

    def _calculate_section_volume(
        self,
        section: CrossSection,
        design_height: float,
        corridor_width: float,
        side_slopes: float
    ) -> Dict:
        """Oblicza objetosc pojedynczego przekroju"""
        half_width = corridor_width / 2

        cut_area = 0.0
        fill_area = 0.0
        veg_area = 0.0
        max_cut = 0.0
        max_fill = 0.0

        dx = section.distances[1] - section.distances[0] if len(section.distances) > 1 else 1.0

        for i, d in enumerate(section.distances):
            if abs(d) > half_width + 20:  # Poza strefą wpływu
                continue

            terrain_h = section.terrain_heights[i]
            surface_h = section.surface_heights[i]

            if np.isnan(terrain_h):
                continue

            # Wysokosc projektowa w tym punkcie (uwzgledniaj skarpe)
            if abs(d) <= half_width:
                # Na koronie
                proj_h = design_height
            else:
                # Na skarpie
                dist_from_edge = abs(d) - half_width
                if terrain_h < design_height:
                    # Nasyp - skarpa w dol
                    proj_h = design_height - dist_from_edge / side_slopes
                else:
                    # Wykop - skarpa w gore
                    proj_h = design_height + dist_from_edge / side_slopes

                # Ogranicz do terenu
                proj_h = max(min(proj_h, terrain_h + 10), terrain_h - 10)

            # Roznica
            diff = terrain_h - proj_h

            if diff > 0:
                # Wykop
                cut_area += diff * dx
                max_cut = max(max_cut, diff)
            else:
                # Nasyp
                fill_area += abs(diff) * dx
                max_fill = max(max_fill, abs(diff))

            # Roslinnosc
            if not np.isnan(surface_h):
                veg_height = surface_h - terrain_h
                if veg_height > 0.5:  # Roslinnosc > 0.5m
                    veg_area += veg_height * dx

        return {
            'cut_area': cut_area,
            'fill_area': fill_area,
            'veg_area': veg_area,
            'max_cut': max_cut,
            'max_fill': max_fill
        }

    def calculate_area_volume(
        self,
        polygon: np.ndarray,
        design_height: float
    ) -> VolumeResult:
        """
        Oblicza objetosc dla obszaru (np. plac budowy)

        Args:
            polygon: (M, 2) wierzcholki wielokata
            design_height: docelowa wysokosc terenu

        Returns:
            VolumeResult
        """
        logger.info("Calculating area volume...")

        dtm = self.terrain.generate_dtm(resolution=1.0)
        dsm = self.terrain.generate_dsm(resolution=1.0)

        # Znajdz punkty wewnatrz wielokata
        from matplotlib.path import Path
        poly_path = Path(polygon)

        # Siatka
        x = dtm.grid_x
        y = dtm.grid_y
        xx, yy = np.meshgrid(x, y)
        points = np.vstack([xx.ravel(), yy.ravel()]).T

        inside = poly_path.contains_points(points).reshape(xx.shape)

        # Oblicz
        cut_volume = 0.0
        fill_volume = 0.0
        veg_volume = 0.0

        resolution = dtm.resolution
        cell_area = resolution ** 2

        for i in range(inside.shape[0]):
            for j in range(inside.shape[1]):
                if not inside[i, j]:
                    continue

                terrain_h = dtm.heights[i, j]
                surface_h = dsm.heights[i, j]

                if np.isnan(terrain_h):
                    continue

                diff = terrain_h - design_height

                if diff > 0:
                    cut_volume += diff * cell_area
                else:
                    fill_volume += abs(diff) * cell_area

                if not np.isnan(surface_h):
                    veg_h = surface_h - terrain_h
                    if veg_h > 0.5:
                        veg_volume += veg_h * cell_area

        area = inside.sum() * cell_area

        return VolumeResult(
            cut_volume_m3=cut_volume,
            fill_volume_m3=fill_volume,
            net_volume_m3=cut_volume - fill_volume,
            vegetation_volume_m3=veg_volume,
            area_m2=area,
            avg_cut_depth_m=cut_volume / area if area > 0 and cut_volume > 0 else 0,
            avg_fill_height_m=fill_volume / area if area > 0 and fill_volume > 0 else 0,
            max_cut_depth_m=0,  # TODO
            max_fill_height_m=0  # TODO
        )

    def generate_report(self, result: VolumeResult) -> Dict:
        """Generuje raport z obliczen objetosciowych"""
        return {
            'summary': {
                'cut_volume_m3': round(result.cut_volume_m3, 1),
                'fill_volume_m3': round(result.fill_volume_m3, 1),
                'net_volume_m3': round(result.net_volume_m3, 1),
                'balance': 'NADMIAR' if result.net_volume_m3 > 0 else 'NIEDOBOR',
                'balance_volume_m3': round(abs(result.net_volume_m3), 1)
            },
            'vegetation': {
                'volume_m3': round(result.vegetation_volume_m3, 1),
                'note': 'Objetosc roslinnosci do usuniecia'
            },
            'area': {
                'total_m2': round(result.area_m2, 1),
                'total_ha': round(result.area_m2 / 10000, 2)
            },
            'depths': {
                'avg_cut_m': round(result.avg_cut_depth_m, 2),
                'max_cut_m': round(result.max_cut_depth_m, 2),
                'avg_fill_m': round(result.avg_fill_height_m, 2),
                'max_fill_m': round(result.max_fill_height_m, 2)
            },
            'costs_estimate': {
                'note': 'Szacunkowe koszty (orientacyjne)',
                'cut_cost_pln': round(result.cut_volume_m3 * 25, 0),  # 25 PLN/m3
                'fill_cost_pln': round(result.fill_volume_m3 * 35, 0),  # 35 PLN/m3
                'veg_removal_pln': round(result.vegetation_volume_m3 * 15, 0),  # 15 PLN/m3
                'total_estimate_pln': round(
                    result.cut_volume_m3 * 25 +
                    result.fill_volume_m3 * 35 +
                    result.vegetation_volume_m3 * 15, 0
                )
            }
        }
