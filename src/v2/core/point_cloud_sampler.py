"""
Point Cloud Sampler - Generalizacja chmur punktów dla wizualizacji

Inteligentne zmniejszanie liczby punktów z zachowaniem:
- Rozkładu przestrzennego
- Charakterystyki geometrycznej
- Reprezentatywności danych

Optymalizowane dla wizualizacji w przeglądarce (50k-100k punktów)
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SamplingResult:
    """Wynik operacji samplingu"""
    coords: np.ndarray
    colors: Optional[np.ndarray]
    intensity: Optional[np.ndarray]
    indices: np.ndarray
    stats: Dict


class PointCloudSampler:
    """
    Sampler chmur punktów dla wizualizacji 3D

    Metody:
    - voxel_downsample(): Preferowana - zachowuje rozkład przestrzenny
    - random_sample(): Fallback - szybka ale mniej reprezentatywna
    - adaptive_sample(): Auto-wybór metody na podstawie danych
    """

    def __init__(
        self,
        coords: np.ndarray,
        colors: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None
    ):
        self.coords = coords
        self.colors = colors
        self.intensity = intensity
        self.n_points = len(coords)

        # Oblicz granice
        self.bounds = {
            'x': (float(coords[:, 0].min()), float(coords[:, 0].max())),
            'y': (float(coords[:, 1].min()), float(coords[:, 1].max())),
            'z': (float(coords[:, 2].min()), float(coords[:, 2].max()))
        }

        # Oblicz wymiary
        self.x_range = self.bounds['x'][1] - self.bounds['x'][0]
        self.y_range = self.bounds['y'][1] - self.bounds['y'][0]
        self.z_range = self.bounds['z'][1] - self.bounds['z'][0]
        self.volume = max(self.x_range * self.y_range * self.z_range, 1.0)
        self.area_xy = max(self.x_range * self.y_range, 1.0)

        logger.info(f"PointCloudSampler: {self.n_points:,} punktów")
        logger.info(f"  Wymiary: {self.x_range:.1f} x {self.y_range:.1f} x {self.z_range:.1f} m")

    def voxel_downsample(
        self,
        target_points: int = 75_000,
        min_voxel_size: float = 0.1,
        max_voxel_size: float = 50.0
    ) -> SamplingResult:
        """
        Próbkowanie metodą siatki wokseli (Open3D)
        ZOPTYMALIZOWANE: nie szuka oryginalnych indeksów - bezpośrednio używa danych z Open3D
        """
        logger.info(f"Voxel downsample: {self.n_points:,} -> ~{target_points:,}")

        if self.n_points <= target_points:
            logger.info("  Liczba punktów <= target, pomijam sampling")
            return self._create_result(
                indices=np.arange(self.n_points),
                method='none',
                voxel_size=0.0
            )

        try:
            import open3d as o3d

            # Oblicz optymalny rozmiar woksela
            voxel_size = self._calculate_voxel_size(target_points)
            voxel_size = np.clip(voxel_size, min_voxel_size, max_voxel_size)

            logger.info(f"  Obliczony rozmiar woksela: {voxel_size:.3f} m")

            # Utwórz chmurę Open3D
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.coords)

            if self.colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(self.colors)

            # Wykonaj voxel downsampling
            pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

            # Pobierz wyniki BEZPOŚREDNIO z Open3D (bez szukania indeksów!)
            coords_down = np.asarray(pcd_down.points)
            colors_down = np.asarray(pcd_down.colors) if self.colors is not None else None
            actual_points = len(coords_down)

            logger.info(f"  Wynik: {actual_points:,} punktów (ratio: {actual_points/self.n_points*100:.2f}%)")

            # Zwróć bezpośrednio dane z Open3D (dummy indices)
            return self._create_result_direct(
                coords=coords_down,
                colors=colors_down,
                intensity=None,  # Intensity nie jest zachowywane w voxel
                method='voxel',
                voxel_size=voxel_size
            )

        except ImportError:
            logger.warning("Open3D niedostępne, używam random sample")
            return self.random_sample(target_points)
        except Exception as e:
            logger.warning(f"Voxel sampling failed: {e}, fallback to random")
            return self.random_sample(target_points)

    def random_sample(self, target_points: int = 75_000) -> SamplingResult:
        """
        Proste losowe próbkowanie (fallback)
        """
        logger.info(f"Random sample: {self.n_points:,} -> {target_points:,}")

        if self.n_points <= target_points:
            return self._create_result(
                indices=np.arange(self.n_points),
                method='none',
                voxel_size=0.0
            )

        indices = np.random.choice(self.n_points, target_points, replace=False)
        indices = np.sort(indices)

        return self._create_result(
            indices=indices,
            method='random',
            voxel_size=0.0
        )

    def grid_sample(self, target_points: int = 75_000) -> SamplingResult:
        """
        Próbkowanie metodą siatki 3D (bez Open3D)
        Zachowuje rozkład przestrzenny lepiej niż random
        """
        logger.info(f"Grid sample: {self.n_points:,} -> ~{target_points:,}")

        if self.n_points <= target_points:
            return self._create_result(
                indices=np.arange(self.n_points),
                method='none',
                voxel_size=0.0
            )

        # Oblicz rozmiar komórki
        cell_size = self._calculate_voxel_size(target_points) * 0.8

        # Normalizuj współrzędne do siatki
        grid_x = ((self.coords[:, 0] - self.bounds['x'][0]) / cell_size).astype(np.int32)
        grid_y = ((self.coords[:, 1] - self.bounds['y'][0]) / cell_size).astype(np.int32)
        grid_z = ((self.coords[:, 2] - self.bounds['z'][0]) / cell_size).astype(np.int32)

        # Unikalne komórki - bierzemy pierwszy punkt z każdej
        grid_keys = grid_x * 1000000 + grid_y * 1000 + grid_z
        _, unique_indices = np.unique(grid_keys, return_index=True)

        # Jeśli za dużo punktów - losowo wybierz
        if len(unique_indices) > target_points:
            selected = np.random.choice(len(unique_indices), target_points, replace=False)
            unique_indices = unique_indices[selected]

        indices = np.sort(unique_indices)

        logger.info(f"  Wynik: {len(indices):,} punktów")

        return self._create_result(
            indices=indices,
            method='grid',
            voxel_size=cell_size
        )

    def adaptive_sample(
        self,
        target_points: int = 75_000,
        prefer_quality: bool = True
    ) -> SamplingResult:
        """
        Adaptacyjne próbkowanie - wybiera metodę automatycznie

        Logika dla szybkości:
        - < 5M: voxel (najlepsza jakość)
        - 5M-50M: random (szybko i wystarczająco dobrze)
        - > 50M: random z pre-samplingiem (ultra szybko)
        """
        if self.n_points <= target_points:
            return self._create_result(
                indices=np.arange(self.n_points),
                method='none',
                voxel_size=0.0
            )

        # Dla BARDZO dużych chmur (>50M) - random jest jedyną sensowną opcją
        if self.n_points > 50_000_000:
            logger.info(f"  Bardzo duża chmura ({self.n_points/1e6:.0f}M) - używam random sample")
            return self.random_sample(target_points)

        # Dla dużych chmur (5M-50M) - random jest szybszy i wystarczający
        if self.n_points > 5_000_000:
            logger.info(f"  Duża chmura ({self.n_points/1e6:.1f}M) - używam random sample")
            return self.random_sample(target_points)

        # Dla średnich chmur (<5M) - voxel daje lepszą jakość
        if prefer_quality:
            try:
                return self.voxel_downsample(target_points)
            except Exception:
                return self.random_sample(target_points)
        else:
            return self.random_sample(target_points)

    def _calculate_voxel_size(self, target_points: int) -> float:
        """Oblicza rozmiar woksela dla docelowej liczby punktów"""
        voxel_size = (self.volume / target_points) ** (1/3)
        voxel_size *= 0.7  # Korekta empiryczna

        density_xy = self.n_points / self.area_xy
        voxel_from_density = np.sqrt(1.0 / density_xy) * np.sqrt(target_points / self.n_points)

        return max(voxel_size, voxel_from_density, 0.1)

    def _create_result(
        self,
        indices: np.ndarray,
        method: str,
        voxel_size: float
    ) -> SamplingResult:
        """Tworzy obiekt SamplingResult z indeksów"""

        coords = self.coords[indices]
        colors = self.colors[indices] if self.colors is not None else None
        intensity = self.intensity[indices] if self.intensity is not None else None

        stats = {
            'original_points': self.n_points,
            'sampled_points': len(indices),
            'sampling_ratio': len(indices) / self.n_points if self.n_points > 0 else 1.0,
            'method': method,
            'voxel_size': voxel_size,
            'bounds': self.bounds,
            'dimensions': {
                'x': self.x_range,
                'y': self.y_range,
                'z': self.z_range
            },
            'area_m2': self.area_xy,
            'density_original': self.n_points / self.area_xy,
            'density_sampled': len(indices) / self.area_xy
        }

        return SamplingResult(
            coords=coords,
            colors=colors,
            intensity=intensity,
            indices=indices,
            stats=stats
        )

    def _create_result_direct(
        self,
        coords: np.ndarray,
        colors: Optional[np.ndarray],
        intensity: Optional[np.ndarray],
        method: str,
        voxel_size: float
    ) -> SamplingResult:
        """Tworzy obiekt SamplingResult bezpośrednio z danych (bez indeksów)"""

        stats = {
            'original_points': self.n_points,
            'sampled_points': len(coords),
            'sampling_ratio': len(coords) / self.n_points if self.n_points > 0 else 1.0,
            'method': method,
            'voxel_size': voxel_size,
            'bounds': self.bounds,
            'dimensions': {
                'x': self.x_range,
                'y': self.y_range,
                'z': self.z_range
            },
            'area_m2': self.area_xy,
            'density_original': self.n_points / self.area_xy,
            'density_sampled': len(coords) / self.area_xy
        }

        return SamplingResult(
            coords=coords,
            colors=colors,
            intensity=intensity,
            indices=np.arange(len(coords)),  # Dummy indices
            stats=stats
        )
