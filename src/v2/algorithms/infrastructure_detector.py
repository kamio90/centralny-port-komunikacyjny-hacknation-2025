"""
Infrastructure Detection for Linear and Vertical Structures

ZOPTYMALIZOWANA WERSJA - O(N) zamiast O(N*k*log(N))

Zamiast PCA dla każdego punktu, używamy:
1. Grid-based feature computation - oblicz cechy per komórka siatki
2. Lookup O(1) - przypisz cechy do punktów przez grid index
3. Proste reguły dla infrastruktury bazowane na HAG + geometrii grid

Detects infrastructure elements:
- Power lines: High, sparse in XY grid cells with large Z variance
- Poles: Vertical clusters (small XY extent, large Z extent)
- Rails: Low HAG, high intensity, linear in XY
- Roads: Low HAG, flat (low Z variance), gray color

ASPRS/CPK Classes:
- 18: Rails
- 19: Power Lines
- 20: Traction Poles
- 30: Road Surface
- 34: Street Lighting
- 64: Telecom Poles
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class InfrastructureParams:
    """Parameters for infrastructure detection - TUNED FOR CPK DATA"""

    # Grid resolution for feature computation
    grid_resolution: float = 0.5  # meters

    # Power lines (high, large Z variance in cell, SPARSE)
    powerline_min_hag: float = 6.0       # Lowered slightly
    powerline_min_z_variance: float = 0.3
    powerline_max_density: float = 50    # Very sparse (reduced from 100)

    # Poles (vertical clusters) - MUCH STRICTER
    pole_min_hag: float = 3.0            # Raised from 2.0
    pole_max_hag: float = 20.0           # Lowered from 30.0
    pole_min_z_range: float = 5.0        # Raised from 3.0 - poles are tall!
    pole_max_xy_extent: float = 0.5      # Stricter - poles are narrow
    pole_max_density: float = 200        # NEW: poles are sparse, not dense like trees

    # Rails (low, high intensity, linear) - MUCH STRICTER
    rail_max_hag: float = 0.15           # Reduced from 0.3 - rails are ON ground
    rail_min_intensity: float = 0.5      # Raised from 0.3 - metal is very reflective
    rail_max_width: float = 0.2          # NEW: rails are narrow

    # Roads (flat, low HAG) - keep similar but stricter
    road_max_hag: float = 0.15           # Reduced from 0.2
    road_max_z_variance: float = 0.05    # Reduced from 0.1 - roads are FLAT
    road_min_density: float = 100        # Raised from 50 - roads are dense

    # Curbs (krawężniki) - linear, slightly elevated, high intensity (concrete)
    curb_min_hag: float = 0.05           # Curbs are slightly above road
    curb_max_hag: float = 0.25           # But not too high
    curb_min_intensity: float = 0.4      # Concrete is reflective
    curb_max_z_variance: float = 0.02    # Very flat top surface

    # Signs (znaki drogowe) - high intensity (reflective), vertical, small
    sign_min_hag: float = 1.5            # Signs are elevated
    sign_max_hag: float = 4.0            # But not too high
    sign_min_intensity: float = 0.7      # Signs are VERY reflective

    # Barriers (bariery drogowe) - linear, medium height, metal
    barrier_min_hag: float = 0.3         # Barriers are above ground
    barrier_max_hag: float = 1.2         # Guardrail height
    barrier_min_intensity: float = 0.5   # Metal/reflective surface

    # Platforms (perony kolejowe) - flat, elevated, dense
    platform_min_hag: float = 0.3        # Platforms are elevated
    platform_max_hag: float = 1.5        # Standard platform height
    platform_max_z_variance: float = 0.03  # Very flat surface
    platform_min_density: float = 150    # Dense surface

    # Water bodies (zbiorniki wodne) - very low, flat, low intensity (water absorbs LiDAR)
    water_max_hag: float = 0.1           # Water is at ground level or below
    water_max_z_variance: float = 0.01   # Water is VERY flat
    water_max_intensity: float = 0.15    # Water absorbs LiDAR - low return
    water_min_density: float = 50        # Some points from water surface

    # Bridges (mosty) - elevated flat surfaces over gaps
    bridge_min_hag: float = 2.0          # Bridges are elevated
    bridge_max_hag: float = 30.0         # Can be high
    bridge_max_z_variance: float = 0.05  # Flat deck surface
    bridge_min_density: float = 100      # Solid surface
    bridge_min_length: float = 10.0      # Minimum bridge length (meters)


class InfrastructureDetectorFast:
    """
    FAST Infrastructure detector using grid-based features

    Complexity: O(N) instead of O(N*k*log(N))

    Usage:
        detector = InfrastructureDetectorFast(coords, hag, intensity)
        results = detector.detect()
    """

    def __init__(
        self,
        coords: np.ndarray,
        hag: np.ndarray,
        intensity: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        params: Optional[InfrastructureParams] = None,
        ground_mask: Optional[np.ndarray] = None,
        vegetation_mask: Optional[np.ndarray] = None
    ):
        self.coords = coords
        self.hag = hag
        self.intensity = intensity
        self.colors = colors
        self.n_points = len(coords)
        self.params = params or InfrastructureParams()
        self.ground_mask = ground_mask if ground_mask is not None else np.zeros(self.n_points, dtype=bool)
        self.vegetation_mask = vegetation_mask if vegetation_mask is not None else np.zeros(self.n_points, dtype=bool)

        logger.info(f"InfrastructureDetectorFast: {self.n_points:,} points")

    def detect(self) -> Dict[str, np.ndarray]:
        """
        Detect infrastructure elements using grid-based approach

        Returns:
            Dict with boolean masks for each infrastructure type
        """
        logger.info("Detecting infrastructure (fast grid-based)...")
        start_time = time.time()

        # Krok 1: Oblicz grid features - O(N)
        grid_features = self._compute_grid_features()

        # Krok 2: Klasyfikuj na podstawie grid features + HAG
        results = {
            'powerlines': self._detect_powerlines(grid_features),
            'poles': self._detect_poles(grid_features),
            'rails': self._detect_rails(grid_features),
            'roads': self._detect_roads(grid_features),
            'curbs': self._detect_curbs(grid_features),
            'signs': self._detect_signs(grid_features),
            'barriers': self._detect_barriers(grid_features),
            'platforms': self._detect_platforms(grid_features),
            'water': self._detect_water(grid_features),
            'bridges': self._detect_bridges(grid_features)
        }

        elapsed = time.time() - start_time
        logger.info(f"Infrastructure detection completed in {elapsed:.2f}s")

        for name, mask in results.items():
            if mask.sum() > 0:
                logger.info(f"  {name}: {mask.sum():,} points ({mask.sum()/self.n_points*100:.2f}%)")

        return results

    def _compute_grid_features(self) -> Dict[str, np.ndarray]:
        """
        Compute features per grid cell, then map back to points - O(N)

        Features per cell:
        - z_variance: variance of Z in cell
        - z_range: max(Z) - min(Z) in cell
        - density: points per cell
        - xy_extent: spread in XY
        """
        logger.info("  Computing grid-based features...")
        p = self.params

        # Get bounds
        x_min, x_max = self.coords[:, 0].min(), self.coords[:, 0].max()
        y_min, y_max = self.coords[:, 1].min(), self.coords[:, 1].max()

        # Create grid indices
        n_cols = max(1, int(np.ceil((x_max - x_min) / p.grid_resolution)))
        n_rows = max(1, int(np.ceil((y_max - y_min) / p.grid_resolution)))

        col_idx = np.floor((self.coords[:, 0] - x_min) / p.grid_resolution).astype(np.int32)
        row_idx = np.floor((self.coords[:, 1] - y_min) / p.grid_resolution).astype(np.int32)

        col_idx = np.clip(col_idx, 0, n_cols - 1)
        row_idx = np.clip(row_idx, 0, n_rows - 1)

        # Linear cell index
        cell_idx = row_idx * n_cols + col_idx
        n_cells = n_rows * n_cols

        logger.info(f"    Grid: {n_cols} x {n_rows} = {n_cells:,} cells")

        # Compute statistics per cell using bincount tricks
        # Count points per cell
        cell_counts = np.bincount(cell_idx, minlength=n_cells)

        # Sum of Z per cell
        z_sum = np.bincount(cell_idx, weights=self.coords[:, 2], minlength=n_cells)

        # Sum of Z^2 per cell (for variance)
        z_sq_sum = np.bincount(cell_idx, weights=self.coords[:, 2]**2, minlength=n_cells)

        # Mean Z per cell
        with np.errstate(divide='ignore', invalid='ignore'):
            z_mean = np.where(cell_counts > 0, z_sum / cell_counts, 0)

        # Variance = E[X^2] - E[X]^2
        with np.errstate(divide='ignore', invalid='ignore'):
            z_variance = np.where(
                cell_counts > 1,
                z_sq_sum / cell_counts - z_mean**2,
                0
            )
        z_variance = np.maximum(z_variance, 0)  # Numerical stability

        # Min/Max Z per cell (for z_range)
        z_min_cell = np.full(n_cells, np.inf)
        z_max_cell = np.full(n_cells, -np.inf)

        # Use numpy's ufunc.at for min/max
        np.minimum.at(z_min_cell, cell_idx, self.coords[:, 2])
        np.maximum.at(z_max_cell, cell_idx, self.coords[:, 2])

        z_range = np.where(cell_counts > 0, z_max_cell - z_min_cell, 0)

        # Density (points per cell)
        cell_area = p.grid_resolution ** 2
        density = cell_counts / cell_area

        # Map features back to points - O(N)
        features = {
            'z_variance': z_variance[cell_idx],
            'z_range': z_range[cell_idx],
            'density': density[cell_idx],
            'cell_count': cell_counts[cell_idx]
        }

        logger.info(f"    Grid features computed")
        return features

    def _detect_powerlines(self, grid_features: Dict) -> np.ndarray:
        """Detect power lines: high, sparse, variable Z"""
        p = self.params

        hag_mask = self.hag >= p.powerline_min_hag
        z_var_mask = grid_features['z_variance'] >= p.powerline_min_z_variance
        sparse_mask = grid_features['density'] <= p.powerline_max_density

        # Exclude ground and vegetation
        exclude_mask = ~self.ground_mask & ~self.vegetation_mask

        mask = hag_mask & z_var_mask & sparse_mask & exclude_mask

        logger.info(f"  Power lines: HAG>={p.powerline_min_hag}m, z_var>={p.powerline_min_z_variance}")
        return mask

    def _detect_poles(self, grid_features: Dict) -> np.ndarray:
        """Detect vertical poles: height in range, large Z span, LOW DENSITY (not trees!)"""
        p = self.params

        hag_mask = (self.hag >= p.pole_min_hag) & (self.hag <= p.pole_max_hag)
        z_range_mask = grid_features['z_range'] >= p.pole_min_z_range

        # KEY: Poles are SPARSE - trees are dense!
        # Low density in cell = likely pole, high density = likely tree
        sparse_mask = grid_features['density'] <= p.pole_max_density

        # Exclude ground and vegetation
        exclude_mask = ~self.ground_mask & ~self.vegetation_mask

        mask = hag_mask & z_range_mask & sparse_mask & exclude_mask

        logger.info(f"  Poles: HAG {p.pole_min_hag}-{p.pole_max_hag}m, z_range>={p.pole_min_z_range}m, density<={p.pole_max_density}")
        return mask

    def _detect_rails(self, grid_features: Dict) -> np.ndarray:
        """Detect rails: low HAG, VERY high intensity (metal), NOT ground class"""
        p = self.params

        # Rails are slightly above ground but very close (on ballast)
        hag_mask = (self.hag >= 0.0) & (self.hag <= p.rail_max_hag)

        # Rails have VERY high reflectivity (metal) - key distinguisher from ground
        if self.intensity is not None:
            intensity_mask = self.intensity >= p.rail_min_intensity
        else:
            # Without intensity data, we can't reliably detect rails
            logger.warning("  Rails: No intensity data - skipping rail detection")
            return np.zeros(self.n_points, dtype=bool)

        # Rails are in linear patterns - use z_variance to filter
        # Rail areas have very low variance (flat metal surfaces)
        low_variance_mask = grid_features['z_variance'] <= 0.02

        # Exclude vegetation and GROUND (rails are separate from classified ground)
        exclude_mask = ~self.vegetation_mask & ~self.ground_mask

        # Gray/silver color for metal (if colors available)
        if self.colors is not None:
            # Metal is grayish (low saturation, medium-high brightness)
            color_mean = self.colors.mean(axis=1)
            color_std = self.colors.std(axis=1)
            gray_mask = (color_std < 0.1) & (color_mean > 0.2) & (color_mean < 0.7)
        else:
            gray_mask = np.ones(self.n_points, dtype=bool)

        mask = hag_mask & intensity_mask & low_variance_mask & exclude_mask & gray_mask

        logger.info(f"  Rails: HAG<={p.rail_max_hag}m, intensity>={p.rail_min_intensity}, z_var<=0.02, gray color")
        return mask

    def _detect_roads(self, grid_features: Dict) -> np.ndarray:
        """Detect roads: low HAG, flat (low Z variance), dense"""
        p = self.params

        hag_mask = self.hag <= p.road_max_hag
        flat_mask = grid_features['z_variance'] <= p.road_max_z_variance
        dense_mask = grid_features['density'] >= p.road_min_density

        # Exclude vegetation
        exclude_mask = ~self.vegetation_mask

        # Roads are typically gray (low saturation) if colors available
        if self.colors is not None:
            saturation = self.colors.std(axis=1)
            gray_mask = saturation < 0.15
        else:
            gray_mask = np.ones(self.n_points, dtype=bool)

        mask = hag_mask & flat_mask & dense_mask & exclude_mask & gray_mask

        logger.info(f"  Roads: HAG<={p.road_max_hag}m, z_var<={p.road_max_z_variance}")
        return mask

    def _detect_curbs(self, grid_features: Dict) -> np.ndarray:
        """Detect curbs (krawężniki): slightly elevated, linear, concrete"""
        p = self.params

        # Curbs are slightly above road level
        hag_mask = (self.hag >= p.curb_min_hag) & (self.hag <= p.curb_max_hag)

        # Flat top surface
        flat_mask = grid_features['z_variance'] <= p.curb_max_z_variance

        # Concrete is reflective
        if self.intensity is not None:
            intensity_mask = self.intensity >= p.curb_min_intensity
        else:
            intensity_mask = np.ones(self.n_points, dtype=bool)

        # Exclude vegetation and ground
        exclude_mask = ~self.vegetation_mask & ~self.ground_mask

        # Light gray color (concrete)
        if self.colors is not None:
            brightness = self.colors.mean(axis=1)
            saturation = self.colors.std(axis=1)
            gray_mask = (saturation < 0.1) & (brightness > 0.4) & (brightness < 0.8)
        else:
            gray_mask = np.ones(self.n_points, dtype=bool)

        mask = hag_mask & flat_mask & intensity_mask & exclude_mask & gray_mask

        logger.info(f"  Curbs: HAG {p.curb_min_hag}-{p.curb_max_hag}m, z_var<={p.curb_max_z_variance}")
        return mask

    def _detect_signs(self, grid_features: Dict) -> np.ndarray:
        """Detect road signs (znaki): elevated, VERY reflective, small"""
        p = self.params

        # Signs are elevated
        hag_mask = (self.hag >= p.sign_min_hag) & (self.hag <= p.sign_max_hag)

        # Signs are EXTREMELY reflective (retroreflective material)
        if self.intensity is not None:
            intensity_mask = self.intensity >= p.sign_min_intensity
        else:
            # Without intensity, we can't reliably detect signs
            logger.warning("  Signs: No intensity data - skipping sign detection")
            return np.zeros(self.n_points, dtype=bool)

        # Signs are sparse (small objects)
        sparse_mask = grid_features['density'] <= 100

        # Exclude vegetation and ground
        exclude_mask = ~self.vegetation_mask & ~self.ground_mask

        mask = hag_mask & intensity_mask & sparse_mask & exclude_mask

        logger.info(f"  Signs: HAG {p.sign_min_hag}-{p.sign_max_hag}m, intensity>={p.sign_min_intensity}")
        return mask

    def _detect_barriers(self, grid_features: Dict) -> np.ndarray:
        """Detect road barriers (bariery): linear, medium height, metal"""
        p = self.params

        # Barriers are at guardrail height
        hag_mask = (self.hag >= p.barrier_min_hag) & (self.hag <= p.barrier_max_hag)

        # Metal is reflective
        if self.intensity is not None:
            intensity_mask = self.intensity >= p.barrier_min_intensity
        else:
            intensity_mask = np.ones(self.n_points, dtype=bool)

        # Barriers are moderately sparse (not dense surfaces)
        density_mask = (grid_features['density'] >= 20) & (grid_features['density'] <= 300)

        # Exclude vegetation and ground
        exclude_mask = ~self.vegetation_mask & ~self.ground_mask

        # Metal color (grayish)
        if self.colors is not None:
            saturation = self.colors.std(axis=1)
            gray_mask = saturation < 0.12
        else:
            gray_mask = np.ones(self.n_points, dtype=bool)

        mask = hag_mask & intensity_mask & density_mask & exclude_mask & gray_mask

        logger.info(f"  Barriers: HAG {p.barrier_min_hag}-{p.barrier_max_hag}m, intensity>={p.barrier_min_intensity}")
        return mask

    def _detect_platforms(self, grid_features: Dict) -> np.ndarray:
        """Detect railway platforms (perony): flat, elevated, dense"""
        p = self.params

        # Platforms are elevated
        hag_mask = (self.hag >= p.platform_min_hag) & (self.hag <= p.platform_max_hag)

        # Very flat surface
        flat_mask = grid_features['z_variance'] <= p.platform_max_z_variance

        # Dense surface (concrete/paving)
        dense_mask = grid_features['density'] >= p.platform_min_density

        # Exclude vegetation
        exclude_mask = ~self.vegetation_mask

        # Concrete/paving color (light gray)
        if self.colors is not None:
            brightness = self.colors.mean(axis=1)
            saturation = self.colors.std(axis=1)
            gray_mask = (saturation < 0.12) & (brightness > 0.3) & (brightness < 0.8)
        else:
            gray_mask = np.ones(self.n_points, dtype=bool)

        mask = hag_mask & flat_mask & dense_mask & exclude_mask & gray_mask

        logger.info(f"  Platforms: HAG {p.platform_min_hag}-{p.platform_max_hag}m, z_var<={p.platform_max_z_variance}")
        return mask

    def _detect_water(self, grid_features: Dict) -> np.ndarray:
        """Detect water bodies: very flat, low intensity (water absorbs LiDAR)"""
        p = self.params

        # Water is at ground level or slightly below
        hag_mask = self.hag <= p.water_max_hag

        # Water is EXTREMELY flat
        flat_mask = grid_features['z_variance'] <= p.water_max_z_variance

        # Water absorbs LiDAR - very low intensity return
        if self.intensity is not None:
            # Water has LOW intensity (absorbs LiDAR)
            low_intensity_mask = self.intensity <= p.water_max_intensity
        else:
            # Without intensity, water detection is less reliable
            # Use color if available - water is often dark blue/green
            low_intensity_mask = np.ones(self.n_points, dtype=bool)

        # Some minimum density (scattered returns from water surface)
        density_mask = grid_features['density'] >= p.water_min_density

        # Exclude vegetation
        exclude_mask = ~self.vegetation_mask

        # Water color: dark blue, dark green, or dark (low brightness, blue-ish)
        if self.colors is not None:
            r, g, b = self.colors[:, 0], self.colors[:, 1], self.colors[:, 2]
            brightness = self.colors.mean(axis=1)

            # Water is dark and bluish
            dark_mask = brightness < 0.4
            blue_ish_mask = (b > r) | (g > r)  # More blue/green than red
            color_mask = dark_mask & blue_ish_mask
        else:
            color_mask = np.ones(self.n_points, dtype=bool)

        mask = hag_mask & flat_mask & low_intensity_mask & density_mask & exclude_mask & color_mask

        logger.info(f"  Water: HAG<={p.water_max_hag}m, z_var<={p.water_max_z_variance}, intensity<={p.water_max_intensity}")
        return mask

    def _detect_bridges(self, grid_features: Dict) -> np.ndarray:
        """Detect bridges: elevated flat surfaces spanning gaps"""
        p = self.params

        # Bridges are elevated
        hag_mask = (self.hag >= p.bridge_min_hag) & (self.hag <= p.bridge_max_hag)

        # Bridge deck is flat
        flat_mask = grid_features['z_variance'] <= p.bridge_max_z_variance

        # Solid surface
        dense_mask = grid_features['density'] >= p.bridge_min_density

        # Exclude vegetation
        exclude_mask = ~self.vegetation_mask

        # Bridge materials: concrete (gray) or steel (metallic gray)
        if self.colors is not None:
            brightness = self.colors.mean(axis=1)
            saturation = self.colors.std(axis=1)
            # Gray colors (low saturation, medium-high brightness)
            gray_mask = (saturation < 0.12) & (brightness > 0.3) & (brightness < 0.8)
        else:
            gray_mask = np.ones(self.n_points, dtype=bool)

        # Medium-high intensity (concrete/steel)
        if self.intensity is not None:
            intensity_mask = self.intensity >= 0.3
        else:
            intensity_mask = np.ones(self.n_points, dtype=bool)

        mask = hag_mask & flat_mask & dense_mask & exclude_mask & gray_mask & intensity_mask

        logger.info(f"  Bridges: HAG {p.bridge_min_hag}-{p.bridge_max_hag}m, z_var<={p.bridge_max_z_variance}")
        return mask


# Alias for backwards compatibility
InfrastructureDetector = InfrastructureDetectorFast
