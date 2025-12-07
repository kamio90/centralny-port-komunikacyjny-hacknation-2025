"""
Height Above Ground (HAG) Normalization

Computes the height of each point above the local ground surface.
This is the foundation for vegetation stratification and building detection.

Algorithm:
1. Take ground points (from CSF)
2. Create a 2D grid of ground elevations (DTM)
3. For each non-ground point, interpolate ground height at XY position
4. HAG = point_z - interpolated_ground_z

Performance optimizations:
- Grid-based interpolation instead of per-point KNN (O(N) vs O(N*k*logN))
- Vectorized operations
- Handles 100M+ points efficiently
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import time

logger = logging.getLogger(__name__)


class HeightAboveGround:
    """
    Compute Height Above Ground (HAG) for point cloud

    Usage:
        hag_calc = HeightAboveGround(coords, ground_mask)
        hag = hag_calc.compute()

        # Classify by HAG
        low_veg = (hag > 0) & (hag < 0.5)
        medium_veg = (hag >= 0.5) & (hag < 3.0)
        high_veg = hag >= 3.0
    """

    def __init__(
        self,
        coords: np.ndarray,
        ground_mask: np.ndarray,
        grid_resolution: float = 1.0,
        method: str = 'grid'
    ):
        """
        Args:
            coords: (N, 3) Point coordinates XYZ
            ground_mask: (N,) Boolean mask for ground points
            grid_resolution: Resolution of ground grid in meters (for 'grid' method)
            method: 'grid' (fast, approximate) or 'knn' (slower, precise)
        """
        self.coords = coords
        self.ground_mask = ground_mask
        self.n_points = len(coords)
        self.grid_resolution = grid_resolution
        self.method = method

        self.ground_coords = coords[ground_mask]
        self.n_ground = len(self.ground_coords)

        logger.info(f"HeightAboveGround: {self.n_points:,} points")
        logger.info(f"  Ground points: {self.n_ground:,}")
        logger.info(f"  Method: {method}")
        logger.info(f"  Grid resolution: {grid_resolution}m")

    def compute(self) -> np.ndarray:
        """
        Compute HAG for all points

        Returns:
            (N,) HAG values in meters
        """
        if self.n_ground < 10:
            logger.warning("Not enough ground points for HAG computation!")
            # Return Z values directly (assume flat ground at min Z)
            return self.coords[:, 2] - self.coords[:, 2].min()

        if self.method == 'grid':
            return self._compute_grid_method()
        else:
            return self._compute_knn_method()

    def _compute_grid_method(self) -> np.ndarray:
        """
        Fast grid-based HAG computation

        Creates a 2D grid of ground elevations, then looks up/interpolates
        for each point. O(N) complexity.
        """
        logger.info("Computing HAG using grid method...")
        logger.info(f"  Processing {self.n_points:,} points with {self.n_ground:,} ground references")
        start_time = time.time()

        # Get bounds
        x_min, x_max = self.coords[:, 0].min(), self.coords[:, 0].max()
        y_min, y_max = self.coords[:, 1].min(), self.coords[:, 1].max()

        # Create grid
        n_cols = int(np.ceil((x_max - x_min) / self.grid_resolution)) + 1
        n_rows = int(np.ceil((y_max - y_min) / self.grid_resolution)) + 1

        logger.info(f"  Grid size: {n_cols} x {n_rows} = {n_cols * n_rows:,} cells")

        # Initialize grid with NaN
        ground_grid = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

        # Fill grid with ground elevations (use minimum in each cell for true ground)
        ground_x = self.ground_coords[:, 0]
        ground_y = self.ground_coords[:, 1]
        ground_z = self.ground_coords[:, 2]

        # Compute grid indices for ground points
        col_idx = np.floor((ground_x - x_min) / self.grid_resolution).astype(np.int32)
        row_idx = np.floor((ground_y - y_min) / self.grid_resolution).astype(np.int32)

        # Clip to valid range
        col_idx = np.clip(col_idx, 0, n_cols - 1)
        row_idx = np.clip(row_idx, 0, n_rows - 1)

        # Fill grid - use minimum Z in each cell (OPTIMIZED with np.minimum.at)
        # Create linear index
        linear_idx = row_idx * n_cols + col_idx

        logger.info(f"  Filling grid with {self.n_ground:,} ground points...")
        fill_start = time.time()

        # Use numpy's ufunc.at for fast minimum computation - O(N) vectorized
        ground_grid_flat = ground_grid.ravel()

        # Initialize with inf for minimum operation
        ground_grid_flat[:] = np.inf

        # Find minimum Z per cell using np.minimum.at (much faster than loop!)
        np.minimum.at(ground_grid_flat, linear_idx, ground_z)

        # Convert inf back to nan
        ground_grid_flat[ground_grid_flat == np.inf] = np.nan
        ground_grid = ground_grid_flat.reshape(n_rows, n_cols)

        fill_elapsed = time.time() - fill_start
        logger.info(f"  Grid filled in {fill_elapsed:.2f}s: {np.sum(~np.isnan(ground_grid)):,} cells with data")

        # Fill gaps using interpolation (simple dilation-like approach)
        ground_grid = self._fill_grid_gaps(ground_grid)

        # Compute HAG for all points
        logger.info(f"  Computing HAG for {self.n_points:,} points...")
        hag_start = time.time()

        all_col_idx = np.floor((self.coords[:, 0] - x_min) / self.grid_resolution).astype(np.int32)
        all_row_idx = np.floor((self.coords[:, 1] - y_min) / self.grid_resolution).astype(np.int32)

        all_col_idx = np.clip(all_col_idx, 0, n_cols - 1)
        all_row_idx = np.clip(all_row_idx, 0, n_rows - 1)

        # Lookup ground elevation for each point
        ground_elevation = ground_grid[all_row_idx, all_col_idx]

        # HAG = point_z - ground_z
        hag = self.coords[:, 2] - ground_elevation

        hag_elapsed = time.time() - hag_start
        logger.info(f"  HAG lookup completed in {hag_elapsed:.2f}s")

        # Handle NaN (should be rare after gap filling)
        nan_mask = np.isnan(hag)
        if nan_mask.sum() > 0:
            logger.warning(f"  {nan_mask.sum():,} points with undefined HAG (using 0)")
            hag[nan_mask] = 0

        elapsed = time.time() - start_time
        logger.info(f"HAG computed in {elapsed:.2f}s")
        logger.info(f"  HAG range: {hag.min():.2f}m to {hag.max():.2f}m")

        return hag

    def _fill_grid_gaps(self, grid: np.ndarray, max_iterations: int = 10) -> np.ndarray:
        """
        Fill NaN gaps in grid using neighbor averaging

        Simple iterative gap filling - each NaN cell gets average of non-NaN neighbors.
        """
        from scipy.ndimage import uniform_filter, binary_dilation, generate_binary_structure

        filled_grid = grid.copy()
        nan_mask = np.isnan(filled_grid)
        n_nan_initial = nan_mask.sum()

        if n_nan_initial == 0:
            return filled_grid

        logger.info(f"  Filling {n_nan_initial:,} grid gaps...")

        # Use scipy's uniform_filter for fast gap filling
        for iteration in range(max_iterations):
            nan_mask = np.isnan(filled_grid)
            n_nan = nan_mask.sum()

            if n_nan == 0:
                break

            # Create temporary grid with zeros for NaN
            temp_grid = np.where(nan_mask, 0, filled_grid)
            valid_mask = (~nan_mask).astype(np.float32)

            # Apply uniform filter (3x3 average)
            sum_values = uniform_filter(temp_grid, size=3, mode='constant', cval=0)
            sum_valid = uniform_filter(valid_mask, size=3, mode='constant', cval=0)

            # Compute average where we have neighbors
            with np.errstate(invalid='ignore', divide='ignore'):
                avg_values = sum_values / sum_valid

            # Fill NaN cells that now have valid neighbors
            fill_mask = nan_mask & (sum_valid > 0)
            filled_grid[fill_mask] = avg_values[fill_mask]

        n_nan_final = np.isnan(filled_grid).sum()
        logger.info(f"  Gap filling: {n_nan_initial - n_nan_final:,} cells filled, {n_nan_final:,} remain")

        # For any remaining NaN, use global minimum
        if n_nan_final > 0:
            global_min = np.nanmin(filled_grid)
            filled_grid[np.isnan(filled_grid)] = global_min

        return filled_grid

    def _compute_knn_method(self, k: int = 10) -> np.ndarray:
        """
        KNN-based HAG computation (slower but more precise)

        For each point, find K nearest ground points and interpolate.
        O(N * k * log(M)) where M = ground points.
        """
        logger.info(f"Computing HAG using KNN method (k={k})...")
        start_time = time.time()

        # Build KD-tree of ground points (2D - only XY)
        ground_xy = self.ground_coords[:, :2]
        ground_z = self.ground_coords[:, 2]

        kdtree = cKDTree(ground_xy)

        # Query all points
        all_xy = self.coords[:, :2]
        distances, indices = kdtree.query(all_xy, k=min(k, len(ground_xy)))

        # Handle case when k > available points
        if len(indices.shape) == 1:
            indices = indices.reshape(-1, 1)
            distances = distances.reshape(-1, 1)

        # IDW (Inverse Distance Weighting) interpolation
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = 1.0 / (distances + 1e-10)
            weights = weights / weights.sum(axis=1, keepdims=True)

        # Interpolated ground elevation
        ground_z_at_neighbors = ground_z[indices]
        interpolated_ground = np.sum(weights * ground_z_at_neighbors, axis=1)

        # HAG
        hag = self.coords[:, 2] - interpolated_ground

        elapsed = time.time() - start_time
        logger.info(f"HAG computed in {elapsed:.2f}s")
        logger.info(f"  HAG range: {hag.min():.2f}m to {hag.max():.2f}m")

        return hag

    def compute_with_stats(self) -> Tuple[np.ndarray, Dict]:
        """
        Compute HAG with detailed statistics

        Returns:
            Tuple of (hag, stats_dict)
        """
        start_time = time.time()
        hag = self.compute()
        elapsed = time.time() - start_time

        stats = {
            'n_points': self.n_points,
            'n_ground': self.n_ground,
            'hag_min': float(hag.min()),
            'hag_max': float(hag.max()),
            'hag_mean': float(hag.mean()),
            'hag_std': float(hag.std()),
            'hag_median': float(np.median(hag)),
            'below_ground_pct': float((hag < 0).sum() / self.n_points * 100),
            'processing_time': elapsed
        }

        return hag, stats


def classify_by_hag(hag: np.ndarray, thresholds: Dict[str, float] = None) -> Dict[str, np.ndarray]:
    """
    Classify points into height zones based on HAG

    Args:
        hag: (N,) HAG values
        thresholds: Custom thresholds dict (default: standard vegetation thresholds)

    Returns:
        Dict of boolean masks for each zone
    """
    if thresholds is None:
        thresholds = {
            'below_ground': 0.0,      # HAG < 0 (underground/errors)
            'ground_level': 0.15,     # HAG < 0.15m (effectively ground)
            'low_veg': 0.5,           # HAG < 0.5m (grass, low shrubs)
            'medium_veg': 3.0,        # HAG < 3m (bushes, small trees)
            'high_veg': 10.0,         # HAG < 10m (trees)
            'very_high': float('inf') # HAG >= 10m (tall trees, structures)
        }

    masks = {
        'below_ground': hag < thresholds['below_ground'],
        'ground_level': (hag >= thresholds['below_ground']) & (hag < thresholds['ground_level']),
        'low_vegetation': (hag >= thresholds['ground_level']) & (hag < thresholds['low_veg']),
        'medium_vegetation': (hag >= thresholds['low_veg']) & (hag < thresholds['medium_veg']),
        'high_vegetation': (hag >= thresholds['medium_veg']) & (hag < thresholds['high_veg']),
        'very_high': hag >= thresholds['high_veg']
    }

    return masks
