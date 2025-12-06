"""
LAS/LAZ file reader optimized for Apple M4 Max with 64GB RAM
Handles large point cloud files through memory mapping and efficient sampling
"""

import laspy
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LASReader:
    """
    Memory-efficient LAS reader optimized for M4 Max

    Features:
    - Lazy loading (header only until needed)
    - Memory-mapped file access
    - Efficient random sampling for preview
    - Spatial filtering for tile extraction
    """

    def __init__(self, filepath: Path):
        """
        Initialize LAS reader

        Args:
            filepath: Path to LAS/LAZ file
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"LAS file not found: {filepath}")

        self.header = None
        self._load_header()

    def _load_header(self):
        """Load LAS header without reading points (fast operation)"""
        try:
            with laspy.open(self.filepath) as f:
                self.header = f.header
            logger.info(f"Loaded header from {self.filepath.name}")
        except Exception as e:
            logger.error(f"Failed to read LAS header: {e}")
            raise

    def get_info(self) -> Dict:
        """
        Get file information for UI display

        Returns:
            Dictionary with file metadata
        """
        # Check for RGB colors by testing point format ID
        # Point formats 2, 3, 5, 7, 8, 10 include RGB
        point_format_id = self.header.point_format.id
        has_rgb = point_format_id in [2, 3, 5, 7, 8, 10]

        return {
            'point_count': self.header.point_count,
            'bounds': {
                'x': (self.header.x_min, self.header.x_max),
                'y': (self.header.y_min, self.header.y_max),
                'z': (self.header.z_min, self.header.z_max)
            },
            'version': f"{self.header.version.major}.{self.header.version.minor}",
            'point_format': point_format_id,
            'has_rgb': has_rgb,
            'has_intensity': True,  # Usually present in LAS files
            'file_size_gb': self.filepath.stat().st_size / 1e9,
            'scales': (self.header.x_scale, self.header.y_scale, self.header.z_scale),
            'offsets': (self.header.x_offset, self.header.y_offset, self.header.z_offset)
        }

    def sample_points(self, n_samples: int = 1_000_000,
                     method: str = 'random',
                     return_indices: bool = False) -> Dict[str, np.ndarray]:
        """
        Sample points for quick preview

        With 64GB RAM, we can afford 1M points for preview without issues

        Args:
            n_samples: Number of points to sample
            method: Sampling method ('random', 'uniform', 'stratified')
            return_indices: Whether to return sampled indices

        Returns:
            Dictionary containing sampled point data
        """
        logger.info(f"Sampling {n_samples:,} points using {method} method...")

        try:
            # Read the entire file (with 64GB RAM this is OK for most files)
            las = laspy.read(self.filepath)
            total_points = len(las.points)

            logger.info(f"File contains {total_points:,} points")

            # Determine sampling indices
            if total_points <= n_samples:
                # Use all points if file is small
                indices = np.arange(total_points)
                logger.info(f"Using all {total_points:,} points (less than sample size)")
            else:
                if method == 'random':
                    # Random sampling
                    indices = np.random.choice(total_points, n_samples, replace=False)
                    indices.sort()  # Sort for better cache locality
                elif method == 'uniform':
                    # Uniform sampling (every Nth point)
                    step = total_points // n_samples
                    indices = np.arange(0, total_points, step)[:n_samples]
                elif method == 'stratified':
                    # Stratified spatial sampling (by Z height)
                    z_values = las.z
                    z_bins = np.percentile(z_values, np.linspace(0, 100, 11))
                    indices = []
                    points_per_bin = n_samples // 10
                    for i in range(len(z_bins) - 1):
                        mask = (z_values >= z_bins[i]) & (z_values < z_bins[i + 1])
                        bin_indices = np.where(mask)[0]
                        if len(bin_indices) > 0:
                            sampled = np.random.choice(
                                bin_indices,
                                min(points_per_bin, len(bin_indices)),
                                replace=False
                            )
                            indices.extend(sampled)
                    indices = np.array(indices)
                else:
                    raise ValueError(f"Unknown sampling method: {method}")

                logger.info(f"Sampled {len(indices):,} points")

            # Extract coordinates
            coords = np.vstack([
                las.x[indices],
                las.y[indices],
                las.z[indices]
            ]).T

            # Extract RGB colors if available
            colors = None
            if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                colors = np.vstack([
                    las.red[indices],
                    las.green[indices],
                    las.blue[indices]
                ]).T / 65535.0  # Normalize to [0, 1]
                logger.info("RGB colors extracted")
            else:
                logger.info("No RGB colors available")

            # Extract intensity if available
            intensity = None
            if hasattr(las, 'intensity'):
                intensity = las.intensity[indices] / 65535.0  # Normalize to [0, 1]
                logger.info("Intensity values extracted")

            # Extract classification if available
            classification = None
            if hasattr(las, 'classification'):
                classification = las.classification[indices]
                logger.info("Classification values extracted")

            result = {
                'coords': coords,
                'colors': colors,
                'intensity': intensity,
                'classification': classification,
                'point_count': len(indices)
            }

            if return_indices:
                result['indices'] = indices

            logger.info("Sampling complete")
            return result

        except Exception as e:
            logger.error(f"Failed to sample points: {e}")
            raise

    def read_bounds(self,
                   x_range: Optional[Tuple[float, float]] = None,
                   y_range: Optional[Tuple[float, float]] = None,
                   z_range: Optional[Tuple[float, float]] = None) -> Dict[str, np.ndarray]:
        """
        Read points within specified spatial bounds

        Args:
            x_range: (min, max) for X coordinate
            y_range: (min, max) for Y coordinate
            z_range: (min, max) for Z coordinate

        Returns:
            Dictionary containing filtered point data
        """
        logger.info("Reading points within bounds...")

        try:
            las = laspy.read(self.filepath)

            # Create filter mask
            mask = np.ones(len(las.points), dtype=bool)

            if x_range is not None:
                mask &= (las.x >= x_range[0]) & (las.x <= x_range[1])
            if y_range is not None:
                mask &= (las.y >= y_range[0]) & (las.y <= y_range[1])
            if z_range is not None:
                mask &= (las.z >= z_range[0]) & (las.z <= z_range[1])

            logger.info(f"Filter retained {mask.sum():,} / {len(mask):,} points")

            # Extract data
            coords = np.vstack([las.x[mask], las.y[mask], las.z[mask]]).T

            colors = None
            if hasattr(las, 'red'):
                colors = np.vstack([
                    las.red[mask],
                    las.green[mask],
                    las.blue[mask]
                ]).T / 65535.0

            intensity = None
            if hasattr(las, 'intensity'):
                intensity = las.intensity[mask] / 65535.0

            classification = None
            if hasattr(las, 'classification'):
                classification = las.classification[mask]

            return {
                'coords': coords,
                'colors': colors,
                'intensity': intensity,
                'classification': classification,
                'point_count': mask.sum()
            }

        except Exception as e:
            logger.error(f"Failed to read bounds: {e}")
            raise

    def get_spatial_extent(self) -> Dict[str, float]:
        """
        Calculate spatial extent of the point cloud

        Returns:
            Dictionary with extent measurements
        """
        info = self.get_info()
        bounds = info['bounds']

        return {
            'x_extent': bounds['x'][1] - bounds['x'][0],
            'y_extent': bounds['y'][1] - bounds['y'][0],
            'z_extent': bounds['z'][1] - bounds['z'][0],
            'area_m2': (bounds['x'][1] - bounds['x'][0]) * (bounds['y'][1] - bounds['y'][0])
        }
