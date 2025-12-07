"""
Ground Classification using CSF (Cloth Simulation Filter)

CSF is the industry-standard algorithm for LiDAR ground classification.
Used by: LAStools, PDAL, CloudCompare, ArcGIS

Algorithm:
1. Invert point cloud (flip Z axis)
2. Drop virtual cloth from above
3. Cloth settles on highest points (which are ground after inversion)
4. Points close to cloth = ground, far from cloth = non-ground

Reference:
Zhang et al. (2016) "An Easy-to-Use Airborne LiDAR Data Filtering Method
Based on Cloth Simulation" Remote Sensing, 8(6), 501.

Performance: O(N log N) - handles 100M+ points efficiently
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import time

try:
    import CSF
    CSF_AVAILABLE = True
except ImportError:
    CSF_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CSFParams:
    """Parameters for CSF algorithm"""

    # Cloth resolution - size of cloth grid cells in meters
    # Smaller = more detail but slower
    # Recommended: 0.5m for urban, 1.0m for forests, 2.0m for flat terrain
    cloth_resolution: float = 0.5

    # Rigidness of cloth (1=soft, 2=medium, 3=hard)
    # 1: For steep slopes and complex terrain
    # 2: Default, works for most cases
    # 3: For flat terrain
    rigidness: int = 2

    # Classification threshold - max distance from cloth to be ground (meters)
    # Smaller = stricter ground classification
    class_threshold: float = 0.5

    # Simulation parameters
    time_step: float = 0.65
    iterations: int = 500

    # Slope post-processing
    slope_smooth: bool = True


class GroundClassifierCSF:
    """
    Ground point classifier using Cloth Simulation Filter

    Usage:
        classifier = GroundClassifierCSF(coords)
        ground_mask = classifier.classify()

        # Or with custom parameters
        classifier = GroundClassifierCSF(coords, params=CSFParams(cloth_resolution=1.0))
        ground_mask = classifier.classify()
    """

    def __init__(
        self,
        coords: np.ndarray,
        params: Optional[CSFParams] = None
    ):
        """
        Args:
            coords: (N, 3) Point coordinates XYZ
            params: CSF parameters (default: CSFParams())
        """
        if not CSF_AVAILABLE:
            raise ImportError(
                "CSF not available. Install with: pip install cloth-simulation-filter"
            )

        self.coords = coords
        self.n_points = len(coords)
        self.params = params or CSFParams()

        logger.info(f"GroundClassifierCSF: {self.n_points:,} points")
        logger.info(f"  cloth_resolution: {self.params.cloth_resolution}m")
        logger.info(f"  rigidness: {self.params.rigidness}")
        logger.info(f"  class_threshold: {self.params.class_threshold}m")

    def classify(self) -> np.ndarray:
        """
        Classify ground points using CSF

        Returns:
            (N,) boolean mask - True for ground points
        """
        logger.info("Running CSF ground classification...")
        start_time = time.time()

        # Initialize CSF
        csf = CSF.CSF()

        # Set parameters
        csf.params.cloth_resolution = self.params.cloth_resolution
        csf.params.rigidness = self.params.rigidness
        csf.params.class_threshold = self.params.class_threshold
        csf.params.time_step = self.params.time_step
        csf.params.interations = self.params.iterations  # Note: typo in CSF API
        csf.params.bSloopSmooth = self.params.slope_smooth

        # Set point cloud (CSF expects numpy array or list)
        csf.setPointCloud(self.coords)

        # Run filtering
        ground_indices = CSF.VecInt()
        non_ground_indices = CSF.VecInt()

        csf.do_filtering(ground_indices, non_ground_indices)

        # Convert to numpy arrays
        ground_idx = np.array(list(ground_indices), dtype=np.int64)

        # Create boolean mask
        ground_mask = np.zeros(self.n_points, dtype=bool)
        if len(ground_idx) > 0:
            ground_mask[ground_idx] = True

        elapsed = time.time() - start_time
        n_ground = ground_mask.sum()

        logger.info(f"CSF completed in {elapsed:.2f}s")
        logger.info(f"  Ground points: {n_ground:,} ({n_ground/self.n_points*100:.1f}%)")
        logger.info(f"  Non-ground: {self.n_points - n_ground:,} ({(self.n_points - n_ground)/self.n_points*100:.1f}%)")

        return ground_mask

    def classify_with_stats(self) -> Tuple[np.ndarray, Dict]:
        """
        Classify with detailed statistics

        Returns:
            Tuple of (ground_mask, stats_dict)
        """
        start_time = time.time()
        ground_mask = self.classify()
        elapsed = time.time() - start_time

        # Compute statistics
        ground_z = self.coords[ground_mask, 2]
        non_ground_z = self.coords[~ground_mask, 2]

        stats = {
            'n_points': self.n_points,
            'n_ground': int(ground_mask.sum()),
            'n_non_ground': int((~ground_mask).sum()),
            'ground_pct': float(ground_mask.sum() / self.n_points * 100),
            'ground_z_min': float(ground_z.min()) if len(ground_z) > 0 else None,
            'ground_z_max': float(ground_z.max()) if len(ground_z) > 0 else None,
            'ground_z_mean': float(ground_z.mean()) if len(ground_z) > 0 else None,
            'processing_time': elapsed,
            'params': {
                'cloth_resolution': self.params.cloth_resolution,
                'rigidness': self.params.rigidness,
                'class_threshold': self.params.class_threshold
            }
        }

        return ground_mask, stats


def auto_tune_params(coords: np.ndarray) -> CSFParams:
    """
    Automatically tune CSF parameters based on point cloud characteristics

    Args:
        coords: (N, 3) Point coordinates

    Returns:
        Optimized CSFParams
    """
    # Compute basic statistics
    z_range = coords[:, 2].max() - coords[:, 2].min()
    xy_extent = max(
        coords[:, 0].max() - coords[:, 0].min(),
        coords[:, 1].max() - coords[:, 1].min()
    )
    density = len(coords) / (xy_extent ** 2) if xy_extent > 0 else 1

    # Estimate terrain type
    params = CSFParams()

    # Cloth resolution based on density and extent
    if density > 100:  # High density (urban, detailed scan)
        params.cloth_resolution = 0.5
    elif density > 20:  # Medium density
        params.cloth_resolution = 1.0
    else:  # Low density
        params.cloth_resolution = 2.0

    # Rigidness based on Z range (terrain complexity)
    if z_range > 50:  # Complex terrain with significant height variation
        params.rigidness = 1  # Soft cloth for complex terrain
    elif z_range > 20:
        params.rigidness = 2  # Medium
    else:
        params.rigidness = 3  # Hard cloth for flat terrain

    # Classification threshold
    if density > 50:
        params.class_threshold = 0.3  # Stricter for high density
    else:
        params.class_threshold = 0.5  # Default

    logger.info(f"Auto-tuned CSF params:")
    logger.info(f"  cloth_resolution: {params.cloth_resolution}m (density: {density:.1f} pts/m2)")
    logger.info(f"  rigidness: {params.rigidness} (z_range: {z_range:.1f}m)")
    logger.info(f"  class_threshold: {params.class_threshold}m")

    return params
