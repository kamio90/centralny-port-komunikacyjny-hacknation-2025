"""
Vegetation Classification using HAG (Height Above Ground) and NDVI

Standard approach for LiDAR vegetation classification:
1. Use HAG to stratify by height (low/medium/high)
2. Use NDVI from RGB colors to identify green vegetation
3. Combine both for accurate classification

ASPRS Classes:
- 3: Low Vegetation (< 0.5m)
- 4: Medium Vegetation (0.5m - 3m)
- 5: High Vegetation (> 3m)
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class VegetationParams:
    """Parameters for vegetation classification"""

    # Height thresholds (meters above ground)
    low_veg_max_height: float = 0.5      # Low vegetation: 0 - 0.5m
    medium_veg_max_height: float = 3.0   # Medium vegetation: 0.5 - 3m
    # High vegetation: > 3m

    # NDVI thresholds (Normalized Difference Vegetation Index)
    # NDVI = (Green - Red) / (Green + Red) for RGB images
    # For true NIR data: NDVI = (NIR - Red) / (NIR + Red)
    ndvi_threshold: float = 0.05  # Minimum NDVI to be considered vegetation

    # Color-based vegetation detection (when NDVI is not reliable)
    green_ratio_threshold: float = 0.33  # G / (R + G + B) minimum

    # Require color confirmation (True = use NDVI/color, False = HAG only)
    require_color: bool = True


class VegetationClassifier:
    """
    Vegetation classifier using HAG and optionally color (NDVI)

    Usage:
        classifier = VegetationClassifier(hag, colors)
        low_mask, medium_mask, high_mask = classifier.classify()

        # Apply to classification array
        classification[low_mask] = 3      # Low vegetation
        classification[medium_mask] = 4   # Medium vegetation
        classification[high_mask] = 5     # High vegetation
    """

    def __init__(
        self,
        hag: np.ndarray,
        colors: Optional[np.ndarray] = None,
        params: Optional[VegetationParams] = None,
        ground_mask: Optional[np.ndarray] = None
    ):
        """
        Args:
            hag: (N,) Height Above Ground values
            colors: (N, 3) RGB colors [0-1] or None
            params: Classification parameters
            ground_mask: (N,) Boolean mask for ground points (excluded from vegetation)
        """
        self.hag = hag
        self.colors = colors
        self.n_points = len(hag)
        self.params = params or VegetationParams()
        self.ground_mask = ground_mask

        # Pre-compute NDVI if colors available
        if colors is not None:
            self.ndvi = self._compute_ndvi(colors)
            self.green_ratio = self._compute_green_ratio(colors)
        else:
            self.ndvi = None
            self.green_ratio = None

        logger.info(f"VegetationClassifier: {self.n_points:,} points")
        logger.info(f"  Colors available: {colors is not None}")
        logger.info(f"  HAG range: {hag.min():.2f}m to {hag.max():.2f}m")

    def _compute_ndvi(self, colors: np.ndarray) -> np.ndarray:
        """
        Compute NDVI from RGB colors

        For RGB data (not NIR), we use: NDVI = (G - R) / (G + R + epsilon)
        This approximates vegetation greenness.
        """
        red = colors[:, 0]
        green = colors[:, 1]

        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (green - red) / (green + red + 1e-8)

        return ndvi

    def _compute_green_ratio(self, colors: np.ndarray) -> np.ndarray:
        """
        Compute green ratio: G / (R + G + B)

        Alternative to NDVI for vegetation detection.
        """
        total = colors.sum(axis=1) + 1e-8
        green_ratio = colors[:, 1] / total
        return green_ratio

    def classify(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Classify vegetation into low/medium/high categories

        Returns:
            Tuple of (low_mask, medium_mask, high_mask)
        """
        logger.info("Classifying vegetation...")
        start_time = time.time()

        # Height-based masks
        low_height = (self.hag > 0) & (self.hag <= self.params.low_veg_max_height)
        medium_height = (self.hag > self.params.low_veg_max_height) & \
                       (self.hag <= self.params.medium_veg_max_height)
        high_height = self.hag > self.params.medium_veg_max_height

        # Vegetation color mask
        if self.colors is not None and self.params.require_color:
            # Use NDVI and green ratio for vegetation detection
            veg_color = (self.ndvi > self.params.ndvi_threshold) | \
                       (self.green_ratio > self.params.green_ratio_threshold)
            logger.info(f"  Color-based vegetation: {veg_color.sum():,} points")
        else:
            # No color data - use height only
            veg_color = np.ones(self.n_points, dtype=bool)
            logger.info("  No color data - using HAG only")

        # Exclude ground points if mask provided
        if self.ground_mask is not None:
            not_ground = ~self.ground_mask
        else:
            not_ground = np.ones(self.n_points, dtype=bool)

        # Combine height and color criteria
        low_mask = low_height & veg_color & not_ground
        medium_mask = medium_height & veg_color & not_ground
        high_mask = high_height & veg_color & not_ground

        elapsed = time.time() - start_time

        logger.info(f"Vegetation classification completed in {elapsed:.2f}s")
        logger.info(f"  Low vegetation (class 3): {low_mask.sum():,} ({low_mask.sum()/self.n_points*100:.1f}%)")
        logger.info(f"  Medium vegetation (class 4): {medium_mask.sum():,} ({medium_mask.sum()/self.n_points*100:.1f}%)")
        logger.info(f"  High vegetation (class 5): {high_mask.sum():,} ({high_mask.sum()/self.n_points*100:.1f}%)")

        return low_mask, medium_mask, high_mask

    def classify_with_stats(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Classify with detailed statistics

        Returns:
            Tuple of (masks_dict, stats_dict)
        """
        start_time = time.time()
        low_mask, medium_mask, high_mask = self.classify()
        elapsed = time.time() - start_time

        masks = {
            'low_vegetation': low_mask,
            'medium_vegetation': medium_mask,
            'high_vegetation': high_mask
        }

        stats = {
            'n_points': self.n_points,
            'low_vegetation': {
                'count': int(low_mask.sum()),
                'percentage': float(low_mask.sum() / self.n_points * 100),
                'hag_mean': float(self.hag[low_mask].mean()) if low_mask.sum() > 0 else 0
            },
            'medium_vegetation': {
                'count': int(medium_mask.sum()),
                'percentage': float(medium_mask.sum() / self.n_points * 100),
                'hag_mean': float(self.hag[medium_mask].mean()) if medium_mask.sum() > 0 else 0
            },
            'high_vegetation': {
                'count': int(high_mask.sum()),
                'percentage': float(high_mask.sum() / self.n_points * 100),
                'hag_mean': float(self.hag[high_mask].mean()) if high_mask.sum() > 0 else 0
            },
            'total_vegetation': {
                'count': int(low_mask.sum() + medium_mask.sum() + high_mask.sum()),
                'percentage': float((low_mask.sum() + medium_mask.sum() + high_mask.sum()) / self.n_points * 100)
            },
            'params': {
                'low_veg_max_height': self.params.low_veg_max_height,
                'medium_veg_max_height': self.params.medium_veg_max_height,
                'ndvi_threshold': self.params.ndvi_threshold,
                'require_color': self.params.require_color
            },
            'processing_time': elapsed
        }

        return masks, stats


def compute_ndvi(colors: np.ndarray) -> np.ndarray:
    """
    Standalone NDVI computation

    Args:
        colors: (N, 3) RGB colors [0-1]

    Returns:
        (N,) NDVI values [-1, 1]
    """
    red = colors[:, 0]
    green = colors[:, 1]

    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (green - red) / (green + red + 1e-8)

    return ndvi


def compute_excess_green(colors: np.ndarray) -> np.ndarray:
    """
    Compute Excess Green Index (ExG)

    ExG = 2*G - R - B

    Useful for distinguishing vegetation from other objects.
    """
    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
    exg = 2 * g - r - b
    return exg
