"""
Noise Detection using Statistical Outlier Removal (SOR)

Standard algorithm for detecting noise/outliers in LiDAR point clouds.
Used by: PCL, Open3D, CloudCompare

Algorithm:
1. For each point, compute mean distance to k nearest neighbors
2. Compute global mean and std of these distances
3. Points with mean_distance > global_mean + std_ratio * global_std are noise

Performance:
- O(N * k * log N) with KD-tree
- For very large clouds, use sampling for global statistics
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional
from scipy.spatial import cKDTree
import time

logger = logging.getLogger(__name__)


class NoiseDetectorSOR:
    """
    Statistical Outlier Removal for noise detection

    Usage:
        detector = NoiseDetectorSOR(coords)
        noise_mask = detector.detect()

        # Or with custom parameters
        detector = NoiseDetectorSOR(coords, k_neighbors=50, std_ratio=1.5)
        noise_mask = detector.detect()
    """

    def __init__(
        self,
        coords: np.ndarray,
        k_neighbors: int = 30,
        std_ratio: float = 2.0,
        sample_size: int = 100_000
    ):
        """
        Args:
            coords: (N, 3) Point coordinates XYZ
            k_neighbors: Number of neighbors to consider
            std_ratio: Points with mean_dist > mean + std_ratio*std are noise
            sample_size: Sample size for computing global statistics (for large clouds)
        """
        self.coords = coords
        self.n_points = len(coords)
        self.k_neighbors = min(k_neighbors, self.n_points - 1)
        self.std_ratio = std_ratio
        self.sample_size = min(sample_size, self.n_points)

        logger.info(f"NoiseDetectorSOR: {self.n_points:,} points")
        logger.info(f"  k_neighbors: {self.k_neighbors}")
        logger.info(f"  std_ratio: {self.std_ratio}")

    def detect(self) -> np.ndarray:
        """
        Detect noise points using SOR

        Returns:
            (N,) boolean mask - True for noise points
        """
        logger.info("Running Statistical Outlier Removal...")
        start_time = time.time()

        # Build KD-tree
        kdtree = cKDTree(self.coords)

        # For large clouds, compute statistics on a sample first
        if self.n_points > self.sample_size * 2:
            mean_threshold = self._compute_threshold_sampled(kdtree)
        else:
            mean_threshold = self._compute_threshold_full(kdtree)

        # Now classify all points
        logger.info(f"  Computing distances for all {self.n_points:,} points...")

        # Process in chunks to manage memory
        chunk_size = 500_000
        noise_mask = np.zeros(self.n_points, dtype=bool)

        for start_idx in range(0, self.n_points, chunk_size):
            end_idx = min(start_idx + chunk_size, self.n_points)
            chunk_coords = self.coords[start_idx:end_idx]

            # Query k+1 neighbors (first is the point itself)
            distances, _ = kdtree.query(chunk_coords, k=self.k_neighbors + 1)

            # Mean distance to neighbors (exclude self - distance 0)
            mean_distances = distances[:, 1:].mean(axis=1)

            # Mark as noise if above threshold
            noise_mask[start_idx:end_idx] = mean_distances > mean_threshold

            if end_idx < self.n_points:
                logger.info(f"  Processed {end_idx:,}/{self.n_points:,} points...")

        elapsed = time.time() - start_time
        n_noise = noise_mask.sum()

        logger.info(f"SOR completed in {elapsed:.2f}s")
        logger.info(f"  Noise points: {n_noise:,} ({n_noise/self.n_points*100:.2f}%)")

        return noise_mask

    def _compute_threshold_sampled(self, kdtree: cKDTree) -> float:
        """
        Compute noise threshold using sampling (for large clouds)
        """
        logger.info(f"  Computing statistics on sample of {self.sample_size:,} points...")

        # Random sample
        sample_indices = np.random.choice(self.n_points, self.sample_size, replace=False)
        sample_coords = self.coords[sample_indices]

        # Query neighbors for sample
        distances, _ = kdtree.query(sample_coords, k=self.k_neighbors + 1)
        mean_distances = distances[:, 1:].mean(axis=1)

        # Compute statistics
        global_mean = mean_distances.mean()
        global_std = mean_distances.std()
        threshold = global_mean + self.std_ratio * global_std

        logger.info(f"  Mean distance: {global_mean:.4f}m")
        logger.info(f"  Std distance: {global_std:.4f}m")
        logger.info(f"  Noise threshold: {threshold:.4f}m")

        return threshold

    def _compute_threshold_full(self, kdtree: cKDTree) -> float:
        """
        Compute noise threshold using all points
        """
        logger.info(f"  Computing statistics on all {self.n_points:,} points...")

        # Query neighbors
        distances, _ = kdtree.query(self.coords, k=self.k_neighbors + 1)
        mean_distances = distances[:, 1:].mean(axis=1)

        # Compute statistics
        global_mean = mean_distances.mean()
        global_std = mean_distances.std()
        threshold = global_mean + self.std_ratio * global_std

        logger.info(f"  Mean distance: {global_mean:.4f}m")
        logger.info(f"  Std distance: {global_std:.4f}m")
        logger.info(f"  Noise threshold: {threshold:.4f}m")

        return threshold

    def detect_with_stats(self) -> Tuple[np.ndarray, Dict]:
        """
        Detect noise with detailed statistics

        Returns:
            Tuple of (noise_mask, stats_dict)
        """
        start_time = time.time()
        noise_mask = self.detect()
        elapsed = time.time() - start_time

        noise_coords = self.coords[noise_mask]
        valid_coords = self.coords[~noise_mask]

        stats = {
            'n_points': self.n_points,
            'n_noise': int(noise_mask.sum()),
            'n_valid': int((~noise_mask).sum()),
            'noise_pct': float(noise_mask.sum() / self.n_points * 100),
            'params': {
                'k_neighbors': self.k_neighbors,
                'std_ratio': self.std_ratio
            },
            'processing_time': elapsed
        }

        # Add spatial distribution info
        if noise_mask.sum() > 0:
            stats['noise_z_mean'] = float(noise_coords[:, 2].mean())
            stats['noise_z_std'] = float(noise_coords[:, 2].std())

        return noise_mask, stats


class NoiseDetectorRadius:
    """
    Alternative noise detection based on point density within radius

    Faster than SOR for very large clouds, but less precise.
    """

    def __init__(
        self,
        coords: np.ndarray,
        radius: float = 1.0,
        min_neighbors: int = 5
    ):
        """
        Args:
            coords: (N, 3) Point coordinates XYZ
            radius: Search radius in meters
            min_neighbors: Minimum neighbors required (points with fewer = noise)
        """
        self.coords = coords
        self.n_points = len(coords)
        self.radius = radius
        self.min_neighbors = min_neighbors

        logger.info(f"NoiseDetectorRadius: {self.n_points:,} points")
        logger.info(f"  radius: {self.radius}m")
        logger.info(f"  min_neighbors: {self.min_neighbors}")

    def detect(self) -> np.ndarray:
        """
        Detect noise points based on local density

        Returns:
            (N,) boolean mask - True for noise points
        """
        logger.info("Running radius-based noise detection...")
        start_time = time.time()

        # Build KD-tree
        kdtree = cKDTree(self.coords)

        # Count neighbors within radius for each point
        # Use query_ball_point which returns variable-length lists
        noise_mask = np.zeros(self.n_points, dtype=bool)

        # Process in chunks
        chunk_size = 100_000

        for start_idx in range(0, self.n_points, chunk_size):
            end_idx = min(start_idx + chunk_size, self.n_points)
            chunk_coords = self.coords[start_idx:end_idx]

            # Query neighbors within radius
            neighbor_lists = kdtree.query_ball_point(chunk_coords, self.radius)

            # Count neighbors (subtract 1 for self)
            neighbor_counts = np.array([len(n) - 1 for n in neighbor_lists])

            # Mark as noise if too few neighbors
            noise_mask[start_idx:end_idx] = neighbor_counts < self.min_neighbors

            if end_idx < self.n_points and (end_idx // chunk_size) % 5 == 0:
                logger.info(f"  Processed {end_idx:,}/{self.n_points:,} points...")

        elapsed = time.time() - start_time
        n_noise = noise_mask.sum()

        logger.info(f"Radius noise detection completed in {elapsed:.2f}s")
        logger.info(f"  Noise points: {n_noise:,} ({n_noise/self.n_points*100:.2f}%)")

        return noise_mask
