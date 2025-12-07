"""
Building Detection using RANSAC Plane Fitting

Detects building structures (roofs, walls) by finding planar surfaces.

Algorithm:
1. Filter points by HAG (above ground, not too high)
2. Exclude vegetation (green points)
3. Apply RANSAC to find dominant planes
4. Classify horizontal planes as roofs, vertical as walls

ASPRS Classes:
- 6: Building (general)
- 40: External Walls (BIM)
- 41: Roofs (BIM)
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import time

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BuildingParams:
    """Parameters for building detection"""

    # Height filters (HAG)
    min_height: float = 2.5      # Minimum height above ground for buildings (raised from 2.0)
    max_height: float = 50.0     # Maximum height (lowered from 100.0)

    # RANSAC parameters
    distance_threshold: float = 0.10  # Max distance from plane to be inlier (tighter)
    ransac_n: int = 3                 # Points to estimate plane
    num_iterations: int = 500         # RANSAC iterations (reduced)

    # Minimum points for a plane to be considered
    min_plane_points: int = 10000     # Raised from 1000 - buildings are large!

    # Normal thresholds for roof/wall classification
    # Roof: normal mostly vertical (Z component > threshold)
    roof_normal_z_threshold: float = 0.8  # Stricter (was 0.7)
    # Wall: normal mostly horizontal (Z component < threshold)
    wall_normal_z_threshold: float = 0.2  # Stricter (was 0.3)

    # Maximum number of planes to extract
    max_planes: int = 15              # Reduced from 50

    # Exclude vegetation (by excess green index)
    exclude_vegetation: bool = True
    ndvi_threshold: float = 0.12      # Raised from 0.05 - better vegetation exclusion

    # Additional filters
    min_planarity: float = 0.6        # Minimum planarity for building surfaces
    max_color_variance: float = 0.15  # Buildings have uniform color


class BuildingDetector:
    """
    Building detector using RANSAC plane fitting

    Usage:
        detector = BuildingDetector(coords, hag, colors)
        building_mask, roof_mask, wall_mask = detector.detect()

        # Apply to classification
        classification[building_mask & ~roof_mask & ~wall_mask] = 6  # Building
        classification[roof_mask] = 41  # Roof
        classification[wall_mask] = 40  # Wall
    """

    def __init__(
        self,
        coords: np.ndarray,
        hag: np.ndarray,
        colors: Optional[np.ndarray] = None,
        params: Optional[BuildingParams] = None
    ):
        """
        Args:
            coords: (N, 3) Point coordinates XYZ
            hag: (N,) Height Above Ground
            colors: (N, 3) RGB colors [0-1] or None
            params: Detection parameters
        """
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D required for building detection. Install with: pip install open3d")

        self.coords = coords
        self.hag = hag
        self.colors = colors
        self.n_points = len(coords)
        self.params = params or BuildingParams()

        logger.info(f"BuildingDetector: {self.n_points:,} points")
        logger.info(f"  HAG filter: {self.params.min_height}m - {self.params.max_height}m")

    def detect(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect buildings, roofs, and walls

        Returns:
            Tuple of (building_mask, roof_mask, wall_mask)
        """
        logger.info("Detecting buildings using RANSAC...")
        start_time = time.time()

        # Create masks
        building_mask = np.zeros(self.n_points, dtype=bool)
        roof_mask = np.zeros(self.n_points, dtype=bool)
        wall_mask = np.zeros(self.n_points, dtype=bool)

        # Pre-filter by HAG
        height_mask = (self.hag >= self.params.min_height) & (self.hag <= self.params.max_height)
        n_height = height_mask.sum()
        logger.info(f"  Height filter ({self.params.min_height}-{self.params.max_height}m): {n_height:,} points")

        # Exclude vegetation if colors available
        if self.colors is not None and self.params.exclude_vegetation:
            # Excess Green Index (proxy for NDVI without NIR)
            egi = (self.colors[:, 1] - self.colors[:, 0]) / \
                  (self.colors[:, 1] + self.colors[:, 0] + 1e-8)
            vegetation_mask = egi > self.params.ndvi_threshold

            # Also check for high green content (absolute)
            green_dominant = (self.colors[:, 1] > self.colors[:, 0]) & \
                            (self.colors[:, 1] > self.colors[:, 2])
            vegetation_mask = vegetation_mask | (green_dominant & (self.colors[:, 1] > 0.3))

            # Buildings typically have low color variance (uniform surfaces)
            color_variance = self.colors.std(axis=1)
            high_variance_mask = color_variance > self.params.max_color_variance

            # Combine exclusions
            exclude_mask = vegetation_mask | high_variance_mask
            candidate_mask = height_mask & ~exclude_mask

            n_veg_excluded = vegetation_mask.sum()
            n_var_excluded = high_variance_mask.sum()
            logger.info(f"  Excluded vegetation: {n_veg_excluded:,}, high color variance: {n_var_excluded:,}")
        else:
            candidate_mask = height_mask

        n_candidates = candidate_mask.sum()
        logger.info(f"  Building candidates after filtering: {n_candidates:,} points ({n_candidates/self.n_points*100:.1f}%)")

        if n_candidates < self.params.min_plane_points:
            logger.warning("  Not enough candidate points for building detection")
            return building_mask, roof_mask, wall_mask

        # Get candidate points
        candidate_indices = np.where(candidate_mask)[0]
        candidate_coords = self.coords[candidate_mask]

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(candidate_coords)

        # Iterative RANSAC to find multiple planes
        remaining_pcd = pcd
        remaining_indices = candidate_indices.copy()
        planes_found = 0
        total_roof_points = 0
        total_wall_points = 0
        ransac_start = time.time()

        logger.info(f"  Starting RANSAC plane extraction (max {self.params.max_planes} planes)...")

        while len(remaining_pcd.points) > self.params.min_plane_points and \
              planes_found < self.params.max_planes:

            # Segment plane using RANSAC
            try:
                plane_model, inlier_indices = remaining_pcd.segment_plane(
                    distance_threshold=self.params.distance_threshold,
                    ransac_n=self.params.ransac_n,
                    num_iterations=self.params.num_iterations
                )
            except Exception as e:
                logger.warning(f"  RANSAC failed: {e}")
                break

            if len(inlier_indices) < self.params.min_plane_points:
                logger.info(f"  Stopping: remaining plane too small ({len(inlier_indices):,} < {self.params.min_plane_points:,})")
                break

            # Get plane normal
            a, b, c, d = plane_model
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)

            # Classify plane as roof or wall based on normal
            z_component = abs(normal[2])

            # Map inlier indices back to original point cloud
            original_inlier_indices = remaining_indices[inlier_indices]

            if z_component > self.params.roof_normal_z_threshold:
                # Horizontal plane = roof
                roof_mask[original_inlier_indices] = True
                plane_type = "roof"
                total_roof_points += len(inlier_indices)
            elif z_component < self.params.wall_normal_z_threshold:
                # Vertical plane = wall
                wall_mask[original_inlier_indices] = True
                plane_type = "wall"
                total_wall_points += len(inlier_indices)
            else:
                # Sloped - could be roof or other structure
                roof_mask[original_inlier_indices] = True
                plane_type = "sloped_roof"
                total_roof_points += len(inlier_indices)

            building_mask[original_inlier_indices] = True
            planes_found += 1

            # Log progress every 5 planes or for significant planes
            if planes_found % 5 == 0 or len(inlier_indices) > 10000:
                elapsed_ransac = time.time() - ransac_start
                logger.info(f"  [RANSAC] Plane {planes_found}: {len(inlier_indices):,} pts ({plane_type}), "
                           f"total buildings: {building_mask.sum():,}, elapsed: {elapsed_ransac:.1f}s")

            logger.debug(f"  Plane {planes_found}: {len(inlier_indices):,} points, "
                        f"type={plane_type}, normal_z={z_component:.2f}")

            # Remove inliers and continue
            outlier_indices = list(set(range(len(remaining_pcd.points))) - set(inlier_indices))
            remaining_pcd = remaining_pcd.select_by_index(outlier_indices)
            remaining_indices = remaining_indices[outlier_indices]

        ransac_elapsed = time.time() - ransac_start
        logger.info(f"  RANSAC extraction completed in {ransac_elapsed:.1f}s")

        elapsed = time.time() - start_time

        logger.info(f"Building detection completed in {elapsed:.2f}s")
        logger.info(f"  Planes found: {planes_found}")
        logger.info(f"  Building points: {building_mask.sum():,} ({building_mask.sum()/self.n_points*100:.1f}%)")
        logger.info(f"  Roof points: {roof_mask.sum():,} ({roof_mask.sum()/self.n_points*100:.1f}%)")
        logger.info(f"  Wall points: {wall_mask.sum():,} ({wall_mask.sum()/self.n_points*100:.1f}%)")

        return building_mask, roof_mask, wall_mask

    def detect_with_stats(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Detect with detailed statistics

        Returns:
            Tuple of (masks_dict, stats_dict)
        """
        start_time = time.time()
        building_mask, roof_mask, wall_mask = self.detect()
        elapsed = time.time() - start_time

        masks = {
            'building': building_mask,
            'roof': roof_mask,
            'wall': wall_mask
        }

        stats = {
            'n_points': self.n_points,
            'building': {
                'count': int(building_mask.sum()),
                'percentage': float(building_mask.sum() / self.n_points * 100)
            },
            'roof': {
                'count': int(roof_mask.sum()),
                'percentage': float(roof_mask.sum() / self.n_points * 100),
                'hag_mean': float(self.hag[roof_mask].mean()) if roof_mask.sum() > 0 else 0
            },
            'wall': {
                'count': int(wall_mask.sum()),
                'percentage': float(wall_mask.sum() / self.n_points * 100)
            },
            'params': {
                'min_height': self.params.min_height,
                'distance_threshold': self.params.distance_threshold
            },
            'processing_time': elapsed
        }

        return masks, stats


class SimplePlanarClassifier:
    """
    Simpler planar surface classifier without iterative RANSAC

    Uses local planarity computed from PCA on neighborhoods.
    Faster but less accurate than full RANSAC.
    """

    def __init__(
        self,
        coords: np.ndarray,
        hag: np.ndarray,
        min_height: float = 2.0,
        planarity_threshold: float = 0.7,
        k_neighbors: int = 30
    ):
        self.coords = coords
        self.hag = hag
        self.n_points = len(coords)
        self.min_height = min_height
        self.planarity_threshold = planarity_threshold
        self.k_neighbors = k_neighbors

    def detect(self) -> np.ndarray:
        """
        Detect planar structures based on local planarity

        Returns:
            (N,) boolean mask for building points
        """
        from scipy.spatial import cKDTree

        logger.info("Detecting planar structures...")
        start_time = time.time()

        # Pre-filter by HAG
        height_mask = self.hag >= self.min_height
        candidate_indices = np.where(height_mask)[0]

        if len(candidate_indices) == 0:
            return np.zeros(self.n_points, dtype=bool)

        # Build KD-tree
        kdtree = cKDTree(self.coords)

        # Compute planarity for candidates
        building_mask = np.zeros(self.n_points, dtype=bool)

        # Sample for speed if too many candidates
        if len(candidate_indices) > 100_000:
            sample_indices = np.random.choice(candidate_indices, 100_000, replace=False)
        else:
            sample_indices = candidate_indices

        for idx in sample_indices:
            # Find neighbors
            _, neighbor_idx = kdtree.query(self.coords[idx], k=self.k_neighbors)
            neighbors = self.coords[neighbor_idx]

            # Compute planarity via PCA
            centered = neighbors - neighbors.mean(axis=0)
            cov = centered.T @ centered
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]

            # Planarity = (λ2 - λ3) / λ1
            planarity = (eigenvalues[1] - eigenvalues[2]) / (eigenvalues[0] + 1e-8)

            if planarity > self.planarity_threshold:
                building_mask[idx] = True

        elapsed = time.time() - start_time
        logger.info(f"Planar detection completed in {elapsed:.2f}s")
        logger.info(f"  Building points: {building_mask.sum():,}")

        return building_mask
