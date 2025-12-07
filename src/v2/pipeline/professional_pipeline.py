"""
Professional LiDAR Classification Pipeline

Industry-standard pipeline using proven algorithms:
1. Statistical Outlier Removal (SOR) → Noise (class 7)
2. Cloth Simulation Filter (CSF) → Ground (class 2)
3. Height Above Ground (HAG) computation
4. Vegetation classification (HAG + NDVI) → classes 3, 4, 5
5. Building detection (RANSAC) → classes 6, 40, 41
6. Infrastructure detection → classes 18, 19, 20, 30, etc.

This replaces the broken feature-based classification that marked 99.68% as noise.
"""

import numpy as np
import logging
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass, field
import time

from ..algorithms import (
    GroundClassifierCSF, CSFParams, auto_tune_csf_params,
    HeightAboveGround,
    NoiseDetectorSOR,
    NoiseDetectorVoxelSOR,
    auto_select_noise_detector,
    VegetationClassifier, VegetationParams,
    BuildingDetector, BuildingParams,
    InfrastructureDetector, InfrastructureParams
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for professional pipeline"""

    # Which steps to run
    detect_noise: bool = True
    classify_ground: bool = True
    classify_vegetation: bool = True
    detect_buildings: bool = True
    detect_infrastructure: bool = True

    # Algorithm parameters
    csf_params: CSFParams = field(default_factory=CSFParams)
    vegetation_params: VegetationParams = field(default_factory=VegetationParams)
    building_params: BuildingParams = field(default_factory=BuildingParams)
    infrastructure_params: InfrastructureParams = field(default_factory=InfrastructureParams)

    # Noise detection parameters
    noise_k_neighbors: int = 30
    noise_std_ratio: float = 2.0
    noise_voxel_size: float = 0.5  # For voxel-accelerated SOR (larger = faster)

    # HAG parameters
    hag_grid_resolution: float = 1.0

    # Auto-tune CSF based on data
    auto_tune_csf: bool = True

    # Performance optimization for large clouds
    use_fast_noise_detection: bool = True  # Auto-select fast detector for large clouds
    large_cloud_threshold: int = 10_000_000  # Use fast algorithms above this


class ProfessionalPipeline:
    """
    Professional LiDAR classification pipeline

    Usage:
        pipeline = ProfessionalPipeline(coords, colors, intensity)
        classification, stats = pipeline.run()

        # Or with progress callback
        def on_progress(step, progress, message):
            print(f"[{step}] {progress:.0f}% - {message}")

        classification, stats = pipeline.run(progress_callback=on_progress)
    """

    # ASPRS class definitions
    CLASS_UNCLASSIFIED = 1
    CLASS_GROUND = 2
    CLASS_LOW_VEG = 3
    CLASS_MEDIUM_VEG = 4
    CLASS_HIGH_VEG = 5
    CLASS_BUILDING = 6
    CLASS_NOISE = 7
    CLASS_WATER = 9
    CLASS_RAIL = 18
    CLASS_POWERLINE = 19
    CLASS_POLE = 20
    CLASS_PLATFORM = 21      # Perony kolejowe
    CLASS_ROAD = 30
    CLASS_CURB = 32          # Krawężniki
    CLASS_SIGN = 35          # Znaki drogowe
    CLASS_BARRIER = 36       # Bariery drogowe
    CLASS_WALL = 40
    CLASS_ROOF = 41

    def __init__(
        self,
        coords: np.ndarray,
        colors: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
        config: Optional[PipelineConfig] = None
    ):
        """
        Args:
            coords: (N, 3) Point coordinates XYZ
            colors: (N, 3) RGB colors [0-1] or None
            intensity: (N,) Intensity values [0-1] or None
            config: Pipeline configuration
        """
        self.coords = coords
        self.colors = colors
        self.intensity = intensity
        self.n_points = len(coords)
        self.config = config or PipelineConfig()

        # Results
        self.classification = np.ones(self.n_points, dtype=np.uint8)  # Default: Unclassified
        self.hag = None
        self.ground_mask = None
        self.vegetation_mask = None

        # Statistics
        self.stats = {
            'n_points': self.n_points,
            'steps': {}
        }

        logger.info(f"ProfessionalPipeline initialized: {self.n_points:,} points")
        logger.info(f"  Colors: {'Yes' if colors is not None else 'No'}")
        logger.info(f"  Intensity: {'Yes' if intensity is not None else 'No'}")

    def run(
        self,
        progress_callback: Optional[Callable[[str, float, str], None]] = None
    ) -> tuple:
        """
        Run the full classification pipeline

        Args:
            progress_callback: Function(step_name, progress_pct, message)

        Returns:
            Tuple of (classification, stats)
        """
        logger.info("=" * 70)
        logger.info("PROFESSIONAL LIDAR CLASSIFICATION PIPELINE")
        logger.info("=" * 70)

        start_time = time.time()
        total_steps = sum([
            self.config.detect_noise,
            self.config.classify_ground,
            True,  # HAG is always computed if ground is classified
            self.config.classify_vegetation,
            self.config.detect_buildings,
            self.config.detect_infrastructure
        ])
        current_step = 0

        def report_progress(step_name: str, pct: float, msg: str):
            if progress_callback:
                overall_pct = (current_step / total_steps * 100) + (pct / total_steps)
                progress_callback(step_name, overall_pct, msg)

        # STEP 1: Noise Detection
        if self.config.detect_noise:
            current_step += 1
            report_progress("noise", 0, "Detecting noise points...")
            self._detect_noise()
            report_progress("noise", 100, "Noise detection complete")

        # STEP 2: Ground Classification
        if self.config.classify_ground:
            current_step += 1
            report_progress("ground", 0, "Classifying ground points...")
            self._classify_ground()
            report_progress("ground", 100, "Ground classification complete")

            # STEP 3: HAG Computation (required for vegetation/building)
            current_step += 1
            report_progress("hag", 0, "Computing Height Above Ground...")
            self._compute_hag()
            report_progress("hag", 100, "HAG computation complete")

        # STEP 4: Vegetation Classification
        if self.config.classify_vegetation and self.hag is not None:
            current_step += 1
            report_progress("vegetation", 0, "Classifying vegetation...")
            self._classify_vegetation()
            report_progress("vegetation", 100, "Vegetation classification complete")

        # STEP 5: Building Detection
        if self.config.detect_buildings and self.hag is not None:
            current_step += 1
            report_progress("buildings", 0, "Detecting buildings...")
            self._detect_buildings()
            report_progress("buildings", 100, "Building detection complete")

        # STEP 6: Infrastructure Detection
        if self.config.detect_infrastructure and self.hag is not None:
            current_step += 1
            report_progress("infrastructure", 0, "Detecting infrastructure...")
            self._detect_infrastructure()
            report_progress("infrastructure", 100, "Infrastructure detection complete")

        # Final statistics
        elapsed = time.time() - start_time
        self.stats['processing_time'] = elapsed
        self.stats['points_per_second'] = self.n_points / elapsed

        self._compute_final_stats()

        logger.info("=" * 70)
        logger.info("CLASSIFICATION COMPLETE")
        logger.info(f"  Total time: {elapsed:.1f}s")
        logger.info(f"  Speed: {self.stats['points_per_second']:,.0f} points/s")
        logger.info("=" * 70)

        return self.classification, self.stats

    def _detect_noise(self):
        """Step 1: Statistical Outlier Removal (optimized for large clouds)"""
        logger.info("\n[STEP 1] NOISE DETECTION (SOR)")

        start = time.time()

        # Use fast voxel-accelerated SOR for large clouds
        use_fast = (
            self.config.use_fast_noise_detection and
            self.n_points > self.config.large_cloud_threshold
        )

        if use_fast:
            # Auto-scale voxel size based on cloud size for optimal performance
            # Larger clouds need larger voxels to stay within time budget
            if self.n_points > 100_000_000:  # > 100M points
                voxel_size = 1.0  # 1m voxels for huge clouds
                k_neighbors = 20  # Fewer neighbors for speed
            elif self.n_points > 50_000_000:  # > 50M points
                voxel_size = 0.75
                k_neighbors = 25
            else:
                voxel_size = self.config.noise_voxel_size
                k_neighbors = self.config.noise_k_neighbors

            logger.info(f"  Using Voxel-Accelerated SOR (cloud: {self.n_points:,})")
            logger.info(f"    voxel_size={voxel_size}m, k_neighbors={k_neighbors}")

            detector = NoiseDetectorVoxelSOR(
                self.coords,
                voxel_size=voxel_size,
                k_neighbors=k_neighbors,
                std_ratio=self.config.noise_std_ratio
            )
        else:
            logger.info("  Using Standard SOR")
            detector = NoiseDetectorSOR(
                self.coords,
                k_neighbors=self.config.noise_k_neighbors,
                std_ratio=self.config.noise_std_ratio
            )

        noise_mask = detector.detect()
        self.classification[noise_mask] = self.CLASS_NOISE

        elapsed = time.time() - start
        self.stats['steps']['noise'] = {
            'count': int(noise_mask.sum()),
            'percentage': float(noise_mask.sum() / self.n_points * 100),
            'time': elapsed,
            'method': 'voxel_sor' if use_fast else 'standard_sor'
        }

        logger.info(f"  Noise points: {noise_mask.sum():,} ({noise_mask.sum()/self.n_points*100:.2f}%)")

    def _classify_ground(self):
        """Step 2: CSF Ground Classification"""
        logger.info("\n[STEP 2] GROUND CLASSIFICATION (CSF)")

        start = time.time()

        # Exclude noise points
        valid_mask = self.classification != self.CLASS_NOISE
        valid_coords = self.coords[valid_mask]

        # Auto-tune CSF parameters if enabled
        if self.config.auto_tune_csf:
            csf_params = auto_tune_csf_params(valid_coords)
        else:
            csf_params = self.config.csf_params

        # Run CSF on valid points
        classifier = GroundClassifierCSF(valid_coords, params=csf_params)
        valid_ground_mask = classifier.classify()

        # Map back to full point cloud
        self.ground_mask = np.zeros(self.n_points, dtype=bool)
        valid_indices = np.where(valid_mask)[0]
        self.ground_mask[valid_indices[valid_ground_mask]] = True

        # Update classification
        self.classification[self.ground_mask] = self.CLASS_GROUND

        elapsed = time.time() - start
        self.stats['steps']['ground'] = {
            'count': int(self.ground_mask.sum()),
            'percentage': float(self.ground_mask.sum() / self.n_points * 100),
            'time': elapsed
        }

        logger.info(f"  Ground points: {self.ground_mask.sum():,} ({self.ground_mask.sum()/self.n_points*100:.2f}%)")

    def _compute_hag(self):
        """Step 3: Height Above Ground Computation"""
        logger.info("\n[STEP 3] HEIGHT ABOVE GROUND (HAG)")

        start = time.time()

        if self.ground_mask is None or self.ground_mask.sum() < 10:
            logger.warning("  Not enough ground points for HAG computation!")
            self.hag = self.coords[:, 2] - self.coords[:, 2].min()
        else:
            hag_calc = HeightAboveGround(
                self.coords,
                self.ground_mask,
                grid_resolution=self.config.hag_grid_resolution
            )
            self.hag = hag_calc.compute()

        elapsed = time.time() - start
        self.stats['steps']['hag'] = {
            'min': float(self.hag.min()),
            'max': float(self.hag.max()),
            'mean': float(self.hag.mean()),
            'time': elapsed
        }

        logger.info(f"  HAG range: {self.hag.min():.2f}m to {self.hag.max():.2f}m")

    def _classify_vegetation(self):
        """Step 4: Vegetation Classification"""
        logger.info("\n[STEP 4] VEGETATION CLASSIFICATION")

        start = time.time()

        classifier = VegetationClassifier(
            self.hag,
            self.colors,
            self.config.vegetation_params,
            self.ground_mask
        )

        low_mask, medium_mask, high_mask = classifier.classify()

        # Only classify points that are still unclassified
        unclassified = self.classification == self.CLASS_UNCLASSIFIED

        self.classification[low_mask & unclassified] = self.CLASS_LOW_VEG
        self.classification[medium_mask & unclassified] = self.CLASS_MEDIUM_VEG
        self.classification[high_mask & unclassified] = self.CLASS_HIGH_VEG

        # Store vegetation mask for later use
        self.vegetation_mask = low_mask | medium_mask | high_mask

        elapsed = time.time() - start
        self.stats['steps']['vegetation'] = {
            'low': int(low_mask.sum()),
            'medium': int(medium_mask.sum()),
            'high': int(high_mask.sum()),
            'total': int(self.vegetation_mask.sum()),
            'percentage': float(self.vegetation_mask.sum() / self.n_points * 100),
            'time': elapsed
        }

    def _detect_buildings(self):
        """Step 5: Building Detection"""
        logger.info("\n[STEP 5] BUILDING DETECTION (RANSAC)")

        start = time.time()

        detector = BuildingDetector(
            self.coords,
            self.hag,
            self.colors,
            self.config.building_params
        )

        building_mask, roof_mask, wall_mask = detector.detect()

        # Only classify points that are still unclassified
        unclassified = self.classification == self.CLASS_UNCLASSIFIED

        # Assign classes (roofs and walls are subsets of building)
        self.classification[roof_mask & unclassified] = self.CLASS_ROOF
        self.classification[wall_mask & unclassified] = self.CLASS_WALL

        # Remaining building points
        other_building = building_mask & ~roof_mask & ~wall_mask & unclassified
        self.classification[other_building] = self.CLASS_BUILDING

        elapsed = time.time() - start
        self.stats['steps']['buildings'] = {
            'total': int(building_mask.sum()),
            'roofs': int(roof_mask.sum()),
            'walls': int(wall_mask.sum()),
            'percentage': float(building_mask.sum() / self.n_points * 100),
            'time': elapsed
        }

    def _detect_infrastructure(self):
        """Step 6: Infrastructure Detection"""
        logger.info("\n[STEP 6] INFRASTRUCTURE DETECTION")

        start = time.time()

        detector = InfrastructureDetector(
            self.coords,
            self.hag,
            self.intensity,
            self.colors,
            self.config.infrastructure_params,
            self.ground_mask,
            self.vegetation_mask
        )

        results = detector.detect()

        # Only classify points that are still unclassified
        unclassified = self.classification == self.CLASS_UNCLASSIFIED

        # Assign infrastructure classes
        self.classification[results['powerlines'] & unclassified] = self.CLASS_POWERLINE
        self.classification[results['poles'] & unclassified] = self.CLASS_POLE
        self.classification[results['rails'] & unclassified] = self.CLASS_RAIL
        self.classification[results['roads'] & unclassified] = self.CLASS_ROAD
        self.classification[results['curbs'] & unclassified] = self.CLASS_CURB
        self.classification[results['signs'] & unclassified] = self.CLASS_SIGN
        self.classification[results['barriers'] & unclassified] = self.CLASS_BARRIER
        self.classification[results['platforms'] & unclassified] = self.CLASS_PLATFORM

        elapsed = time.time() - start
        self.stats['steps']['infrastructure'] = {
            'powerlines': int(results['powerlines'].sum()),
            'poles': int(results['poles'].sum()),
            'rails': int(results['rails'].sum()),
            'roads': int(results['roads'].sum()),
            'curbs': int(results['curbs'].sum()),
            'signs': int(results['signs'].sum()),
            'barriers': int(results['barriers'].sum()),
            'platforms': int(results['platforms'].sum()),
            'time': elapsed
        }

    def _compute_final_stats(self):
        """Compute final classification statistics"""
        unique, counts = np.unique(self.classification, return_counts=True)

        self.stats['classification'] = {}
        for cls, count in zip(unique, counts):
            self.stats['classification'][int(cls)] = {
                'count': int(count),
                'percentage': float(count / self.n_points * 100)
            }

        # Summary
        classified = self.classification != self.CLASS_UNCLASSIFIED
        self.stats['summary'] = {
            'classified_count': int(classified.sum()),
            'classified_percentage': float(classified.sum() / self.n_points * 100),
            'unclassified_count': int((~classified).sum()),
            'unclassified_percentage': float((~classified).sum() / self.n_points * 100)
        }

        logger.info("\n[FINAL STATISTICS]")
        logger.info(f"  Classified: {classified.sum():,} ({classified.sum()/self.n_points*100:.1f}%)")
        logger.info(f"  Unclassified: {(~classified).sum():,} ({(~classified).sum()/self.n_points*100:.1f}%)")


def run_professional_classification(
    coords: np.ndarray,
    colors: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None,
    config: Optional[PipelineConfig] = None,
    progress_callback: Optional[Callable] = None
) -> tuple:
    """
    Convenience function to run professional classification

    Args:
        coords: (N, 3) Point coordinates
        colors: (N, 3) RGB colors or None
        intensity: (N,) Intensity values or None
        config: Pipeline configuration
        progress_callback: Progress callback function

    Returns:
        Tuple of (classification, stats)
    """
    pipeline = ProfessionalPipeline(coords, colors, intensity, config)
    return pipeline.run(progress_callback)
