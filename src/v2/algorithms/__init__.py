"""
Professional LiDAR Classification Algorithms

This module contains industry-standard algorithms for point cloud classification:

- ground_csf: Cloth Simulation Filter for ground classification (Zhang et al. 2016)
- height_above_ground: HAG normalization for all points
- noise_sor: Statistical Outlier Removal for noise detection
- vegetation_classifier: HAG + NDVI based vegetation stratification
- building_detector: RANSAC-based planar structure detection
- infrastructure_detector: Linear feature detection (poles, rails, power lines)

Pipeline Order:
1. Noise Detection (SOR) → class 7
2. Ground Classification (CSF) → class 2
3. HAG Computation
4. Vegetation Classification (HAG + NDVI) → classes 3, 4, 5
5. Building Detection (RANSAC) → classes 6, 40, 41
6. Infrastructure Detection → classes 18, 19, 20, 30, etc.
"""

from .ground_csf import (
    GroundClassifierCSF,
    CSFParams,
    auto_tune_params as auto_tune_csf_params
)

from .height_above_ground import (
    HeightAboveGround,
    classify_by_hag
)

from .noise_sor import (
    NoiseDetectorSOR,
    NoiseDetectorRadius
)

from .noise_sor_fast import (
    NoiseDetectorVoxelSOR,
    NoiseDetectorGridDensity,
    auto_select_noise_detector
)

from .vegetation_classifier import (
    VegetationClassifier,
    VegetationParams,
    compute_ndvi,
    compute_excess_green
)

from .building_detector import (
    BuildingDetector,
    BuildingParams,
    SimplePlanarClassifier
)

from .infrastructure_detector import (
    InfrastructureDetector,
    InfrastructureParams
)

__all__ = [
    # Ground
    'GroundClassifierCSF',
    'CSFParams',
    'auto_tune_csf_params',

    # HAG
    'HeightAboveGround',
    'classify_by_hag',

    # Noise
    'NoiseDetectorSOR',
    'NoiseDetectorRadius',
    'NoiseDetectorVoxelSOR',
    'NoiseDetectorGridDensity',
    'auto_select_noise_detector',

    # Vegetation
    'VegetationClassifier',
    'VegetationParams',
    'compute_ndvi',
    'compute_excess_green',

    # Buildings
    'BuildingDetector',
    'BuildingParams',
    'SimplePlanarClassifier',

    # Infrastructure
    'InfrastructureDetector',
    'InfrastructureParams'
]
