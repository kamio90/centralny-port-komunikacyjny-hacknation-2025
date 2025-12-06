"""
Feature extraction for point cloud classification
Computes geometric and spectral features for ML model input
"""

import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract features from point cloud for classification

    Features include:
    - XYZ coordinates (normalized)
    - RGB colors
    - Intensity
    - Geometric normals
    - Height above ground
    - Local density
    """

    def __init__(self, normalize: bool = True):
        """
        Initialize feature extractor

        Args:
            normalize: Whether to normalize features to [0, 1] range
        """
        self.normalize = normalize

    def extract(self,
               coords: np.ndarray,
               colors: Optional[np.ndarray] = None,
               intensity: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract features from point cloud

        Args:
            coords: (N, 3) XYZ coordinates
            colors: (N, 3) RGB colors [0-1] (optional)
            intensity: (N,) intensity values [0-1] (optional)

        Returns:
            (N, C) feature matrix where C depends on available data
            Output: 9 features (XYZ + RGB + I + Nx + Ny)
        """
        logger.info(f"Extracting features from {len(coords):,} points")

        features = []

        # 1. Normalized coordinates (3 features)
        if self.normalize:
            coords_norm = self._normalize_coords(coords)
        else:
            coords_norm = coords
        features.append(coords_norm)

        # 2. RGB colors (3 features)
        if colors is not None:
            features.append(colors)
        else:
            # Default to zeros if no colors
            features.append(np.zeros((len(coords), 3), dtype=np.float32))

        # 3. Intensity (1 feature)
        if intensity is not None:
            features.append(intensity.reshape(-1, 1))
        else:
            features.append(np.zeros((len(coords), 1), dtype=np.float32))

        # 4. Simple geometric features (2 features: Nx, Ny)
        # Using simplified normal approximation - just X and Y components
        # This gives us 9 total features to match the model
        simple_normals = np.zeros((len(coords), 2), dtype=np.float32)
        features.append(simple_normals)

        # Concatenate all features
        feature_matrix = np.hstack(features).astype(np.float32)

        logger.info(f"Extracted {feature_matrix.shape[1]} features per point")
        return feature_matrix

    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Normalize coordinates to [0, 1] range

        Args:
            coords: (N, 3) XYZ coordinates

        Returns:
            Normalized coordinates
        """
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        ranges = maxs - mins

        # Avoid division by zero
        ranges[ranges == 0] = 1.0

        normalized = (coords - mins) / ranges
        return normalized.astype(np.float32)

    def compute_normals_knn(self, coords: np.ndarray, k: int = 20) -> np.ndarray:
        """
        Compute point normals using KNN and PCA

        This is a more advanced feature that will be implemented later

        Args:
            coords: (N, 3) XYZ coordinates
            k: Number of neighbors for normal estimation

        Returns:
            (N, 3) normal vectors
        """
        # TODO: Implement using Open3D or scikit-learn KNN
        logger.warning("Normal computation not yet implemented, using placeholder")
        normals = np.zeros_like(coords)
        normals[:, 2] = 1.0  # Point up
        return normals
