"""
LAS/LAZ file writer for classified point clouds

Writes classified point cloud data back to LAS/LAZ format with:
- Original coordinates, colors, intensity
- New classification labels
- Confidence scores (in user_data field)
- Proper LAS 1.2+ format compliance
"""

import numpy as np
import laspy
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LASWriter:
    """
    Write classified point cloud to LAS/LAZ file

    Features:
    - Preserves original point attributes
    - Adds classification labels (standard LAS field)
    - Stores confidence scores in user_data field
    - Supports LAZ compression
    - LAS 1.2+ format compliance
    """

    def __init__(self, output_path: str, compress: bool = True):
        """
        Initialize LAS writer

        Args:
            output_path: Path to output file (.las or .laz)
            compress: Whether to compress output (LAZ format)
        """
        self.output_path = Path(output_path)
        self.compress = compress

        # Auto-adjust extension based on compression
        if self.compress and not self.output_path.suffix == '.laz':
            self.output_path = self.output_path.with_suffix('.laz')
        elif not self.compress and not self.output_path.suffix == '.las':
            self.output_path = self.output_path.with_suffix('.las')

        logger.info(f"LAS writer initialized: {self.output_path} (compressed={compress})")

    def write(self,
             coords: np.ndarray,
             classifications: np.ndarray,
             colors: Optional[np.ndarray] = None,
             intensity: Optional[np.ndarray] = None,
             confidences: Optional[np.ndarray] = None,
             original_header: Optional[laspy.LasHeader] = None) -> Dict:
        """
        Write classified point cloud to LAS/LAZ file

        Args:
            coords: (N, 3) XYZ coordinates
            classifications: (N,) class labels [0-255]
            colors: (N, 3) RGB colors [0-1] (optional)
            intensity: (N,) intensity values [0-1] (optional)
            confidences: (N,) confidence scores [0-1] (optional)
            original_header: Original LAS header to preserve metadata (optional)

        Returns:
            Dictionary with export statistics
        """
        logger.info(f"Writing {len(coords):,} points to {self.output_path}...")

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine point format
        # Format 2: XYZ + Intensity + Classification + RGB
        # Format 3: XYZ + Intensity + Classification + RGB + GPS time
        has_color = colors is not None
        point_format = 2 if has_color else 0

        # Create new LAS file
        if original_header is not None:
            # Use original header as template
            header = laspy.LasHeader(version=original_header.version, point_format=point_format)
            header.scales = original_header.scales
            header.offsets = original_header.offsets
        else:
            # Create new header
            header = laspy.LasHeader(version="1.2", point_format=point_format)

            # Set scales and offsets
            header.offsets = coords.min(axis=0)
            header.scales = [0.001, 0.001, 0.001]  # 1mm precision

        # Create LasData
        las = laspy.LasData(header)

        # Write coordinates
        las.x = coords[:, 0]
        las.y = coords[:, 1]
        las.z = coords[:, 2]

        # Write classification
        las.classification = classifications.astype(np.uint8)

        # Write RGB colors if available
        if has_color:
            # Convert [0-1] to [0-65535]
            las.red = (colors[:, 0] * 65535).astype(np.uint16)
            las.green = (colors[:, 1] * 65535).astype(np.uint16)
            las.blue = (colors[:, 2] * 65535).astype(np.uint16)

        # Write intensity if available
        if intensity is not None:
            # Convert [0-1] to [0-65535]
            las.intensity = (intensity * 65535).astype(np.uint16)

        # Write confidence in user_data field if available
        if confidences is not None:
            # Convert [0-1] to [0-255]
            las.user_data = (confidences * 255).astype(np.uint8)

        # Write to file
        try:
            las.write(self.output_path)
            file_size_mb = self.output_path.stat().st_size / 1e6

            logger.info(f"✅ Successfully wrote {len(coords):,} points to {self.output_path}")
            logger.info(f"   File size: {file_size_mb:.2f} MB")

            stats = {
                'output_path': str(self.output_path),
                'point_count': len(coords),
                'file_size_mb': file_size_mb,
                'compressed': self.compress,
                'point_format': point_format,
                'has_color': has_color,
                'has_intensity': intensity is not None,
                'has_confidence': confidences is not None
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to write LAS file: {e}", exc_info=True)
            raise


def test_las_writer():
    """Test LAS writer"""
    print("Testing LASWriter...")

    # Create synthetic data
    np.random.seed(42)
    n_points = 1000

    coords = np.random.uniform(0, 100, size=(n_points, 3))
    classifications = np.random.randint(0, 12, size=n_points)
    colors = np.random.uniform(0, 1, size=(n_points, 3))
    intensity = np.random.uniform(0, 1, size=n_points)
    confidences = np.random.uniform(0.5, 1.0, size=n_points)

    # Test 1: Write LAZ with all attributes
    output_path = "test_output/classified_test.laz"
    writer = LASWriter(output_path, compress=True)

    stats = writer.write(
        coords=coords,
        classifications=classifications,
        colors=colors,
        intensity=intensity,
        confidences=confidences
    )

    print(f"\n✅ Test Results:")
    print(f"   Output: {stats['output_path']}")
    print(f"   Points: {stats['point_count']:,}")
    print(f"   Size: {stats['file_size_mb']:.2f} MB")
    print(f"   Format: {stats['point_format']}")
    print(f"   Compressed: {stats['compressed']}")

    # Verify file exists
    assert Path(output_path).exists(), "Output file not created"

    # Test 2: Read back and verify
    las_read = laspy.read(output_path)
    assert len(las_read.points) == n_points, "Point count mismatch"
    assert np.allclose(las_read.x, coords[:, 0], atol=0.001), "X coordinates mismatch"
    assert np.allclose(las_read.classification, classifications), "Classification mismatch"

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_las_writer()
