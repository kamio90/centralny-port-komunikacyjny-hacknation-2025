"""
Spatial tiling engine for processing large point clouds in manageable chunks
Optimized for Apple M4 Max with 64GB RAM
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Iterator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class Tile:
    """
    Represents a spatial tile in the point cloud

    Attributes:
        id: Unique tile identifier
        bounds: (min_xyz, max_xyz) tuple defining tile boundaries
        center: Center point of tile
        point_count: Number of points in tile (0 if not yet counted)
        buffered_bounds: Expanded bounds including buffer zone
    """
    id: int
    bounds: Tuple[np.ndarray, np.ndarray]  # (min_xyz, max_xyz)
    center: np.ndarray
    point_count: int = 0
    buffered_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None


class TilingEngine:
    """
    Tiling engine optimized for M4 Max with 64GB RAM

    Strategy:
    - 150m × 150m tiles (larger than typical due to more RAM)
    - 20m buffer around each tile (prevents edge artifacts)
    - On-demand tile extraction (don't precompute)
    - Parallel processing support (14 CPU cores)

    The buffer ensures smooth transitions at tile boundaries during classification.
    Points in buffer zones are used for context but only core points are saved.
    """

    def __init__(self, tile_size: float = 150.0, buffer_size: float = 20.0):
        """
        Initialize tiling engine

        Args:
            tile_size: Size of each tile in meters (default 150m for 64GB RAM)
            buffer_size: Buffer zone around each tile in meters (default 20m)
        """
        self.tile_size = tile_size
        self.buffer_size = buffer_size
        logger.info(f"Initialized tiling engine: {tile_size}m tiles, {buffer_size}m buffer")

    def create_tile_grid(self, bounds: Dict) -> List[Tile]:
        """
        Create a grid of tiles covering the point cloud bounds

        Args:
            bounds: Dictionary with 'x', 'y', 'z' ranges
                   e.g., {'x': (min_x, max_x), 'y': (min_y, max_y), 'z': (min_z, max_z)}

        Returns:
            List of Tile objects covering the area
        """
        x_min, x_max = bounds['x']
        y_min, y_max = bounds['y']
        z_min, z_max = bounds['z']

        logger.info(f"Creating tile grid for bounds: X=[{x_min:.2f}, {x_max:.2f}], "
                   f"Y=[{y_min:.2f}, {y_max:.2f}], Z=[{z_min:.2f}, {z_max:.2f}]")

        # Create grid edges
        x_edges = np.arange(x_min, x_max + self.tile_size, self.tile_size)
        y_edges = np.arange(y_min, y_max + self.tile_size, self.tile_size)

        # Ensure we cover the entire area
        if x_edges[-1] < x_max:
            x_edges = np.append(x_edges, x_max)
        if y_edges[-1] < y_max:
            y_edges = np.append(y_edges, y_max)

        logger.info(f"Grid: {len(x_edges)-1} × {len(y_edges)-1} tiles")

        # Create tiles
        tiles = []
        tile_id = 0

        for i in range(len(x_edges) - 1):
            for j in range(len(y_edges) - 1):
                # Core tile bounds (no buffer)
                min_xyz = np.array([x_edges[i], y_edges[j], z_min])
                max_xyz = np.array([x_edges[i+1], y_edges[j+1], z_max])
                center = (min_xyz + max_xyz) / 2

                # Buffered bounds (for point extraction)
                buffered_min = min_xyz - np.array([self.buffer_size, self.buffer_size, 0])
                buffered_max = max_xyz + np.array([self.buffer_size, self.buffer_size, 0])

                # Clip buffered bounds to original area (don't go outside data)
                buffered_min = np.maximum(buffered_min, np.array([x_min, y_min, z_min]))
                buffered_max = np.minimum(buffered_max, np.array([x_max, y_max, z_max]))

                tile = Tile(
                    id=tile_id,
                    bounds=(min_xyz, max_xyz),
                    center=center,
                    buffered_bounds=(buffered_min, buffered_max)
                )

                tiles.append(tile)
                tile_id += 1

        logger.info(f"Created {len(tiles)} tiles")
        return tiles

    def extract_tile_points(self, las_reader, tile: Tile) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract points for a single tile (with buffer)

        This is the key method that will be called per-tile during processing.
        It reads points within the buffered bounds and marks which points are
        in the buffer vs. core tile.

        Args:
            las_reader: LASReader instance
            tile: Tile object to extract

        Returns:
            Dictionary with point data or None if tile is empty
        """
        import laspy

        logger.debug(f"Extracting points for tile {tile.id}")

        # Get buffered bounds
        buffered_min, buffered_max = tile.buffered_bounds
        core_min, core_max = tile.bounds

        try:
            # Read entire file (with 64GB RAM this is acceptable)
            # For even larger files, consider memory-mapped reading
            las = laspy.read(las_reader.filepath)

            # Filter points within buffered bounds
            mask = (
                (las.x >= buffered_min[0]) & (las.x < buffered_max[0]) &
                (las.y >= buffered_min[1]) & (las.y < buffered_max[1])
            )

            points_in_tile = mask.sum()

            if points_in_tile == 0:
                logger.debug(f"Tile {tile.id} is empty")
                return None  # Empty tile

            logger.debug(f"Tile {tile.id}: {points_in_tile:,} points (with buffer)")

            # Extract coordinates
            coords = np.vstack([las.x[mask], las.y[mask], las.z[mask]]).T

            # Extract colors if available
            colors = None
            if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                colors = np.vstack([
                    las.red[mask],
                    las.green[mask],
                    las.blue[mask]
                ]).T / 65535.0

            # Extract intensity
            intensity = None
            if hasattr(las, 'intensity'):
                intensity = las.intensity[mask] / 65535.0

            # Extract classification if present
            classification = None
            if hasattr(las, 'classification'):
                classification = las.classification[mask]

            # Create buffer mask (points outside core tile)
            # These points are used for context but won't be saved in final output
            buffer_mask = ~(
                (las.x[mask] >= core_min[0]) & (las.x[mask] < core_max[0]) &
                (las.y[mask] >= core_min[1]) & (las.y[mask] < core_max[1])
            )

            core_points = (~buffer_mask).sum()
            buffer_points = buffer_mask.sum()
            logger.debug(f"Tile {tile.id}: {core_points:,} core + {buffer_points:,} buffer points")

            return {
                'coords': coords,
                'colors': colors,
                'intensity': intensity,
                'classification': classification,
                'buffer_mask': buffer_mask,
                'tile_id': tile.id,
                'point_count': points_in_tile,
                'core_point_count': core_points
            }

        except Exception as e:
            logger.error(f"Failed to extract tile {tile.id}: {e}")
            raise

    def count_points_per_tile(self, las_reader, tiles: List[Tile]) -> List[Tile]:
        """
        Count points in each tile (useful for load balancing)

        Args:
            las_reader: LASReader instance
            tiles: List of tiles to count

        Returns:
            Updated list of tiles with point_count filled in
        """
        import laspy

        logger.info("Counting points per tile...")

        las = laspy.read(las_reader.filepath)

        for tile in tiles:
            core_min, core_max = tile.bounds

            mask = (
                (las.x >= core_min[0]) & (las.x < core_max[0]) &
                (las.y >= core_min[1]) & (las.y < core_max[1])
            )

            tile.point_count = mask.sum()

        total_points = sum(t.point_count for t in tiles)
        logger.info(f"Total points across all tiles: {total_points:,}")

        return tiles

    def get_tile_statistics(self, tiles: List[Tile]) -> Dict:
        """
        Get statistics about the tile grid

        Args:
            tiles: List of tiles

        Returns:
            Dictionary with statistics
        """
        point_counts = [t.point_count for t in tiles]
        non_empty = [t for t in tiles if t.point_count > 0]

        return {
            'total_tiles': len(tiles),
            'non_empty_tiles': len(non_empty),
            'empty_tiles': len(tiles) - len(non_empty),
            'total_points': sum(point_counts),
            'mean_points_per_tile': np.mean(point_counts) if point_counts else 0,
            'max_points_per_tile': max(point_counts) if point_counts else 0,
            'min_points_per_tile': min(point_counts) if point_counts else 0,
            'tile_size_m': self.tile_size,
            'buffer_size_m': self.buffer_size
        }

    def get_tiles_for_visualization(self, tiles: List[Tile]) -> Dict:
        """
        Get tile boundaries for visualization

        Returns data suitable for plotting tile grid

        Args:
            tiles: List of tiles

        Returns:
            Dictionary with coordinates for plotting
        """
        rectangles = []

        for tile in tiles:
            min_xyz, max_xyz = tile.bounds

            # Create rectangle corners (in XY plane)
            corners = np.array([
                [min_xyz[0], min_xyz[1]],
                [max_xyz[0], min_xyz[1]],
                [max_xyz[0], max_xyz[1]],
                [min_xyz[0], max_xyz[1]],
                [min_xyz[0], min_xyz[1]]  # Close the rectangle
            ])

            rectangles.append({
                'corners': corners,
                'tile_id': tile.id,
                'point_count': tile.point_count,
                'center': tile.center[:2]  # XY only
            })

        return {'rectangles': rectangles}


class ProgressTracker:
    """
    Progress tracker for Streamlit integration

    Provides real-time updates during tile processing
    """

    def __init__(self, total_steps: int, description: str = "Processing"):
        """
        Initialize progress tracker

        Args:
            total_steps: Total number of steps to complete
            description: Description of the task
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.progress_bar = None
        self.status_text = None

    def initialize_ui(self, streamlit_module):
        """
        Initialize Streamlit UI elements

        Args:
            streamlit_module: Streamlit module (st)
        """
        self.progress_bar = streamlit_module.progress(0)
        self.status_text = streamlit_module.empty()
        self.update(f"{self.description} - Starting...")

    def update(self, message: str = ""):
        """
        Update progress

        Args:
            message: Status message to display
        """
        self.current_step += 1
        progress = min(self.current_step / self.total_steps, 1.0)

        if self.progress_bar:
            self.progress_bar.progress(progress)
        if self.status_text:
            step_info = f"[{self.current_step}/{self.total_steps}]"
            self.status_text.text(f"{step_info} {message}")

    def finish(self, message: str = "Complete!"):
        """
        Mark as finished

        Args:
            message: Completion message
        """
        if self.progress_bar:
            self.progress_bar.progress(1.0)
        if self.status_text:
            self.status_text.text(f"✅ {message}")
