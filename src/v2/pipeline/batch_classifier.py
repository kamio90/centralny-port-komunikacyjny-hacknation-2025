"""
Batch Classifier - przetwarzanie ogromnych chmur punktów (100M+)

Strategia:
1. Wczytuj dane chunkami (10M punktów)
2. Dla każdego chunka uruchom pipeline
3. Łącz wyniki i zapisuj inkrementalnie

Optymalizacje:
- Minimalny footprint pamięciowy
- Checkpointing (wznowienie po błędzie)
- Progress tracking
"""

import numpy as np
import laspy
import logging
import time
import json
from pathlib import Path
from typing import Optional, Callable, Dict
from dataclasses import dataclass

from ..core import LASLoader
from .professional_pipeline import ProfessionalPipeline, PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Konfiguracja batch processingu"""
    chunk_size: int = 10_000_000  # 10M punktów na chunk
    checkpoint_dir: Optional[str] = None  # Katalog na checkpointy
    resume_from_chunk: int = 0  # Wznów od tego chunka


class BatchClassifier:
    """
    Batch classifier dla bardzo dużych chmur punktów

    Usage:
        classifier = BatchClassifier(input_path, output_path)
        stats = classifier.run(progress_callback=my_callback)
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        pipeline_config: Optional[PipelineConfig] = None,
        batch_config: Optional[BatchConfig] = None
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.pipeline_config = pipeline_config or PipelineConfig()
        self.batch_config = batch_config or BatchConfig()

        # Pobierz info o pliku
        self.file_info = LASLoader.get_file_info(str(input_path))
        self.total_points = self.file_info['n_points']

        # Oblicz liczbę chunków
        self.n_chunks = int(np.ceil(self.total_points / self.batch_config.chunk_size))

        logger.info(f"BatchClassifier: {self.total_points:,} punktów w {self.n_chunks} chunkach")

    def run(
        self,
        progress_callback: Optional[Callable[[int, int, float, str], None]] = None
    ) -> Dict:
        """
        Uruchom batch classification

        Args:
            progress_callback: Funkcja(chunk_idx, n_chunks, overall_pct, message)

        Returns:
            Dict ze statystykami
        """
        logger.info("=" * 70)
        logger.info("BATCH CLASSIFICATION")
        logger.info(f"Input: {self.input_path.name}")
        logger.info(f"Chunks: {self.n_chunks}")
        logger.info("=" * 70)

        start_time = time.time()

        # Przygotuj output
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Otwórz plik źródłowy
        with laspy.open(str(self.input_path)) as src:
            # Przygotuj header dla outputu
            src_header = src.header

            # Potrzebujemy LAS 1.4 dla extended classification
            has_rgb = src_header.point_format.id in [2, 3, 5, 7, 8, 10]
            point_format = 7 if has_rgb else 6
            out_header = laspy.LasHeader(point_format=point_format, version="1.4")
            out_header.scales = src_header.scales
            out_header.offsets = src_header.offsets

            # Statystyki
            all_stats = {
                'chunks_processed': 0,
                'total_points': 0,
                'classification': {},
                'processing_times': []
            }

            # Przetwarzaj chunk po chunku
            chunk_idx = 0
            points_processed = 0

            # Otwórz output do zapisu
            with laspy.open(str(self.output_path), mode="w", header=out_header) as out_file:

                for las_chunk in src.chunk_iterator(self.batch_config.chunk_size):
                    chunk_start = time.time()

                    # Skip jeśli resuming
                    if chunk_idx < self.batch_config.resume_from_chunk:
                        points_processed += len(las_chunk.x)
                        chunk_idx += 1
                        continue

                    n_chunk_points = len(las_chunk.x)

                    # Report progress
                    if progress_callback:
                        pct = points_processed / self.total_points * 100
                        progress_callback(
                            chunk_idx, self.n_chunks, pct,
                            f"Przetwarzanie chunka {chunk_idx+1}/{self.n_chunks}"
                        )

                    logger.info(f"\n[Chunk {chunk_idx+1}/{self.n_chunks}] {n_chunk_points:,} punktów")

                    # Przygotuj dane
                    coords = np.vstack([las_chunk.x, las_chunk.y, las_chunk.z]).T

                    colors = None
                    if has_rgb:
                        colors = np.vstack([
                            las_chunk.red / 65535.0,
                            las_chunk.green / 65535.0,
                            las_chunk.blue / 65535.0
                        ]).T

                    intensity = None
                    if hasattr(las_chunk, 'intensity') and las_chunk.intensity is not None:
                        max_int = las_chunk.intensity.max()
                        if max_int > 0:
                            intensity = las_chunk.intensity / max_int

                    # Uruchom pipeline
                    pipeline = ProfessionalPipeline(
                        coords, colors, intensity, self.pipeline_config
                    )
                    classification, chunk_stats = pipeline.run()

                    # Agreguj statystyki
                    for cls_id, info in chunk_stats.get('classification', {}).items():
                        if cls_id not in all_stats['classification']:
                            all_stats['classification'][cls_id] = {'count': 0}
                        all_stats['classification'][cls_id]['count'] += info['count']

                    # Przygotuj output chunk
                    out_las = laspy.ScaleAwarePointRecord.zeros(
                        n_chunk_points, header=out_header
                    )
                    out_las.x = las_chunk.x
                    out_las.y = las_chunk.y
                    out_las.z = las_chunk.z
                    out_las.intensity = las_chunk.intensity

                    if has_rgb:
                        out_las.red = las_chunk.red
                        out_las.green = las_chunk.green
                        out_las.blue = las_chunk.blue

                    out_las.classification = classification.astype(np.uint8)

                    # Zapisz chunk
                    out_file.write_points(out_las)

                    # Update stats
                    chunk_time = time.time() - chunk_start
                    all_stats['processing_times'].append(chunk_time)
                    all_stats['chunks_processed'] += 1
                    all_stats['total_points'] += n_chunk_points

                    points_processed += n_chunk_points
                    chunk_idx += 1

                    # Checkpoint
                    if self.batch_config.checkpoint_dir:
                        self._save_checkpoint(chunk_idx, all_stats)

                    logger.info(f"  Chunk time: {chunk_time:.1f}s ({n_chunk_points/chunk_time:,.0f} pkt/s)")

                    # Cleanup
                    del pipeline, classification, coords, colors, intensity
                    import gc
                    gc.collect()

        # Finalne statystyki
        elapsed = time.time() - start_time

        # Oblicz procenty
        for cls_id in all_stats['classification']:
            count = all_stats['classification'][cls_id]['count']
            all_stats['classification'][cls_id]['percentage'] = \
                count / all_stats['total_points'] * 100

        all_stats['processing_time'] = elapsed
        all_stats['points_per_second'] = all_stats['total_points'] / elapsed

        # Summary
        classified = sum(
            info['count'] for cls_id, info in all_stats['classification'].items()
            if cls_id != 1
        )
        all_stats['summary'] = {
            'classified_count': classified,
            'classified_percentage': classified / all_stats['total_points'] * 100
        }

        if progress_callback:
            progress_callback(self.n_chunks, self.n_chunks, 100, "Zakończono!")

        logger.info("=" * 70)
        logger.info("BATCH CLASSIFICATION COMPLETE")
        logger.info(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logger.info(f"  Points: {all_stats['total_points']:,}")
        logger.info(f"  Speed: {all_stats['points_per_second']:,.0f} pts/s")
        logger.info(f"  Output: {self.output_path}")
        logger.info("=" * 70)

        return all_stats

    def _save_checkpoint(self, chunk_idx: int, stats: Dict):
        """Zapisz checkpoint"""
        checkpoint_path = Path(self.batch_config.checkpoint_dir) / f"checkpoint_{chunk_idx}.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'chunk_idx': chunk_idx,
            'stats': stats,
            'input_path': str(self.input_path),
            'output_path': str(self.output_path)
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"  Checkpoint saved: {checkpoint_path}")


def batch_classify(
    input_path: str,
    output_path: str,
    chunk_size: int = 10_000_000,
    progress_callback: Optional[Callable] = None
) -> Dict:
    """
    Convenience function dla batch classification

    Args:
        input_path: Ścieżka do pliku LAS/LAZ
        output_path: Ścieżka wyjściowa
        chunk_size: Rozmiar chunka
        progress_callback: Callback funkcja

    Returns:
        Dict ze statystykami
    """
    config = BatchConfig(chunk_size=chunk_size)
    classifier = BatchClassifier(input_path, output_path, batch_config=config)
    return classifier.run(progress_callback)
