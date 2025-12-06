"""
Główny pipeline klasyfikacji chmur punktów

Orchestrator łączący wszystkie komponenty:
- Wczytywanie LAS/LAZ
- Tiling (podział na kafelki)
- Feature extraction
- Klasyfikacja (45 klas)
- Zapis wyników

CLEAN, MODULAR, THREAD-SAFE
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import logging

from ..core import LASLoader, LASWriter, TilingEngine, Tile
from ..features import GeometricFeatureExtractor
from ..classifiers import ClassifierRegistry, HeightZoneCalculator

logger = logging.getLogger(__name__)


class ClassificationPipeline:
    """
    Pipeline klasyfikacji chmur punktów

    Automatycznie obsługuje:
    - Duże chmury (auto-tiling)
    - Równoległe przetwarzanie (ThreadPoolExecutor)
    - Progress tracking
    - Robust error handling
    """

    def __init__(self,
                 input_path: str,
                 output_path: str,
                 enabled_classes: Optional[List[int]] = None,
                 n_threads: int = 4,
                 demo_mode: bool = False):
        """
        Args:
            input_path: Ścieżka do pliku wejściowego LAS/LAZ
            output_path: Ścieżka do pliku wyjściowego
            enabled_classes: Lista ID klas do klasyfikacji (None = wszystkie)
            n_threads: Liczba wątków do równoległego przetwarzania
            demo_mode: Tryb demo (szybsze, mniej dokładne)
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.enabled_classes = enabled_classes
        self.n_threads = n_threads
        self.demo_mode = demo_mode

        # Pobierz klasyfikatory
        all_classifiers = ClassifierRegistry.get_all()

        if enabled_classes is not None:
            # Filtruj tylko włączone klasy
            self.classifiers = {
                cid: all_classifiers[cid]
                for cid in enabled_classes
                if cid in all_classifiers
            }
        else:
            self.classifiers = all_classifiers

        # Sortuj klasyfikatory według priorytetu
        self.classifiers = dict(sorted(
            self.classifiers.items(),
            key=lambda x: x[1]().priority
        ))

        logger.info(f"Pipeline zainicjalizowany:")
        logger.info(f"  Plik wejściowy: {self.input_path.name}")
        logger.info(f"  Plik wyjściowy: {self.output_path.name}")
        logger.info(f"  Włączonych klas: {len(self.classifiers)}")
        logger.info(f"  Wątki: {self.n_threads}")
        logger.info(f"  Tryb DEMO: {self.demo_mode}")

        # Progress tracking - track POINTS not just tiles (dla stabilnego ETA)
        self.progress_lock = threading.Lock()
        self.completed_tiles = 0
        self.total_tiles = 0
        self.completed_points = 0
        self.total_points = 0
        self.start_time = None

    def run(self, progress_callback: Optional[Callable] = None) -> Dict:
        """
        Uruchamia pełny pipeline klasyfikacji

        Args:
            progress_callback: Funkcja callback(progress_dict) wywoływana podczas postępu

        Returns:
            Dict ze statystykami:
                - n_points: Całkowita liczba punktów
                - n_tiles: Liczba kafelków
                - processing_time: Czas przetwarzania (sekundy)
                - classification_stats: Statystyki klasyfikacji
        """
        self.start_time = time.time()
        logger.info("=" * 70)
        logger.info("ROZPOCZYNAM KLASYFIKACJĘ")
        logger.info("=" * 70)

        # KROK 1: Wczytaj chmurę punktów
        logger.info("KROK 1/5: Wczytywanie chmury punktów...")
        loader = LASLoader(str(self.input_path))
        data = loader.load()

        coords = data['coords']
        colors = data['colors']
        intensity = data['intensity']
        original_header = data['header']
        n_points = len(coords)

        logger.info(f"  Wczytano: {n_points:,} punktów")

        # KROK 2: Podziel na kafelki
        logger.info("KROK 2/5: Tworzenie kafelków...")
        tiling_engine = TilingEngine(
            coords=coords,
            target_points_per_tile=500_000 if not self.demo_mode else 10_000_000  # DEMO: ~28 tiles dla 277M
        )
        tiles = tiling_engine.create_tiles()

        # Filtruj puste kafelki
        tiles = [t for t in tiles if t.point_count > 0]
        self.total_tiles = len(tiles)
        self.total_points = n_points  # Total points dla ETA

        logger.info(f"  Utworzono: {self.total_tiles} niepustych kafelków")

        # KROK 3: Przetwórz kafelki
        logger.info("KROK 3/5: Klasyfikacja kafelków...")
        classification = np.ones(n_points, dtype=np.uint8)  # Domyślnie: Unclassified (1)

        if self.n_threads > 1:
            # Równoległe przetwarzanie
            logger.info(f"  Przetwarzanie równoległe ({self.n_threads} wątków)...")
            self._process_tiles_parallel(tiles, coords, colors, intensity,
                                        classification, progress_callback)
        else:
            # Sekwencyjne przetwarzanie
            logger.info("  Przetwarzanie sekwencyjne...")
            self._process_tiles_sequential(tiles, coords, colors, intensity,
                                          classification, progress_callback)

        # KROK 4: Post-processing (opcjonalnie)
        logger.info("KROK 4/5: Post-processing...")
        # TODO: Context refinement, noise filtering

        # KROK 5: Zapisz wyniki
        logger.info("KROK 5/5: Zapisywanie wyników...")
        LASWriter.write(
            output_path=str(self.output_path),
            coords=coords,
            classification=classification,
            original_header=original_header,
            colors=colors,
            intensity=intensity
        )

        # Statystyki
        processing_time = time.time() - self.start_time
        unique, counts = np.unique(classification, return_counts=True)
        classification_stats = {int(cls): int(count) for cls, count in zip(unique, counts)}

        stats = {
            'n_points': n_points,
            'n_tiles': self.total_tiles,
            'processing_time': processing_time,
            'points_per_second': n_points / processing_time,
            'classification_stats': classification_stats
        }

        logger.info("=" * 70)
        logger.info("KLASYFIKACJA ZAKOŃCZONA!")
        logger.info(f"  Czas: {processing_time:.1f}s")
        logger.info(f"  Prędkość: {stats['points_per_second']:,.0f} pkt/s")
        logger.info("=" * 70)

        return stats

    def _process_tiles_sequential(self, tiles, coords, colors, intensity, classification, progress_callback):
        """Przetwarzanie sekwencyjne (pojedynczy wątek)"""
        for i, tile in enumerate(tiles):
            self._process_single_tile(tile, coords, colors, intensity, classification)

            # Progress update - track both tiles AND points
            self.completed_tiles += 1
            self.completed_points += tile.point_count  # CRITICAL dla stabilnego ETA!
            if progress_callback:
                progress_callback(self._get_progress_dict())

            logger.info(f"  Kafelek {i+1}/{self.total_tiles}: {tile.point_count:,} punktów")

    def _process_tiles_parallel(self, tiles, coords, colors, intensity, classification, progress_callback):
        """Przetwarzanie równoległe (ThreadPoolExecutor)"""

        def process_tile_wrapper(tile):
            """Wrapper dla ThreadPoolExecutor"""
            try:
                tile_classification = self._classify_tile(tile, coords, colors, intensity)
                return tile, tile_classification, None
            except Exception as e:
                logger.error(f"Błąd w kafelku {tile.tile_id}: {e}")
                return tile, None, str(e)

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = {executor.submit(process_tile_wrapper, tile): tile for tile in tiles}

            for future in as_completed(futures):
                tile, tile_classification, error = future.result()

                if error is None:
                    # Przypisz klasyfikację do globalnej tablicy
                    classification[tile.indices] = tile_classification

                # Progress update - track both tiles AND points
                with self.progress_lock:
                    self.completed_tiles += 1
                    self.completed_points += tile.point_count  # CRITICAL dla stabilnego ETA!
                    if progress_callback:
                        progress_callback(self._get_progress_dict())

                logger.info(f"  ✓ Kafelek {self.completed_tiles}/{self.total_tiles}: "
                           f"{tile.point_count:,} punktów")

    def _process_single_tile(self, tile: Tile, coords, colors, intensity, classification):
        """
        Przetwarza pojedynczy kafelek (in-place update)

        Thread-safe: każdy kafelek ma unikalne indeksy
        """
        tile_classification = self._classify_tile(tile, coords, colors, intensity)
        classification[tile.indices] = tile_classification

    def _classify_tile(self, tile: Tile, coords, colors, intensity) -> np.ndarray:
        """
        Klasyfikuje punkty w kafelku

        Args:
            tile: Obiekt Tile
            coords: Pełna chmura punktów
            colors: Pełne kolory
            intensity: Pełna intensywność

        Returns:
            (M,) Klasyfikacja punktów w kafelku
        """
        # Pobierz dane kafelka
        tile_coords = coords[tile.indices]
        tile_colors = colors[tile.indices] if colors is not None else None
        tile_intensity = intensity[tile.indices] if intensity is not None else None

        n_tile_points = len(tile_coords)

        # Ekstrakcja cech
        sample_rate = 0.005 if not self.demo_mode else 0.0002  # Demo: 0.02% próbek (ULTRA FAST dla hackatonu!)
        search_radius = 1.0 if not self.demo_mode else 0.5  # Demo: mniejszy radius = szybsze KD-tree
        feature_extractor = GeometricFeatureExtractor(
            coords=tile_coords,
            sample_rate=sample_rate,
            search_radius=search_radius
        )
        features = feature_extractor.extract()

        # Dodaj cechy kolorów (jeśli dostępne)
        if tile_colors is not None:
            features['ndvi'] = GeometricFeatureExtractor.compute_ndvi(tile_colors)
            features['brightness'] = GeometricFeatureExtractor.compute_brightness(tile_colors)
            features['saturation'] = GeometricFeatureExtractor.compute_saturation(tile_colors)

        # Oblicz strefy wysokości
        height_zones, _ = HeightZoneCalculator.calculate(tile_coords)

        # Klasyfikacja hierarchiczna (priorytet)
        tile_classification = np.ones(n_tile_points, dtype=np.uint8)  # Domyślnie: 1 (Unclassified)

        for class_id, classifier_class in self.classifiers.items():
            if class_id == 1:  # Pomiń Unclassified
                continue

            # Instancjonuj klasyfikator
            classifier = classifier_class()

            # Klasyfikuj
            mask = classifier.classify(
                coords=tile_coords,
                features=features,
                height_zones=height_zones,
                colors=tile_colors,
                intensity=tile_intensity
            )

            # Przypisz tylko do niesk lasyfikowanych punktów
            mask = mask & (tile_classification == 1)
            tile_classification[mask] = class_id

        return tile_classification

    def _get_progress_dict(self) -> Dict:
        """Zwraca aktualny stan postępu - NAPRAWIONY ETA bazuje na PUNKTACH nie kafelkach!"""
        elapsed = time.time() - self.start_time

        # Progress w procentach bazuje na PUNKTACH (bardziej dokładne)
        progress_pct = (self.completed_points / self.total_points * 100) if self.total_points > 0 else 0

        # Estymacja czasu pozostałego bazuje na PUNKTACH/SEKUNDA (STABILNE!)
        if self.completed_points > 0 and elapsed > 0:
            points_per_second = self.completed_points / elapsed
            remaining_points = self.total_points - self.completed_points
            eta_seconds = remaining_points / points_per_second
        else:
            eta_seconds = 0

        return {
            'completed_tiles': self.completed_tiles,
            'total_tiles': self.total_tiles,
            'progress_pct': progress_pct,
            'elapsed_seconds': elapsed,
            'eta_seconds': eta_seconds
        }
