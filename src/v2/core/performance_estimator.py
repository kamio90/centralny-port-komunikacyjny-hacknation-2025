"""
Performance Estimator - Benchmark dla trybu DEMO

Mierzy wydajność przetwarzania i estymuje maksymalny rozmiar
obszaru do analizy w zadanym budżecie czasowym.

CLEAN, FAST, ACCURATE
"""

import numpy as np
import time
import logging
from typing import Dict, Optional, Tuple

from .tiling_engine import TilingEngine
from ..features import GeometricFeatureExtractor
from ..classifiers import HeightZoneCalculator

logger = logging.getLogger(__name__)


class PerformanceEstimator:
    """
    Estymuje wydajność pipeline i oblicza maksymalny obszar dla trybu DEMO

    Proces:
    1. Pobiera losową próbkę punktów
    2. Mierzy pełny pipeline (tiling + features + classification overhead)
    3. Oblicza punkty/sekunda
    4. Estymuje maksymalny rozmiar obszaru dla budżetu czasowego
    """

    def __init__(self,
                 coords: np.ndarray,
                 colors: Optional[np.ndarray] = None,
                 intensity: Optional[np.ndarray] = None,
                 sample_size: int = 50_000):
        """
        Args:
            coords: (N, 3) Współrzędne XYZ całej chmury
            colors: (N, 3) RGB [0-1] lub None
            intensity: (N,) [0-1] lub None
            sample_size: Rozmiar próbki do benchmarku (domyślnie 50k)
        """
        self.coords = coords
        self.colors = colors
        self.intensity = intensity
        self.n_points = len(coords)
        self.sample_size = min(sample_size, self.n_points)

        # Oblicz gęstość punktów (punkty/m²)
        self.density = self._calculate_density()

        logger.info(f"PerformanceEstimator zainicjalizowany:")
        logger.info(f"  Całkowita liczba punktów: {self.n_points:,}")
        logger.info(f"  Rozmiar próbki benchmarku: {self.sample_size:,}")
        logger.info(f"  Gęstość punktów: {self.density:.1f} pkt/m²")

    def _calculate_density(self) -> float:
        """
        Oblicza średnią gęstość punktów (punkty/m²)

        Returns:
            Gęstość w punktach na metr kwadratowy
        """
        x_range = self.coords[:, 0].max() - self.coords[:, 0].min()
        y_range = self.coords[:, 1].max() - self.coords[:, 1].min()
        area = x_range * y_range

        if area > 0:
            return self.n_points / area
        else:
            return 1000.0  # Fallback

    def benchmark(self) -> Dict:
        """
        Wykonuje benchmark na losowej próbce punktów

        Mierzy:
        - Czas tilingu
        - Czas ekstrakcji cech (PCA, KD-tree)
        - Overhead klasyfikacji

        Returns:
            Dict z metrykami:
                - points_per_second: Prędkość przetwarzania
                - sample_time: Czas benchmarku (sekundy)
                - sample_points: Liczba punktów w próbce
        """
        logger.info("=" * 60)
        logger.info("BENCHMARK - Pomiar wydajności")
        logger.info("=" * 60)

        # 1. Wybierz losową próbkę
        logger.info(f"Losowanie {self.sample_size:,} punktów...")
        sample_indices = np.random.choice(self.n_points, self.sample_size, replace=False)

        sample_coords = self.coords[sample_indices]
        sample_colors = self.colors[sample_indices] if self.colors is not None else None
        sample_intensity = self.intensity[sample_indices] if self.intensity is not None else None

        # 2. Benchmark START
        start_time = time.time()

        # TILING (małe próbki nie wymagają tilingu, ale symulujemy overhead)
        logger.info("  [1/3] Tiling overhead...")
        tiling_engine = TilingEngine(
            coords=sample_coords,
            target_points_per_tile=500_000  # Standardowe parametry
        )
        tiles = tiling_engine.create_tiles()

        # FEATURE EXTRACTION (główny bottleneck)
        logger.info("  [2/3] Feature extraction (PCA + KD-tree)...")
        feature_extractor = GeometricFeatureExtractor(
            coords=sample_coords,
            sample_rate=0.005,  # NORMALNY parametr (0.5%)
            search_radius=1.0   # NORMALNY parametr
        )
        features = feature_extractor.extract()

        # CLASSIFICATION OVERHEAD (mały, ale wliczamy)
        logger.info("  [3/3] Classification overhead...")
        height_zones, _ = HeightZoneCalculator.calculate(sample_coords)

        # 3. Benchmark END
        elapsed = time.time() - start_time

        # 4. Oblicz metryki
        points_per_second = self.sample_size / elapsed

        logger.info(f"  ✓ Benchmark zakończony w {elapsed:.2f}s")
        logger.info(f"  ✓ Wydajność: {points_per_second:,.0f} pkt/s")
        logger.info("=" * 60)

        return {
            'points_per_second': points_per_second,
            'sample_time': elapsed,
            'sample_points': self.sample_size,
            'density': self.density
        }

    def estimate_for_time_budget(self, time_budget_seconds: int = 600) -> Tuple[int, Dict]:
        """
        Estymuje maksymalny rozmiar obszaru dla budżetu czasowego

        Args:
            time_budget_seconds: Budżet czasowy w sekundach (domyślnie 600 = 10 min)

        Returns:
            Tuple:
                - max_points: Maksymalna liczba punktów do przetworzenia
                - metadata: Dict z metadanymi:
                    - area_side: Wymiar kwadratu (metry)
                    - area_m2: Powierzchnia (m²)
                    - estimated_time: Estymowany czas (sekundy)
                    - coverage_pct: Procent całej chmury
        """
        # 1. Benchmark
        benchmark_results = self.benchmark()
        points_per_second = benchmark_results['points_per_second']

        # 2. Oblicz max punktów dla budżetu
        # Zostawiamy 10% marginesu bezpieczeństwa
        # CORRECTION FACTOR: Benchmark na małej próbce jest zbyt optymistyczny
        # Dla dużych chmur wydajność spada (większe kafelki, więcej overhead)
        correction_factor = 0.3  # Realistyczna wydajność to ~30% benchmarku
        max_points_from_benchmark = int(points_per_second * time_budget_seconds * 0.9 * correction_factor)

        # HARD CAP: DEMO max 2 kafelki = 1M punktów (2 × 500k)
        # To da <10 min z PEŁNĄ JAKOŚCIĄ 1:1
        max_points_cap = 1_000_000  # Hard limit dla DEMO

        # Wybierz mniejszą wartość
        max_points = min(max_points_from_benchmark, max_points_cap)

        logger.info(f"  Benchmark sugeruje: {max_points_from_benchmark:,} punktów")
        logger.info(f"  Hard cap (10% chmury): {max_points_cap:,} punktów")
        logger.info(f"  Używam: {max_points:,} punktów (mniejsza wartość)")

        # 3. Oblicz wymiary obszaru
        area_m2 = max_points / self.density
        area_side = np.sqrt(area_m2)  # Kwadrat

        # 4. Estymowany rzeczywisty czas
        estimated_time = max_points / points_per_second

        # 5. Pokrycie
        coverage_pct = (max_points / self.n_points) * 100

        metadata = {
            'area_side': area_side,
            'area_m2': area_m2,
            'estimated_time': estimated_time,
            'coverage_pct': coverage_pct,
            'points_per_second': points_per_second,
            'time_budget': time_budget_seconds
        }

        logger.info(f"ESTYMACJA dla budżetu {time_budget_seconds}s:")
        logger.info(f"  Maksymalna liczba punktów: {max_points:,}")
        logger.info(f"  Wymiar kwadratu: {area_side:.1f}m × {area_side:.1f}m")
        logger.info(f"  Powierzchnia: {area_m2:.1f} m²")
        logger.info(f"  Pokrycie całości: {coverage_pct:.2f}%")
        logger.info(f"  Estymowany czas: {estimated_time:.1f}s ({estimated_time/60:.1f} min)")

        return max_points, metadata
