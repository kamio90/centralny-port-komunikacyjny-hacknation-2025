#!/usr/bin/env python3
"""
CPK Chmura+ Benchmark - Test wydajnosci klasyfikatora

Uruchom:
    python benchmark.py                    # Auto-benchmark na dostepnych plikach
    python benchmark.py --quick            # Szybki test (100k punktow)
    python benchmark.py --full             # Pelny test na wszystkich plikach
    python benchmark.py input.las          # Test na konkretnym pliku
"""

import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from src.v2.core import LASLoader
from src.v2.pipeline import ProfessionalPipeline, PipelineConfig


@dataclass
class BenchmarkResult:
    """Wynik pojedynczego testu"""
    file_name: str
    n_points: int
    processing_time: float
    points_per_second: float
    n_classes: int
    classified_percentage: float
    memory_mb: float
    quality_mode: str


def get_memory_mb() -> float:
    """Pobiera uzycie pamieci w MB"""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def run_benchmark(
    file_path: str,
    sample_size: Optional[int] = None,
    quality: str = "standard"
) -> BenchmarkResult:
    """
    Uruchamia benchmark na pojedynczym pliku

    Args:
        file_path: sciezka do pliku LAS/LAZ
        sample_size: liczba punktow do pobrania (None = wszystkie)
        quality: tryb jakosci (fast/standard/high)

    Returns:
        BenchmarkResult z wynikami
    """
    path = Path(file_path)

    # Wczytaj dane
    loader = LASLoader(str(path))
    data = loader.load(sample_size=sample_size)

    n_points = len(data['coords'])

    # Konfiguracja
    if quality == "fast":
        config = PipelineConfig(
            detect_noise=True,
            classify_ground=True,
            classify_vegetation=True,
            detect_buildings=False,
            detect_infrastructure=True,
            use_fast_noise_detection=True,
            noise_voxel_size=1.0,
            noise_k_neighbors=15,
            hag_grid_resolution=2.0
        )
    elif quality == "high":
        config = PipelineConfig(
            detect_noise=True,
            classify_ground=True,
            classify_vegetation=True,
            detect_buildings=True,
            detect_infrastructure=True,
            use_fast_noise_detection=False,
            noise_k_neighbors=50,
            hag_grid_resolution=0.5
        )
    else:
        config = PipelineConfig(
            detect_noise=True,
            classify_ground=True,
            classify_vegetation=True,
            detect_buildings=True,
            detect_infrastructure=True,
            use_fast_noise_detection=(n_points > 5_000_000)
        )

    # Benchmark
    mem_before = get_memory_mb()
    start_time = time.perf_counter()

    pipeline = ProfessionalPipeline(
        data['coords'],
        data['colors'],
        data['intensity'],
        config
    )

    classification, stats = pipeline.run()

    elapsed = time.perf_counter() - start_time
    mem_after = get_memory_mb()

    # Wyniki
    n_classes = len([k for k in stats.get('classification', {}).keys() if k != 1])
    classified_pct = stats.get('summary', {}).get('classified_percentage', 0)

    return BenchmarkResult(
        file_name=path.name,
        n_points=n_points,
        processing_time=elapsed,
        points_per_second=n_points / elapsed,
        n_classes=n_classes,
        classified_percentage=classified_pct,
        memory_mb=max(0, mem_after - mem_before),
        quality_mode=quality
    )


def find_test_files() -> List[Path]:
    """Znajduje pliki testowe"""
    data_dir = Path(__file__).parent / "data"
    files = []
    if data_dir.exists():
        for ext in ['*.las', '*.laz', '*.LAS', '*.LAZ']:
            files.extend(data_dir.glob(ext))
    return sorted(files, key=lambda x: x.stat().st_size)


def print_result(result: BenchmarkResult, idx: int = 0):
    """Drukuje wynik benchmarku"""
    print(f"\n{'='*60}")
    print(f"Test #{idx+1}: {result.file_name}")
    print(f"{'='*60}")
    print(f"  Punkty:           {result.n_points:,}")
    print(f"  Czas:             {result.processing_time:.2f} s")
    print(f"  Predkosc:         {result.points_per_second:,.0f} pkt/s")
    print(f"  Klasy:            {result.n_classes}")
    print(f"  Sklasyfikowane:   {result.classified_percentage:.1f}%")
    print(f"  Pamiec:           {result.memory_mb:.0f} MB")
    print(f"  Tryb:             {result.quality_mode}")


def print_summary(results: List[BenchmarkResult]):
    """Drukuje podsumowanie wszystkich testow"""
    if not results:
        return

    total_points = sum(r.n_points for r in results)
    total_time = sum(r.processing_time for r in results)
    avg_speed = total_points / total_time if total_time > 0 else 0
    avg_classes = sum(r.n_classes for r in results) / len(results)
    avg_classified = sum(r.classified_percentage for r in results) / len(results)

    print("\n" + "="*60)
    print("PODSUMOWANIE BENCHMARKU")
    print("="*60)
    print(f"  Testow:           {len(results)}")
    print(f"  Punktow lacznie:  {total_points:,}")
    print(f"  Czas lacznie:     {total_time:.2f} s")
    print(f"  Srednia predkosc: {avg_speed:,.0f} pkt/s")
    print(f"  Srednia klas:     {avg_classes:.1f}")
    print(f"  Srednia %:        {avg_classified:.1f}%")
    print("="*60)

    # Performance rating
    if avg_speed > 300_000:
        rating = "SWIETNA"
        emoji = "🚀"
    elif avg_speed > 150_000:
        rating = "DOBRA"
        emoji = "✅"
    elif avg_speed > 80_000:
        rating = "AKCEPTOWALNA"
        emoji = "👍"
    else:
        rating = "DO OPTYMALIZACJI"
        emoji = "⚠️"

    print(f"\n{emoji} Ocena wydajnosci: {rating}")
    print(f"   ({avg_speed/1000:.0f}k punktow/sekunde)")


def main():
    parser = argparse.ArgumentParser(
        description="CPK Chmura+ Benchmark - Test wydajnosci"
    )

    parser.add_argument(
        "input",
        type=str,
        nargs='?',
        default=None,
        help="Plik LAS/LAZ do testu (opcjonalnie)"
    )

    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Szybki test (100k punktow)"
    )

    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Pelny test na wszystkich plikach"
    )

    parser.add_argument(
        "--quality",
        choices=["fast", "standard", "high"],
        default="standard",
        help="Tryb jakosci (default: standard)"
    )

    parser.add_argument(
        "--json", "-j",
        type=str,
        default=None,
        help="Zapisz wyniki do pliku JSON"
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Liczba punktow do probkowania"
    )

    args = parser.parse_args()

    print("="*60)
    print("🔷 CPK CHMURA+ BENCHMARK")
    print("="*60)
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tryb: {args.quality}")

    results: List[BenchmarkResult] = []

    # Okresl sample size
    if args.quick:
        sample_size = 100_000
        print(f"Sample: {sample_size:,} (quick mode)")
    elif args.sample:
        sample_size = args.sample
        print(f"Sample: {sample_size:,}")
    else:
        sample_size = None
        print("Sample: pelny plik")

    try:
        if args.input:
            # Test na podanym pliku
            input_path = Path(args.input)
            if not input_path.exists():
                print(f"\n❌ Plik nie istnieje: {input_path}")
                sys.exit(1)

            print(f"\n🔄 Testuje: {input_path.name}")
            result = run_benchmark(str(input_path), sample_size, args.quality)
            results.append(result)
            print_result(result, 0)

        else:
            # Auto-discover test files
            test_files = find_test_files()

            if not test_files:
                print("\n⚠️ Brak plikow testowych w data/")
                print("   Umiesc pliki LAS/LAZ w folderze data/")
                sys.exit(1)

            print(f"\nZnaleziono {len(test_files)} plikow testowych")

            if not args.full:
                # Tylko pierwszy plik w trybie domyslnym
                test_files = test_files[:1]
                print("(uzyj --full dla wszystkich)")

            for i, file_path in enumerate(test_files):
                print(f"\n🔄 Test {i+1}/{len(test_files)}: {file_path.name}")
                try:
                    result = run_benchmark(str(file_path), sample_size, args.quality)
                    results.append(result)
                    print_result(result, i)
                except Exception as e:
                    print(f"   ❌ Blad: {e}")

        # Podsumowanie
        print_summary(results)

        # Zapis JSON
        if args.json and results:
            json_path = Path(args.json)
            json_path.parent.mkdir(parents=True, exist_ok=True)

            output = {
                "timestamp": datetime.now().isoformat(),
                "quality_mode": args.quality,
                "results": [asdict(r) for r in results]
            }

            with open(json_path, 'w') as f:
                json.dump(output, f, indent=2)

            print(f"\n📊 Wyniki zapisane: {json_path}")

        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⚠️ Przerwano przez uzytkownika")
        if results:
            print_summary(results)
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Blad: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
