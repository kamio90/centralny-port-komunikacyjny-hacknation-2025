#!/usr/bin/env python3
"""
CPK Point Cloud Classifier - Command Line Interface

Automatyczna klasyfikacja chmur punktów LAS/LAZ dla CPK.

Usage:
    python cli.py input.las output_classified.las
    python cli.py input.las output.las --report report.json
    python cli.py input.las output.laz --fast --threads 4

Examples:
    # Podstawowe użycie
    python cli.py data/chmura.las output/chmura_classified.las

    # Z raportem JSON
    python cli.py data/chmura.las output/chmura_classified.las --report output/raport.json

    # Tryb szybki (dla dużych chmur)
    python cli.py data/chmura.las output/chmura_classified.laz --fast

    # Bez detekcji budynków (szybciej)
    python cli.py data/chmura.las output/chmura_classified.las --no-buildings
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import laspy

from src.v2.core import LASLoader
from src.v2.pipeline import ProfessionalPipeline, PipelineConfig, BatchClassifier, BatchConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="CPK Point Cloud Classifier - Automatyczna klasyfikacja LAS/LAZ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Klasy ASPRS:
  2  - Grunt
  3  - Roślinność niska (<0.5m)
  4  - Roślinność średnia (0.5-2m)
  5  - Roślinność wysoka (>2m)
  6  - Budynek
  7  - Szum
  18 - Tory kolejowe
  19 - Linie energetyczne
  20 - Słupy trakcyjne
  30 - Droga
  40 - Ściany budynków
  41 - Dachy budynków

Przykłady:
  python cli.py input.las output.las
  python cli.py input.las output.laz --report raport.json --fast
        """
    )

    # Required arguments
    parser.add_argument(
        "input",
        type=str,
        help="Ścieżka do pliku wejściowego LAS/LAZ"
    )
    parser.add_argument(
        "output",
        type=str,
        help="Ścieżka do pliku wyjściowego LAS/LAZ"
    )

    # Optional arguments
    parser.add_argument(
        "--report", "-r",
        type=str,
        default=None,
        help="Ścieżka do pliku raportu JSON (opcjonalne)"
    )
    parser.add_argument(
        "--report-txt",
        type=str,
        default=None,
        help="Ścieżka do pliku raportu TXT (opcjonalne)"
    )
    parser.add_argument(
        "--ifc",
        type=str,
        default=None,
        help="Ścieżka do pliku IFC (opcjonalne, format BIM)"
    )
    parser.add_argument(
        "--geojson",
        type=str,
        default=None,
        help="Ścieżka do pliku GeoJSON (opcjonalne, dla integracji GIS)"
    )
    parser.add_argument(
        "--html",
        type=str,
        default=None,
        help="Ścieżka do pliku HTML (opcjonalne, interaktywny viewer 3D)"
    )

    # Pipeline options
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Tryb szybki (mniej dokładny, ale 2-3x szybszy)"
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Pomiń detekcję szumu"
    )
    parser.add_argument(
        "--no-buildings",
        action="store_true",
        help="Pomiń detekcję budynków (szybciej)"
    )
    parser.add_argument(
        "--no-infrastructure",
        action="store_true",
        help="Pomiń detekcję infrastruktury"
    )

    # Batch mode
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Tryb batch dla dużych plików (>50M punktów) - mniejsze zużycie pamięci"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000_000,
        help="Rozmiar chunka dla trybu batch (domyślnie 10M punktów)"
    )

    # Verbosity
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimalne wyjście (tylko błędy)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Szczegółowe wyjście"
    )

    return parser.parse_args()


def print_progress(step: str, pct: float, msg: str, quiet: bool = False):
    """Print progress to terminal"""
    if quiet:
        return

    # Simple progress bar
    bar_width = 30
    filled = int(bar_width * pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)

    print(f"\r[{bar}] {pct:5.1f}% | {step}: {msg}", end="", flush=True)

    if pct >= 100:
        print()  # New line at 100%


def main():
    """Main CLI entry point"""
    args = parse_args()

    # Setup logging
    import logging
    if args.quiet:
        logging.basicConfig(level=logging.ERROR)
    elif args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    logger = logging.getLogger(__name__)

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Błąd: Plik wejściowy nie istnieje: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not input_path.suffix.lower() in ['.las', '.laz']:
        print(f"❌ Błąd: Nieobsługiwany format pliku: {input_path.suffix}", file=sys.stderr)
        sys.exit(1)

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Print header
    if not args.quiet:
        print("=" * 60)
        print("🔷 CPK POINT CLOUD CLASSIFIER")
        print("=" * 60)
        print(f"📥 Input:  {input_path}")
        print(f"📤 Output: {output_path}")
        if args.report:
            print(f"📊 Report: {args.report}")
        print("-" * 60)

    start_time = time.time()

    try:
        # Check file size for auto-batch mode
        file_info = LASLoader.get_file_info(str(input_path))
        n_points = file_info['n_points']

        # Auto-enable batch mode for large files
        use_batch = args.batch or n_points > 50_000_000

        if use_batch:
            if not args.quiet:
                print(f"📂 Tryb BATCH: {n_points:,} punktów")
                print(f"   Chunk size: {args.chunk_size:,}")

            # Configure pipeline
            config = PipelineConfig(
                detect_noise=not args.no_noise,
                detect_buildings=not args.no_buildings,
                detect_infrastructure=not args.no_infrastructure,
                use_fast_noise_detection=True
            )

            if args.fast:
                config.noise_voxel_size = 1.0
                config.noise_k_neighbors = 15
                config.hag_grid_resolution = 2.0

            batch_config = BatchConfig(chunk_size=args.chunk_size)

            def batch_progress_cb(chunk_idx, n_chunks, pct, msg):
                if not args.quiet:
                    bar_width = 30
                    filled = int(bar_width * pct / 100)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    print(f"\r[{bar}] {pct:5.1f}% | Chunk {chunk_idx+1}/{n_chunks}: {msg}", end="", flush=True)

            classifier = BatchClassifier(
                str(input_path),
                str(output_path),
                pipeline_config=config,
                batch_config=batch_config
            )

            stats = classifier.run(progress_callback=batch_progress_cb if not args.quiet else None)
            classification = None  # Już zapisane przez BatchClassifier

            if not args.quiet:
                print()  # New line after progress bar

        else:
            # Standard mode
            if not args.quiet:
                print("📂 Wczytywanie danych...")

            loader = LASLoader(str(input_path))
            data = loader.load()

            n_points = len(data['coords'])
            if not args.quiet:
                print(f"   ✅ Wczytano {n_points:,} punktów")

            # Configure pipeline
            config = PipelineConfig(
                detect_noise=not args.no_noise,
                detect_buildings=not args.no_buildings,
                detect_infrastructure=not args.no_infrastructure
            )

            if args.fast:
                config.noise_voxel_size = 1.0
                config.noise_k_neighbors = 15
                config.hag_grid_resolution = 2.0

            # Run classification
            if not args.quiet:
                print("\n🔄 Klasyfikacja...")

            pipeline = ProfessionalPipeline(
                coords=data['coords'],
                colors=data['colors'],
                intensity=data['intensity'],
                config=config
            )

            def progress_callback(step, pct, msg):
                print_progress(step, pct, msg, args.quiet)

            classification, stats = pipeline.run(
                progress_callback=progress_callback if not args.quiet else None
            )

            # Save output
            if not args.quiet:
                print("\n💾 Zapisywanie wyników...")

        # Save output (skip if batch mode - already saved)
        if not use_batch:
            with laspy.open(str(input_path)) as src:
                las_orig = src.read()

                # Check if we need LAS 1.4 for extended classification
                max_class = int(classification.max())

                if max_class > 31:
                    # LAS 1.4 with extended classification
                    has_rgb = hasattr(las_orig, 'red') and las_orig.red is not None
                    point_format = 7 if has_rgb else 6

                    header = laspy.LasHeader(point_format=point_format, version="1.4")
                    header.scales = las_orig.header.scales
                    header.offsets = las_orig.header.offsets

                    las = laspy.LasData(header)
                    las.x = las_orig.x
                    las.y = las_orig.y
                    las.z = las_orig.z
                    las.intensity = las_orig.intensity

                    if has_rgb:
                        las.red = las_orig.red
                        las.green = las_orig.green
                        las.blue = las_orig.blue

                    las.classification = classification.astype(np.uint8)
                else:
                    las = las_orig
                    las.classification = classification.astype(np.uint8)

                # Write output
                if str(output_path).endswith('.laz'):
                    las.write(str(output_path), laz_backend=laspy.compression.LazBackend.Laszip)
                else:
                    las.write(str(output_path))

        if not args.quiet:
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ Zapisano: {output_path.name} ({file_size_mb:.1f} MB)")

        # Step 5: Save reports
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)

            # Calculate classified points from stats
            class_stats = stats.get('classification', {})
            classified_pts = sum(
                info['count'] for cls_id, info in class_stats.items()
                if cls_id != 1
            )
            unclassified_pts = class_stats.get(1, {}).get('count', 0)

            report_data = {
                "metadata": {
                    "tool": "CPK Chmura+ Classifier v2.0",
                    "input_file": str(input_path),
                    "output_file": str(output_path),
                    "processing_time_seconds": stats.get('processing_time', 0),
                    "points_per_second": stats.get('points_per_second', 0),
                    "fast_mode": args.fast,
                    "batch_mode": use_batch
                },
                "statistics": {
                    "total_points": n_points,
                    "classified_points": classified_pts,
                    "unclassified_points": unclassified_pts,
                    "classified_percentage": stats.get('summary', {}).get('classified_percentage', 0)
                },
                "classification": class_stats,
                "pipeline_steps": stats.get('steps', {})
            }

            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            if not args.quiet:
                print(f"   ✅ Raport JSON: {report_path}")

        if args.report_txt:
            report_txt_path = Path(args.report_txt)
            report_txt_path.parent.mkdir(parents=True, exist_ok=True)

            lines = [
                "=" * 60,
                "RAPORT KLASYFIKACJI CHMURY PUNKTÓW",
                "=" * 60,
                f"Plik wejściowy: {input_path.name}",
                f"Plik wyjściowy: {output_path.name}",
                f"Całkowita liczba punktów: {n_points:,}",
                f"Czas przetwarzania: {stats.get('processing_time', 0):.1f}s",
                f"Prędkość: {stats.get('points_per_second', 0):,.0f} pkt/s",
                "",
                "Rozkład klasyfikacji:",
                "-" * 60
            ]

            CLASS_NAMES = {
                1: "Nieklasyfikowane",
                2: "Grunt",
                3: "Roślinność niska",
                4: "Roślinność średnia",
                5: "Roślinność wysoka",
                6: "Budynek",
                7: "Szum",
                18: "Tory kolejowe",
                19: "Linie energetyczne",
                20: "Słupy trakcyjne",
                21: "Peron kolejowy",
                30: "Droga",
                32: "Krawężnik",
                35: "Znak drogowy",
                36: "Bariera drogowa",
                40: "Ściany budynków",
                41: "Dachy budynków"
            }

            for cls_id, info in sorted(stats.get('classification', {}).items()):
                name = CLASS_NAMES.get(cls_id, f"Klasa {cls_id}")
                lines.append(f"  [{cls_id:2d}] {name:25s} {info['count']:12,} ({info['percentage']:5.2f}%)")

            lines.append("=" * 60)

            with open(report_txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

            if not args.quiet:
                print(f"   ✅ Raport TXT: {report_txt_path}")

        # Step 6: IFC Export (opcjonalny)
        if args.ifc:
            from src.v2.core import IFCExporter

            ifc_path = Path(args.ifc)
            ifc_path.parent.mkdir(parents=True, exist_ok=True)

            # Wczytaj dane z zapisanego pliku
            with laspy.open(str(output_path)) as src:
                las_out = src.read()
                coords_out = np.vstack([las_out.x, las_out.y, las_out.z]).T
                class_out = np.array(las_out.classification)

            exporter = IFCExporter(
                coords=coords_out,
                classification=class_out,
                project_name=f"CPK Classification - {input_path.stem}",
                project_description="Automatic LiDAR Point Cloud Classification",
                author="CPK Classifier v2.0"
            )

            ifc_stats = exporter.export(str(ifc_path))

            if not args.quiet:
                print(f"   ✅ IFC (BIM): {ifc_path} ({ifc_stats['n_classes']} klas)")

        # Step 7: GeoJSON Export (opcjonalny)
        if args.geojson:
            from src.v2.exporters import GeoJSONExporter

            geojson_path = Path(args.geojson)
            geojson_path.parent.mkdir(parents=True, exist_ok=True)

            # Wczytaj dane z zapisanego pliku
            with laspy.open(str(output_path)) as src:
                las_out = src.read()
                coords_out = np.vstack([las_out.x, las_out.y, las_out.z]).T
                class_out = np.array(las_out.classification)

            exporter = GeoJSONExporter(
                coords=coords_out,
                classification=class_out
            )

            exporter.save(str(geojson_path))

            if not args.quiet:
                n_features = len(np.unique(class_out)) + 1  # klasy + bounding box
                print(f"   ✅ GeoJSON: {geojson_path} ({n_features} features)")

        # Step 8: HTML Viewer Export (opcjonalny)
        if args.html:
            from src.v2.exporters import HTMLViewerExporter

            html_path = Path(args.html)
            html_path.parent.mkdir(parents=True, exist_ok=True)

            # Wczytaj dane z zapisanego pliku
            with laspy.open(str(output_path)) as src:
                las_out = src.read()
                coords_out = np.vstack([las_out.x, las_out.y, las_out.z]).T
                class_out = np.array(las_out.classification)

            exporter = HTMLViewerExporter(
                coords=coords_out,
                classification=class_out,
                title=f"CPK - {input_path.stem}"
            )

            exporter.save(str(html_path))

            if not args.quiet:
                print(f"   ✅ HTML Viewer: {html_path}")

        # Final summary
        elapsed = time.time() - start_time

        if not args.quiet:
            print("\n" + "=" * 60)
            print("✅ KLASYFIKACJA ZAKOŃCZONA")
            print("=" * 60)
            print(f"⏱️  Czas całkowity: {elapsed:.1f}s")
            print(f"🚀 Prędkość: {n_points/elapsed:,.0f} punktów/s")

            # Show class distribution
            print("\n📊 Rozkład klas:")
            for cls_id, info in sorted(stats.get('classification', {}).items()):
                if info['percentage'] >= 0.1:  # Show only classes with >= 0.1%
                    name = {
                        1: "Nieklasyfikowane",
                        2: "Grunt",
                        3: "Roślinność niska",
                        4: "Roślinność średnia",
                        5: "Roślinność wysoka",
                        6: "Budynek",
                        7: "Szum",
                        18: "Tory",
                        19: "Linie energ.",
                        20: "Słupy",
                        21: "Perony",
                        30: "Droga",
                        32: "Krawężniki",
                        35: "Znaki",
                        36: "Bariery",
                        40: "Ściany",
                        41: "Dachy"
                    }.get(cls_id, f"Klasa {cls_id}")
                    print(f"   {cls_id:2d}: {name:15s} {info['percentage']:5.1f}%")

            print("=" * 60)

        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Błąd: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
