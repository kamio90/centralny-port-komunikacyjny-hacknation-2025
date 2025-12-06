"""
Moduł do zapisu chmur punktów LAS/LAZ z klasyfikacją

Tworzy pliki LAS/LAZ zachowując oryginalne metadane
i dodając wyniki klasyfikacji do pola 'classification'.
"""

import laspy
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LASWriter:
    """Zapis chmur punktów z wynikami klasyfikacji"""

    @staticmethod
    def write(
        output_path: str,
        coords: np.ndarray,
        classification: np.ndarray,
        original_header: Optional[laspy.LasHeader] = None,
        colors: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None
    ) -> None:
        """
        Zapisuje chmurę punktów z klasyfikacją do pliku LAS/LAZ

        Args:
            output_path: Ścieżka wyjściowa (*.las lub *.laz)
            coords: (N, 3) Współrzędne XYZ
            classification: (N,) Klasy (0-255)
            original_header: Oryginalny header (jeśli dostępny)
            colors: (N, 3) RGB [0-1] (opcjonalne)
            intensity: (N,) Intensywność [0-1] (opcjonalne)
        """
        output_path = Path(output_path)
        n_points = len(coords)

        logger.info(f"Zapisywanie {n_points:,} punktów do: {output_path.name}")

        # Określ format punktów
        if colors is not None:
            point_format = 3  # XYZ + Intensity + RGB + Classification
        else:
            point_format = 1  # XYZ + Intensity + Classification

        # Stwórz nowy LAS
        if original_header is not None:
            # Użyj oryginalnego headera jako bazy
            header = original_header
            # Zaktualizuj format jeśli potrzeba
            if point_format != header.point_format.id:
                header.point_format = laspy.PointFormat(point_format)
        else:
            # Stwórz nowy header od zera
            header = laspy.LasHeader(point_format=point_format, version="1.2")
            header.offsets = coords.min(axis=0)
            header.scales = [0.001, 0.001, 0.001]  # 1mm precision

        # Stwórz LasData
        las = laspy.LasData(header)

        # Ustaw współrzędne
        las.x = coords[:, 0]
        las.y = coords[:, 1]
        las.z = coords[:, 2]

        # Ustaw klasyfikację - REMAP klas > 31 (LAS format limit)
        # Standard LAS 1.2/1.3 ma tylko 5-bit classification (0-31)
        classification_remapped = classification.copy()

        # Klasy > 31 mapujemy do 19-31 (User Defined)
        overflow_mask = classification_remapped > 31
        if overflow_mask.any():
            overflow_classes = np.unique(classification_remapped[overflow_mask])
            logger.warning(f"UWAGA: Klasy > 31 wykryte: {overflow_classes}")
            logger.warning(f"Remapowanie do zakresu 19-31 (User Defined)")

            # Mapowanie: 32→19, 33→20, ..., 67→31 (z cyklicznym wraparound)
            classification_remapped[overflow_mask] = 19 + ((classification_remapped[overflow_mask] - 32) % 13)

        las.classification = classification_remapped.astype(np.uint8)

        # Ustaw kolory (jeśli dostępne)
        if colors is not None:
            # Konwertuj z [0-1] do uint16 [0-65535]
            las.red = (colors[:, 0] * 65535).astype(np.uint16)
            las.green = (colors[:, 1] * 65535).astype(np.uint16)
            las.blue = (colors[:, 2] * 65535).astype(np.uint16)
            logger.info("Zapisano kolory RGB")

        # Ustaw intensywność (jeśli dostępna)
        if intensity is not None:
            # Konwertuj z [0-1] do uint16
            las.intensity = (intensity * 65535).astype(np.uint16)
            logger.info("Zapisano intensywność")

        # Zapisz do pliku
        las.write(output_path)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Zapisano: {output_path.name} ({file_size_mb:.1f} MB)")

        # Pokaż statystyki klasyfikacji
        unique, counts = np.unique(classification, return_counts=True)
        logger.info(f"Statystyki klasyfikacji:")
        for cls, count in zip(unique, counts):
            pct = count / n_points * 100
            logger.info(f"  Klasa {cls}: {count:,} punktów ({pct:.1f}%)")

    @staticmethod
    def create_classification_report(
        classification: np.ndarray,
        class_names: Dict[int, str],
        output_path: Optional[str] = None
    ) -> str:
        """
        Tworzy raport tekstowy z wynikami klasyfikacji

        Args:
            classification: (N,) Klasy
            class_names: {id: 'Nazwa klasy'}
            output_path: Ścieżka do zapisu raportu (opcjonalne)

        Returns:
            str: Raport tekstowy
        """
        n_points = len(classification)
        unique, counts = np.unique(classification, return_counts=True)

        # Buduj raport
        lines = []
        lines.append("=" * 60)
        lines.append("RAPORT KLASYFIKACJI CHMURY PUNKTÓW")
        lines.append("=" * 60)
        lines.append(f"Całkowita liczba punktów: {n_points:,}")
        lines.append(f"Liczba wykrytych klas: {len(unique)}")
        lines.append("")
        lines.append("Rozkład klasyfikacji:")
        lines.append("-" * 60)

        for cls, count in zip(unique, counts):
            pct = count / n_points * 100
            name = class_names.get(cls, f"Nieznana klasa {cls}")
            lines.append(f"  [{cls:2d}] {name:30s} {count:12,} ({pct:5.2f}%)")

        lines.append("=" * 60)

        report = "\n".join(lines)

        # Zapisz do pliku (jeśli podano)
        if output_path:
            Path(output_path).write_text(report, encoding='utf-8')
            logger.info(f"Raport zapisany: {output_path}")

        return report
