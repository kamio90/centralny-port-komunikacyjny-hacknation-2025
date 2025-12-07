"""
Clash Detection - Wykrywanie kolizji przestrzennych

Typy kolizji:
- Hard clash - fizyczne przeciecie obiektow
- Soft clash - naruszenie minimalnego odstępu
- Clearance clash - naruszenie skrajni/strefy ochronnej

Zastosowania:
- Kontrola kolizji infrastruktury
- Weryfikacja projektowa
- Analiza BIM

HackNation 2025 - CPK Chmura+
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from scipy.spatial import cKDTree
import logging

logger = logging.getLogger(__name__)


class ClashType(Enum):
    """Typy kolizji"""
    HARD = "hard"  # Fizyczne przeciecie
    SOFT = "soft"  # Naruszenie minimalnego odstępu
    CLEARANCE = "clearance"  # Naruszenie strefy ochronnej


class ClashSeverity(Enum):
    """Waznosc kolizji"""
    CRITICAL = "critical"  # Krytyczna - wymaga natychmiastowej naprawy
    MAJOR = "major"  # Powazna - wymaga naprawy
    MINOR = "minor"  # Drobna - do rozważenia
    INFO = "info"  # Informacyjna


@dataclass
class Clash:
    """Wykryta kolizja"""
    id: int
    clash_type: ClashType
    severity: ClashSeverity
    element_a: str  # Nazwa/typ elementu A
    element_b: str  # Nazwa/typ elementu B
    location: np.ndarray  # (3,) lokalizacja kolizji
    distance: float  # Odleglosc (ujemna = przeciecie)
    overlap_volume: float  # Objetosc przeciecia [m3]
    description: str
    points_involved: int  # Liczba punktow w kolizji


@dataclass
class ClashReport:
    """Raport kolizji"""
    total_clashes: int
    critical_count: int
    major_count: int
    minor_count: int
    clashes_by_type: Dict[str, int]
    clashes_by_elements: Dict[str, int]
    clashes: List[Clash]
    analysis_time_s: float
    total_points_analyzed: int


class ClashDetector:
    """
    Detektor kolizji przestrzennych

    Wykrywa kolizje miedzy:
    - Budynkami a infrastruktura
    - Roslinnoscia a infrastruktura
    - Elementami liniowymi (linie, tory)
    - Strefami ochronnymi

    Usage:
        detector = ClashDetector(coords, classification)
        report = detector.detect_all_clashes(clearance_distance=2.0)
    """

    # Klasy elementow
    CLASS_NAMES = {
        2: "ground",
        3: "low_vegetation",
        4: "medium_vegetation",
        5: "high_vegetation",
        6: "building",
        7: "noise",
        18: "railway",
        19: "wire",
        20: "pole"
    }

    # Pary elementow do sprawdzenia kolizji
    CLASH_PAIRS = [
        (6, 19),   # Budynek - przewody
        (5, 19),   # Roslinnosc wysoka - przewody
        (6, 18),   # Budynek - tory
        (5, 20),   # Roslinnosc - slupy
        (19, 5),   # Przewody - roslinnosc
        (6, 20),   # Budynek - slupy
    ]

    # Minimalne odleglosci [m]
    MIN_CLEARANCES = {
        (6, 19): 3.0,   # Budynek od przewodow
        (5, 19): 2.0,   # Roslinnosc od przewodow
        (6, 18): 5.0,   # Budynek od torow
        (5, 20): 1.0,   # Roslinnosc od slupow
        (19, 5): 2.0,   # Przewody od roslinnosci
        (6, 20): 2.0,   # Budynek od slupow
    }

    def __init__(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        custom_clearances: Optional[Dict] = None
    ):
        """
        Args:
            coords: wspolrzedne punktow
            classification: klasyfikacja
            custom_clearances: niestandardowe odleglosci
        """
        self.coords = coords
        self.classification = classification
        self.clearances = self.MIN_CLEARANCES.copy()
        if custom_clearances:
            self.clearances.update(custom_clearances)

        # Przygotuj indeksy dla kazdej klasy
        self.class_indices = {}
        self.class_trees = {}
        self.class_coords = {}

        for cls in np.unique(classification):
            mask = classification == cls
            if mask.sum() > 0:
                self.class_indices[cls] = np.where(mask)[0]
                self.class_coords[cls] = coords[mask]
                self.class_trees[cls] = cKDTree(coords[mask])

        logger.info(f"Clash detector initialized with {len(coords):,} points")

    def detect_all_clashes(
        self,
        default_clearance: float = 1.0,
        include_soft: bool = True,
        max_clashes: int = 1000
    ) -> ClashReport:
        """
        Wykryj wszystkie kolizje

        Args:
            default_clearance: domyslna minimalna odleglosc [m]
            include_soft: czy uwzgledniac soft clashes
            max_clashes: maksymalna liczba kolizji do zwrocenia

        Returns:
            ClashReport z wykrytymi kolizjami
        """
        import time
        start_time = time.time()

        clashes = []
        clash_id = 0

        for class_a, class_b in self.CLASH_PAIRS:
            if class_a not in self.class_coords or class_b not in self.class_coords:
                continue

            min_dist = self.clearances.get((class_a, class_b), default_clearance)

            logger.info(f"Checking clashes: {self.CLASS_NAMES.get(class_a)} vs {self.CLASS_NAMES.get(class_b)}")

            pair_clashes = self._detect_pair_clashes(
                class_a, class_b,
                min_dist,
                include_soft
            )

            for clash in pair_clashes:
                clash.id = clash_id
                clashes.append(clash)
                clash_id += 1

                if len(clashes) >= max_clashes:
                    logger.warning(f"Max clashes reached ({max_clashes})")
                    break

            if len(clashes) >= max_clashes:
                break

        # Statystyki
        elapsed = time.time() - start_time

        report = ClashReport(
            total_clashes=len(clashes),
            critical_count=sum(1 for c in clashes if c.severity == ClashSeverity.CRITICAL),
            major_count=sum(1 for c in clashes if c.severity == ClashSeverity.MAJOR),
            minor_count=sum(1 for c in clashes if c.severity == ClashSeverity.MINOR),
            clashes_by_type={
                ClashType.HARD.value: sum(1 for c in clashes if c.clash_type == ClashType.HARD),
                ClashType.SOFT.value: sum(1 for c in clashes if c.clash_type == ClashType.SOFT),
                ClashType.CLEARANCE.value: sum(1 for c in clashes if c.clash_type == ClashType.CLEARANCE),
            },
            clashes_by_elements=self._count_by_elements(clashes),
            clashes=clashes,
            analysis_time_s=elapsed,
            total_points_analyzed=len(self.coords)
        )

        logger.info(f"Detected {len(clashes)} clashes in {elapsed:.1f}s")
        return report

    def _detect_pair_clashes(
        self,
        class_a: int,
        class_b: int,
        min_distance: float,
        include_soft: bool
    ) -> List[Clash]:
        """Wykryj kolizje miedzy dwoma klasami"""
        clashes = []

        coords_a = self.class_coords[class_a]
        tree_b = self.class_trees[class_b]

        # Dla kazdego punktu klasy A znajdz najblizszy w B
        distances, indices = tree_b.query(coords_a, k=1)

        # Klastruj kolizje
        collision_mask = distances < min_distance
        if not collision_mask.any():
            return clashes

        # Grupuj kolizje w klastry
        collision_points = coords_a[collision_mask]
        collision_distances = distances[collision_mask]

        if len(collision_points) == 0:
            return clashes

        # Klastruj przestrzennie
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=min_distance, min_samples=3).fit(collision_points)
        labels = clustering.labels_

        for label_id in set(labels):
            if label_id == -1:
                continue

            cluster_mask = labels == label_id
            cluster_points = collision_points[cluster_mask]
            cluster_distances = collision_distances[cluster_mask]

            # Srednia lokalizacja i odleglosc
            location = cluster_points.mean(axis=0)
            avg_distance = cluster_distances.mean()
            min_dist_in_cluster = cluster_distances.min()

            # Okresl typ kolizji
            if min_dist_in_cluster < 0.1:
                clash_type = ClashType.HARD
            elif min_dist_in_cluster < min_distance * 0.5:
                clash_type = ClashType.SOFT
            else:
                clash_type = ClashType.CLEARANCE

            if not include_soft and clash_type == ClashType.SOFT:
                continue

            # Waznosc
            if clash_type == ClashType.HARD:
                severity = ClashSeverity.CRITICAL
            elif min_dist_in_cluster < min_distance * 0.3:
                severity = ClashSeverity.MAJOR
            elif min_dist_in_cluster < min_distance * 0.7:
                severity = ClashSeverity.MINOR
            else:
                severity = ClashSeverity.INFO

            # Objetosc (przyblizona)
            overlap_volume = len(cluster_points) * 0.001  # m3 na punkt

            element_a = self.CLASS_NAMES.get(class_a, f"class_{class_a}")
            element_b = self.CLASS_NAMES.get(class_b, f"class_{class_b}")

            clash = Clash(
                id=0,
                clash_type=clash_type,
                severity=severity,
                element_a=element_a,
                element_b=element_b,
                location=location,
                distance=min_dist_in_cluster,
                overlap_volume=overlap_volume,
                description=f"{element_a} vs {element_b}: {min_dist_in_cluster:.2f}m (min: {min_distance}m)",
                points_involved=len(cluster_points)
            )

            clashes.append(clash)

        return clashes

    def _count_by_elements(self, clashes: List[Clash]) -> Dict[str, int]:
        """Policz kolizje wedlug par elementow"""
        counts = {}
        for clash in clashes:
            key = f"{clash.element_a}-{clash.element_b}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def detect_clearance_violations(
        self,
        reference_points: np.ndarray,
        clearance_profile: np.ndarray,
        axis_direction: np.ndarray
    ) -> List[Clash]:
        """
        Wykryj naruszenia skrajni wzdluz osi

        Args:
            reference_points: punkty osi referencyjnej
            clearance_profile: profil skrajni (odleglosci od osi)
            axis_direction: kierunek osi

        Returns:
            Lista naruszen skrajni
        """
        clashes = []

        # Dla kazdego punktu sprawdz czy jest w skrajni
        tree = cKDTree(reference_points[:, :2])

        for cls in [5, 6]:  # Roslinnosc i budynki
            if cls not in self.class_coords:
                continue

            coords = self.class_coords[cls]
            distances, indices = tree.query(coords[:, :2])

            # Sprawdz profile
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                if idx < len(clearance_profile):
                    allowed_dist = clearance_profile[idx]
                    if dist < allowed_dist:
                        clashes.append(Clash(
                            id=len(clashes),
                            clash_type=ClashType.CLEARANCE,
                            severity=ClashSeverity.MAJOR,
                            element_a=self.CLASS_NAMES.get(cls, f"class_{cls}"),
                            element_b="clearance_zone",
                            location=coords[i],
                            distance=dist - allowed_dist,
                            overlap_volume=0.001,
                            description=f"Clearance violation: {dist:.2f}m < {allowed_dist:.2f}m",
                            points_involved=1
                        ))

        return clashes

    def check_building_infrastructure_clash(
        self,
        building_coords: np.ndarray,
        infrastructure_coords: np.ndarray,
        min_clearance: float = 5.0
    ) -> Optional[Clash]:
        """
        Sprawdz kolizje budynku z infrastruktura

        Args:
            building_coords: punkty budynku
            infrastructure_coords: punkty infrastruktury
            min_clearance: minimalna odleglosc [m]

        Returns:
            Clash jesli wykryto kolizje, None w przeciwnym razie
        """
        tree = cKDTree(infrastructure_coords)
        distances, _ = tree.query(building_coords)

        min_dist = distances.min()

        if min_dist >= min_clearance:
            return None

        # Znajdz lokalizacje kolizji
        closest_idx = np.argmin(distances)
        location = building_coords[closest_idx]

        if min_dist < 0.5:
            clash_type = ClashType.HARD
            severity = ClashSeverity.CRITICAL
        elif min_dist < min_clearance * 0.5:
            clash_type = ClashType.SOFT
            severity = ClashSeverity.MAJOR
        else:
            clash_type = ClashType.CLEARANCE
            severity = ClashSeverity.MINOR

        return Clash(
            id=0,
            clash_type=clash_type,
            severity=severity,
            element_a="building",
            element_b="infrastructure",
            location=location,
            distance=min_dist,
            overlap_volume=(distances < min_clearance).sum() * 0.001,
            description=f"Building-Infrastructure clash: {min_dist:.2f}m",
            points_involved=(distances < min_clearance).sum()
        )

    def generate_clash_report_html(self, report: ClashReport, output_path: str):
        """Generuj raport HTML"""
        html = f"""<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <title>Clash Detection Report - CPK Chmura+</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #c62828 0%, #b71c1c 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0; }}
        .section {{ background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }}
        .metric {{ padding: 15px; border-radius: 8px; text-align: center; }}
        .metric-critical {{ background: #ffebee; color: #c62828; }}
        .metric-major {{ background: #fff3e0; color: #e65100; }}
        .metric-minor {{ background: #fff8e1; color: #f9a825; }}
        .metric-value {{ font-size: 2em; font-weight: bold; }}
        .clash-list {{ max-height: 400px; overflow-y: auto; }}
        .clash-item {{ padding: 10px; border-bottom: 1px solid #eee; }}
        .clash-critical {{ border-left: 4px solid #c62828; }}
        .clash-major {{ border-left: 4px solid #e65100; }}
        .clash-minor {{ border-left: 4px solid #f9a825; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #c62828; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Clash Detection Report</h1>
        <p>CPK Chmura+ - BIM Analysis</p>
        <p>Analyzed {report.total_points_analyzed:,} points in {report.analysis_time_s:.1f}s</p>
    </div>

    <div class="section">
        <h2>Summary</h2>
        <div class="metrics">
            <div class="metric metric-critical">
                <div class="metric-value">{report.critical_count}</div>
                <div>Critical</div>
            </div>
            <div class="metric metric-major">
                <div class="metric-value">{report.major_count}</div>
                <div>Major</div>
            </div>
            <div class="metric metric-minor">
                <div class="metric-value">{report.minor_count}</div>
                <div>Minor</div>
            </div>
            <div class="metric" style="background: #e3f2fd;">
                <div class="metric-value">{report.total_clashes}</div>
                <div>Total</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Clashes by Type</h2>
        <table>
            <tr><th>Type</th><th>Count</th></tr>
            {''.join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in report.clashes_by_type.items())}
        </table>
    </div>

    <div class="section">
        <h2>Clashes by Elements</h2>
        <table>
            <tr><th>Element Pair</th><th>Count</th></tr>
            {''.join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in report.clashes_by_elements.items())}
        </table>
    </div>

    <div class="section">
        <h2>Clash Details</h2>
        <div class="clash-list">
            {''.join(self._clash_to_html(c) for c in report.clashes[:50])}
        </div>
        {f'<p><i>Showing 50 of {report.total_clashes} clashes</i></p>' if report.total_clashes > 50 else ''}
    </div>
</body>
</html>"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Clash report saved to {output_path}")

    def _clash_to_html(self, clash: Clash) -> str:
        """Konwertuj kolizje do HTML"""
        css_class = f"clash-{clash.severity.value}"
        return f"""
        <div class="clash-item {css_class}">
            <strong>#{clash.id}</strong> [{clash.clash_type.value.upper()}] {clash.element_a} vs {clash.element_b}<br>
            <small>Location: ({clash.location[0]:.1f}, {clash.location[1]:.1f}, {clash.location[2]:.1f})</small><br>
            <small>Distance: {clash.distance:.2f}m | Points: {clash.points_involved}</small>
        </div>"""


def detect_clashes(
    coords: np.ndarray,
    classification: np.ndarray,
    clearance: float = 2.0
) -> ClashReport:
    """Convenience function dla wykrywania kolizji"""
    detector = ClashDetector(coords, classification)
    return detector.detect_all_clashes(default_clearance=clearance)
