"""
Railway Clearance Analyzer - Analiza skrajni kolejowej

Skrajnia kolejowa (clearance gauge) definiuje przestrzen wolna od przeszkod
wokol torow kolejowych. Ten modul wykrywa naruszenia skrajni.

Standardy:
- UIC 505-1: Europejski standard skrajni
- PLK: Polski standard (zbliżony do UIC)

Wymiary skrajni (przyblizene):
- Szerokosc: 3.15m na wysokosci peronu
- Wysokosc: 4.65m nad glowka szyny
- Wysokosc trakcji: min 5.0m
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClearanceProfile:
    """Profil skrajni kolejowej"""
    # Polowa szerokosci na danej wysokosci [wysokosc, polowa_szerokosci]
    profile_points: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 1.575),    # Poziom szyny
        (0.4, 1.575),    # Pod peronem
        (0.55, 1.645),   # Krawedz peronu
        (1.1, 1.645),    # Powyzej peronu
        (3.5, 1.575),    # Gorna czesc
        (4.65, 1.435),   # Sufit podstawowy
        (5.3, 0.75),     # Strefa trakcyjna (przewody)
    ])

    # Dodatkowy margines bezpieczenstwa [m]
    safety_margin: float = 0.1

    def get_half_width_at_height(self, height: float) -> float:
        """Zwraca polowe szerokosci skrajni na danej wysokosci"""
        points = sorted(self.profile_points, key=lambda x: x[0])

        if height <= points[0][0]:
            return points[0][1]
        if height >= points[-1][0]:
            return points[-1][1]

        # Interpolacja liniowa
        for i in range(len(points) - 1):
            h1, w1 = points[i]
            h2, w2 = points[i + 1]
            if h1 <= height <= h2:
                t = (height - h1) / (h2 - h1)
                return w1 + t * (w2 - w1)

        return points[-1][1]


@dataclass
class ClearanceViolation:
    """Naruszenie skrajni"""
    position: np.ndarray  # [x, y, z]
    distance_to_track: float  # Odleglosc od osi toru
    height_above_rail: float  # Wysokosc nad szyna
    violation_distance: float  # O ile narusza skrajnie [m]
    point_class: int  # Klasa punktu
    severity: str  # 'critical', 'warning', 'minor'


class RailwayClearanceAnalyzer:
    """
    Analizator skrajni kolejowej

    Wykrywa punkty naruszajace skrajnie wzdluz torow kolejowych.

    Usage:
        analyzer = RailwayClearanceAnalyzer(coords, classification)
        violations = analyzer.analyze()
        report = analyzer.generate_report()
    """

    # Klasy zwiazane z torami
    TRACK_CLASSES = [18]  # Tory kolejowe

    # Klasy potencjalnie naruszajace skrajnie
    OBSTACLE_CLASSES = [
        3, 4, 5,   # Roslinnosc
        6,         # Budynki
        17,        # Mosty
        19,        # Linie energetyczne
        20,        # Slupy trakcyjne
        35, 36,    # Znaki, bariery
    ]

    def __init__(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        clearance_profile: Optional[ClearanceProfile] = None,
        track_direction: Optional[np.ndarray] = None
    ):
        """
        Args:
            coords: (N, 3) wspolrzedne XYZ
            classification: (N,) klasy punktow
            clearance_profile: profil skrajni (domyslnie UIC)
            track_direction: kierunek torow [dx, dy] (wykrywany automatycznie)
        """
        self.coords = coords
        self.classification = classification
        self.profile = clearance_profile or ClearanceProfile()
        self.track_direction = track_direction

        self.violations: List[ClearanceViolation] = []
        self.track_points = None
        self.track_center = None
        self.rail_level = None

    def analyze(self, sample_interval: float = 1.0) -> List[ClearanceViolation]:
        """
        Analizuje skrajnie wzdluz torow

        Args:
            sample_interval: co ile metrow sprawdzac [m]

        Returns:
            Lista naruszen skrajni
        """
        logger.info("Analyzing railway clearance...")

        # 1. Znajdz punkty torow
        track_mask = np.isin(self.classification, self.TRACK_CLASSES)
        if not track_mask.any():
            logger.warning("No track points found (class 18)")
            return []

        self.track_points = self.coords[track_mask]

        # 2. Oblicz os torow i poziom szyny
        self.track_center = self.track_points.mean(axis=0)
        self.rail_level = np.percentile(self.track_points[:, 2], 10)  # Dolny percentyl

        logger.info(f"Track center: {self.track_center[:2]}, rail level: {self.rail_level:.2f}m")

        # 3. Wyznacz kierunek torow (PCA)
        if self.track_direction is None:
            self.track_direction = self._estimate_track_direction()

        # 4. Znajdz potencjalne przeszkody
        obstacle_mask = np.isin(self.classification, self.OBSTACLE_CLASSES)
        obstacle_coords = self.coords[obstacle_mask]
        obstacle_classes = self.classification[obstacle_mask]

        if len(obstacle_coords) == 0:
            logger.info("No potential obstacles found")
            return []

        # 5. Sprawdz kazda przeszkode
        self.violations = []

        for i in range(len(obstacle_coords)):
            point = obstacle_coords[i]
            point_class = obstacle_classes[i]

            # Odleglosc od osi toru (prostopadle)
            distance = self._distance_to_track_axis(point)

            # Wysokosc nad szyna
            height = point[2] - self.rail_level

            # Sprawdz czy w strefie skrajni
            if height < 0 or height > 6.0:  # Poza zakresem
                continue

            # Dozwolona polowa szerokosci na tej wysokosci
            allowed_half_width = self.profile.get_half_width_at_height(height)
            allowed_half_width -= self.profile.safety_margin

            if distance < allowed_half_width:
                # Naruszenie!
                violation_dist = allowed_half_width - distance

                # Okresl powage
                if violation_dist > 0.3:
                    severity = 'critical'
                elif violation_dist > 0.1:
                    severity = 'warning'
                else:
                    severity = 'minor'

                self.violations.append(ClearanceViolation(
                    position=point,
                    distance_to_track=distance,
                    height_above_rail=height,
                    violation_distance=violation_dist,
                    point_class=int(point_class),
                    severity=severity
                ))

        # Podsumowanie
        n_critical = sum(1 for v in self.violations if v.severity == 'critical')
        n_warning = sum(1 for v in self.violations if v.severity == 'warning')
        n_minor = sum(1 for v in self.violations if v.severity == 'minor')

        logger.info(f"Found {len(self.violations)} violations: "
                   f"{n_critical} critical, {n_warning} warning, {n_minor} minor")

        return self.violations

    def _estimate_track_direction(self) -> np.ndarray:
        """Estymuje kierunek torow za pomoca PCA"""
        if len(self.track_points) < 10:
            return np.array([1.0, 0.0])

        # PCA na XY
        xy = self.track_points[:, :2]
        xy_centered = xy - xy.mean(axis=0)

        # Kowariancja
        cov = np.cov(xy_centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Kierunek glowny
        main_dir = eigenvectors[:, np.argmax(eigenvalues)]
        return main_dir / np.linalg.norm(main_dir)

    def _distance_to_track_axis(self, point: np.ndarray) -> float:
        """Oblicza odleglosc punktu od osi toru (prostopadle)"""
        # Wektor od centrum toru do punktu (tylko XY)
        to_point = point[:2] - self.track_center[:2]

        # Rzut na kierunek prostopadly
        perp_dir = np.array([-self.track_direction[1], self.track_direction[0]])
        distance = abs(np.dot(to_point, perp_dir))

        return distance

    def generate_report(self) -> Dict:
        """Generuje raport z analizy skrajni"""
        if not self.violations:
            return {
                'status': 'OK',
                'message': 'Brak naruszen skrajni',
                'violations': [],
                'summary': {
                    'total': 0,
                    'critical': 0,
                    'warning': 0,
                    'minor': 0
                }
            }

        # Grupuj po klasach
        by_class = {}
        for v in self.violations:
            cls = v.point_class
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(v)

        # Grupuj po severity
        by_severity = {
            'critical': [v for v in self.violations if v.severity == 'critical'],
            'warning': [v for v in self.violations if v.severity == 'warning'],
            'minor': [v for v in self.violations if v.severity == 'minor']
        }

        return {
            'status': 'VIOLATIONS_FOUND',
            'message': f'Wykryto {len(self.violations)} naruszen skrajni',
            'rail_level': float(self.rail_level) if self.rail_level else None,
            'track_center': self.track_center.tolist() if self.track_center is not None else None,
            'summary': {
                'total': len(self.violations),
                'critical': len(by_severity['critical']),
                'warning': len(by_severity['warning']),
                'minor': len(by_severity['minor'])
            },
            'by_class': {
                str(cls): len(violations)
                for cls, violations in by_class.items()
            },
            'worst_violations': [
                {
                    'position': v.position.tolist(),
                    'class': v.point_class,
                    'violation_distance_m': round(v.violation_distance, 3),
                    'height_above_rail_m': round(v.height_above_rail, 2),
                    'severity': v.severity
                }
                for v in sorted(self.violations, key=lambda x: -x.violation_distance)[:10]
            ]
        }

    def get_violation_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Zwraca punkty naruszen do wizualizacji

        Returns:
            (coords, severities) - wspolrzedne i poziomy powagi
        """
        if not self.violations:
            return np.array([]).reshape(0, 3), np.array([])

        coords = np.array([v.position for v in self.violations])
        severity_map = {'critical': 3, 'warning': 2, 'minor': 1}
        severities = np.array([severity_map[v.severity] for v in self.violations])

        return coords, severities
