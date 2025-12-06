"""
Bazowa klasa dla klasyfikatorów oraz system rejestracji

System rejestracji pozwala na łatwe dodawanie nowych klas
bez modyfikacji głównego kodu.
"""

import numpy as np
from typing import Dict, Optional, Callable
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class ClassifierRegistry:
    """
    Rejestr klasyfikatorów (Singleton pattern)

    Pozwala na dynamiczną rejestrację klasyfikatorów dla poszczególnych klas.
    Używając dekoratora @register_classifier można łatwo dodawać nowe klasyfikatory.
    """
    _classifiers = {}

    @classmethod
    def register(cls, class_id: int, classifier: Callable):
        """Rejestruje klasyfikator dla danej klasy"""
        cls._classifiers[class_id] = classifier
        logger.info(f"Zarejestrowano klasyfikator dla klasy {class_id}")

    @classmethod
    def get(cls, class_id: int) -> Optional[Callable]:
        """Pobiera klasyfikator dla danej klasy"""
        return cls._classifiers.get(class_id)

    @classmethod
    def get_all(cls) -> Dict[int, Callable]:
        """Zwraca wszystkie zarejestrowane klasyfikatory"""
        return cls._classifiers.copy()

    @classmethod
    def clear(cls):
        """Czyści rejestr"""
        cls._classifiers.clear()


def register_classifier(class_id: int):
    """
    Dekorator do rejestracji klasyfikatorów

    Przykład użycia:
        @register_classifier(class_id=2)
        class GroundClassifier(BaseClassifier):
            ...
    """
    def decorator(classifier_class):
        ClassifierRegistry.register(class_id, classifier_class)
        return classifier_class
    return decorator


class BaseClassifier(ABC):
    """
    Bazowa klasa dla wszystkich klasyfikatorów

    Każdy klasyfikator implementuje logikę rozpoznawania
    konkretnej klasy obiektów (np. grunt, budynki, zieleń).
    """

    def __init__(self,
                 class_id: int,
                 class_name: str,
                 priority: int = 100):
        """
        Args:
            class_id: ID klasy (zgodnie z ASPRS/CPK)
            class_name: Polska nazwa klasy
            priority: Priorytet (1=najwyższy, 100=najniższy)
        """
        self.class_id = class_id
        self.class_name = class_name
        self.priority = priority

    @abstractmethod
    def classify(self,
                 coords: np.ndarray,
                 features: Dict[str, np.ndarray],
                 height_zones: np.ndarray,
                 colors: Optional[np.ndarray] = None,
                 intensity: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Klasyfikuje punkty

        Args:
            coords: (N, 3) Współrzędne XYZ
            features: Dict z cechami geometrycznymi
            height_zones: (N,) Strefy wysokości ['A', 'B', 'C', 'D']
            colors: (N, 3) RGB [0-1] (opcjonalne)
            intensity: (N,) Intensywność [0-1] (opcjonalne)

        Returns:
            (N,) bool mask - True dla punktów należących do tej klasy
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.class_id}, name='{self.class_name}', priority={self.priority})"


class HeightZoneCalculator:
    """
    Kalkulator stref wysokości

    Dzieli punkty na strefy A/B/C/D bazując na percentylach wysokości:
    - A: Poniżej terenu (< 5 percentyl)
    - B: Poziom terenu (5-30 percentyl)
    - C: Średnia wysokość (30-90 percentyl)
    - D: Góra (> 90 percentyl)
    """

    @staticmethod
    def calculate(coords: np.ndarray) -> tuple[np.ndarray, Dict[str, float]]:
        """
        Oblicza strefy wysokości

        Args:
            coords: (N, 3) Współrzędne XYZ

        Returns:
            - zones: (N,) Strefy ['A', 'B', 'C', 'D']
            - thresholds: Dict z progami wysokości
        """
        z = coords[:, 2]

        # Oblicz progi
        z_ground = np.percentile(z, 5)
        z_low = np.percentile(z, 30)
        z_high = np.percentile(z, 90)

        # Przypisz strefy
        zones = np.full(len(coords), 'B', dtype='U1')
        zones[z < z_ground] = 'A'
        zones[(z >= z_low) & (z < z_high)] = 'C'
        zones[z >= z_high] = 'D'

        thresholds = {
            'z_min': z.min(),
            'z_max': z.max(),
            'z_ground': z_ground,
            'z_low': z_low,
            'z_high': z_high
        }

        logger.info(f"Strefy wysokości: A={np.sum(zones=='A'):,}, "
                   f"B={np.sum(zones=='B'):,}, "
                   f"C={np.sum(zones=='C'):,}, "
                   f"D={np.sum(zones=='D'):,}")

        return zones, thresholds
