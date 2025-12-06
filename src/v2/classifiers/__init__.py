"""
System klasyfikatorów infrastruktury CPK

Zawiera:
- BaseClassifier: Bazowa klasa dla klasyfikatorów
- ClassifierRegistry: System rejestracji klasyfikatorów
- 45 klasyfikatorów infrastruktury (auto-rejestrowane)

Użycie:
    from src.v2.classifiers import ClassifierRegistry

    # Pobierz wszystkie zarejestrowane klasyfikatory
    classifiers = ClassifierRegistry.get_all()
"""

from .base import BaseClassifier, ClassifierRegistry, HeightZoneCalculator, register_classifier

# Import wszystkich klasyfikatorów (auto-rejestracja przez dekoratory)
from . import infrastructure_classifiers

__all__ = [
    'BaseClassifier',
    'ClassifierRegistry',
    'HeightZoneCalculator',
    'register_classifier'
]
