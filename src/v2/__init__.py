"""
CPK Cloud Classifier v2.0 - Czysta, modularna architektura

Moduły:
- core: Podstawowe operacje (loader, writer, tiling)
- features: Ekstrakcja cech geometrycznych
- classifiers: 45 klasyfikatorów infrastruktury
- pipeline: Główny pipeline klasyfikacji
- ml: Machine Learning (Random Forest, PointNet, Ensemble)
- railway: Infrastruktura kolejowa (catenary, tracks, poles, signals)
- analysis: Analiza (clearance, terrain, volume)

Przykład użycia:
    from src.v2 import ClassificationPipeline

    pipeline = ClassificationPipeline(
        input_path="input.las",
        output_path="output.las",
        n_threads=4,
        demo_mode=False
    )

    stats = pipeline.run()
"""

from .core import LASLoader, LASWriter, TilingEngine, Tile
from .features import GeometricFeatureExtractor
from .classifiers import (
    BaseClassifier,
    ClassifierRegistry,
    HeightZoneCalculator,
    register_classifier
)
from .pipeline import ClassificationPipeline

__version__ = "2.0.0"
__all__ = [
    'LASLoader',
    'LASWriter',
    'TilingEngine',
    'Tile',
    'GeometricFeatureExtractor',
    'BaseClassifier',
    'ClassifierRegistry',
    'HeightZoneCalculator',
    'register_classifier',
    'ClassificationPipeline'
]
