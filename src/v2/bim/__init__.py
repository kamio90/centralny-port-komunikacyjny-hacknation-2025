"""
BIM Module - Building Information Modeling dla chmur punktow LiDAR

Funkcjonalnosci:
- Ekstrakcja budynkow (Building Extraction)
- Detekcja kolizji (Clash Detection)
- Eksport IFC (Industry Foundation Classes)
- Klasyfikacja LOD (Level of Detail)
- Analiza geometrii 3D
- Raporty BIM

HackNation 2025 - CPK Chmura+
"""

from .building_extraction import (
    BuildingExtractor,
    Building,
    BuildingFootprint,
    RoofType
)

from .clash_detection import (
    ClashDetector,
    Clash,
    ClashType,
    ClashReport
)

from .ifc_export import (
    IFCExporter,
    export_to_ifc,
    BIMElement
)

from .lod_classifier import (
    LODClassifier,
    LODLevel,
    LODResult
)

from .geometry_analyzer import (
    GeometryAnalyzer,
    BoundingBox,
    ConvexHull3D,
    GeometryMetrics
)

__all__ = [
    # Building Extraction
    'BuildingExtractor',
    'Building',
    'BuildingFootprint',
    'RoofType',

    # Clash Detection
    'ClashDetector',
    'Clash',
    'ClashType',
    'ClashReport',

    # IFC Export
    'IFCExporter',
    'export_to_ifc',
    'BIMElement',

    # LOD Classification
    'LODClassifier',
    'LODLevel',
    'LODResult',

    # Geometry Analysis
    'GeometryAnalyzer',
    'BoundingBox',
    'ConvexHull3D',
    'GeometryMetrics',
]
