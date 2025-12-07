"""
Analysis - modul analizy infrastruktury dla CPK

Zawiera narzedzia do:
- Analizy skrajni kolejowej (clearance)
- Ekstrakcji przekrojow poprzecznych
- Generowania DTM/DSM
- Detekcji zarosli w skrajni
- Obliczania objetosci (wykopy/nasypy)
"""

from .railway_clearance import (
    RailwayClearanceAnalyzer,
    ClearanceProfile,
    ClearanceViolation
)

from .terrain_analysis import (
    TerrainAnalyzer,
    CrossSection,
    TerrainModel
)

from .volume_calculator import (
    VolumeCalculator,
    VolumeResult
)

__all__ = [
    'RailwayClearanceAnalyzer',
    'ClearanceProfile',
    'ClearanceViolation',
    'TerrainAnalyzer',
    'CrossSection',
    'TerrainModel',
    'VolumeCalculator',
    'VolumeResult',
]
