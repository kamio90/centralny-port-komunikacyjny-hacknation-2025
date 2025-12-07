"""
Railway Infrastructure Module - Moduły specjalistyczne dla infrastruktury kolejowej

Zawiera:
- Detekcja sieci trakcyjnej (catenary)
- Ekstrakcja osi torów
- Detekcja słupów i masztów
- Detekcja sygnalizacji
- Generator raportów infrastruktury

HackNation 2025 - CPK Chmura+
"""

from .catenary import (
    CatenaryDetector,
    WireSegment,
    CatenarySystem,
    detect_catenary_wires,
    detect_contact_wire,
    detect_messenger_wire
)

from .track_extraction import (
    TrackExtractor,
    RailAxis,
    TrackSegment,
    extract_rail_axes,
    detect_track_geometry
)

from .pole_detection import (
    PoleDetector,
    Pole,
    PoleType,
    detect_poles,
    classify_pole_type
)

from .signal_detection import (
    SignalDetector,
    RailwaySignal,
    SignalType,
    detect_signals
)

from .infrastructure_report import (
    InfrastructureReporter,
    generate_infrastructure_report,
    export_to_geojson,
    export_to_kml
)

__all__ = [
    # Catenary
    'CatenaryDetector',
    'WireSegment',
    'CatenarySystem',
    'detect_catenary_wires',
    'detect_contact_wire',
    'detect_messenger_wire',

    # Track
    'TrackExtractor',
    'RailAxis',
    'TrackSegment',
    'extract_rail_axes',
    'detect_track_geometry',

    # Poles
    'PoleDetector',
    'Pole',
    'PoleType',
    'detect_poles',
    'classify_pole_type',

    # Signals
    'SignalDetector',
    'RailwaySignal',
    'SignalType',
    'detect_signals',

    # Reports
    'InfrastructureReporter',
    'generate_infrastructure_report',
    'export_to_geojson',
    'export_to_kml',
]
