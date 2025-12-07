"""
Centralna konfiguracja aplikacji CPK Klasyfikator

Wszystkie stałe i parametry w jednym miejscu dla łatwej modyfikacji.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Konfiguracja aplikacji Streamlit"""
    TITLE: str = "Chmura+ | Klasyfikator Chmur Punktów"
    FAVICON: str = "assets/favicon-chmura.png"
    LOGO: str = "assets/logo_chmura.png"
    VERSION: str = "2.0.0"


@dataclass(frozen=True)
class PipelineConfig:
    """Konfiguracja pipeline klasyfikacji"""
    TARGET_POINTS_PER_TILE: int = 500_000
    SAMPLE_RATE: float = 0.005  # 0.5%
    SEARCH_RADIUS: float = 1.0  # metry
    DEMO_TIME_BUDGET: int = 600  # sekund (10 min)
    DEMO_BENCHMARK_SAMPLE: int = 50_000


@dataclass(frozen=True)
class PathConfig:
    """Konfiguracja ścieżek"""
    DATA_DIR: Path = Path("data")
    OUTPUT_DIR: Path = Path("output")
    ASSETS_DIR: Path = Path("assets")


@dataclass(frozen=True)
class UIConfig:
    """Konfiguracja UI"""
    PRIMARY_COLOR: str = "#0A1E42"
    SUCCESS_COLOR: str = "#28a745"
    WARNING_COLOR: str = "#ffc107"
    INFO_BG_COLOR: str = "#e6f0ff"


# Singleton instances
APP = AppConfig()
PIPELINE = PipelineConfig()
PATHS = PathConfig()
UI = UIConfig()
