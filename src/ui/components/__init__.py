"""
Komponenty UI - reużywalne elementy interfejsu

Każdy komponent jest odpowiedzialny za jedną funkcjonalność.
"""

from .file_loader import render_file_loader
from .preview import render_preview
from .classification import render_classification
from .hackathon_classification import render_hackathon_classification
from .sidebar import render_sidebar
from .header import render_header
from .footer import render_footer

__all__ = [
    'render_file_loader',
    'render_preview',
    'render_classification',
    'render_hackathon_classification',
    'render_sidebar',
    'render_header',
    'render_footer',
]
