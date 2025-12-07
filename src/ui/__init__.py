"""
Moduł UI - komponenty interfejsu Streamlit

Modularny podział odpowiedzialności:
- styles: CSS i stylowanie
- components: Reużywalne komponenty UI
"""

from .styles import apply_styles
from .components import (
    render_file_loader,
    render_preview,
    render_classification,
    render_hackathon_classification,
    render_sidebar,
    render_header,
    render_footer,
)

__all__ = [
    'apply_styles',
    'render_file_loader',
    'render_preview',
    'render_classification',
    'render_hackathon_classification',
    'render_sidebar',
    'render_header',
    'render_footer',
]
