"""
Pipeline klasyfikacji chmur punktów

- ClassificationPipeline: Legacy orchestrator (feature-based)
- ProfessionalPipeline: Industry-standard algorithms (CSF, HAG, RANSAC)
"""

from .classification_pipeline import ClassificationPipeline
from .professional_pipeline import (
    ProfessionalPipeline,
    PipelineConfig,
    run_professional_classification
)

__all__ = [
    'ClassificationPipeline',
    'ProfessionalPipeline',
    'PipelineConfig',
    'run_professional_classification'
]
