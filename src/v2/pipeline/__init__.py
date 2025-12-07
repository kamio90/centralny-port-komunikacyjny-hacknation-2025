"""
Pipeline klasyfikacji chmur punktów

- ClassificationPipeline: Legacy orchestrator (feature-based)
- ProfessionalPipeline: Industry-standard algorithms (CSF, HAG, RANSAC)
- BatchClassifier: Batch processing dla ogromnych chmur (100M+)
"""

from .classification_pipeline import ClassificationPipeline
from .professional_pipeline import (
    ProfessionalPipeline,
    PipelineConfig,
    run_professional_classification
)
from .batch_classifier import (
    BatchClassifier,
    BatchConfig,
    batch_classify
)

__all__ = [
    'ClassificationPipeline',
    'ProfessionalPipeline',
    'PipelineConfig',
    'run_professional_classification',
    'BatchClassifier',
    'BatchConfig',
    'batch_classify'
]
