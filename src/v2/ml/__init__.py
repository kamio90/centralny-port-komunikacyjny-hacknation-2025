"""
ML Module - Machine Learning dla klasyfikacji chmur punktow

Zawiera:
- Ekstrakcja cech geometrycznych (features.py)
- Klasyfikatory ML: Random Forest (classifiers.py)
- Deep Learning: PointNet (pointnet.py)
- Ensemble: Laczenie wielu modeli (ensemble.py)
- Active Learning: Interaktywne uczenie (active_learning.py)
- Auto-tuning: Optymalizacja hiperparametrow (auto_tuning.py)
- Post-processing: Wygladzanie przestrzenne (post_processing.py)
- Model Comparison: Porownywanie modeli (model_comparison.py)
- Pipeline treningowy (training.py)
- Inference (inference.py)
"""

from .features import (
    FeatureExtractor,
    GeometricFeatures,
    extract_point_features
)

from .classifiers import (
    PointCloudClassifier,
    RandomForestPointClassifier,
    # XGBoostPointClassifier,  # opcjonalnie
)

from .training import (
    TrainingPipeline,
    TrainingConfig,
    train_classifier
)

from .inference import (
    MLInference,
    load_model,
    predict
)

from .pointnet import (
    PointNetConfig,
    PointNetTrainer,
    create_pointnet_model,
    is_torch_available,
    get_device_info
)

from .ensemble import (
    EnsembleClassifier,
    EnsembleConfig,
    EnsembleResult,
    create_ensemble
)

from .active_learning import (
    ActiveLearningManager,
    UncertaintySampler,
    QueryByCommittee,
    create_active_learning_session
)

from .auto_tuning import (
    AutoMLPipeline,
    GridSearchTuner,
    RandomSearchTuner,
    CrossValidator,
    quick_tune_rf,
    is_optuna_available
)

from .post_processing import (
    SpatialSmoother,
    RegionGrowing,
    MorphologicalFilter,
    CRFRefinement,
    OutlierRemover,
    PostProcessingPipeline,
    smooth_classification
)

from .model_comparison import (
    ModelComparator,
    ErrorAnalyzer,
    calculate_metrics,
    quick_compare
)

__all__ = [
    # Features
    'FeatureExtractor',
    'GeometricFeatures',
    'extract_point_features',

    # Classifiers
    'PointCloudClassifier',
    'RandomForestPointClassifier',

    # Training
    'TrainingPipeline',
    'TrainingConfig',
    'train_classifier',

    # Inference
    'MLInference',
    'load_model',
    'predict',

    # Deep Learning
    'PointNetConfig',
    'PointNetTrainer',
    'create_pointnet_model',
    'is_torch_available',
    'get_device_info',

    # Ensemble
    'EnsembleClassifier',
    'EnsembleConfig',
    'EnsembleResult',
    'create_ensemble',

    # Active Learning
    'ActiveLearningManager',
    'UncertaintySampler',
    'QueryByCommittee',
    'create_active_learning_session',

    # Auto-tuning
    'AutoMLPipeline',
    'GridSearchTuner',
    'RandomSearchTuner',
    'CrossValidator',
    'quick_tune_rf',
    'is_optuna_available',

    # Post-processing
    'SpatialSmoother',
    'RegionGrowing',
    'MorphologicalFilter',
    'CRFRefinement',
    'OutlierRemover',
    'PostProcessingPipeline',
    'smooth_classification',

    # Model Comparison
    'ModelComparator',
    'ErrorAnalyzer',
    'calculate_metrics',
    'quick_compare',
]
