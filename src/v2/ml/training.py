"""
Training Pipeline - Pipeline do trenowania modeli ML

Zawiera:
- TrainingPipeline - kompletny pipeline treningowy
- TrainingConfig - konfiguracja treningu
- Funkcje pomocnicze do przygotowania danych
"""

import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import logging

from .features import FeatureExtractor, GeometricFeatures
from .classifiers import RandomForestPointClassifier, ClassifierMetrics

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Konfiguracja treningu"""
    # Feature extraction
    k_neighbors: int = 30
    use_colors: bool = True
    use_intensity: bool = True

    # Sampling
    max_samples_per_class: int = 50000
    min_samples_per_class: int = 100
    balance_classes: bool = True

    # Model
    model_type: str = "random_forest"
    n_estimators: int = 100
    max_depth: Optional[int] = 20

    # Training
    validation_split: float = 0.15
    test_split: float = 0.15
    random_state: int = 42

    # Output
    model_path: str = "models/point_classifier.pkl"


@dataclass
class TrainingResult:
    """Wynik treningu"""
    model_path: str
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    training_time: float
    n_samples: int
    n_features: int
    n_classes: int
    feature_names: List[str]
    class_distribution: Dict[int, int]
    metrics: ClassifierMetrics


class TrainingPipeline:
    """
    Pipeline do trenowania klasyfikatora ML

    Usage:
        pipeline = TrainingPipeline(config)
        result = pipeline.train(coords, labels)
        pipeline.save_model("model.pkl")
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.classifier = None
        self.feature_extractor = None
        self.feature_names: List[str] = []

    def train(
        self,
        coords: np.ndarray,
        labels: np.ndarray,
        colors: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
        ground_mask: Optional[np.ndarray] = None,
        progress_callback=None
    ) -> TrainingResult:
        """
        Trenuje model na danych

        Args:
            coords: (N, 3) wspolrzedne
            labels: (N,) etykiety klas
            colors: (N, 3) kolory RGB
            intensity: (N,) intensywnosc
            ground_mask: (N,) maska gruntu
            progress_callback: callback(step, pct, msg)

        Returns:
            TrainingResult
        """
        start_time = time.time()

        logger.info(f"Starting training pipeline on {len(coords):,} points")

        if progress_callback:
            progress_callback("Training", 0, "Inicjalizacja...")

        # 1. Przygotuj dane
        if progress_callback:
            progress_callback("Training", 5, "Przygotowanie danych...")

        X, y, sample_mask = self._prepare_data(coords, labels, colors, intensity, ground_mask)

        logger.info(f"Prepared {len(X):,} samples, {X.shape[1]} features")

        # 2. Podzial train/val/test
        if progress_callback:
            progress_callback("Training", 20, "Podzial danych...")

        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y)

        logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

        # 3. Trenuj model
        if progress_callback:
            progress_callback("Training", 30, "Trening modelu...")

        self.classifier = self._create_classifier()
        self.classifier.fit(X_train, y_train, feature_names=self.feature_names)

        if progress_callback:
            progress_callback("Training", 70, "Ewaluacja...")

        # 4. Ewaluacja
        train_acc = self.classifier.model.score(
            self.classifier.scaler.transform(X_train), y_train
        )
        val_acc = self.classifier.model.score(
            self.classifier.scaler.transform(X_val), y_val
        )
        test_acc = self.classifier.model.score(
            self.classifier.scaler.transform(X_test), y_test
        )

        metrics = self.classifier.evaluate(X_test, y_test)

        logger.info(f"Train accuracy: {train_acc:.4f}")
        logger.info(f"Val accuracy: {val_acc:.4f}")
        logger.info(f"Test accuracy: {test_acc:.4f}")

        # 5. Zapisz model
        if progress_callback:
            progress_callback("Training", 90, "Zapisywanie modelu...")

        model_path = Path(self.config.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.classifier.save(str(model_path))

        # 6. Wynik
        training_time = time.time() - start_time

        if progress_callback:
            progress_callback("Training", 100, "Gotowe!")

        unique_classes, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique_classes.astype(int), counts.astype(int)))

        return TrainingResult(
            model_path=str(model_path),
            train_accuracy=train_acc,
            val_accuracy=val_acc,
            test_accuracy=test_acc,
            training_time=training_time,
            n_samples=len(X),
            n_features=X.shape[1],
            n_classes=len(unique_classes),
            feature_names=self.feature_names,
            class_distribution=class_dist,
            metrics=metrics
        )

    def _prepare_data(
        self,
        coords: np.ndarray,
        labels: np.ndarray,
        colors: Optional[np.ndarray],
        intensity: Optional[np.ndarray],
        ground_mask: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Przygotowuje dane do treningu"""

        # Sampling per class
        unique_classes = np.unique(labels)
        sample_indices = []

        for cls in unique_classes:
            cls_mask = labels == cls
            cls_indices = np.where(cls_mask)[0]

            n_samples = len(cls_indices)

            if n_samples < self.config.min_samples_per_class:
                logger.warning(f"Class {cls} has only {n_samples} samples (min: {self.config.min_samples_per_class})")
                continue

            if n_samples > self.config.max_samples_per_class:
                # Losowe probkowanie
                cls_indices = np.random.choice(
                    cls_indices,
                    self.config.max_samples_per_class,
                    replace=False
                )

            sample_indices.extend(cls_indices)

        sample_indices = np.array(sample_indices)
        np.random.shuffle(sample_indices)

        # Ekstrakcja cech
        sampled_coords = coords[sample_indices]
        sampled_colors = colors[sample_indices] if colors is not None else None
        sampled_intensity = intensity[sample_indices] if intensity is not None else None
        sampled_ground = ground_mask[sample_indices] if ground_mask is not None else None
        sampled_labels = labels[sample_indices]

        self.feature_extractor = FeatureExtractor(
            sampled_coords,
            sampled_colors if self.config.use_colors else None,
            sampled_intensity if self.config.use_intensity else None,
            sampled_ground,
            k_neighbors=self.config.k_neighbors
        )

        features = self.feature_extractor.extract_all(compute_colors=self.config.use_colors)
        self.feature_names = features.feature_names

        X = features.to_array()
        y = sampled_labels

        # Usun NaN
        valid_mask = ~np.isnan(X).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        return X, y, sample_indices[valid_mask]

    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Dzieli dane na train/val/test"""
        from sklearn.model_selection import train_test_split

        # Najpierw wydziel test
        test_size = self.config.test_split
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config.random_state, stratify=y
        )

        # Potem wydziel validation
        val_size = self.config.validation_split / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size,
            random_state=self.config.random_state, stratify=y_trainval
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _create_classifier(self) -> RandomForestPointClassifier:
        """Tworzy klasyfikator"""
        if self.config.model_type == "random_forest":
            return RandomForestPointClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")


def train_classifier(
    coords: np.ndarray,
    labels: np.ndarray,
    colors: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None,
    config: Optional[TrainingConfig] = None,
    progress_callback=None
) -> TrainingResult:
    """
    Convenience function do trenowania

    Args:
        coords: wspolrzedne
        labels: etykiety
        colors: kolory
        intensity: intensywnosc
        config: konfiguracja
        progress_callback: callback

    Returns:
        TrainingResult
    """
    pipeline = TrainingPipeline(config)
    return pipeline.train(coords, labels, colors, intensity, progress_callback=progress_callback)


def save_training_report(result: TrainingResult, path: str):
    """Zapisuje raport z treningu"""
    report = {
        'model_path': result.model_path,
        'accuracy': {
            'train': result.train_accuracy,
            'validation': result.val_accuracy,
            'test': result.test_accuracy
        },
        'training': {
            'time_seconds': result.training_time,
            'n_samples': result.n_samples,
            'n_features': result.n_features,
            'n_classes': result.n_classes
        },
        'features': result.feature_names,
        'class_distribution': result.class_distribution,
        'classification_report': result.metrics.classification_report
    }

    with open(path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
