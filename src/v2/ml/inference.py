"""
ML Inference - Inference pipeline dla wytrenowanych modeli

Uzywa wytrenowanego modelu do klasyfikacji nowych chmur punktow.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import time
import logging

from .features import FeatureExtractor
from .classifiers import RandomForestPointClassifier, PointCloudClassifier

logger = logging.getLogger(__name__)


class MLInference:
    """
    Pipeline do inference z modelem ML

    Usage:
        inference = MLInference("model.pkl")
        classification, confidence = inference.predict(coords, colors, intensity)
    """

    def __init__(
        self,
        model_path: str,
        k_neighbors: int = 30,
        batch_size: int = 100000
    ):
        """
        Args:
            model_path: sciezka do modelu
            k_neighbors: liczba sasiadow do ekstrakcji cech
            batch_size: rozmiar batcha dla duzych chmur
        """
        self.model_path = model_path
        self.k_neighbors = k_neighbors
        self.batch_size = batch_size

        # Wczytaj model
        self.classifier = load_model(model_path)
        logger.info(f"Loaded model from {model_path}")

    def predict(
        self,
        coords: np.ndarray,
        colors: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
        ground_mask: Optional[np.ndarray] = None,
        return_confidence: bool = True,
        progress_callback=None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Przewiduje klasy dla chmury punktow

        Args:
            coords: (N, 3) wspolrzedne
            colors: (N, 3) kolory RGB
            intensity: (N,) intensywnosc
            ground_mask: (N,) maska gruntu
            return_confidence: czy zwracac confidence
            progress_callback: callback(step, pct, msg)

        Returns:
            (classification, confidence) - klasyfikacja i opcjonalnie confidence
        """
        n_points = len(coords)
        logger.info(f"ML Inference on {n_points:,} points")

        start_time = time.time()

        if progress_callback:
            progress_callback("ML Inference", 0, "Ekstrakcja cech...")

        # Dla duzych chmur - przetwarzaj w batchach
        if n_points > self.batch_size:
            return self._predict_batched(
                coords, colors, intensity, ground_mask,
                return_confidence, progress_callback
            )

        # Ekstrakcja cech
        extractor = FeatureExtractor(
            coords, colors, intensity, ground_mask,
            k_neighbors=self.k_neighbors
        )
        features = extractor.extract_all()
        X = features.to_array()

        if progress_callback:
            progress_callback("ML Inference", 50, "Klasyfikacja...")

        # Usun NaN (zastap srednia)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            col_means = np.nanmean(X, axis=0)
            for i in range(X.shape[1]):
                X[nan_mask[:, i], i] = col_means[i]

        # Predykcja
        classification = self.classifier.predict(X)

        confidence = None
        if return_confidence:
            proba = self.classifier.predict_proba(X)
            confidence = proba.max(axis=1)

        elapsed = time.time() - start_time

        if progress_callback:
            progress_callback("ML Inference", 100, f"Gotowe! ({n_points/elapsed:.0f} pkt/s)")

        logger.info(f"Inference complete: {elapsed:.2f}s ({n_points/elapsed:.0f} pts/s)")

        return classification, confidence

    def _predict_batched(
        self,
        coords: np.ndarray,
        colors: Optional[np.ndarray],
        intensity: Optional[np.ndarray],
        ground_mask: Optional[np.ndarray],
        return_confidence: bool,
        progress_callback
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predykcja w batchach dla duzych chmur"""
        n_points = len(coords)
        n_batches = (n_points + self.batch_size - 1) // self.batch_size

        logger.info(f"Processing in {n_batches} batches of {self.batch_size:,}")

        all_classifications = np.zeros(n_points, dtype=np.int32)
        all_confidences = np.zeros(n_points) if return_confidence else None

        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, n_points)

            if progress_callback:
                pct = int((batch_idx / n_batches) * 100)
                progress_callback("ML Inference", pct, f"Batch {batch_idx+1}/{n_batches}")

            # Wytnij batch
            batch_coords = coords[start:end]
            batch_colors = colors[start:end] if colors is not None else None
            batch_intensity = intensity[start:end] if intensity is not None else None
            batch_ground = ground_mask[start:end] if ground_mask is not None else None

            # Ekstrakcja cech
            extractor = FeatureExtractor(
                batch_coords, batch_colors, batch_intensity, batch_ground,
                k_neighbors=self.k_neighbors
            )
            features = extractor.extract_all()
            X = features.to_array()

            # Usun NaN
            nan_mask = np.isnan(X)
            if nan_mask.any():
                col_means = np.nanmean(X, axis=0)
                for i in range(X.shape[1]):
                    X[nan_mask[:, i], i] = col_means[i]

            # Predykcja
            batch_class = self.classifier.predict(X)
            all_classifications[start:end] = batch_class

            if return_confidence:
                proba = self.classifier.predict_proba(X)
                all_confidences[start:end] = proba.max(axis=1)

        if progress_callback:
            progress_callback("ML Inference", 100, "Gotowe!")

        return all_classifications, all_confidences

    def get_model_info(self) -> Dict:
        """Zwraca informacje o modelu"""
        return {
            'model_path': self.model_path,
            'n_features': len(self.classifier.feature_names),
            'feature_names': self.classifier.feature_names,
            'classes': self.classifier.classes_.tolist(),
            'n_classes': len(self.classifier.classes_)
        }


def load_model(path: str) -> PointCloudClassifier:
    """Wczytuje model z pliku"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    return RandomForestPointClassifier.load(str(path))


def predict(
    model_path: str,
    coords: np.ndarray,
    colors: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convenience function do predykcji

    Args:
        model_path: sciezka do modelu
        coords: wspolrzedne
        colors: kolory
        intensity: intensywnosc

    Returns:
        klasyfikacja
    """
    inference = MLInference(model_path)
    classification, _ = inference.predict(coords, colors, intensity, return_confidence=False)
    return classification


def predict_with_confidence(
    model_path: str,
    coords: np.ndarray,
    colors: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predykcja z confidence

    Returns:
        (klasyfikacja, confidence)
    """
    inference = MLInference(model_path)
    return inference.predict(coords, colors, intensity, return_confidence=True)
