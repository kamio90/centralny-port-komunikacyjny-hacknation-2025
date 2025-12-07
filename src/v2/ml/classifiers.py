"""
ML Classifiers - Klasyfikatory Machine Learning dla chmur punktow

Zawiera:
- RandomForestPointClassifier - Random Forest
- GradientBoostingClassifier - Gradient Boosting (opcjonalnie)

Klasyfikatory ucza sie na cechach geometrycznych i przewiduja klasy ASPRS.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pickle
import logging

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

logger = logging.getLogger(__name__)


@dataclass
class ClassifierMetrics:
    """Metryki klasyfikatora"""
    accuracy: float
    per_class_accuracy: Dict[int, float]
    confusion_matrix: np.ndarray
    classification_report: str
    feature_importance: Optional[Dict[str, float]] = None


class PointCloudClassifier(ABC):
    """Abstrakcyjna klasa bazowa dla klasyfikatorow"""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PointCloudClassifier':
        """Trenuje klasyfikator"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Przewiduje klasy"""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Przewiduje prawdopodobienstwa klas"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Zapisuje model"""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'PointCloudClassifier':
        """Wczytuje model"""
        pass


class RandomForestPointClassifier(PointCloudClassifier):
    """
    Random Forest Classifier dla chmur punktow

    Zalety:
    - Szybki trening i inference
    - Nie wymaga normalizacji
    - Dostarcza feature importance
    - Odporny na overfitting

    Usage:
        clf = RandomForestPointClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        clf.save("model.pkl")
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        n_jobs: int = -1,
        random_state: int = 42,
        class_weight: str = 'balanced'
    ):
        """
        Args:
            n_estimators: liczba drzew
            max_depth: max glebokosc drzewa (None = bez limitu)
            min_samples_split: min probek do podzialu
            min_samples_leaf: min probek w lisciu
            n_jobs: liczba watkow (-1 = wszystkie)
            random_state: ziarno losowosci
            class_weight: wagi klas ('balanced' dla niezbalansowanych danych)
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight=class_weight,
            verbose=0
        )

        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.classes_: np.ndarray = np.array([])
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        validation_split: float = 0.1
    ) -> 'RandomForestPointClassifier':
        """
        Trenuje klasyfikator

        Args:
            X: (N, F) macierz cech
            y: (N,) etykiety klas
            feature_names: nazwy cech
            validation_split: czesc danych do walidacji

        Returns:
            self
        """
        logger.info(f"Training Random Forest on {len(X):,} samples, {X.shape[1]} features")

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Podziel na train/val
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        # Skalowanie (opcjonalne dla RF, ale pomocne)
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Trening
        logger.info("Fitting Random Forest...")
        self.model.fit(X_train_scaled, y_train)
        self.classes_ = self.model.classes_
        self.is_fitted = True

        # Walidacja
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_acc = self.model.score(X_val_scaled, y_val)
            logger.info(f"Validation accuracy: {val_acc:.4f}")

        # Feature importance
        importances = self.model.feature_importances_
        for name, imp in sorted(zip(self.feature_names, importances), key=lambda x: -x[1])[:5]:
            logger.info(f"  Feature '{name}': {imp:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Przewiduje klasy"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Przewiduje prawdopodobienstwa klas"""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ClassifierMetrics:
        """
        Ewaluuje klasyfikator

        Args:
            X: cechy
            y: prawdziwe etykiety

        Returns:
            ClassifierMetrics
        """
        predictions = self.predict(X)

        acc = accuracy_score(y, predictions)
        cm = confusion_matrix(y, predictions, labels=self.classes_)
        report = classification_report(y, predictions, labels=self.classes_, zero_division=0)

        # Per-class accuracy
        per_class_acc = {}
        for cls in self.classes_:
            mask = y == cls
            if mask.sum() > 0:
                per_class_acc[int(cls)] = accuracy_score(y[mask], predictions[mask])

        # Feature importance
        feature_imp = dict(zip(self.feature_names, self.model.feature_importances_))

        return ClassifierMetrics(
            accuracy=acc,
            per_class_accuracy=per_class_acc,
            confusion_matrix=cm,
            classification_report=report,
            feature_importance=feature_imp
        )

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
        """
        Cross-validation

        Args:
            X: cechy
            y: etykiety
            cv: liczba foldow

        Returns:
            Dict z wynikami
        """
        logger.info(f"Running {cv}-fold cross-validation...")

        X_scaled = self.scaler.fit_transform(X)
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')

        return {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'fold_scores': scores.tolist()
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """Zwraca waznosc cech"""
        if not self.is_fitted:
            return {}
        return dict(zip(self.feature_names, self.model.feature_importances_))

    def save(self, path: str) -> None:
        """Zapisuje model do pliku"""
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'classes': self.classes_,
            'is_fitted': self.is_fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'RandomForestPointClassifier':
        """Wczytuje model z pliku"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        instance = cls()
        instance.model = data['model']
        instance.scaler = data['scaler']
        instance.feature_names = data['feature_names']
        instance.classes_ = data['classes']
        instance.is_fitted = data['is_fitted']

        logger.info(f"Model loaded from {path}")
        return instance


class EnsemblePointClassifier(PointCloudClassifier):
    """
    Ensemble klasyfikator laczacy wiele modeli

    Uzywa glosowania wiekszosciowego lub usredniania prawdopodobienstw.
    """

    def __init__(self, classifiers: List[PointCloudClassifier] = None):
        self.classifiers = classifiers or []
        self.is_fitted = False

    def add_classifier(self, clf: PointCloudClassifier):
        """Dodaje klasyfikator do ensemble"""
        self.classifiers.append(clf)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsemblePointClassifier':
        """Trenuje wszystkie klasyfikatory"""
        for i, clf in enumerate(self.classifiers):
            logger.info(f"Training classifier {i+1}/{len(self.classifiers)}")
            clf.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Przewiduje przez glosowanie wiekszosciowe"""
        if not self.classifiers:
            raise ValueError("No classifiers in ensemble")

        predictions = np.array([clf.predict(X) for clf in self.classifiers])
        # Majority voting
        from scipy.stats import mode
        result, _ = mode(predictions, axis=0, keepdims=False)
        return result.flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Usrednia prawdopodobienstwa"""
        probas = np.array([clf.predict_proba(X) for clf in self.classifiers])
        return probas.mean(axis=0)

    def save(self, path: str) -> None:
        """Zapisuje ensemble"""
        with open(path, 'wb') as f:
            pickle.dump(self.classifiers, f)

    @classmethod
    def load(cls, path: str) -> 'EnsemblePointClassifier':
        """Wczytuje ensemble"""
        with open(path, 'rb') as f:
            classifiers = pickle.load(f)
        instance = cls(classifiers)
        instance.is_fitted = True
        return instance
