"""
Active Learning - Interaktywne uczenie z feedbackiem uzytkownika

Pozwala uzytkownikowi:
1. Wybrac punkty o niskiej pewnosci
2. Poprawic ich klasyfikacje
3. Dotrenowac model na poprawionych danych

HackNation 2025 - CPK Chmura+
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time
import pickle

logger = logging.getLogger(__name__)


@dataclass
class CorrectionRecord:
    """Rekord korekty uzytkownika"""
    point_indices: np.ndarray
    original_labels: np.ndarray
    corrected_labels: np.ndarray
    timestamp: float = field(default_factory=time.time)
    confidence_before: Optional[np.ndarray] = None


@dataclass
class ActiveLearningSession:
    """Sesja active learning"""
    corrections: List[CorrectionRecord] = field(default_factory=list)
    total_corrections: int = 0
    retrain_count: int = 0
    improvement_history: List[Dict] = field(default_factory=list)


class UncertaintySampler:
    """
    Strategie wyboru punktow do anotacji

    Wspierane strategie:
    - least_confident: punkty o najnizszej pewnosci
    - margin: punkty z najmniejsza roznica miedzy top-2 klasami
    - entropy: punkty o najwyzszej entropii predykcji
    - random: losowy wybor (baseline)
    """

    @staticmethod
    def least_confident(
        confidence: np.ndarray,
        n_samples: int,
        exclude_indices: Optional[Set[int]] = None
    ) -> np.ndarray:
        """Wybierz punkty o najnizszej pewnosci"""
        if exclude_indices:
            mask = np.ones(len(confidence), dtype=bool)
            mask[list(exclude_indices)] = False
            valid_indices = np.where(mask)[0]
            valid_conf = confidence[valid_indices]
            sorted_idx = np.argsort(valid_conf)[:n_samples]
            return valid_indices[sorted_idx]
        else:
            return np.argsort(confidence)[:n_samples]

    @staticmethod
    def margin_sampling(
        probabilities: np.ndarray,
        n_samples: int,
        exclude_indices: Optional[Set[int]] = None
    ) -> np.ndarray:
        """Wybierz punkty z najmniejsza roznica miedzy top-2 klasami"""
        if probabilities.ndim == 1:
            # Tylko confidence, uzyj least_confident
            return UncertaintySampler.least_confident(probabilities, n_samples, exclude_indices)

        # Sortuj prawdopodobienstwa dla kazdego punktu
        sorted_probs = np.sort(probabilities, axis=1)
        # Roznica miedzy najwyzsza a druga najwyzsza
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]

        if exclude_indices:
            mask = np.ones(len(margins), dtype=bool)
            mask[list(exclude_indices)] = False
            valid_indices = np.where(mask)[0]
            valid_margins = margins[valid_indices]
            sorted_idx = np.argsort(valid_margins)[:n_samples]
            return valid_indices[sorted_idx]
        else:
            return np.argsort(margins)[:n_samples]

    @staticmethod
    def entropy_sampling(
        probabilities: np.ndarray,
        n_samples: int,
        exclude_indices: Optional[Set[int]] = None
    ) -> np.ndarray:
        """Wybierz punkty o najwyzszej entropii"""
        if probabilities.ndim == 1:
            return UncertaintySampler.least_confident(1 - probabilities, n_samples, exclude_indices)

        # Oblicz entropie
        # Dodaj maly epsilon zeby uniknac log(0)
        eps = 1e-10
        entropy = -np.sum(probabilities * np.log(probabilities + eps), axis=1)

        if exclude_indices:
            mask = np.ones(len(entropy), dtype=bool)
            mask[list(exclude_indices)] = False
            valid_indices = np.where(mask)[0]
            valid_entropy = entropy[valid_indices]
            sorted_idx = np.argsort(valid_entropy)[::-1][:n_samples]
            return valid_indices[sorted_idx]
        else:
            return np.argsort(entropy)[::-1][:n_samples]

    @staticmethod
    def diversity_sampling(
        coords: np.ndarray,
        confidence: np.ndarray,
        n_samples: int,
        n_candidates: int = 1000,
        exclude_indices: Optional[Set[int]] = None
    ) -> np.ndarray:
        """
        Wybierz roznorodne punkty o niskiej pewnosci

        Najpierw wybiera kandydatow o niskiej pewnosci,
        potem wybiera z nich punkty maksymalizujac roznorodnosc przestrzenna.
        """
        # Najpierw wybierz kandydatow o niskiej pewnosci
        candidates = UncertaintySampler.least_confident(
            confidence, min(n_candidates, len(confidence)), exclude_indices
        )

        if len(candidates) <= n_samples:
            return candidates

        # Greedy diversity selection
        selected = [candidates[0]]
        candidate_coords = coords[candidates]

        for _ in range(n_samples - 1):
            # Oblicz min odleglosc do juz wybranych
            selected_coords = coords[selected]

            min_distances = np.full(len(candidates), np.inf)
            for sel_idx in selected:
                sel_coord = coords[sel_idx]
                distances = np.linalg.norm(candidate_coords - sel_coord, axis=1)
                min_distances = np.minimum(min_distances, distances)

            # Wybierz punkt najdalej od wybranych
            # Ale tylko sposrod kandydatow ktorzy nie sa jeszcze wybrani
            for i, cand in enumerate(candidates):
                if cand in selected:
                    min_distances[i] = -1

            next_idx = candidates[np.argmax(min_distances)]
            selected.append(next_idx)

        return np.array(selected)


class ActiveLearningManager:
    """
    Manager Active Learning

    Zarzadza sesja active learning, korektami i dotrenowywaniem modelu.
    """

    def __init__(
        self,
        classifier,
        strategy: str = 'least_confident',
        batch_size: int = 100
    ):
        """
        Args:
            classifier: wytrenowany klasyfikator (RF, PointNet, Ensemble)
            strategy: strategia wyboru punktow
            batch_size: ile punktow do anotacji na raz
        """
        self.classifier = classifier
        self.strategy = strategy
        self.batch_size = batch_size
        self.session = ActiveLearningSession()
        self.corrected_indices: Set[int] = set()

        # Dane
        self.coords: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.predictions: Optional[np.ndarray] = None
        self.confidence: Optional[np.ndarray] = None
        self.probabilities: Optional[np.ndarray] = None

    def initialize(
        self,
        coords: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        confidence: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ):
        """Inicjalizuj sesje z danymi"""
        self.coords = coords
        self.labels = labels.copy()  # Kopia - bedziemy modyfikowac
        self.predictions = predictions
        self.confidence = confidence
        self.probabilities = probabilities
        self.corrected_indices = set()

        logger.info(f"Active Learning initialized with {len(coords)} points")

    def get_uncertain_samples(self, n_samples: Optional[int] = None) -> np.ndarray:
        """
        Pobierz indeksy punktow do anotacji

        Returns:
            Indeksy punktow wymagajacych anotacji
        """
        if self.coords is None:
            raise RuntimeError("Call initialize() first")

        n = n_samples or self.batch_size

        if self.strategy == 'least_confident':
            indices = UncertaintySampler.least_confident(
                self.confidence, n, self.corrected_indices
            )
        elif self.strategy == 'margin':
            indices = UncertaintySampler.margin_sampling(
                self.probabilities if self.probabilities is not None else self.confidence,
                n, self.corrected_indices
            )
        elif self.strategy == 'entropy':
            indices = UncertaintySampler.entropy_sampling(
                self.probabilities if self.probabilities is not None else self.confidence,
                n, self.corrected_indices
            )
        elif self.strategy == 'diversity':
            indices = UncertaintySampler.diversity_sampling(
                self.coords, self.confidence, n,
                exclude_indices=self.corrected_indices
            )
        else:
            # Random
            valid = np.array([i for i in range(len(self.coords))
                           if i not in self.corrected_indices])
            indices = np.random.choice(valid, min(n, len(valid)), replace=False)

        return indices

    def apply_corrections(
        self,
        point_indices: np.ndarray,
        new_labels: np.ndarray
    ) -> CorrectionRecord:
        """
        Zastosuj korekty uzytkownika

        Args:
            point_indices: indeksy poprawianych punktow
            new_labels: nowe etykiety

        Returns:
            CorrectionRecord z zapisanymi zmianami
        """
        original = self.labels[point_indices].copy()

        # Zapisz korekty
        record = CorrectionRecord(
            point_indices=point_indices,
            original_labels=original,
            corrected_labels=new_labels,
            confidence_before=self.confidence[point_indices].copy()
        )

        # Zastosuj
        self.labels[point_indices] = new_labels
        self.corrected_indices.update(point_indices.tolist())

        self.session.corrections.append(record)
        self.session.total_corrections += len(point_indices)

        logger.info(f"Applied {len(point_indices)} corrections")

        return record

    def get_correction_stats(self) -> Dict:
        """Statystyki korekt"""
        if not self.session.corrections:
            return {
                'total_corrections': 0,
                'unique_points': 0,
                'by_class': {}
            }

        all_original = np.concatenate([c.original_labels for c in self.session.corrections])
        all_corrected = np.concatenate([c.corrected_labels for c in self.session.corrections])

        # Zlicz korekty per klasa
        by_class = {}
        for orig, corr in zip(all_original, all_corrected):
            key = f"{orig}->{corr}"
            by_class[key] = by_class.get(key, 0) + 1

        return {
            'total_corrections': self.session.total_corrections,
            'unique_points': len(self.corrected_indices),
            'correction_records': len(self.session.corrections),
            'by_class': by_class
        }

    def get_improvement_potential(self) -> Dict:
        """Oszacuj potencjalna poprawe po dotrenowaniu"""
        if self.predictions is None or len(self.corrected_indices) == 0:
            return {'potential_improvement': 0.0}

        corrected_list = list(self.corrected_indices)

        # Ile predykcji bylo blednych
        wrong_before = np.sum(self.predictions[corrected_list] != self.labels[corrected_list])

        # Teraz wszystkie poprawione sa "poprawne" (zgodne z nowymi labelami)
        # Potencjalna poprawa
        improvement = wrong_before / len(self.predictions) * 100

        return {
            'corrected_points': len(corrected_list),
            'wrong_predictions_fixed': int(wrong_before),
            'potential_accuracy_gain': improvement,
            'current_accuracy': np.mean(self.predictions == self.labels) * 100
        }

    def prepare_retrain_data(
        self,
        colors: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
        focus_corrected: bool = True,
        oversample_factor: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Przygotuj dane do dotrenowania

        Args:
            colors: kolory punktow
            intensity: intensywnosc
            focus_corrected: czy nadreprezentowac poprawione punkty
            oversample_factor: ile razy powielac poprawione punkty

        Returns:
            (coords, labels, colors, intensity) - dane do treningu
        """
        if not focus_corrected or len(self.corrected_indices) == 0:
            return self.coords, self.labels, colors, intensity

        # Oversample poprawione punkty
        corrected_list = list(self.corrected_indices)

        # Powtorz poprawione punkty
        repeated_indices = np.repeat(corrected_list, oversample_factor)
        all_indices = np.concatenate([np.arange(len(self.coords)), repeated_indices])

        coords_aug = self.coords[all_indices]
        labels_aug = self.labels[all_indices]
        colors_aug = colors[all_indices] if colors is not None else None
        intensity_aug = intensity[all_indices] if intensity is not None else None

        logger.info(f"Prepared retrain data: {len(coords_aug)} points "
                   f"({len(corrected_list)} corrected x{oversample_factor})")

        return coords_aug, labels_aug, colors_aug, intensity_aug

    def save_session(self, path: str):
        """Zapisz sesje"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump({
                'session': self.session,
                'corrected_indices': self.corrected_indices,
                'labels': self.labels
            }, f)

        logger.info(f"Session saved to {path}")

    def load_session(self, path: str):
        """Wczytaj sesje"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.session = data['session']
        self.corrected_indices = data['corrected_indices']
        if 'labels' in data:
            self.labels = data['labels']

        logger.info(f"Session loaded from {path}")


class QueryByCommittee:
    """
    Query by Committee - wybor punktow przez glosowanie wielu modeli

    Trenuje wiele modeli i wybiera punkty gdzie modele sie nie zgadzaja.
    """

    def __init__(self, n_committee: int = 5):
        self.n_committee = n_committee
        self.models: List = []

    def train_committee(
        self,
        coords: np.ndarray,
        labels: np.ndarray,
        colors: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
        progress_callback=None
    ):
        """Trenuj komitet modeli z bootstrapping"""
        from .classifiers import RandomForestPointClassifier
        from .features import FeatureExtractor

        n_points = len(coords)

        # Ekstrakcja cech raz
        extractor = FeatureExtractor(coords, colors, intensity)
        features = extractor.extract_all()
        X = features.to_array()

        # Usun NaN
        nan_mask = np.isnan(X)
        if nan_mask.any():
            col_means = np.nanmean(X, axis=0)
            for i in range(X.shape[1]):
                X[nan_mask[:, i], i] = col_means[i]

        self.models = []

        for i in range(self.n_committee):
            if progress_callback:
                pct = int((i + 1) / self.n_committee * 100)
                progress_callback("QBC", pct, f"Training model {i+1}/{self.n_committee}")

            # Bootstrap sample
            boot_indices = np.random.choice(n_points, n_points, replace=True)
            X_boot = X[boot_indices]
            y_boot = labels[boot_indices]

            # Trenuj model z innymi hiperparametrami
            model = RandomForestPointClassifier(
                n_estimators=50 + i * 20,
                max_depth=15 + i * 2
            )
            model.fit(X_boot, y_boot, features.feature_names())
            self.models.append(model)

        logger.info(f"Trained committee of {self.n_committee} models")

    def get_disagreement_samples(
        self,
        coords: np.ndarray,
        colors: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
        n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wybierz punkty gdzie modele sie nie zgadzaja

        Returns:
            (indices, disagreement_scores)
        """
        from .features import FeatureExtractor

        if not self.models:
            raise RuntimeError("Train committee first")

        # Ekstrakcja cech
        extractor = FeatureExtractor(coords, colors, intensity)
        features = extractor.extract_all()
        X = features.to_array()

        # Usun NaN
        nan_mask = np.isnan(X)
        if nan_mask.any():
            col_means = np.nanmean(X, axis=0)
            for i in range(X.shape[1]):
                X[nan_mask[:, i], i] = col_means[i]

        # Predykcje wszystkich modeli
        all_preds = np.array([m.predict(X) for m in self.models])

        # Oblicz vote entropy dla kazdego punktu
        n_points = len(coords)
        disagreement = np.zeros(n_points)

        for i in range(n_points):
            votes = all_preds[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            probs = counts / len(votes)
            # Entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            disagreement[i] = entropy

        # Wybierz punkty z najwyzsza niezgodnoscia
        top_indices = np.argsort(disagreement)[::-1][:n_samples]

        return top_indices, disagreement[top_indices]


def create_active_learning_session(
    classifier,
    coords: np.ndarray,
    predictions: np.ndarray,
    confidence: np.ndarray,
    strategy: str = 'least_confident'
) -> ActiveLearningManager:
    """
    Convenience function do tworzenia sesji active learning
    """
    manager = ActiveLearningManager(classifier, strategy=strategy)
    manager.initialize(coords, predictions.copy(), predictions, confidence)
    return manager
