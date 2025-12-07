"""
Model Comparison - Porownywanie i analiza modeli ML

Funkcjonalnosci:
- Porownanie metryk roznych modeli
- Analiza bledow
- Confusion matrix
- Per-class performance
- Wizualizacja roznic

HackNation 2025 - CPK Chmura+
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ClassMetrics:
    """Metryki dla pojedynczej klasy"""
    class_id: int
    class_name: str
    precision: float
    recall: float
    f1_score: float
    support: int  # liczba probek
    iou: float  # Intersection over Union


@dataclass
class ModelMetrics:
    """Kompletne metryki modelu"""
    model_name: str
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class: List[ClassMetrics]
    confusion_matrix: np.ndarray
    class_names: Dict[int, str]
    predictions: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None


@dataclass
class ComparisonResult:
    """Wynik porownania modeli"""
    models: List[ModelMetrics]
    best_model: str
    best_accuracy: float
    agreement_matrix: np.ndarray  # macierz zgodnosci miedzy modelami
    per_class_winner: Dict[int, str]  # najlepszy model dla kazdej klasy
    error_analysis: Dict


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    class_names: Optional[Dict[int, str]] = None,
    confidence: Optional[np.ndarray] = None
) -> ModelMetrics:
    """
    Oblicz kompletne metryki dla modelu

    Args:
        y_true: prawdziwe etykiety
        y_pred: predykcje
        model_name: nazwa modelu
        class_names: mapowanie id -> nazwa
        confidence: pewnosc predykcji

    Returns:
        ModelMetrics
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    # Domyslne nazwy klas
    if class_names is None:
        class_names = {c: f"Class {c}" for c in classes}

    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for true, pred in zip(y_true, y_pred):
        cm[class_to_idx[true], class_to_idx[pred]] += 1

    # Per-class metrics
    per_class = []

    for cls in classes:
        idx = class_to_idx[cls]

        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # IoU
        union = tp + fp + fn
        iou = tp / union if union > 0 else 0

        support = cm[idx, :].sum()

        per_class.append(ClassMetrics(
            class_id=int(cls),
            class_name=class_names.get(cls, f"Class {cls}"),
            precision=precision,
            recall=recall,
            f1_score=f1,
            support=int(support),
            iou=iou
        ))

    # Overall metrics
    accuracy = np.trace(cm) / cm.sum()

    precisions = [m.precision for m in per_class]
    recalls = [m.recall for m in per_class]
    f1s = [m.f1_score for m in per_class]
    supports = [m.support for m in per_class]

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    # Weighted F1
    total_support = sum(supports)
    weighted_f1 = sum(f * s for f, s in zip(f1s, supports)) / total_support if total_support > 0 else 0

    return ModelMetrics(
        model_name=model_name,
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class=per_class,
        confusion_matrix=cm,
        class_names=class_names,
        predictions=y_pred,
        confidence=confidence
    )


class ModelComparator:
    """
    Porownuje wiele modeli na tych samych danych
    """

    def __init__(self, class_names: Optional[Dict[int, str]] = None):
        """
        Args:
            class_names: mapowanie class_id -> nazwa
        """
        self.class_names = class_names or {
            2: "Grunt",
            3: "Rosl. niska",
            4: "Rosl. srednia",
            5: "Rosl. wysoka",
            6: "Budynek",
            7: "Szum",
            18: "Tory",
            19: "Linie",
            20: "Slupy"
        }
        self.models: Dict[str, ModelMetrics] = {}

    def add_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence: Optional[np.ndarray] = None
    ):
        """Dodaj model do porownania"""
        metrics = calculate_metrics(
            y_true, y_pred, model_name,
            self.class_names, confidence
        )
        self.models[model_name] = metrics
        logger.info(f"Added model '{model_name}': acc={metrics.accuracy:.2%}")

    def compare(self) -> ComparisonResult:
        """
        Porownaj wszystkie modele

        Returns:
            ComparisonResult
        """
        if len(self.models) < 2:
            raise ValueError("Need at least 2 models to compare")

        model_list = list(self.models.values())
        model_names = list(self.models.keys())

        # Best model by accuracy
        best_idx = np.argmax([m.accuracy for m in model_list])
        best_model = model_names[best_idx]
        best_accuracy = model_list[best_idx].accuracy

        # Agreement matrix
        n_models = len(model_list)
        agreement_matrix = np.zeros((n_models, n_models))

        for i in range(n_models):
            for j in range(n_models):
                if model_list[i].predictions is not None and model_list[j].predictions is not None:
                    agreement = np.mean(
                        model_list[i].predictions == model_list[j].predictions
                    )
                    agreement_matrix[i, j] = agreement
                else:
                    agreement_matrix[i, j] = 1.0 if i == j else np.nan

        # Per-class winner
        per_class_winner = {}
        all_classes = set()
        for m in model_list:
            for cm in m.per_class:
                all_classes.add(cm.class_id)

        for cls in all_classes:
            best_f1 = -1
            winner = None
            for name, metrics in self.models.items():
                for cm in metrics.per_class:
                    if cm.class_id == cls and cm.f1_score > best_f1:
                        best_f1 = cm.f1_score
                        winner = name
            per_class_winner[cls] = winner

        # Error analysis
        error_analysis = self._analyze_errors(model_list)

        return ComparisonResult(
            models=model_list,
            best_model=best_model,
            best_accuracy=best_accuracy,
            agreement_matrix=agreement_matrix,
            per_class_winner=per_class_winner,
            error_analysis=error_analysis
        )

    def _analyze_errors(self, models: List[ModelMetrics]) -> Dict:
        """Analiza bledow"""
        if models[0].predictions is None:
            return {}

        n_points = len(models[0].predictions)

        # Punkty gdzie wszystkie modele sie myla
        all_wrong = np.ones(n_points, dtype=bool)

        # Punkty gdzie wszystkie modele sie zgadzaja
        all_agree = np.ones(n_points, dtype=bool)

        first_preds = models[0].predictions
        for m in models:
            if m.predictions is not None:
                # Zakladamy ze wszystkie modele maja te same y_true
                all_agree &= (m.predictions == first_preds)

        # Confusion patterns
        confusion_patterns = defaultdict(int)

        for m in models:
            if m.predictions is not None:
                cm = m.confusion_matrix
                classes = list(self.class_names.keys())

                for i, true_cls in enumerate(classes):
                    if i >= cm.shape[0]:
                        continue
                    for j, pred_cls in enumerate(classes):
                        if j >= cm.shape[1]:
                            continue
                        if i != j and cm[i, j] > 0:
                            key = f"{true_cls}->{pred_cls}"
                            confusion_patterns[key] += cm[i, j]

        # Top confusions
        sorted_confusions = sorted(
            confusion_patterns.items(),
            key=lambda x: -x[1]
        )[:10]

        return {
            'agreement_ratio': float(all_agree.mean()),
            'top_confusions': sorted_confusions,
            'n_samples': n_points
        }

    def get_summary_table(self) -> List[Dict]:
        """Zwroc tabele podsumowujaca"""
        summary = []

        for name, metrics in self.models.items():
            summary.append({
                'Model': name,
                'Accuracy': f"{metrics.accuracy:.2%}",
                'Macro F1': f"{metrics.macro_f1:.2%}",
                'Weighted F1': f"{metrics.weighted_f1:.2%}",
                'Precision': f"{metrics.macro_precision:.2%}",
                'Recall': f"{metrics.macro_recall:.2%}"
            })

        return summary

    def get_per_class_comparison(self) -> Dict[int, Dict[str, float]]:
        """Porownanie F1 per klasa dla wszystkich modeli"""
        comparison = {}

        for name, metrics in self.models.items():
            for cm in metrics.per_class:
                if cm.class_id not in comparison:
                    comparison[cm.class_id] = {'class_name': cm.class_name}
                comparison[cm.class_id][name] = cm.f1_score

        return comparison


class ErrorAnalyzer:
    """
    Szczegolowa analiza bledow klasyfikacji
    """

    def __init__(
        self,
        coords: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        class_names: Optional[Dict[int, str]] = None
    ):
        self.coords = coords
        self.y_true = y_true
        self.y_pred = y_pred
        self.confidence = confidence
        self.class_names = class_names or {}

        # Maska bledow
        self.error_mask = y_true != y_pred
        self.n_errors = self.error_mask.sum()

    def get_error_statistics(self) -> Dict:
        """Statystyki bledow"""
        n_total = len(self.y_true)

        stats = {
            'total_samples': n_total,
            'total_errors': int(self.n_errors),
            'error_rate': self.n_errors / n_total,
            'accuracy': 1 - self.n_errors / n_total
        }

        # Bledy per klasa
        errors_by_true_class = defaultdict(int)
        errors_by_pred_class = defaultdict(int)

        for true, pred, is_error in zip(self.y_true, self.y_pred, self.error_mask):
            if is_error:
                errors_by_true_class[true] += 1
                errors_by_pred_class[pred] += 1

        stats['errors_by_true_class'] = dict(errors_by_true_class)
        stats['errors_by_pred_class'] = dict(errors_by_pred_class)

        # Confidence dla bledow vs poprawnych
        if self.confidence is not None:
            stats['mean_conf_correct'] = float(self.confidence[~self.error_mask].mean())
            stats['mean_conf_errors'] = float(self.confidence[self.error_mask].mean())
            stats['conf_diff'] = stats['mean_conf_correct'] - stats['mean_conf_errors']

        return stats

    def get_spatial_error_clusters(
        self,
        cluster_radius: float = 5.0
    ) -> List[Dict]:
        """
        Znajdz klastry bledow w przestrzeni

        Bledy ktore wystepuja blisko siebie moga wskazywac
        na problematyczne obszary.
        """
        from scipy.spatial import cKDTree

        error_coords = self.coords[self.error_mask]
        error_indices = np.where(self.error_mask)[0]

        if len(error_coords) == 0:
            return []

        tree = cKDTree(error_coords)

        # Znajdz klastry
        visited = set()
        clusters = []

        for i in range(len(error_coords)):
            if i in visited:
                continue

            # Znajdz wszystkie bledy w promieniu
            neighbors = tree.query_ball_point(error_coords[i], cluster_radius)

            if len(neighbors) >= 3:  # Minimum 3 bledy w klastrze
                cluster_indices = [error_indices[n] for n in neighbors]
                visited.update(neighbors)

                # Analizuj klaster
                cluster_true = self.y_true[cluster_indices]
                cluster_pred = self.y_pred[cluster_indices]

                true_classes, true_counts = np.unique(cluster_true, return_counts=True)
                pred_classes, pred_counts = np.unique(cluster_pred, return_counts=True)

                clusters.append({
                    'center': error_coords[neighbors].mean(axis=0).tolist(),
                    'size': len(neighbors),
                    'indices': cluster_indices,
                    'dominant_true_class': int(true_classes[np.argmax(true_counts)]),
                    'dominant_pred_class': int(pred_classes[np.argmax(pred_counts)]),
                    'true_class_distribution': dict(zip(true_classes.tolist(), true_counts.tolist())),
                    'pred_class_distribution': dict(zip(pred_classes.tolist(), pred_counts.tolist()))
                })

        # Sortuj po rozmiarze
        clusters.sort(key=lambda x: -x['size'])

        return clusters

    def get_boundary_errors(self, k_neighbors: int = 10) -> Dict:
        """
        Analiza bledow na granicach klas

        Bledy czesto wystepuja na granicach miedzy klasami.
        """
        from scipy.spatial import cKDTree

        tree = cKDTree(self.coords)

        boundary_errors = 0
        interior_errors = 0

        for i in np.where(self.error_mask)[0]:
            _, neighbors = tree.query(self.coords[i], k=k_neighbors + 1)
            neighbors = neighbors[1:]

            neighbor_classes = self.y_true[neighbors]
            unique_classes = np.unique(neighbor_classes)

            # Jesli sasiedzi maja rozne klasy - to granica
            if len(unique_classes) > 1:
                boundary_errors += 1
            else:
                interior_errors += 1

        total_errors = boundary_errors + interior_errors

        return {
            'boundary_errors': boundary_errors,
            'interior_errors': interior_errors,
            'boundary_error_ratio': boundary_errors / total_errors if total_errors > 0 else 0,
            'total_errors': total_errors
        }

    def get_low_confidence_errors(
        self,
        threshold: float = 0.5
    ) -> Dict:
        """Analiza bledow o niskiej pewnosci"""
        if self.confidence is None:
            return {}

        low_conf_mask = self.confidence < threshold
        low_conf_errors = self.error_mask & low_conf_mask
        high_conf_errors = self.error_mask & ~low_conf_mask

        return {
            'low_conf_errors': int(low_conf_errors.sum()),
            'high_conf_errors': int(high_conf_errors.sum()),
            'low_conf_error_rate': low_conf_errors.sum() / low_conf_mask.sum() if low_conf_mask.sum() > 0 else 0,
            'high_conf_error_rate': high_conf_errors.sum() / (~low_conf_mask).sum() if (~low_conf_mask).sum() > 0 else 0
        }


def quick_compare(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    class_names: Optional[Dict[int, str]] = None
) -> ComparisonResult:
    """
    Szybkie porownanie wielu modeli

    Args:
        y_true: prawdziwe etykiety
        predictions: Dict[nazwa_modelu, predykcje]
        class_names: nazwy klas

    Returns:
        ComparisonResult
    """
    comparator = ModelComparator(class_names)

    for name, preds in predictions.items():
        comparator.add_model(name, y_true, preds)

    return comparator.compare()
