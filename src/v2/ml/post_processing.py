"""
Post-Processing - Wygładzanie przestrzenne klasyfikacji

Wspiera:
- Spatial smoothing (k-NN voting)
- Region growing
- Morphological operations
- CRF-like refinement

HackNation 2025 - CPK Chmura+
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.spatial import cKDTree
from scipy import ndimage
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class PostProcessingResult:
    """Wynik post-processingu"""
    classification: np.ndarray
    confidence: np.ndarray
    changes_count: int
    changes_ratio: float
    processing_time: float


class SpatialSmoother:
    """
    Wygladzanie przestrzenne przez glosowanie sasiadow

    Dla kazdego punktu, jesli wiekszosc sasiadow ma inna klase,
    zmien klase punktu na klase wiekszosci.
    """

    def __init__(
        self,
        k_neighbors: int = 15,
        min_agreement: float = 0.6,
        confidence_threshold: float = 0.3
    ):
        """
        Args:
            k_neighbors: liczba sasiadow do glosowania
            min_agreement: minimalny procent zgodnosci sasiadow do zmiany
            confidence_threshold: punkty z confidence ponizej beda smoothowane
        """
        self.k_neighbors = k_neighbors
        self.min_agreement = min_agreement
        self.confidence_threshold = confidence_threshold

    def smooth(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        iterations: int = 1
    ) -> PostProcessingResult:
        """
        Wygladzanie klasyfikacji

        Args:
            coords: (N, 3) wspolrzedne
            classification: (N,) klasyfikacja
            confidence: (N,) pewnosc (opcjonalnie)
            iterations: liczba iteracji

        Returns:
            PostProcessingResult
        """
        start_time = time.time()

        result = classification.copy()
        result_conf = confidence.copy() if confidence is not None else np.ones(len(classification))

        total_changes = 0

        # KD-Tree dla szybkiego wyszukiwania sasiadow
        tree = cKDTree(coords)

        for iteration in range(iterations):
            new_result = result.copy()
            new_conf = result_conf.copy()
            changes = 0

            # Znajdz punkty do smoothowania
            if confidence is not None:
                candidates = np.where(result_conf < self.confidence_threshold)[0]
            else:
                candidates = np.arange(len(coords))

            # Dla kazdego kandydata
            for idx in candidates:
                # Znajdz sasiadow
                _, neighbor_indices = tree.query(coords[idx], k=self.k_neighbors + 1)
                neighbor_indices = neighbor_indices[1:]  # Usun sam punkt

                # Policz glosy
                neighbor_classes = result[neighbor_indices]
                unique, counts = np.unique(neighbor_classes, return_counts=True)
                majority_class = unique[np.argmax(counts)]
                majority_ratio = counts.max() / len(neighbor_indices)

                # Jesli wiekszosc sasiadow ma inna klase
                if majority_class != result[idx] and majority_ratio >= self.min_agreement:
                    new_result[idx] = majority_class
                    new_conf[idx] = majority_ratio
                    changes += 1

            result = new_result
            result_conf = new_conf
            total_changes += changes

            logger.debug(f"Iteration {iteration + 1}: {changes} changes")

            if changes == 0:
                break

        elapsed = time.time() - start_time

        return PostProcessingResult(
            classification=result,
            confidence=result_conf,
            changes_count=total_changes,
            changes_ratio=total_changes / len(classification),
            processing_time=elapsed
        )


class RegionGrowing:
    """
    Region Growing - rozszerzanie regionow od seed points

    Uzyteczne do:
    - Wypelniania luk w klasyfikacji
    - Laczenia fragmentarycznych regionow
    """

    def __init__(
        self,
        distance_threshold: float = 1.0,
        max_iterations: int = 50
    ):
        self.distance_threshold = distance_threshold
        self.max_iterations = max_iterations

    def grow_regions(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        seed_class: int,
        target_class: int = 0
    ) -> PostProcessingResult:
        """
        Rozszerz regiony klasy seed_class na sasiednie punkty klasy target_class

        Args:
            coords: wspolrzedne
            classification: klasyfikacja
            seed_class: klasa do rozszerzenia
            target_class: klasa do zastapienia (0 = niesklasyfikowane)
        """
        start_time = time.time()

        result = classification.copy()
        tree = cKDTree(coords)

        total_changes = 0

        for iteration in range(self.max_iterations):
            # Znajdz punkty seed_class
            seed_indices = np.where(result == seed_class)[0]
            if len(seed_indices) == 0:
                break

            # Znajdz sasiadow w distance_threshold
            changes = 0

            for seed_idx in seed_indices:
                neighbors = tree.query_ball_point(coords[seed_idx], self.distance_threshold)

                for n_idx in neighbors:
                    if result[n_idx] == target_class:
                        result[n_idx] = seed_class
                        changes += 1

            total_changes += changes

            if changes == 0:
                break

        elapsed = time.time() - start_time

        return PostProcessingResult(
            classification=result,
            confidence=np.ones(len(classification)),
            changes_count=total_changes,
            changes_ratio=total_changes / len(classification),
            processing_time=elapsed
        )


class MorphologicalFilter:
    """
    Operacje morfologiczne na klasyfikacji 3D

    Projektuje punkty na voxel grid i stosuje operacje morfologiczne,
    potem mapuje wynik z powrotem na punkty.
    """

    def __init__(self, voxel_size: float = 0.5):
        self.voxel_size = voxel_size

    def _points_to_voxels(
        self,
        coords: np.ndarray,
        classification: np.ndarray
    ) -> Tuple[np.ndarray, Dict, np.ndarray]:
        """Konwertuj punkty na voxel grid"""
        # Normalizuj do voxel indices
        min_coords = coords.min(axis=0)
        voxel_indices = ((coords - min_coords) / self.voxel_size).astype(int)

        # Rozmiar grida
        grid_shape = voxel_indices.max(axis=0) + 1

        # Stworz grid dla kazdej klasy
        unique_classes = np.unique(classification)
        grids = {}

        for cls in unique_classes:
            grid = np.zeros(grid_shape, dtype=bool)
            cls_mask = classification == cls
            cls_indices = voxel_indices[cls_mask]
            grid[cls_indices[:, 0], cls_indices[:, 1], cls_indices[:, 2]] = True
            grids[cls] = grid

        return grids, {'min_coords': min_coords, 'shape': grid_shape}, voxel_indices

    def _voxels_to_points(
        self,
        grids: Dict[int, np.ndarray],
        voxel_indices: np.ndarray,
        original_classification: np.ndarray
    ) -> np.ndarray:
        """Mapuj voxel grid z powrotem na punkty"""
        result = original_classification.copy()

        for i, (vx, vy, vz) in enumerate(voxel_indices):
            # Znajdz klase dla tego voxela
            best_class = original_classification[i]

            for cls, grid in grids.items():
                if grid[vx, vy, vz]:
                    best_class = cls
                    break

            result[i] = best_class

        return result

    def erode(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        target_class: int,
        iterations: int = 1
    ) -> PostProcessingResult:
        """Erozja - usun male/cienkie elementy klasy"""
        start_time = time.time()

        grids, meta, voxel_indices = self._points_to_voxels(coords, classification)

        if target_class in grids:
            struct = ndimage.generate_binary_structure(3, 1)
            grids[target_class] = ndimage.binary_erosion(
                grids[target_class], struct, iterations=iterations
            )

        result = self._voxels_to_points(grids, voxel_indices, classification)
        changes = np.sum(result != classification)

        return PostProcessingResult(
            classification=result,
            confidence=np.ones(len(classification)),
            changes_count=int(changes),
            changes_ratio=changes / len(classification),
            processing_time=time.time() - start_time
        )

    def dilate(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        target_class: int,
        iterations: int = 1
    ) -> PostProcessingResult:
        """Dylatacja - rozszerz regiony klasy"""
        start_time = time.time()

        grids, meta, voxel_indices = self._points_to_voxels(coords, classification)

        if target_class in grids:
            struct = ndimage.generate_binary_structure(3, 1)
            grids[target_class] = ndimage.binary_dilation(
                grids[target_class], struct, iterations=iterations
            )

        result = self._voxels_to_points(grids, voxel_indices, classification)
        changes = np.sum(result != classification)

        return PostProcessingResult(
            classification=result,
            confidence=np.ones(len(classification)),
            changes_count=int(changes),
            changes_ratio=changes / len(classification),
            processing_time=time.time() - start_time
        )

    def opening(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        target_class: int,
        iterations: int = 1
    ) -> PostProcessingResult:
        """Opening (erozja + dylatacja) - usun szum zachowujac ksztalt"""
        start_time = time.time()

        grids, meta, voxel_indices = self._points_to_voxels(coords, classification)

        if target_class in grids:
            struct = ndimage.generate_binary_structure(3, 1)
            grids[target_class] = ndimage.binary_opening(
                grids[target_class], struct, iterations=iterations
            )

        result = self._voxels_to_points(grids, voxel_indices, classification)
        changes = np.sum(result != classification)

        return PostProcessingResult(
            classification=result,
            confidence=np.ones(len(classification)),
            changes_count=int(changes),
            changes_ratio=changes / len(classification),
            processing_time=time.time() - start_time
        )

    def closing(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        target_class: int,
        iterations: int = 1
    ) -> PostProcessingResult:
        """Closing (dylatacja + erozja) - wypelnij luki"""
        start_time = time.time()

        grids, meta, voxel_indices = self._points_to_voxels(coords, classification)

        if target_class in grids:
            struct = ndimage.generate_binary_structure(3, 1)
            grids[target_class] = ndimage.binary_closing(
                grids[target_class], struct, iterations=iterations
            )

        result = self._voxels_to_points(grids, voxel_indices, classification)
        changes = np.sum(result != classification)

        return PostProcessingResult(
            classification=result,
            confidence=np.ones(len(classification)),
            changes_count=int(changes),
            changes_ratio=changes / len(classification),
            processing_time=time.time() - start_time
        )


class CRFRefinement:
    """
    CRF-like refinement dla klasyfikacji chmur punktow

    Uproszczona wersja Conditional Random Fields:
    - Unary potentials z predykcji modelu
    - Pairwise potentials z sasiedztwa przestrzennego
    """

    def __init__(
        self,
        k_neighbors: int = 10,
        spatial_weight: float = 0.3,
        iterations: int = 5
    ):
        """
        Args:
            k_neighbors: sasiedzi do pairwise potentials
            spatial_weight: waga dla spatial consistency (0-1)
            iterations: liczba iteracji mean-field inference
        """
        self.k_neighbors = k_neighbors
        self.spatial_weight = spatial_weight
        self.iterations = iterations

    def refine(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        probabilities: np.ndarray
    ) -> PostProcessingResult:
        """
        Refine klasyfikacje uzywajac CRF

        Args:
            coords: (N, 3) wspolrzedne
            classification: (N,) poczatkowa klasyfikacja
            probabilities: (N, C) prawdopodobienstwa klas

        Returns:
            PostProcessingResult
        """
        start_time = time.time()
        n_points = len(coords)
        n_classes = probabilities.shape[1]

        # KD-Tree
        tree = cKDTree(coords)

        # Q - unary potentials (log probabilities)
        Q = np.log(probabilities + 1e-10)

        # Mean-field iterations
        for iteration in range(self.iterations):
            Q_new = np.zeros_like(Q)

            for i in range(n_points):
                # Unary potential
                Q_new[i] = (1 - self.spatial_weight) * Q[i]

                # Pairwise potential - spatial consistency
                _, neighbors = tree.query(coords[i], k=self.k_neighbors + 1)
                neighbors = neighbors[1:]

                # Message from neighbors
                neighbor_probs = np.exp(Q[neighbors])
                neighbor_probs = neighbor_probs / neighbor_probs.sum(axis=1, keepdims=True)
                message = np.mean(neighbor_probs, axis=0)

                Q_new[i] += self.spatial_weight * np.log(message + 1e-10)

            Q = Q_new

        # Final classification
        result = np.argmax(Q, axis=1)

        # Mapuj z powrotem na oryginalne klasy
        unique_classes = np.unique(classification)
        if len(unique_classes) == n_classes:
            result = unique_classes[result]

        # Confidence
        probs = np.exp(Q)
        probs = probs / probs.sum(axis=1, keepdims=True)
        confidence = probs.max(axis=1)

        changes = np.sum(result != classification)
        elapsed = time.time() - start_time

        return PostProcessingResult(
            classification=result,
            confidence=confidence,
            changes_count=int(changes),
            changes_ratio=changes / n_points,
            processing_time=elapsed
        )


class OutlierRemover:
    """Usuwanie outlierow w klasyfikacji"""

    def __init__(
        self,
        min_cluster_size: int = 10,
        distance_threshold: float = 2.0
    ):
        self.min_cluster_size = min_cluster_size
        self.distance_threshold = distance_threshold

    def remove_isolated_points(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        target_class: int
    ) -> PostProcessingResult:
        """
        Usun izolowane punkty klasy target_class

        Punkty ktore nie maja wystarczajacej liczby sasiadow
        tej samej klasy sa reklasyfikowane.
        """
        start_time = time.time()

        result = classification.copy()
        tree = cKDTree(coords)

        # Punkty klasy docelowej
        target_mask = classification == target_class
        target_indices = np.where(target_mask)[0]

        changes = 0

        for idx in target_indices:
            # Sasiedzi w distance_threshold
            neighbors = tree.query_ball_point(coords[idx], self.distance_threshold)

            # Ile sasiadow tej samej klasy?
            same_class_count = sum(1 for n in neighbors if classification[n] == target_class)

            if same_class_count < self.min_cluster_size:
                # Znajdz najpopularniejsza klase wsrod sasiadow
                neighbor_classes = classification[neighbors]
                neighbor_classes = neighbor_classes[neighbor_classes != target_class]

                if len(neighbor_classes) > 0:
                    unique, counts = np.unique(neighbor_classes, return_counts=True)
                    result[idx] = unique[np.argmax(counts)]
                    changes += 1

        elapsed = time.time() - start_time

        return PostProcessingResult(
            classification=result,
            confidence=np.ones(len(classification)),
            changes_count=changes,
            changes_ratio=changes / len(classification),
            processing_time=elapsed
        )


class PostProcessingPipeline:
    """
    Pipeline laczacy rozne metody post-processingu
    """

    def __init__(self):
        self.steps: List[Tuple[str, callable]] = []

    def add_step(self, name: str, processor: callable):
        """Dodaj krok do pipeline"""
        self.steps.append((name, processor))
        return self

    def add_smoothing(
        self,
        k_neighbors: int = 15,
        min_agreement: float = 0.6
    ):
        """Dodaj spatial smoothing"""
        smoother = SpatialSmoother(k_neighbors, min_agreement)
        self.steps.append(('smoothing', lambda c, cls, conf: smoother.smooth(c, cls, conf)))
        return self

    def add_outlier_removal(
        self,
        target_class: int,
        min_cluster_size: int = 10
    ):
        """Dodaj usuwanie outlierow"""
        remover = OutlierRemover(min_cluster_size)
        self.steps.append((
            f'outlier_removal_{target_class}',
            lambda c, cls, conf, tc=target_class: remover.remove_isolated_points(c, cls, tc)
        ))
        return self

    def process(
        self,
        coords: np.ndarray,
        classification: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        progress_callback=None
    ) -> PostProcessingResult:
        """Wykonaj caly pipeline"""
        start_time = time.time()

        current_classification = classification.copy()
        current_confidence = confidence.copy() if confidence is not None else np.ones(len(classification))
        total_changes = 0

        for i, (name, processor) in enumerate(self.steps):
            if progress_callback:
                pct = int((i + 1) / len(self.steps) * 100)
                progress_callback("Post-processing", pct, f"Step: {name}")

            result = processor(coords, current_classification, current_confidence)
            current_classification = result.classification
            current_confidence = result.confidence
            total_changes += result.changes_count

        elapsed = time.time() - start_time

        return PostProcessingResult(
            classification=current_classification,
            confidence=current_confidence,
            changes_count=total_changes,
            changes_ratio=total_changes / len(classification),
            processing_time=elapsed
        )


def smooth_classification(
    coords: np.ndarray,
    classification: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    k_neighbors: int = 15
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function dla szybkiego smoothowania

    Returns:
        (smoothed_classification, new_confidence)
    """
    smoother = SpatialSmoother(k_neighbors=k_neighbors)
    result = smoother.smooth(coords, classification, confidence)
    return result.classification, result.confidence
