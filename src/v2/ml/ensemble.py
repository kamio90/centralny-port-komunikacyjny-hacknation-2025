"""
Ensemble Classifier - Laczenie predykcji z wielu modeli

Kombinuje Random Forest i PointNet (jesli dostepny) dla lepszej dokładnosci.
Wspiera rozne strategie voting: hard, soft, weighted.

HackNation 2025 - CPK Chmura+
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time
import pickle

from .features import FeatureExtractor
from .classifiers import RandomForestPointClassifier, PointCloudClassifier
from .pointnet import is_torch_available, PointNetTrainer, PointNetConfig

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Konfiguracja ensemble"""
    voting: str = 'soft'  # 'hard', 'soft', 'weighted'
    weights: Optional[List[float]] = None  # wagi dla weighted voting
    use_rf: bool = True
    use_pointnet: bool = True
    rf_n_estimators: int = 100
    rf_max_depth: int = 20
    pointnet_epochs: int = 30
    pointnet_hidden_dim: int = 256
    k_neighbors: int = 30


@dataclass
class EnsembleResult:
    """Wyniki ensemble"""
    classification: np.ndarray
    confidence: np.ndarray
    model_predictions: Dict[str, np.ndarray]
    model_confidences: Dict[str, np.ndarray]
    agreement_ratio: float  # procent zgody miedzy modelami
    training_time: float = 0.0
    inference_time: float = 0.0


class EnsembleClassifier:
    """
    Ensemble klasyfikator laczacy wiele modeli

    Strategie votingu:
    - hard: klasyfikacja = moda predykcji
    - soft: klasyfikacja = argmax sredniej prawdopodobienstw
    - weighted: jak soft, ale z wagami dla modeli
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.models: Dict[str, Union[PointCloudClassifier, PointNetTrainer]] = {}
        self.is_trained = False
        self.label_mapping = {}
        self.reverse_mapping = {}

    def train(
        self,
        coords: np.ndarray,
        labels: np.ndarray,
        colors: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
        progress_callback=None
    ) -> Dict:
        """
        Trenuje wszystkie modele w ensemble

        Args:
            coords: (N, 3) wspolrzedne
            labels: (N,) etykiety
            colors: (N, 3) kolory RGB
            intensity: (N,) intensywnosc
            progress_callback: callback(step, pct, msg)
        """
        start_time = time.time()
        results = {}

        # Mapowanie etykiet
        unique_classes = np.unique(labels)
        self.label_mapping = {c: i for i, c in enumerate(unique_classes)}
        self.reverse_mapping = {i: c for c, i in self.label_mapping.items()}
        n_classes = len(unique_classes)

        total_steps = int(self.config.use_rf) + int(self.config.use_pointnet and is_torch_available())
        current_step = 0

        # 1. Random Forest
        if self.config.use_rf:
            if progress_callback:
                progress_callback("Ensemble", 0, "Trening Random Forest...")

            logger.info("Training Random Forest...")
            rf_start = time.time()

            # Ekstrakcja cech
            extractor = FeatureExtractor(
                coords, colors, intensity,
                k_neighbors=self.config.k_neighbors
            )
            features = extractor.extract_all()
            X = features.to_array()

            # Usun NaN
            nan_mask = np.isnan(X)
            if nan_mask.any():
                col_means = np.nanmean(X, axis=0)
                for i in range(X.shape[1]):
                    X[nan_mask[:, i], i] = col_means[i]

            # Trenuj RF
            rf = RandomForestPointClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth
            )
            rf.fit(X, labels, features.feature_names())

            self.models['random_forest'] = rf
            rf_time = time.time() - rf_start
            results['rf_time'] = rf_time
            results['rf_val_accuracy'] = rf.validation_accuracy

            current_step += 1
            if progress_callback:
                pct = int(current_step / total_steps * 50)
                progress_callback("Ensemble", pct, f"RF gotowy (acc: {rf.validation_accuracy:.1%})")

            logger.info(f"RF trained in {rf_time:.1f}s, val_acc: {rf.validation_accuracy:.2%}")

        # 2. PointNet
        if self.config.use_pointnet and is_torch_available():
            if progress_callback:
                progress_callback("Ensemble", 50, "Trening PointNet...")

            logger.info("Training PointNet...")
            pn_start = time.time()

            # Oblicz input channels
            input_channels = 3  # xyz
            if colors is not None:
                input_channels += 3
            if intensity is not None:
                input_channels += 1

            config = PointNetConfig(
                n_classes=n_classes,
                input_channels=input_channels,
                hidden_dims=[64, 128, self.config.pointnet_hidden_dim]
            )

            pn_trainer = PointNetTrainer(config)

            def pn_progress(step, pct, msg):
                if progress_callback:
                    overall = 50 + int(pct * 0.45)
                    progress_callback("Ensemble", overall, f"PointNet: {msg}")

            pn_result = pn_trainer.train(
                coords, labels, colors, intensity,
                epochs=self.config.pointnet_epochs,
                progress_callback=pn_progress
            )

            self.models['pointnet'] = pn_trainer
            pn_time = time.time() - pn_start
            results['pn_time'] = pn_time
            results['pn_val_accuracy'] = pn_result['best_val_acc']

            current_step += 1
            if progress_callback:
                progress_callback("Ensemble", 95, f"PointNet gotowy (acc: {pn_result['best_val_acc']:.1%})")

            logger.info(f"PointNet trained in {pn_time:.1f}s, val_acc: {pn_result['best_val_acc']:.2%}")

        elif self.config.use_pointnet and not is_torch_available():
            logger.warning("PointNet requested but PyTorch not available")

        self.is_trained = True
        elapsed = time.time() - start_time

        if progress_callback:
            progress_callback("Ensemble", 100, f"Ensemble gotowy! ({elapsed:.1f}s)")

        results['total_time'] = elapsed
        results['n_models'] = len(self.models)
        results['models'] = list(self.models.keys())

        return results

    def predict(
        self,
        coords: np.ndarray,
        colors: Optional[np.ndarray] = None,
        intensity: Optional[np.ndarray] = None,
        return_details: bool = False,
        progress_callback=None
    ) -> Union[EnsembleResult, Tuple[np.ndarray, np.ndarray]]:
        """
        Predykcja ensemble

        Args:
            coords: wspolrzedne
            colors: kolory
            intensity: intensywnosc
            return_details: czy zwrocic szczegoly
            progress_callback: callback

        Returns:
            EnsembleResult lub (classification, confidence)
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble not trained")

        start_time = time.time()
        n_points = len(coords)

        all_preds = {}
        all_probs = {}

        step = 0
        n_models = len(self.models)

        # Zbierz predykcje z kazdego modelu
        for name, model in self.models.items():
            if progress_callback:
                pct = int(step / n_models * 80)
                progress_callback("Ensemble Inference", pct, f"Model: {name}")

            if name == 'random_forest':
                # Ekstrakcja cech
                extractor = FeatureExtractor(
                    coords, colors, intensity,
                    k_neighbors=self.config.k_neighbors
                )
                features = extractor.extract_all()
                X = features.to_array()

                # Usun NaN
                nan_mask = np.isnan(X)
                if nan_mask.any():
                    col_means = np.nanmean(X, axis=0)
                    for i in range(X.shape[1]):
                        X[nan_mask[:, i], i] = col_means[i]

                preds = model.predict(X)
                probs = model.predict_proba(X)

                all_preds[name] = preds
                all_probs[name] = probs

            elif name == 'pointnet':
                preds, conf = model.predict(
                    coords, colors, intensity,
                    return_confidence=True
                )

                all_preds[name] = preds
                # Dla PointNet mamy tylko confidence, nie pelne probs
                all_probs[name] = conf

            step += 1

        if progress_callback:
            progress_callback("Ensemble Inference", 85, "Laczenie predykcji...")

        # Voting
        if self.config.voting == 'hard':
            final_preds, final_conf = self._hard_voting(all_preds, all_probs)
        elif self.config.voting == 'soft':
            final_preds, final_conf = self._soft_voting(all_preds, all_probs)
        else:  # weighted
            final_preds, final_conf = self._weighted_voting(all_preds, all_probs)

        # Oblicz agreement ratio
        if len(all_preds) > 1:
            models = list(all_preds.keys())
            agreement = np.sum(all_preds[models[0]] == all_preds[models[1]]) / n_points
        else:
            agreement = 1.0

        elapsed = time.time() - start_time

        if progress_callback:
            progress_callback("Ensemble Inference", 100, f"Gotowe! ({elapsed:.1f}s)")

        result = EnsembleResult(
            classification=final_preds,
            confidence=final_conf,
            model_predictions=all_preds,
            model_confidences={k: (v.max(axis=1) if v.ndim == 2 else v) for k, v in all_probs.items()},
            agreement_ratio=agreement,
            inference_time=elapsed
        )

        if return_details:
            return result
        else:
            return final_preds, final_conf

    def _hard_voting(self, preds: Dict, probs: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Hard voting - moda predykcji"""
        models = list(preds.keys())
        n_points = len(preds[models[0]])

        # Stack all predictions
        stacked = np.stack([preds[m] for m in models], axis=0)

        # Moda dla kazdego punktu
        from scipy import stats
        mode_result = stats.mode(stacked, axis=0, keepdims=False)
        final_preds = mode_result.mode.flatten()

        # Confidence = procent modeli ktore sie zgadzaja
        agreement = np.sum(stacked == final_preds, axis=0) / len(models)

        return final_preds, agreement

    def _soft_voting(self, preds: Dict, probs: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Soft voting - srednia prawdopodobienstw"""
        models = list(probs.keys())
        n_points = len(preds[models[0]])

        # Jesli mamy tylko RF - uzyj jego probs
        if 'random_forest' in probs and len(models) == 1:
            rf_probs = probs['random_forest']
            final_preds = preds['random_forest']
            final_conf = rf_probs.max(axis=1)
            return final_preds, final_conf

        # Jesli mamy RF probs i PointNet confidence
        if 'random_forest' in probs:
            rf_probs = probs['random_forest']

            if 'pointnet' in probs:
                # Kombinuj - uzyj RF class z najwyzszym combined score
                pn_preds = preds['pointnet']
                pn_conf = probs['pointnet']
                rf_preds = preds['random_forest']
                rf_conf = rf_probs.max(axis=1)

                # Srednia confidence
                avg_conf = (rf_conf + pn_conf) / 2

                # Wybierz predykcje modelu z wyzsza pewnoscia
                use_pn = pn_conf > rf_conf
                final_preds = np.where(use_pn, pn_preds, rf_preds)
                final_conf = np.maximum(pn_conf, rf_conf)

                return final_preds, final_conf
            else:
                return preds['random_forest'], rf_probs.max(axis=1)

        # Fallback - hard voting
        return self._hard_voting(preds, probs)

    def _weighted_voting(self, preds: Dict, probs: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted voting z wagami"""
        if self.config.weights is None:
            # Domyslne wagi
            weights = [1.0 / len(preds)] * len(preds)
        else:
            weights = self.config.weights

        models = list(preds.keys())

        # Wazone glosowanie
        weighted_scores = {}
        for i, model in enumerate(models):
            w = weights[i] if i < len(weights) else weights[-1]
            weighted_scores[model] = w

        # Similar to soft but with weights
        if 'random_forest' in probs and 'pointnet' in probs:
            rf_w = weighted_scores.get('random_forest', 0.5)
            pn_w = weighted_scores.get('pointnet', 0.5)

            rf_conf = probs['random_forest'].max(axis=1)
            pn_conf = probs['pointnet']

            # Wazona suma
            weighted_rf = rf_w * rf_conf
            weighted_pn = pn_w * pn_conf

            use_pn = weighted_pn > weighted_rf
            final_preds = np.where(use_pn, preds['pointnet'], preds['random_forest'])
            final_conf = (weighted_rf + weighted_pn) / (rf_w + pn_w)

            return final_preds, final_conf

        return self._hard_voting(preds, probs)

    def save(self, path: str):
        """Zapisz ensemble"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'config': self.config,
            'label_mapping': self.label_mapping,
            'reverse_mapping': self.reverse_mapping,
            'models': {}
        }

        # Zapisz kazdy model osobno
        for name, model in self.models.items():
            if name == 'random_forest':
                save_dict['models']['random_forest'] = {
                    'type': 'rf',
                    'model': model
                }
            elif name == 'pointnet':
                # PointNet zapisujemy osobno
                pn_path = path.parent / f"{path.stem}_pointnet.pth"
                model.save(str(pn_path))
                save_dict['models']['pointnet'] = {
                    'type': 'pointnet',
                    'path': str(pn_path)
                }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

        logger.info(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'EnsembleClassifier':
        """Wczytaj ensemble"""
        path = Path(path)

        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        ensemble = cls(save_dict['config'])
        ensemble.label_mapping = save_dict['label_mapping']
        ensemble.reverse_mapping = save_dict['reverse_mapping']

        for name, model_info in save_dict['models'].items():
            if model_info['type'] == 'rf':
                ensemble.models['random_forest'] = model_info['model']
            elif model_info['type'] == 'pointnet':
                if is_torch_available():
                    ensemble.models['pointnet'] = PointNetTrainer.load(model_info['path'])
                else:
                    logger.warning("PointNet model found but PyTorch not available")

        ensemble.is_trained = True
        logger.info(f"Ensemble loaded from {path}")

        return ensemble

    def get_info(self) -> Dict:
        """Informacje o ensemble"""
        return {
            'voting': self.config.voting,
            'n_models': len(self.models),
            'models': list(self.models.keys()),
            'is_trained': self.is_trained
        }


def create_ensemble(
    use_rf: bool = True,
    use_pointnet: bool = True,
    voting: str = 'soft'
) -> EnsembleClassifier:
    """Convenience function"""
    config = EnsembleConfig(
        use_rf=use_rf,
        use_pointnet=use_pointnet,
        voting=voting
    )
    return EnsembleClassifier(config)
