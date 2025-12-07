"""
Auto-Tuning - Automatyczna optymalizacja hiperparametrow

Wspiera:
- Grid Search
- Random Search
- Bayesian Optimization (jesli optuna dostepne)
- Cross-validation

HackNation 2025 - CPK Chmura+
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Sprawdz czy optuna jest dostepna
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class HyperparameterSpace:
    """Definicja przestrzeni hiperparametrow"""
    name: str
    param_type: str  # 'int', 'float', 'categorical'
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List] = None
    log_scale: bool = False

    def sample_random(self) -> Any:
        """Losowa wartosc z przestrzeni"""
        if self.param_type == 'int':
            if self.log_scale:
                return int(np.exp(np.random.uniform(np.log(self.low), np.log(self.high))))
            return np.random.randint(self.low, self.high + 1)
        elif self.param_type == 'float':
            if self.log_scale:
                return np.exp(np.random.uniform(np.log(self.low), np.log(self.high)))
            return np.random.uniform(self.low, self.high)
        else:  # categorical
            return np.random.choice(self.choices)

    def get_grid(self, n_points: int = 5) -> List:
        """Punkty dla grid search"""
        if self.param_type == 'categorical':
            return list(self.choices)
        elif self.param_type == 'int':
            if self.log_scale:
                return [int(x) for x in np.logspace(
                    np.log10(self.low), np.log10(self.high), n_points
                )]
            return [int(x) for x in np.linspace(self.low, self.high, n_points)]
        else:  # float
            if self.log_scale:
                return list(np.logspace(np.log10(self.low), np.log10(self.high), n_points))
            return list(np.linspace(self.low, self.high, n_points))


@dataclass
class TuningResult:
    """Wynik tuningu"""
    best_params: Dict
    best_score: float
    all_results: List[Dict]
    search_time: float
    n_trials: int
    best_model: Any = None


# Domyslne przestrzenie dla roznych modeli
RF_PARAM_SPACE = [
    HyperparameterSpace('n_estimators', 'int', 50, 300),
    HyperparameterSpace('max_depth', 'int', 5, 30),
    HyperparameterSpace('min_samples_split', 'int', 2, 20),
    HyperparameterSpace('min_samples_leaf', 'int', 1, 10),
    HyperparameterSpace('max_features', 'categorical', choices=['sqrt', 'log2', None]),
]

POINTNET_PARAM_SPACE = [
    HyperparameterSpace('learning_rate', 'float', 0.0001, 0.01, log_scale=True),
    HyperparameterSpace('hidden_dim', 'int', 128, 512),
    HyperparameterSpace('dropout', 'float', 0.1, 0.5),
    HyperparameterSpace('batch_size', 'categorical', choices=[4, 8, 16, 32]),
]


class CrossValidator:
    """K-Fold Cross Validation dla chmur punktow"""

    def __init__(self, n_folds: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generuj indeksy train/val dla kazdego folda"""
        indices = np.arange(n_samples)

        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(indices)

        fold_size = n_samples // self.n_folds
        folds = []

        for i in range(self.n_folds):
            start = i * fold_size
            end = start + fold_size if i < self.n_folds - 1 else n_samples

            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])

            folds.append((train_indices, val_indices))

        return folds

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_fn: Callable,
        eval_fn: Callable,
        progress_callback=None
    ) -> Dict:
        """
        Przeprowadz cross-validation

        Args:
            X: features
            y: labels
            train_fn: funkcja(X_train, y_train) -> model
            eval_fn: funkcja(model, X_val, y_val) -> score

        Returns:
            Dict z wynikami CV
        """
        folds = self.split(len(X))
        scores = []

        for i, (train_idx, val_idx) in enumerate(folds):
            if progress_callback:
                pct = int((i + 1) / self.n_folds * 100)
                progress_callback("CV", pct, f"Fold {i+1}/{self.n_folds}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = train_fn(X_train, y_train)
            score = eval_fn(model, X_val, y_val)
            scores.append(score)

        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'fold_scores': scores,
            'n_folds': self.n_folds
        }


class GridSearchTuner:
    """Grid Search - przeszukuje wszystkie kombinacje"""

    def __init__(
        self,
        param_space: List[HyperparameterSpace],
        n_points_per_param: int = 3
    ):
        self.param_space = param_space
        self.n_points = n_points_per_param

    def _generate_grid(self) -> List[Dict]:
        """Generuj wszystkie kombinacje parametrow"""
        from itertools import product

        grids = {p.name: p.get_grid(self.n_points) for p in self.param_space}
        keys = list(grids.keys())
        values = [grids[k] for k in keys]

        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def search(
        self,
        objective_fn: Callable[[Dict], float],
        progress_callback=None
    ) -> TuningResult:
        """
        Przeprowadz grid search

        Args:
            objective_fn: funkcja(params) -> score (wyzszy = lepszy)

        Returns:
            TuningResult
        """
        start_time = time.time()
        combinations = self._generate_grid()

        logger.info(f"Grid search: {len(combinations)} combinations")

        results = []
        best_score = -np.inf
        best_params = None

        for i, params in enumerate(combinations):
            if progress_callback:
                pct = int((i + 1) / len(combinations) * 100)
                progress_callback("Grid Search", pct, f"Trial {i+1}/{len(combinations)}")

            try:
                score = objective_fn(params)
                results.append({'params': params, 'score': score})

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

            except Exception as e:
                logger.warning(f"Trial {i+1} failed: {e}")
                results.append({'params': params, 'score': None, 'error': str(e)})

        elapsed = time.time() - start_time

        return TuningResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=results,
            search_time=elapsed,
            n_trials=len(combinations)
        )


class RandomSearchTuner:
    """Random Search - losowe probkowanie przestrzeni"""

    def __init__(
        self,
        param_space: List[HyperparameterSpace],
        n_trials: int = 50
    ):
        self.param_space = param_space
        self.n_trials = n_trials

    def _sample_params(self) -> Dict:
        """Losuj zestaw parametrow"""
        return {p.name: p.sample_random() for p in self.param_space}

    def search(
        self,
        objective_fn: Callable[[Dict], float],
        progress_callback=None
    ) -> TuningResult:
        """Przeprowadz random search"""
        start_time = time.time()

        results = []
        best_score = -np.inf
        best_params = None

        for i in range(self.n_trials):
            if progress_callback:
                pct = int((i + 1) / self.n_trials * 100)
                progress_callback("Random Search", pct, f"Trial {i+1}/{self.n_trials}")

            params = self._sample_params()

            try:
                score = objective_fn(params)
                results.append({'params': params, 'score': score})

                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    logger.info(f"New best: {score:.4f} with {params}")

            except Exception as e:
                logger.warning(f"Trial {i+1} failed: {e}")
                results.append({'params': params, 'score': None, 'error': str(e)})

        elapsed = time.time() - start_time

        return TuningResult(
            best_params=best_params or {},
            best_score=best_score,
            all_results=results,
            search_time=elapsed,
            n_trials=self.n_trials
        )


class BayesianTuner:
    """Bayesian Optimization z Optuna"""

    def __init__(
        self,
        param_space: List[HyperparameterSpace],
        n_trials: int = 100
    ):
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna not installed. Run: pip install optuna")

        self.param_space = param_space
        self.n_trials = n_trials

    def search(
        self,
        objective_fn: Callable[[Dict], float],
        progress_callback=None
    ) -> TuningResult:
        """Przeprowadz Bayesian optimization"""
        start_time = time.time()

        def optuna_objective(trial):
            params = {}
            for p in self.param_space:
                if p.param_type == 'int':
                    params[p.name] = trial.suggest_int(
                        p.name, p.low, p.high, log=p.log_scale
                    )
                elif p.param_type == 'float':
                    params[p.name] = trial.suggest_float(
                        p.name, p.low, p.high, log=p.log_scale
                    )
                else:  # categorical
                    params[p.name] = trial.suggest_categorical(p.name, p.choices)

            return objective_fn(params)

        # Tworz study
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)

        # Callback dla progress
        if progress_callback:
            def callback(study, trial):
                pct = int((trial.number + 1) / self.n_trials * 100)
                progress_callback(
                    "Bayesian Opt",
                    pct,
                    f"Trial {trial.number+1}/{self.n_trials}, best={study.best_value:.4f}"
                )

            study.optimize(optuna_objective, n_trials=self.n_trials, callbacks=[callback])
        else:
            study.optimize(optuna_objective, n_trials=self.n_trials)

        elapsed = time.time() - start_time

        # Zbierz wyniki
        all_results = [
            {'params': t.params, 'score': t.value}
            for t in study.trials
        ]

        return TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            all_results=all_results,
            search_time=elapsed,
            n_trials=len(study.trials)
        )


class AutoMLPipeline:
    """
    Automatyczny pipeline ML z tuningiem

    Laczy:
    - Automatyczna selekcja cech
    - Tuning hiperparametrow
    - Cross-validation
    - Model selection
    """

    def __init__(
        self,
        search_method: str = 'random',  # 'grid', 'random', 'bayesian'
        n_trials: int = 30,
        cv_folds: int = 3
    ):
        self.search_method = search_method
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_model = None
        self.best_params = None
        self.tuning_history = []

    def auto_tune_rf(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        progress_callback=None
    ) -> TuningResult:
        """
        Automatyczny tuning Random Forest

        Args:
            X: features (N, F)
            y: labels (N,)
            feature_names: nazwy cech
            progress_callback: callback

        Returns:
            TuningResult z najlepszym modelem
        """
        from .classifiers import RandomForestPointClassifier

        cv = CrossValidator(n_folds=self.cv_folds)

        def objective(params: Dict) -> float:
            """Objective function dla tunera"""

            def train_fn(X_train, y_train):
                model = RandomForestPointClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', 20),
                    min_samples_split=params.get('min_samples_split', 2),
                    min_samples_leaf=params.get('min_samples_leaf', 1),
                    max_features=params.get('max_features', 'sqrt')
                )
                model.fit(X_train, y_train, feature_names)
                return model

            def eval_fn(model, X_val, y_val):
                preds = model.predict(X_val)
                return np.mean(preds == y_val)

            cv_result = cv.cross_validate(X, y, train_fn, eval_fn)
            return cv_result['mean_score']

        # Wybierz tuner
        if self.search_method == 'grid':
            tuner = GridSearchTuner(RF_PARAM_SPACE, n_points_per_param=3)
        elif self.search_method == 'bayesian' and OPTUNA_AVAILABLE:
            tuner = BayesianTuner(RF_PARAM_SPACE, n_trials=self.n_trials)
        else:
            tuner = RandomSearchTuner(RF_PARAM_SPACE, n_trials=self.n_trials)

        # Szukaj
        result = tuner.search(objective, progress_callback)

        # Trenuj finalny model z najlepszymi parametrami
        if result.best_params:
            final_model = RandomForestPointClassifier(
                n_estimators=result.best_params.get('n_estimators', 100),
                max_depth=result.best_params.get('max_depth', 20),
                min_samples_split=result.best_params.get('min_samples_split', 2),
                min_samples_leaf=result.best_params.get('min_samples_leaf', 1),
                max_features=result.best_params.get('max_features', 'sqrt')
            )
            final_model.fit(X, y, feature_names)
            result.best_model = final_model

        self.best_model = result.best_model
        self.best_params = result.best_params
        self.tuning_history.append(result)

        return result

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Waznosc cech z najlepszego modelu"""
        if self.best_model is None:
            return None

        if hasattr(self.best_model, 'get_feature_importance'):
            return self.best_model.get_feature_importance()

        return None


def is_optuna_available() -> bool:
    """Sprawdz czy optuna jest dostepna"""
    return OPTUNA_AVAILABLE


def quick_tune_rf(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 20,
    progress_callback=None
) -> Tuple[Dict, float]:
    """
    Szybki tuning Random Forest

    Returns:
        (best_params, best_score)
    """
    pipeline = AutoMLPipeline(search_method='random', n_trials=n_trials)
    result = pipeline.auto_tune_rf(X, y, progress_callback=progress_callback)
    return result.best_params, result.best_score
