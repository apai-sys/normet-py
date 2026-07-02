# src/normet/backends/lgb_backend.py
"""LightGBM backend: cross-validated hyperparameter search, persistence, and :class:`LgbModel`."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils._config import DEFAULT_SEED
from ..utils._lazy import require
from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "train_lgb",
    "save_lgb",
    "load_lgb",
    "LgbModel",
]

_DEFAULT_CONFIG: dict[str, Any] = {
    "n_trials": 50,
    "cv_folds": 5,
    "nrounds": 1000,
    "early_stopping_rounds": 20,
    "learning_rate_min": 0.01,
    "learning_rate_max": 0.3,
}


def _import_lightgbm():
    """Dynamically import lightgbm."""
    lgb = require("lightgbm", hint="pip install lightgbm")
    return lgb


def _infer_num_leaves_range(n: int, cfg: dict[str, Any]) -> tuple[int, int]:
    """Compute [min, max] num_leaves from dataset size and user config."""
    leaves_min = cfg.get("num_leaves_min")
    if leaves_min is None:
        leaves_min = max(8, min(127, n // 20))
    leaves_max = cfg.get("num_leaves_max")
    if leaves_max is None:
        leaves_max = min(127, max(16, n // 3))
    leaves_max = max(leaves_min + 1, leaves_max)
    return leaves_min, leaves_max


def _infer_min_data_in_leaf_range(n: int) -> tuple[int, int]:
    """Compute [min, max] min_data_in_leaf from dataset size."""
    low = max(3, min(100, n // 50))
    high = min(500, max(20, n // 5))
    return low, high


class LgbModel:
    """Wrapper around a lightgbm Booster that carries normet metadata.

    Attributes
    ----------
    backend : str
        Always ``"lightgbm"``.
    feature_names : list of str
        Predictor column names the model was trained on.
    booster : lightgbm.Booster
        The underlying trained booster.
    """

    __slots__ = ("_booster", "backend", "feature_names")

    def __init__(self, booster: Any, feature_names: list[str]) -> None:
        self._booster = booster
        self.backend = "lightgbm"
        self.feature_names = list(feature_names)

    @property
    def booster(self) -> object:
        """The underlying ``lightgbm.Booster``."""
        return self._booster

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict on ``X``, treating non-finite inputs as 0."""
        if isinstance(X, pd.DataFrame):
            # Booster.predict is positional: realign columns to training order
            # regardless of the order the caller passed them in.
            X = X.reindex(columns=self.feature_names)
        X_arr = np.asarray(X, dtype=float)
        non_finite = ~np.isfinite(X_arr)
        if np.any(non_finite):
            X_arr = X_arr.copy()
            X_arr[non_finite] = 0.0
        return np.asarray(self._booster.predict(X_arr), dtype=float).reshape(-1)

    def feature_name(self) -> list[str]:
        """Return the predictor column names the model was trained on."""
        return self.feature_names

    def feature_importance(self, importance_type: str = "gain") -> list[float]:
        """Return per-feature importances from the underlying booster."""
        return list(self._booster.feature_importance(importance_type=importance_type))


def save_lgb(
    model: LgbModel,
    path: str | Path = ".",
    filename: str = "automl.joblib",
) -> str:
    """Persist a LightGBM model to disk via joblib.

    Parameters
    ----------
    model : LgbModel
        Wrapped LightGBM model to save.
    path : str | Path, default="."
        Directory to save into.
    filename : str, default="automl.joblib"
        Desired filename.  If no extension is given, ``.joblib`` is appended.

    Returns
    -------
    str
        Absolute path of the saved file.
    """
    joblib = require("joblib", hint="pip install joblib")
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    if not Path(filename).suffix:
        filename = f"{filename}.joblib"
    model_path = folder / filename
    joblib.dump(model, str(model_path))
    log.info("Saved LightGBM model to %s", model_path)
    return str(model_path)


def load_lgb(
    path: str | Path = ".",
    filename: str | None = None,
) -> LgbModel:
    """Load a LightGBM model saved with :func:`save_lgb`.

    See :func:`.flaml_backend.load_flaml` for the resolution logic when
    *filename* is ``None``.

    Parameters
    ----------
    path : str | Path, default="."
        Directory or file path.
    filename : str | None, optional
        Specific filename.  If ``None``, auto-pick newest ``.joblib``/``.pkl``.

    Returns
    -------
    LgbModel
        The deserialised wrapped LightGBM model.

    Raises
    ------
    FileNotFoundError
        If no suitable model file is found.
    ImportError
        If *joblib* is not installed.
    """
    joblib = require("joblib", hint="pip install joblib")
    p = Path(path)

    if filename is None and p.is_file():
        target = p
        if target.suffix.lower() not in {".joblib", ".pkl"}:
            raise FileNotFoundError(
                f"Unsupported model file (expect .joblib/.pkl): '{target.name}'"
            )
    else:
        folder = p if p.is_dir() else p.parent
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: '{folder}'")
        if filename:
            target = folder / filename
            if not target.exists():
                raise FileNotFoundError(f"Specified model file not found: {target}")
        else:
            candidates = [
                f
                for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in {".joblib", ".pkl"}
            ]
            if not candidates:
                raise FileNotFoundError(
                    f"No LightGBM model files (.joblib/.pkl) found under '{folder}'."
                )
            target = max(candidates, key=lambda f: f.stat().st_mtime)

    model = joblib.load(str(target))
    if not isinstance(model, LgbModel):
        raise TypeError(f"Loaded object is not an LgbModel (got {type(model).__name__}).")
    log.info("Loaded LightGBM model from %s", target)
    return model


def train_lgb(
    df: pd.DataFrame,
    value: str = "value",
    feature_names: list[str] | None = None,
    variables: list[str] | None = None,
    model_config: dict[str, Any] | None = None,
    seed: int = DEFAULT_SEED,
    verbose: bool = False,
    n_cores: int | None = None,
) -> LgbModel:
    """Train a LightGBM model with random hyperparameter search + CV.

    Runs a random search over hyperparameters using ``lgb.cv`` for each
    trial, picks the configuration with the lowest validation RMSE, then
    trains a final model on all training data with those best parameters.

    Parameters
    ----------
    df : pandas.DataFrame
        Training dataset.  If a ``'set'`` column is present, only rows
        with ``set == 'training'`` are used.
    value : str, default="value"
        Name of the target column.
    feature_names : list of str, optional
        Predictor column names.  Must be non-empty and unique.
    variables : list of str, optional
        .. deprecated::
            Use *feature_names* instead.
    model_config : dict, optional
        LightGBM configuration overrides.  Supported keys and **defaults**:

        - ``n_trials`` : int (default 50)
        - ``cv_folds`` : int (default 5)
        - ``nrounds`` : int (default 1000)
        - ``early_stopping_rounds`` : int (default 20)
        - ``num_leaves_min`` / ``num_leaves_max`` : int | None (auto-inferred)
        - ``learning_rate_min`` / ``learning_rate_max`` : float (0.01 – 0.3)

    seed : int, default=7654321
        Random seed for reproducibility.
    verbose : bool, default=False
        Whether to log progress messages.

    Returns
    -------
    LgbModel
        Wrapped LightGBM Booster with ``.backend == "lightgbm"`` and
        ``.feature_names`` set.

    Raises
    ------
    ValueError
        If *feature_names* are missing, empty, duplicated, or columns not found.
    """
    if variables is not None:
        warnings.warn(
            "`variables` is deprecated, use `feature_names` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if feature_names is None:
            feature_names = variables
    if not feature_names:
        raise ValueError("`feature_names` must be a non-empty list.")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("`feature_names` contains duplicates.")
    missing = set(feature_names + [value]) - set(df.columns)
    if missing:
        raise ValueError(f"Columns not found in df: {sorted(missing)}")

    if "set" in df.columns:
        df_train = df.loc[df["set"] == "training", [value] + feature_names]
        if df_train.empty:
            df_train = df[[value] + feature_names]
    else:
        df_train = df[[value] + feature_names]

    if df_train[feature_names].shape[1] == 0:
        raise ValueError("No predictor columns available after preprocessing.")

    n = len(df_train)
    n_feat = len(feature_names)

    cfg = dict(_DEFAULT_CONFIG)
    if model_config:
        cfg.update(model_config)

    leaves_min, leaves_max = _infer_num_leaves_range(n, cfg)
    min_data_low, min_data_high = _infer_min_data_in_leaf_range(n)

    rng = np.random.default_rng(seed)
    lgb = _import_lightgbm()

    x_mat = np.asarray(df_train[feature_names], dtype=float)
    y_vec = np.asarray(df_train[value], dtype=float)

    if np.any(np.isnan(y_vec)):
        raise ValueError("Target column contains NA values.")
    non_finite = ~np.isfinite(x_mat)
    if np.any(non_finite):
        x_mat = x_mat.copy()
        x_mat[non_finite] = 0.0

    dtrain = lgb.Dataset(x_mat, label=y_vec)

    best_score = float("inf")
    best_params: dict[str, Any] | None = None
    best_nrounds = cfg["nrounds"]

    (log.info if verbose else log.debug)(
        "LightGBM random search: %d trials, %d-fold CV, %d predictors, %d rows",
        cfg["n_trials"],
        cfg["cv_folds"],
        n_feat,
        n,
    )

    for i in range(1, cfg["n_trials"] + 1):
        params: dict[str, Any] = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "feature_pre_filter": False,
            "num_threads": n_cores if n_cores is not None else 0,
            "device_type": "cpu",
            "num_leaves": int(rng.integers(leaves_min, leaves_max + 1)),
            "learning_rate": float(rng.uniform(cfg["learning_rate_min"], cfg["learning_rate_max"])),
            "min_data_in_leaf": int(rng.integers(min_data_low, min_data_high + 1)),
            "feature_fraction": float(rng.uniform(0.5, 1.0)),
            "bagging_fraction": float(rng.uniform(0.5, 1.0)),
            "bagging_freq": int(rng.choice([0, 1, 5])),
            "lambda_l1": 0.0 if rng.uniform() < 0.5 else 10.0 ** float(rng.uniform(-3, 1)),
            "lambda_l2": 0.0 if rng.uniform() < 0.5 else 10.0 ** float(rng.uniform(-3, 1)),
        }

        params["early_stopping_rounds"] = cfg["early_stopping_rounds"]
        cv_results = lgb.cv(
            params=params,
            train_set=dtrain,
            num_boost_round=cfg["nrounds"],
            nfold=cfg["cv_folds"],
            stratified=False,
            seed=seed,
        )

        # Locate the metric column regardless of prefix ("rmse-mean" / "valid rmse-mean")
        mean_key = next((k for k in cv_results if k.endswith("rmse-mean")), None)
        if mean_key is None:
            raise KeyError(f"Expected 'rmse-mean' in CV results, got keys: {list(cv_results)}")
        rmse_vals = cv_results[mean_key]
        best_idx = int(np.argmin(rmse_vals))
        score = float(rmse_vals[best_idx])
        best_iter = best_idx + 1

        if score < best_score:
            best_score = score
            best_params = params
            best_nrounds = best_iter
            (log.info if verbose else log.debug)(
                "  Trial %d/%d: best RMSE = %.4f (leaves=%d, lr=%.3f, rounds=%d)",
                i,
                cfg["n_trials"],
                score,
                params["num_leaves"],
                params["learning_rate"],
                best_nrounds,
            )
        elif verbose and i % 10 == 0:
            log.info(
                "  Trial %d/%d: RMSE = %.4f (best = %.4f)",
                i,
                cfg["n_trials"],
                score,
                best_score,
            )

    if best_params is None:
        raise RuntimeError("LightGBM random search produced no valid trial.")

    (log.info if verbose else log.debug)(
        "Training final model (rounds=%d, leaves=%d)",
        best_nrounds,
        best_params["num_leaves"],
    )

    final_params = {k: v for k, v in best_params.items() if k != "early_stopping_rounds"}
    booster = lgb.train(
        params=final_params,
        train_set=dtrain,
        num_boost_round=best_nrounds,
    )

    return LgbModel(booster, feature_names)


class _LgbBackend:
    """Backend-compatible wrapper around the LightGBM-specific functions."""

    name = "lightgbm"

    def train(
        self,
        df: pd.DataFrame,
        value: str = "value",
        feature_names: list[str] | None = None,
        variables: list[str] | None = None,
        model_config: dict[str, Any] | None = None,
        seed: int = DEFAULT_SEED,
        verbose: bool = False,
        n_cores: int | None = None,
    ) -> LgbModel:
        return train_lgb(
            df,
            value=value,
            feature_names=feature_names,
            variables=variables,
            model_config=model_config,
            seed=seed,
            verbose=verbose,
            n_cores=n_cores,
        )

    def save(
        self,
        model: Any,
        path: str = ".",
        filename: str = "automl.joblib",
    ) -> str:
        return save_lgb(model, path=path, filename=filename)

    def load(
        self,
        path: str = ".",
        filename: str | None = None,
    ) -> LgbModel:
        return load_lgb(path=path, filename=filename)


backend = _LgbBackend()
