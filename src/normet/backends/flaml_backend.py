# src/normet/backends/flaml_backend.py
"""FLAML AutoML backend: training, persistence, and the ``flaml`` registry entry."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils._config import DEFAULT_SEED
from ..utils._lazy import require
from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "train_flaml",
    "save_flaml",
    "load_flaml",
]


def _import_flaml_automl():
    """Dynamically import FLAML's AutoML class."""
    AutoML = require("flaml.automl:AutoML", hint="pip install flaml")
    return AutoML


def save_flaml(
    model,
    path: str | Path = ".",
    filename: str = "automl.joblib",
) -> str:
    """
    Save a FLAML AutoML model to the specified directory with the given filename.

    Parameters
    ----------
    model : AutoML
        The FLAML AutoML model to save.
    path : str | Path, default="."
        Directory path to save the model.
    filename : str, default="automl.joblib"
        Desired filename. If no extension is given, ``.joblib`` will be added.

    Returns
    -------
    str
        The path of the saved model.
    """
    joblib = require("joblib", hint="pip install joblib")  # <-- added
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)

    # Ensure extension
    if not Path(filename).suffix:
        filename = f"{filename}.joblib"

    model_path = folder / filename

    joblib.dump(model, str(model_path))
    log.info("Saved FLAML model to %s", model_path)
    return str(model_path)


def load_flaml(
    path: str | Path = ".",
    filename: str | None = None,
) -> object:
    """
    Load a FLAML model saved with ``save_flaml``.

    Resolution rules
    ----------------
    - If ``filename`` is provided, load exactly ``path/filename``.
    - Otherwise, scan ``path`` and pick the most recently modified
      ``.joblib`` or ``.pkl`` file.

    Parameters
    ----------
    path : str | Path, default "."
        Directory containing the saved model, or a file path if you pass the
        file directly via ``filename=None`` and ``path`` is a file.
    filename : str | None, optional
        Specific filename to load (e.g., "automl.joblib"). If None, auto-pick.

    Returns
    -------
    object
        The loaded FLAML AutoML object. Ensures ``backend == "flaml"``.

    Raises
    ------
    FileNotFoundError
        If no suitable model file is found.
    ImportError
        If ``joblib`` is not installed.
    """
    joblib = require("joblib", hint="pip install joblib")
    p = Path(path)

    # If user passed a direct file path via path and no filename
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
                    f"No FLAML model files (.joblib/.pkl) found under '{folder}'."
                )
            target = max(candidates, key=lambda f: f.stat().st_mtime)

    model = joblib.load(str(target))
    if getattr(model, "backend", None) != "flaml":
        try:
            model.backend = "flaml"
        except Exception as e:
            log.debug("Could not set .backend on loaded model: %s", e)
    log.info("Loaded FLAML model from %s", target)
    return model


def train_flaml(
    df: pd.DataFrame,
    value: str = "value",
    feature_names: list[str] | None = None,
    variables: list[str] | None = None,
    model_config: dict[str, Any] | None = None,
    seed: int = DEFAULT_SEED,
    verbose: bool = False,
    n_cores: int | None = None,
    use_gpu: bool = False,
) -> object:
    """
    Train a model with FLAML AutoML and tag it with ``backend='flaml'``.

    Parameters
    ----------
    df : pandas.DataFrame
        Training dataset containing predictors and target column.
        If a ``'set'`` column is present, rows with ``set == 'training'`` are used.
    value : str, default="value"
        Name of the target column.
    feature_names : list of str, optional
        Predictor column names. Must be non-empty and unique.
    variables : list of str, optional
        .. deprecated::
            Use *feature_names* instead.
    model_config : dict, optional
        FLAML configuration overrides. Recognized keys and **defaults**:
          - ``time_budget`` : int (default 90)
          - ``metric`` : str (default "r2")
          - ``estimator_list`` : list[str] (default ["lgbm"])
          - ``task`` : {"regression","classification"} (default "regression")
          - ``eval_method`` : {"auto","cv","holdout"} (default "auto")
          - ``save_model`` : bool (default False)
          - ``path`` : str (default ".")
          - ``filename`` : str (default "automl.joblib")
          - ``verbose`` : bool (defaults to this function's ``verbose``)

    seed : int, default=7654321
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to log progress.

    Returns
    -------
    object
        A trained FLAML AutoML model, tagged with ``backend="flaml"``.

    Raises
    ------
    ValueError
        If feature_names are missing/empty, duplicated, or columns not found.

    .. versionchanged:: 0.3.0
        ``variables`` is deprecated in favour of ``feature_names``.
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

    # Pick training rows if a split is present
    if "set" in df.columns:
        df_train = df.loc[df["set"] == "training", [value] + feature_names]
        if df_train.empty:
            df_train = df[[value] + feature_names]
    else:
        df_train = df[[value] + feature_names]

    if df_train[feature_names].shape[1] == 0:
        raise ValueError("No predictor columns available after preprocessing.")

    # Defaults
    default_cfg: dict[str, Any] = {
        "time_budget": 90,
        "metric": "r2",
        "estimator_list": ["lgbm"],
        "task": "regression",
        "eval_method": "auto",
        "save_model": False,
        "path": ".",
        "filename": "automl.joblib",
        "verbose": verbose,
    }
    # ``split_type`` / ``n_splits`` let callers request time-ordered internal
    # validation (split_type="time"), which prevents the model-selection step
    # from being fooled by temporally-leaked random folds on autocorrelated
    # air-quality series. Only forwarded to FLAML when explicitly set.
    if n_cores is not None:
        default_cfg.setdefault("n_jobs", n_cores)
    if model_config:
        default_cfg.update(model_config)

    # Build kwargs for AutoML.fit
    passthrough = {
        "time_budget",
        "metric",
        "estimator_list",
        "task",
        "eval_method",
        "verbose",
        "n_jobs",
        "split_type",
        "n_splits",
    }
    automl_kwargs = {k: default_cfg[k] for k in passthrough if k in default_cfg}

    if use_gpu:
        import platform as _platform
        import sys as _sys

        _is_apple_silicon = _sys.platform == "darwin" and _platform.machine() in {
            "arm64",
            "aarch64",
        }
        if _is_apple_silicon:
            log.warning(
                "use_gpu=True has no effect on Apple Silicon: neither FLAML nor its "
                "LightGBM estimator support Metal/MPS. Training will run on CPU."
            )
        else:
            log.info(
                "FLAML GPU training requested. Injecting device_type='cuda' for lgbm estimator. "
                "Requires LightGBM built with CUDA support and flaml[tune] installed."
            )
        if not _is_apple_silicon:
            try:
                from flaml import tune as _flaml_tune

                automl_kwargs.setdefault("custom_hp", {}).setdefault("lgbm", {})["device_type"] = {
                    "domain": _flaml_tune.choice(["cuda"])
                }
            except Exception as _e:
                log.warning("Could not inject GPU config into FLAML (%s). Training on CPU.", _e)

    AutoML = _import_flaml_automl()
    automl = AutoML()

    (log.info if verbose else log.debug)(
        "Training FLAML AutoML: X shape=%s, target='%s'", df_train[feature_names].shape, value
    )

    automl.fit(
        X_train=df_train[feature_names],
        y_train=df_train[value],
        seed=seed,
        **automl_kwargs,
    )

    (log.info if verbose else log.debug)(
        "FLAML best_estimator=%s | best_config=%s", automl.best_estimator, automl.best_config
    )

    # Optional persistence
    if default_cfg.get("save_model", False):
        save_flaml(
            automl,
            path=default_cfg.get("path", "."),
            filename=default_cfg.get("filename", "automl.joblib"),
        )

    automl.backend = "flaml"
    automl.use_gpu = use_gpu
    return automl


class _FlamlBackend:
    """Backend-compatible wrapper around the FLAML-specific functions."""

    name = "flaml"

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
        use_gpu: bool = False,
    ) -> object:
        return train_flaml(
            df,
            value=value,
            feature_names=feature_names,
            variables=variables,
            model_config=model_config,
            seed=seed,
            verbose=verbose,
            n_cores=n_cores,
            use_gpu=use_gpu,
        )

    def save(
        self,
        model: object,
        path: str = ".",
        filename: str = "automl.joblib",
    ) -> str:
        return save_flaml(model, path=path, filename=filename)

    def load(
        self,
        path: str = ".",
        filename: str | None = None,
    ) -> object:
        return load_flaml(path=path, filename=filename)


backend = _FlamlBackend()
