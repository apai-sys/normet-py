# src/normet/model/train.py
"""Model training helpers: :func:`build_model` and :func:`train_model`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..backends import backend_registry
from ..utils._config import DEFAULT_SEED
from ..utils.logging import get_logger
from ..utils.prepare import prepare_data

log = get_logger(__name__)

__all__ = ["build_model", "train_model"]


def _train_backend(_cache_key: str, *, be: Any, df: pd.DataFrame, **train_kw: Any) -> object:
    """Backend training call, written so joblib can memoize it on ``_cache_key``.

    All arguments other than ``_cache_key`` are ignored for cache-key purposes
    (see :func:`train_model`); ``_cache_key`` already encodes the data + config.
    """
    return be.train(df, **train_kw)


def build_model(
    df: pd.DataFrame,
    value: str,
    *,
    backend: str = "flaml",
    feature_names: list[str] | None = None,
    split_method: str = "random",
    fraction: float = 0.75,
    model_config: dict[str, Any] | None = None,
    seed: int = DEFAULT_SEED,
    verbose: bool = False,
    drop_time_features: bool = False,
    n_cores: int | None = None,
    cache: str | Path | None = None,
) -> tuple[pd.DataFrame, object]:
    """
    Prepare the data and train a model with the selected AutoML backend.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw input data.
    value : str
        Target column in `df`.
    backend : str, default="flaml"
        AutoML backend name. Must be registered in :data:`backend_registry`.
    feature_names : List[str], optional
        Predictors to use. Must be non-empty when training a model.
    split_method : str, default="random"
        Data split strategy for training.
    fraction : float, default=0.75
        Train fraction for the split.
    model_config : dict, optional
        Backend-specific configuration passed through to the trainer.
    seed : int, default=7654321
        Random seed.
    verbose : bool, default=True
        Verbose logging.
    drop_time_features : bool, default=False
        If True, drop helper time features like {"date_unix","day_julian","weekday","hour"}.
        By default we keep them.

    Returns
    -------
    (pandas.DataFrame, object)
        Tuple of (prepared_df, trained_model).

    Examples
    --------
    >>> import pandas as pd
    >>> from normet import build_model
    >>> df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=48, freq="h"),
    ...                    "PM2.5": range(48), "t2m": 10.0, "blh": 500.0})
    >>> df_prep, model = build_model(df, value="PM2.5",
    ...                              feature_names=["t2m", "blh"])  # doctest: +SKIP
    """
    backend = (backend or "flaml").lower()
    backend_registry.get(backend)  # validate early

    if not feature_names:
        raise ValueError("`feature_names` must be provided and non-empty.")

    # Optionally drop helper time features (default: keep them)
    if drop_time_features:
        drop_cols = {"date_unix", "day_julian", "weekday", "hour"}
        variables = [c for c in feature_names if c not in drop_cols]
    else:
        variables = list(feature_names)

    # Prepare data (ensures 'date', renames target to 'value', splits sets, etc.)
    df_prep = prepare_data(
        df=df,
        value=value,
        feature_names=variables,
        split_method=split_method,
        fraction=fraction,
        seed=seed,
    )

    # Align variables to what survived prepare_data (be explicit if none remain)
    variables = [c for c in variables if c in df_prep.columns]
    if not variables:
        raise ValueError(
            "None of the requested features remain after prepare_data(). "
            "Check `feature_names` and your input columns."
        )

    # Resolve target column consistently
    if "value" in df_prep.columns:
        target_col = "value"
    elif value in df_prep.columns:
        target_col = value
        df_prep = df_prep.copy()
        df_prep["value"] = df_prep[value]
    else:
        raise ValueError(
            "Target column not found after prepare_data(); "
            "tried 'value' and '" + value + "'. Columns: " + str(list(df_prep.columns))
        )

    # Train
    model = train_model(
        df=df_prep,
        value=target_col,
        backend=backend,
        feature_names=variables,
        model_config=model_config,
        seed=seed,
        verbose=verbose,
        n_cores=n_cores,
        cache=cache,
    )

    log.info("Model trained with backend=%s", backend)
    return df_prep, model


def train_model(
    df: pd.DataFrame,
    *,
    value: str = "value",
    backend: str = "flaml",
    feature_names: list[str] | None = None,
    model_config: dict[str, Any] | None = None,
    seed: int = DEFAULT_SEED,
    verbose: bool = False,
    n_cores: int | None = None,
    cache: str | Path | None = None,
) -> object:
    """
    Train an AutoML model via the registered backend.

    Parameters
    ----------
    df : pandas.DataFrame
        Prepared dataset (must contain the target and predictor columns;
        may include a 'set' column for train/test partition).
    value : str, default="value"
        Name of the target column.
    backend : str, default="flaml"
        AutoML backend name. Must be registered in :data:`backend_registry`.
    feature_names : list[str], optional
        Predictor names. Must be non-empty and unique.
    model_config : dict, optional
        Backend-specific configuration options.
    seed : int, default=7654321
        Random seed for reproducibility.
    verbose : bool, default=True
        Verbose logging.
    cache : str or pathlib.Path, optional
        If given, memoize the trained model to this directory (a
        :class:`joblib.Memory` location). Repeat calls with the same data
        (hashed) and configuration are served from disk instead of retraining.
        Off by default.

    Returns
    -------
    object
        Trained model object. The returned model has an attribute
        ``backend`` set to the backend name.

    Raises
    ------
    ValueError
        If feature_names are missing, empty, duplicated, or not present in ``df``.
    RuntimeError
        If backend training fails.

    Notes
    -----
    - If a 'set' column exists, only rows with ``set == 'training'`` are used.
    - ``model_config`` keys are backend-specific; see the backend's docstring.

    Examples
    --------
    >>> import pandas as pd
    >>> from normet import train_model
    >>> df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=48, freq="h"),
    ...                    "value": range(48), "t2m": 10.0, "blh": 500.0})
    >>> model = train_model(df, feature_names=["t2m", "blh"])  # doctest: +SKIP
    """
    backend = (backend or "flaml").lower()
    be = backend_registry.get(backend)

    if not feature_names:
        raise ValueError("`feature_names` must be a non-empty list.")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("`feature_names` contains duplicates.")
    missing = set(feature_names + [value]) - set(df.columns)
    if missing:
        raise ValueError("Columns not found in df: " + str(sorted(missing)))

    train_kw: dict[str, Any] = dict(
        value=value,
        feature_names=feature_names,
        model_config=model_config,
        seed=seed,
        verbose=verbose,
        n_cores=n_cores,
    )

    if cache is None:
        return be.train(df, **train_kw)

    # Opt-in on-disk memoization keyed by data content + config (not the raw
    # DataFrame / backend object, which we tell joblib to ignore).
    from ..utils.cache import config_hash, dataframe_hash, make_memory

    key_cols = list(
        dict.fromkeys([*feature_names, value, *(["set"] if "set" in df.columns else [])])
    )
    cache_key = config_hash(
        backend,
        value,
        sorted(feature_names),
        model_config,
        seed,
        dataframe_hash(df[key_cols]),
    )
    memory = make_memory(cache)
    cached = memory.cache(_train_backend, ignore=["be", "df"])
    return cached(cache_key, be=be, df=df, **train_kw)
