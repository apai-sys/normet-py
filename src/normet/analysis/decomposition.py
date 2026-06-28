# src/normet/analysis/decomposition.py
"""Split a normalised series into emission- and meteorology-driven components.

Provides :func:`decompose` (and the convenience wrappers :func:`decom_emi`,
:func:`decom_met`) plus :class:`DecomposeConfig`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import ConfigError, DataError, ModelError
from ..model.train import build_model
from ..utils._config import DEFAULT_SEED, resolve_config
from ..utils.features import extract_features
from ..utils.logging import get_logger
from ..utils.prepare import add_date_variables, process_date
from .normalise import normalise

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DecomposeConfig:
    """Consolidated configuration for :func:`decompose` / :func:`decom_emi` / :func:`decom_met`.

    Every field has a default so the dataclass can be constructed with only
    the overrides that differ from the standard values.
    """

    value: str = "value"
    feature_names: list[str] | None = None
    backend: str | None = None
    split_method: str = "random"
    fraction: float = 0.75
    model_config: dict[str, Any] | None = None
    n_samples: int = 300
    seed: int = DEFAULT_SEED
    n_cores: int | None = None
    memory_save: bool = False
    use_gpu: bool = False
    verbose: bool = False
    importance_ascending: bool = False
    method: str = "emission"


def _resolve_config(config: DecomposeConfig | None = None, **kwargs) -> DecomposeConfig:
    return resolve_config(DecomposeConfig, config, **kwargs)


def _effective_cores(n_cores: int | None) -> int:
    """Resolve parallel worker count (>=1)."""
    return max(1, n_cores if n_cores is not None else (os.cpu_count() or 2) - 1)


def decompose(
    df: pd.DataFrame,
    model: object | None = None,
    *,
    config: DecomposeConfig | None = None,
    method: str = "emission",
    **kwargs,
) -> pd.DataFrame:
    """
    High-level wrapper for time series decomposition.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with datetime and target column.
    model : object, optional
        Pre-trained model. If None, a new model will be trained.
    config : DecomposeConfig, optional
        Consolidated config object. Individual keyword arguments (``value``,
        ``backend``, ``feature_names``, …) override the corresponding field
        when provided.
    method : {"emission", "meteorology"}
        Decomposition strategy.

    Returns
    -------
    pandas.DataFrame
        Decomposed result.

    Examples
    --------
    >>> import pandas as pd
    >>> from normet import decompose
    >>> df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=24, freq="h"),
    ...                    "value": range(24), "t2m": 10.0, "blh": 500.0})
    >>> result = decompose(df, method="emission", feature_names=["t2m", "blh"],
    ...                    n_samples=2)  # doctest: +SKIP
    """
    _cfg = _resolve_config(config=config, method=method, **kwargs)

    if df is None:
        raise DataError("`df` must be provided.")
    if _cfg.value is None:
        raise DataError("`value` must be provided.")
    if model is None and _cfg.feature_names is None:
        raise ConfigError("Either `model` or `feature_names` must be provided.")
    if model is None and _cfg.backend is None:
        raise ConfigError("When training a model, `backend` must be specified.")

    if _cfg.method == "emission":
        return decom_emi(df=df, model=model, config=_cfg)

    if _cfg.method == "meteorology":
        return decom_met(df=df, model=model, config=_cfg)

    raise ConfigError(
        f"Unsupported decomposition method: '{_cfg.method}'. "
        "Must be one of 'emission' or 'meteorology'."
    )


def decom_emi(
    df: pd.DataFrame,
    model: object | None = None,
    *,
    config: DecomposeConfig | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Emission-based decomposition via leave-one-out normalisation.

    Sequentially fixes time variables (``base``, ``date_unix``, ``day_julian``,
    ``weekday``, ``hour``) to isolate the contribution of each temporal
    component to the predicted concentration.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Input data with datetime index and target column.
    model : object, optional
        Pre-trained model.
    config : DecomposeConfig, optional
        Consolidated config object.
    **kwargs
        Supported shorthand for overriding individual :class:`DecomposeConfig`
        fields without constructing a config object. Any field passed both via
        ``config`` and as a keyword is resolved in favour of the keyword.

    Returns
    -------
    pandas.DataFrame
        Decomposition results.
    """
    _cfg = _resolve_config(config=config, **kwargs)

    if df is None:
        raise DataError("`df` must be provided.")
    if _cfg.value is None:
        raise DataError("`value` (target column name) must be provided.")
    if model is None and _cfg.feature_names is None:
        raise ConfigError("Either `model` or `feature_names` must be provided.")
    if model is None and _cfg.backend is None:
        raise ConfigError("When training a model, `backend` must be specified.")

    df_work = process_date(df.copy()) if "date" not in df.columns else df.copy()
    if "date" not in df_work.columns:
        raise DataError("Could not find or create a 'date' column.")

    if _cfg.value not in df_work.columns:
        raise DataError(f"`df` does not contain the target column '{_cfg.value}'.")

    observed_series = df_work[_cfg.value].copy()
    if _cfg.value != "value":
        df_work = df_work.rename(columns={_cfg.value: "value"})

    mask_valid = df_work["date"].notna() & df_work["value"].notna()
    df_work = df_work.loc[mask_valid].sort_values("date").reset_index(drop=True)
    observed_series = observed_series.loc[mask_valid].reset_index(drop=True)

    if _cfg.feature_names:
        missing_time_vars = [
            v
            for v in ["date_unix", "day_julian", "weekday", "hour"]
            if v in _cfg.feature_names and v not in df_work.columns
        ]
        if missing_time_vars:
            try:
                df_work = add_date_variables(df_work)
                (log.info if _cfg.verbose else log.debug)(
                    "Generated time variables: %s", missing_time_vars
                )
            except Exception:
                log.warning(
                    "Could not generate some time features: %s", missing_time_vars, exc_info=False
                )

    if model is None:
        if _cfg.feature_names is None:
            raise ValueError("feature_names must be provided")
        (log.info if _cfg.verbose else log.debug)(
            "Training model via backend='%s' with features=%d...",
            _cfg.backend or "flaml",
            len(_cfg.feature_names),
        )
        df_work, model = build_model(
            df=df_work,
            value="value",
            backend=_cfg.backend or "flaml",
            feature_names=_cfg.feature_names,
            split_method=_cfg.split_method,
            fraction=_cfg.fraction,
            model_config=_cfg.model_config,
            seed=_cfg.seed,
            n_cores=_cfg.n_cores,
            verbose=_cfg.verbose,
        )

    try:
        model_feats = [str(c) for c in extract_features(model)]
    except Exception as exc:
        if not _cfg.feature_names:
            raise ModelError(
                "Cannot infer model features; please provide `feature_names`."
            ) from exc
        model_feats = [str(c) for c in _cfg.feature_names]

    model_feats = [c for c in model_feats if c in df_work.columns]
    if not model_feats:
        raise DataError("No valid model features found in the provided `df` for decomposition.")

    result = (
        pd.DataFrame({"date": df_work["date"].to_numpy(), "observed": observed_series.to_numpy()})
        .set_index("date")
        .sort_index()
    )

    time_vars_order = ["base", "date_unix", "day_julian", "weekday", "hour"]
    present_time_vars = ["base"] + [
        v for v in time_vars_order[1:] if v in model_feats and v in df_work.columns
    ]

    n_cores_eff = _effective_cores(_cfg.n_cores)
    start = time.time()

    resample_vars = [v for v in model_feats if v != "value"]

    for i, var_to_fix in enumerate(present_time_vars, start=1):
        if var_to_fix != "base":
            resample_vars = [v for v in resample_vars if v != var_to_fix]

        elapsed = time.time() - start
        eta = (elapsed / max(i - 1, 1)) * (len(present_time_vars) - (i - 1)) if i > 1 else None
        eta_str = (
            ""
            if eta is None
            else (
                f" | ETA: {eta:.1f}s"
                if eta < 60
                else f" | ETA: {eta / 60:.1f}m"
                if eta < 3600
                else f" | ETA: {eta / 3600:.1f}h"
            )
        )
        (log.info if _cfg.verbose else log.debug)(
            "%s: Decomposing %s%s",
            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            var_to_fix,
            eta_str,
        )

        df_norm = normalise(
            df=df_work,
            model=model,
            feature_names=model_feats,
            variables_resample=resample_vars,
            n_samples=_cfg.n_samples,
            replace=True,
            aggregate=True,
            seed=_cfg.seed,
            n_cores=n_cores_eff,
            resample_df=None,
            memory_save=_cfg.memory_save,
            use_gpu=_cfg.use_gpu,
        )
        if "normalised" not in df_norm.columns:
            log.exception("`normalise` did not return 'normalised' column (aggregate=True).")
            raise ModelError(
                "`normalise` must return a DataFrame with column 'normalised' when aggregate=True."
            )

        result[var_to_fix] = df_norm.reindex(result.index)["normalised"].to_numpy()

    result["emi_total"] = result.get("hour", result["observed"])

    for higher_freq, lower_freq, target_col in [
        ("hour", "weekday", "hour"),
        ("weekday", "day_julian", "weekday"),
        ("day_julian", "date_unix", "day_julian"),
        ("date_unix", "base", "date_unix"),
    ]:
        if higher_freq in result.columns and lower_freq in result.columns:
            result[target_col] = result[higher_freq] - result[lower_freq]

    base_mean = float(result["base"].mean())
    result["emi_noise"] = result["base"] - base_mean
    result["emi_base"] = base_mean
    del result["base"]

    return result


def decom_met(
    df: pd.DataFrame,
    model: object | None = None,
    *,
    config: DecomposeConfig | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Meteorological decomposition via leave-one-out normalisation.

    Iterates over meteorological features ordered by importance, isolating
    the contribution of each feature to the predicted concentration.  Time
    variables are excluded from the meteorological contribution set.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Input data with datetime and target column.
    model : object, optional
        Pre-trained model. If None, a new model will be trained.
    config : DecomposeConfig, optional
        Consolidated config object. Individual keyword arguments (``value``,
        ``backend``, ``feature_names``, …) override the corresponding field
        when provided.

    Returns
    -------
    pandas.DataFrame
        Columns include ``observed``, ``emi_total``, per-meteorological-feature
        contributions, ``met_total``, ``met_base``, and ``met_noise``.

    Raises
    ------
    ValueError
        If required arguments are missing or columns are not found.
    RuntimeError
        If ``normalise`` does not return an ``aggregate`` column.
    """
    _cfg = _resolve_config(config=config, **kwargs)

    if df is None:
        raise DataError("`df` must be provided.")
    if _cfg.value is None:
        raise DataError("`value` (target column name) must be provided.")
    if model is None and _cfg.feature_names is None:
        raise ConfigError("Either `model` or `feature_names` must be provided.")
    if model is None and _cfg.backend is None:
        raise ConfigError("When training a model, `backend` must be specified.")

    df = df.copy()
    if "date" not in df.columns:
        df = process_date(df)
    df = df[df["date"].notna()].sort_values("date").reset_index(drop=True)

    if _cfg.value not in df.columns:
        raise DataError(f"`df` does not contain the target column '{_cfg.value}'.")
    observed_series = df[_cfg.value].copy()

    df_work = df.copy()
    if _cfg.value != "value":
        df_work = df_work.rename(columns={_cfg.value: "value"})

    if _cfg.feature_names:
        time_vars = ["date_unix", "day_julian", "weekday", "hour"]
        missing_time_vars = [
            v for v in time_vars if v in _cfg.feature_names and v not in df_work.columns
        ]
        if missing_time_vars:
            try:
                df_work = add_date_variables(df_work)
                (log.info if _cfg.verbose else log.debug)(
                    "Generated time variables: %s", missing_time_vars
                )
            except Exception:
                log.warning(
                    "Missing time features not generated: %s", missing_time_vars, exc_info=False
                )

    if model is None:
        if _cfg.feature_names is None:
            raise ValueError("feature_names must be provided")
        (log.info if _cfg.verbose else log.debug)(
            "Training model via backend='%s' with features=%d...",
            _cfg.backend or "flaml",
            len(_cfg.feature_names),
        )
        df_work, model = build_model(
            df=df_work,
            value="value",
            backend=_cfg.backend or "flaml",
            feature_names=_cfg.feature_names,
            split_method=_cfg.split_method,
            fraction=_cfg.fraction,
            model_config=_cfg.model_config,
            seed=_cfg.seed,
            verbose=_cfg.verbose,
        )

    try:
        feat_sorted = extract_features(model, importance_ascending=_cfg.importance_ascending)
    except Exception as exc:
        if not _cfg.feature_names:
            raise ModelError(
                "Cannot infer model features; please provide `feature_names`."
            ) from exc
        feat_sorted = list(_cfg.feature_names)

    feat_sorted = [f for f in feat_sorted if f in df_work.columns]
    if not feat_sorted:
        raise DataError("No valid model features found in `df`.")

    time_var_set: set[str] = {"hour", "weekday", "day_julian", "date_unix"}
    contrib_candidates = [f for f in feat_sorted if f not in time_var_set]

    result = (
        pd.DataFrame({"date": df_work["date"].to_numpy(), "observed": observed_series.to_numpy()})
        .set_index("date")
        .sort_index()
    )

    n_cores_eff = _effective_cores(_cfg.n_cores)
    decomp_order = ["emi_total"] + contrib_candidates[:]
    resample_vars = contrib_candidates[:]

    start = time.time()
    tmp: dict[str, np.ndarray] = {}

    for i, var_to_fix in enumerate(decomp_order, start=1):
        if var_to_fix != "emi_total" and var_to_fix in resample_vars:
            resample_vars = [v for v in resample_vars if v != var_to_fix]

        elapsed = time.time() - start
        eta = (elapsed / max(i - 1, 1)) * (len(decomp_order) - (i - 1)) if i > 1 else None
        eta_str = (
            ""
            if eta is None
            else (
                f" | ETA: {eta:.1f}s"
                if eta < 60
                else f" | ETA: {eta / 60:.1f}m"
                if eta < 3600
                else f" | ETA: {eta / 3600:.1f}h"
            )
        )
        (log.info if _cfg.verbose else log.debug)(
            "%s: Decomposing %s%s",
            pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            var_to_fix,
            eta_str,
        )

        df_norm = normalise(
            df=df_work,
            model=model,
            feature_names=feat_sorted,
            variables_resample=resample_vars,
            n_samples=_cfg.n_samples,
            replace=True,
            aggregate=True,
            seed=_cfg.seed,
            n_cores=n_cores_eff,
            resample_df=None,
            memory_save=_cfg.memory_save,
            use_gpu=_cfg.use_gpu,
        )
        if "normalised" not in df_norm.columns:
            log.exception("`normalise` did not return 'normalised' column (aggregate=True).")
            raise ModelError(
                "`normalise` must return a DataFrame with column 'normalised' when aggregate=True."
            )

        tmp[var_to_fix] = df_norm.reindex(result.index)["normalised"].to_numpy()

    result["emi_total"] = tmp["emi_total"]

    prev_key = "emi_total"
    for feat in contrib_candidates:
        result[feat] = tmp[feat] - tmp[prev_key]
        prev_key = feat

    result["met_total"] = result["observed"] - result["emi_total"]
    result["met_base"] = float(result["met_total"].mean())
    contrib_sum = result[contrib_candidates].sum(axis=1) if contrib_candidates else 0.0
    result["met_noise"] = result["met_total"] - (result["met_base"] + contrib_sum)

    return result
