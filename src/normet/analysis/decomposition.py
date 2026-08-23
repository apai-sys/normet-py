# src/normet/analysis/decomposition.py
"""Split a normalised series into emission- and meteorology-driven components.

Provides :func:`decompose` (and the convenience wrappers :func:`decom_emi`,
:func:`decom_met`) plus :class:`DecomposeConfig`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import ConfigError, DataError, ModelError
from ..foundation.estimator import CHRONOS_BACKEND, resolve_n_samples
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

    target: str = "value"
    covariates: list[str] | None = None
    backend: str | None = None
    split_method: str = "random"
    train_fraction: float = 0.75
    model_config: dict[str, Any] | None = None
    #: ``None`` resolves per backend: 300 for the AutoML backends, 8 for
    #: Chronos-2, where each sample is a full transformer forward pass.
    n_samples: int | None = None
    seed: int = DEFAULT_SEED
    n_cores: int | None = None
    memory_save: bool = False
    verbose: bool = False
    importance_ascending: bool = False
    method: str = "emission"
    variable_order: list[str] | None = None
    """Explicit meteorological-feature decomposition order for
    :func:`decom_met` (ignored by :func:`decom_emi`, which always uses its
    own hardcoded calendar order -- see its docstring). If None (default),
    order is derived from fitted feature importance via
    ``importance_ascending``, which can silently reorder "which component
    comes first" across refits of the same features/data with a different
    seed -- results aren't directly comparable run to run. Pass an
    explicit list (must be exactly the model's non-time-variable features,
    in any permutation) to get a decomposition order that stays fixed and
    comparable across runs regardless of the underlying model's importance
    ranking."""
    cache: str | Path | None = None
    """If given, memoize expensive sub-calls to this directory (a
    :class:`joblib.Memory` location): the internal :func:`build_model` fit
    (when ``model=None``) and every per-time-variable :func:`normalise`
    call in the decomposition loop -- ``decom_emi``/``decom_met`` call
    ``normalise`` once per fixed variable, each a full Monte Carlo
    resample-and-predict over ``n_samples`` draws. Off by default."""


def _resolve_config(config: DecomposeConfig | None = None, **kwargs: Any) -> DecomposeConfig:
    return resolve_config(DecomposeConfig, config, **kwargs)


def _effective_cores(n_cores: int | None) -> int:
    """Resolve parallel worker count (>=1)."""
    return max(1, n_cores if n_cores is not None else (os.cpu_count() or 2) - 1)


def _log_decomposition_progress(
    verbose: bool, start: float, i: int, total: int, var_to_fix: str
) -> None:
    """Log a "Decomposing <var>" progress line with an ETA, shared by decom_emi/decom_met."""
    elapsed = time.time() - start
    eta = (elapsed / max(i - 1, 1)) * (total - (i - 1)) if i > 1 else None
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
    (log.info if verbose else log.debug)(
        "%s: Decomposing %s%s",
        pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        var_to_fix,
        eta_str,
    )


def decompose(
    df: pd.DataFrame,
    model: object | None = None,
    *,
    config: DecomposeConfig | None = None,
    method: str = "emission",
    **kwargs: Any,
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
        Consolidated config object. Individual keyword arguments (``target``,
        ``backend``, ``covariates``, …) override the corresponding field
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
    >>> result = decompose(df, method="emission", covariates=["t2m", "blh"],
    ...                    n_samples=2)  # doctest: +SKIP
    """
    _cfg = _resolve_config(config=config, method=method, **kwargs)
    # n_samples is None-by-default so it can follow the backend; the AutoML
    # paths below hand it straight to normalise(), which needs a number.
    _cfg.n_samples = resolve_n_samples(_cfg.n_samples, _cfg.backend)

    if df is None:
        raise DataError("`df` must be provided.")
    if _cfg.target is None:
        raise DataError("`target` must be provided.")
    if model is None and _cfg.covariates is None:
        raise ConfigError("Either `model` or `covariates` must be provided.")
    if model is None and _cfg.backend is None:
        raise ConfigError("When training a model, `backend` must be specified.")

    if _cfg.backend == CHRONOS_BACKEND:
        if _cfg.method == "emission":
            raise ConfigError(
                "method='emission' is not available on the chronos-2 backend. The "
                "emission decomposition isolates the calendar components by resampling "
                "date_unix / day_julian / weekday / hour, which works because an AutoML "
                "model sees them as ordinary features. Chronos-2 conditions on the "
                "target's own history and deweather() never resamples history, so a "
                "repeating calendar signal survives every draw -- the model reads it off "
                "the past and takes nothing from the covariate. Measured covariate "
                "sensitivity, calendar encoders supplied as ordinary covariates: "
                "trend 0.53%, weekly 0.75%, diurnal 1.50%, all six together 2.47%, "
                "against 4.78-10.72% for meteorology in the same runs -- and the diurnal "
                "signal was the largest injected component of all, so attribution runs "
                "opposite to signal size. The components would come back near zero and "
                "read as 'no trend' rather than 'not separable'. Use "
                "method='meteorology', or an AutoML backend."
            )
        if _cfg.method == "meteorology":
            return _decom_met_zero_shot(df=df, model=model, cfg=_cfg)

    if _cfg.method == "emission":
        return decom_emi(df=df, model=model, config=_cfg)

    if _cfg.method == "meteorology":
        return decom_met(df=df, model=model, config=_cfg)

    raise ConfigError(
        f"Unsupported decomposition method: '{_cfg.method}'. "
        "Must be one of 'emission' or 'meteorology'."
    )


def _decom_met_zero_shot(
    df: pd.DataFrame, model: object | None, cfg: DecomposeConfig
) -> pd.DataFrame:
    """Meteorological decomposition on Chronos-2, one nested de-weathering per feature.

    Structurally identical to :func:`decom_met`: start with every meteorological
    feature resampled, fix them one at a time, and take successive differences.
    Only the inner call changes -- :meth:`Chronos2Estimator.deweather` in place
    of :func:`normalise` -- because both answer the same question, "what would
    this series be with these covariates averaged over".

    Ordering is the one real difference. :func:`decom_met` ranks features by
    fitted importance, which does not exist here. ``variable_order`` is used when
    given; otherwise features are ranked by their individual covariate
    sensitivity, which is the zero-shot analogue: how far the forecast moves when
    that one feature is shuffled. Ranking needs more rows than
    ``context_length + prediction_length``; below that the covariate order is
    kept as given and a warning is logged, since an arbitrary order still yields
    a valid decomposition, only a less interpretable one.
    """
    from ..foundation import Chronos2Estimator, to_indexed_frame

    if cfg.target is None:
        raise DataError("`target` must be provided.")
    work = df.copy()
    if "date" not in work.columns:
        work = process_date(work)
    work = work[work["date"].notna()].sort_values("date").reset_index(drop=True)
    if cfg.target not in work.columns:
        raise DataError(f"`df` does not contain the target column '{cfg.target}'.")
    observed = work[cfg.target].to_numpy()
    if cfg.target != "value":
        work = work.rename(columns={cfg.target: "value"})

    time_var_set = {"hour", "weekday", "day_julian", "date_unix"}
    met = [c for c in (cfg.covariates or []) if c not in time_var_set and c in work.columns]
    if not met:
        raise ConfigError(
            "the chronos-2 backend needs meteorological covariates to decompose: "
            "pass covariates, excluding the time variables it cannot use"
        )

    indexed = to_indexed_frame(work)
    est = model if isinstance(model, Chronos2Estimator) else None
    if est is None:
        est = Chronos2Estimator(met_covariates=met, **(cfg.model_config or {}))
        est._load_pipeline()
    est._check_index(indexed.index)

    if cfg.variable_order is not None:
        requested = list(cfg.variable_order)
        if set(requested) != set(met):
            raise ConfigError(
                "`variable_order` must be exactly the meteorological features, in any "
                f"order. Missing: {sorted(set(met) - set(requested))}. "
                f"Not in the model: {sorted(set(requested) - set(met))}."
            )
        met = requested
    else:
        met = _rank_by_sensitivity(est, indexed, met, cfg)

    n_samples = resolve_n_samples(cfg.n_samples, CHRONOS_BACKEND)
    result = pd.DataFrame({"observed": observed}, index=pd.DatetimeIndex(work["date"]))
    result.index.name = "date"

    decomp_order = ["emi_total", *met]
    resample_vars = met[:]
    tmp: dict[str, np.ndarray] = {}
    start = time.time()
    for i, var_to_fix in enumerate(decomp_order, start=1):
        if var_to_fix != "emi_total":
            resample_vars = [v for v in resample_vars if v != var_to_fix]
        _log_decomposition_progress(cfg.verbose, start, i, len(decomp_order), var_to_fix)
        out = est.deweather(
            indexed,
            "value",
            met_features=resample_vars,
            n_samples=n_samples,
            quantiles=(0.5,),
            random_state=cfg.seed,
            schema="normet",
        )
        tmp[var_to_fix] = out["normalised"].reindex(result.index).to_numpy()

    result["emi_total"] = tmp["emi_total"]
    prev = "emi_total"
    for feat in met:
        result[feat] = tmp[feat] - tmp[prev]
        prev = feat

    result["met_total"] = result["observed"] - result["emi_total"]
    result["met_base"] = float(result["met_total"].mean())
    contrib_sum = result[met].sum(axis=1) if met else 0.0
    result["met_noise"] = result["met_total"] - (result["met_base"] + contrib_sum)
    return result


def _rank_by_sensitivity(
    est: Any, indexed: pd.DataFrame, met: list[str], cfg: DecomposeConfig
) -> list[str]:
    """Order meteorological features by how far each one alone moves the forecast."""
    horizon = est.prediction_length
    if len(indexed) <= est.context_length + horizon:
        log.warning(
            "only %d rows: need more than %d to rank features by covariate sensitivity, "
            "so the given covariate order is kept. Pass variable_order to pin it explicitly.",
            len(indexed),
            est.context_length + horizon,
        )
        return met
    anchor = indexed.index[-horizon]
    scores: dict[str, float] = {}
    for feat in met:
        shift = est.covariate_sensitivity(
            indexed,
            "value",
            anchor=anchor,
            horizon=horizon,
            met_features=[feat],
            random_state=cfg.seed,
        )
        scores[feat] = float(shift["mean_abs_shift"])
    ordered = sorted(met, key=lambda f: scores[f], reverse=not cfg.importance_ascending)
    (log.info if cfg.verbose else log.debug)(
        "covariate sensitivity ranking: %s",
        ", ".join(f"{f}={scores[f]:.3f}" for f in ordered),
    )
    return ordered


def decom_emi(
    df: pd.DataFrame,
    model: object | None = None,
    *,
    config: DecomposeConfig | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Emission-based decomposition via leave-one-out normalisation.

    Sequentially fixes time variables, in the fixed order ``base`` ->
    ``date_unix`` -> ``day_julian`` -> ``weekday`` -> ``hour``, to isolate
    the marginal contribution of each temporal component to the predicted
    concentration. Each returned component is the difference between two
    consecutive nested predictions (previous variables already fixed at
    their observed values, current variable now also fixed, everything
    else still resampled).

    .. important::
        **This fixed order is not just bookkeeping -- it determines what
        each component can and cannot represent.** Because ``date_unix``
        is fixed *before* ``day_julian``, ``weekday``, and ``hour``, the
        returned ``date_unix`` ("trend") component is computed while every
        within-year calendar position is still being averaged over
        resampling -- it cannot carry a recurring, calendar-aligned signal
        (e.g. a Christmas/New Year dip that recurs every year), only a
        genuine long-term drift. Conversely, ``day_julian`` (nominally the
        "seasonal" component) is computed with ``date_unix`` already fixed
        at *each row's own observed value*, so it is NOT a pooled,
        climatological quantity the way a bottom-up seasonal factor would
        be -- it stays native to the specific year and can register a
        one-off, non-repeating event (e.g. a single year's holiday dip, or
        a structural break such as a lockdown) despite its "seasonal"
        label. If you need to examine a recurring calendar effect, use
        ``day_julian``, not ``date_unix``, even though "trend" sounds like
        the more natural place to look for it.

    .. note::
        Time variables are opt-in at the model level (see
        :func:`normet.build_model`'s ``covariates``), not mandatory --
        this function adapts automatically. Only whichever of
        ``date_unix``/``day_julian``/``weekday``/``hour`` actually ended up
        as a model feature get decomposed into their own component; the
        rest are simply absent from the result (no error). A model trained
        on none of the four (e.g. meteorology/traffic predictors only)
        still decomposes cleanly into ``base``/``emi_base``/``emi_noise``
        with no time-variable columns at all.

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
    # n_samples is None-by-default so it can follow the backend; the AutoML
    # paths below hand it straight to normalise(), which needs a number.
    _cfg.n_samples = resolve_n_samples(_cfg.n_samples, _cfg.backend)

    if df is None:
        raise DataError("`df` must be provided.")
    if _cfg.target is None:
        raise DataError("`target` (target column name) must be provided.")
    if model is None and _cfg.covariates is None:
        raise ConfigError("Either `model` or `covariates` must be provided.")
    if model is None and _cfg.backend is None:
        raise ConfigError("When training a model, `backend` must be specified.")

    df_work = process_date(df.copy()) if "date" not in df.columns else df.copy()
    if "date" not in df_work.columns:
        raise DataError("Could not find or create a 'date' column.")

    if _cfg.target not in df_work.columns:
        raise DataError(f"`df` does not contain the target column '{_cfg.target}'.")

    observed_series = df_work[_cfg.target].copy()
    if _cfg.target != "value":
        df_work = df_work.rename(columns={_cfg.target: "value"})

    mask_valid = df_work["date"].notna() & df_work["value"].notna()
    df_work = df_work.loc[mask_valid].sort_values("date").reset_index(drop=True)
    observed_series = observed_series.loc[mask_valid].reset_index(drop=True)

    if _cfg.covariates:
        missing_time_vars = [
            v
            for v in ["date_unix", "day_julian", "weekday", "hour"]
            if v in _cfg.covariates and v not in df_work.columns
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
        if _cfg.covariates is None:
            raise ValueError("covariates must be provided")
        (log.info if _cfg.verbose else log.debug)(
            "Training model via backend='%s' with features=%d...",
            _cfg.backend or "flaml",
            len(_cfg.covariates),
        )
        df_work, model = build_model(
            df=df_work,
            target="value",
            backend=_cfg.backend or "flaml",
            covariates=_cfg.covariates,
            split_method=_cfg.split_method,
            train_fraction=_cfg.train_fraction,
            model_config=_cfg.model_config,
            seed=_cfg.seed,
            n_cores=_cfg.n_cores,
            verbose=_cfg.verbose,
            cache=_cfg.cache,
        )

    try:
        model_feats = [str(c) for c in extract_features(model)]
    except Exception as exc:
        if not _cfg.covariates:
            raise ModelError("Cannot infer model features; please provide `covariates`.") from exc
        model_feats = [str(c) for c in _cfg.covariates]

    # The missing-time-vars generation above only runs when the caller
    # passes `covariates` explicitly. Repeat it here against model_feats
    # (covers the case where covariates is instead auto-derived from
    # `model` via extract_features()) so date_unix/day_julian/weekday/hour
    # get generated whenever the model needs them, regardless of which
    # path supplied the feature list. add_date_variables() is idempotent,
    # so this is a no-op whenever the first pass above already handled it.
    missing_time_vars = [
        v
        for v in ["date_unix", "day_julian", "weekday", "hour"]
        if v in model_feats and v not in df_work.columns
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

        _log_decomposition_progress(_cfg.verbose, start, i, len(present_time_vars), var_to_fix)

        df_norm = normalise(
            df=df_work,
            model=model,
            covariates=model_feats,
            variables_resample=resample_vars,
            n_samples=_cfg.n_samples,
            replace=True,
            aggregate=True,
            seed=_cfg.seed,
            n_cores=n_cores_eff,
            resample_df=None,
            memory_save=_cfg.memory_save,
            cache=_cfg.cache,
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
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Meteorological decomposition via leave-one-out normalisation.

    Iterates over meteorological features ordered by importance, isolating
    the contribution of each feature to the predicted concentration.  Time
    variables are excluded from the meteorological contribution set.

    Note the asymmetry with :func:`decom_emi`: that function fixes time
    variables in a *hardcoded* calendar order (``date_unix`` before
    ``day_julian`` before ``weekday`` before ``hour``, chosen so each
    component has a specific temporal-frequency meaning -- see its
    docstring), whereas this function orders meteorological features by
    *fitted importance*, which can vary run to run with the underlying
    model. The two are not directly comparable in how "which component
    comes first" was decided. Pass ``variable_order`` (via ``config`` or
    as a keyword) to pin an explicit order instead, for results that stay
    comparable across model refits.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Input data with datetime and target column.
    model : object, optional
        Pre-trained model. If None, a new model will be trained.
    config : DecomposeConfig, optional
        Consolidated config object. Individual keyword arguments (``target``,
        ``backend``, ``covariates``, …) override the corresponding field
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
    # n_samples is None-by-default so it can follow the backend; the AutoML
    # paths below hand it straight to normalise(), which needs a number.
    _cfg.n_samples = resolve_n_samples(_cfg.n_samples, _cfg.backend)

    if df is None:
        raise DataError("`df` must be provided.")
    if _cfg.target is None:
        raise DataError("`target` (target column name) must be provided.")
    if model is None and _cfg.covariates is None:
        raise ConfigError("Either `model` or `covariates` must be provided.")
    if model is None and _cfg.backend is None:
        raise ConfigError("When training a model, `backend` must be specified.")

    df = df.copy()
    if "date" not in df.columns:
        df = process_date(df)
    df = df[df["date"].notna()].sort_values("date").reset_index(drop=True)

    if _cfg.target not in df.columns:
        raise DataError(f"`df` does not contain the target column '{_cfg.target}'.")
    observed_series = df[_cfg.target].copy()

    df_work = df.copy()
    if _cfg.target != "value":
        df_work = df_work.rename(columns={_cfg.target: "value"})

    if _cfg.covariates:
        time_vars = ["date_unix", "day_julian", "weekday", "hour"]
        missing_time_vars = [
            v for v in time_vars if v in _cfg.covariates and v not in df_work.columns
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
        if _cfg.covariates is None:
            raise ValueError("covariates must be provided")
        (log.info if _cfg.verbose else log.debug)(
            "Training model via backend='%s' with features=%d...",
            _cfg.backend or "flaml",
            len(_cfg.covariates),
        )
        df_work, model = build_model(
            df=df_work,
            target="value",
            backend=_cfg.backend or "flaml",
            covariates=_cfg.covariates,
            split_method=_cfg.split_method,
            train_fraction=_cfg.train_fraction,
            model_config=_cfg.model_config,
            seed=_cfg.seed,
            verbose=_cfg.verbose,
            cache=_cfg.cache,
        )

    try:
        feat_sorted = extract_features(model, importance_ascending=_cfg.importance_ascending)
    except Exception as exc:
        if not _cfg.covariates:
            raise ModelError("Cannot infer model features; please provide `covariates`.") from exc
        feat_sorted = list(_cfg.covariates)

    feat_sorted = [f for f in feat_sorted if f in df_work.columns]
    if not feat_sorted:
        raise DataError("No valid model features found in `df`.")

    time_var_set: set[str] = {"hour", "weekday", "day_julian", "date_unix"}
    contrib_candidates = [f for f in feat_sorted if f not in time_var_set]

    if _cfg.variable_order is not None:
        requested = list(_cfg.variable_order)
        actual_set = set(contrib_candidates)
        requested_set = set(requested)
        if requested_set != actual_set:
            missing = sorted(actual_set - requested_set)
            extra = sorted(requested_set - actual_set)
            raise ConfigError(
                "`variable_order` must be exactly the model's meteorological "
                f"(non-time) features, in any order. Missing: {missing}. "
                f"Not in model: {extra}."
            )
        contrib_candidates = requested

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

        _log_decomposition_progress(_cfg.verbose, start, i, len(decomp_order), var_to_fix)

        df_norm = normalise(
            df=df_work,
            model=model,
            covariates=feat_sorted,
            variables_resample=resample_vars,
            n_samples=_cfg.n_samples,
            replace=True,
            aggregate=True,
            seed=_cfg.seed,
            n_cores=n_cores_eff,
            resample_df=None,
            memory_save=_cfg.memory_save,
            cache=_cfg.cache,
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
