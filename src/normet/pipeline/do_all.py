# src/normet/pipeline/do_all.py
"""Single-call pipelines that prepare data, train a model, and normalise.

Provides :func:`do_all` (one model) and :func:`do_all_unc` (an ensemble of
models for uncertainty quantification), plus their config dataclasses
:class:`SingleConfig` and :class:`UncConfig`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from ..analysis.normalise import normalise
from ..foundation.estimator import (
    CHRONOS_BACKEND,
    resolve_n_samples,
)
from ..model.train import build_model
from ..utils._config import resolve_config
from ..utils.logging import _progress_str, get_logger
from ..utils.metrics import modStats
from ..utils.prepare import prepare_data

log = get_logger(__name__)

__all__ = ["CHRONOS_BACKEND", "SingleConfig", "UncConfig", "do_all", "do_all_unc"]


@dataclass
class SingleConfig:
    """Configuration for :func:`do_all`."""

    target: str = "value"
    backend: str = "flaml"
    covariates: list[str] | None = None
    variables_resample: list[str] | None = None
    split_method: str = "random"
    train_fraction: float = 0.75
    model_config: dict[str, Any] | None = None
    #: ``None`` resolves per backend: 300 for the AutoML backends, 8 for
    #: Chronos-2, where each sample is a full transformer forward pass.
    n_samples: int | None = None
    aggregate: bool = True
    seed: int = 7_654_321
    n_cores: int | None = None
    resample_df: pd.DataFrame | None = None
    memory_save: bool = False
    verbose: bool = False


@dataclass
class UncConfig(SingleConfig):
    """Configuration for :func:`do_all_unc`; adds ensemble settings."""

    n_models: int = 10
    confidence_level: float = 0.95


def _do_all_zero_shot(
    df: pd.DataFrame, cfg: SingleConfig
) -> tuple[pd.DataFrame, object, pd.DataFrame]:
    """The Chronos-2 branch of :func:`do_all`: prepare -> load -> de-weather.

    The AutoML path is prepare -> train -> normalise, and the middle step has no
    counterpart here: nothing is fitted, so there is no held-out set, no model
    to persist and no feature importance. What is returned in the model slot is
    the loaded :class:`~normet.foundation.Chronos2Estimator`, which is what the
    caller needs to run further projections or the covariate-sensitivity check.

    ``split_method``, ``train_fraction`` and ``model_config``'s search settings
    have nothing to act on. Rather than ignore ``model_config`` entirely, its
    keys are passed to the estimator's constructor, which is where ``device``,
    ``context_length`` and ``prediction_length`` belong.
    """
    from ..foundation import Chronos2Estimator, to_indexed_frame

    covariates = list(cfg.covariates or [])
    met = list(cfg.variables_resample) if cfg.variables_resample else covariates
    if not met:
        raise ValueError(
            "the chronos-2 backend needs meteorological covariates to condition on: "
            "pass variables_resample (or covariates), otherwise the run is a plain "
            "forecast rather than a de-weathering"
        )

    df_prep = prepare_data(
        df,
        target=cfg.target,
        covariates=covariates,
        split_method=cfg.split_method,
        train_fraction=cfg.train_fraction,
        seed=cfg.seed,
    )
    indexed = to_indexed_frame(df_prep)

    estimator = Chronos2Estimator(met_covariates=met, **(cfg.model_config or {}))
    estimator._load_pipeline()
    log.info(
        "Chronos-2 ready on %s | context=%d | horizon=%d",
        estimator.device,
        estimator.context_length,
        estimator.prediction_length,
    )
    estimator._check_index(indexed.index)

    out = estimator.deweather(
        indexed,
        "value",
        met_features=met,
        n_samples=cast(int, cfg.n_samples),
        random_state=cfg.seed,
        schema="normet",
    )
    out.index.name = "date"

    seeded = min(estimator.context_length, len(out))
    log.info(
        "do_all finished: %d timestamps, of which the first %d are seeded with the "
        "observed values (Chronos-2 has no history to condition on until then)",
        len(out),
        seeded,
    )
    return out, estimator, df_prep


def _resolve_single_config(config: SingleConfig | None = None, **kwargs: Any) -> SingleConfig:
    return resolve_config(SingleConfig, config, **kwargs)


def _resolve_unc_config(config: UncConfig | None = None, **kwargs: Any) -> UncConfig:
    return resolve_config(UncConfig, config, **kwargs)


def do_all(
    df: pd.DataFrame,
    target: str | None = None,
    *,
    config: SingleConfig | None = None,
    backend: str = "flaml",
    covariates: list[str] | None = None,
    variables_resample: list[str] | None = None,
    split_method: str = "random",
    train_fraction: float = 0.75,
    model_config: dict[str, Any] | None = None,
    n_samples: int | None = None,
    seed: int = 7_654_321,
    n_cores: int | None = None,
    resample_df: pd.DataFrame | None = None,
    memory_save: bool = False,
    verbose: bool = False,
    aggregate: bool = True,
    **kwargs: Any,
) -> tuple[pd.DataFrame, object, pd.DataFrame]:
    r"""Run the standard pipeline: prepare → build_model → normalise.

    With ``backend="chronos-2"`` the middle step is skipped: nothing is trained,
    the Chronos-2 checkpoint is loaded instead and the de-weathering runs through
    :meth:`~normet.foundation.Chronos2Estimator.deweather`. The return shape is
    unchanged, but the model slot holds the loaded estimator, ``split_method`` /
    ``train_fraction`` have nothing to act on, and ``model_config`` is forwarded
    to the estimator's constructor (``device``, ``context_length``,
    ``prediction_length``). Note that the first ``context_length`` rows of the
    result are the observed values: the model has no history to condition on
    until then. Needs the ``foundation`` extra.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    target : str or None, default=None
        Target column in *df*. Can be ``None`` when *config* is provided
        with a non-None ``target`` field.
    config : SingleConfig or None, default=None
        Optional :class:`SingleConfig` holding all parameters.
        Individual keyword arguments override fields on *config*\ .
    **kwargs
        Any additional keyword arguments are forwarded to
        :func:`_resolve_single_config`.

    Returns
    -------
    (pandas.DataFrame, object, pandas.DataFrame)
        Normalised contributions, trained model, prepared dataframe.

    Examples
    --------
    >>> import pandas as pd
    >>> from normet import do_all
    >>> df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=48, freq="h"),
    ...                    "PM2.5": range(48), "t2m": 10.0, "blh": 500.0})
    >>> result, model, df_prep = do_all(df, target="PM2.5",
    ...                                 covariates=["t2m", "blh"])  # doctest: +SKIP
    """
    _cfg = _resolve_single_config(
        config=config,
        target=target,
        backend=backend,
        covariates=covariates,
        variables_resample=variables_resample,
        split_method=split_method,
        train_fraction=train_fraction,
        model_config=model_config,
        n_samples=n_samples,
        seed=seed,
        n_cores=n_cores,
        resample_df=resample_df,
        memory_save=memory_save,
        verbose=verbose,
        aggregate=aggregate,
        **kwargs,
    )

    if _cfg.target is None:
        raise ValueError("`target` must be provided either directly or via config.")

    _cfg.n_samples = resolve_n_samples(_cfg.n_samples, _cfg.backend)
    log.info(
        "Starting do_all | backend=%s | target=%s | n_samples=%d",
        _cfg.backend,
        _cfg.target,
        _cfg.n_samples,
    )

    if _cfg.covariates is None:
        raise ValueError("covariates must be provided")

    if _cfg.backend == CHRONOS_BACKEND:
        return _do_all_zero_shot(df, _cfg)
    df_prep = prepare_data(
        df,
        target=_cfg.target,
        covariates=_cfg.covariates,
        split_method=_cfg.split_method,
        train_fraction=_cfg.train_fraction,
        seed=_cfg.seed,
    )
    log.info(
        "Data prepared: %d rows (%d training, %d testing)",
        len(df_prep),
        int((df_prep["set"] == "training").sum()) if "set" in df_prep.columns else len(df_prep),
        int((df_prep["set"] == "testing").sum()) if "set" in df_prep.columns else 0,
    )

    df_prep, model = build_model(
        df=df_prep,
        target="value",
        backend=_cfg.backend,
        covariates=_cfg.covariates,
        split_method=_cfg.split_method,
        train_fraction=_cfg.train_fraction,
        model_config=_cfg.model_config,
        seed=_cfg.seed,
        n_cores=_cfg.n_cores,
        verbose=_cfg.verbose,
    )
    log.info("Model trained with backend=%s", _cfg.backend)

    out = normalise(
        df=df_prep,
        model=model,
        covariates=_cfg.covariates or [c for c in df_prep.columns if c not in {"value"}],
        variables_resample=_cfg.variables_resample,
        n_samples=_cfg.n_samples,
        aggregate=_cfg.aggregate,
        seed=_cfg.seed,
        n_cores=_cfg.n_cores,
        resample_df=_cfg.resample_df,
        memory_save=_cfg.memory_save,
        verbose=_cfg.verbose,
    )

    log.info("do_all finished: %d timestamps", len(out))
    return out, model, df_prep


def do_all_unc(
    df: pd.DataFrame,
    target: str | None = None,
    *,
    config: UncConfig | None = None,
    backend: str = "flaml",
    covariates: list[str] | None = None,
    variables_resample: list[str] | None = None,
    split_method: str = "random",
    train_fraction: float = 0.75,
    model_config: dict[str, Any] | None = None,
    n_samples: int | None = None,
    n_models: int = 10,
    confidence_level: float = 0.95,
    seed: int = 7_654_321,
    n_cores: int | None = None,
    resample_df: pd.DataFrame | None = None,
    memory_save: bool = False,
    verbose: bool = False,
    weighted_method: str = "r2",
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    r"""Run the uncertainty pipeline: :func:`do_all` repeated *n_models* times.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset.
    target : str or None, default=None
        Target column in *df*. Can be ``None`` when *config* is provided
        with a non-None ``target`` field.
    config : UncConfig or None, default=None
        Optional :class:`UncConfig` holding all parameters.
        Individual keyword arguments override fields on *config*\ .
    **kwargs
        Any additional keyword arguments are forwarded to
        :func:`_resolve_unc_config`.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Normalised contributions and model statistics.
    """
    _cfg = _resolve_unc_config(
        config=config,
        target=target,
        backend=backend,
        covariates=covariates,
        variables_resample=variables_resample,
        split_method=split_method,
        train_fraction=train_fraction,
        model_config=model_config,
        n_samples=n_samples,
        n_models=n_models,
        confidence_level=confidence_level,
        seed=seed,
        n_cores=n_cores,
        resample_df=resample_df,
        memory_save=memory_save,
        verbose=verbose,
        **kwargs,
    )

    weighted_method = (weighted_method or "r2").lower()
    if weighted_method not in {"r2", "rmse"}:
        raise ValueError("`weighted_method` must be 'r2' or 'rmse'.")

    rng = np.random.default_rng(_cfg.seed)
    seeds = rng.choice(1_000_001, size=_cfg.n_models, replace=False).tolist()

    series_list: list[pd.DataFrame] = []
    stats_list: list[pd.DataFrame] = []
    observed_ref: pd.Series | None = None

    t0 = time.time()
    for i, s in enumerate(seeds, start=1):
        (log.info if _cfg.verbose else log.debug)(
            "do_all_unc: running model %d/%d (seed=%d) %s",
            i,
            _cfg.n_models,
            s,
            _progress_str(i, _cfg.n_models, t0),
        )

        out_i, model_i, df_prep_i = do_all(
            df=df,
            target=_cfg.target,
            backend=_cfg.backend,
            covariates=_cfg.covariates,
            variables_resample=_cfg.variables_resample,
            split_method=_cfg.split_method,
            train_fraction=_cfg.train_fraction,
            model_config=_cfg.model_config,
            n_samples=_cfg.n_samples,
            seed=int(s),
            n_cores=_cfg.n_cores,
            resample_df=_cfg.resample_df,
            memory_save=_cfg.memory_save,
            aggregate=True,
        )

        if observed_ref is None:
            observed_ref = out_i["observed"].copy()

        col = f"normalised_{s}"
        series_list.append(out_i[["normalised"]].rename(columns={"normalised": col}))

        try:
            stats_i = modStats(df=df_prep_i, model=model_i, subset=None, statistic=None)
            if isinstance(stats_i, pd.DataFrame) and not stats_i.empty:
                stats_i = stats_i.copy()
                stats_i["seed"] = int(s)
                stats_list.append(stats_i)
        except Exception as e:
            log.warning("Failed to compute metrics for seed %d: %s", s, e)

    if observed_ref is None:
        raise RuntimeError("do_all_unc produced no outputs — verify inputs and seeds.")

    out = observed_ref.to_frame(name="observed")
    for s in series_list:
        out = out.join(s, how="outer")

    pred_cols = [c for c in out.columns if c.startswith("normalised_")]
    P = out[pred_cols]

    out["mean"] = P.mean(axis=1)
    out["std"] = P.std(axis=1)
    out["median"] = P.median(axis=1)

    alpha = (1.0 - _cfg.confidence_level) / 2.0
    out["lower_bound"] = P.quantile(alpha, axis=1)
    out["upper_bound"] = P.quantile(1.0 - alpha, axis=1)

    def _pick_metric(df_in: pd.DataFrame, names: list[str]) -> float | None:
        for n in names:
            if n in df_in.columns:
                try:
                    return float(df_in[n].iloc[0])
                except Exception:
                    continue
        return None

    perf_rows = []
    for si in stats_list:
        sd = int(si["seed"].iloc[0])
        r2_val = _pick_metric(si, ["r2", "R2", "r_squared", "R2_score"])
        rmse_val = _pick_metric(si, ["rmse", "RMSE", "root_mean_squared_error"])
        perf_rows.append({"seed": sd, "r2": r2_val, "rmse": rmse_val})
    perf_df = (
        pd.DataFrame(perf_rows).set_index("seed")
        if perf_rows
        else pd.DataFrame(columns=["r2", "rmse"])
    )

    def _parse_seed(col: str) -> int | None:
        try:
            return int(col.split("_", 1)[1])
        except Exception:
            return None

    seeds_in_P = [s for s in map(_parse_seed, pred_cols)]

    scores = np.zeros(len(seeds_in_P), dtype=float)
    if not perf_df.empty:
        if weighted_method == "r2":
            for i, s in enumerate(seeds_in_P):
                if s is None or s not in perf_df.index:
                    continue
                r2 = perf_df.loc[s, "r2"]
                if pd.notna(r2):
                    scores[i] = max(float(cast(Any, r2)), 0.0)
        else:
            eps = 1e-9
            for i, s in enumerate(seeds_in_P):
                if s is None or s not in perf_df.index:
                    continue
                rmse = perf_df.loc[s, "rmse"]
                if pd.notna(rmse):
                    scores[i] = 1.0 / (float(cast(Any, rmse)) + eps)

    if np.all(~np.isfinite(scores)) or np.all(scores <= 0):
        w = np.full(len(pred_cols), 1.0 / len(pred_cols)) if pred_cols else np.array([])
    else:
        scores[:] = np.where(np.isfinite(scores) & (scores > 0), scores, 0.0)
        ssum = scores.sum()
        w = scores / ssum if ssum > 0 else np.full(len(pred_cols), 1.0 / len(pred_cols))

    if pred_cols:
        out["weighted"] = (P.values * w[np.newaxis, :]).sum(axis=1)
    else:
        out["weighted"] = np.nan

    w_by_seed = {s: float(w[i]) for i, s in enumerate(seeds_in_P) if i < len(w)}
    mod_stats = pd.concat(stats_list, ignore_index=True) if stats_list else pd.DataFrame()
    if not mod_stats.empty and "seed" in mod_stats.columns:
        mod_stats["weight"] = mod_stats["seed"].map(w_by_seed).astype(float)

    return out, mod_stats
