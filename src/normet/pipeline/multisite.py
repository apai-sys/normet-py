# src/normet/pipeline/multisite.py
"""
Multi-site (multi-station) parallel drivers.

Loops a per-site callable across the unique values of a site column and
concatenates the results in long format. Use these when you have a single
DataFrame that spans many stations and you want to fit / normalise / decompose
each station independently.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from ..foundation.estimator import CHRONOS_BACKEND, resolve_n_samples
from ..utils._config import dispatch_batch_size, resolve_workers
from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "multisite_apply",
    "do_all_multisite",
    "decompose_multisite",
    "embed_multisite",
]


def _resolve_workers(n_cores: int | None) -> int:
    return resolve_workers(n_cores)


def _inner_site_cores(n_sites: int) -> int | None:
    """Per-site worker count for a driver whose outer pool already spans sites.

    Every worker process resolves ``n_cores=None`` to the *whole* job
    allocation, so with P concurrent sites each one opened its own full-width
    thread pool inside its process: P × (C − 1) threads on C cores, with
    nothing in the call signalling it. The outer pool fills the allocation,
    so more than one site runs single-threaded inside its worker; a one-site
    call keeps the callee's own default (there is nothing to share).
    """
    return 1 if n_sites > 1 else None


def multisite_apply(
    df: pd.DataFrame,
    site_col: str,
    func: Callable[..., pd.DataFrame],
    *,
    n_cores: int | None = None,
    inner_n_cores: int | None = None,
    site_kwarg: str | None = None,
    keep_index: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Run ``func(df=site_df, **kwargs)`` for each unique site and concatenate results.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format input containing a ``site_col`` column.
    site_col : str
        Column that identifies the site / station / unit.
    func : callable
        Per-site function. Must return a pandas DataFrame.
    n_cores : int, optional
        Parallel workers. Defaults to the job's allocation minus one
        (see :func:`normet.utils.available_cores`), not the machine's
        core count. Workers are processes that each inherit the whole
        allocation, so a ``func`` that sizes its own pool from
        ``n_cores=None`` would run one full pool per site; pass
        ``inner_n_cores`` for that.
    inner_n_cores : int, optional
        Forwarded to ``func`` as ``n_cores``; a ``func`` that takes no such
        keyword must be left without it. With P concurrent sites, a ``func``
        that resolves ``n_cores=None`` to the whole allocation opens P full
        pools inside one allocation (P × (C − 1) threads on C cores); pass
        ``1`` when the outer pool is the parallelism and the callee takes an
        ``n_cores`` argument. Opt-in rather than injected, because the
        generic ``func`` need not accept it.
    site_kwarg : str, optional
        If set, the site id will also be passed to ``func`` under this kwarg
        (useful when ``func`` needs to know its site).
    keep_index : bool, default True
        If True, the returned DataFrame's original index becomes a ``date``
        column (if it's a DatetimeIndex), then the site column is appended.
    **kwargs :
        Forwarded to ``func`` as-is.

    Returns
    -------
    pandas.DataFrame
        Long-format concatenation with the ``site_col`` column appended/restored.
    """
    if site_col not in df.columns:
        raise ValueError(f"`site_col` '{site_col}' not in df.")

    sites = list(pd.unique(df[site_col]))
    if not sites:
        return pd.DataFrame()

    n_workers = _resolve_workers(n_cores)
    log.info("multisite_apply: %d sites × workers=%d", len(sites), n_workers)

    def _run(site_value: Any) -> pd.DataFrame | None:
        sub = df[df[site_col] == site_value]
        if sub.empty:
            return None
        try:
            extra = {site_kwarg: site_value} if site_kwarg else {}
            call_kwargs = dict(kwargs)
            if inner_n_cores is not None:
                call_kwargs.setdefault("n_cores", inner_n_cores)
            res = func(df=sub, **call_kwargs, **extra)
            if isinstance(res, tuple):  # type: ignore[unreachable]
                res = res[0]  # type: ignore[unreachable]
            if res is None or len(res) == 0:
                return None
            out = res.copy()
            if keep_index and isinstance(out.index, pd.DatetimeIndex):
                out = out.reset_index().rename(columns={out.index.name or "index": "date"})
            out[site_col] = site_value
            return out
        except Exception as e:
            log.warning("Site %s failed: %s", site_value, e)
            return None

    # joblib is ~100 ms of `import normet`; only this driver needs it.
    from joblib import Parallel, delayed

    pieces = Parallel(n_jobs=n_workers, batch_size=dispatch_batch_size(len(sites), n_workers))(
        delayed(_run)(s) for s in sites
    )
    pieces = [p for p in pieces if p is not None]
    if not pieces:
        raise RuntimeError("All per-site runs failed.")
    return pd.concat(pieces, ignore_index=True)


def _do_all_multisite_chronos(
    df: pd.DataFrame,
    site_col: str,
    target: str,
    *,
    covariates: list[str] | None,
    return_models: bool,
    do_all_kwargs: dict[str, Any],
) -> Any:
    """Single-checkpoint fast path for ``do_all_multisite(..., backend="chronos-2")``.

    Loads Chronos-2 once in-process instead of spawning one loky worker per site,
    and routes ``model_config={"method": "icn"}`` through
    :meth:`Chronos2Estimator.normalise_in_context_multisite` so cross-learning
    information sharing can span all sites in one batched call.
    """
    from ..foundation import (
        Chronos2Estimator,
        ICNDesign,
        MetPool,
        add_calendar_covariates,
        to_indexed_frame,
    )
    from ..utils.prepare import expm1_columns, log1p_target, prepare_data

    if site_col not in df.columns:
        raise ValueError(f"`site_col` '{site_col}' not in df.")

    sites = list(pd.unique(df[site_col]))
    if not sites:
        raise RuntimeError("All per-site do_all runs failed.")

    mc = dict(do_all_kwargs.get("model_config") or {})
    model_path = mc.pop("model_path", None)
    method = str(mc.pop("method", "deweather")).lower()
    estimate = str(mc.pop("estimate", "level")).lower()
    icn_design = mc.pop("icn_design", None)
    ext_pool = mc.pop("pool", None)
    cross_learning = bool(mc.pop("cross_learning", True))
    estimator_arg = mc.pop("estimator", None)
    if method not in ("deweather", "icn"):
        raise ValueError(f"model_config['method'] must be 'deweather' or 'icn', got {method!r}")
    if estimate not in ("level", "paired"):
        raise ValueError(f"model_config['estimate'] must be 'level' or 'paired', got {estimate!r}")

    cov_list = list(covariates or [])
    variables_resample = do_all_kwargs.get("variables_resample")
    met = list(variables_resample) if variables_resample else cov_list
    if not met:
        raise ValueError(
            "the chronos-2 backend needs meteorological covariates to condition on: "
            "pass variables_resample (or covariates), otherwise the run is a plain "
            "forecast rather than a de-weathering"
        )
    kept = cov_list + [c for c in met if c not in cov_list]

    n_samples = resolve_n_samples(do_all_kwargs.get("n_samples"), CHRONOS_BACKEND)
    seed = int(do_all_kwargs.get("seed", 7_654_321))
    log_transform = bool(do_all_kwargs.get("log_transform", False))
    split_method = str(do_all_kwargs.get("split_method", "random"))
    train_fraction = float(do_all_kwargs.get("train_fraction", 0.75))
    resample_df = do_all_kwargs.get("resample_df")

    if estimator_arg is not None:
        estimator = estimator_arg
        if estimator.met_covariates is None:
            estimator.met_covariates = list(met)
    elif model_path is not None:
        estimator = Chronos2Estimator.load(model_path, device=mc.get("device"))
        if estimator.met_covariates is None:
            estimator.met_covariates = list(met)
    else:
        estimator = Chronos2Estimator(met_covariates=met, **mc)
    estimator._load_pipeline()
    log.info(
        "do_all_multisite (chronos-2, %s): %d sites sharing one checkpoint on %s",
        method,
        len(sites),
        estimator.device,
    )

    prepared: list[tuple[Any, pd.DataFrame]] = []
    for site_value in sites:
        sub = df[df[site_col] == site_value]
        if sub.empty:
            continue
        try:
            df_prep = prepare_data(
                sub.drop(columns=[site_col]),
                target=target,
                covariates=kept,
                split_method=split_method,
                train_fraction=train_fraction,
                seed=seed,
            )
            if log_transform and method == "deweather":
                df_prep = log1p_target(df_prep)
            indexed = to_indexed_frame(df_prep)
            estimator._check_index(indexed.index)
            prepared.append((site_value, indexed))
        except Exception as exc:
            log.warning("Site %s failed in do_all preparation: %s", site_value, exc)

    if not prepared:
        raise RuntimeError("All per-site do_all runs failed.")

    pieces: list[pd.DataFrame] = []
    models: dict[Any, Any] = {}

    if method == "icn":
        if isinstance(icn_design, ICNDesign):
            design = icn_design
        else:
            overrides: dict[str, Any] = {
                "context_length": estimator.context_length,
                "n_resamples": n_samples,
                "seed": seed,
                "log_transform": log_transform,
            }
            if "prediction_length" in mc:
                bh = int(mc["prediction_length"])
                overrides["block_h"] = bh
                overrides["stride"] = max(1, bh // 2)
            if isinstance(icn_design, Mapping):
                overrides.update(icn_design)
            design = ICNDesign.v1(**overrides)

        pool_arg: Any = ext_pool
        if pool_arg is None and resample_df is not None:
            pool_arg = MetPool.from_frame(to_indexed_frame(resample_df), met)

        site_frames: dict[str, pd.DataFrame] = {}
        orig_keys: dict[str, Any] = {}
        for site_value, indexed in prepared:
            key = str(site_value)
            site_frames[key] = add_calendar_covariates(indexed)
            orig_keys[key] = site_value

        res_map = estimator.normalise_in_context_multisite(
            site_frames,
            "value",
            design=design,
            met_features=met,
            pool=pool_arg,
            quantiles=(0.1, 0.5, 0.9),
            cross_learning=cross_learning,
        )
        for key, res in res_map.items():
            site_value = orig_keys[key]
            out = res.to_normet_frame(estimate=estimate)
            df_out = out.reset_index().rename(columns={out.index.name or "index": "date"})
            df_out[site_col] = site_value
            pieces.append(df_out)
            models[site_value] = estimator
    else:
        for site_value, indexed in prepared:
            try:
                out = estimator.deweather(
                    indexed,
                    "value",
                    met_features=met,
                    n_samples=n_samples,
                    random_state=seed,
                    schema="normet",
                )
                out.index.name = "date"
                if log_transform:
                    out = expm1_columns(out)
                df_out = out.reset_index().rename(columns={out.index.name or "index": "date"})
                df_out[site_col] = site_value
                pieces.append(df_out)
                models[site_value] = estimator
            except Exception as exc:
                log.warning("Site %s failed in do_all: %s", site_value, exc)

    if not pieces:
        raise RuntimeError("All per-site do_all runs failed.")

    df_norm = pd.concat(pieces, ignore_index=True)
    return (df_norm, models) if return_models else df_norm


def do_all_multisite(
    df: pd.DataFrame,
    site_col: str,
    target: str,
    *,
    covariates: list[str] | None = None,
    backend: str = "flaml",
    n_cores: int | None = None,
    return_models: bool = False,
    **do_all_kwargs: Any,
) -> Any:
    """
    Run :func:`normet.do_all` independently per site.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format input with ``site_col`` and the target ``target``.
    site_col : str
        Site/station identifier column.
    target : str
        Target column name (e.g., "PM2.5").
    covariates : list of str, optional
        Predictor columns. Required by ``do_all`` for non-trivial runs.
    backend : {"flaml", "lightgbm", "chronos-2"}, default "flaml"
        AutoML, model training, or zero-shot Chronos-2 backend. On
        ``chronos-2``, one checkpoint is loaded in-process and shared across
        all sites (supporting ``model_config={"method": "icn", ...}`` via
        cross-learning multi-site ICN).
    n_cores : int, optional
        Parallel workers across sites. With more than one site each site's
        own model training and resampling run single-threaded, so the outer
        pool is the parallelism that fits the allocation (see
        :func:`_inner_site_cores`).
    return_models : bool, default False
        If True, also return a ``dict`` mapping site → trained model.
    **do_all_kwargs :
        Forwarded to :func:`do_all` (e.g., ``n_samples``, ``model_config``,
        ``split_method``, ``train_fraction``).

    Returns
    -------
    pandas.DataFrame
        Long-format normalised series with the ``site_col`` column appended.
        If ``return_models=True``, returns ``(df_norm, models)``.
    """
    if str(backend).lower() == CHRONOS_BACKEND:
        return _do_all_multisite_chronos(
            df,
            site_col,
            target,
            covariates=covariates,
            return_models=return_models,
            do_all_kwargs=do_all_kwargs,
        )

    from .do_all import do_all

    if site_col not in df.columns:
        raise ValueError(f"`site_col` '{site_col}' not in df.")

    sites = list(pd.unique(df[site_col]))
    n_workers = _resolve_workers(n_cores)
    inner_cores = _inner_site_cores(len(sites))
    log.info("do_all_multisite: %d sites × workers=%d | backend=%s", len(sites), n_workers, backend)

    def _run(site_value: Any) -> tuple[Any, ...] | None:
        sub = df[df[site_col] == site_value]
        if sub.empty:
            return None
        try:
            out, model, df_prep = do_all(
                df=sub.drop(columns=[site_col]),
                target=target,
                backend=backend,
                covariates=covariates,
                n_cores=inner_cores,
                **do_all_kwargs,
            )
            return site_value, out, model
        except Exception as e:
            log.warning("Site %s failed in do_all: %s", site_value, e)
            return None

    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_workers, batch_size=dispatch_batch_size(len(sites), n_workers))(
        delayed(_run)(s) for s in sites
    )
    results = [r for r in results if r is not None]
    if not results:
        raise RuntimeError("All per-site do_all runs failed.")

    pieces = []
    models: dict[Any, Any] = {}
    for site_value, out, model in results:
        df_out = out.reset_index().rename(columns={out.index.name or "index": "date"})
        df_out[site_col] = site_value
        pieces.append(df_out)
        models[site_value] = model

    df_norm = pd.concat(pieces, ignore_index=True)
    return (df_norm, models) if return_models else df_norm


def _decompose_multisite_chronos(
    df: pd.DataFrame,
    site_col: str,
    target: str,
    *,
    method: str,
    covariates: list[str] | None,
    decompose_kwargs: dict[str, Any],
) -> pd.DataFrame:
    """Single-checkpoint in-process fast path for ``decompose_multisite(..., backend="chronos-2")``."""
    from ..analysis.decomposition import decompose
    from ..foundation import Chronos2Estimator

    if site_col not in df.columns:
        raise ValueError(f"`site_col` '{site_col}' not in df.")

    sites = list(pd.unique(df[site_col]))
    if not sites:
        return pd.DataFrame()

    kw = dict(decompose_kwargs)
    kw.setdefault("n_cores", 1)
    model_arg = kw.pop("model", None)
    if model_arg is None:
        mc = dict(kw.get("model_config") or {})
        model_path = mc.pop("model_path", None)
        for non_ctor in ("method", "estimate", "icn_design", "pool", "cross_learning", "estimator"):
            mc.pop(non_ctor, None)
        estimator = (
            Chronos2Estimator.load(model_path, device=mc.get("device"))
            if model_path is not None
            else Chronos2Estimator(**mc)
        )
    else:
        estimator = model_arg

    if getattr(estimator, "_pipeline", None) is None:
        estimator._load_pipeline()

    pieces: list[pd.DataFrame] = []
    for site_value in sites:
        sub = df[df[site_col] == site_value]
        if sub.empty:
            continue
        try:
            res = decompose(
                method=method,
                df=sub.drop(columns=[site_col]),
                target=target,
                covariates=covariates,
                backend=CHRONOS_BACKEND,
                model=estimator,
                **kw,
            )
            if len(res) == 0:
                continue
            out = res.copy()
            if isinstance(out.index, pd.DatetimeIndex):
                out = out.reset_index().rename(columns={out.index.name or "index": "date"})
            out[site_col] = site_value
            pieces.append(out)
        except Exception as exc:
            log.warning("Site %s failed: %s", site_value, exc)

    if not pieces:
        raise RuntimeError("All per-site runs failed.")
    return pd.concat(pieces, ignore_index=True)


def decompose_multisite(
    df: pd.DataFrame,
    site_col: str,
    target: str,
    *,
    method: str = "emission",
    covariates: list[str] | None = None,
    backend: str = "flaml",
    n_cores: int | None = None,
    **decompose_kwargs: Any,
) -> pd.DataFrame:
    """
    Run :func:`normet.decompose` independently per site.

    Parameters
    ----------
    df : pandas.DataFrame
        Combined multi-site input data.
    site_col : str
        Column identifying each site.
    target : str
        Target column name.
    method : {"emission", "meteorology"}
        Decomposition strategy forwarded to :func:`normet.decompose`.
    covariates : list of str, optional
        Features used for training and decomposition.
    backend : str
        Model training backend. On ``chronos-2``, loads one checkpoint in-process
        and reuses it across sites instead of spawning worker processes.
    n_cores : int, optional
        Number of parallel workers. With more than one site each site's
        decomposition runs its normalisation calls single-threaded, so the
        outer pool is the parallelism that fits the allocation.
    **decompose_kwargs
        Additional keyword arguments passed to :func:`normet.decompose`.

    Returns
    -------
    pandas.DataFrame
        Decomposition results for all sites, concatenated.
    """
    if str(backend).lower() == CHRONOS_BACKEND:
        return _decompose_multisite_chronos(
            df,
            site_col,
            target,
            method=method,
            covariates=covariates,
            decompose_kwargs=decompose_kwargs,
        )

    from ..analysis.decomposition import decompose

    if site_col in df.columns and df[site_col].nunique() > 1:
        # The outer pool spans sites; a decompose() that resolved
        # ``n_cores=None`` to the whole allocation would open one full pool
        # per site (see :func:`_inner_site_cores`).
        decompose_kwargs.setdefault("n_cores", 1)

    return multisite_apply(
        df,
        site_col=site_col,
        func=lambda df: decompose(
            method=method,
            df=df.drop(columns=[site_col]) if site_col in df.columns else df,
            target=target,
            covariates=covariates,
            backend=backend,
            **decompose_kwargs,
        ),
        n_cores=n_cores,
    )


# --------------------------------------------------------------- embeddings


def _wide_by_site(df: pd.DataFrame, site_col: str, target: str, date_col: str) -> pd.DataFrame:
    """Pivot a long multi-site frame to one column per site, on a shared time axis."""
    for col in (site_col, target):
        if col not in df.columns:
            raise ValueError(f"column {col!r} not in df")
    work = df.copy()
    if date_col in work.columns:
        work = work.set_index(date_col)
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index)
    work.index.name = work.index.name or date_col
    wide = work.pivot_table(index=work.index.name, columns=site_col, values=target, aggfunc="mean")
    return wide.sort_index()


def embed_multisite(
    df: pd.DataFrame,
    site_col: str,
    target: str,
    *,
    date_col: str = "date",
    context_length: int = 2048,
    batch_size: int = 32,
    device: str | None = None,
    min_coverage: float = 0.1,
    **embedder_kwargs: Any,
) -> dict[Any, Any]:
    """Embed every site's target series with Chronos-2, from one long-format frame.

    The rest of this module runs a *model* per site. This runs no model at all:
    it reads each site's series through Chronos-2's encoder and returns the
    768-D vector that describes its dynamics, which is what lets sites be
    compared by how they behave rather than by where they are.

    The long-to-wide pivot is the whole bridge. ``ChronosEmbedder.embed_stations``
    wants ``{site: series}`` or one column per site; multi-site frames in this
    package are long, so nothing connected the two before.

    Sites are embedded in a single batched pass, and every series is cut or
    NaN-padded to ``context_length`` first, so a site's vector does not depend on
    which other sites shared its batch.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format frame with ``site_col``, ``target`` and a timestamp.
    site_col, target : str
        Site identifier and the series to embed (e.g. ``"PM2.5"``).
    date_col : str, default "date"
        Timestamp column. Ignored if the frame is already time-indexed.
    device : str, optional
        ``None`` picks CUDA, then Apple Silicon's Metal backend, then the CPU.
    **embedder_kwargs
        Forwarded to :class:`~normet.foundation.ChronosEmbedder`.

    Returns
    -------
    dict
        ``{site: 768-D array}``. Needs the ``foundation`` extra.
    """
    from ..foundation import ChronosEmbedder

    wide = _wide_by_site(df, site_col, target, date_col)
    log.info("embed_multisite: %d sites x %d timestamps", wide.shape[1], wide.shape[0])

    embedder = ChronosEmbedder(device=device, min_coverage=min_coverage, **embedder_kwargs)
    by_name = embedder.embed_stations(wide, context_length=context_length, batch_size=batch_size)

    # embed_stations keys by str(site); hand back the caller's own site values so
    # the result joins against their frame without a string round trip.
    originals = {str(site): site for site in wide.columns}
    return {originals.get(name, name): vec for name, vec in by_name.items()}
