# src/normet/pipeline/multisite.py
"""
Multi-site (multi-station) parallel drivers.

Loops a per-site callable across the unique values of a site column and
concatenates the results in long format. Use these when you have a single
DataFrame that spans many stations and you want to fit / normalise / decompose
each station independently.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pandas as pd
from joblib import Parallel, delayed

from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["multisite_apply", "do_all_multisite", "decompose_multisite"]


def _resolve_workers(n_cores: int | None) -> int:
    return max(1, n_cores if n_cores is not None else (os.cpu_count() or 2) - 1)


def multisite_apply(
    df: pd.DataFrame,
    site_col: str,
    func: Callable[..., pd.DataFrame],
    *,
    n_cores: int | None = None,
    site_kwarg: str | None = None,
    keep_index: bool = True,
    **kwargs,
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
        Parallel workers. Defaults to ``cpu_count() - 1``.
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
            res = func(df=sub, **kwargs, **extra)
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

    pieces = Parallel(n_jobs=n_workers)(delayed(_run)(s) for s in sites)
    pieces = [p for p in pieces if p is not None]
    if not pieces:
        raise RuntimeError("All per-site runs failed.")
    return pd.concat(pieces, ignore_index=True)


def do_all_multisite(
    df: pd.DataFrame,
    site_col: str,
    target: str,
    *,
    covariates: list[str] | None = None,
    backend: str = "flaml",
    n_cores: int | None = None,
    return_models: bool = False,
    **do_all_kwargs,
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
    backend : {"flaml", "lightgbm"}, default "flaml"
        AutoML or model training backend.
    n_cores : int, optional
        Parallel workers across sites.
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
    from .do_all import do_all

    sites = list(pd.unique(df[site_col]))
    n_workers = _resolve_workers(n_cores)
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
                **do_all_kwargs,
            )
            return site_value, out, model
        except Exception as e:
            log.warning("Site %s failed in do_all: %s", site_value, e)
            return None

    results = Parallel(n_jobs=n_workers)(delayed(_run)(s) for s in sites)
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


def decompose_multisite(
    df: pd.DataFrame,
    site_col: str,
    target: str,
    *,
    method: str = "emission",
    covariates: list[str] | None = None,
    backend: str = "flaml",
    n_cores: int | None = None,
    **decompose_kwargs,
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
        Model training backend.
    n_cores : int, optional
        Number of parallel workers.
    **decompose_kwargs
        Additional keyword arguments passed to :func:`normet.decompose`.

    Returns
    -------
    pandas.DataFrame
        Decomposition results for all sites, concatenated.
    """
    from ..analysis.decomposition import decompose

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
