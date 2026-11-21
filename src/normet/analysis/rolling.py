# src/normet/analysis/rolling.py
"""Rolling-window weather normalisation for trend analysis.

Provides :func:`rolling`, which normalises a sliding window of the data
against a model and returns one normalised series per window, plus
:class:`RollingConfig` and :func:`window_columns`, which picks those series out
of the result.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..model.train import build_model
from ..utils import progress
from ..utils._config import DEFAULT_SEED, resolve_config, resolve_workers
from ..utils.features import extract_features
from ..utils.logging import _progress_str, get_logger
from ..utils.prepare import add_date_variables, log1p_target, process_date
from .normalise import _is_time_feature, normalise

log = get_logger(__name__)

_WINDOW_COLUMN = re.compile(r"window_\d+")


def window_column(i: int, total: int) -> str:
    """The column of window *i* (counted from 1) of *total*: ``window_01``,
    ``window_02``, ..., padded to the width of *total* so the names sort in the
    windows' order."""
    return f"window_{i:0{max(2, len(str(total)))}d}"


def window_columns(df: pd.DataFrame) -> list[str]:
    """The window columns of a :func:`rolling` result, in the windows' order."""
    return [c for c in df.columns if isinstance(c, str) and _WINDOW_COLUMN.fullmatch(c)]


@dataclass
class RollingConfig:
    """Configuration for :func:`rolling`."""

    target: str = "value"
    backend: str = "flaml"
    covariates: list[str] | None = None
    variables_resample: list[str] | None = None
    split_method: str = "random"
    train_fraction: float = 0.75
    model_config: dict[str, Any] | None = None
    n_samples: int = 300
    window_days: int = 14
    rolling_every: int = 7
    seed: int = DEFAULT_SEED
    n_cores: int | None = None
    memory_save: bool = False
    verbose: bool = False
    log_transform: bool = False
    back_transform: str = "arithmetic"
    smearing_factor: float | None = None
    stratify_by: str | Sequence[str] | None = None
    conditional_k: int | None = None
    conditional_on: Mapping[str, Any] | None = None
    resample_pools: Mapping[str, pd.DataFrame] | None = None
    sampling: str = "iid"
    block_hours: int | None = None
    cache: str | Path | None = None
    paired: bool | str = False
    include_summary: bool = False
    refit_per_window: bool = False
    min_pool_rows: int = 30


def _resolve_rolling_config(config: RollingConfig | None = None, **kwargs: Any) -> RollingConfig:
    return resolve_config(RollingConfig, config, **kwargs)


def rolling(
    df: pd.DataFrame | None = None,
    model: object | None = None,
    *,
    config: RollingConfig | None = None,
    target: str | None = None,
    backend: str | None = None,
    covariates: list[str] | None = None,
    variables_resample: list[str] | None = None,
    split_method: str | None = None,
    train_fraction: float | None = None,
    model_config: dict[str, Any] | None = None,
    n_samples: int | None = None,
    window_days: int | None = None,
    rolling_every: int | None = None,
    seed: int | None = None,
    n_cores: int | None = None,
    memory_save: bool | None = None,
    verbose: bool | None = None,
    log_transform: bool | None = None,
    back_transform: str | None = None,
    smearing_factor: float | None = None,
    stratify_by: str | Sequence[str] | None = None,
    conditional_k: int | None = None,
    conditional_on: Mapping[str, Any] | None = None,
    resample_pools: Mapping[str, pd.DataFrame] | None = None,
    sampling: str | None = None,
    block_hours: int | None = None,
    cache: str | Path | None = None,
    paired: bool | str | None = None,
    include_summary: bool | None = None,
    refit_per_window: bool | None = None,
    min_pool_rows: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Apply weather normalisation over a sliding window of the data.

    For each window of ``window_days`` days, stepped every ``rolling_every``
    days, the data in that window is normalised against *model* and the
    result is appended as a column of its own: ``window_01`` for the first window,
    ``window_02`` for the second, and so on (:func:`window_column`).

    Parameters
    ----------
    df : pandas.DataFrame or None, default=None
        Input dataset with ``date`` and target columns. Required.
    model : object or None, default=None
        Trained model. If ``None``, one is trained via :func:`build_model`
        using ``covariates``.
    config : RollingConfig or None, default=None
        Optional :class:`RollingConfig` holding all parameters.
        Individual keyword arguments override fields on *config* -- but only
        when they are given: a keyword left at ``None`` (every one of the
        function's own defaults) leaves the config's value, or the
        dataclass's default, in place.
    **kwargs
        Any additional keyword arguments are forwarded to
        :func:`_resolve_rolling_config`.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``date`` with column ``observed`` and one ``window_<i>``
        column of normalised values per window, counted from 1 and zero-padded
        (``window_01``, ``window_02``, ...); :func:`window_columns` lists them.
        When ``include_summary=True``, also includes ``rolling_mean``,
        ``rolling_std``, and ``n_windows``.
    """
    _cfg = _resolve_rolling_config(
        config=config or RollingConfig(),
        target=target,
        backend=backend,
        covariates=covariates,
        variables_resample=variables_resample,
        split_method=split_method,
        train_fraction=train_fraction,
        model_config=model_config,
        n_samples=n_samples,
        window_days=window_days,
        rolling_every=rolling_every,
        seed=seed,
        n_cores=n_cores,
        memory_save=memory_save,
        verbose=verbose,
        log_transform=log_transform,
        back_transform=back_transform,
        smearing_factor=smearing_factor,
        stratify_by=stratify_by,
        conditional_k=conditional_k,
        conditional_on=conditional_on,
        resample_pools=resample_pools,
        sampling=sampling,
        block_hours=block_hours,
        cache=cache,
        paired=paired,
        include_summary=include_summary,
        refit_per_window=refit_per_window,
        min_pool_rows=min_pool_rows,
        **kwargs,
    )

    if df is None:
        raise ValueError("`df` must be provided.")
    if _cfg.target is None:
        raise ValueError("`target` (target column name) must be provided.")

    if "date" not in df.columns:
        df = process_date(df)
    df = df[df["date"].notna()].sort_values("date").reset_index(drop=True)
    assert df is not None

    if _cfg.target not in df.columns:
        raise ValueError(f"`df` does not contain the target column '{_cfg.target}'.")
    df_work = df.copy()
    if _cfg.target != "value":
        df_work = df_work.rename(columns={_cfg.target: "value"})

    def _maybe_add_time_vars(frame: pd.DataFrame) -> pd.DataFrame:
        time_vars = {"date_unix", "day_julian", "weekday", "hour"}
        if _cfg.covariates is None:
            return frame
        need = [v for v in time_vars if v in _cfg.covariates and v not in frame.columns]
        if need:
            try:
                frame = add_date_variables(frame)
            except Exception:
                (log.info if _cfg.verbose else log.debug)(
                    "Missing time features not generated: %s", need
                )
        return frame

    df_work = _maybe_add_time_vars(df_work)
    result = df_work.set_index("date")[["value"]].rename(columns={"value": "observed"})

    if _cfg.log_transform:
        df_work = log1p_target(df_work, "value")

    if model is None:
        if not _cfg.covariates:
            raise ValueError("When `model` is None you must provide `covariates` for training.")
        df_work, model = build_model(
            df=df_work,
            target="value",
            backend=_cfg.backend,
            covariates=_cfg.covariates,
            split_method=_cfg.split_method,
            train_fraction=_cfg.train_fraction,
            model_config=_cfg.model_config,
            seed=_cfg.seed,
            verbose=_cfg.verbose,
            cache=_cfg.cache,
        )

    covariates_resolved = _cfg.covariates
    if covariates_resolved is None:
        try:
            covariates_resolved = extract_features(model)
        except Exception as exc:
            raise ValueError(
                "`covariates` must be provided, or the model must expose "
                "features via extract_features()."
            ) from exc

    variables_resample_resolved = _cfg.variables_resample
    if variables_resample_resolved is None:
        variables_resample_resolved = [f for f in covariates_resolved if not _is_time_feature(f)]

    n_cores_eff = resolve_workers(_cfg.n_cores)

    d_floor = df_work["date"].dt.floor("D")
    min_day = d_floor.min()
    max_day = d_floor.max()
    last_start = max_day - pd.Timedelta(days=_cfg.window_days - 1)
    if last_start < min_day:
        raise ValueError("Window is larger than the entire time span of `df`.")
    start_days = pd.date_range(min_day, last_start, freq=f"{_cfg.rolling_every}D")

    t0 = time.time()
    total = len(start_days)
    small_pools = 0

    for i, start_day in enumerate(start_days, start=1):
        progress.report(i - 1, total)
        end_excl = start_day + pd.Timedelta(days=_cfg.window_days)
        mask = (d_floor >= start_day) & (d_floor < end_excl)
        dfa = df_work.loc[mask]

        name = window_column(i, total)
        if len(dfa) < 2:
            (log.info if _cfg.verbose else log.debug)(
                "%s: %s skipped (not enough rows).",
                pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                name,
            )
            continue

        if len(dfa) < int(_cfg.min_pool_rows):
            small_pools += 1

        try:
            win_model = model
            if _cfg.refit_per_window and len(dfa) >= max(20, int(_cfg.min_pool_rows)):
                _, win_model = build_model(
                    df=dfa,
                    target="value",
                    backend=_cfg.backend,
                    covariates=covariates_resolved,
                    split_method=_cfg.split_method,
                    train_fraction=_cfg.train_fraction,
                    model_config=_cfg.model_config,
                    seed=_cfg.seed + (i * 997),
                    n_cores=n_cores_eff,
                    verbose=False,
                )

            with progress.part(i - 1, total):
                df_norm = normalise(
                    df=dfa,
                    model=win_model,
                    covariates=covariates_resolved,
                    variables_resample=variables_resample_resolved,
                    n_samples=_cfg.n_samples,
                    aggregate=True,
                    seed=_cfg.seed + (i * 997),
                    n_cores=n_cores_eff,
                    resample_df=dfa,
                    memory_save=_cfg.memory_save,
                    log_transform=_cfg.log_transform,
                    back_transform=_cfg.back_transform,
                    smearing_factor=_cfg.smearing_factor,
                    stratify_by=_cfg.stratify_by,
                    conditional_k=_cfg.conditional_k,
                    conditional_on=_cfg.conditional_on,
                    resample_pools=_cfg.resample_pools,
                    sampling=_cfg.sampling,
                    block_hours=_cfg.block_hours,
                    cache=_cfg.cache,
                    paired=_cfg.paired,
                )
            pick = "paired" if (_cfg.paired and "paired" in df_norm.columns) else "normalised"
            result[name] = df_norm[pick]
        except Exception as e:
            start_str = start_day.strftime("%Y-%m-%d")
            end_str = (end_excl - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            log.warning(
                "%s: error in %s [%s..%s]: %s",
                pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                name,
                start_str,
                end_str,
                e,
            )

        if i == 1 or i % 10 == 0 or i == total:
            s0 = start_day.strftime("%Y-%m-%d")
            s1 = (end_excl - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            (log.info if _cfg.verbose else log.debug)(
                "window %d/%d [%s..%s] %s",
                i,
                total,
                s0,
                s1,
                _progress_str(i, total, t0),
            )

    if small_pools:
        log.warning(
            "%d/%d rolling window(s) had fewer than min_pool_rows=%d rows in their local resample pool.",
            small_pools,
            total,
            int(_cfg.min_pool_rows),
        )

    result = result.sort_index()
    if _cfg.include_summary:
        win_cols = window_columns(result)
        if win_cols:
            vals_df = result[win_cols]
            result["rolling_mean"] = vals_df.mean(axis=1, skipna=True)
            result["rolling_std"] = vals_df.std(axis=1, skipna=True)
            result["n_windows"] = vals_df.notna().sum(axis=1).astype(int)
        else:
            result["rolling_mean"] = float("nan")
            result["rolling_std"] = float("nan")
            result["n_windows"] = 0

    return result
