# src/normet/analysis/rolling.py
"""Rolling-window weather normalisation for trend analysis.

Provides :func:`rolling`, which normalises a sliding window of the data
against a model and returns one normalised series per window, plus
:class:`RollingConfig`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..model.train import build_model
from ..utils._config import DEFAULT_SEED, resolve_config
from ..utils.features import extract_features
from ..utils.logging import _progress_str, get_logger
from ..utils.prepare import add_date_variables, process_date
from .normalise import normalise

log = get_logger(__name__)


@dataclass
class RollingConfig:
    """Configuration for :func:`rolling`."""

    value: str = "value"
    backend: str = "flaml"
    feature_names: list[str] | None = None
    variables_resample: list[str] | None = None
    split_method: str = "random"
    fraction: float = 0.75
    model_config: dict[str, Any] | None = None
    n_samples: int = 300
    window_days: int = 14
    rolling_every: int = 7
    seed: int = DEFAULT_SEED
    n_cores: int | None = None
    memory_save: bool = False
    use_gpu: bool = False
    verbose: bool = False


def _resolve_rolling_config(config: RollingConfig | None = None, **kwargs: Any) -> RollingConfig:
    return resolve_config(RollingConfig, config, **kwargs)


def rolling(
    df: pd.DataFrame | None = None,
    model: object | None = None,
    *,
    config: RollingConfig | None = None,
    value: str = "value",
    backend: str = "flaml",
    feature_names: list[str] | None = None,
    variables_resample: list[str] | None = None,
    split_method: str = "random",
    fraction: float = 0.75,
    model_config: dict[str, Any] | None = None,
    n_samples: int = 300,
    window_days: int = 14,
    rolling_every: int = 7,
    seed: int = DEFAULT_SEED,
    n_cores: int | None = None,
    memory_save: bool = False,
    use_gpu: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Apply weather normalisation over a sliding window of the data.

    For each window of ``window_days`` days, stepped every ``rolling_every``
    days, the data in that window is normalised against *model* and the
    result is appended as a ``rolling_<i>`` column.

    Parameters
    ----------
    df : pandas.DataFrame or None, default=None
        Input dataset with ``date`` and target columns. Required.
    model : object or None, default=None
        Trained model. If ``None``, one is trained via :func:`build_model`
        using ``feature_names``.
    config : RollingConfig or None, default=None
        Optional :class:`RollingConfig` holding all parameters.
        Individual keyword arguments override fields on *config*.
    **kwargs
        Any additional keyword arguments are forwarded to
        :func:`_resolve_rolling_config`.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``date`` with column ``observed`` and one
        ``rolling_<i>`` column of normalised values per window.
    """
    _cfg = _resolve_rolling_config(
        config=config,
        value=value,
        backend=backend,
        feature_names=feature_names,
        variables_resample=variables_resample,
        split_method=split_method,
        fraction=fraction,
        model_config=model_config,
        n_samples=n_samples,
        window_days=window_days,
        rolling_every=rolling_every,
        seed=seed,
        n_cores=n_cores,
        memory_save=memory_save,
        use_gpu=use_gpu,
        verbose=verbose,
        **kwargs,
    )

    if df is None:
        raise ValueError("`df` must be provided.")
    if _cfg.value is None:
        raise ValueError("`value` (target column name) must be provided.")

    if "date" not in df.columns:
        df = process_date(df)
    df = df[df["date"].notna()].sort_values("date").reset_index(drop=True)
    assert df is not None  # narrowing: the pandas chain above always yields a DataFrame

    if _cfg.value not in df.columns:
        raise ValueError(f"`df` does not contain the target column '{_cfg.value}'.")
    df_work = df.copy()
    if _cfg.value != "value":
        df_work = df_work.rename(columns={_cfg.value: "value"})

    def _maybe_add_time_vars(frame: pd.DataFrame) -> pd.DataFrame:
        time_vars = {"date_unix", "day_julian", "weekday", "hour"}
        if _cfg.feature_names is None:
            return frame
        need = [v for v in time_vars if v in _cfg.feature_names and v not in frame.columns]
        if need:
            try:
                frame = add_date_variables(frame)
            except Exception:
                (log.info if _cfg.verbose else log.debug)(
                    "Missing time features not generated: %s", need
                )
        return frame

    df_work = _maybe_add_time_vars(df_work)

    if model is None:
        if not _cfg.feature_names:
            raise ValueError("When `model` is None you must provide `feature_names` for training.")
        df_work, model = build_model(
            df=df_work,
            value="value",
            backend=_cfg.backend,
            feature_names=_cfg.feature_names,
            split_method=_cfg.split_method,
            fraction=_cfg.fraction,
            model_config=_cfg.model_config,
            seed=_cfg.seed,
            verbose=_cfg.verbose,
        )

    feature_names_resolved = _cfg.feature_names
    if feature_names_resolved is None:
        try:
            feature_names_resolved = extract_features(model)
        except Exception as exc:
            raise ValueError(
                "`feature_names` must be provided, or the model must expose "
                "features via extract_features()."
            ) from exc

    variables_resample_resolved = _cfg.variables_resample
    if variables_resample_resolved is None:
        time_vars = {"date_unix", "day_julian", "weekday", "hour"}
        variables_resample_resolved = [f for f in feature_names_resolved if f not in time_vars]

    n_cores_eff = max(1, _cfg.n_cores if _cfg.n_cores is not None else (os.cpu_count() or 2) - 1)

    d_floor = df_work["date"].dt.floor("D")
    min_day = d_floor.min()
    max_day = d_floor.max()
    last_start = max_day - pd.Timedelta(days=_cfg.window_days - 1)
    if last_start < min_day:
        raise ValueError("Window is larger than the entire time span of `df`.")
    start_days = pd.date_range(min_day, last_start, freq=f"{_cfg.rolling_every}D")

    result = df_work.set_index("date")[["value"]].rename(columns={"value": "observed"})

    t0 = time.time()
    total = len(start_days)

    for i, start_day in enumerate(start_days, start=1):
        end_excl = start_day + pd.Timedelta(days=_cfg.window_days)
        mask = (d_floor >= start_day) & (d_floor < end_excl)
        dfa = df_work.loc[mask]

        if len(dfa) < 2:
            (log.info if _cfg.verbose else log.debug)(
                "%s: window %d skipped (not enough rows).",
                pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                i - 1,
            )
            continue

        try:
            df_norm = normalise(
                df=dfa,
                model=model,
                feature_names=feature_names_resolved,
                variables_resample=variables_resample_resolved,
                n_samples=_cfg.n_samples,
                aggregate=True,
                seed=_cfg.seed + (i * 997),
                n_cores=n_cores_eff,
                resample_df=None,
                memory_save=_cfg.memory_save,
                use_gpu=_cfg.use_gpu,
            )
            result[f"rolling_{i - 1}"] = df_norm["normalised"]
        except Exception as e:
            start_str = start_day.strftime("%Y-%m-%d")
            end_str = (end_excl - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            log.warning(
                "%s: error in window %d [%s..%s]: %s",
                pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                i - 1,
                start_str,
                end_str,
                e,
            )

        if i == 1 or i % 10 == 0 or i == total:
            s0 = start_day.strftime("%Y-%m-%d")
            s1 = (end_excl - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            (log.info if _cfg.verbose else log.debug)(
                "window %d/%d [%s..%s] %s",
                i - 1,
                total - 1,
                s0,
                s1,
                _progress_str(i, total, t0),
            )

    return result.sort_index()
