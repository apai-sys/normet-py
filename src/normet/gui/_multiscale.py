# src/normet/gui/_multiscale.py
"""Multi-scale meteorological-influence decomposition via differencing
between rolling-deweathered series at different window widths.

Scientific rationale
---------------------
``Y_W(t)``, the window-``W`` rolling-deweathered value (with *local* —
within-window — resampling, exactly what :func:`normet.rolling` computes),
forms a scale-space family indexed by ``W``. As ``W`` grows, the resample
pool becomes progressively closer to the full-record climatological
distribution, so ``Y_W -> Y_inf`` as ``W`` approaches the full record —
``Y_inf`` is exactly :func:`normet.normalise`'s ordinary (unwindowed) result,
since it resamples from the whole record by default.

Differences between adjacent scales isolate the meteorological-influence
component specific to that timescale band, directly analogous to
wavelet/multi-resolution "detail coefficients", or to how MSTL separates
trend from seasonal from remainder via LOESS at different bandwidths::

    D_fast(t) = Y_fast(t) - Y_meso(t)   synoptic / sub-monthly residual
    D_meso(t) = Y_meso(t) - Y_slow(t)   intra-seasonal residual
    D_slow(t) = Y_slow(t) - Y_inf(t)    residual non-stationarity even a
                                        full slow-window's local pool
                                        doesn't average out

with the defaults ``fast=14 d``, ``meso=90 d``, ``slow=365 d``.
"""

from __future__ import annotations

import pandas as pd

from ..utils._config import DEFAULT_SEED
from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["rolling_mean_series", "compute_multiscale"]

#: A scale needs at least this many overlapping windows to be "rolling" at
#: all — with exactly one window, Y_W degenerates to Y_inf (the window
#: *is* the whole record), so the difference against it would be pure noise.
MIN_WINDOWS = 2


def rolling_mean_series(
    *,
    df_prep: pd.DataFrame,
    model: object,
    covariates: list[str],
    variables_resample: list[str] | None,
    window_days: int,
    n_samples: int,
    n_cores: int | None,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.Series | None, int, str | None]:
    """``Y_W(t)``: the row-wise mean of :func:`normet.rolling`'s overlapping
    windows at width *window_days*.

    Returns
    -------
    (series, n_windows, reason)
        *series* is None (with *reason* explaining why) when the record is
        too short for even one window, or too short for :data:`MIN_WINDOWS`
        overlapping ones.
    """
    from normet import rolling

    span_days = (df_prep["date"].max() - df_prep["date"].min()).total_seconds() / 86400.0 + 1
    if span_days < window_days:
        return (
            None,
            0,
            f"the record spans {span_days:.0f} d, shorter than the {window_days} d window",
        )

    step = max(1, window_days // 4)
    res = rolling(
        df=df_prep,
        model=model,
        covariates=covariates,
        variables_resample=variables_resample,
        window_days=window_days,
        rolling_every=step,
        n_samples=n_samples,
        n_cores=n_cores,
        seed=seed,
    )
    cols = [c for c in res.columns if c.startswith("rolling_")]
    if len(cols) < MIN_WINDOWS:
        return (
            None,
            len(cols),
            f"only {len(cols)} window(s) fit in the record — need at least "
            f"{MIN_WINDOWS} for a meaningful {window_days} d scale "
            "(a single window degenerates to the full-record baseline)",
        )
    return res[cols].mean(axis=1, skipna=True), len(cols), None


def compute_multiscale(
    *,
    df_prep: pd.DataFrame,
    model: object,
    covariates: list[str],
    variables_resample: list[str] | None,
    y_inf: pd.Series,
    fast_days: int = 14,
    meso_days: int = 90,
    slow_days: int = 365,
    n_samples: int = 100,
    n_cores: int | None = None,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Compute the fast/meso/slow scale-space family and their differences.

    Parameters
    ----------
    y_inf : pandas.Series
        ``Y_inf(t)`` — the ordinary (unwindowed) normalisation result,
        i.e. ``normalise(df_prep, model, ...)["normalised"]`` (Step 2's
        output, reused rather than recomputed).

    Returns
    -------
    dict
        ``Y_fast``, ``Y_meso``, ``Y_slow``, ``Y_inf`` (each a Series or
        None), ``D_fast``, ``D_meso``, ``D_slow`` (present only where both
        operands were available), ``{fast,meso,slow}_days``,
        ``n_{fast,meso,slow}`` (window counts), and ``notes`` (list of str
        explaining any skipped scale).
    """
    scales = {}
    notes: list[str] = []
    for key, days in (("fast", fast_days), ("meso", meso_days), ("slow", slow_days)):
        series, n_windows, reason = rolling_mean_series(
            df_prep=df_prep,
            model=model,
            covariates=covariates,
            variables_resample=variables_resample,
            window_days=days,
            n_samples=n_samples,
            n_cores=n_cores,
            seed=seed + hash(key) % 1000,
        )
        scales[key] = series
        if reason:
            notes.append(f"{days} d scale skipped: {reason}")
        log.info(
            "Multi-scale: Y_%s (%d d) -> %s",
            key,
            days,
            f"{n_windows} windows" if series is not None else f"skipped ({reason})",
        )

    out: dict = {
        "fast_days": fast_days,
        "meso_days": meso_days,
        "slow_days": slow_days,
        "Y_fast": scales["fast"],
        "Y_meso": scales["meso"],
        "Y_slow": scales["slow"],
        "Y_inf": y_inf,
        "notes": notes,
    }

    def _diff(a: pd.Series | None, b: pd.Series | None) -> pd.Series | None:
        if a is None or b is None:
            return None
        idx = a.index.intersection(b.index)
        d = (a.reindex(idx) - b.reindex(idx)).dropna()
        return d if len(d) else None

    out["D_fast"] = _diff(scales["fast"], scales["meso"])
    out["D_meso"] = _diff(scales["meso"], scales["slow"])
    out["D_slow"] = _diff(scales["slow"], y_inf)
    return out
