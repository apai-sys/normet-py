# src/normet/utils/featureeng.py
"""
Feature-engineering helpers for environmental time-series.

These utilities operate on tidy long DataFrames with a ``date`` column
(or DatetimeIndex). They never modify the input in place.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ._lazy import require
from .logging import get_logger

log = get_logger(__name__)

__all__ = [
    "DEFAULT_MET_BOUNDS",
    "add_lag_features",
    "add_rolling_features",
    "add_met_physics_features",
    "analyze_lag",
    "cyclical_encode",
    "screen_aq_data",
    "wind_to_uv",
    "LagDiagnostics",
]

DEFAULT_MET_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "wd": (0.0, 360.0),
    "ws": (0.0, 100.0),
    "rh": (0.0, 100.0),
    "blh": (0.0, 10_000.0),
}


def _ensure_sorted_by_date(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Return a copy sorted by date_col; require the column to be present and datetime-like."""
    if date_col not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": date_col})
        else:
            raise ValueError(f"`{date_col}` column not found and index is not DatetimeIndex.")
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise ValueError(f"`{date_col}` must be datetime-like.")
    return df.sort_values(date_col).reset_index(drop=True)


def add_lag_features(
    df: pd.DataFrame,
    cols: Sequence[str],
    lags: Sequence[int],
    *,
    group_col: str | None = None,
    date_col: str = "date",
    suffix: str = "_lag",
    freq: str | None = None,
) -> pd.DataFrame:
    """
    Add lagged copies of ``cols`` shifted by each value in ``lags``.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format dataset with a datetime column.
    cols : sequence of str
        Columns to lag.
    lags : sequence of int
        Positive integers; ``lag=1`` looks one row (or one ``freq`` step when
        ``freq`` is set) back.
    group_col : str, optional
        If provided, lags are computed within each group (e.g., per station).
    date_col : str, default "date"
        Datetime column used to sort before lagging.
    suffix : str, default "_lag"
        New columns are named ``f"{col}{suffix}{k}"``.
    freq : str, optional
        When set (e.g. ``"h"`` or ``"D"``), shifts along a uniform calendar grid
        ``pd.date_range(min, max, freq=freq)`` instead of physical row positions
        so missing timestamps emit ``NaN`` rather than shifting across gaps.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` (sorted by date) with new lag columns appended.
    """
    if not cols or not lags:
        return df.copy()
    bad = [c for c in cols if c not in df.columns]
    if bad:
        raise ValueError(f"Columns not in df: {bad}")
    if any(int(k) <= 0 for k in lags):
        raise ValueError("`lags` must contain positive integers.")
    fractional = [k for k in lags if float(k) != int(k)]
    if fractional:
        # int(1.9) == 1: a fractional lag used to be silently truncated to a
        # different shift than the caller asked for.
        raise ValueError(f"`lags` must be whole numbers of rows; got {fractional}.")

    out = _ensure_sorted_by_date(df, date_col=date_col)
    if freq is None:
        grouped = out.groupby(group_col, sort=False) if group_col else None
        for c in cols:
            for k in lags:
                name = f"{c}{suffix}{int(k)}"
                out[name] = (
                    grouped[c].shift(int(k)) if grouped is not None else out[c].shift(int(k))
                )
        return out

    col_list = list(dict.fromkeys(cols))
    slices = [sub for _, sub in out.groupby(group_col, sort=False)] if group_col else [out]
    for k in lags:
        for c in col_list:
            out[f"{c}{suffix}{int(k)}"] = np.nan

    for sub in slices:
        if sub.empty:
            continue
        if sub[date_col].duplicated().any():
            raise ValueError(
                f"Duplicate timestamps in `{date_col}` cannot be uniquely aligned to `freq={freq!r}`."
            )
        orig_dates = pd.Index(sub[date_col])
        full_idx = pd.date_range(orig_dates.min(), orig_dates.max(), freq=freq)
        grid = sub.set_index(date_col)[col_list].reindex(full_idx)
        for c in col_list:
            for k in lags:
                vals = grid[c].shift(int(k)).reindex(orig_dates).to_numpy()
                out.loc[sub.index, f"{c}{suffix}{int(k)}"] = vals
    return out


def add_rolling_features(
    df: pd.DataFrame,
    cols: Sequence[str],
    windows: Sequence[int],
    *,
    aggs: Sequence[str] = ("mean",),
    min_periods: int | None = None,
    group_col: str | None = None,
    date_col: str = "date",
    suffix: str = "_roll",
    causal: bool = True,
    freq: str | None = None,
) -> pd.DataFrame:
    """
    Add rolling-window statistics for ``cols``.

    Parameters
    ----------
    df : pandas.DataFrame
    cols : sequence of str
        Columns to roll over.
    windows : sequence of int
        Window sizes in rows (or ``freq`` steps when ``freq`` is set).
    aggs : sequence of {"mean","std","min","max","median","sum"}, default ("mean",)
    min_periods : int, optional
        Forwarded to pandas rolling; defaults to the window size.
    group_col : str, optional
        Roll within each group (e.g., per station) if provided.
    date_col : str, default "date"
        Datetime column used to sort before rolling.
    suffix : str, default "_roll"
        New columns are named ``f"{col}{suffix}{w}_{agg}"``.
    causal : bool, default True
        If True, use trailing windows (no look-ahead). If False, the window is
        centered (``center=True``) — only appropriate for diagnostics, not
        for predictors used in forecasting tasks.
    freq : str, optional
        When set (e.g. ``"h"``), reindexes each series onto a uniform calendar
        grid before rolling so missing intervals are preserved as ``NaN``.

    Returns
    -------
    pandas.DataFrame
        Copy with rolling-statistics columns appended.
    """
    if not cols or not windows:
        return df.copy()

    allowed = {"mean", "std", "min", "max", "median", "sum"}
    bad_aggs = [a for a in aggs if a not in allowed]
    if bad_aggs:
        raise ValueError(f"Unsupported aggs: {bad_aggs}. Choose from {sorted(allowed)}.")
    bad_cols = [c for c in cols if c not in df.columns]
    if bad_cols:
        raise ValueError(f"Columns not in df: {bad_cols}")
    if any(int(w) <= 0 for w in windows):
        raise ValueError("`windows` must contain positive integers.")

    out = _ensure_sorted_by_date(df, date_col=date_col)
    if freq is None:
        grouped = out.groupby(group_col, sort=False) if group_col else None
        for c in cols:
            for w in windows:
                w_int = int(w)
                mp = w_int if min_periods is None else int(min_periods)
                if grouped is not None:
                    r = grouped[c].rolling(window=w_int, min_periods=mp, center=not causal)
                else:
                    r = out[c].rolling(window=w_int, min_periods=mp, center=not causal)  # type: ignore[assignment]
                for agg in aggs:
                    name = f"{c}{suffix}{w_int}_{agg}"
                    vals = getattr(r, agg)()
                    # When grouped, rolling returns a multi-index Series; align back.
                    if grouped is not None:
                        vals = vals.reset_index(level=0, drop=True).sort_index()
                    out[name] = vals.to_numpy() if hasattr(vals, "to_numpy") else np.asarray(vals)
        return out

    col_list = list(dict.fromkeys(cols))
    slices = [sub for _, sub in out.groupby(group_col, sort=False)] if group_col else [out]
    for c in col_list:
        for w in windows:
            for agg in aggs:
                out[f"{c}{suffix}{int(w)}_{agg}"] = np.nan

    for sub in slices:
        if sub.empty:
            continue
        if sub[date_col].duplicated().any():
            raise ValueError(
                f"Duplicate timestamps in `{date_col}` cannot be uniquely aligned to `freq={freq!r}`."
            )
        orig_dates = pd.Index(sub[date_col])
        full_idx = pd.date_range(orig_dates.min(), orig_dates.max(), freq=freq)
        grid = sub.set_index(date_col)[col_list].reindex(full_idx)
        for c in col_list:
            for w in windows:
                w_int = int(w)
                mp = w_int if min_periods is None else int(min_periods)
                roll = grid[c].rolling(window=w_int, min_periods=mp, center=not causal)
                for agg in aggs:
                    name = f"{c}{suffix}{w_int}_{agg}"
                    vals = getattr(roll, agg)().reindex(orig_dates).to_numpy()
                    out.loc[sub.index, name] = vals
    return out


def cyclical_encode(
    df: pd.DataFrame,
    col: str,
    period: float,
    *,
    drop: bool = False,
    prefix: str | None = None,
) -> pd.DataFrame:
    """
    Replace a periodic feature with its sine/cosine encoding.

    Useful for hour-of-day (period=24), day-of-week (period=7),
    day-of-year (period=365.25), month (period=12), wind direction (period=360).

    Parameters
    ----------
    df : pandas.DataFrame
    col : str
        Column to encode. Must be numeric and roughly in ``[0, period)``.
    period : float
        Cycle length (e.g., 24 for hour).
    drop : bool, default False
        If True, drop the original column.
    prefix : str, optional
        Prefix for the new columns; defaults to ``col``. Produces
        ``f"{prefix}_sin"`` and ``f"{prefix}_cos"``.

    Returns
    -------
    pandas.DataFrame
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not in df.")
    if period <= 0:
        raise ValueError("`period` must be positive.")

    out = df.copy()
    base = prefix or col
    theta = 2.0 * np.pi * pd.to_numeric(out[col], errors="coerce") / float(period)
    out[f"{base}_sin"] = np.sin(theta)
    out[f"{base}_cos"] = np.cos(theta)
    if drop:
        out = out.drop(columns=[col])
    return out


def _magnus_es(temp_c: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure (kPa) via the Magnus formula for T in Celsius."""
    denom = temp_c + 237.3
    safe_denom = np.where(denom > 1.0, denom, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        return 0.6108 * np.exp((17.27 * temp_c) / safe_denom)


def _causal_ema(
    values: np.ndarray, decay: np.ndarray | float, *, reset: np.ndarray | None = None
) -> np.ndarray:
    """First-order causal ODE exponential accumulation ``s[t] = d[t]*s[t-1] + (1-d[t])*x[t]``.

    ``reset`` marks the first row of each independent segment (a site, a series).
    Without it a panel is accumulated as one long record, so every row after the
    first site starts at the previous site's level and the feature depends on
    how the rows happen to be ordered.
    """
    n = len(values)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    d_arr = np.broadcast_to(np.asarray(decay, dtype=np.float64), (n,))
    r_arr = np.zeros(n, dtype=bool) if reset is None else np.asarray(reset, dtype=bool)
    acc = float("nan")
    for t in range(n):
        if t == 0 or r_arr[t]:
            acc = float(values[t])
        else:
            d_t = float(d_arr[t])
            acc = d_t * acc + (1.0 - d_t) * float(values[t])
        out[t] = acc
    return out


def add_met_physics_features(
    df: pd.DataFrame,
    *,
    ws_col: str | None = "ws",
    wd_col: str | None = "wd",
    blh_col: str | None = "blh",
    temp_col: str | None = "t2m",
    rh_col: str | None = "rh",
    dewpoint_col: str | None = "d2m",
    ssrd_col: str | None = "ssrd",
    precip_col: str | None = "tp",
    temp_unit: str = "auto",
    include_memory: bool = False,
    residence_tau_hours: float = 12.0,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Derive physically meaningful boundary-layer and thermodynamic meteorological features.

    Adds whichever columns can be computed from available inputs without mutating *df*:
      - ``u_wind``, ``v_wind`` from ``ws_col`` & ``wd_col`` via :func:`wind_to_uv`
      - ``ventilation_coef = ws * blh`` from ``ws_col`` & ``blh_col``
      - ``vpd`` (vapour pressure deficit, kPa) and ``dewpoint_depression`` (deg C),
        plus ``rh_derived`` (%) when ``dewpoint_col`` is present without ``rh_col``
      - when ``include_memory=True``, causal first-order ODE residence integrals:
        ``stagnation_memory``, ``photochemical_dose``, and ``wet_scavenging_memory``.

    The memory integrals are causal and stateful, so a panel of several sites must
    be accumulated per site: with ``group_col`` set, the state resets at each
    group and rows are ordered by group then date. Without it the whole frame is
    treated as one record, which mixes one site's air mass into the next site's
    features. Decay is applied per elapsed hour (``exp(-dt / tau)``), so hourly,
    daily and irregular records all correspond to the same physical kernel; when
    no datetime column or index is available, one row is taken as one hour.
    """
    if temp_unit not in ("auto", "C", "K"):
        raise ValueError(f"`temp_unit` must be 'auto', 'C', or 'K', got {temp_unit!r}.")
    if float(residence_tau_hours) <= 0.0:
        raise ValueError(f"`residence_tau_hours` must be positive, got {residence_tau_hours!r}.")

    out = df.copy()
    if ws_col and wd_col and ws_col in out.columns and wd_col in out.columns:
        u_arr, v_arr = wind_to_uv(
            pd.to_numeric(out[ws_col], errors="coerce"),
            pd.to_numeric(out[wd_col], errors="coerce"),
            convention="meteorological",
        )
        out["u_wind"] = u_arr
        out["v_wind"] = v_arr

    if ws_col and blh_col and ws_col in out.columns and blh_col in out.columns:
        out["ventilation_coef"] = pd.to_numeric(out[ws_col], errors="coerce") * pd.to_numeric(
            out[blh_col], errors="coerce"
        )

    if temp_col and temp_col in out.columns:
        t_raw = pd.to_numeric(out[temp_col], errors="coerce").to_numpy(dtype=float)
        fin_t = t_raw[np.isfinite(t_raw)]
        is_k = (temp_unit == "K") or (
            temp_unit == "auto" and fin_t.size > 0 and float(np.median(fin_t)) > 150.0
        )
        t_c = t_raw - 273.15 if is_k else t_raw
        es_t = _magnus_es(t_c)

        has_rh = bool(rh_col and rh_col in out.columns)
        has_td = bool(dewpoint_col and dewpoint_col in out.columns)

        if has_rh:
            assert rh_col is not None
            rh_vals = np.clip(
                pd.to_numeric(out[rh_col], errors="coerce").to_numpy(dtype=float), 0.0, 100.0
            )
            out["vpd"] = np.maximum(0.0, es_t * (1.0 - rh_vals / 100.0))
            if not has_td:
                rh_pos = np.clip(rh_vals, 1e-3, 100.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    gamma = np.log(rh_pos / 100.0) + (17.27 * t_c) / np.where(
                        t_c + 237.3 > 1.0, t_c + 237.3, np.nan
                    )
                    td_est = (237.3 * gamma) / np.where(
                        np.abs(17.27 - gamma) > 1e-6, 17.27 - gamma, np.nan
                    )
                out["dewpoint_depression"] = np.maximum(0.0, t_c - td_est)

        if has_td:
            assert dewpoint_col is not None
            td_raw = pd.to_numeric(out[dewpoint_col], errors="coerce").to_numpy(dtype=float)
            fin_td = td_raw[np.isfinite(td_raw)]
            is_td_k = (temp_unit == "K") or (
                temp_unit == "auto" and fin_td.size > 0 and float(np.median(fin_td)) > 150.0
            )
            td_c = td_raw - 273.15 if is_td_k else td_raw
            out["dewpoint_depression"] = t_c - td_c
            es_td = _magnus_es(td_c)
            if not has_rh:
                out["vpd"] = np.maximum(0.0, es_t - es_td)
                with np.errstate(divide="ignore", invalid="ignore"):
                    out["rh_derived"] = 100.0 * np.clip(es_td / np.maximum(es_t, 1e-12), 0.0, 1.0)

    if include_memory and len(out) > 0:
        n = len(out)
        date_idx: pd.DatetimeIndex | None = None
        if "date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["date"]):
            date_idx = pd.DatetimeIndex(out["date"])
        elif isinstance(out.index, pd.DatetimeIndex):
            date_idx = out.index

        # Integer sort keys for the datetime axis. `asi8` is not in the type
        # stubs; converting to UTC datetime64 then int64 gives the same order
        # for tz-aware and naive indexes alike.
        date_codes = (
            date_idx.to_numpy(dtype="datetime64[ns]").astype("int64")
            if date_idx is not None
            else None
        )

        reset_mask: np.ndarray | None = None
        if group_col is not None:
            if group_col not in out.columns:
                raise ValueError(f"`group_col` {group_col!r} is not a column of `df`.")
            groups_arr = out[group_col].to_numpy(dtype=object)
            group_codes = pd.factorize(groups_arr)[0]
            if date_codes is not None:
                ord_idx = np.lexsort((date_codes, group_codes))
            else:
                ord_idx = np.argsort(group_codes, kind="stable")
            groups_sorted = groups_arr[ord_idx]
            reset_mask = np.empty(n, dtype=bool)
            reset_mask[0] = True
            reset_mask[1:] = groups_sorted[1:] != groups_sorted[:-1]
        elif date_codes is not None:
            ord_idx = np.argsort(date_codes, kind="stable")
        else:
            ord_idx = np.arange(n, dtype=np.int64)
        inv_idx = np.empty(n, dtype=np.int64)
        inv_idx[ord_idx] = np.arange(n, dtype=np.int64)

        # Elapsed hours between consecutive rows within a segment. The kernel is
        # then exp(-dt / tau) at every row, so the physical memory is the same
        # whether the record is hourly, daily or irregular; one row per hour is
        # assumed only when no datetime axis exists. Duplicate timestamps get
        # dt = 0 (the state is carried forward, not advanced).
        dt_hours: np.ndarray = np.ones(n, dtype=np.float64)
        if date_idx is not None:
            ordered = pd.Series(date_idx[ord_idx])
            step = ordered.diff().dt.total_seconds().to_numpy(dtype=float) / 3600.0
            step[0] = 1.0
            if reset_mask is not None:
                step[reset_mask] = 1.0
            dt_hours = np.where(np.isfinite(step) & (step >= 0.0), step, 1.0)

        tau_base = float(residence_tau_hours)
        if blh_col and blh_col in out.columns:
            blh_arr = pd.to_numeric(out[blh_col], errors="coerce").to_numpy(dtype=float)[ord_idx]
            inv_blh = 1000.0 / np.maximum(np.where(np.isfinite(blh_arr), blh_arr, 500.0), 50.0)
            if ws_col and ws_col in out.columns:
                ws_arr = pd.to_numeric(out[ws_col], errors="coerce").to_numpy(dtype=float)[ord_idx]
                ws_clean = np.maximum(np.where(np.isfinite(ws_arr), ws_arr, 2.0), 0.0)
                tau_eff = tau_base / (1.0 + ws_clean / 3.0)
            else:
                tau_eff = np.full(n, tau_base, dtype=np.float64)
            decay_stag = np.exp(-dt_hours / np.maximum(tau_eff, 0.25))
            out["stagnation_memory"] = _causal_ema(inv_blh, decay_stag, reset=reset_mask)[inv_idx]

        if ssrd_col and ssrd_col in out.columns:
            rad_arr = pd.to_numeric(out[ssrd_col], errors="coerce").to_numpy(dtype=float)[ord_idx]
            rad_clean = np.maximum(np.where(np.isfinite(rad_arr), rad_arr, 0.0), 0.0)
            out["photochemical_dose"] = _causal_ema(
                rad_clean, np.exp(-dt_hours / 6.0), reset=reset_mask
            )[inv_idx]

        if precip_col and precip_col in out.columns:
            tp_arr = pd.to_numeric(out[precip_col], errors="coerce").to_numpy(dtype=float)[ord_idx]
            tp_clean = np.maximum(np.where(np.isfinite(tp_arr), tp_arr, 0.0), 0.0)
            decay_tp = np.exp(-dt_hours / max(tau_base, 0.5))
            out["wet_scavenging_memory"] = _causal_ema(tp_clean, decay_tp, reset=reset_mask)[
                inv_idx
            ]

    return out


def screen_aq_data(
    df: pd.DataFrame,
    target: str,
    *,
    date_col: str = "date",
    min_value: float | None = 0.0,
    max_value: float | None = None,
    negative_action: str = "nan",
    flatline_window: int | None = 6,
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    iqr_multiplier: float | None = None,
    group_col: str | None = None,
    return_flags: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Screen air-quality observations and meteorology for QA/QC anomalies.

    Detects values below ``min_value`` (negatives / below zero), above
    ``max_value``, constant sensor plateaus (flatlines of length >=
    ``flatline_window``), robust Tukey IQR outliers, and out-of-range
    meteorological bounds.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format observations. May hold several sites; see ``group_col``.
    target : str
        Observation column to screen and clean.
    date_col : str, default "date"
        Datetime column establishing temporal order. A DatetimeIndex is used
        when the column is absent.
    min_value, max_value : float or None
        Fixed validity bounds for ``target``. Values below ``min_value``
        (or above ``max_value``) are flagged; ``None`` disables that check.
    negative_action : {"nan", "clip", "drop", "keep"}, default "nan"
        What to do with below-``min_value`` values in the cleaned frame.
    flatline_window : int or None, default 6
        Minimum run length (in consecutive rows of the same group) for a
        constant value to count as a stuck sensor. ``None`` disables.
    bounds : mapping, optional
        Extra columns mapped to ``(low, high)`` plausibility bounds (e.g.
        ``DEFAULT_MET_BOUNDS``); out-of-range cells become NaN and are
        flagged. Bounds are absolute, so they are unaffected by grouping.
    iqr_multiplier : float or None, default None
        Tukey fence multiplier: values outside
        ``Q1 - m*IQR .. Q3 + m*IQR`` are flagged. The quartiles are computed
        on the group's own distribution. ``None`` disables.
    group_col : str or None, default None
        Column identifying independent series (typically a station id).
        When set, temporal ordering, flatline runs and IQR quartiles are
        computed *within each group*, so a value from another site neither
        breaks a stuck-sensor run nor widens the IQR fence. When None the
        whole frame is treated as one series, as before. Without this,
        a constant station in a multi-site panel is never flagged because
        each of its rows is separated by another site in the global order.
    return_flags : bool, default False
        If True, return ``(cleaned_df, flags_df)``; otherwise just the
        cleaned frame.

    Returns
    -------
    pandas.DataFrame or tuple of pandas.DataFrame
        Cleaned frame, and optionally a per-row boolean flag frame with the
        same index as ``df``.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")
    if group_col is not None and group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in DataFrame.")
    if negative_action not in ("nan", "clip", "drop", "keep"):
        raise ValueError(
            f"`negative_action` must be 'nan', 'clip', 'drop', or 'keep', got {negative_action!r}."
        )
    if flatline_window is not None and (
        isinstance(flatline_window, bool) or int(flatline_window) < 2
    ):
        raise ValueError(
            f"`flatline_window` must be an integer >= 2 or None, got {flatline_window!r}."
        )
    if iqr_multiplier is not None and float(iqr_multiplier) <= 0.0:
        raise ValueError(f"`iqr_multiplier` must be positive or None, got {iqr_multiplier!r}.")

    out = df.copy()
    n = len(out)
    orig_index = out.index

    if n == 0:
        flags_empty = pd.DataFrame(
            columns=[
                "flag_negative",
                "flag_above_max",
                "flag_flatline",
                "flag_iqr_outlier",
                "flag_met_bounds",
                "flag_any",
            ],
            index=orig_index,
        )
        return (out, flags_empty) if return_flags else out

    if date_col in out.columns and pd.api.types.is_datetime64_any_dtype(out[date_col]):
        date_values: np.ndarray | None = out[date_col].to_numpy()
    elif isinstance(out.index, pd.DatetimeIndex):
        date_values = out.index.to_numpy()
    else:
        date_values = None

    # ``flatnonzero`` (intp) and ``arange(dtype=int64)`` produce different
    # numpy shape/dtype spellings; annotate the list so the two branches
    # agree under numpy 2.2's stricter shape inference.
    group_positions: list[np.ndarray[Any, Any]]
    if group_col is None:
        group_positions = [np.arange(n, dtype=np.int64)]
    else:
        # Factorize with a sentinel so rows with a missing group still form
        # one group; factorize's -1 code would otherwise be skipped and
        # those rows would never be screened.
        groups = out[group_col].astype(object)
        missing_group = groups.isna().to_numpy()
        if missing_group.any():
            groups = groups.copy()
            groups.loc[missing_group] = "__normet_missing_group__"
        codes = pd.factorize(groups, sort=False)[0]
        group_positions = [np.flatnonzero(codes == code) for code in range(int(codes.max()) + 1)]

    def _ordered_positions(positions: np.ndarray) -> np.ndarray:
        if date_values is None:
            return positions
        return positions[np.argsort(date_values[positions], kind="stable")]

    vals = pd.to_numeric(out[target], errors="coerce").to_numpy(dtype=float)
    fin = np.isfinite(vals)
    flag_neg = (vals < float(min_value)) & fin if min_value is not None else np.zeros(n, dtype=bool)
    flag_max = (vals > float(max_value)) & fin if max_value is not None else np.zeros(n, dtype=bool)

    flag_flat = np.zeros(n, dtype=bool)
    if flatline_window is not None:
        w_min = int(flatline_window)
        for positions in group_positions:
            if positions.size < w_min:
                continue
            pos_ord = _ordered_positions(positions)
            v_ord = vals[pos_ord]
            f_ord = np.isfinite(v_ord)
            n_ord = pos_ord.size
            flat_ord = np.zeros(n_ord, dtype=bool)
            i = 0
            while i < n_ord:
                if not f_ord[i]:
                    i += 1
                    continue
                j = i + 1
                while j < n_ord and f_ord[j] and abs(v_ord[j] - v_ord[i]) <= 1e-12:
                    j += 1
                if j - i >= w_min:
                    flat_ord[i:j] = True
                i = j
            flag_flat[pos_ord] = flat_ord

    flag_iqr = np.zeros(n, dtype=bool)
    if iqr_multiplier is not None:
        m = float(iqr_multiplier)
        for positions in group_positions:
            base_valid = positions[fin[positions] & (~flag_neg[positions]) & (~flag_max[positions])]
            v_base = vals[base_valid]
            if v_base.size < 4:
                continue
            q1, q3 = np.nanpercentile(v_base, [25.0, 75.0])
            iqr = float(q3 - q1)
            if iqr > 0.0:
                lo_t, hi_t = float(q1) - m * iqr, float(q3) + m * iqr
                flag_iqr[positions] = fin[positions] & (
                    (vals[positions] < lo_t) | (vals[positions] > hi_t)
                )

    flag_bounds = np.zeros(n, dtype=bool)
    if bounds:
        for col, (lo_b, hi_b) in bounds.items():
            if col not in out.columns:
                continue
            c_vals = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
            c_fin = np.isfinite(c_vals)
            bad = np.zeros(n, dtype=bool)
            if lo_b is not None:
                bad |= c_fin & (c_vals < float(lo_b))
            if hi_b is not None:
                bad |= c_fin & (c_vals > float(hi_b))
            if np.any(bad):
                c_vals_clean = c_vals.copy()
                c_vals_clean[bad] = np.nan
                out[col] = c_vals_clean
                flag_bounds |= bad

    # Update target column according to flags
    t_clean = vals.copy()
    bad_target_other = flag_max | flag_flat | flag_iqr
    t_clean[bad_target_other] = np.nan
    if negative_action == "nan":
        t_clean[flag_neg] = np.nan
    elif negative_action == "clip" and min_value is not None:
        t_clean[flag_neg] = float(min_value)
    out[target] = t_clean

    if negative_action == "drop" and np.any(flag_neg):
        out = out.loc[~flag_neg].copy()

    flag_any = flag_neg | flag_max | flag_flat | flag_iqr | flag_bounds
    flags_df = pd.DataFrame(
        {
            "flag_negative": flag_neg,
            "flag_above_max": flag_max,
            "flag_flatline": flag_flat,
            "flag_iqr_outlier": flag_iqr,
            "flag_met_bounds": flag_bounds,
            "flag_any": flag_any,
        },
        index=orig_index,
    )
    return (out, flags_df) if return_flags else out


def wind_to_uv(
    speed: pd.Series | np.ndarray | Iterable[float],
    direction_deg: pd.Series | np.ndarray | Iterable[float],
    *,
    convention: str = "meteorological",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose wind speed + direction into orthogonal components (u, v).

    Parameters
    ----------
    speed : array-like
        Wind speed (any consistent unit; output uses the same unit).
    direction_deg : array-like
        Wind direction in degrees.
    convention : {"meteorological","oceanographic"}, default "meteorological"
        - "meteorological": direction is the angle the wind is *from*,
          measured clockwise from North. u is eastward, v is northward.
        - "oceanographic": direction is the angle the wind is *toward*.

    Returns
    -------
    (u, v) : tuple of numpy.ndarray
        Zonal (east-west) and meridional (north-south) components.
    """
    s = np.asarray(speed, dtype=float)
    d = np.asarray(direction_deg, dtype=float)
    if s.shape != d.shape:
        raise ValueError(f"`speed` and `direction_deg` shape mismatch: {s.shape} vs {d.shape}")

    rad = np.deg2rad(d)
    if convention == "meteorological":
        u = -s * np.sin(rad)
        v = -s * np.cos(rad)
    elif convention == "oceanographic":
        u = s * np.sin(rad)
        v = s * np.cos(rad)
    else:
        raise ValueError("`convention` must be 'meteorological' or 'oceanographic'.")
    return u, v


# ---------------------------------------------------------------------------
# Lag-structure diagnostics (ACF / PACF / pre-whitened CCF)
# ---------------------------------------------------------------------------


@dataclass
class LagDiagnostics:
    """Result of :func:`analyze_lag`.

    Attributes
    ----------
    target : str
        Name of the response series (e.g. the pollutant).
    driver : str or None
        Name of the driver series (e.g. a meteorological variable). ``None``
        when only the target's own lag structure was requested.
    n : int
        Effective number of observations used for the significance bands.
    alpha : float
        Significance level used for the two-sided bands.
    band : float
        White-noise significance threshold ``z_{1-alpha/2} / sqrt(n)``. A
        coefficient is flagged "significant" when ``abs(value) > band``.
    acf, pacf : pandas.DataFrame
        Columns ``lag`` (>= 0) and ``value`` for the target's autocorrelation
        and partial autocorrelation.
    ccf : pandas.DataFrame or None
        Columns ``lag`` and ``value`` for the cross-correlation between driver
        and target. ``lag = k > 0`` means the driver *leads* the target by
        ``k`` rows, i.e. ``corr(driver.shift(k), target)`` — the orientation
        you want for predictive lag features. ``None`` when ``driver`` is
        ``None``.
    target_ar_lags : list of int
        Significant PACF lags (>= 1) — suggested autoregressive lags of the
        target itself.
    driver_lags : list of int
        Significant CCF lags (>= 0) where the driver leads the target —
        suggested lags to feed to :func:`add_lag_features`. The bands are
        per-lag thresholds, so with ``2 * max_lag + 1`` lags tested a few
        exceed the band by chance alone (about 3 of 50 unrelated white
        pairs at ``alpha=0.05``): treat the list as candidates, and prefer
        a lag that is also the peak or repeats at other sites/periods.
    peak_lag : int or None
        Driver-leading lag (>= 0) with the largest absolute CCF.
    prewhitened : bool
        Whether the CCF was computed on pre-whitened series.
    """

    target: str
    driver: str | None
    n: int
    alpha: float
    band: float
    acf: pd.DataFrame
    pacf: pd.DataFrame
    ccf: pd.DataFrame | None = None
    target_ar_lags: list[int] = field(default_factory=list)
    driver_lags: list[int] = field(default_factory=list)
    peak_lag: int | None = None
    prewhitened: bool = False

    def summary(self) -> str:
        """Return a short human-readable summary of the recommended lags."""
        lines = [
            f"Lag diagnostics for target='{self.target}'"
            + (f", driver='{self.driver}'" if self.driver else ""),
            f"  n={self.n}, significance band=±{self.band:.3f} (alpha={self.alpha})",
            f"  suggested target AR lags (PACF): {self.target_ar_lags or '—'}",
        ]
        if self.driver is not None:
            tag = "pre-whitened" if self.prewhitened else "raw"
            lines.append(f"  suggested driver lags (CCF, {tag}): {self.driver_lags or '—'}")
            lines.append(f"  peak driver-leading lag: {self.peak_lag}")
        return "\n".join(lines)

    def plot(self, ax: Any = None) -> Any:  # pragma: no cover - visual helper
        """Stem-plot the ACF, PACF and (if present) CCF with significance bands.

        Requires matplotlib. Returns the array of axes.
        """
        plt = require("matplotlib.pyplot", hint="pip install matplotlib")
        npanels = 3 if self.ccf is not None else 2
        if ax is None:
            _, axes = plt.subplots(npanels, 1, figsize=(8, 2.4 * npanels), sharex=False)
        else:
            axes = np.atleast_1d(ax)
        axes = np.atleast_1d(axes)

        def _stem(a: Any, frame: pd.DataFrame, title: str) -> None:
            a.stem(frame["lag"].to_numpy(), frame["value"].to_numpy(), basefmt=" ")
            a.axhline(0.0, color="0.5", lw=0.8)
            a.axhline(self.band, color="crimson", ls="--", lw=0.8)
            a.axhline(-self.band, color="crimson", ls="--", lw=0.8)
            a.set_title(title)
            a.set_ylabel("corr")

        _stem(axes[0], self.acf, f"ACF — {self.target}")
        _stem(axes[1], self.pacf, f"PACF — {self.target}")
        if self.ccf is not None:
            tag = "pre-whitened" if self.prewhitened else "raw"
            _stem(axes[2], self.ccf, f"CCF ({tag}) — {self.driver} → {self.target}")
            axes[2].set_xlabel("lag (driver leads target →)")
        axes[-1].set_xlabel(axes[-1].get_xlabel() or "lag")
        return axes


def _z_value(alpha: float) -> float:
    """Two-sided normal quantile ``z_{1-alpha/2}`` (uses SciPy if available)."""
    try:
        stats = require("scipy.stats")
        return float(stats.norm.ppf(1.0 - alpha / 2.0))
    except ImportError:
        # Fall back to the common 95% value; good enough for a rule-of-thumb band.
        return 1.959963984540054


def _clean_target(df: pd.DataFrame, col: str, date_col: str) -> pd.Series:
    """Sort by date and return ``col`` as a float Series indexed 0..n-1."""
    out = _ensure_sorted_by_date(df, date_col=date_col)
    if col not in out.columns:
        raise ValueError(f"Column '{col}' not in df.")
    s = pd.to_numeric(out[col], errors="coerce").reset_index(drop=True)
    # ``to_numeric`` passes +-inf through, but the docstring promises that
    # non-finite values are dropped; inf otherwise poisons every correlation
    # computed from the series.
    return s.replace([np.inf, -np.inf], np.nan).astype(float)


def _ccf_at_lags(driver: pd.Series, target: pd.Series, lags: Sequence[int]) -> np.ndarray:
    """Pearson cross-correlation; lag k>0 => corr(driver.shift(k), target)."""
    vals = []
    for k in lags:
        pair = pd.concat([driver.shift(k), target], axis=1).dropna()
        vals.append(pair.iloc[:, 0].corr(pair.iloc[:, 1]) if len(pair) >= 3 else np.nan)
    return np.asarray(vals, dtype=float)


def _pairwise_acf_pacf(y: pd.Series, nlags: int) -> tuple[np.ndarray, np.ndarray]:
    """Gap-preserving ACF and Yule-Walker PACF when missing steps remain on a grid."""
    arr = y.to_numpy(dtype=float)
    acf = np.full(nlags + 1, np.nan, dtype=float)
    acf[0] = 1.0
    for k in range(1, nlags + 1):
        a = arr[:-k]
        b = arr[k:]
        m = np.isfinite(a) & np.isfinite(b)
        if int(m.sum()) < 3:
            continue
        sa = float(np.std(a[m]))
        sb = float(np.std(b[m]))
        if sa > 1e-12 and sb > 1e-12:
            acf[k] = float(np.corrcoef(a[m], b[m])[0, 1])

    pacf_lags = min(nlags, max(1, int(np.isfinite(arr).sum()) // 2 - 1))
    pacf = np.full(pacf_lags + 1, np.nan, dtype=float)
    pacf[0] = 1.0
    for k in range(1, pacf_lags + 1):
        sub_r = acf[: k + 1]
        if not np.all(np.isfinite(sub_r)):
            break
        mat = np.empty((k, k), dtype=float)
        for i in range(k):
            for j in range(k):
                mat[i, j] = sub_r[abs(i - j)]
        try:
            phi = np.linalg.solve(mat + 1e-12 * np.eye(k), sub_r[1:])
            pacf[k] = float(np.clip(phi[-1], -1.0, 1.0))
        except np.linalg.LinAlgError:
            break
    return acf, pacf


def _prewhiten(
    driver: pd.Series, target: pd.Series, max_ar: int
) -> tuple[pd.Series, pd.Series, int]:
    """Box-Jenkins pre-whitening preserving true grid index positions across gaps."""
    ar_model = require("statsmodels.tsa.ar_model", hint="pip install statsmodels")
    if driver.notna().all():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = ar_model.ar_select_order(driver, maxlag=max(1, int(max_ar)), ic="aic")
            p = max(sel.ar_lags) if sel.ar_lags else 1
            res = ar_model.AutoReg(driver, lags=p).fit()
        phi = np.asarray(res.params[1 : p + 1], dtype=float)
        driver_white = res.resid
    else:
        d_fit = (
            driver.interpolate(method="linear", limit_area="inside").dropna().reset_index(drop=True)
        )
        eff_max = min(max(1, int(max_ar)), max(1, len(d_fit) // 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = ar_model.ar_select_order(d_fit, maxlag=eff_max, ic="aic")
            p = max(sel.ar_lags) if sel.ar_lags else 1
            res = ar_model.AutoReg(d_fit, lags=p).fit()
        phi = np.asarray(res.params[1 : p + 1], dtype=float)
        dc = driver - driver.mean()
        driver_white = dc.copy()
        for i in range(1, p + 1):
            driver_white = driver_white - phi[i - 1] * dc.shift(i)
        driver_white = driver_white.iloc[p:]

    # Apply the same AR polynomial (1 - phi(B)) to the centred target on un-compressed grid steps.
    yc = target - target.mean()
    y_f = yc.copy()
    for i in range(1, p + 1):
        y_f = y_f - phi[i - 1] * yc.shift(i)
    y_f = y_f.iloc[p:]

    common = driver_white.index.intersection(y_f.index)
    return driver_white.loc[common], y_f.loc[common], p


def analyze_lag(
    df: pd.DataFrame,
    target: str,
    driver: str | None = None,
    *,
    max_lag: int = 48,
    date_col: str = "date",
    prewhiten: bool = True,
    max_ar: int = 24,
    alpha: float = 0.05,
    freq: str | None = None,
) -> LagDiagnostics:
    """Diagnose the lag structure of a target and its lead-lag with a driver.

    Computes the target's autocorrelation (ACF) and partial autocorrelation
    (PACF) to suggest autoregressive lags, and — when ``driver`` is given — the
    cross-correlation (CCF) between driver and target to suggest predictive
    driver lags. By default the CCF is computed on *pre-whitened* series so that
    shared seasonality / autocorrelation does not produce spurious peaks
    (Box-Jenkins pre-whitening).

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format dataset with a datetime column.
    target : str
        Response column (e.g. the pollutant).
    driver : str, optional
        Driver column (e.g. a meteorological variable). If omitted, only ACF
        and PACF of the target are returned.
    max_lag : int, default 48
        Maximum lag (rows, or ``freq`` steps when ``freq`` is set) for
        ACF/PACF and the positive/negative CCF range.
    date_col : str, default "date"
        Datetime column used to sort before the analysis.
    prewhiten : bool, default True
        Pre-whiten before the CCF. Strongly recommended for autocorrelated,
        seasonal environmental series. Ignored when ``driver`` is ``None``.
    max_ar : int, default 24
        Maximum AR order considered for the pre-whitening filter (AIC-selected).
    alpha : float, default 0.05
        Two-sided significance level for the white-noise bands.
    freq : str, optional
        Regular frequency (e.g. ``"h"``) to reindex timestamps onto before
        computing ACF/PACF/CCF so gaps keep their true temporal spacing.

    Returns
    -------
    LagDiagnostics
        Tables of ACF/PACF/CCF plus the suggested lags. ``CCF`` lag ``k > 0``
        means the driver leads the target by ``k`` rows, matching the sign of
        ``add_lag_features(..., lags=[k])``.
    """
    if max_lag < 1:
        raise ValueError("`max_lag` must be >= 1.")
    stattools = require("statsmodels.tsa.stattools", hint="pip install statsmodels")

    work = df
    if freq is not None:
        sorted_df = _ensure_sorted_by_date(df, date_col=date_col)
        if sorted_df[date_col].duplicated().any():
            raise ValueError(
                f"Duplicate timestamps in `{date_col}` cannot be uniquely aligned to `freq={freq!r}`."
            )
        full_idx = pd.date_range(sorted_df[date_col].min(), sorted_df[date_col].max(), freq=freq)
        work = (
            sorted_df.set_index(date_col)
            .reindex(full_idx)
            .reset_index()
            .rename(columns={"index": date_col})
        )

    y = _clean_target(work, target, date_col=date_col)
    n_target = int(y.notna().sum())
    if n_target < 3:
        # acf/pacf below divide by a zero deviation and return warnings, not
        # a result: say what is wrong instead.
        raise ValueError(f"`{target}` has {n_target} finite observation(s); at least 3 are needed.")
    z = _z_value(alpha)

    if freq is not None and y.isna().any():
        nlags = min(int(max_lag), max(1, len(y) - 2))
        acf_vals, pacf_vals = _pairwise_acf_pacf(y, nlags)
    else:
        nlags = min(int(max_lag), max(1, len(y.dropna()) - 2))
        acf_vals = stattools.acf(y, nlags=nlags, missing="drop", fft=True)
        pacf_vals = stattools.pacf(y.dropna(), nlags=min(nlags, len(y.dropna()) // 2 - 1))

    acf_df = pd.DataFrame({"lag": np.arange(len(acf_vals)), "value": acf_vals})
    pacf_df = pd.DataFrame({"lag": np.arange(len(pacf_vals)), "value": pacf_vals})

    band = z / np.sqrt(n_target)

    target_ar_lags = [
        int(k)
        for k, v in zip(pacf_df["lag"], pacf_df["value"], strict=False)
        if k >= 1 and np.isfinite(v) and abs(v) > band
    ]

    if driver is None:
        return LagDiagnostics(
            target=target,
            driver=None,
            n=n_target,
            alpha=alpha,
            band=float(band),
            acf=acf_df,
            pacf=pacf_df,
            target_ar_lags=target_ar_lags,
        )

    x = _clean_target(work, driver, date_col=date_col)
    prewhitened = bool(prewhiten)
    if prewhitened:
        try:
            x_use, y_use, _p = _prewhiten(x, y, max_ar=max_ar)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully to raw CCF
            log.warning("Pre-whitening failed (%s); falling back to raw CCF.", exc)
            prewhitened = False
            x_use, y_use = x, y
    else:
        x_use, y_use = x, y

    lags = list(range(-int(max_lag), int(max_lag) + 1))
    ccf_vals = _ccf_at_lags(x_use, y_use, lags)
    ccf_df = pd.DataFrame({"lag": lags, "value": ccf_vals})

    n_ccf = int(pd.concat([x_use, y_use], axis=1).dropna().shape[0])
    band_ccf = z / np.sqrt(max(n_ccf, 1))

    lead = ccf_df[(ccf_df["lag"] >= 0) & np.isfinite(ccf_df["value"])]
    driver_lags = [
        int(k) for k, v in zip(lead["lag"], lead["value"], strict=False) if abs(v) > band_ccf
    ]
    peak_lag = int(lead.loc[lead["value"].abs().idxmax(), "lag"]) if not lead.empty else None

    return LagDiagnostics(
        target=target,
        driver=driver,
        n=n_ccf,
        alpha=alpha,
        band=float(band_ccf),
        acf=acf_df,
        pacf=pacf_df,
        ccf=ccf_df,
        target_ar_lags=target_ar_lags,
        driver_lags=driver_lags,
        peak_lag=peak_lag,
        prewhitened=prewhitened,
    )
