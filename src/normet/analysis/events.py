# src/normet/analysis/events.py
"""
Event / anomaly detection on environmental time-series.

Provides three complementary methods:

- ``"iqr"``           : robust univariate threshold (median ± k·IQR).
- ``"isolation"``     : sklearn IsolationForest on lagged features.
- ``"stl_residual"``  : STL decomposition (statsmodels) then IQR on residuals.

All return a DataFrame of consecutive anomalous intervals, ranked by score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils._lazy import require
from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["detect_events", "anomaly_scores"]


def _to_series(series: pd.Series | pd.DataFrame, value_col: str | None) -> pd.Series:
    if isinstance(series, pd.DataFrame):
        if value_col is None:
            raise ValueError("`value_col` is required when passing a DataFrame.")
        if value_col not in series.columns:
            raise ValueError(f"Column '{value_col}' not in DataFrame.")
        s = series[value_col]
    else:
        s = series
    if not isinstance(s.index, pd.DatetimeIndex):
        s = s.copy()
        s.index = pd.to_datetime(s.index)
    return s


def _intervals_from_mask(
    idx: pd.DatetimeIndex, mask: np.ndarray, scores: np.ndarray
) -> pd.DataFrame:
    """Collapse a boolean mask into consecutive (start, end, score) intervals."""
    if not mask.any():
        return pd.DataFrame(columns=["start", "end", "n", "max_score", "mean_score"])
    rows = []
    in_run = False
    run_start = 0
    for i in range(len(mask)):
        if mask[i] and not in_run:
            run_start = i
            in_run = True
        elif (not mask[i]) and in_run:
            seg_scores = scores[run_start:i]
            rows.append(
                {
                    "start": idx[run_start],
                    "end": idx[i - 1],
                    "n": int(i - run_start),
                    "max_score": float(np.nanmax(seg_scores)),
                    "mean_score": float(np.nanmean(seg_scores)),
                }
            )
            in_run = False
    if in_run:
        seg_scores = scores[run_start:]
        rows.append(
            {
                "start": idx[run_start],
                "end": idx[-1],
                "n": int(len(mask) - run_start),
                "max_score": float(np.nanmax(seg_scores)),
                "mean_score": float(np.nanmean(seg_scores)),
            }
        )
    return pd.DataFrame(rows).sort_values("max_score", ascending=False).reset_index(drop=True)


def anomaly_scores(
    series: pd.Series | pd.DataFrame,
    *,
    value_col: str | None = None,
    method: str = "iqr",
    stl_period: int | None = None,
    seed: int = 7_654_321,
) -> pd.Series:
    """
    Compute a per-timestamp anomaly score (larger = more anomalous).

    Parameters
    ----------
    series : Series or DataFrame
        Time-indexed series. If DataFrame, pass ``value_col``.
    value_col : str, optional
        Column to score when ``series`` is a DataFrame.
    method : {"iqr","isolation","stl_residual"}, default "iqr"
    stl_period : int, optional
        Period for STL decomposition (e.g., 24 for hourly w/ daily cycle).
    seed : int, default=7654321
        Random seed for reproducible methods (IsolationForest).

    Returns
    -------
    pandas.Series
        Same index as input; anomaly score per timestamp.
    """
    s = _to_series(series, value_col)
    s = s.astype(float)

    if method == "iqr":
        med = float(s.median())
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            return pd.Series(0.0, index=s.index, name="score")
        score = (s - med).abs() / iqr
        return score.rename("score")

    if method == "isolation":
        sk = require("sklearn.ensemble", hint="pip install scikit-learn")
        # Feature engineering: value + a small lag bank for context
        feats = pd.DataFrame({"x": s})
        for lag in (1, 2, 3, 6, 12):
            feats[f"lag_{lag}"] = s.shift(lag)
        feats = feats.bfill().ffill().fillna(0.0)
        iso = sk.IsolationForest(contamination="auto", random_state=int(seed))
        iso.fit(feats.values)
        # Higher = more anomalous: invert sklearn's decision_function (higher = normal)
        raw = -iso.decision_function(feats.values)
        return pd.Series(raw, index=s.index, name="score")

    if method == "stl_residual":
        sm = require("statsmodels.tsa.seasonal", hint="pip install statsmodels")
        if stl_period is None:
            # Try to infer from index frequency
            freq = pd.infer_freq(pd.DatetimeIndex(s.index))
            stl_period = {"H": 24, "h": 24, "D": 7, "W": 52, "M": 12}.get(freq or "", 12)
        stl = sm.STL(s.ffill().bfill(), period=int(stl_period), robust=True).fit()
        resid = stl.resid
        med = float(resid.median())
        q1, q3 = float(resid.quantile(0.25)), float(resid.quantile(0.75))
        iqr = max(q3 - q1, 1e-12)
        return ((resid - med).abs() / iqr).rename("score")

    raise ValueError(f"Unknown method: {method}")


def detect_events(
    series: pd.Series | pd.DataFrame,
    *,
    value_col: str | None = None,
    method: str = "iqr",
    k: float = 3.0,
    min_length: int = 1,
    stl_period: int | None = None,
    seed: int = 7_654_321,
) -> pd.DataFrame:
    """
    Identify consecutive anomalous time intervals.

    Parameters
    ----------
    series, value_col, method, k, stl_period :
        See :func:`anomaly_scores`.
    min_length : int, default 1
        Minimum number of consecutive flagged timestamps for an event.
    seed : int, default=7654321
        Random seed forwarded to :func:`anomaly_scores`.

    Returns
    -------
    pandas.DataFrame
        Columns: ``start``, ``end``, ``n``, ``max_score``, ``mean_score``,
        sorted by ``max_score`` descending. Returns an empty DataFrame
        with these columns if nothing is flagged.
    """
    score = anomaly_scores(
        series, value_col=value_col, method=method, stl_period=stl_period, seed=seed
    )
    arr = score.to_numpy(dtype=float)

    # IsolationForest uses 0 as the natural decision boundary; other methods use k·IQR.
    threshold = 0.0 if method == "isolation" else float(k)

    mask = (arr > threshold) & np.isfinite(arr)
    events = _intervals_from_mask(pd.DatetimeIndex(score.index), mask, arr)  # type: ignore[arg-type]
    if events.empty:
        return events
    return events[events["n"] >= int(min_length)].reset_index(drop=True)
