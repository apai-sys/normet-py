# src/normet/analysis/events.py
"""
Event / anomaly detection on environmental time-series.

Provides three complementary methods:

- ``"iqr"``           : robust univariate threshold (median ± k·IQR).
- ``"isolation"``     : sklearn IsolationForest on lagged features (IQR-standardised).
- ``"stl_residual"``  : STL decomposition (statsmodels) then IQR on residuals.

All return a DataFrame of consecutive anomalous intervals, ranked by score.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..utils._lazy import require
from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "detect_events",
    "anomaly_scores",
    "HindsightEpisodeResult",
    "hindsight_episode_counterfactual",
]


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


def _prepare_chronos_frame(series: pd.Series | pd.DataFrame, value_col: str | None) -> pd.DataFrame:
    """Validate and sort a DataFrame input for ``method='chronos'`` anomaly scoring."""
    if not isinstance(series, pd.DataFrame) or value_col is None:
        raise ValueError(
            "method='chronos' requires a DataFrame `series` and `value_col` target column."
        )
    work = series.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        if "date" in work.columns:
            work = work.set_index("date")
        work.index = pd.to_datetime(work.index)
    return work.sort_index()


def anomaly_scores(
    series: pd.Series | pd.DataFrame,
    *,
    value_col: str | None = None,
    method: str = "iqr",
    stl_period: int | None = None,
    seed: int = 7_654_321,
    estimator: Any | None = None,
    covariates: Sequence[str] | None = None,
    past_only_covariates: Sequence[str] | None = None,
    calibration: Any | None = None,
    adaptive: bool = False,
) -> pd.Series:
    """
    Compute a per-timestamp anomaly score (larger = more anomalous).

    Parameters
    ----------
    series : Series or DataFrame
        Time-indexed series. If DataFrame, pass ``value_col``.
    value_col : str, optional
        Column to score when ``series`` is a DataFrame.
    method : {"iqr","isolation","stl_residual","chronos"}, default "iqr"
    stl_period : int, optional
        Period for STL decomposition (e.g., 24 for hourly w/ daily cycle).
    seed : int, default=7654321
        Random seed for reproducible methods (IsolationForest).
    estimator : Chronos2Estimator, optional
        Pre-loaded Chronos-2 estimator when ``method="chronos"``.
    covariates, past_only_covariates, calibration, adaptive :
        Forwarded to :func:`normet.foundation.chronos_anomaly_scores` when
        ``method="chronos"``.

    Returns
    -------
    pandas.Series
        Same index as input; anomaly score per timestamp.
    """
    if method == "chronos":
        work = _prepare_chronos_frame(series, value_col)
        assert value_col is not None
        from ..foundation import Chronos2Estimator, chronos_anomaly_scores

        est = estimator if estimator is not None else Chronos2Estimator()
        scores_df = chronos_anomaly_scores(
            est,
            work,
            value_col,
            covariates=covariates,
            past_only_covariates=past_only_covariates,
            calibration=calibration,
            adaptive=adaptive,
        )
        return scores_df["surprise"].rename("score")

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
        if len(s) == 0:
            return pd.Series(dtype=float, index=s.index, name="score")
        # Feature engineering: value + a small lag bank for context
        feats = pd.DataFrame({"x": s})
        for lag in (1, 2, 3, 6, 12):
            feats[f"lag_{lag}"] = s.shift(lag)
        feats = feats.bfill().ffill().fillna(0.0)
        iso = sk.IsolationForest(contamination="auto", random_state=int(seed))
        iso.fit(feats.values)
        # Higher = more anomalous: invert sklearn's decision_function (higher = normal),
        # then standardise via median/IQR onto the same k·IQR scale used by iqr and
        # stl_residual so k tunes sensitivity and constant series return 0.
        raw = -iso.decision_function(feats.values)
        med = float(np.median(raw))
        q1, q3 = float(np.quantile(raw, 0.25)), float(np.quantile(raw, 0.75))
        iqr = q3 - q1
        if iqr <= 1e-12:
            return pd.Series(0.0, index=s.index, name="score")
        return pd.Series((raw - med) / iqr, index=s.index, name="score")

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
    estimator: Any | None = None,
    covariates: Sequence[str] | None = None,
    past_only_covariates: Sequence[str] | None = None,
    calibration: Any | None = None,
    adaptive: bool = False,
    alpha: float = 0.05,
    direction: str = "both",
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
    estimator, covariates, past_only_covariates, calibration, adaptive, alpha, direction :
        Forwarded to :func:`normet.foundation.detect_chronos_events` when
        ``method="chronos"``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``start``, ``end``, ``n``, ``max_score``, ``mean_score``,
        sorted by ``max_score`` descending. Returns an empty DataFrame
        with these columns if nothing is flagged.
    """
    if method == "chronos":
        work = _prepare_chronos_frame(series, value_col)
        assert value_col is not None
        from ..foundation import Chronos2Estimator, detect_chronos_events

        est = estimator if estimator is not None else Chronos2Estimator()
        return detect_chronos_events(
            est,
            work,
            value_col,
            alpha=alpha,
            min_length=min_length,
            direction=direction,
            covariates=covariates,
            past_only_covariates=past_only_covariates,
            calibration=calibration,
            adaptive=adaptive,
        )

    score = anomaly_scores(
        series, value_col=value_col, method=method, stl_period=stl_period, seed=seed
    )
    arr = score.to_numpy(dtype=float)

    # All univariate/residual methods (iqr, isolation, stl_residual) return IQR-scaled scores.
    threshold = float(k)

    mask = (arr > threshold) & np.isfinite(arr)
    events = _intervals_from_mask(pd.DatetimeIndex(score.index), mask, arr)
    if events.empty:
        return events
    return events[events["n"] >= int(min_length)].reset_index(drop=True)


@dataclass
class HindsightEpisodeResult:
    """Counterfactual trajectory and attribution for an observed episode window."""

    episode_start: pd.Timestamp
    episode_end: pd.Timestamp
    n_steps: int
    actual_mean: float
    counterfactual_mean: float
    attribution_abs: float
    attribution_pct: float
    trajectory: pd.DataFrame
    summary_metrics: dict[str, Any] = field(default_factory=dict)


def hindsight_episode_counterfactual(
    df_norm: pd.DataFrame,
    episode_start: str | pd.Timestamp,
    episode_end: str | pd.Timestamp,
    *,
    observed_col: str = "observed",
    normalised_col: str = "normalised",
    factual_col: str = "factual",
    coordinate: str = "auto",
    alternative_residual: str | float | np.ndarray | pd.Series = "zero",
) -> HindsightEpisodeResult:
    """Construct a hindsight counterfactual trajectory for a specific episode window.

    Combines the model's baseline prediction (``factual`` when available, falling
    back to ``normalised``) with a baseline or user-supplied latent disturbance
    in raw or ``log1p`` coordinates while preserving realized episode meteorology.

    Parameters
    ----------
    df_norm : pandas.DataFrame
        DataFrame with a DatetimeIndex (or ``date`` column) and ``observed`` plus
        ``factual`` and/or ``normalised`` columns.
    episode_start, episode_end : str or pandas.Timestamp
        Inclusive boundaries of the hindsight episode window.
    observed_col, normalised_col, factual_col : str
        Column names for observed levels, deweathered levels, and factual predictions.
    coordinate : {"auto", "raw", "log1p"}, default "auto"
        Residual space. ``"auto"`` uses ``"log1p"`` when observed and baseline are
        non-negative, else ``"raw"``.
    alternative_residual : {"zero", "pre_mean", "pre_median"} or float/array/Series, default "zero"
        Latent disturbance applied under the counterfactual hypothesis. ``"zero"``
        ablates the episode's unmodeled disturbance (yielding the no-episode trajectory
        under realized weather); ``"pre_mean"`` / ``"pre_median"`` substitutes pre-episode
        background residual levels.

    Returns
    -------
    HindsightEpisodeResult
    """
    if coordinate not in ("auto", "raw", "log1p"):
        raise ValueError(f"`coordinate` must be 'auto', 'raw', or 'log1p', got {coordinate!r}.")

    work = df_norm.copy()
    if not isinstance(work.index, pd.DatetimeIndex):
        if "date" not in work.columns:
            raise ValueError("`df_norm` must have a DatetimeIndex or a 'date' column.")
        work = work.set_index(pd.to_datetime(work["date"]))
    work = work.sort_index()
    idx = pd.DatetimeIndex(work.index)

    if observed_col not in work.columns:
        raise KeyError(f"Column {observed_col!r} not found in `df_norm`.")
    has_factual = factual_col in work.columns
    if has_factual:
        base_col = factual_col
    elif normalised_col in work.columns:
        base_col = normalised_col
    else:
        raise KeyError(f"Neither {factual_col!r} nor {normalised_col!r} found in `df_norm`.")

    def _align_ts(val: str | pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(val)
        if idx.tz is None:
            return ts.tz_localize(None) if ts.tzinfo is not None else ts
        return ts.tz_localize(idx.tz) if ts.tzinfo is None else ts.tz_convert(idx.tz)

    ep_s = _align_ts(episode_start)
    ep_e = _align_ts(episode_end)
    if ep_e < ep_s:
        raise ValueError(f"`episode_end` ({ep_e}) precedes `episode_start` ({ep_s}).")

    ep_mask = np.asarray((idx >= ep_s) & (idx <= ep_e), dtype=bool)
    n_ep = int(ep_mask.sum())
    if n_ep == 0:
        raise ValueError("No timestamps fall inside the requested episode window.")

    obs_full = pd.to_numeric(work[observed_col], errors="coerce").to_numpy(dtype=float)
    base_full = pd.to_numeric(work[base_col], errors="coerce").to_numpy(dtype=float)
    fin_full = np.isfinite(obs_full) & np.isfinite(base_full)

    if coordinate == "auto":
        non_neg = bool(np.all(obs_full[fin_full] >= -1e-9)) and bool(
            np.all(base_full[fin_full] >= -1e-9)
        )
        coord_eff = "log1p" if (non_neg and np.any(fin_full)) else "raw"
    else:
        coord_eff = coordinate

    if coord_eff == "log1p":
        z_obs = np.log1p(np.maximum(obs_full, 0.0))
        z_base = np.log1p(np.maximum(base_full, 0.0))
        r_coord = z_obs - z_base
    else:
        r_coord = obs_full - base_full

    pre_mask = np.asarray(idx < ep_s, dtype=bool) & fin_full
    if isinstance(alternative_residual, str):
        if alternative_residual == "zero":
            r_alt = np.zeros(n_ep, dtype=float)
        elif alternative_residual == "pre_mean":
            fill = float(np.nanmean(r_coord[pre_mask])) if np.any(pre_mask) else 0.0
            r_alt = np.full(n_ep, fill, dtype=float)
        elif alternative_residual == "pre_median":
            fill = float(np.nanmedian(r_coord[pre_mask])) if np.any(pre_mask) else 0.0
            r_alt = np.full(n_ep, fill, dtype=float)
        else:
            raise ValueError(
                "`alternative_residual` string must be 'zero', 'pre_mean', or 'pre_median', "
                f"got {alternative_residual!r}."
            )
    elif isinstance(alternative_residual, pd.Series):
        r_alt = pd.to_numeric(alternative_residual.reindex(idx[ep_mask]), errors="coerce").to_numpy(
            dtype=float
        )
    elif np.ndim(alternative_residual) == 0:
        r_alt = np.full(n_ep, float(alternative_residual), dtype=float)
    else:
        arr_alt = np.asarray(alternative_residual, dtype=float).reshape(-1)
        if len(arr_alt) != n_ep:
            raise ValueError(
                f"Length of `alternative_residual` ({len(arr_alt)}) does not match episode length ({n_ep})."
            )
        r_alt = arr_alt

    obs_ep = obs_full[ep_mask]
    base_ep = base_full[ep_mask]
    if coord_eff == "log1p":
        cf_ep = np.maximum(np.expm1(np.log1p(np.maximum(base_ep, 0.0)) + r_alt), 0.0)
    else:
        cf_ep = base_ep + r_alt

    diff_ep = obs_ep - cf_ep
    traj = pd.DataFrame(
        {
            "actual": obs_ep,
            "counterfactual": cf_ep,
            "baseline": base_ep,
            "difference": diff_ep,
            "disturbance_raw": obs_ep - base_ep,
        },
        index=idx[ep_mask],
    )

    valid_ep = np.isfinite(obs_ep) & np.isfinite(cf_ep)
    actual_mean = float(np.mean(obs_ep[valid_ep])) if np.any(valid_ep) else float("nan")
    cf_mean = float(np.mean(cf_ep[valid_ep])) if np.any(valid_ep) else float("nan")
    attr_abs = float(np.mean(diff_ep[valid_ep])) if np.any(valid_ep) else float("nan")
    attr_tot = float(np.sum(diff_ep[valid_ep])) if np.any(valid_ep) else float("nan")
    attr_pct = (
        float(100.0 * attr_abs / max(abs(cf_mean), 1e-12))
        if np.isfinite(attr_abs) and np.isfinite(cf_mean)
        else float("nan")
    )

    r_ep_fin = r_coord[ep_mask & fin_full]
    ep_ac1 = float("nan")
    if len(r_ep_fin) >= 4:
        a, b = r_ep_fin[:-1], r_ep_fin[1:]
        if float(np.std(a)) > 1e-12 and float(np.std(b)) > 1e-12:
            ep_ac1 = float(np.corrcoef(a, b)[0, 1])

    summary: dict[str, Any] = {
        "coordinate": coord_eff,
        "baseline_col": base_col,
        "has_factual": has_factual,
        "alternative_residual": (
            alternative_residual if isinstance(alternative_residual, str) else "custom"
        ),
        "n_steps": n_ep,
        "actual_mean": actual_mean,
        "counterfactual_mean": cf_mean,
        "attribution_abs": attr_abs,
        "attribution_total": attr_tot,
        "attribution_pct": attr_pct,
        "episode_residual_autocorr": ep_ac1,
    }
    return HindsightEpisodeResult(
        episode_start=idx[ep_mask][0],
        episode_end=idx[ep_mask][-1],
        n_steps=n_ep,
        actual_mean=actual_mean,
        counterfactual_mean=cf_mean,
        attribution_abs=attr_abs,
        attribution_pct=attr_pct,
        trajectory=traj,
        summary_metrics=summary,
    )
