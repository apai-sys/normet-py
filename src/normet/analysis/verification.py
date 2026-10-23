# src/normet/analysis/verification.py
"""Skill scores for probabilistic forecasts: CRPS and its pinball components.

Point forecasts are verified by :func:`normet.modStats` (FAC2, RMSE, ...).
The question those cannot answer is the one every comparison with a baseline
must settle: *how much better is the forecast, in the spread it carries, than
the climatology or persistence it is up against?* The Continuous Ranked
Probability Score (CRPS) scores a forecast's whole distribution against the
observation; it is non-negative, in the same units as the quantity, and is
the standard skill score for probabilistic forecasts (Gneiting and Raftery,
2005, Journal of the American Statistical Association 100:724).

- :func:`pinball_loss`        : the quantile (pinball) loss at one quantile
                                 level -- the elementary score the others
                                 are built from.
- :func:`crps_from_quantiles` : the CRPS of a quantile forecast, by exact
                                 integration; the ``qNNN`` columns a
                                 ``normalise`` run returns (with
                                 ``return_quantiles``) are its direct input.
- :func:`crps_from_samples`   : the CRPS of ensemble samples, each sample
                                 carrying its rank as a quantile level. The
                                 wide table a ``normalise`` run returns with
                                 ``aggregate=False`` is its direct input.
- :func:`crps_observed`       : the point forecast's reference on a skill table:
                                  its MAE read as a normal's own MAE (sigma * sqrt(2/pi)),
                                  so the score is the sigma that would carry it -- the MAE
                                  times ``sqrt(pi/2) ~= 1.253`` -- which puts model and
                                  baseline on one scale.

The first three scores are comparable on equal records with the same units;
``crps_observed`` is the reference a point forecast carries on the same
record.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import DataError

__all__ = [
    "pinball_loss",
    "crps_from_quantiles",
    "crps_from_samples",
    "crps_observed",
]


#: Reads a point forecast's MAE as a normal's own MAE (sigma * sqrt(2/pi))
#: and multiplies back to the sigma that would carry it: ``sqrt(pi/2)``.
_POINT_MASS_FACTOR = float(np.sqrt(np.pi / 2.0))


def _as_quantile_level(q: Any, what: str = "quantile level") -> float:
    """Validate a quantile level and return it as a float in (0, 1)."""
    try:
        level = float(q)
    except (TypeError, ValueError):
        raise DataError(f"{what} {q!r} is not a number.") from None
    if not (0.0 < level < 1.0):
        raise DataError(f"{what} must be in (0, 1): got {level}")
    return level


def _column_level(name: Any) -> float:
    """The level a quantile column names: ``q025`` -> 0.025, else itself."""
    s = str(name)
    if s.startswith("q") and s[1:].isdigit():
        return float(int(s[1:])) / 1000.0
    return _as_quantile_level(name, f"the quantile label {name!r}")


def _obs_array(obs: Any) -> np.ndarray:
    """The observation as a 1-D float array."""
    o = (
        obs.to_numpy(dtype=float)
        if isinstance(obs, (pd.DataFrame, pd.Series))
        else np.asarray(obs, dtype=float)
    )
    if o.ndim != 1:
        raise DataError("The observation must be one value per row.")
    return o


def _levels_and_values(
    preds: pd.DataFrame | np.ndarray[Any, Any] | list,
    quantiles: None | list[Any],
) -> tuple[np.ndarray, list[float]]:
    """The quantile forecast as a (rows, levels) float array and its levels.

    A frame carries one column per level: its ``qNNN`` columns when
    ``quantiles`` is left out, else the named columns. An array's last axis
    is the levels, and then ``quantiles`` must be given. One level is a
    point prediction, not a distribution to score.
    """
    # The branches below assign from pandas and numpy stubs that disagree on
    # the shape parameter; one explicit annotation keeps the reassignment
    # clean under every numpy 2.x stub generation.
    p: np.ndarray
    if isinstance(preds, pd.DataFrame):
        frame = preds
        if quantiles is None:
            names = sorted(
                c
                for c in frame.columns
                if isinstance(c, str) and c.startswith("q") and c[1:].isdigit()
            )
            if len(names) < 2:
                raise DataError(
                    "A quantile frame needs its levels labelled qNNN "
                    "(e.g. q025, q500, q975): pass quantiles= to name them."
                )
        else:
            names = [str(c) for c in quantiles]
            missing = [c for c in names if c not in frame.columns]
            if missing:
                raise DataError(f"No such column(s) in the frame: {missing}")
        levels = [_column_level(c) for c in names]
        p = frame[names].to_numpy(dtype=float)  # (records, levels)
    else:
        p = np.asarray(preds, dtype=float)
        if p.ndim == 1:
            raise DataError("One level is a point prediction: score it with crps_observed.")
        if p.ndim != 2:
            raise DataError("A quantile array needs a shape (rows, levels).")
        if quantiles is None:
            raise DataError("An array of quantiles needs quantiles= to say what level each is.")
        if len(quantiles) != p.shape[1]:
            raise DataError(f"{len(quantiles)} levels for a table of {p.shape[1]} levels.")
        levels = [_as_quantile_level(q) for q in quantiles]
    if p.shape[0] == 0:
        raise DataError("Nothing to score: the record is empty.")
    if len(set(levels)) != len(levels):
        raise DataError("The quantile levels repeat.")
    return p, levels


def _row_crps(vs: np.ndarray, tv: np.ndarray, y: float) -> float:
    """One row's CRPS, exactly.

    The CDF is the piecewise-linear curve through the row's quantile
    points (value, level): flat 0 below the smallest value, flat 1 above
    the largest, a repeated value a vertical piece of measure zero, and
    the score is the integral of ``(F - 1{x >= y})^2``. On the
    break-point grid of the distinct values and y the integrand is a
    quadratic per interval -- the indicator is constant on each, since
    y is one of the break points -- and the pieces are integrated
    exactly, tails and off-span observations included.
    """
    order = np.argsort(vs, kind="stable")
    vs, tv = vs[order], tv[order].astype(float)
    v, inv = np.unique(vs, return_inverse=True)  # the distinct values
    m = v.size
    u = np.zeros(m)
    np.maximum.at(u, inv, tv)  # a value group's level: its highest
    grid = np.unique(np.concatenate([v, [y]]))
    pos = np.searchsorted(v, grid, side="right") - 1  # largest v <= grid
    # The CDF's value on the open intervals, by its right- and left-limits
    # at each break point: 0 below the smallest value, 1 above the
    # largest, the level u at an interior value, linearly between.
    if m >= 2:
        i = np.clip(pos, 0, m - 2)
        denom = v[i + 1] - v[i]
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.where(denom > 0, (grid - v[i]) / np.where(denom > 0, denom, 1.0), 0.0)
        base = u[i] + t * (u[i + 1] - u[i])
    else:
        base = np.zeros(grid.size)
    fr = np.where(
        grid >= v[-1],
        1.0,
        np.where(grid == v[0], u[0], np.where(grid > v[0], base, 0.0)),
    )
    fl = np.where(grid <= v[0], 0.0, np.where(grid > v[-1], 1.0, base))
    a = fr[:-1]  # each interval's CDF at its left end
    b = fl[1:]  # and its right end
    d = np.diff(grid)
    ind = (grid[:-1] >= y).astype(float)  # 1{x >= y} on the interval
    c = a - ind
    slope = np.where(d > 0, (b - a) / np.where(d > 0, d, 1.0), 0.0)
    seg = c * c * d + c * slope * d * d + (slope * slope * d * d * d) / 3.0
    return float(seg.sum())


def _crps_rows(q: np.ndarray, levels: list[float], y: np.ndarray) -> float:
    """The CRPS of quantile forecasts ``q`` (rows, levels) at ``levels``
    against ``y`` (rows,): the mean of the rows' exact scores."""
    tau = np.asarray(levels, dtype=float)
    n = q.shape[0]
    total = 0.0
    for i in range(n):
        total += _row_crps(q[i], tau, y[i])
    return total / n


def pinball_loss(preds: Any, obs: Any, q: float) -> float:
    """The quantile (pinball) loss of ``preds`` against ``obs`` at level ``q``.

    For the residual ``r = obs - pred`` the loss is ``max(q r, (q-1) r)``:
    an under-forecast costs ``q`` per unit short, an over-forecast
    ``1 - q``; the mean is lowest when the forecast is the true ``q``-
    quantile, and at ``q = 0.5`` it is half the mean absolute error.
    Non-finite rows are dropped, as the rest of the package scores them.

    Parameters
    ----------
    preds, obs : array-like, pd.DataFrame or pd.Series
        The forecast's quantile column (or one sample of it) and the
        observation; one value per row, the same rows.
    q : float
        The quantile level, in (0, 1), that ``preds`` forecasts.

    Returns
    -------
    float
        The mean pinball loss over the finite rows, in the data's units.
    """
    level = _as_quantile_level(q)
    p = (
        preds.to_numpy(dtype=float)
        if isinstance(preds, (pd.DataFrame, pd.Series))
        else np.asarray(preds, dtype=float)
    )
    if p.ndim != 1:
        raise DataError("A single quantile's forecast must be one value per row.")
    o = _obs_array(obs)
    if len(p) != len(o):
        raise DataError(f"The forecast has {len(p)} rows and the observation {len(o)}.")
    mask = np.isfinite(p) & np.isfinite(o)
    if not mask.any():
        raise DataError("Nothing to score: no finite forecast-observation row.")
    r = o[mask] - p[mask]
    return float(np.mean(np.maximum(level * r, (level - 1.0) * r)))


def crps_from_quantiles(
    preds: pd.DataFrame | np.ndarray[Any, Any] | list,
    obs: Any,
    quantiles: None | list[float] = None,
) -> float:
    """The CRPS of a quantile forecast against the observation.

    The forecast's distribution is the CDF linearly interpolated through
    its quantile points, and the score is the squared distance of that
    CDF from the observation's, integrated exactly -- the continuous form
    the pinball sum is a trapezoidal rule for (Gneiting and Raftery 2005,
    in the quantile form). A frame's ``qNNN`` columns are its levels; an
    array's last axis is the levels, named by ``quantiles``. With few
    levels the interpolation is coarse and the score reads high: pass
    the levels the forecast actually carries.

    Parameters
    ----------
    preds : pd.DataFrame or (rows, levels) array
        One value per level and per row, in any column order.
    obs : array-like, pd.DataFrame or pd.Series
        The observation, one per row of ``preds``.
    quantiles : list, optional
        The levels the frame's columns (or the array's rows) carry, for
        when the frame's columns are not its ``qNNN`` names.

    Returns
    -------
    float
        The CRPS over the finite rows, in the data's units.
    """
    p, lev = _levels_and_values(preds, quantiles)
    o = _obs_array(obs)
    if p.shape[0] != len(o):
        raise DataError(f"The quantiles have {p.shape[0]} rows and the observation {len(o)}.")
    mask = np.isfinite(p).all(axis=1) & np.isfinite(o)
    if not mask.any():
        raise DataError("Nothing to score: no finite forecast-observation row.")
    return _crps_rows(p[mask], lev, o[mask])


def crps_from_samples(samples: pd.DataFrame | np.ndarray[Any, Any] | list, obs: Any) -> float:
    """The CRPS of ensemble samples against the observation.

    Each sample carries its rank as a quantile level: a row's sorted
    samples stand at the levels ``1/(K+1), ..., K/(K+1)``, and the score
    is the same integral the quantile one is. A frame's columns are the
    samples -- the wide table a ``normalise`` run returns with
    ``aggregate=False`` is its direct input; an array's last axis is the
    samples.

    Parameters
    ----------
    samples : pd.DataFrame or (rows, K) array
        The ensemble, one value per sample and per row.
    obs : array-like, pd.DataFrame or pd.Series
        The observation, one per row.

    Returns
    -------
    float
        The CRPS over the finite rows, in the data's units.
    """
    s = (
        samples.to_numpy(dtype=float)
        if isinstance(samples, pd.DataFrame)
        else np.asarray(samples, dtype=float)
    )
    if s.ndim == 1:
        s = s[:, None]
    if s.ndim != 2:
        raise DataError("The ensemble needs a shape (rows, samples).")
    if s.shape[1] < 2:
        raise DataError("One sample is a point prediction: score it with crps_observed.")
    o = _obs_array(obs)
    if s.shape[0] != len(o):
        raise DataError(f"The samples have {s.shape[0]} rows and the observation {len(o)}.")
    k = s.shape[1]
    levels = [i / (k + 1) for i in range(1, k + 1)]
    mask = np.isfinite(s).all(axis=1) & np.isfinite(o)
    if not mask.any():
        raise DataError("Nothing to score: no finite sample-observation row.")
    s = np.sort(s[mask], axis=1, kind="stable")  # the rank is the order
    return _crps_rows(s, levels, o[mask])


def crps_observed(y_pred: Any, y_true: Any) -> float:
    """The CRPS reference for a point forecast: ``sqrt(pi/2)`` times its MAE.

    A point forecast carries no distribution, so there is no ensemble score
    to compute for it; what is needed on a skill table is a reference value
    on the same scale as a spread score. The convention is to read the
    forecast's mean absolute error as the normal distribution's own MAE --
    a Gaussian of standard deviation sigma has MAE ``sigma * sqrt(2/pi)`` --
    and return the sigma that would carry it, which is the MAE times
    ``sqrt(pi/2) ~= 1.253``. Quote it as *the point forecast's* CRPS: it
    reads about 25 % above its MAE, and a spread forecast's score against
    it answers "is the spread worth its cost?" -- the value of the spread
    the spread forecast actually carries, not an error in either number.
    (A point mass's own CRPS is its MAE exactly; the factor is the
    normal-scale convention, not the degenerate integral.)

    Parameters
    ----------
    y_pred, y_true : array-like, pd.DataFrame or pd.Series
        The point forecast and the observation; one value per row, the
        same rows.

    Returns
    -------
    float
        The point forecast's CRPS reference = ``sqrt(pi/2) * MAE`` over
        the finite rows, in the data's units.
    """
    p = (
        y_pred.to_numpy(dtype=float)
        if isinstance(y_pred, (pd.DataFrame, pd.Series))
        else np.asarray(y_pred, dtype=float)
    )
    if p.ndim != 1:
        raise DataError("The point forecast must be one value per row.")
    o = _obs_array(y_true)
    if len(p) != len(o):
        raise DataError(f"The forecast has {len(p)} rows and the observation {len(o)}.")
    mask = np.isfinite(p) & np.isfinite(o)
    if not mask.any():
        raise DataError("Nothing to score: no finite forecast-observation row.")
    return _POINT_MASS_FACTOR * float(np.mean(np.abs(o[mask] - p[mask])))
