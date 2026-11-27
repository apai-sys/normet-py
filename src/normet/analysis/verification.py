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

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..core.exceptions import ConfigError, DataError
from ..utils._config import DEFAULT_SEED

__all__ = [
    "pinball_loss",
    "crps_from_quantiles",
    "crps_from_samples",
    "crps_observed",
    "fair_crps_from_samples",
    "crps_skill_score",
    "interval_score",
    "interval_coverage",
    "pit_diagnostics",
    "calibration_curve",
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
    the largest, and the score is the integral of ``(F - 1{x >= y})^2``.
    On the break-point grid of the distinct values and y the integrand is
    a quadratic per interval -- the indicator is constant on each, since
    y is one of the break points -- and the pieces are integrated
    exactly, tails and off-span observations included.

    A value carrying several levels is a flat stretch of the quantile
    function, and therefore a *jump* of the CDF from the lowest level at
    that value to the highest: the interval below it rises to the lowest
    level, the interval above starts from the highest. Keeping only the
    highest (as an earlier version did) lost the lower level entirely and
    understated the score of any forecast whose quantiles tied.
    """
    order = np.argsort(vs, kind="stable")
    vs, tv = vs[order], tv[order].astype(float)
    v, inv = np.unique(vs, return_inverse=True)  # the distinct values
    m = v.size
    lo = np.full(m, np.inf)
    hi = np.zeros(m)
    np.minimum.at(lo, inv, tv)  # a value group's levels: its lowest ...
    np.maximum.at(hi, inv, tv)  # ... and its highest
    grid = np.unique(np.concatenate([v, [y]]))
    pos = np.searchsorted(v, grid, side="right") - 1  # largest v <= grid

    # The CDF's one-sided limits at each break point. 0 below the smallest
    # value and 1 above the largest; at a data value, a jump from lo to hi
    # (the smallest value jumps from 0; the largest jumps to 1); strictly
    # between two values, the straight line the inverse of the piecewise-
    # linear quantile function gives, from the lower value's highest level
    # to the upper value's lowest.
    left = np.zeros(grid.size)
    right = np.zeros(grid.size)
    below = grid < v[0]
    above = grid > v[-1]
    at_min = grid == v[0]
    at_max = grid == v[-1]
    between = ~(below | above | at_min | at_max)
    k = np.clip(pos, 0, max(m - 2, 0))
    denom = v[k + 1] - v[k] if m >= 2 else np.zeros(k.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(denom > 0, (grid - v[k]) / np.where(denom > 0, denom, 1.0), 0.0)
    line = hi[k] + t * (lo[k + 1] - hi[k]) if m >= 2 else np.zeros(grid.size)
    at_value = np.isin(grid, v)
    idx = np.clip(pos, 0, m - 1)
    left[between] = np.where(at_value[between], lo[idx[between]], line[between])
    right[between] = np.where(at_value[between], hi[idx[between]], line[between])
    if m == 1:
        # One value carrying every level: the forecast is a point mass,
        # and the CDF is a step (0 below it, 1 at and above it).
        left[at_min] = 0.0
        right[at_min] = 1.0
    else:
        right[at_min] = hi[0]
        left[at_max] = lo[-1]
        right[at_max] = 1.0
    left[above] = 1.0
    right[above] = 1.0

    a = right[:-1]  # each interval's CDF at its left end
    b = left[1:]  # and its right end
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


def _crps_rows_rust(q: np.ndarray, levels: list[float], y: np.ndarray) -> float:
    """The same score as :func:`_crps_rows`, through the optional kernel.

    The row loop -- a Python iteration building about twenty temporary
    arrays per row -- is what a normalise run's scoring pass spends its time
    in; the kernel repeats the arithmetic natively with one scratch buffer.
    The number is bit-identical: the kernel also reproduces numpy's pairwise
    summation for each row's segment integral and accumulates the rows in the
    same order. Requires ``normet_rust`` (``rust/normet-python``).
    """
    from ..utils._rust import rust_core

    core = rust_core()
    return float(
        core.crps_rows(
            np.ascontiguousarray(q, dtype=float),
            np.ascontiguousarray(levels, dtype=float),
            np.ascontiguousarray(y, dtype=float),
        )
    )


def _check_backend(backend: str) -> None:
    """Reject an unknown backend with the two names this module supports."""
    if backend not in ("python", "rust"):
        raise ConfigError(f"`backend` must be 'python' or 'rust', got {backend!r}.")


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
    *,
    backend: str = "python",
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
    backend : {"python", "rust"}, default "python"
        ``"rust"`` runs the same integration through the optional
        ``normet_rust`` extension, moving the per-row loop out of Python;
        the score is bit-identical (the kernel reproduces numpy's pairwise
        summation). The default path is untouched.

    Returns
    -------
    float
        The CRPS over the finite rows, in the data's units.
    """
    _check_backend(backend)
    p, lev = _levels_and_values(preds, quantiles)
    o = _obs_array(obs)
    if p.shape[0] != len(o):
        raise DataError(f"The quantiles have {p.shape[0]} rows and the observation {len(o)}.")
    mask = np.isfinite(p).all(axis=1) & np.isfinite(o)
    if not mask.any():
        raise DataError("Nothing to score: no finite forecast-observation row.")
    if backend == "rust":
        return _crps_rows_rust(p[mask], lev, o[mask])
    return _crps_rows(p[mask], lev, o[mask])


def crps_from_samples(
    samples: pd.DataFrame | np.ndarray[Any, Any] | list,
    obs: Any,
    *,
    backend: str = "python",
) -> float:
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
    backend : {"python", "rust"}, default "python"
        ``"rust"`` runs the same integration through the optional
        ``normet_rust`` extension, moving the per-row loop out of Python;
        the score is bit-identical. The default path is untouched.

    Returns
    -------
    float
        The CRPS over the finite rows, in the data's units.
    """
    _check_backend(backend)
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
    if backend == "rust":
        return _crps_rows_rust(s, levels, o[mask])
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


def _sample_matrix_and_obs(
    samples: pd.DataFrame | np.ndarray[Any, Any] | list,
    obs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate an ensemble sample table ``(rows, K)`` and align finite rows with *obs*."""
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
        raise DataError("Ensemble diagnostics require at least 2 samples per row.")
    o = _obs_array(obs)
    if s.shape[0] != len(o):
        raise DataError(f"The samples have {s.shape[0]} rows and the observation {len(o)}.")
    mask = np.isfinite(s).all(axis=1) & np.isfinite(o)
    if not mask.any():
        raise DataError("Nothing to score: no finite sample-observation row.")
    return s[mask], o[mask]


def fair_crps_from_samples(samples: pd.DataFrame | np.ndarray[Any, Any] | list, obs: Any) -> float:
    """Unbiased Fair CRPS (Ferro 2014) of ensemble samples against observations.

    Uses ``1 / (K * (K - 1))`` for within-ensemble pairwise spread instead of
    the biased empirical ``1 / K^2`` scaling, removing the finite-ensemble-size
    penalty when comparing ensembles of different member counts *K*.
    """
    s_clean, o_clean = _sample_matrix_and_obs(samples, obs)
    k = s_clean.shape[1]
    s_sorted = np.sort(s_clean, axis=1, kind="stable")
    term_obs = np.mean(np.abs(s_sorted - o_clean[:, None]), axis=1)
    weights = 2.0 * np.arange(1, k + 1, dtype=float) - k - 1.0
    term_pair = (s_sorted @ weights) / float(k * (k - 1))
    return float(np.mean(term_obs - term_pair))


def crps_skill_score(crps_model: float, crps_reference: float) -> float:
    """Compute Continuous Ranked Probability Skill Score ``1 - CRPS_model / CRPS_ref``."""
    cm = float(crps_model)
    cr = float(crps_reference)
    if not np.isfinite(cm) or not np.isfinite(cr):
        raise DataError("Both `crps_model` and `crps_reference` must be finite floats.")
    if cr <= 0.0:
        raise DataError(f"`crps_reference` must be strictly positive, got {crps_reference!r}.")
    return float(1.0 - cm / cr)


def interval_score(lower: Any, upper: Any, obs: Any, alpha: float = 0.1) -> float:
    """Proper Winkler / Gneiting-Raftery (2007) interval score for ``(1 - alpha)`` intervals."""
    level = _as_quantile_level(alpha, "alpha")
    lo = (
        lower.to_numpy(dtype=float)
        if isinstance(lower, (pd.DataFrame, pd.Series))
        else np.asarray(lower, dtype=float)
    ).reshape(-1)
    hi = (
        upper.to_numpy(dtype=float)
        if isinstance(upper, (pd.DataFrame, pd.Series))
        else np.asarray(upper, dtype=float)
    ).reshape(-1)
    o = _obs_array(obs)
    if not (len(lo) == len(hi) == len(o)):
        raise DataError(
            f"Length mismatch among lower ({len(lo)}), upper ({len(hi)}), and obs ({len(o)})."
        )
    mask = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(o)
    if not mask.any():
        raise DataError("Nothing to score: no finite interval-observation row.")
    lm, hm, om = lo[mask], hi[mask], o[mask]
    if np.any(lm > hm):
        raise DataError("Lower interval bound exceeds upper bound.")
    penalty = (2.0 / level) * (np.where(om < lm, lm - om, 0.0) + np.where(om > hm, om - hm, 0.0))
    return float(np.mean((hm - lm) + penalty))


def interval_coverage(
    preds: pd.DataFrame | None = None,
    obs: Any = None,
    *,
    lower: Any = None,
    upper: Any = None,
    alpha: float | None = None,
    pairs: Sequence[tuple[str, str, float]] | None = None,
) -> pd.DataFrame:
    """Evaluate prediction interval empirical coverage, sharpness, and Winkler score.

    Accepts either explicit ``(lower, upper)`` arrays or a prediction DataFrame
    with ``qNNN`` columns (automatically pairing symmetric lower/upper quantiles)
    or explicit ``pairs=[(lo_col, hi_col, alpha), ...]``.
    """
    o = _obs_array(obs)
    rows: list[dict[str, Any]] = []

    def _eval_one(
        name: str, lo_arr: np.ndarray, hi_arr: np.ndarray, a_val: float | None
    ) -> dict[str, Any]:
        mask = np.isfinite(lo_arr) & np.isfinite(hi_arr) & np.isfinite(o)
        if not mask.any():
            raise DataError(f"Interval {name!r} has no finite rows to evaluate.")
        lm, hm, om = lo_arr[mask], hi_arr[mask], o[mask]
        if np.any(lm > hm):
            raise DataError(f"Lower bound exceeds upper bound in interval {name!r}.")
        emp = float(np.mean((om >= lm) & (om <= hm)))
        widths = hm - lm
        if a_val is not None:
            nom = float(1.0 - float(a_val))
            cov_err = emp - nom
            iscore = interval_score(lm, hm, om, alpha=float(a_val))
        else:
            nom = float("nan")
            cov_err = float("nan")
            iscore = float("nan")
        return {
            "interval": name,
            "nominal_coverage": nom,
            "empirical_coverage": emp,
            "coverage_error": cov_err,
            "mean_width": float(np.mean(widths)),
            "median_width": float(np.median(widths)),
            "interval_score": iscore,
            "n": int(mask.sum()),
        }

    if lower is not None and upper is not None:
        lo_a = np.asarray(lower, dtype=float).reshape(-1)
        hi_a = np.asarray(upper, dtype=float).reshape(-1)
        if len(lo_a) != len(o) or len(hi_a) != len(o):
            raise DataError("`lower`, `upper`, and `obs` must have the same length.")
        if alpha is not None:
            _as_quantile_level(alpha, "alpha")
        rows.append(_eval_one("custom", lo_a, hi_a, alpha))
    else:
        if not isinstance(preds, pd.DataFrame):
            raise DataError("Pass a DataFrame to `preds` or explicit `lower`/`upper` arrays.")
        if len(preds) != len(o):
            raise DataError(f"`preds` has {len(preds)} rows and `obs` has {len(o)}.")
        spec_pairs: list[tuple[str, str, float]] = []
        if pairs is not None:
            spec_pairs = [
                (str(lo_c), str(hi_c), _as_quantile_level(a_c, "alpha"))
                for lo_c, hi_c, a_c in pairs
            ]
        else:
            q_cols = sorted(
                (
                    (_column_level(c), str(c))
                    for c in preds.columns
                    if isinstance(c, str) and c.startswith("q") and c[1:].isdigit()
                ),
                key=lambda item: item[0],
            )
            by_level = {round(lvl, 6): col for lvl, col in q_cols}
            used: set[str] = set()
            for lvl, lo_col in q_cols:
                if lo_col in used or lvl >= 0.5:
                    continue
                mate_lvl = round(1.0 - lvl, 6)
                hi_col = by_level.get(mate_lvl)
                if hi_col is not None and hi_col != lo_col:
                    used.update({lo_col, hi_col})
                    spec_pairs.append((lo_col, hi_col, float(2.0 * lvl)))
        if not spec_pairs:
            raise DataError("No symmetric quantile pairs or interval bounds found to evaluate.")
        for lo_col, hi_col, a_c in spec_pairs:
            for col in (lo_col, hi_col):
                if col not in preds.columns:
                    raise DataError(f"Column {col!r} not found in `preds`.")
            lo_v = pd.to_numeric(preds[lo_col], errors="coerce").to_numpy(dtype=float)
            hi_v = pd.to_numeric(preds[hi_col], errors="coerce").to_numpy(dtype=float)
            rows.append(_eval_one(f"{lo_col}..{hi_col}", lo_v, hi_v, a_c))

    return pd.DataFrame(rows)


def pit_diagnostics(
    samples: pd.DataFrame | np.ndarray[Any, Any] | list,
    obs: Any,
    *,
    n_bins: int = 10,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Compute randomized Probability Integral Transform (PIT) uniformity diagnostics."""
    from scipy import stats as _stats

    if int(n_bins) < 2:
        raise DataError("`n_bins` must be at least 2.")
    s_clean, o_clean = _sample_matrix_and_obs(samples, obs)
    n, k = s_clean.shape

    rng = np.random.default_rng(seed)
    c_lt = np.sum(s_clean < o_clean[:, None], axis=1).astype(float)
    c_eq = np.sum(s_clean == o_clean[:, None], axis=1).astype(float)
    u = rng.uniform(0.0, 1.0, size=n)
    pit_vals = np.clip((c_lt + u * (c_eq + 1.0)) / float(k + 1), 0.0, 1.0)

    counts, edges = np.histogram(pit_vals, bins=int(n_bins), range=(0.0, 1.0))
    mean_p = float(np.mean(pit_vals))
    std_p = float(np.std(pit_vals, ddof=1)) if n > 1 else 0.0
    ks_res = _stats.kstest(pit_vals, "uniform")
    ks_stat = float(ks_res.statistic)
    ks_pval = float(ks_res.pvalue)

    uniform_std = float(1.0 / np.sqrt(12.0))
    if abs(mean_p - 0.5) > 0.08:
        diag = "biased"
    elif std_p > uniform_std + 0.035 and ks_pval < 0.05:
        diag = "under_dispersed"
    elif std_p < uniform_std - 0.035 and ks_pval < 0.05:
        diag = "over_dispersed"
    elif ks_pval >= 0.05 or (abs(mean_p - 0.5) <= 0.05 and abs(std_p - uniform_std) <= 0.03):
        diag = "well_calibrated"
    else:
        diag = "under_dispersed" if std_p > uniform_std else "over_dispersed"

    return {
        "pit": pit_vals,
        "counts": counts,
        "edges": edges,
        "mean": mean_p,
        "std": std_p,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pval,
        "diagnosis": diag,
    }


def calibration_curve(
    y_pred: Any,
    y_true: Any,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Compute binned prediction reliability and conditional-bias diagnostics."""
    if int(n_bins) < 2:
        raise DataError("`n_bins` must be at least 2.")
    if strategy not in ("quantile", "uniform"):
        raise DataError(f"`strategy` must be 'quantile' or 'uniform', got {strategy!r}.")

    p = (
        y_pred.to_numpy(dtype=float)
        if isinstance(y_pred, (pd.DataFrame, pd.Series))
        else np.asarray(y_pred, dtype=float)
    ).reshape(-1)
    o = _obs_array(y_true)
    if len(p) != len(o):
        raise DataError(f"The forecast has {len(p)} rows and the observation {len(o)}.")
    mask = np.isfinite(p) & np.isfinite(o)
    if not mask.any():
        raise DataError("Nothing to score: no finite forecast-observation row.")
    pm, om = p[mask], o[mask]

    nb = int(n_bins)
    if strategy == "quantile":
        edges = np.unique(np.quantile(pm, np.linspace(0.0, 1.0, nb + 1)))
    else:
        edges = np.unique(np.linspace(float(pm.min()), float(pm.max()), nb + 1))

    if len(edges) < 2:
        return pd.DataFrame(
            [
                {
                    "bin": 0,
                    "n": int(len(pm)),
                    "pred_mean": float(np.mean(pm)),
                    "obs_mean": float(np.mean(om)),
                    "bias": float(np.mean(pm - om)),
                    "rmse": float(np.sqrt(np.mean((pm - om) ** 2))),
                    "obs_p10": float(np.percentile(om, 10.0)),
                    "obs_p90": float(np.percentile(om, 90.0)),
                }
            ]
        )

    bin_idx = np.clip(np.searchsorted(edges, pm, side="left") - 1, 0, len(edges) - 2)
    rows: list[dict[str, Any]] = []
    for k in range(len(edges) - 1):
        sel = bin_idx == k
        if not np.any(sel):
            continue
        p_b, o_b = pm[sel], om[sel]
        rows.append(
            {
                "bin": k,
                "n": int(sel.sum()),
                "pred_mean": float(np.mean(p_b)),
                "obs_mean": float(np.mean(o_b)),
                "bias": float(np.mean(p_b - o_b)),
                "rmse": float(np.sqrt(np.mean((p_b - o_b) ** 2))),
                "obs_p10": float(np.percentile(o_b, 10.0)),
                "obs_p90": float(np.percentile(o_b, 90.0)),
            }
        )
    return pd.DataFrame(rows)
