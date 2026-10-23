# src/normet/foundation/calibration.py
"""Quantile calibration for the zero-shot de-weathering's bands.

The probabilistic output of the chronos-2 path is its native quantile
forecast over resampled weather (see :func:`normet.gui._chronos.deweather`
and ``Chronos2Estimator.deweather``): a band that is wide or narrow by the
model's training habit, not because the record is noisy. A policy effect is
compared against that band, so the band's nominal coverage is exactly what
must be true before the comparison means anything, and nothing in the
zero-shot path checks it. The de-weathered columns' docstring already
admits it: those are forecast quantiles, not checked confidence bands.

:func:`calibrate_quantiles` checks it on the record's own history, under
the same conditioning the counterfactual is made under -- the future
meteorology is known-future covariate there. Rolling hold-out blocks each
see the model's quantile forecast of the block, and the calibration works
in the one quantity a quantile correction can act on: the error,
observation minus the block's median forecast, measured in the block's own
reference span (its 5th-to-95th percentile half-width). The correction is
symmetric in that reference and per band level: the band's both edges go
to the median plus or minus the fitted multiple of the span. That multiple
is the split-conformal threshold of the held-out errors at the level's
nominal rate, made non-decreasing in the level (the PAVA leg,
``method="isotonic"`` -- a corrected band's edges must not cross).

Honesty is the design. :class:`CalibratedEstimator` carries the
``coverage`` table -- per level, the nominal rate, the empirical rate on
the held-out under the model's own band, and the empirical rate under the
corrected one, cross-validated: each block is scored by the thresholds
fitted without it (the fit-time alternative would read good on the
correction's training data and be worthless). Where a corrected band's
cross-validated coverage still deviates from the nominal by more than ten
points, the fit is returned anyway with a
:class:`~normet.exceptions.CalibrationWarning`: the record's behaviour went
beyond the hold-out, and the warning is what keeps a band that would only
look calibrated out of a paper.

The estimator is duck-typed on what the calibration calls: its
``predict_quantiles(df, target, anchor, *, horizon, quantiles, covariates)``
and its ``prediction_length`` / ``context_length``. The chronos-2 estimator
and the test stubs are alike there, so the tests keep their promise without
torch.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import CalibrationWarning, ConfigError

log = logging.getLogger(__name__)

__all__ = [
    "CalibratedEstimator",
    "QuantileCalibration",
    "calibrate_quantiles",
]

#: The cross-validated coverage deviation from the nominal that earns a
#: :class:`~normet.exceptions.CalibrationWarning`: ten points is where a
#: band stops being the band it claims to be.
DEFAULT_TOLERANCE = 0.10

#: Fewer held-out error multiples per band level than this, and the fit is
#: a sample of one: the fit falls to the plain per-level thresholds and
#: names itself ``constant`` in the result, which the table carries.
_MIN_MULTIPLES_PER_LEVEL = 30

#: The reference half-width the errors are measured in: the model's own
#: 5th-to-95th percentile span, halved, at each forecast step.
_SPAN = (0.05, 0.95)

#: The coverage table's columns, in order (the table's schema, the name a
#: paper quotes under).
COVERAGE_COLUMNS = (
    "level",
    "nominal_coverage",
    "empirical_uncalibrated",
    "empirical_calibrated",
    "abs_deviation",
)


def _round_q(level: float) -> float:
    """A quantile level as the estimator will name it.

    ``predict_quantiles`` names its columns ``q{level}`` with the level as
    Python writes it; a derived level such as ``1.0 - 0.9``
    (``0.09999999999999998``) would earn a name the calibration never
    reads. Rounding to six significant digits is what both sides agree on.
    """
    return float(f"{level:.6g}")


def _col(level: float) -> str:
    """The estimator's column name for a quantile level.

    ``predict_quantiles`` names its columns ``q{level}`` with the level
    Python writes it (``q0.05``, ``q0.5``); the names must round-trip or
    the calibration reads the wrong edge.
    """
    return f"q{float(f'{level:g}')}"


def _band_levels(quantiles: Sequence[float]) -> list[float]:
    """The bands above the median, each named by its upper edge.

    A requested level below the median is a band's lower edge, corrected
    with its upper one: the name fit and scored is the level whose upper
    edge it is, and 0.5 is the centre, not a band.
    """
    return sorted(_round_q(q) for q in quantiles if q > 0.5)


def _conformal_threshold(errors: np.ndarray, nominal: float) -> float:
    """The split-conformal half-width multiple: the ``ceil((n+1) nominal)``-th
    smallest of the held-out error multiples.

    One step past the sample quantile of the multiples would give. At small
    holds-out that step is the correction's honesty -- a sample quantile of
    the errors the band will be scored on reads its own training data, and
    the corrected band's coverage of that data would be good forever and
    worthless always.
    """
    if errors.size == 0:
        return 1.0
    order = int(np.ceil((errors.size + 1) * nominal))
    order = min(max(order, 1), errors.size)
    return float(np.sort(np.abs(errors))[order - 1])


def _pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pool adjacent violators: the non-decreasing fit of ``values`` weighted by
    ``weights``. The isotone leg of the correction, without scikit-learn --
    a corrected band's edges must not cross, and the fit that enforces it is
    the isotone one."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = values.size
    blocks: list[list[float]] = []  # each: [weighted sum, total weight]
    for v, w in zip(values, weights, strict=False):
        blocks.append([float(v * w), float(w)])
        # A new block lower than its predecessor breaks the non-decreasing
        # order: merge the two until the order holds again.
        while len(blocks) > 1 and blocks[-1][0] / blocks[-1][1] < blocks[-2][0] / blocks[-2][1]:
            s1, w1 = blocks.pop()
            s0, w0 = blocks.pop()
            blocks.append([s0 + s1, w0 + w1])
    out = np.empty(n)
    filled = 0
    for s, w in blocks:
        size = int(round(w))
        out[filled : filled + size] = s / w
        filled += size
    return out


@dataclass
class QuantileCalibration:
    """The fitted correction and the coverage table that earns its trust.

    ``threshold`` is the multiple of the reference span each band level's
    edges get (the split-conformal threshold of the held-out error multiples
    at the level's nominal rate, isotone in the levels);
    ``half_width_factor`` carries the same numbers under the name a paper
    would call them. ``coverage`` is the table a paper should quote: per
    level, the nominal rate, the empirical rate on the held-out under the
    model's own band, the empirical rate under the corrected one
    (cross-validated), and the absolute deviation of the corrected from the
    nominal. ``warnings`` names the levels whose corrected coverage
    deviated past the fit's tolerance: the empty tuple is the only ending
    the fit calls its own.
    """

    levels: tuple[float, ...]
    threshold: list[float]
    half_width_factor: list[float]
    coverage: pd.DataFrame
    warnings: tuple[str, ...]
    method: str
    n_points: int
    n_blocks: int

    def __str__(self) -> str:  # pragma: no cover - the fit's own words
        bad = "none" if not self.warnings else ", ".join(self.warnings)
        return (
            f"quantile calibration ({self.method}): {self.n_points} held-out errors in "
            f"{self.n_blocks} blocks over {len(self.levels)} band levels; levels off "
            f"target: {bad}"
        )


def _fit_pass(
    errors: list[np.ndarray],
    bands: list[float],
    scored: set[int],
    method: str,
) -> list[float]:
    """The thresholds one scoring pass fits: from every held-out block's error
    multiples except the ones it scores (the conformal way), one per band
    level, isotone in the level where the method asks.

    ``errors[j]`` is block ``j``'s error multiples, one per forecast step.
    A pool too thin to trust keeps the plain conformal threshold of
    whatever the pool holds; the fit's thinness is what ``method`` names in
    the result.
    """
    fit = [j for j in range(len(errors)) if j not in scored] or list(range(len(errors)))
    raw: list[float] = []
    pools: list[np.ndarray] = []
    nominals = [2.0 * (q - 0.5) for q in bands]
    for li in range(len(bands)):
        pool = np.concatenate([errors[j][:, li] for j in fit])
        pools.append(pool)
        raw.append(_conformal_threshold(pool, nominals[li]))
    if method == "constant" or all(p.size < _MIN_MULTIPLES_PER_LEVEL for p in pools):
        return list(raw)
    weights = np.array([p.size + 1 for p in pools], dtype=float)
    fixed = _pava(np.asarray(raw, dtype=float), weights)
    return [float(t) for t in fixed]


def calibrate_quantiles(
    estimator: Any,
    df: pd.DataFrame,
    target: str,
    anchor: pd.Timestamp | str,
    *,
    quantiles: Sequence[float] = (0.1, 0.5, 0.9),
    n_blocks: int | None = None,
    method: str = "isotonic",
    tolerance: float = DEFAULT_TOLERANCE,
    covariates: list[str] | None = None,
) -> CalibratedEstimator:
    """Calibrate the estimator's quantile bands and hand back the corrected estimator.

    The estimator is duck-typed on what the calibration calls (its
    ``predict_quantiles`` and its ``prediction_length`` /
    ``context_length``), so a chronos-2 estimator and a deterministic stub
    are alike; the fit is the calibration's, not the model's. See the
    module docstring for what fit and correction mean here.

    Parameters
    ----------
    estimator
        An object whose ``predict_quantiles(df, target, anchor, *,
        horizon, quantiles, covariates)`` returns a frame of ``q{level}``
        columns over the forecast block, carrying ``prediction_length``
        and ``context_length``.
    df
        The record, indexed by its own time (a :class:`~pandas.DatetimeIndex`),
        holding ``target`` and the covariates.
    target
        The target column the calibration scores against.
    anchor
        Where the corrected forecasts start: the held-out blocks are the
        ones directly before it; everything from it on is forecast,
        calibrated.
    quantiles
        The band levels the corrected estimator must carry, the median in
        the middle. Each distinct pair of edges above the median is a band,
        corrected under the level of its upper edge (the default
        ``(0.1, 0.5, 0.9)`` corrects the 90 % band, one of the
        de-weathering's own).
    n_blocks
        How many held-out blocks the span before the anchor splits into, each
        one forecast block of ``prediction_length`` steps (default: what the
        record gives, up to seven).
    method
        ``isotonic`` (the default): each level's conformal threshold through
        the PAVA non-decreasing leg, so a corrected band's edges cannot
        cross. ``constant``: the plain per-level thresholds, the fit the
        record names as such where it is too short for one.
    tolerance
        The cross-validated coverage deviation from the nominal that earns a
        :class:`~normet.exceptions.CalibrationWarning` on return.
    covariates
        The estimator's covariate set for the fit's forecasts (the chronos-2
        estimator resolves it itself without one).

    Returns
    -------
    CalibratedEstimator
        The estimator with its quantile paths rescaled by the fitted
        thresholds; its ``coverage`` the table a paper quotes, its
        ``calibration`` the fit's words.

    Raises
    ------
    ConfigError
        Where the levels miss the median, the record holds no room for the
        span, or the estimator gives no usable spread to score against.
    """
    if len(quantiles) < 3 or 0.5 not in list(quantiles):
        raise ConfigError(
            f"`quantiles` must be three or more levels with the median in the middle: "
            f"got {list(quantiles)!r}."
        )
    if list(quantiles) != sorted(quantiles) or quantiles[0] <= 0.0 or quantiles[-1] >= 1.0:
        raise ConfigError(
            f"`quantiles` must be strictly increasing inside (0, 1): {list(quantiles)!r}."
        )
    if method not in ("isotonic", "constant"):
        raise ConfigError(f"`method` is 'isotonic' or 'constant', got {method!r}.")
    if not hasattr(estimator, "predict_quantiles"):
        raise ConfigError(
            "the estimator has no predict_quantiles to calibrate; "
            "load one (normet.foundation.Chronos2Estimator)."
        )
    bands = _band_levels(quantiles)
    if not bands:
        raise ConfigError(f"`quantiles` names no band above the median: {list(quantiles)!r}.")

    index = df.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ConfigError("`df` must be indexed by its own time (a DatetimeIndex).")
    if target not in df.columns:
        raise ConfigError(f"`df` has no column {target!r} to score against.")

    horizon = int(getattr(estimator, "prediction_length", 168))
    context = int(getattr(estimator, "context_length", 2048))
    anchor = pd.Timestamp(anchor)
    if len(index) < context + horizon:
        raise ConfigError(
            f"the record holds no room for a held-out span: {len(index)} rows, and the "
            f"estimator's context alone is {context}."
        )

    # The held-out blocks: whole forecast blocks of `horizon` steps, back from
    # the anchor, each with its context of rows behind it. The step is the
    # record's own (an hour, a day, a month -- the estimator's horizon is in
    # steps, not seconds).
    step = index[1] - index[0]
    if n_blocks is None:
        n_blocks = min(7, max(1, int((anchor - index[0]) / step) // horizon or 1))
    blocks: list[pd.Timestamp] = []  # forecast anchors: each block's start
    end = anchor
    for _ in range(int(n_blocks)):
        start = end - step * int(horizon)
        if (index < start).sum() < context:
            break
        blocks.append(start)  # the block [start, start + horizon) is forecast from start
        end = start
    if not blocks:
        raise ConfigError(
            f"the record holds no room for a held-out span before {anchor:%Y-%m-%d}: "
            f"context {context} plus a block of {horizon} steps does not fit."
        )
    blocks.reverse()  # old to young

    # The levels one forecast is asked for: the span, the median, the bands' own
    # pairs.
    ask = sorted(
        {_round_q(0.5), _round_q(float(_SPAN[0])), _round_q(float(_SPAN[1]))}
        | set(bands)
        | {_round_q(1.0 - b) for b in bands}
    )
    lo_c, hi_c = _col(float(_SPAN[0])), _col(float(_SPAN[1]))
    errors: list[np.ndarray] = []  # per block: (steps, n_bands) error multiples
    model_edges: list[
        list[tuple[float, float]]
    ] = []  # per block: per band, the model's own (lo, hi) in multiples
    for b in blocks:
        try:
            frame = estimator.predict_quantiles(
                df, target, b, horizon=horizon, quantiles=ask, covariates=covariates
            )
        except ValueError as exc:
            if not errors:
                raise ConfigError(
                    f"the held-out span begins before the record can give the context "
                    f"the estimator needs ({exc})."
                ) from exc
            log.warning("calibration: a held-out block gave no forecast (%s); skipped.", exc)
            continue
        end_ts = b + step * int(horizon)
        obs = df.loc[(index >= b) & (index < end_ts), target].to_numpy(dtype=float)
        if obs.size != horizon:
            break  # a record that truncates mid-block: the block is not whole
        half = (frame[hi_c] - frame[lo_c]) / 2.0
        med = frame[_col(0.5)]
        half_u, med_u = half.to_numpy(), med.to_numpy()
        finite = np.isfinite(half_u) & (half_u > 0.0)
        if int(finite.sum()) < 3:
            raise ConfigError(
                f"the held-out block {b:%Y-%m-%d} has no usable spread "
                f"({int(finite.sum())} of {horizon} steps); the record holds nothing "
                "to calibrate against there."
            )
        e = (obs[finite] - med_u[finite]) / half_u[finite]
        row_e = np.repeat(
            e[:, None], len(bands), axis=1
        )  # the same error in its own units, per band
        edges: list[tuple[float, float]] = []
        for li, q in enumerate(bands):
            lo_b_u, hi_b_u = (
                frame[_col(1.0 - q)].to_numpy()[finite],
                frame[_col(q)].to_numpy()[finite],
            )
            c_lo = float(np.median((lo_b_u - med_u[finite]) / half_u[finite]))
            c_hi = float(np.median((hi_b_u - med_u[finite]) / half_u[finite]))
            edges.append((c_lo, c_hi))
        errors.append(row_e)
        model_edges.append(edges)
    n_points = int(sum(e.shape[0] for e in errors))
    if n_points < 6:
        raise ConfigError(
            f"the held-out span gave {n_points} scored errors; the calibration needs "
            "more than two blocks of usable spread."
        )
    if method == "isotonic" and (n_points // len(bands)) < _MIN_MULTIPLES_PER_LEVEL:
        method = "constant"  # a fit of one sample per level is not a split: the table says so

    # The model's own coverage of each of its bands, on the same held-out
    # errors (independent of the scoring pass: it scores the model, not the fit).
    raw_hits = [0 for _ in bands]
    for j in range(len(errors)):
        e = errors[j][:, 0] if errors[j].shape[1] else np.array([])
        for li, (c_lo, c_hi) in enumerate(model_edges[j]):
            raw_hits[li] += int(np.sum((e >= c_lo) & (e <= c_hi)))

    # The corrected coverage, cross-validated: each scoring pass scores its
    # blocks with thresholds fitted without them.
    k = max(1, len(errors) // 7)
    passes = [set(range(s, min(s + k, len(errors)))) for s in range(0, len(errors), k)]
    cv_hits = [0 for _ in bands]
    for scored in passes:
        th = _fit_pass(errors, bands, scored, method)
        for j in sorted(scored):
            m = np.abs(errors[j][:, 0]) if errors[j].shape[1] else np.array([])
            for li in range(len(bands)):
                cv_hits[li] += int(np.sum(m <= th[li]))

    total = n_points
    rows = []
    for li, q in enumerate(bands):
        nominal = float(2.0 * (q - 0.5))
        raw_rate = float(raw_hits[li] / total) if total else float("nan")
        cv_rate = float(cv_hits[li] / total) if total else float("nan")
        rows.append([float(q), nominal, raw_rate, cv_rate, abs(cv_rate - nominal)])
    coverage = pd.DataFrame(rows, columns=list(COVERAGE_COLUMNS))

    thresholds = _fit_pass(errors, bands, set(), method)
    names = [
        f"q{float(f'{q:g}')} ({cv:.0%} against the nominal {nom:.0%})"
        for (q, nom, cv) in zip(
            bands, coverage["nominal_coverage"], coverage["empirical_calibrated"], strict=False
        )
        if (abs(cv - nom) if np.isfinite(cv) else float("inf")) > tolerance
    ]
    calibration = QuantileCalibration(
        levels=tuple(bands),
        threshold=list(thresholds),
        half_width_factor=list(thresholds),
        coverage=coverage,
        warnings=tuple(names),
        method=method,
        n_points=n_points,
        n_blocks=len(errors),
    )
    est = CalibratedEstimator(estimator, calibration=calibration, quantiles=tuple(quantiles))
    if calibration.warnings:
        warnings.warn(
            f"calibrated band coverage deviates from the nominal: {', '.join(calibration.warnings)}. "
            "The correction was fit, but the record's behaviour went beyond the hold-out; "
            "the band is returned corrected and labelled -- quote the coverage table with it.",
            CalibrationWarning,
            stacklevel=2,
        )
    return est


class CalibratedEstimator:
    """The estimator with its quantile paths through the calibration.

    :meth:`predict_quantiles` rescales a forecast block's band edges around
    the block's own median and in the block's own reference span: each
    corrected edge is the median plus or minus the fitted threshold's
    multiple of the span. The move is monotone in the span the threshold
    was fit on -- the fit could act on that quantity, and the correction
    acting on it is the same move. The median, the levels the fit does not
    carry, ``predict``, and everything the estimator has that the
    calibration does not shadow (``finetune``, the context lengths, the
    whole rest of its surface) are carried through. ``deweather`` and the
    counterfactual reach the base's own paths and so stay uncorrected until
    the wiring takes them (the counterfactual's ``calibrated`` flag); the
    correction is on :meth:`predict_quantiles`, which is what they call.

    ``coverage`` is the table a paper quotes and ``calibration`` the fit
    behind it; its ``warnings`` are the levels the fit could not bring onto
    target, named.
    """

    def __init__(
        self, base: Any, *, calibration: QuantileCalibration, quantiles: Sequence[float]
    ) -> None:
        self.__dict__["base"] = base
        self.__dict__["calibration"] = calibration
        self.__dict__["coverage"] = calibration.coverage
        self.__dict__["threshold"] = calibration.threshold
        self.__dict__["_bands"] = list(_band_levels(quantiles))
        self.__dict__["_requested"] = list(quantiles)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["base"], name)

    def _span(self, frame: pd.DataFrame) -> pd.Series:
        return (frame[_col(float(_SPAN[1]))] - frame[_col(float(_SPAN[0]))]) / 2.0

    def _rescale(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "q0.5" not in frame.columns or not self._bands:
            return frame
        need = set()
        for b in self._bands:
            need |= {_col(b), _col(1.0 - b)}
        if not need <= set(frame.columns):
            return frame  # a frame its own edges are not in stays as made
        half = self._span(frame)
        if not (half > 0).any():
            return frame
        out = frame.copy()
        for bi, b in enumerate(self._bands):
            t = self.threshold[bi]
            out[_col(b)] = frame[_col(0.5)] + half * t
            out[_col(1.0 - b)] = frame[_col(0.5)] - half * t
        return out

    def predict_quantiles(
        self,
        df: pd.DataFrame,
        target: str,
        anchor: pd.Timestamp | str,
        *,
        horizon: int | None = None,
        quantiles: Any = (0.1, 0.5, 0.9),
        covariates: list[str] | None = None,
    ) -> pd.DataFrame:
        """The estimator's quantile forecast, each corrected band's edges through
        its fitted thresholds; the median and any level the fit does not carry
        come back as the model made them."""
        # The base is asked for everything the rescale needs (its bands' own
        # pairs, the span, the median), returned is only what was requested.
        need = (
            {
                _round_q(0.5),
                _round_q(float(_SPAN[0])),
                _round_q(float(_SPAN[1])),
            }
            | set(self._bands)
            | {_round_q(1.0 - b) for b in self._bands}
        )
        request = list(dict.fromkeys([_round_q(q) for q in quantiles] + sorted(need)))
        out = self.base.predict_quantiles(
            df, target, anchor, horizon=horizon, quantiles=request, covariates=covariates
        )
        out = self._rescale(out)
        keep = [_col(q) for q in dict.fromkeys(list(quantiles)) if _col(q) in out.columns]
        return out[keep]

    def predict(self, X: pd.DataFrame) -> Any:
        """The median path: the estimator's own, uncorrected (the correction
        acts on the band edges; the median is where the band is centred)."""
        return self.base.predict(X)
