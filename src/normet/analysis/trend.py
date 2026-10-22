# src/normet/analysis/trend.py
"""Trend analysis for environmental time-series.

The deliverable of de-weathering is an emission-driven series, and the
question every such analysis must answer next is: *is that series trending,
how fast, and can we say so honestly?* Hourly air-quality records are strongly
autocorrelated (diurnal and weekly structure carried through the model), and a
plain Mann-Kendall test on such a series treats its 8760 hourly values as 8760
independent draws -- inflating the significance of every trend it finds.
This module therefore ships the test with the corrections the data demand:

- :func:`mann_kendall` : S statistic with tie correction (Hirsch et al. 1982),
  including the seasonal variant of Sen (1968) for records that carry a
  repeating calendar structure.
- :func:`theil_sen`    : robust median pair-wise slope with the analytic
  Sen (1968) 95 % confidence interval, in units of *per day* when the input
  is a datetime-indexed series, so hourly, daily and monthly records compare.
- :func:`trend`        : the entry point. Runs both, diagnoses serial
  correlation of the trend residuals, and -- when it finds a problem --
  replaces the slope's error estimate with a moving-block bootstrap (block
  length long enough to keep the calendar structure together) and a Wald
  p-value computed from that bootstrap standard error.

Nothing here resamples the *weather*; this is statistics on the series the
user has already normalised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import DataError
from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "TrendTest",
    "TrendSlope",
    "TrendResult",
    "mann_kendall",
    "theil_sen",
    "trend",
]


@dataclass(frozen=True)
class TrendTest:
    """Result of a Mann-Kendall trend test.

    Attributes
    ----------
    S : int
        The Mann-Kendall S statistic (sum of sign terms over all pairs).
    var_S : float
        Variance of S under H0, with tie correction.
    z : float
        Normalised statistic with continuity correction.
    p : float
        Two-sided p-value.
    n : int
        Number of values used (missing values dropped before testing).
    seasonal : bool, default False
        Whether the seasonal (Sen 1968) variant was used.
    n_periods : int or None
        Number of calendar periods (groups) the seasonal test was run on,
        when ``seasonal`` -- for hourly data with ``seasonal_period=24``
        that is 24 (one group per hour-of-day), not the number of days.
    """

    S: int
    var_S: float
    z: float
    p: float
    n: int
    seasonal: bool = False
    n_periods: int | None = None

    @property
    def direction(self) -> str:
        """``"increasing"`` / ``"decreasing"`` at the 5 % level, else ``"none"``."""
        if self.p < 0.05 and self.S > 0:
            return "increasing"
        if self.p < 0.05 and self.S < 0:
            return "decreasing"
        return "none"


@dataclass(frozen=True)
class TrendSlope:
    """Theil-Sen slope of a series against time.

    ``slope`` is per day when the input carried a datetime index, per index
    step otherwise; ``slope_per_year`` converts it to the analyst's units.
    """

    slope: float
    ci_low: float
    ci_high: float
    n: int
    time_unit: str = "step"
    slope_per_year: float | None = None
    ci_low_per_year: float | None = None
    ci_high_per_year: float | None = None


@dataclass
class TrendResult:
    """Everything ``trend`` computed for one series.

    ``boot_*`` fields are ``None`` unless the AR(1) diagnostic flagged
    serial correlation and the bootstrap ran.
    """

    name: str
    n: int
    n_dropped: int
    test: TrendTest
    slope: TrendSlope
    ar1_acf: float
    ar1_p: float
    ar_autocorrelated: bool
    boot_slope_se: float | None = None
    boot_ci_low: float | None = None
    boot_ci_high: float | None = None
    p_wald_boot: float | None = None
    block_length: int | None = None
    n_bootstrap: int | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        """Tidy two-column summary: metric / value, report-ready."""
        rows = [
            ("n", self.n),
            ("n_missing_dropped", self.n_dropped),
            ("MK_S", self.test.S),
            ("MK_z", round(self.test.z, 3)),
            ("MK_p_two_sided", round(self.test.p, 5)),
            ("direction", self.test.direction),
            (
                "slope_per_day" if self.slope.time_unit == "day" else "slope_per_step",
                self.slope.slope,
            ),
            (
                "slope_95ci",
                f"[{self.slope.ci_low:.4g}, {self.slope.ci_high:.4g}]"
                + (" /day" if self.slope.time_unit == "day" else " /step"),
            ),
            ("ar1_residual_acf", round(self.ar1_acf, 3)),
            ("ar1_p", round(self.ar1_p, 5)),
        ]
        if self.ar_autocorrelated:
            rows += [
                (
                    "bootstrap_SE",
                    None if self.boot_slope_se is None else round(self.boot_slope_se, 4),
                ),
                (
                    "bootstrap_95ci",
                    None
                    if self.boot_ci_low is None
                    else f"[{self.boot_ci_low:.4g}, {self.boot_ci_high:.4g}]",
                ),
                (
                    "p_wald_block_bootstrap",
                    None if self.p_wald_boot is None else round(self.p_wald_boot, 5),
                ),
                ("block_length", self.block_length),
                ("n_bootstrap", self.n_bootstrap),
            ]
        return pd.DataFrame(rows, columns=["metric", "value"])

    def __str__(self) -> str:
        s = self.slope
        unit = "u/m3/day" if s.time_unit == "day" else "u/step"
        line = (
            f"{self.name}: n={self.n}"
            + (f" (dropped {self.n_dropped} missing)" if self.n_dropped else "")
            + f" | MK p={self.test.p:.3g} ({self.test.direction})"
            + f" | slope {s.slope:.4g} {unit}"
            + f" | 95% CI [{s.ci_low:.4g}, {s.ci_high:.4g}]"
        )
        if self.ar_autocorrelated:
            # The AR(1) trigger can fire yet leave no bootstrap -- e.g. more
            # than n_bootstrap replicates are finite when many bootstrap
            # blocks land entirely on a single repeated timestamp (collinear
            # x -> undefined OLS slope, filtered out below). Then report the
            # trigger and say the interval is unavailable rather than
            # formatting None.
            if self.boot_slope_se is not None and self.p_wald_boot is not None:
                line += (
                    f" | AR(1) acf={self.ar1_acf:.2f} -> block bootstrap: "
                    f"SE={self.boot_slope_se:.4g}, p={self.p_wald_boot:.3g}"
                )
            else:
                line += (
                    f" | AR(1) acf={self.ar1_acf:.2f} -> block bootstrap unavailable (see notes)"
                )
        return line


def _to_series(series: pd.Series | pd.DataFrame, value_col: str | None) -> tuple[pd.Series, bool]:
    """Reduce a DataFrame via ``value_col``, or pass a Series through.

    Returns the sorted float series and ``positional``: True when the input
    carries no real timestamps -- a :class:`~pandas.RangeIndex` left as
    0..n-1 (the default when no index is given), or one with a
    :class:`~pandas.Timedelta` frequency, which *is* read as time. Such
    positional records are tested on the 0..n-1 grid and their slopes are
    per *step* -- the series is not assumed hourly, which a blanket
    ``to_datetime`` conversion of positions would silently be.
    """
    if isinstance(series, pd.DataFrame):
        if value_col is None:
            raise ValueError("`value_col` is required when passing a DataFrame.")
        if value_col not in series.columns:
            raise ValueError(f"Column '{value_col}' not in DataFrame.")
        s = series[value_col]
    else:
        s = series
    idx = s.index
    positional = not isinstance(idx, pd.DatetimeIndex) and (
        isinstance(idx, pd.RangeIndex) and not isinstance(getattr(idx, "freq", None), pd.Timedelta)
    )
    s = s.astype(float)
    if not positional:
        if not isinstance(idx, pd.DatetimeIndex):
            s = s.copy()
            s.index = pd.DatetimeIndex(s.index)
        else:
            s = s.copy()
    return s.sort_index(), positional


def _time_grid(s: pd.Series, x: pd.Series, positional: bool) -> np.ndarray:
    """Time grid for the finite values of ``s`` (``x`` is ``s.dropna()``).

    Datetime records are measured in days, so hourly, daily and monthly all
    share one unit. Positional records use the *original step indices* with
    gaps preserved: a missing row is a hole in the grid, not a renumbering of
    what survives -- otherwise a positional record with missing values would
    be re-spaced (and, worse, ``trend()`` reading the post-``dropna`` integer
    index as timestamps would silently treat positions 0..n-1 as nanoseconds
    since the epoch). Keeping the original indices makes "per step" mean
    per *real* step and mirror how the datetime branch handles gaps.
    """
    if not positional:
        assert isinstance(x.index, pd.DatetimeIndex)
        return x.index.to_numpy(dtype="datetime64[ns]").astype("int64") / (10**9 * 86400)
    finite = np.isfinite(s.to_numpy())
    return np.arange(len(s), dtype=float)[finite]


def _has_time_spacing(s: pd.Series, is_dt: bool) -> bool:
    """Whether a per-year slope conversion is meaningful for this record.

    True only when the index actually measures time (a datetime index with a
    positive median spacing). Positional records get a per-step slope and no
    per-year figure, rather than a conversion that presumes a frequency the
    user never stated.
    """
    return _median_index_days(s, is_dt) is not None


def _median_index_days(s: pd.Series, is_dt: bool) -> float | None:
    """Median spacing of a datetime index in days, else None."""
    if not is_dt or len(s) < 2:
        return None
    deltas = np.diff(s.index.to_numpy("datetime64[ns]").astype("int64"))
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return None
    med = float(np.median(deltas)) / (10**9 * 86400)
    return med if med > 0 else None


# ---------------------------------------------------------------------------
# Mann-Kendall
# ---------------------------------------------------------------------------


def _s_statistic(group: np.ndarray) -> int:
    """S = sum over i<j of sign(y_j - y_i) within one group.

    Row-band by row-band so the temporary matrices stay around 2 MB: band row
    r pairs with every later value, and ``d[r, r:]`` is exactly those pairs
    (columns are indexed from ``i+1`` on, so diagonal ``d[r, r]`` is the pair
    ``(i+r, i+r+1)``). Banded in time (O(n^2)), O(n) in memory -- which bounds
    what a classic test can take: a 10-year hourly record in one group is
    ~3.8 billion pairs (tens of seconds, measured); for such records prefer
    the ``seasonal_period`` form, whose groups stay smaller.
    """
    n = group.size
    if n < 2:
        return 0
    total = 0
    chunk = max(1, int(2_000_000 // n))
    for i in range(0, n, chunk):
        # d[r, c] = group[i + r] - group[i + 1 + c]; valid pair when i+1+c > i+r,
        # i.e. c >= r; row i+r itself (r on the band's top row when the band is
        # shorter than its remaining columns) has no later partner and is skipped.
        d = group[i + 1 : n] - group[i : i + chunk, None]
        rows, cols = d.shape
        mask = np.arange(cols)[None, :] >= np.arange(rows)[:, None]
        total += int(np.sign(d[mask]).sum())
    return total


def _tie_correction(counts: pd.Series) -> int:
    """Sum over tied value-groups of t(t-1)(2t+5); counts holds value frequencies."""
    t = counts[counts > 1].to_numpy(dtype=np.float64)
    if t.size == 0:
        return 0
    return int(np.sum(t * (t - 1) * (2 * t + 5)))


def mann_kendall(
    series: pd.Series | pd.DataFrame,
    *,
    value_col: str | None = None,
    seasonal_period: int | None = None,
) -> TrendTest:
    """Mann-Kendall trend test on one series.

    Parameters
    ----------
    series : Series or DataFrame
        Datetime-indexed (or positionally ordered) values. Missing values are
        dropped and reported in ``n``.
    value_col : str, optional
        Column to test when ``series`` is a DataFrame.
    seasonal_period : int, optional
        Group the test into calendar periods of this length (Sen 1968),
        e.g. 24 for hourly-with-diurnal-structure, 7 for daily-with-weekly,
        12 for monthly-with-annual. The record must contain a whole number of
        full periods after dropping missing values; anything else raises
        :class:`~normet.exceptions.DataError` rather than testing uneven
        groups, which is where the variance formula silently breaks. Grouping
        is by *record position* (value ``m`` of each block of ``period``), so
        the alignment assumes a gap-free record: missing hours shift what
        lands in each slot.

    Returns
    -------
    TrendTest
        S, tie-corrected variance, continuity-corrected z and two-sided p.
    """
    s, _ = _to_series(series, value_col)
    x = s.dropna()
    if x.size < 8:
        raise DataError(
            f"mann_kendall needs at least 8 observed values, got {x.size} "
            f"(the test's normal approximation is the wrong tool before that)."
        )

    if seasonal_period is not None:
        period = int(seasonal_period)
        if period < 2:
            raise DataError(f"seasonal_period must be >= 2, got {period}.")
        n_periods, remainder = divmod(x.size, period)
        if remainder != 0:
            raise DataError(
                f"a seasonal test with period {period} needs a whole number of "
                f"periods, but the record holds {x.size} values "
                f"({n_periods} full periods + {remainder}). Drop the trailing "
                f"partial period or leave seasonal_period=None."
            )
        g = period
        arr = x.to_numpy()
        # Sen (1968): the S statistic is the sum of the classical S computed
        # *within each calendar slot* -- for a 24 h period, slot m is every
        # value recorded at hour m of the day, ordered by day. Grouping
        # consecutive periods instead (a whole 24 h stretch as one group)
        # measures the cycle's own rise-then-fall shape, not the between-year
        # trend, and comes out systematically signed for any cyclic signal.
        groups = [arr[m::g] for m in range(g)]
        S = sum(map(_s_statistic, groups))
        tie_sum = sum(_tie_correction(pd.Series(gr).value_counts()) for gr in groups)
        # Variance: each slot is an independent classical test with n = the
        # number of groups it holds; the groups are equal in length because
        # the whole-number-of-periods check above passed.
        n_slots = g
        # Per-slot classical variance, summed across the 24 slots:
        # Var(S) of one group of size m is m(m-1)(2m+5)/18 (see Hirsch
        # et al. 1982); the slots are disjoint so their variances add.
        m = len(groups[0])
        var_S = float(n_slots * (m * (m - 1) * (2 * m + 5)) / 18 - tie_sum / 18)
        n_periods_out: int | None = n_slots
    else:
        y = x.to_numpy()
        S = _s_statistic(y)
        # Standard MK variance with tie correction (Hirsch et al. 1982):
        # Var(S) = [n(n-1)(2n+5) - sum_t t(t-1)(2t+5)] / 18.
        var_S = (
            float(x.size * (x.size - 1) * (2 * x.size + 5)) - _tie_correction(x.value_counts())
        ) / 18.0
        n_periods_out = None

    if var_S <= 0:
        # Every value identical: S is 0 and its variance is 0 by the same
        # formula. Report no trend rather than dividing by zero.
        return TrendTest(
            S=0, var_S=0.0, z=0.0, p=1.0, n=x.size, seasonal=False, n_periods=n_periods_out
        )

    if S > 0:
        z = (S - 1) / np.sqrt(var_S)
    elif S < 0:
        z = (S + 1) / np.sqrt(var_S)
    else:
        z = 0.0

    p = 2.0 * (1.0 - NormalDist().cdf(abs(z)))
    return TrendTest(
        S=int(S),
        var_S=var_S,
        z=float(z),
        p=float(p),
        n=x.size,
        seasonal=seasonal_period is not None,
        n_periods=n_periods_out,
    )


# ---------------------------------------------------------------------------
# Theil-Sen
# ---------------------------------------------------------------------------


def theil_sen(
    series: pd.Series | pd.DataFrame,
    *,
    value_col: str | None = None,
) -> TrendSlope:
    """Robust Theil-Sen slope against time.

    The slope is the median pair-wise slope (``scipy.stats.theilslopes``),
    robust to the occasional extreme spike a pollution event leaves in the
    record. ``ci_low`` / ``ci_high`` are the analytic Sen (1968) 95 %
    interval on the slope (equation 2.6 there, as implemented by scipy),
    which unlike a percentile of pair slopes does account for the number of
    pairs and shrinks as the record grows.
    """
    stats = require_scipy_stats()
    s, positional = _to_series(series, value_col)
    x = s.dropna()
    if x.size < 4:
        raise DataError(f"theil_sen needs at least 4 observed values, got {x.size}.")
    xtime = _time_grid(s, x, positional)
    y = x.to_numpy()
    res = stats.theilslopes(y, xtime)
    out = TrendSlope(
        slope=float(res.slope),
        ci_low=float(res.low_slope),
        ci_high=float(res.high_slope),
        n=x.size,
        time_unit="step" if positional else "day",
    )
    if _has_time_spacing(s, is_dt=not positional):
        # xtime is in days, so the slope is per day; per-year is the fixed
        # 365.25 days regardless of sampling frequency.
        days = 365.25
        return TrendSlope(
            slope=out.slope,
            ci_low=out.ci_low,
            ci_high=out.ci_high,
            n=out.n,
            time_unit=out.time_unit,
            slope_per_year=out.slope * days,
            ci_low_per_year=out.ci_low * days,
            ci_high_per_year=out.ci_high * days,
        )
    return out


# ---------------------------------------------------------------------------
# Serial-correlation diagnostic + block bootstrap
# ---------------------------------------------------------------------------


def _ar1_residuals_acf(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """ACF(1) of the OLS trend residuals and a large-sample p-value.

    ``acf1 = sum(r_t r_{t-1}) / sum(r_t^2)`` is the sample autocorrelation
    (the divisor is the summed squares, not divided by n again -- a
    ``/n`` on the numerator that survives to the division would scale the
    statistic by 1/n and make it always read as zero). Returns ``(acf1, p)``
    with ``p = 2(1-Phi(|acf1| * sqrt(n)))``: under a no-autocorrelation
    null the sample ACF(1) is approximately N(0, 1/n). The divisor for OLS
    residuals is slightly different in principle (they are constrained to
    sum to zero), but for the diagnostic's purpose the plain sample formula
    is the right size.
    """
    n = y.size
    if n < 8:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    denom = float(np.sum(resid**2))
    if denom <= 0:
        return 0.0, 1.0
    acf1 = float(np.sum(resid[1:] * resid[:-1])) / denom
    z = acf1 * np.sqrt(n)
    p = 2.0 * (1.0 - NormalDist().cdf(abs(z)))
    return float(acf1), float(p)


def _default_block_length(n: int, med_days: float | None = None) -> int:
    """Moving-block length: the longer of ``3 n^(1/3)`` and one calendar week.

    The Politis-Romano ``3 n^(1/3)`` asymptote is what consistency needs as
    n grows; the week floor is what the *data* needs while n is still small
    -- for hourly records it keeps the weekly cycle inside a block, which the
    raw heuristic would not (``3 * 8760^(1/3)`` is only ~2.6 days). The floor
    applies to sub-daily records; daily and coarser records use the plain
    heuristic, whose annual-scale blocks already cover their cycle.
    """
    length = int(max(3, round(3 * n ** (1.0 / 3.0))))
    if med_days is not None and 0 < med_days < 1:
        length = max(length, int(-(-7 // med_days)))  # ceil of one week
    return int(min(length, n // 3)) if n >= 9 else 3


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Ordinary-least-squares slope for one bootstrap replicate (O(n))."""
    sx = x.sum()
    sy = y.sum()
    sxy = float(np.dot(x, y))
    sxx = float(np.dot(x, x))
    denom = x.size * sxx - sx * sx
    if denom == 0:
        return float("nan")
    return (x.size * sxy - sx * sy) / denom


def _block_bootstrap_slopes(
    x: np.ndarray,
    y: np.ndarray,
    *,
    block_length: int,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Moving-block bootstrap of the OLS trend slope.

    Blocks of ``block_length`` contiguous points are drawn with replacement
    until ``n`` points are assembled; the last block is truncated. Each
    replicate uses the OLS slope (Theil-Sen per replicate would cost
    O(n^2) per draw and 1000 draws of a year of hourly data would not
    finish in a work session); the block sampling is what fixes the
    autocorrelation, not the per-replicate estimator.
    """
    n = x.size
    rng = np.random.default_rng(seed)
    positions = np.arange(n)
    n_starts = n - block_length + 1
    full = n // block_length
    remain = n - full * block_length
    slopes = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        starts = rng.integers(0, n_starts, size=full)
        idx = (
            np.concatenate([positions[s : s + block_length] for s in starts])
            if full
            else np.empty(0, dtype=int)
        )
        if remain > 0 and n_starts > 0:
            s0 = int(rng.integers(0, n_starts))
            idx = np.concatenate([idx, positions[s0 : s0 + remain]])
        xb = x[idx]
        slopes[b] = _ols_slope(xb, y[idx])
    return slopes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def require_scipy_stats() -> Any:
    # Deferred: ~0.8 s to import, and `import normet` should stay fast (see
    # normet.utils.metrics for the same call). scipy ships without type
    # stubs for this purpose and the call sites treat the result as Any.
    from scipy import stats  # noqa: PLC0415

    return stats


def trend(
    series: pd.Series | pd.DataFrame,
    *,
    value_col: str | None = None,
    name: str | None = None,
    seasonal_period: int | None = None,
    test_alpha: float = 0.05,
    block_length: int | None = None,
    n_bootstrap: int = 1000,
    seed: int = 7_654_321,
) -> TrendResult:
    """Trend test + robust slope + honest error bars, for one series.

    Runs :func:`mann_kendall` and :func:`theil_sen`, then asks whether the
    residuals of a linear fit are autocorrelated. If they are (which hourly
    normalised almost always is), the Theil-Sen interval -- which assumes
    independent sampling -- is supplemented with a moving-block bootstrap
    standard error, bootstrap interval and a Wald p-value from that
    bootstrap SE. The Mann-Kendall p-value is always the classical one:
    there is no equally standard autocorrelated variant, so the result
    carries an explicit note rather than pretending it was corrected.

    A known asymmetry, documented rather than hidden: the bootstrap SE is
    estimated from OLS replicate slopes (fast, O(n) per replicate), while the
    point estimate and the classical interval are Theil-Sen. The block
    sampling is what repairs the autocorrelation in either case, but on a
    record riddled with extreme spikes the OLS variance -- not the robust
    variance -- is what the bootstrap measures, so read ``p_wald_boot`` as
    the corrected significance of the *OLS* trend.

    Parameters
    ----------
    series : Series or DataFrame
        The series to test -- typically the emission component from
        :func:`normet.decompose` or a deweathered ``normalised`` column.
    value_col : str, optional
        Column when ``series`` is a DataFrame.
    name : str, optional
        Label reported in ``str(result)``; defaults to the series name.
    seasonal_period : int, optional
        Forwarded to :func:`mann_kendall` (Sen 1968 seasonal test).
    test_alpha : float, default 0.05
        Threshold at which the AR(1) diagnostic triggers the bootstrap.
    block_length : int, optional
        Bootstrap block length in points. Defaults to the longer of the
        Politis-Romano ``3 n^(1/3)`` heuristic and one calendar week for
        sub-daily records, so the weekly cycle stays inside a block.
    n_bootstrap : int, default 1000
        Bootstrap replicates.
    seed : int, default 7654321
        RNG seed; the bootstrap output is bit-identical across runs.

    Returns
    -------
    TrendResult
    """
    s, positional = _to_series(series, value_col)
    x = s.dropna()
    n_dropped = int(s.size - x.size)
    if x.size < 15:
        raise DataError(
            f"trend() needs at least 15 observed values, got {x.size} -- a trend "
            "test on fewer is a statement about the noise, not the trend."
        )

    # The *original* series goes to both sub-tests: each re-runs
    # ``_to_series`` internally, and feeding them the post-``dropna`` series
    # would change what an integer index resolves to (a ``RangeIndex`` with a
    # hole becomes a plain int index, which a fresh ``_to_series`` would read
    # as nanoseconds-since-epoch timestamps).
    test = mann_kendall(s, seasonal_period=seasonal_period)
    slope = theil_sen(s)

    xtime = _time_grid(s, x, positional)
    y = x.to_numpy()
    acf1, acf_p = _ar1_residuals_acf(y, xtime)
    autocorrelated = bool(np.isfinite(acf_p) and acf_p < test_alpha)

    result = TrendResult(
        name=name or (str(s.name) if s.name is not None else "series"),
        n=x.size,
        n_dropped=n_dropped,
        test=test,
        slope=slope,
        ar1_acf=acf1,
        ar1_p=acf_p,
        ar_autocorrelated=autocorrelated,
    )
    if not autocorrelated:
        result.notes.append(
            "AR(1) residual autocorrelation not significant; the Theil-Sen interval is used as-is."
        )
        return result

    bl = (
        int(block_length)
        if block_length is not None
        else _default_block_length(x.size, med_days=_median_index_days(x, is_dt=not positional))
    )
    bl = max(1, min(bl, x.size // 3))
    boots = _block_bootstrap_slopes(
        xtime, y, block_length=bl, n_bootstrap=int(n_bootstrap), seed=int(seed)
    )
    finite = np.isfinite(boots)
    if finite.sum() < 10:
        result.notes.append(
            f"only {int(finite.sum())} of {n_bootstrap} bootstrap replicates were "
            "finite; no bootstrap interval reported."
        )
        return result

    b = boots[finite]
    se = float(b.std(ddof=1))
    lo, hi = (float(q) for q in np.percentile(b, [2.5, 97.5]))
    result.block_length = bl
    result.n_bootstrap = int(finite.sum())
    log.debug(
        "trend(%s): block bootstrap %d replicates, block length %d",
        result.name,
        int(finite.sum()),
        bl,
    )
    if se > 0:
        result.boot_slope_se = se
        result.boot_ci_low = lo
        result.boot_ci_high = hi
        z_wald = slope.slope / se
        result.p_wald_boot = float(2 * (1 - NormalDist().cdf(abs(z_wald))))
    result.notes.append(
        f"AR(1) residual autocorrelation {acf1:.2f} (p={acf_p:.2g}) < {test_alpha}: "
        f"block bootstrap (length {bl}, {int(finite.sum())} replicates) used for the "
        "slope interval. The Mann-Kendall p-value above is the classical one and "
        "is anti-conservative under this autocorrelation; read the bootstrap "
        "p_wald as the corrected test."
    )
    return result
