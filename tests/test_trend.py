"""Tests for normet.analysis.trend (PR 1 of the 1.1 plan).

The tests pin the three things the module exists for:

1. the classical Mann-Kendall S / tie correction / continuity correction
   against hand-computed and cross-checked values;
2. the Theil-Sen slope against the median of all pair-wise slopes;
3. the reason the module exists at all -- an AR(1) record inflates the
   classical p-value, and the block-bootstrap correction must bring the
   inference back to the nominal level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from normet.analysis.trend import (
    TrendResult,
    mann_kendall,
    theil_sen,
    trend,
)
from normet.exceptions import DataError


def _hourly_index(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="h")


# ---------------------------------------------------------------------------
# mann_kendall
# ---------------------------------------------------------------------------


def test_mk_hand_computed_two_groups_equivalence():
    # A record whose values repeat in an exact period: S, the tie correction
    # and the classical formula must all agree with the small case computed
    # by brute force here.
    y = np.array([1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 2.0, 4.0])
    n = len(y)
    iu = np.triu_indices(n, k=1)
    S_brute = int(np.sign(y[iu[1]] - y[iu[0]]).sum())  # sign(y_j - y_i) for i<j
    res = mann_kendall(pd.Series(y, index=_hourly_index(n)))
    assert S_brute == res.S
    # Variance with ties: n(n-1)(2n+5)/18 minus sum t(t-1)(2t+5)/18
    counts = pd.Series(y).value_counts().to_numpy(dtype=float)
    tie = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var = (n * (n - 1) * (2 * n + 5) - tie) / 18
    assert res.var_S == pytest.approx(var, rel=1e-12)


def test_mk_direction_and_p_on_obvious_trends():
    up = pd.Series(np.arange(60, dtype=float), index=_hourly_index(60))
    dn = pd.Series(-np.arange(60, dtype=float), index=_hourly_index(60))
    assert mann_kendall(up).direction == "increasing"
    assert mann_kendall(dn).direction == "decreasing"
    assert mann_kendall(up).p < 1e-6
    assert mann_kendall(dn).p < 1e-6


def test_mk_flat_series_no_trend():
    flat = pd.Series(np.full(50, 3.2), index=_hourly_index(50))
    res = mann_kendall(flat)
    assert res.S == 0
    assert res.p == 1.0
    assert res.direction == "none"


def test_mk_missing_values_dropped():
    idx = _hourly_index(20)
    y = np.arange(20, dtype=float)
    y[[3, 9, 15]] = np.nan
    res = mann_kendall(pd.Series(y, index=idx))
    assert res.n == 17


def test_mk_too_few_values_raises():
    with pytest.raises(DataError):
        mann_kendall(pd.Series([1.0, 2.0, 3.0, 4.0], index=_hourly_index(4)))


def test_mk_seasonal_requires_whole_periods():
    # 10 hourly values cannot divide into weekly periods whole.
    s = pd.Series(np.arange(10, dtype=float), index=_hourly_index(10))
    with pytest.raises(DataError, match="whole number of periods"):
        mann_kendall(s, seasonal_period=7)


def test_mk_seasonal_detects_trend_under_diurnality():
    # One year of hourly, diurnal cycle on a rising ramp: the seasonal test
    # must detect the ramp while the seasonal cycle stays a group, not noise.
    rng = np.random.default_rng(1)
    n_years = 3
    n = n_years * 8760
    t = np.arange(n)
    season = 3.0 * np.sin(2 * np.pi * t / 24.0)
    y = 0.002 * t / 24.0 + season + rng.normal(0, 1.0, n)
    s = pd.Series(y, index=_hourly_index(n))
    classical = mann_kendall(s)
    seasonal = mann_kendall(s, seasonal_period=24)
    assert seasonal.n_periods == 24
    # Both must call this a rising trend; the seasonal one is the honest one.
    assert classical.direction == "increasing"
    assert seasonal.direction == "increasing"
    assert classical.p < 0.05
    assert seasonal.p < 0.05


def test_mk_accepts_dataframe_with_value_col():
    y = np.linspace(0, 10, 30)
    df = pd.DataFrame({"a": np.ones(30), "target": y}, index=_hourly_index(30))
    res = mann_kendall(df, value_col="target")
    assert res.direction == "increasing"


def test_mk_dataframe_requires_value_col():
    df = pd.DataFrame({"a": np.ones(30)}, index=_hourly_index(30))
    with pytest.raises(ValueError, match="value_col.*is required"):
        mann_kendall(df)


# ---------------------------------------------------------------------------
# theil_sen
# ---------------------------------------------------------------------------


def test_theil_sen_equals_brute_force_median_pairwise_slope():
    rng = np.random.default_rng(2)
    n = 120
    x = np.arange(n, dtype=float)
    y = 1.5 + 0.04 * x + rng.normal(0, 0.5, n)
    iu = np.triu_indices(n, k=1)
    slopes = np.median((y[iu[1]] - y[iu[0]]) / (x[iu[1]] - x[iu[0]]))
    res = theil_sen(pd.Series(y, index=pd.date_range("2020-01-01", periods=n, freq="D")))
    assert res.slope == pytest.approx(slopes, rel=1e-9)
    # CI ordered, slope inside it
    assert res.ci_low < res.slope < res.ci_high
    # A true slope of 0.04/day must land inside the 95% interval here.
    assert res.ci_low < 0.04 < res.ci_high


def test_theil_sen_robust_to_outlier_spike():
    # A 100x spike at one point: the median pair-wise slope barely moves.
    n = 200
    x = np.arange(n, dtype=float)
    y = 0.1 + 0.01 * x
    y[100] = 1000.0
    res = theil_sen(pd.Series(y, index=pd.date_range("2020-01-01", periods=n, freq="D")))
    assert res.slope == pytest.approx(0.01, abs=5e-3)


def test_theil_sen_positional_index_reports_steps():
    y = np.linspace(0, 10, 50)
    res = theil_sen(pd.Series(y))
    assert res.time_unit == "step"
    # 10 units over 49 steps
    assert res.slope == pytest.approx(10.0 / 49)
    assert res.slope_per_year is None


def test_theil_sen_datetime_reports_per_day_and_per_year():
    y = np.linspace(0, 365.25, 8760)  # +1/day over one year of hourly data
    res = theil_sen(pd.Series(y, index=_hourly_index(8760)))
    assert res.time_unit == "day"
    assert res.slope == pytest.approx(1.0, rel=1e-3)
    assert res.slope_per_year == pytest.approx(365.25, rel=1e-3)
    assert res.ci_low_per_year is not None and res.ci_high_per_year is not None


def test_theil_sen_too_few_values_raises():
    with pytest.raises(DataError):
        theil_sen(pd.Series([1.0, 2.0], index=_hourly_index(2)))


def test_positional_series_with_nan_stays_per_step():
    # Regression: dropping NaNs from a RangeIndex leaves a plain int index,
    # which a re-derivation of "positional" used to misread as timestamps
    # (positions 0..n-1 as nanoseconds since epoch), giving a slope of
    # ~1e13 "per day" for what is a per-step ramp.
    y = np.linspace(0, 10, 100)
    y[10:15] = np.nan
    direct = theil_sen(pd.Series(y))
    assert direct.time_unit == "step"
    assert direct.slope_per_year is None
    res = trend(pd.Series(y, name="ramp"))
    assert res.slope.time_unit == "step"
    assert res.slope.slope_per_year is None
    # Same values, same steps -> same slope, whether or not NaNs were dropped
    # first; both must be the per-step ramp (~10 units / 99 steps).
    full = theil_sen(pd.Series(np.linspace(0, 10, 100)))
    assert direct.slope == pytest.approx(full.slope, rel=1e-9)
    assert res.slope.slope == pytest.approx(full.slope, rel=1e-9)


def test_positional_gap_preserves_real_stops():
    # A hole in a positional record is a missing stop, not a renumbering:
    # the per-step slope of the ramp survives the gap, and the datetime
    # analogue behaves identically.
    y = np.linspace(0, 10, 24)
    y[11:13] = np.nan
    res = trend(pd.Series(y))
    assert res.slope.time_unit == "step"
    assert res.slope.slope == pytest.approx(10.0 / 23, rel=1e-9)


# ---------------------------------------------------------------------------
# trend(): the autocorrelation correction is the point of the module
# ---------------------------------------------------------------------------


def _independent_series(slope_per_day: float, n_hours: int, sd: float, seed: int = 3):
    rng = np.random.default_rng(seed)
    n = n_hours
    noise = rng.normal(0, sd, n)
    y = slope_per_day * n / 24.0 * (np.arange(n) / n) + noise
    return pd.Series(y, index=_hourly_index(n))


def test_trend_rejects_on_independent_data_with_no_trend():
    # No trend, independent errors: the classical p should not be significant.
    s = _independent_series(0.0, 8760, sd=5.0)
    res = trend(s)
    assert isinstance(res, TrendResult)
    assert res.test.p > 0.05
    assert not res.ar_autocorrelated
    assert res.boot_slope_se is None
    assert res.slope.slope == pytest.approx(0.0, abs=0.05)


def test_trend_detects_a_real_trend_with_correlated_noise():
    # A 2 ug/m3/year rise in hourly data, AR(1) error at rho=0.5: the
    # classical MK says "insanely significant", but it is overconfident.
    rng = np.random.default_rng(4)
    n = 8760 * 2
    eps = np.zeros(n)
    rho = 0.5
    z = rng.normal(0, 1, n)
    for i in range(n):
        eps[i] = rho * eps[i - 1] + z[i] if i else z[0]
    y = (2.0 / 365.25) * (np.arange(n) / 24.0) + eps
    s = pd.Series(y, index=_hourly_index(n))
    res = trend(s)
    assert res.ar_autocorrelated
    assert res.ar1_acf > 0.4
    # The trend must be found by BOTH tests: classical (inflated) and
    # bootstrap-corrected.
    assert res.test.p < 0.05
    assert res.p_wald_boot is not None
    assert res.p_wald_boot < 0.05
    assert res.boot_slope_se is not None and res.boot_slope_se > 0
    assert res.boot_ci_low is not None and res.boot_ci_high is not None
    assert res.slope.slope_per_year is not None
    assert res.slope.slope_per_year == pytest.approx(2.0, abs=0.3)


def test_trend_bootstrap_se_wider_than_classical_under_autocorrelation():
    # The whole reason for the bootstrap: under AR(1), the independence-based
    # Theil-Sen interval is too narrow. The bootstrap SE must come out wider
    # than the Sen (1968) analytic half-width.
    rng = np.random.default_rng(5)
    n = 8760
    rho = 0.6
    eps = np.zeros(n)
    z = rng.normal(0, 1, n)
    for i in range(n):
        eps[i] = rho * eps[i - 1] + z[i] if i else z[0]
    y = (1.0 / 365.25) * (np.arange(n) / 24.0) + eps
    s = pd.Series(y, index=_hourly_index(n))
    res = trend(s, block_length=24, n_bootstrap=400, seed=11)
    analytic_half = res.slope.ci_high - res.slope.slope
    assert res.boot_slope_se * 1.96 > analytic_half


def test_trend_independent_record_uses_sen_interval_without_bootstrap():
    # White noise (no trend): no AR(1) trigger, classical interval kept.
    s = _independent_series(0.0, 4380, sd=3.0, seed=6)
    res = trend(s)
    assert not res.ar_autocorrelated
    assert "AR(1) residual autocorrelation not significant" in res.notes[0]
    assert res.p_wald_boot is None


def test_trend_is_bit_identical_under_seed():
    # AR(1) errors so the block bootstrap actually runs (the point of the
    # fixed-seed comparison is that replicate set is reproducible, not just
    # the classical test that is deterministic regardless).
    rng = np.random.default_rng(7)
    n = 8760
    rho = 0.5
    eps = np.zeros(n)
    z = rng.normal(0, 1, n)
    for i in range(n):
        eps[i] = rho * eps[i - 1] + z[i] if i else z[0]
    y = (0.5 / 365.25) * (np.arange(n) / 24.0) + eps
    s = pd.Series(y, index=_hourly_index(n))
    a = trend(s, block_length=24, n_bootstrap=200, seed=42)
    b = trend(s, block_length=24, n_bootstrap=200, seed=42)
    assert a.ar_autocorrelated
    assert a.boot_slope_se == b.boot_slope_se
    assert a.boot_ci_low == b.boot_ci_low
    assert a.p_wald_boot == b.p_wald_boot
    # A different seed moves the bootstrap, and only the bootstrap.
    c = trend(s, block_length=24, n_bootstrap=200, seed=43)
    assert c.boot_slope_se != a.boot_slope_se
    assert a.test.S == c.test.S


def test_trend_requires_minimum_length():
    with pytest.raises(DataError, match="at least 15"):
        trend(pd.Series(np.arange(12, dtype=float), index=_hourly_index(12)))


def test_trend_nan_rows_reported_not_silently_dropped():
    y = np.arange(100, dtype=float) * 0.01
    y[10:20] = np.nan
    s = pd.Series(y, index=_hourly_index(100))
    res = trend(s)
    assert res.n == 90
    assert res.n_dropped == 10
    assert "dropped 10 missing" in str(res)


def test_trend_summary_is_a_dataframe():
    s = _independent_series(0.0 / 365.25, 4380, sd=1.0, seed=8)
    res = trend(s)
    df = res.summary()
    assert list(df.columns) == ["metric", "value"]
    metrics = set(df["metric"])
    assert {"MK_S", "MK_p_two_sided", "direction"} <= metrics
    assert "slope_per_day" in metrics


def test_trend_str_reports_correction_when_applied():
    rng = np.random.default_rng(9)
    n = 4380
    eps = np.zeros(n)
    for i in range(n):
        eps[i] = 0.5 * eps[i - 1] + rng.normal(0, 1) if i else rng.normal(0, 1)
    y = (1.0 / 365.25) * (np.arange(n) / 24.0) + eps
    s = pd.Series(y, index=_hourly_index(n))
    res = trend(s, block_length=24, n_bootstrap=200)
    line = str(res)
    assert "AR(1)" in line
    assert "block bootstrap" in line


def test_trend_explicit_block_length_respected():
    s = _independent_series(0.0, 4380, sd=1.0, seed=10)
    res = trend(s, block_length=168)
    # Force the bootstrap to run by lowering the alpha for the trigger.
    assert res.block_length is None  # white noise, no trigger
    res2 = trend(s, block_length=168, test_alpha=2.0)  # trigger always on
    assert res2.block_length == 168
    assert res2.n_bootstrap == 1000


def test_str_does_not_crash_when_bootstrap_not_ran():
    # Regression: str() used to format boot_slope_se / p_wald_boot
    # unconditionally. When the AR(1) trigger fires but the bootstrap
    # produced no usable replicate (e.g. all blocks collinear), those
    # fields stayed None and str() raised TypeError.
    from normet.analysis.trend import TrendResult, TrendSlope, TrendTest

    res = TrendResult(
        name="degenerate",
        n=1000,
        n_dropped=0,
        test=TrendTest(S=500, var_S=1000.0, z=15.8, p=0.0, n=1000),
        slope=TrendSlope(slope=0.3, ci_low=0.1, ci_high=0.5, n=1000),
        ar1_acf=0.8,
        ar1_p=1e-6,
        ar_autocorrelated=True,
        # boot_* intentionally left at the None defaults
    )
    line = str(res)
    assert "AR(1)" in line
    assert "bootstrap unavailable" in line
    # summary() must still work in the same state
    df = res.summary()
    assert "p_wald_block_bootstrap" in set(df["metric"])
    assert df.loc[df["metric"] == "p_wald_block_bootstrap", "value"].iloc[0] is None
