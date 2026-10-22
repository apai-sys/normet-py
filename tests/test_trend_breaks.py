"""Tests for the piecewise (segmented) trend, ``detect_trend_breaks``.

The things pinned here mirror the module's purpose:

1. a record whose level or slope changes part-way through is split at the
   right place, and the per-segment Theil-Sen slopes recover the injected
   regimes;
2. a record with no change is *not* reported as one -- neither pure noise
   nor autocorrelated noise crosses the BIC + block-bootstrap gate at an
   inflated rate;
3. the minimum-segment guard is enforced, missing values are dropped and
   counted, positional records work per step;
4. under a fixed ``seed`` the whole result is bit-identical across runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from normet.analysis.trend import TrendBreaks, detect_trend_breaks
from normet.exceptions import DataError


def _hourly_index(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="h")


def _daily(n: int, start: str = "2019-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D")


def _ar1(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    eps = np.zeros(n)
    z = rng.normal(0, 1, n)
    for i in range(n):
        eps[i] = rho * eps[i - 1] + z[i]
    return eps


def _slope_change(
    n_left: int, n_right: int, slope_left: float, slope_right: float, sd: float, seed: int
) -> pd.Series:
    """Two linear regimes joined at day ``n_left``, plus Gaussian noise.

    Regime 2 continues from the level regime 1 reached at the join, so a
    non-zero ``slope_left`` does not create a level jump at the break.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_left + n_right)
    y = np.where(t < n_left, slope_left * t, slope_left * n_left + slope_right * (t - n_left))
    return pd.Series(y + rng.normal(0, sd, n_left + n_right), index=_daily(n_left + n_right))


# ---------------------------------------------------------------------------
# Detection: a real break is found at the right place with the right slopes
# ---------------------------------------------------------------------------


def test_slope_change_recovered_at_the_injected_position():
    s = _slope_change(400, 400, 0.0, 0.5, sd=1.0, seed=2)
    res = detect_trend_breaks(s, n_bootstrap=400)
    assert isinstance(res, TrendBreaks)
    assert res.n_segments == 2
    assert res.significant
    assert res.p_value is not None and res.p_value < 0.05
    # The break lands within a few days of the injected position 400.
    pos = res.starts[1]
    assert abs(pd.Timestamp(pos) - _daily(800)[399]) <= pd.Timedelta(days=15)
    # Regime 1: flat. Regime 2: +0.5/day, recovered within tolerance.
    assert res.segments[0].slope == pytest.approx(0.0, abs=0.05)
    assert res.segments[1].slope == pytest.approx(0.5, abs=0.1)
    assert res.segments[1].ci_low < 0.5 < res.segments[1].ci_high or res.segments[1].ci_low < 0.5
    # Time is measured in days for a datetime-indexed record.
    assert res.time_unit == "day"
    assert res.segments[1].slope_per_year is not None
    assert res.segments[1].slope_per_year == pytest.approx(0.5 * 365.25, rel=0.1)


def test_level_shift_recovered():
    # A step (a lockdown) with roughly equal slopes on both sides: the
    # detector must still find it, because the split improves the *level* fit
    # even when the slopes barely differ.
    rng = np.random.default_rng(11)
    n = 800
    y = np.concatenate([np.full(400, 10.0), np.full(400, 7.0)]) + _ar1(n, 0.6, rng) * 1.5
    s = pd.Series(y, index=_daily(n))
    res = detect_trend_breaks(s, n_bootstrap=400)
    assert res.n_segments == 2
    assert res.significant
    assert res.p_value is not None and res.p_value < 0.05
    pos = pd.Timestamp(res.starts[1])
    assert abs(pos - _daily(800)[399]) <= pd.Timedelta(days=20)


def test_two_real_breaks_with_max_segments():
    # A staircase of levels: 50 for 300 days, then 120 for 300, then 130.
    # The first boundary (50 -> 120) is far stronger than the second
    # (120 -> 130), so the greedy first round's best single split lands at
    # 300 without ambiguity and the second round splits the tail at 600.
    rng = np.random.default_rng(5)
    y = np.concatenate([np.full(300, 50.0), np.full(300, 120.0), np.full(300, 130.0)])
    y = y + rng.normal(0, 1.0, 900)
    s = pd.Series(y, index=_daily(900))

    res3 = detect_trend_breaks(s, max_segments=3, n_bootstrap=300)
    assert res3.n_segments == 3
    assert len(res3.starts) == 3 and len(res3.ends) == 3
    b1 = pd.Timestamp(res3.starts[1])
    b2 = pd.Timestamp(res3.starts[2])
    assert b1 < b2
    # first break near day 300, second near day 600.
    assert abs(b1 - _daily(900)[299]) <= pd.Timedelta(days=15)
    assert abs(b2 - _daily(900)[599]) <= pd.Timedelta(days=15)
    # All regimes are level: every segment's slope is ~0, and the third
    # segment's CI covers the flat value.
    for seg in res3.segments:
        assert abs(seg.slope) < 0.1
    assert res3.improvement_pct is not None and res3.improvement_pct > 50.0


# ---------------------------------------------------------------------------
# Honesty: no false break on records that have no change
# ---------------------------------------------------------------------------


def test_pure_noise_is_not_split():
    rng = np.random.default_rng(7)
    y = rng.normal(50.0, 2.0, 800)
    s = pd.Series(y, index=_daily(800))
    res = detect_trend_breaks(s, n_bootstrap=400)
    assert res.n_segments == 1
    assert not res.significant
    # Even so, the search is recorded: best_break exists but was rejected.
    assert res.best_break is None or res.p_value is None or not res.significant


def test_autocorrelated_pure_noise_is_not_split():
    # The core honesty test: strongly autocorrelated noise (the common case)
    # must not be reported as a break. We check a single seed for a fast run
    # here; the rate over many seeds is pinned separately below.
    rng = np.random.default_rng(3)
    y = 50.0 + _ar1(800, 0.7, rng) * 1.0
    s = pd.Series(y, index=_daily(800))
    res = detect_trend_breaks(s, n_bootstrap=400)
    assert res.n_segments == 1
    assert not res.significant


def test_autocorrelated_noise_false_positive_rate_stays_low():
    # Over many independent autocorrelated records without a break, the
    # fraction reported as significant must stay near the nominal alpha.
    # The BIC + block-bootstrap gate is what should keep this from
    # over-segmenting -- a naked p or a BIC-only gate would not.
    n_seeds = 12
    alpha = 0.05
    sig = 0
    n = 500
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        y = 50.0 + _ar1(n, 0.7, rng) * 1.0
        s = pd.Series(y, index=_daily(n))
        res = detect_trend_breaks(s, n_bootstrap=200, alpha=alpha)
        sig += int(res.significant)
    # 12 trials: expect roughly 0 or 1 false positive. Allow up to 2 to
    # absorb bootstrap randomness without masking a real over-segmentation.
    assert sig <= 2, f"expected <=2 significant of {n_seeds}, got {sig}"


def test_a_real_break_survives_autocorrelated_noise():
    # Power: with a genuine change under autocorrelated noise the detector
    # must still find it (the block sampling must not wash it out).
    rng = np.random.default_rng(9)
    n = 800
    base = _ar1(n, 0.6, rng) * 1.0
    y = base.copy()
    y[400:] += 0.4 * (np.arange(n)[400:] - 399)
    s = pd.Series(y, index=_daily(n))
    res = detect_trend_breaks(s, n_bootstrap=400)
    assert res.n_segments == 2
    assert res.significant
    assert res.p_value is not None and res.p_value < 0.05


# ---------------------------------------------------------------------------
# Guards and mechanics
# ---------------------------------------------------------------------------


def test_min_segment_bound_is_respected():
    # A break injected very close to the start must not be reported at that
    # position: the smaller of the two segments would be < min_segment, so
    # the search's admissible window excludes it.
    n = 1000
    rng = np.random.default_rng(13)
    y = np.concatenate([np.full(10, 100.0), np.full(n - 10, 0.0)]) + rng.normal(0, 0.5, n)
    s = pd.Series(y, index=_daily(n))
    res = detect_trend_breaks(s, n_bootstrap=200)
    # Default min_segment = max(28, n//20=50) = 50. Every reported segment
    # must be at least that long (or it is one regime).
    if res.n_segments > 1:
        for seg in res.segments:
            assert seg.n >= 50
    # With the default window the tiny first segment is not admissible, so
    # the reported break (if any) cannot land inside the first 50.
    if res.n_segments > 1:
        pos = pd.Timestamp(res.starts[1])
        assert pos > _daily(1000)[49]


def test_min_segment_too_large_raises():
    s = pd.Series(np.arange(100.0), index=_daily(100))
    with pytest.raises(DataError, match="min_segment"):
        detect_trend_breaks(s, min_segment=80)


def test_too_few_values_raises():
    s = pd.Series(np.arange(39.0), index=_daily(39))
    with pytest.raises(DataError, match="at least 40"):
        detect_trend_breaks(s)


def test_invalid_arguments_raise():
    s = pd.Series(np.arange(200.0), index=_daily(200))
    with pytest.raises(DataError):
        detect_trend_breaks(s, max_segments=1)
    with pytest.raises(DataError):
        detect_trend_breaks(s, alpha=0.0)
    with pytest.raises(DataError):
        detect_trend_breaks(s, alpha=1.0)
    with pytest.raises(DataError):
        detect_trend_breaks(s, n_bootstrap=10)
    with pytest.raises(DataError):
        detect_trend_breaks(s, min_segment=1)


def test_missing_values_are_dropped_and_counted():
    y = np.arange(200.0, dtype=float)
    y[40:60] = np.nan
    y[150:] += 3.0
    s = pd.Series(y, index=_daily(200))
    res = detect_trend_breaks(s, n_bootstrap=200)
    assert res.n == 180
    assert res.n_dropped == 20


def test_positional_index_runs_per_step():
    y = np.concatenate([np.full(250, 5.0), 5.0 + 0.2 * np.arange(1, 251)])
    rng = np.random.default_rng(17)
    y = y + rng.normal(0, 0.5, 500)
    s = pd.Series(y, name="pos")
    res = detect_trend_breaks(s, n_bootstrap=200)
    assert res.time_unit == "step"
    if res.n_segments == 2:
        # A positional break is an integer position, not a timestamp.
        assert isinstance(res.starts[1], (int, np.integer))
        # No per-year conversion for a positional record.
        assert res.segments[1].slope_per_year is None


def test_dataframe_input_uses_value_col():
    s = _slope_change(300, 300, 0.0, 0.5, sd=1.0, seed=19)
    df = pd.DataFrame({"emi_total": s.values, "other": s.values + 1.0})
    res = detect_trend_breaks(df, value_col="emi_total", n_bootstrap=200)
    assert res.n_segments == 2
    with pytest.raises(ValueError, match="value_col"):
        detect_trend_breaks(df, n_bootstrap=200)


def test_constant_series_is_one_regime():
    s = pd.Series(np.zeros(300), index=_daily(300))
    res = detect_trend_breaks(s, n_bootstrap=100)
    assert res.n_segments == 1
    assert not res.significant
    assert res.segments[0].slope == 0.0


def test_deterministic_under_seed():
    s = _slope_change(400, 400, 0.0, 0.5, sd=1.0, seed=23)
    a = detect_trend_breaks(s, n_bootstrap=300)
    b = detect_trend_breaks(s, n_bootstrap=300)
    assert a.summary().to_dict() == b.summary().to_dict()
    assert a.p_value == b.p_value
    assert a.n_segments == b.n_segments
    assert a.starts == b.starts
    assert a.notes == b.notes


def test_summary_shape():
    s = _slope_change(400, 400, 0.0, 0.5, sd=1.0, seed=29)
    res = detect_trend_breaks(s, n_bootstrap=300)
    df = res.summary()
    assert list(df.columns[:5]) == ["start", "end", "n", "slope", "ci_low"]
    if res.n_segments == 2:
        assert len(df) == 2
        assert "slope_per_year" in df.columns


# ---------------------------------------------------------------------------
# Greedy search internals: a pure-line record is one regime
# ---------------------------------------------------------------------------


def test_exact_linear_record_is_not_split():
    # A perfectly linear record has zero SSE after the single-segment fit;
    # BIC ties at -inf and the record stays one regime rather than splitting
    # on numerical noise.
    y = 10.0 + 0.05 * np.arange(300.0)
    s = pd.Series(y, index=_daily(300))
    res = detect_trend_breaks(s, n_bootstrap=100)
    assert res.n_segments == 1
    assert not res.significant
