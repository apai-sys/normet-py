"""Parity tests for the optional Rust Theil-Sen kernel.

``normet_rust`` is built from ``rust/normet-python`` (see ``AGENTS.md``) and
is opt-in: ``normet.theil_sen(..., backend="rust")``. The contract is
bit-identity -- switching backends must not move a number -- so every test
below compares exact floats, against scipy below the 4,000-point switch and
against :func:`_theil_sen_large` above it, which is the path the default
backend actually runs there. The module skips itself when the extension is
not installed, so the suite still passes from a source checkout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from normet.analysis.trend import (
    _THEIL_SEN_LARGE,
    _theil_sen_large,
    _theil_sen_rust,
    theil_sen,
)
from normet.core.exceptions import ConfigError
from normet.utils._rust import rust_available

normet_rust = pytest.importorskip("normet_rust")


def _linear_record(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=float)
    y = 0.5 + 0.0002 * x + rng.normal(0.0, 0.1, n)
    return x, y


@pytest.mark.parametrize("n", [5, 12, 57, 400, 1_000, _THEIL_SEN_LARGE - 1])
def test_below_the_switch_matches_scipy(n: int) -> None:
    """Below 4,000 points the default backend calls scipy; Rust must agree."""
    from scipy import stats as scipy_stats

    x, y = _linear_record(n, seed=n)
    ref = scipy_stats.theilslopes(y, x)
    slope, low, high = _theil_sen_rust(x, y)
    assert slope == ref.slope
    assert low == ref.low_slope
    assert high == ref.high_slope


@pytest.mark.parametrize("n", [_THEIL_SEN_LARGE + 1, _THEIL_SEN_LARGE + 200, 5_000])
def test_above_the_switch_matches_the_bounded_python_path(n: int) -> None:
    """Above the switch the default backend runs the bounded selector."""
    from scipy import stats as scipy_stats

    x, y = _linear_record(n, seed=n)
    assert _theil_sen_rust(x, y) == _theil_sen_large(x, y, scipy_stats)


def test_duplicate_days_tied_values_and_unsorted_input_match() -> None:
    """The equal-x exclusion, tie correction and internal sort are replicated."""
    from scipy import stats as scipy_stats

    rng = np.random.default_rng(11)
    n = 500
    x = rng.integers(0, n // 2, n).astype(float)  # duplicate days
    y = np.round(0.02 * x + rng.normal(0, 1.0, n), 1)  # tied values
    order = rng.permutation(n)
    ref = scipy_stats.theilslopes(y[order], x[order])
    slope, low, high = _theil_sen_rust(x[order], y[order])
    assert slope == ref.slope
    assert low == ref.low_slope
    assert high == ref.high_slope


def test_ties_above_the_switch_match_the_bounded_python_path() -> None:
    from scipy import stats as scipy_stats

    rng = np.random.default_rng(12)
    n = _THEIL_SEN_LARGE + 200
    x = np.sort(rng.integers(0, n // 2, n)).astype(float)
    y = np.round(0.02 * x + rng.normal(0, 1.0, n), 1)
    assert _theil_sen_rust(x, y) == _theil_sen_large(x, y, scipy_stats)


def test_public_backend_switches_without_moving_a_number() -> None:
    """End to end on a datetime-indexed record above the switch."""
    n = _THEIL_SEN_LARGE + 200
    rng = np.random.default_rng(21)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    y = 0.5 + 0.0002 * np.arange(n) + rng.normal(0.0, 0.1, n)
    series = pd.Series(y, index=idx)
    default = theil_sen(series)
    rust = theil_sen(series, backend="rust")
    assert rust.slope == default.slope
    assert rust.ci_low == default.ci_low
    assert rust.ci_high == default.ci_high
    assert rust.n == default.n == n
    assert rust.time_unit == default.time_unit == "day"


def test_constant_x_warns_and_returns_nan() -> None:
    """No distinct x pair means no slope; the warning mirrors scipy's."""
    y = np.arange(10, dtype=float)
    with pytest.warns(RuntimeWarning, match="identical"):
        slope, low, high = _theil_sen_rust(np.zeros(10), y)
    assert np.isnan(slope) and np.isnan(low) and np.isnan(high)


def test_unknown_backend_raises_config_error() -> None:
    series = pd.Series(np.arange(10, dtype=float))
    with pytest.raises(ConfigError, match="backend"):
        theil_sen(series, backend="numba")


def test_runtime_probe_reports_the_imported_extension() -> None:
    assert rust_available() is True


# ---------------------------------------------------------------------------
# Selection-path tests (2026-11-27): the Rust kernel no longer materialises
# the pair multiset above 4 194 304 slopes. The window and radix branches
# must be bit-identical to the materialised path, which the public backend
# already runs below the cutoff.
# ---------------------------------------------------------------------------


def _z() -> float:
    from scipy import stats as scipy_stats

    return float(scipy_stats.norm.ppf(0.025))


def _bits(value: float) -> int:
    return int(np.float64(value).view(np.uint64))


def _assert_same_tuple(got: tuple, want: tuple, what: str) -> None:
    names = ("slope", "low", "high", "n_pairs", "degenerate")
    for name, a, b in zip(names, got, want, strict=True):
        if name in ("slope", "low", "high"):
            assert _bits(a) == _bits(b), f"{what}: {name}"
        else:
            assert a == b, f"{what}: {name} {a!r} != {b!r}"


def test_pair_cutoff_sits_at_4194304_pairs() -> None:
    """m = n(n-1)/2 crosses 1 << 22 between 2 896 and 2 897 distinct x."""
    from scipy import stats as scipy_stats

    for n, pairs in ((2896, 4_191_960), (2897, 4_194_856)):
        x, y = _linear_record(n, seed=n)
        ref = scipy_stats.theilslopes(y, x)
        assert _theil_sen_rust(x, y) == (ref.slope, ref.low_slope, ref.high_slope)
        assert normet_rust.theil_sen_interval(x, y, _z())[3] == pairs


def test_cutoff_is_on_pairs_not_points() -> None:
    from scipy import stats as scipy_stats

    y = 0.1 * np.arange(4000, dtype=float)
    x = np.array([0.0] * 2000 + [1.0] * 2000)
    ref = scipy_stats.theilslopes(y, x)
    assert _theil_sen_rust(x, y) == (ref.slope, ref.low_slope, ref.high_slope)
    assert normet_rust.theil_sen_interval(x, y, _z())[3] == 4_000_000

    y = 0.1 * np.arange(4100, dtype=float)
    x = np.array([0.0] * 2050 + [1.0] * 2050)
    ref = _theil_sen_large(x, y, scipy_stats)
    assert _theil_sen_rust(x, y) == ref
    assert normet_rust.theil_sen_interval(x, y, _z())[3] == 4_202_500


def test_window_path_with_duplicate_days_and_tied_values() -> None:
    from scipy import stats as scipy_stats

    rng = np.random.default_rng(314)
    n = 4100
    x = np.sort(rng.integers(0, n // 2, n)).astype(float)
    y = np.round(0.02 * x + rng.normal(0.0, 1.0, n), 1)
    ref = _theil_sen_large(x, y, scipy_stats)
    assert _theil_sen_rust(x, y) == ref

    perm = rng.permutation(n)
    base = np.array(_theil_sen_rust(x, y), dtype=float).view(np.uint64)
    shuffled = np.array(_theil_sen_rust(x[perm], y[perm]), dtype=float).view(np.uint64)
    assert np.array_equal(base, shuffled)


def test_uniform_and_constant_slope_records() -> None:
    from scipy import stats as scipy_stats

    n = 5000
    x = np.arange(n, dtype=float)
    z = _z()
    for a in (2.0, -3.5, 0.0):
        y = a * x
        slope, low, high, n_pairs, degenerate = normet_rust.theil_sen_interval(x, y, z)
        assert (slope, low, high) == (a, a, a)
        assert n_pairs == 12_497_500
        assert degenerate is False

    y = 2.0 * x
    assert _theil_sen_rust(x, y) == _theil_sen_large(x, y, scipy_stats)

    y = np.zeros(n)
    slope, low, high, _, _ = normet_rust.theil_sen_interval(x, y, z)
    assert _bits(slope) == _bits(0.0)
    assert _bits(low) == _bits(0.0)
    assert _bits(high) == _bits(0.0)


def test_signed_zero_record_is_deterministic() -> None:
    """The u64 key order makes -0.0 < +0.0; numpy's sort may mix the signs."""
    n = 3000
    x = np.arange(n, dtype=float)
    y = np.concatenate([np.zeros(1500), np.full(1500, -0.0)])
    slope, low, high, _, _ = normet_rust.theil_sen_interval(x, y, _z())
    for value in (slope, low, high):
        assert _bits(value) == _bits(-0.0)

    x4 = np.array([0.0, 1.0, 2.0, 3.0])
    for y4 in (
        np.array([0.0, 0.0, -0.0, -0.0]),
        np.array([0.0, -0.0, 0.0, -0.0]),
    ):
        default = normet_rust.theil_sen_interval(x4, y4, _z())
        for cfg in ((0, 1, 1), (0, 2, 1 << 20)):
            forced = normet_rust.theil_sen_interval_cfg(x4, y4, _z(), *cfg)
            _assert_same_tuple(forced, default, f"cfg={cfg}")
    assert _bits(normet_rust.theil_sen_interval(x4, x4 * 0 + 0.0, _z())[0]) == _bits(0.0)


def test_window_path_is_permutation_invariant() -> None:
    from scipy import stats as scipy_stats

    n = 4200
    x, y = _linear_record(n, seed=42)
    ref = _theil_sen_large(x, y, scipy_stats)
    assert _theil_sen_rust(x, y) == ref

    base = np.array(_theil_sen_rust(x, y), dtype=float).view(np.uint64)
    for perm in (np.arange(n)[::-1], (np.arange(n) * 13 + 777) % n):
        got = _theil_sen_rust(x[perm], y[perm])
        assert np.array_equal(base, np.array(got, dtype=float).view(np.uint64))


def test_overflow_slopes_take_the_exact_fallback() -> None:
    """Finite y near 1e308 overflows the differences into +/-inf slopes."""
    from scipy import stats as scipy_stats

    n = 8700
    x = np.arange(n, dtype=float)
    y = np.full(n, 1e308)
    y[1::2] = -1e308
    with np.errstate(over="ignore", invalid="ignore"):
        ref = _theil_sen_large(x, y, scipy_stats)
        got = _theil_sen_rust(x, y)
    assert got == ref == (0.0, 0.0, 0.0)
    assert normet_rust.theil_sen_interval(x, y, _z())[3] == 37_840_650


def test_forced_configs_are_bit_identical_to_the_public_kernel() -> None:
    from scipy import stats as scipy_stats

    rng = np.random.default_rng(400)
    n = 400
    x = np.arange(n, dtype=float)
    records = [
        ("canonical", *_linear_record(n, seed=400)),
        ("duplicate-x", rng.integers(0, n // 2, n).astype(float), rng.normal(0.0, 1.0, n)),
        ("rounded-y", x, np.round(0.02 * x + rng.normal(0.0, 1.0, n), 1)),
        ("linear2", x, 2.0 * x),
        ("constant-x", np.zeros(50), np.arange(50, dtype=float)),
    ]
    z = _z()
    configs = (
        (0, 8, 4),
        (0, 64, 32),
        (0, 1, 1),
        (0, 1024, 1 << 20),
        (1 << 22, 1 << 17, 1 << 24),
    )
    for name, rx, ry in records:
        want = normet_rust.theil_sen_interval(rx, ry, z)
        for cfg in configs:
            got = normet_rust.theil_sen_interval_cfg(rx, ry, z, *cfg)
            _assert_same_tuple(got, want, f"{name} cfg={cfg}")

    x, y = _linear_record(n, seed=401)
    ref = scipy_stats.theilslopes(y, x)
    assert _theil_sen_rust(x, y) == (ref.slope, ref.low_slope, ref.high_slope)


def test_public_backend_at_the_pair_cutoff_matches_the_default_backend() -> None:
    """2 897 hourly points is the first record the window path serves."""
    n = 2897
    rng = np.random.default_rng(21)
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    y = 0.5 + 0.0002 * np.arange(n) + rng.normal(0.0, 0.1, n)
    series = pd.Series(y, index=idx)
    default = theil_sen(series)
    rust = theil_sen(series, backend="rust")
    assert rust.slope == default.slope
    assert rust.ci_low == default.ci_low
    assert rust.ci_high == default.ci_high
    assert rust.time_unit == default.time_unit == "day"


def test_n_pairs_and_degenerate_from_the_binding_are_unchanged() -> None:
    x = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
    y = np.array([5.0, 7.0, 4.0, 6.0, 7.0, 8.0, 9.0, 7.0, 10.0, 12.0])
    slope, low, high, n_pairs, degenerate = normet_rust.theil_sen_interval(x, y, _z())
    assert n_pairs == 40
    assert degenerate is False
    assert slope == 1.291_666_666_666_666_5
    assert low == 0.5
    assert high == 2.0

    x = np.zeros(10)
    slope, low, high, n_pairs, degenerate = normet_rust.theil_sen_interval(x, np.arange(10.0), _z())
    assert (n_pairs, degenerate) == (0, True)
    assert np.isnan(slope) and np.isnan(low) and np.isnan(high)
    for cfg in ((0, 1, 1), (0, 64, 32)):
        forced = normet_rust.theil_sen_interval_cfg(x, np.arange(10.0), _z(), *cfg)
        assert forced[3:] == (0, True)
        assert np.isnan(forced[0])
