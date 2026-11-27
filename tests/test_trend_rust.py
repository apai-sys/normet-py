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

pytest.importorskip("normet_rust")


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
