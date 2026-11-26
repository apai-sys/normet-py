"""Parity tests for the optional Rust CRPS kernel.

The backend is opt-in (``crps_from_quantiles(..., backend="rust")`` and
``crps_from_samples(..., backend="rust")``), and switching must not move the
score: comparisons between runs are made on the number. Every test compares
exact floats against the pure-Python path, and the module skips when the
extension is not installed, so a source checkout still passes the suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from normet.analysis.verification import (
    _crps_rows,
    _crps_rows_rust,
    crps_from_quantiles,
    crps_from_samples,
)
from normet.core.exceptions import ConfigError

pytest.importorskip("normet_rust")


@pytest.mark.parametrize("rows, k", [(17, 5), (500, 64), (3, 257)])
def test_kernel_matches_the_python_row_loop(rows: int, k: int) -> None:
    """Same score bit for bit, with tied values and out-of-range observations."""
    rng = np.random.default_rng(rows * 1000 + k)
    levels = rng.uniform(0.001, 0.999, k)
    q = np.round(rng.normal(10.0, 2.0, (rows, k)), 1)  # ties on purpose
    y = np.round(rng.normal(10.0, 3.0, rows), 1)
    assert _crps_rows_rust(q, list(levels), y) == _crps_rows(q, list(levels), y)


@pytest.mark.parametrize("rows, k", [(20, 6), (400, 80)])
def test_samples_backend_matches_default(rows: int, k: int) -> None:
    rng = np.random.default_rng(rows * 10 + k)
    samples = rng.normal(10.0, 2.0, (rows, k))
    obs = rng.normal(10.0, 3.0, rows)
    # The scorer drops non-finite rows; both backends must do so identically.
    samples[::7, 0] = np.nan
    obs[::11] = np.nan
    default = crps_from_samples(samples, obs)
    assert crps_from_samples(samples, obs, backend="rust") == default


def test_quantile_frame_backend_matches_default() -> None:
    rng = np.random.default_rng(7)
    levels = np.linspace(0.025, 0.975, 39)
    columns = [f"q{int(round(1000 * level)):03d}" for level in levels]
    frame = pd.DataFrame(rng.normal(10.0, 2.0, (300, levels.size)), columns=columns)
    obs = pd.Series(rng.normal(10.0, 3.0, 300))
    default = crps_from_quantiles(frame, obs)
    assert crps_from_quantiles(frame, obs, backend="rust") == default


def test_constant_forecast_row_is_scored_identically() -> None:
    """A row whose values all tie is one CDF jump, the kernel's m = 1 case."""
    samples = np.full((5, 4), 2.0)
    samples[0] = [1.0, 2.0, 2.0, 3.0]
    obs = np.array([2.0, 1.0, 2.5, 4.0, 0.0])
    default = crps_from_samples(samples, obs)
    assert crps_from_samples(samples, obs, backend="rust") == default


def test_unknown_backend_raises_config_error() -> None:
    samples = np.arange(8, dtype=float).reshape(2, 4)
    obs = np.array([1.0, 2.0])
    with pytest.raises(ConfigError, match="backend"):
        crps_from_samples(samples, obs, backend="numba")
    with pytest.raises(ConfigError, match="backend"):
        crps_from_quantiles(samples, obs, quantiles=[0.1, 0.3, 0.6, 0.9], backend="numba")
