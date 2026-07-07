"""Tests for normet.analysis.events module."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from normet.analysis.events import _intervals_from_mask, _to_series, anomaly_scores, detect_events

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPECTED_COLS = {"start", "end", "n", "max_score", "mean_score"}


def _spike_series(n: int = 60, spike_idx: int = 10, spike_val: float = 200.0) -> pd.Series:
    """DatetimeIndex series with Gaussian noise plus one massive spike.

    Using random noise ensures IQR > 0 so the IQR scorer doesn't short-circuit.
    The spike value is far enough from the bulk that its score > k=3.0.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    values = rng.normal(5.0, 1.0, n)  # realistic spread → IQR > 0
    values[spike_idx] = spike_val  # massive outlier
    return pd.Series(values, index=dates, name="value")


def _uniform_series(n: int = 30, val: float = 5.0) -> pd.Series:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(np.full(n, val), index=dates, name="value")


# ---------------------------------------------------------------------------
# anomaly_scores tests
# ---------------------------------------------------------------------------


def test_anomaly_scores_iqr_basic():
    """Score at spike index must be highest; result is pd.Series with same index."""
    s = _spike_series()
    scores = anomaly_scores(s, method="iqr")

    assert isinstance(scores, pd.Series)
    assert scores.index.equals(s.index)
    # The spike at position 10 should have the maximum score
    assert scores.iloc[10] == scores.max()


def test_anomaly_scores_value_col():
    """Works when a DataFrame + value_col is supplied."""
    s = _spike_series()
    df = s.rename("pm25").to_frame()

    scores = anomaly_scores(df, value_col="pm25", method="iqr")

    assert isinstance(scores, pd.Series)
    assert len(scores) == len(df)
    assert scores.max() == scores.iloc[10]


def test_anomaly_scores_missing_col():
    """Raises ValueError when DataFrame passed without value_col."""
    df = _spike_series().to_frame()
    with pytest.raises(ValueError, match="value_col"):
        anomaly_scores(df, method="iqr")


def test_anomaly_scores_unknown_method():
    """Raises ValueError for unsupported method name."""
    s = _spike_series()
    with pytest.raises(ValueError, match="Unknown method"):
        anomaly_scores(s, method="xyz")


def test_anomaly_scores_isolation():
    """IsolationForest path returns a pd.Series of same length."""
    sklearn_available = importlib.util.find_spec("sklearn") is not None
    if not sklearn_available:
        pytest.skip("scikit-learn not installed")

    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    values = rng.normal(5.0, 1.0, 50)
    s = pd.Series(values, index=dates)

    scores = anomaly_scores(s, method="isolation")

    assert isinstance(scores, pd.Series)
    assert len(scores) == 50


# ---------------------------------------------------------------------------
# detect_events tests
# ---------------------------------------------------------------------------


def test_detect_events_iqr_finds_spike():
    """detect_events returns DataFrame with expected columns and at least 1 event."""
    s = _spike_series()
    result = detect_events(s, method="iqr")

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == _EXPECTED_COLS
    assert len(result) >= 1


def test_detect_events_empty_when_no_anomaly():
    """Uniform constant series should yield no events."""
    s = _uniform_series()
    result = detect_events(s, method="iqr")

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == _EXPECTED_COLS
    assert len(result) == 0


def test_detect_events_min_length():
    """min_length=3 should filter out single-point events."""
    # Create two isolated single-point spikes on a noisy background
    # (IQR must be > 0 for the scorer to work)
    rng = np.random.default_rng(7)
    n = 80
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    values = rng.normal(5.0, 0.5, n)  # noisy background → IQR > 0
    values[10] = 500.0  # single-point spike → run length 1
    values[50] = 500.0  # single-point spike → run length 1
    s = pd.Series(values, index=dates)

    result = detect_events(s, method="iqr", min_length=3)
    # All runs are length 1, so nothing passes min_length=3
    assert len(result) == 0


def test_detect_events_from_dataframe():
    """detect_events works when given a DataFrame with a named column."""
    s = _spike_series()
    df = s.rename("conc").to_frame()

    result = detect_events(df, value_col="conc", method="iqr")

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == _EXPECTED_COLS
    assert len(result) >= 1


def test_to_series_series_passthrough():
    s = pd.Series([1.0], index=pd.date_range("2024-01-01", periods=1))
    result = _to_series(s, None)
    assert result is s


def test_to_series_dataframe_missing_col():
    df = pd.DataFrame({"x": [1.0]})
    with pytest.raises(ValueError, match="value_col"):
        _to_series(df, None)


def test_to_series_converts_index():
    s = pd.Series([1.0], index=["2024-01-01"])
    result = _to_series(s, None)
    assert isinstance(result.index, pd.DatetimeIndex)


def test_intervals_from_mask_empty():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    out = _intervals_from_mask(
        idx,
        np.array([False, False, False, False, False]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert out.empty


def test_intervals_from_mask_tail_run():
    """When anomaly extends to the last element, the tail must be captured."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    mask = np.array([False, True, True, True, True])
    scores = np.array([0.0, 3.0, 4.0, 5.0, 6.0])
    out = _intervals_from_mask(idx, mask, scores)
    assert len(out) == 1
    assert out.iloc[0]["start"] == idx[1]
    assert out.iloc[0]["end"] == idx[-1]


def test_anomaly_scores_uniform_iqr_zero():
    """When IQR is 0 (all identical values), anomaly_scores should not crash."""
    s = pd.Series(np.full(10, 5.0), index=pd.date_range("2024-01-01", periods=10, freq="D"))
    scores = anomaly_scores(s, method="iqr")
    assert (scores == 0.0).all()
