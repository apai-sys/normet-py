"""Tests for normet.gui._multiscale (pure pandas, no Qt/matplotlib needed).

`normet.rolling` is mocked (returning hand-built ``rolling_<i>`` columns) so
these verify the aggregation/differencing/gating logic in isolation, the
same convention used by test_analysis_decomposition.py for `normalise`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import normet
from normet.gui import _multiscale as ms


def _df_prep(n_days: int, start: str = "2020-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame({"date": dates, "value": np.arange(n_days, dtype=float)})


def _rolling_result(dates: pd.DatetimeIndex, n_windows: int, level: float) -> pd.DataFrame:
    """A hand-built normet.rolling()-shaped result: n_windows identical
    columns holding *level* everywhere (so the row-wise mean is exactly
    *level*, regardless of window mechanics)."""
    result = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"))
    result["observed"] = 1.0
    for i in range(n_windows):
        result[f"rolling_{i}"] = level
    return result


@pytest.fixture()
def patch_rolling(monkeypatch):
    """Install a fake normet.rolling(); `_multiscale.py` imports it lazily
    (`from normet import rolling` inside the function body) so patching the
    attribute on the real `normet` package affects the next call."""

    calls: list[dict] = []

    def _install(dispatch):
        def fake_rolling(*, df, window_days, **kwargs):
            calls.append({"window_days": window_days, **kwargs})
            return dispatch(df, window_days)

        monkeypatch.setattr(normet, "rolling", fake_rolling)
        return calls

    return _install


def test_rolling_mean_series_span_too_short(patch_rolling):
    patch_rolling(lambda df, w: (_ for _ in ()).throw(AssertionError("should not be called")))
    df = _df_prep(30)
    series, n_windows, reason = ms.rolling_mean_series(
        df_prep=df,
        model=object(),
        covariates=["t2m"],
        variables_resample=["t2m"],
        window_days=90,
        n_samples=10,
        n_cores=1,
    )
    assert series is None
    assert n_windows == 0
    assert "30 d" in reason and "90 d window" in reason


def test_rolling_mean_series_below_min_windows_is_skipped(patch_rolling):
    df = _df_prep(100)
    patch_rolling(lambda d, w: _rolling_result(d["date"], n_windows=1, level=5.0))
    series, n_windows, reason = ms.rolling_mean_series(
        df_prep=df,
        model=object(),
        covariates=["t2m"],
        variables_resample=["t2m"],
        window_days=90,
        n_samples=10,
        n_cores=1,
    )
    assert series is None
    assert n_windows == 1
    assert "1 window" in reason and "90 d" in reason


def test_rolling_mean_series_aggregates_row_mean(patch_rolling):
    df = _df_prep(60)
    patch_rolling(lambda d, w: _rolling_result(d["date"], n_windows=3, level=7.5))
    series, n_windows, reason = ms.rolling_mean_series(
        df_prep=df,
        model=object(),
        covariates=["t2m"],
        variables_resample=["t2m"],
        window_days=14,
        n_samples=10,
        n_cores=1,
    )
    assert reason is None
    assert n_windows == 3
    assert series is not None
    assert (series == 7.5).all()


def test_compute_multiscale_all_bands_available(patch_rolling):
    df = _df_prep(400)

    def dispatch(d, window_days):
        level = {14: 10.0, 90: 8.0, 365: 6.0}[window_days]
        return _rolling_result(d["date"], n_windows=3, level=level)

    patch_rolling(dispatch)
    y_inf = pd.Series(5.0, index=pd.date_range("2020-01-01", periods=400, freq="D"))

    out = ms.compute_multiscale(
        df_prep=df,
        model=object(),
        covariates=["t2m"],
        variables_resample=["t2m"],
        y_inf=y_inf,
        fast_days=14,
        meso_days=90,
        slow_days=365,
        n_samples=10,
        n_cores=1,
    )
    assert out["notes"] == []
    assert (out["D_fast"] == 10.0 - 8.0).all()
    assert (out["D_meso"] == 8.0 - 6.0).all()
    assert (out["D_slow"] == 6.0 - 5.0).all()


def test_compute_multiscale_skips_unavailable_slow_scale(patch_rolling):
    """A record too short for the slow (365 d) scale still yields D_fast."""
    df = _df_prep(120)  # long enough for fast(14)/meso(90), not slow(365)

    def dispatch(d, window_days):
        level = {14: 10.0, 90: 8.0}[window_days]
        return _rolling_result(d["date"], n_windows=3, level=level)

    patch_rolling(dispatch)
    y_inf = pd.Series(5.0, index=pd.date_range("2020-01-01", periods=120, freq="D"))

    out = ms.compute_multiscale(
        df_prep=df,
        model=object(),
        covariates=["t2m"],
        variables_resample=["t2m"],
        y_inf=y_inf,
        fast_days=14,
        meso_days=90,
        slow_days=365,
        n_samples=10,
        n_cores=1,
    )
    assert out["Y_slow"] is None
    assert out["D_meso"] is None
    assert out["D_slow"] is None
    assert (out["D_fast"] == 10.0 - 8.0).all()
    assert len(out["notes"]) == 1
    assert "365 d scale skipped" in out["notes"][0]
