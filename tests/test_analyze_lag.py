import numpy as np
import pandas as pd
import pytest

from normet.utils.featureeng import LagDiagnostics, analyze_lag

statsmodels = pytest.importorskip("statsmodels")


def _make_driver_response(n=2000, true_lag=6, seed=0):
    """AR(1) driver; response = driver shifted by `true_lag` plus noise.

    Both series carry a shared diurnal cycle so that a naive (non-prewhitened)
    CCF would be confounded by seasonality.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    season = np.sin(2 * np.pi * t / 24.0)

    driver = np.zeros(n)
    for i in range(1, n):
        driver[i] = 0.6 * driver[i - 1] + rng.normal(0, 1)
    driver = driver + 2.0 * season

    resp = np.full(n, np.nan)
    resp[true_lag:] = 0.8 * driver[:-true_lag] + 1.5 * season[true_lag:]
    resp += rng.normal(0, 0.3, n)

    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="h"),
            "ws": driver,
            "pm25": resp,
        }
    )


def test_acf_pacf_only_when_no_driver():
    df = _make_driver_response()
    res = analyze_lag(df, target="pm25", max_lag=24)
    assert isinstance(res, LagDiagnostics)
    assert res.driver is None
    assert res.ccf is None
    # ACF lag 0 is exactly 1.0
    assert res.acf.loc[res.acf["lag"] == 0, "value"].iloc[0] == pytest.approx(1.0)
    # A strongly autocorrelated series flags at least one AR lag.
    assert res.target_ar_lags


def test_prewhitened_ccf_recovers_true_lag():
    true_lag = 6
    df = _make_driver_response(true_lag=true_lag)
    res = analyze_lag(df, target="pm25", driver="ws", max_lag=24, prewhiten=True)
    assert res.prewhitened is True
    assert res.ccf is not None
    # The peak driver-leading lag should match the injected lag.
    assert res.peak_lag == true_lag
    assert true_lag in res.driver_lags


def test_ccf_lag_orientation_is_positive_for_driver_leading():
    # Peak must be on the positive (driver-leads-target) side, not negative.
    df = _make_driver_response(true_lag=4)
    res = analyze_lag(df, target="pm25", driver="ws", max_lag=24, prewhiten=True)
    assert res.peak_lag is not None and res.peak_lag > 0


def test_significance_band_shrinks_with_n():
    small = analyze_lag(_make_driver_response(n=300), target="pm25", driver="ws", max_lag=12)
    large = analyze_lag(_make_driver_response(n=3000), target="pm25", driver="ws", max_lag=12)
    assert large.band < small.band


def test_summary_is_stringy():
    df = _make_driver_response()
    res = analyze_lag(df, target="pm25", driver="ws", max_lag=12)
    s = res.summary()
    assert "pm25" in s and "ws" in s


def test_bad_max_lag_raises():
    df = _make_driver_response(n=100)
    with pytest.raises(ValueError):
        analyze_lag(df, target="pm25", max_lag=0)
