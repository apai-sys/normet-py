import numpy as np
import pandas as pd
import pytest

from normet.utils.featureeng import (
    add_lag_features,
    add_rolling_features,
    cyclical_encode,
    wind_to_uv,
)


@pytest.fixture
def short_series():
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="h"),
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def test_add_lag_features_shifts_correctly(short_series):
    out = add_lag_features(short_series, cols=["x"], lags=[1, 2])
    np.testing.assert_array_equal(out["x_lag1"].to_numpy()[1:], [1.0, 2.0, 3.0, 4.0, 5.0])
    assert pd.isna(out["x_lag1"].iloc[0])
    np.testing.assert_array_equal(out["x_lag2"].to_numpy()[2:], [1.0, 2.0, 3.0, 4.0])


def test_add_lag_features_groupwise():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="h").tolist() * 2,
            "site": ["A", "A", "A", "B", "B", "B"],
            "x": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )
    out = add_lag_features(df, cols=["x"], lags=[1], group_col="site")
    a = out[out["site"] == "A"]
    b = out[out["site"] == "B"]
    # First row of each group has NaN lag — they don't bleed across groups
    assert pd.isna(a["x_lag1"].iloc[0])
    assert pd.isna(b["x_lag1"].iloc[0])
    assert a["x_lag1"].iloc[1] == 1.0
    assert b["x_lag1"].iloc[1] == 10.0


def test_add_rolling_features_mean(short_series):
    out = add_rolling_features(short_series, cols=["x"], windows=[3], aggs=["mean"])
    # Trailing window of size 3
    np.testing.assert_allclose(out["x_roll3_mean"].iloc[2:].to_numpy(), [2.0, 3.0, 4.0, 5.0])


def test_cyclical_encode_hour():
    df = pd.DataFrame({"hour": [0, 6, 12, 18]})
    out = cyclical_encode(df, "hour", period=24)
    # hour=0 → (0,1); hour=12 → (0,-1)
    np.testing.assert_allclose(out["hour_sin"].iloc[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(out["hour_cos"].iloc[0], 1.0, atol=1e-12)
    np.testing.assert_allclose(out["hour_sin"].iloc[2], 0.0, atol=1e-12)
    np.testing.assert_allclose(out["hour_cos"].iloc[2], -1.0, atol=1e-12)


def test_wind_to_uv_meteorological_convention():
    # Wind FROM the north at 5 m/s → blows southward → v = -5, u = 0
    u, v = wind_to_uv([5.0], [0.0], convention="meteorological")
    np.testing.assert_allclose(u[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(v[0], -5.0, atol=1e-12)
    # Wind FROM the east → blows westward → u = -5, v = 0
    u, v = wind_to_uv([5.0], [90.0], convention="meteorological")
    np.testing.assert_allclose(u[0], -5.0, atol=1e-12)
    np.testing.assert_allclose(v[0], 0.0, atol=1e-12)
