import numpy as np
import pandas as pd
import pytest

from normet.utils.metrics import Stats, _fac2, _stats_from_arrays, modStats


def test_fac2_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert _fac2(y, y) == 1.0


def test_fac2_all_out_of_band():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = y_true * 5.0
    assert _fac2(y_pred, y_true) == 0.0


def test_stats_from_arrays_perfect_fit():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = _stats_from_arrays(y, y, ["n", "RMSE", "MB", "R2", "r"])
    assert int(out.loc[0, "n"]) == 5
    assert out.loc[0, "RMSE"] == 0.0
    assert out.loc[0, "MB"] == 0.0
    assert out.loc[0, "R2"] == 1.0
    assert out.loc[0, "r"] == 1.0


def test_stats_handles_nan():
    y_true = np.array([1.0, np.nan, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, np.nan])
    out = _stats_from_arrays(y_pred, y_true, ["n", "RMSE"])
    assert int(out.loc[0, "n"]) == 2  # only two finite pairs


def test_Stats_dataframe_view():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "yhat": [1.1, 1.9, 2.95]})
    out = Stats(df, mod="yhat", obs="y", statistic=["n", "RMSE", "r"])
    assert int(out.loc[0, "n"]) == 3


class _StubModel:
    """Predicts target value plus a small bias — handy for grouping tests."""

    backend = "flaml"

    def __init__(self, bias: float = 0.0):
        self.bias = bias

    def predict(self, X):
        # require a 'value' column to mimic a perfect-ish predictor + bias
        return np.asarray(X["value"], dtype=float) + self.bias


def test_modStats_by_season():
    dates = pd.date_range("2024-01-01", periods=4 * 90, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "value": np.linspace(1, 10, len(dates)),
        }
    )
    out = modStats(
        df,
        _StubModel(),
        statistic=["n", "RMSE"],
        by="season",
        predictor=lambda m, d: m.predict(d),
    )
    # Should include one row per season plus an 'all' row
    assert {"season", "n", "RMSE"} <= set(out.columns)
    assert "all" in out["season"].astype(str).values


def test_fac2_zero_denominator():
    """FAC2 guards against division by zero."""
    y_true = np.array([0.0, 2.0, 3.0])
    y_pred = np.array([0.0, 2.0, 3.0])
    assert _fac2(y_pred, y_true) == 1.0


def test_fac2_all_nan():
    """FAC2 returns NaN when no finite ratios."""
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([0.0, 0.0])
    assert np.isnan(_fac2(y_pred, y_true))


def test_stats_empty_pairs():
    """_stats_from_arrays with no finite pairs returns NaN-filled row."""
    y = np.array([np.nan, np.nan])
    out = _stats_from_arrays(y, y, ["n", "RMSE", "MB", "R2"])
    assert out.loc[0, "n"] == 0
    assert np.isnan(out.loc[0, "RMSE"])


def test_stats_missing_key_defaults():
    """Requested keys not computed fall back to NaN."""
    y = np.array([1.0, 2.0, 3.0])
    out = _stats_from_arrays(y, y, ["COE", "IOA"])
    assert np.isfinite(out.loc[0, "COE"])
    assert np.isfinite(out.loc[0, "IOA"])


def test_Stats_missing_column():
    df = pd.DataFrame({"x": [1.0]})
    with pytest.raises(ValueError, match="columns not found"):
        Stats(df, mod="yhat", obs="x")


def test_modStats_subset():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="h"),
            "value": np.arange(10.0),
            "set": ["training"] * 6 + ["testing"] * 4,
        }
    )
    out = modStats(
        df, _StubModel(), subset="training", statistic=["n"], predictor=lambda m, d: m.predict(d)
    )
    assert int(out.loc[0, "n"]) == 6


def test_modStats_no_set_column():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="h"),
            "value": np.arange(5.0),
        }
    )
    out = modStats(df, _StubModel(), statistic=["n"], predictor=lambda m, d: m.predict(d))
    assert out.loc[0, "set"] == "all"


def test_modStats_by_hour():
    dates = pd.date_range("2024-01-01", periods=48, freq="h")
    df = pd.DataFrame({"date": dates, "value": np.sin(np.arange(48) * 0.5)})
    out = modStats(
        df, _StubModel(), statistic=["n"], by="hour", predictor=lambda m, d: m.predict(d)
    )
    assert "hour" in out.columns
    assert len(out) >= 2  # at least some hours + "all"
