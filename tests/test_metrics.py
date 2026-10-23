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


# ---------------------------------------------------------------------------
# logMAE / logRMSE (accuracy in log1p space)
# ---------------------------------------------------------------------------


def test_log_metrics_perfect_fit():
    y = np.array([0.5, 2.0, 10.0, 100.0])
    out = _stats_from_arrays(y, y, ["logMAE", "logRMSE"])
    assert out.loc[0, "logMAE"] == 0.0
    assert out.loc[0, "logRMSE"] == 0.0


def test_log_metrics_known_values():
    y_true = np.array([1.0, 10.0, 100.0])
    y_pred = np.array([1.5, 12.0, 90.0])
    ldiff = np.log1p(y_pred) - np.log1p(y_true)
    out = _stats_from_arrays(y_pred, y_true, ["logMAE", "logRMSE"])
    assert np.isclose(out.loc[0, "logMAE"], float(np.mean(np.abs(ldiff))))
    assert np.isclose(out.loc[0, "logRMSE"], float(np.sqrt(np.mean(ldiff**2))))


def test_log_metrics_bound_proportional_errors_that_mge_does_not():
    # A constant 10% overprediction: MGE grows with the level (1 -> 100), the
    # log1p error stays bounded near a tenth of a log unit -- why a log fit is
    # judged in log space. (log1p, not log: the weight of a relative miss
    # grows from ~half a relative unit at low levels toward the full unit.)
    y = np.array([10.0, 100.0, 1000.0])
    yhat = 1.1 * y
    mge = np.abs(yhat - y)
    log_err = np.abs(np.log1p(yhat) - np.log1p(y))
    assert mge[-1] / mge[0] == 100.0
    assert log_err.max() < 0.1
    out = _stats_from_arrays(yhat, y, ["MGE", "logMAE"])
    assert np.isclose(out.loc[0, "logMAE"], float(np.mean(log_err)))


def test_log_metrics_drop_out_of_domain_pairs_only():
    # One pair below -1 (outside log1p's domain) is dropped from the log
    # metrics only; the linear statistics keep it.
    y_true = np.array([1.0, -2.0])
    y_pred = np.array([1.5, -1.5])
    out = _stats_from_arrays(y_pred, y_true, ["logMAE", "logRMSE", "MGE", "n"])
    assert int(out.loc[0, "n"]) == 2
    assert np.isfinite(out.loc[0, "MGE"])
    assert np.isclose(out.loc[0, "logMAE"], abs(np.log1p(1.5) - np.log1p(1.0)))
    assert np.isclose(out.loc[0, "logRMSE"], abs(np.log1p(1.5) - np.log1p(1.0)))


def test_log_metrics_all_out_of_domain_are_nan():
    out = _stats_from_arrays(np.array([-2.5, -3.5]), np.array([-2.0, -3.0]), ["logMAE", "logRMSE"])
    assert np.isnan(out.loc[0, "logMAE"])
    assert np.isnan(out.loc[0, "logRMSE"])


def test_default_stats_do_not_include_log_metrics():
    y = np.array([1.0, 2.0, 3.0])
    out = Stats(pd.DataFrame({"value_predict": y, "value": y}), mod="value_predict", obs="value")
    assert "logMAE" not in out.columns
    assert "logRMSE" not in out.columns
