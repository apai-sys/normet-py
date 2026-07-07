"""Regression tests — lock known input/output of core functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from normet.utils.cv import time_series_cv
from normet.utils.metrics import _DEFAULT_STATS, _fac2, _stats_from_arrays, modStats
from normet.utils.prepare import check_data, prepare_data, process_date, split_into_sets


class TestMetricsRegression:
    @pytest.fixture
    def y_true(self):
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    @pytest.fixture
    def y_pred_perfect(self):
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_fac2_perfect(self):
        assert _fac2(np.array([1.0]), np.array([1.0])) == 1.0

    def test_fac2_all_outside(self):
        assert _fac2(np.array([10.0]), np.array([1.0])) == 0.0

    def test_stats_from_arrays_shape(self, y_true, y_pred_perfect):
        row = _stats_from_arrays(y_pred_perfect, y_true, _DEFAULT_STATS)
        for k in _DEFAULT_STATS:
            assert k in row.columns
        assert int(row["n"].iloc[0]) == 5

    def test_stats_from_arrays_empty(self):
        row = _stats_from_arrays(np.array([]), np.array([]), ["n"])
        assert int(row["n"].iloc[0]) == 0

    def test_modStats_requires_model(self):
        df = pd.DataFrame({"value": [1.0], "date": pd.Timestamp("2024-01-01")})
        with pytest.raises((ValueError, TypeError)):
            modStats(df=df, model=None, subset=None, statistic=None)


class TestCVRegression:
    def test_time_series_cv_fold_count(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=100, freq="h"),
            }
        )
        folds = list(time_series_cv(df, n_splits=5, test_size=10))
        assert len(folds) == 5

    def test_time_series_cv_expanding_window(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=50, freq="h"),
            }
        )
        folds = list(time_series_cv(df, n_splits=4, test_size=5))
        # Train sizes should be non-decreasing (expanding window)
        train_sizes = [len(tr) for tr, _ in folds]
        assert all(train_sizes[i] <= train_sizes[i + 1] for i in range(len(train_sizes) - 1))

    def test_time_series_cv_sliding_window(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=100, freq="h"),
            }
        )
        folds = list(time_series_cv(df, n_splits=5, test_size=5, max_train_size=20))
        # Train sizes should all be <= 20
        for tr, _ in folds:
            assert len(tr) <= 20


class TestPrepareRegression:
    @pytest.fixture
    def df(self):
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=20, freq="D"),
                "PM2.5": np.arange(20.0, dtype=float),
                "t2m": np.random.default_rng(42).normal(10, 2, 20),
            }
        )

    def test_process_date_with_datetimeindex(self):
        df = pd.DataFrame(
            {"x": [1.0, 2.0]},
            index=pd.date_range("2024-01-01", periods=2, freq="h"),
        )
        out = process_date(df)
        assert "date" in out.columns
        assert isinstance(out["date"].dtype, pd.DatetimeTZDtype) or str(
            out["date"].dtype
        ).startswith("datetime64")

    def test_check_data_value_missing(self, df):
        with pytest.raises(ValueError, match="not in the DataFrame"):
            check_data(df, feature_names=["t2m"], value="NO2")

    def test_prepare_data_roundtrip(self, df):
        out = prepare_data(
            df,
            value="PM2.5",
            feature_names=["t2m"],
            split_method="ts",
            fraction=0.7,
            seed=42,
        )
        assert "value" in out.columns
        assert "set" in out.columns
        assert "training" in out["set"].values
        assert "testing" in out["set"].values
        assert out["set"].value_counts()["training"] == 14
        assert out["set"].value_counts()["testing"] == 6

    def test_prepare_data_split_methods(self, df):
        for method in ["random", "ts", "season", "month"]:
            out = prepare_data(
                df,
                value="PM2.5",
                feature_names=["t2m"],
                split_method=method,
                fraction=0.75,
                seed=42,
            )
            assert "set" in out.columns
            assert out["set"].nunique() == 2

    def test_split_into_sets_unknown_method(self, df):
        with pytest.raises(ValueError, match="Unknown"):
            split_into_sets(df, split_method="invalid", fraction=0.75, seed=42)
