import numpy as np
import pandas as pd
import pytest

from normet.utils.cv import time_series_cv


@pytest.fixture
def hourly_frame():
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=100, freq="h"),
            "x": np.arange(100, dtype=float),
        }
    )


def test_time_series_cv_basic_shapes(hourly_frame):
    folds = list(time_series_cv(hourly_frame, n_splits=4, test_size=10))
    assert len(folds) == 4
    for tr, te in folds:
        assert tr.size > 0 and te.size == 10
        # No overlap
        assert np.intersect1d(tr, te).size == 0
        # All test indices come after all train indices (temporal ordering)
        assert tr.max() < te.min()


def test_time_series_cv_gap_respected(hourly_frame):
    folds = list(time_series_cv(hourly_frame, n_splits=2, test_size=10, gap=5))
    for tr, te in folds:
        # The gap between train end and test start must be >= 5
        assert te.min() - tr.max() >= 5


def test_time_series_cv_max_train_window(hourly_frame):
    folds = list(time_series_cv(hourly_frame, n_splits=3, test_size=10, max_train_size=20))
    for tr, _ in folds:
        assert tr.size <= 20


def test_time_series_cv_invalid_params(hourly_frame):
    with pytest.raises(ValueError):
        list(time_series_cv(hourly_frame, n_splits=0))
    with pytest.raises(ValueError):
        list(time_series_cv(hourly_frame, n_splits=10, test_size=50))  # no train room


def test_time_series_cv_negative_gap():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=20, freq="h")})
    with pytest.raises(ValueError, match="gap"):
        list(time_series_cv(df, n_splits=2, gap=-1))


def test_time_series_cv_too_few_rows():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, freq="h")})
    with pytest.raises(ValueError, match="Need at least"):
        list(time_series_cv(df, n_splits=5))


def test_time_series_cv_datetimeindex():
    df = pd.DataFrame(
        {"x": np.arange(30.0)},
        index=pd.date_range("2024-01-01", periods=30, freq="h"),
    )
    folds = list(time_series_cv(df, n_splits=2, test_size=5))
    assert len(folds) == 2


def test_time_series_cv_no_date_column_no_datetimeindex():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="not found"):
        list(time_series_cv(df, n_splits=2))


def test_cv_score_raises_no_features():
    df = pd.DataFrame({"value": [1.0], "date": pd.Timestamp("2024-01-01")})
    with pytest.raises(ValueError, match="feature_names"):
        from normet.utils.cv import cv_score

        cv_score(df, feature_names=None)


def test_time_series_cv_empty_fold_logged(caplog):
    caplog.set_level("WARNING")
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=50, freq="h"),
            "x": np.arange(50.0),
        }
    )
    # n_splits=10, test_size=5, gap=0 → total_test=50 → no room
    with pytest.raises(ValueError, match="no room"):
        list(time_series_cv(df, n_splits=10, test_size=5, gap=0))


def test_time_series_cv_non_datetime_column_raises():
    df = pd.DataFrame({"date": [1, 2, 3], "x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="must be datetime"):
        list(time_series_cv(df, n_splits=2))


def test_time_series_cv_explicit_test_size():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=30, freq="h"),
            "x": np.arange(30.0),
        }
    )
    folds = list(time_series_cv(df, n_splits=2, test_size=3))
    for _, te in folds:
        assert len(te) == 3


def test_cv_score_raises_missing_value():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=24, freq="h"),
            "x": np.arange(24.0),
        }
    )
    from normet.utils.cv import cv_score

    with pytest.raises(ValueError, match="not found"):
        cv_score(df, value="y", feature_names=["x"])


def test_cv_score_bad_date_col():
    """Bad date column name should raise."""
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=24, freq="h"),
            "x": np.arange(24.0),
            "value": np.random.default_rng(42).random(24),
        }
    )
    from normet.utils.cv import cv_score

    with pytest.raises(ValueError, match="must be present"):
        cv_score(df, value="value", feature_names=["x"], date_col="bad_date")


def test_cv_score_no_train_room():
    """Raise when total_test covers the entire dataset."""
    from normet.utils.cv import cv_score

    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="h"),
            "x": np.arange(5.0),
            "value": np.arange(5.0),
        }
    )
    with pytest.raises(ValueError, match="no room"):
        cv_score(df, value="value", feature_names=["x"], n_splits=2, test_size=5)
