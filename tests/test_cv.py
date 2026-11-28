import numpy as np
import pandas as pd
import pytest

from normet.utils.cv import cv_score, time_series_cv


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
    with pytest.raises(ValueError, match="covariates"):
        from normet.utils.cv import cv_score

        cv_score(df, covariates=None)


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
        cv_score(df, target="y", covariates=["x"])


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
        cv_score(df, target="value", covariates=["x"], date_col="bad_date")


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
        cv_score(df, target="value", covariates=["x"], n_splits=2, test_size=5)


def test_the_gap_is_counted_once_not_once_per_split():
    """Only the oldest fold's gap can empty its training set.

    With 30 rows, 5 splits of 5 and gap 2 the old check counted 5 gaps
    (35 >= 30) and refused a run whose first fold still had 3 training rows.
    """
    df = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=30, freq="h"), "x": np.arange(30.0)}
    )
    folds = list(time_series_cv(df, n_splits=5, test_size=5, gap=2))
    assert len(folds) == 5
    assert len(folds[0][0]) == 3  # the binding (oldest) fold
    # And a genuinely empty first fold is still refused.
    with pytest.raises(ValueError, match="no room"):
        list(time_series_cv(df, n_splits=5, test_size=5, gap=5))


# ---------------------------------------------------------------------------
# Fold-local imputation: the docstring's promise, not the full-frame median
# ---------------------------------------------------------------------------


def test_cv_score_fits_median_imputation_inside_the_fold_training_rows(monkeypatch):
    """``impute="median"`` inside ``cv_score`` must mean "train_median".

    ``impute_values(strategy="median")`` is the legacy full-frame fill; used
    on a fold it takes the median over the validation rows as well, so their
    scale leaks into the values the fold model is trained on.
    """
    from normet.utils import cv as cv_module

    seen: list[str] = []
    original = cv_module.impute_values

    def spy(df, **kwargs):
        seen.append(kwargs.get("strategy"))
        return original(df, **kwargs)

    monkeypatch.setattr(cv_module, "impute_values", spy)
    n = 90
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "x": rng.normal(size=n),
            "value": rng.normal(size=n) + 20.0,
        }
    )
    df.loc[5, "x"] = np.nan
    cv_score(
        df,
        target="value",
        covariates=["x"],
        backend="lightgbm",
        n_splits=3,
        impute="median",
        model_config={"n_estimators": 20},
        seed=1,
    )
    assert seen and set(seen) == {"train_median"}


def test_cv_score_median_imputation_ignores_validation_scale(monkeypatch):
    """Two frames differing only in the *validation* rows' scale score alike.

    With a full-frame median the fill value moves when the held-out rows
    change scale, and so do the out-of-fold predictions.
    """
    from normet.utils import cv as cv_module

    original = cv_module.impute_values
    current: list[float] = []
    runs: list[list[float]] = []

    def spy(df, **kwargs):
        out = original(df, **kwargs)
        if "x" in out.columns:
            current.append(float(out["x"].iloc[5]))
        return out

    monkeypatch.setattr(cv_module, "impute_values", spy)
    n = 90
    rng = np.random.default_rng(3)
    base = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="h"),
            "x": rng.normal(size=n),
            "value": rng.normal(size=n) + 20.0,
        }
    )
    base.loc[5, "x"] = np.nan
    for scale in (1e3, 1e6):
        df = base.copy()
        df.loc[df.index > 60, "x"] = df.loc[df.index > 60, "x"] + scale
        current = []
        cv_score(
            df,
            target="value",
            covariates=["x"],
            backend="lightgbm",
            n_splits=2,
            impute="median",
            model_config={"n_estimators": 10},
            seed=1,
        )
        runs.append(list(current))
    assert runs[0] and runs[1]
    assert runs[0] == pytest.approx(runs[1], rel=1e-12)
