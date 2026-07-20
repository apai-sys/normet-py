import numpy as np
import pandas as pd
import pytest

from normet.utils.prepare import (
    add_date_variables,
    check_data,
    impute_values,
    prepare_data,
    process_date,
    split_into_sets,
)


def test_process_date_from_datetime_index():
    n = 10
    df = pd.DataFrame({"x": range(n)}, index=pd.date_range("2024-01-01", periods=n, freq="h"))
    out = process_date(df)
    assert "date" in out.columns
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_process_date_coerces_string_column():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "x": [1, 2, 3],
        }
    )
    out = process_date(df)
    assert pd.api.types.is_datetime64_any_dtype(out["date"])


def test_process_date_raises_when_no_datetime():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        process_date(df)


def test_check_data_renames_target_to_value():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="h"),
            "PM2.5": [1.0, 2.0, 3.0, 4.0],
            "t2m": [10, 11, 12, 13],
        }
    )
    out = check_data(df, covariates=["t2m"], target="PM2.5")
    assert "value" in out.columns
    assert "PM2.5" not in out.columns


def test_impute_values_fills_numeric_with_median():
    df = pd.DataFrame(
        {
            "value": [1.0, 2.0, np.nan, 4.0],
            "x": [10.0, np.nan, np.nan, 40.0],
        }
    )
    out = impute_values(df, dropna=True)
    # NaN in target → row dropped
    assert len(out) == 3
    # NaN in feature → filled with median (which is 25.0 after target row drop)
    assert not out["x"].isna().any()


def test_add_date_variables_creates_time_features():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="h"),
            "value": [1.0, 2.0, 3.0],
        }
    )
    out = add_date_variables(df)
    for col in ("date_unix", "day_julian", "weekday", "hour"):
        assert col in out.columns


def test_split_into_sets_partitions_all_rows():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=100, freq="h"),
            "value": np.arange(100, dtype=float),
        }
    )
    for method in ("random", "ts", "month_ts", "season_ts"):
        out = split_into_sets(df, split_method=method, train_fraction=0.75, seed=0)
        assert set(out["set"].unique()) <= {"training", "testing"}
        assert len(out) == 100


def test_prepare_data_full_pipeline(synthetic_aq):
    out = prepare_data(
        synthetic_aq.copy(),
        target="PM2.5",
        covariates=["t2m", "blh", "u10", "v10"],
        split_method="ts",
        train_fraction=0.8,
    )
    assert "value" in out.columns
    assert "set" in out.columns
    for col in ("date_unix", "day_julian", "weekday", "hour"):
        assert col in out.columns
    # Time-ordered split: training rows precede testing rows in time.
    train_max = out.loc[out["set"] == "training", "date"].max()
    test_min = out.loc[out["set"] == "testing", "date"].min()
    assert train_max <= test_min
