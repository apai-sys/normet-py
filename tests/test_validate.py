"""Tests for :mod:`normet.utils.validate`."""

import pandas as pd
import pytest

from normet.exceptions import DataError
from normet.utils.validate import (
    require_column,
    require_no_duplicates,
    require_no_nan_in,
    require_not_empty,
)


def test_require_not_empty_passes():
    df = pd.DataFrame({"x": [1, 2, 3]})
    require_not_empty(df)


def test_require_not_empty_raises():
    with pytest.raises(DataError, match="empty"):
        require_not_empty(pd.DataFrame())


def test_require_column_passes():
    df = pd.DataFrame({"a": [1], "b": [2]})
    require_column(df, "a")


def test_require_column_raises():
    with pytest.raises(DataError, match="not found"):
        require_column(pd.DataFrame({"a": [1]}), "b")


def test_require_no_nan_in_passes():
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    require_no_nan_in(df, ["x", "y"])


def test_require_no_nan_in_raises():
    df = pd.DataFrame({"x": [1.0, None], "y": [3.0, 4.0]})
    with pytest.raises(DataError, match="NaN"):
        require_no_nan_in(df, ["x"])


def test_require_no_nan_in_skips_missing_column():
    df = pd.DataFrame({"x": [1.0, None]})
    require_no_nan_in(df, ["y"])  # no error because y doesn't exist


def test_require_no_duplicates_passes():
    df = pd.DataFrame({"x": [1, 2, 3]})
    require_no_duplicates(df, "x")


def test_require_no_duplicates_raises():
    df = pd.DataFrame({"x": [1, 1, 2]})
    with pytest.raises(DataError, match="duplicate"):
        require_no_duplicates(df, "x")


def test_require_no_duplicates_skips_missing_column():
    df = pd.DataFrame({"x": [1, 2]})
    require_no_duplicates(df, "y")
