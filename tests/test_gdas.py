"""Tests for the GDAS1 met-data downloader (filename logic; no network)."""

from __future__ import annotations

import pandas as pd

from normet.io import gdas


def test_gdas1_week_boundaries():
    assert gdas._gdas1_week(1) == 1
    assert gdas._gdas1_week(7) == 1
    assert gdas._gdas1_week(8) == 2
    assert gdas._gdas1_week(28) == 4
    assert gdas._gdas1_week(29) == 5
    assert gdas._gdas1_week(31) == 5


def test_gdas1_filenames_single_week():
    assert gdas.gdas1_filenames("2020-04-05", "2020-04-06") == ["gdas1.apr20.w1"]


def test_gdas1_filenames_week_boundary():
    assert gdas.gdas1_filenames("2020-04-07", "2020-04-08") == [
        "gdas1.apr20.w1",
        "gdas1.apr20.w2",
    ]


def test_gdas1_filenames_cross_month_chronological_unique():
    # 30 Apr -> w5, 1 May -> w1; ordered, de-duplicated
    assert gdas.gdas1_filenames("2020-04-30", "2020-05-01") == [
        "gdas1.apr20.w5",
        "gdas1.may20.w1",
    ]


def test_gdas1_file_range_weeks():
    ts = pd.Timestamp
    assert gdas._gdas1_file_range("gdas1.apr20.w1") == (ts("2020-04-01"), ts("2020-04-07 23:59:59"))
    assert gdas._gdas1_file_range("/cache/dir/gdas1.apr20.w2") == (
        ts("2020-04-08"),
        ts("2020-04-14 23:59:59"),
    )
    # w5 runs to the end of the month, however long it is.
    assert gdas._gdas1_file_range("gdas1.apr20.w5") == (ts("2020-04-29"), ts("2020-04-30 23:59:59"))
    assert gdas._gdas1_file_range("gdas1.dec20.w5") == (ts("2020-12-29"), ts("2020-12-31 23:59:59"))


def test_gdas1_file_range_short_month_is_clamped():
    # A leap-year February has a 1-day w5; w4 must not spill into March either.
    ts = pd.Timestamp
    assert gdas._gdas1_file_range("gdas1.feb20.w5") == (ts("2020-02-29"), ts("2020-02-29 23:59:59"))
    assert gdas._gdas1_file_range("gdas1.feb21.w4") == (ts("2021-02-22"), ts("2021-02-28 23:59:59"))


def test_gdas1_file_range_unrecognised_name_is_none():
    assert gdas._gdas1_file_range("oct1618.BIN") is None
    assert gdas._gdas1_file_range("gdas1.xyz20.w1") is None  # not a month


def test_gdas1_file_range_agrees_with_filenames():
    # Every day of a year must fall inside the span of the file gdas1_filenames names for it.
    for day in pd.date_range("2020-01-01", "2020-12-31", freq="D"):
        (name,) = gdas.gdas1_filenames(day, day)
        start, end = gdas._gdas1_file_range(name)
        assert start <= day <= end, (day, name)


def test_gdas1_filenames_reversed_range_is_normalised():
    assert gdas.gdas1_filenames("2020-01-31", "2020-01-29") == ["gdas1.jan20.w5"]
