"""Tests for the GDAS1 met-data downloader (filename logic; no network)."""

from __future__ import annotations

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


def test_gdas1_filenames_reversed_range_is_normalised():
    assert gdas.gdas1_filenames("2020-01-31", "2020-01-29") == ["gdas1.jan20.w5"]
