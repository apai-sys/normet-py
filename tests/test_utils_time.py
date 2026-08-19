"""Tests for the permissive timestamp parser shared by the date-taking entry points."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from normet.utils._time import to_datetime_coerced


def test_unparseable_values_become_nat_rather_than_raising():
    """Callers reject NaT themselves; the parser's job is to report, not to raise."""
    out = to_datetime_coerced(pd.Series(["2024-01-01", "not a date", "2024-01-03"]))
    assert out.isna().tolist() == [False, True, False]
    assert out[0] == pd.Timestamp("2024-01-01")


def test_a_second_format_is_parsed_rather_than_silently_dropped():
    """The reason this helper exists, and the reason pandas>=2.0 is required.

    Left to itself pd.to_datetime infers a format from the first value and
    applies it to every other one, so "01/02/2024" after an ISO date became NaT
    with no error and no warning -- a date column half-destroyed in silence.
    format="mixed" parses each value on its own terms.
    """
    out = to_datetime_coerced(pd.Series(["2024-01-01", "01/02/2024"]))
    assert out.notna().all()
    assert out.tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]


def test_an_unparseable_first_value_does_not_warn():
    """Without format="mixed" pandas announces its per-element fallback here.

    The caller checks NaT on the very next line and raises with a message that
    names the real problem, so the notice was noise -- and it fired from nine
    call sites.
    """
    values = pd.Series(["not a date", "2024-01-01", "2024-01-02"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        out = to_datetime_coerced(values)
    assert out.isna().tolist() == [True, False, False]


def test_an_explicit_format_wins():
    """A caller who knows the format should get strict parsing, not the mixed path."""
    out = to_datetime_coerced(pd.Series(["2024-01-01", "01/02/2024"]), format="%Y-%m-%d")
    assert out.isna().tolist() == [False, True]


def test_keyword_arguments_reach_pandas():
    out = to_datetime_coerced(pd.Series(["2024-01-01 00:00:00"]), utc=True)
    assert out.dt.tz is not None
