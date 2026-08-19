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


def test_an_unparseable_first_value_does_not_warn():
    """This is the case that produced the warning, and the case callers handle.

    pandas can infer nothing from a leading garbage value, so it parses each
    element with dateutil and says so. The caller checks NaT on the very next
    line and raises with a message that names the real problem, which makes the
    notice noise -- but silencing it must not break the parse.
    """
    values = pd.Series(["not a date", "2024-01-01", "2024-01-02"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        out = to_datetime_coerced(values)
    assert out.isna().tolist() == [True, False, False]


def test_other_user_warnings_still_get_through():
    """The filter is scoped to pandas' format message, not to UserWarning at large."""
    with pytest.warns(UserWarning, match="unrelated"):
        warnings.warn("an unrelated user warning", UserWarning, stacklevel=1)
        to_datetime_coerced(pd.Series(["2024-01-01"]))


def test_a_format_inferred_from_the_first_value_is_applied_to_the_rest():
    """Pinning pandas' actual contract, which is sharper than it looks.

    The format comes from the first value and everything that does not match it
    becomes NaT -- silently, with no warning at all. So "01/02/2024" after an
    ISO date is dropped rather than parsed. This predates the helper and is not
    changed by it; the test exists so the behaviour is visible rather than
    discovered in the field.
    """
    out = to_datetime_coerced(pd.Series(["2024-01-01", "01/02/2024"]))
    assert out.isna().tolist() == [False, True]


def test_keyword_arguments_reach_pandas():
    out = to_datetime_coerced(pd.Series(["2024-01-01 00:00:00"]), utc=True)
    assert out.dt.tz is not None
