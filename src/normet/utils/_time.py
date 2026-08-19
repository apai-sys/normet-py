"""Permissive timestamp parsing for user-supplied date columns."""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

__all__ = ["to_datetime_coerced"]


def to_datetime_coerced(values: Any, **kwargs: Any) -> Any:
    """``pd.to_datetime(values, errors="coerce")`` without the format notice.

    pandas infers a format from the first value and applies it to the rest,
    coercing whatever does not match to ``NaT``. When that first value is itself
    unparseable it cannot infer anything, falls back to parsing each element with
    dateutil, and emits "Could not infer format, so each element will be parsed
    individually". That case is the one this package's date-taking entry points
    are built for -- every caller checks the resulting ``NaT`` values on the next
    line and reports them -- so the notice is redundant, and redundant at nine
    call sites.

    Only that message is silenced; any other ``UserWarning`` passes through.

    ``format="mixed"`` would tell pandas the same thing directly, but it landed
    in pandas 2.0 and this package still supports 1.5.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not infer format",
            category=UserWarning,
        )
        return pd.to_datetime(values, errors="coerce", **kwargs)
