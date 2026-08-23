"""Permissive timestamp parsing for user-supplied date columns."""

from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = ["to_datetime_coerced"]


def to_datetime_coerced(values: Any, **kwargs: Any) -> Any:
    """Parse mixed-format dates element by element, coercing failures to ``NaT``.

    Date columns arrive from CSVs, config files and GUI pickers, so a single
    format cannot be assumed. Left to itself ``pd.to_datetime`` infers one from
    the first value and applies it to the rest, so an ISO date followed by
    ``"01/02/2024"`` silently drops the second value -- no error, no warning,
    just ``NaT``. ``format="mixed"`` parses each value on its own terms instead,
    and stays quiet about it: without it pandas emits "Could not infer format,
    so each element will be parsed individually" whenever the first value is
    unparseable, which is noise for callers that check ``NaT`` on the next line.

    An explicit ``format`` in *kwargs* wins.
    """
    kwargs.setdefault("format", "mixed")
    return pd.to_datetime(values, errors="coerce", **kwargs)
