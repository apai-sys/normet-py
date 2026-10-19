"""Permissive timestamp parsing for user-supplied date columns."""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any

import pandas as pd

from .logging import get_logger

log = get_logger(__name__)

__all__ = ["to_datetime_coerced"]

# pandas 2.x's notice, and pandas 3.x's error, for values whose UTC offsets differ.
_MIXED_ZONES_WARNING = ".*parsing datetimes with mixed time zones"
_MIXED_ZONES_ERROR = "Mixed timezones detected"


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

    Values whose UTC offsets differ -- local time across a daylight-saving
    change, ``+00:00`` in winter and ``+01:00`` in summer -- have no one time
    zone to keep, so unless ``utc`` is given they are converted to UTC (a value
    with no offset among them is read as UTC). pandas would otherwise return
    them as objects that no ``.dt`` accessor or ``DatetimeIndex`` takes
    (pandas 2.x, with a warning), or raise (pandas 3.x).

    An explicit ``format`` in *kwargs* wins.
    """
    kwargs.setdefault("format", "mixed")
    if "utc" in kwargs:
        return pd.to_datetime(values, errors="coerce", **kwargs)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", _MIXED_ZONES_WARNING, FutureWarning)
            out = pd.to_datetime(values, errors="coerce", **kwargs)
    except ValueError as exc:
        if _MIXED_ZONES_ERROR not in str(exc):
            raise
    else:
        if not _mixes_offsets(out):
            return out
    log.info("Dates with differing UTC offsets (daylight saving, say) were converted to UTC.")
    return pd.to_datetime(values, errors="coerce", utc=True, **kwargs)


def _mixes_offsets(parsed: Any) -> bool:
    """Whether pandas 2.x fell back to objects because the offsets differ."""
    if not isinstance(parsed, pd.Series | pd.Index) or parsed.dtype != object:
        return False
    return any(isinstance(v, datetime) and v.tzinfo is not None for v in parsed)
