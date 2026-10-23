# src/normet/io/eea.py
"""
European Environment Agency (EEA) air-quality download adapter.

Fetches measurement time series from the EEA's official Parquet download
service, which answers a JSON query document with a single Parquet file and
needs no account or key. The dataset behind it is the agency's "verified and
Up To Date" reporting: hourly measurements from 2013, reported by member
countries.

The adapter used to scrape the Discomap FME service instead
(``fme.discomap.eea.europa.eu/.../AQData_Extract.fmw``). That service is
gone: every request now answers ``401`` with
``WWW-Authenticate: Basic realm="FME Server Authentication"`` (observed
2026-10-23), and the agency's datahub points scripted access at the download
service above. Two consequences callers should know, beyond the call working
again:

- sites are identified by the reporting sampling-point id
  (``"GB/GB_SamplingPoint_65449"``), not by the FME-era station codes
  (``GB0001A``), and the response carries no latitude/longitude;
- missing observations arrive as the numeric sentinel ``-99``, converted to
  ``NaN`` here before the frame is returned.

The service answers in Parquet, so reading it needs ``pyarrow`` (carried by
the ``[data]`` extra).
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from typing import Any

import pandas as pd

from ..utils._lazy import require
from ..utils.logging import get_logger
from ._http import request_with_retry

log = get_logger(__name__)

__all__ = ["EEA_POLLUTANT_CODES", "fetch_eea_data"]


EEA_POLLUTANT_CODES = {
    "PM2.5": 6001,
    "PM10": 5,
    "NO2": 8,
    "O3": 7,
    "SO2": 1,
    "CO": 10,
}

_NOTATION_BY_CODE = {code: name for name, code in EEA_POLLUTANT_CODES.items()}

_DOWNLOAD_API = "https://eeadmz1-downloads-api-appservice.azurewebsites.net/ParquetFile/dynamic"

# The datahub lists exactly one dataset behind the service ("verified and Up
# To Date", 2013-now), and the request is rejected outright without it
# ("Dataset parameter is mandatory").
_DATASET = 1


def _month_windows(year_from: int, year_to: int) -> list[tuple[str, str]]:
    """Calendar-month ``(start, end)`` timestamp strings for the range.

    The service caps what one synchronous request may answer ("Request too
    large...", observed 2026-10-23 at six months for GB, while one month of
    Germany's PM10 -- 4.5 MB -- passes). A month per country and pollutant
    is therefore the unit of request; a year range becomes one POST per
    month, concatenated.
    """
    starts = pd.date_range(f"{int(year_from):04d}-01-01", f"{int(year_to):04d}-12-01", freq="MS")
    return [
        (
            s.strftime("%Y-%m-%dT00:00:00"),
            (s + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%dT23:59:59"),
        )
        for s in starts
    ]


def _resolve_pollutant_notation(pollutant: str | int) -> str:
    """The pollutant as the service names it: a notation, not a numeric code.

    Accepts the notations in :data:`EEA_POLLUTANT_CODES` (case-insensitive)
    and the EEA codes they map to, so the names that worked against the FME
    service keep working with the same meanings.
    """
    if isinstance(pollutant, int):
        notation = _NOTATION_BY_CODE.get(pollutant)
    else:
        key = str(pollutant).upper()
        notation = key if key in EEA_POLLUTANT_CODES else None
    if notation is None:
        valid = ", ".join(EEA_POLLUTANT_CODES)
        raise ValueError(f"Unknown pollutant '{pollutant}'. Known: {valid}")
    return notation


def _tidy(raw: pd.DataFrame, country: str) -> pd.DataFrame:
    """A Parquet response frame, renamed to the tidy normet columns.

    The mapping exists because the service speaks the reporting schema
    (``Samplingpoint``/``Value``/``AggType``/...) and every normet table
    speaks ``site``/``value``/``aggregation``. The pollutant's numeric code
    becomes the notation that names it, where one is known.
    """
    values = pd.to_numeric(raw["Value"], errors="coerce").astype(float)
    # -99 is EEA's sentinel for "no measurement" (a reporting convention, not
    # a concentration); left as a number it would poison every mean.
    values = values.mask(values == -99.0)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(raw["Start"], utc=True, errors="coerce"),
            "site": raw["Samplingpoint"].astype(str),
            "country": country.upper(),
            "pollutant": (
                pd.to_numeric(raw["Pollutant"])
                .astype(int)
                .map(lambda c: _NOTATION_BY_CODE.get(c, str(c)))
            ),
            "value": values,
            "unit": raw["Unit"].astype(str),
            "aggregation": raw["AggType"].astype(str),
            "validity": pd.to_numeric(raw["Validity"], errors="coerce").astype("Int64"),
        }
    )


def fetch_eea_data(
    *,
    country: str,
    pollutant: str | int,
    year_from: int,
    year_to: int,
    station: str | None = None,
    source: str = "All",
    keep_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """One country's and pollutant's hourly measurement table from the EEA.

    Parameters
    ----------
    country : str
        ISO-2 country code as the EEA reporting uses it (``"FR"``, ``"DE"``,
        ``"GB"``, ...). An unknown code is not an error: the service answers
        with no rows.
    pollutant : str or int
        Notation (``"PM2.5"``, ``"PM10"``, ``"NO2"``, ``"O3"``, ``"SO2"``,
        ``"CO"``) or the numeric EEA code (6001, 5, 8, 7, 1, 10).
    year_from, year_to : int
        Inclusive year range. The service caps what one synchronous request
        may answer ("Request too large..."), so the adapter asks one calendar
        month at a time and concatenates the results: expect one request per
        month of the range, and minutes of waiting for a multi-year pull.
    station : str, optional
        Keep only sampling points whose identifier contains this string
        (case-insensitive) -- e.g. ``"65449"`` for
        ``GB/GB_SamplingPoint_65449``. The service has no station parameter,
        so this filter runs client side, after the download.
    source : str, default "All"
        EEA reporting source (``"E1a"``, ``"E2a"``); the default asks for
        every source.
    keep_columns : iterable of str, optional
        Restrict the returned DataFrame to these columns.

    Returns
    -------
    pandas.DataFrame
        Long-format ``[date, site, country, pollutant, value, unit,
        aggregation, validity]``, sorted by ``site`` then ``date``. ``date``
        is the measurement's opening hour as UTC; ``site`` is the reporting
        sampling-point identifier; ``value`` has the service's ``-99``
        missing sentinel converted to ``NaN``; ``validity`` is the reported
        validity code (2 = validated -- filter on it if only validated data
        will do). No latitude/longitude: the service does not report them.
        An empty frame when the query matches nothing.

    Raises
    ------
    ValueError
        On an unknown pollutant notation or code.
    RuntimeError
        If the service request fails after retries.
    """
    notation = _resolve_pollutant_notation(pollutant)
    windows = _month_windows(year_from, year_to)
    log.info(
        "Querying the EEA download service for %s %s %d-%d (%d monthly request(s))",
        country,
        notation,
        year_from,
        year_to,
        len(windows),
    )
    require("pyarrow", hint="pip install 'normet[data]' -- the EEA service answers in Parquet")

    frames: list[pd.DataFrame] = []
    for t0, t1 in windows:
        body: dict[str, Any] = {
            "countries": [country.upper()],
            "cities": [],  # a required key; [] means "every city of the country"
            "pollutants": [notation],
            "dataset": _DATASET,
            "dateTimeStart": t0,
            "dateTimeEnd": t1,
            "compress": False,  # a bare Parquet file; the zip adds nothing here
        }
        if source and source.upper() != "ALL":
            body["source"] = source
        resp = request_with_retry(_DOWNLOAD_API, json=body, timeout=300.0, source="EEA")
        if not resp.content or not resp.content.startswith(b"PAR1"):
            # A month with nothing to report answers with a 22-byte empty zip,
            # not an error and not a (readable) empty Parquet file.
            continue
        raw = pd.read_parquet(io.BytesIO(resp.content))
        if not raw.empty:
            frames.append(_tidy(raw, country))

    if not frames:
        log.warning("EEA returned no rows for %s %s %d-%d.", country, notation, year_from, year_to)
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if station:
        df = df[df["site"].str.contains(str(station), case=False, regex=False)]

    df = df.sort_values(["site", "date"])
    default_keep = [
        "date",
        "site",
        "country",
        "pollutant",
        "value",
        "unit",
        "aggregation",
        "validity",
    ]
    cols = list(keep_columns) if keep_columns else [c for c in default_keep if c in df.columns]
    return df[cols].reset_index(drop=True)
