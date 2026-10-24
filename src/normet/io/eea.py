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
- missing observations arrive as a numeric sentinel, converted to ``NaN`` here
  before the frame is returned: the bulk service writes ``-99``, the
  per-station Parquet blobs ``-999`` (measured on DEUB044's NO2 record,
  2026-10-24: 2,037 of its 26,304 rows, every one flagged invalid).

The service answers in Parquet, so reading it needs ``pyarrow`` (carried by
the ``[data]`` extra).

Station level
-------------
The bulk service above has no per-station query path: its ``cities``
parameter accepts neither the station (EoI) codes nor city names (both
probed empty against it, 2026-10-23). The agency's own web viewer therefore
lists stations from the ArcGIS map service and downloads *one Parquet file
per station and pollutant* from the public Azure blobs it publishes there.
:func:`list_eea_stations` (the map service, keyless, one country at a time)
and :func:`fetch_eea_station` (those blobs) follow exactly that mechanism;
together they are what feeds the GUI's per-station workflow.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterable
from typing import Any

import pandas as pd

from ..utils._lazy import require
from ..utils.logging import get_logger
from ._http import request_with_retry

log = get_logger(__name__)

__all__ = ["EEA_POLLUTANT_CODES", "fetch_eea_data", "list_eea_stations", "fetch_eea_station"]


EEA_POLLUTANT_CODES = {
    "PM2.5": 6001,
    "PM10": 5,
    "NO2": 8,
    "O3": 7,
    "SO2": 1,
    "CO": 10,
}

_NOTATION_BY_CODE = {code: name for name, code in EEA_POLLUTANT_CODES.items()}

#: The service's "no measurement" sentinels: the bulk download writes -99, the
#: per-station Parquet blobs -999 (both measured; see :func:`_tidy`).
_MISSING_BULK = -99.0
_MISSING_BLOB = -999.0

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
    # Two sentinels mean "no measurement": the bulk service writes -99, the
    # per-station Parquet blobs -999 (measured on DEUB044's NO2 record,
    # 2026-10-24: 2,037 of its 26,304 rows, every one flagged invalid). Left as
    # numbers they would poison every mean; a real concentration is never -99.
    values = values.mask(values.isin((_MISSING_BULK, _MISSING_BLOB)))
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


# ---------------------------------------------------------------------------
# Station level: the ArcGIS map service + its public per-pollutant blobs
# ---------------------------------------------------------------------------

# The agency's own web viewer lists stations here (keyless, no account) and
# serves one Parquet file per station and pollutant from the Azure blobs the
# layer's per-station HTML names. The bulk download service has no per-station
# query path, so this is the mechanism a single-station workflow must use.
_STATIONS_MAP = (
    "https://air.discomap.eea.europa.eu/arcgis/rest/services/AirQuality/"
    "AirQualityDownloadServiceEUMonitoringStations/MapServer/0/query"
)
# One ArcGIS query answers at most this many features; page with the offset.
_ARCGIS_PAGE = 2000

# The blob links sit in the station's popup HTML, one line per pollutant:
#   - Nitrogen dioxide (air): <a … href="….parquet">NO2</a>, ug/m<sup>3</sup>,
#     (2013 - 2019)
# The anchor text is exactly the reporting notation ("NO2", "PM2.5",
# "NOX as NO2"), so it — not the prose title — is what identifies a pollutant:
# the titles carry "<" ("Particulate matter < 2.5 µm") and share words with
# heavy-metal species ("Benzo(a)pyrene in PM10"), which makes them unreliable
# to match, while the anchor grammar is what the agency's own viewer keys on.
# The unit between the anchor and the year range carries markup
# ("ug/m<sup>3</sup>"), so it is skipped with [^()] rather than [^<].
_BLOB_LINE = re.compile(r'href="([^"]+\.parquet)"[^>]*>([^<]*)</a>[^()]*\((\d{4})\s*-\s*(\d{4})\)')
# Anchor text (normalised) -> normet notation. Anything else (PAHs, metals,
# benzene, black carbon) is not in the app's pollutant set and is skipped.
_NOTATION_BY_ANCHOR = {
    "pm2.5": "PM2.5",
    "pm10": "PM10",
    "no2": "NO2",
    "no": "NO",
    "nox as no2": "NOXasNO2",
    "o3": "O3",
    "so2": "SO2",
    "co": "CO",
}


def _notation_for_anchor(text: str) -> str | None:
    """normet notation for a popup anchor text, or None if not our set."""
    return _NOTATION_BY_ANCHOR.get(" ".join(text.lower().split()))


def _list_features(country: str) -> list[dict[str, Any]]:
    """Every station feature for *country* from the ArcGIS layer, paged."""
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        params = {
            "where": f"CountryCode='{country.upper()}'",
            "outFields": "*",
            "f": "geojson",
            "outSR": "4326",
            "resultOffset": offset,
            "resultRecordCount": _ARCGIS_PAGE,
        }
        resp = request_with_retry(
            _STATIONS_MAP, params=params, timeout=120.0, source="EEA:stations"
        )
        import json as _json

        data = _json.loads(resp.text)
        feats = data.get("features") or []
        rows.extend(feats)
        offset += len(feats)
        if len(feats) < _ARCGIS_PAGE:
            break
    return rows


def _parse_feature(f: dict[str, Any], country: str) -> dict[str, Any] | None:
    """One ArcGIS feature row -> (code, name, lat, lon, {notation: blob_url})."""
    a = f.get("properties") or f.get("attributes") or {}
    code = a.get("AirQualityStationEoICode") or a.get("AirQualityStation") or ""
    if not code:
        return None
    name = a.get("AQStationName") or ""
    g = f.get("geometry") or {}
    if "coordinates" in g:  # geojson
        lon, lat = g["coordinates"][:2]
    else:  # standard JSON
        lat, lon = g.get("y"), g.get("x")
    if lat is None or lon is None:
        return None
    html = a.get("PopupInfo") or ""
    pollutants: dict[str, str] = {}
    years: dict[str, tuple[int, int]] = {}
    for m in _BLOB_LINE.finditer(html):
        url, anchor, y0, y1 = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        norm = _notation_for_anchor(anchor)
        if norm and norm not in pollutants:
            pollutants[norm] = url
            years[norm] = (y0, y1)
    return {
        "site": name or str(code),
        "code": str(code),
        "site_type": "",
        "country": country.upper(),
        "lat": float(lat),
        "lon": float(lon),
        "pollutants": pollutants,
        "years": years,
    }


def list_eea_stations(
    *,
    country: str,
    pollutants: Iterable[str] | None = None,
) -> pd.DataFrame:
    """The stations of one EEA country that measure any of *pollutants*.

    Parameters
    ----------
    country : str
        ISO-2 code as EEA reporting uses it (``"GB"``, ``"DE"``, ``"FR"``,
        ``"PL"``, ...). One country at a time: the ArcGIS layer answers it
        without paging for every member state (a few hundred to under a
        thousand stations each).
    pollutants : iterable of str, optional
        Restrict to stations reporting any of these normet notations. When
        omitted, every station is returned (the picker then shows what it has).

    Returns
    -------
    pandas.DataFrame
        One row per station: ``site`` (the reporting name), ``code`` (the EoI
        id), ``lat``/``lon``, ``country``, ``pollutants`` (the ticked ones the
        station reports, comma-joined), ``n``, and ``links`` (a JSON object of
        notation -> public Parquet URL, kept so the fetch can reuse it without
        a second map-service round trip).
    """
    import json as _json

    feats = _list_features(country)
    rows = []
    for f in feats:
        p = _parse_feature(f, country)
        if p is None:
            continue
        have = [k for k in (list(pollutants) if pollutants else []) if k in p["pollutants"]]
        if pollutants and not have:
            continue
        rows.append(
            {
                "site": p["site"],
                "code": p["code"],
                "site_type": p["site_type"],
                "country": p["country"],
                "pollutants": ", ".join(have) if have else "see links",
                "n": len(have) if have else len(p["pollutants"]),
                "from": "",
                "to": "",
                "lat": p["lat"],
                "lon": p["lon"],
                # notation -> {url, from, to}: the picker reads the years to
                # prefill its date range, the fetch reads the url.
                "links": _json.dumps(
                    {
                        k: {
                            "url": u,
                            "from": p["years"][k][0],
                            "to": p["years"][k][1],
                        }
                        for k, u in p["pollutants"].items()
                    }
                ),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["n", "site"], ascending=[False, True]).reset_index(drop=True)
    return df


def fetch_eea_station(
    *,
    station: str,
    country: str,
    pollutant: str,
) -> pd.DataFrame:
    """One station's full history for one pollutant, from its public blob.

    The blob is one Parquet file holding the whole record (2013 to now) for
    that station and pollutant, so this is the cheapest possible per-station
    pull: a single download. Caller trims to the wanted date range.

    Parameters
    ----------
    station : str
        EoI id (``"GB0777A"``); resolved to the sampling-point blob via the
        ArcGIS layer (one map-service call).
    country : str
        ISO-2 code; narrows the station lookup to one feature set.
    pollutant : str
        A normet notation in :data:`EEA_POLLUTANT_CODES`.

    Raises
    ------
    RuntimeError
        If the station has no blob for that pollutant or it fails to download.
    """

    feats = _list_features(country)
    links: dict[str, str] = {}
    site_name = ""
    lat = lon = None
    for f in feats:
        p = _parse_feature(f, country)
        if p is not None and p["code"].upper() == str(station).upper():
            site_name, links, lat, lon = p["site"], p["pollutants"], p["lat"], p["lon"]
            break
    if not links:
        raise RuntimeError(f"No EEA station {station!r} found for country {country!r}.")
    url = links.get(pollutant)
    if url is None:
        raise RuntimeError(
            f"EEA station {station} ({site_name}) reports no {pollutant} blob. "
            f"Its pollutants: {', '.join(links) or 'none'}."
        )
    resp = request_with_retry(url, timeout=300.0, source="EEA:blob")
    require("pyarrow", hint="pip install 'normet[data]' -- the EEA service answers in Parquet")
    raw = pd.read_parquet(io.BytesIO(resp.content))
    if raw.empty:
        return pd.DataFrame()
    df = _tidy(raw, country)
    df = df[df["pollutant"] == pollutant]
    df["site"] = site_name or str(station)
    df["lat"] = lat
    df["lon"] = lon
    return df[
        [
            "date",
            "site",
            "country",
            "pollutant",
            "value",
            "unit",
            "aggregation",
            "validity",
            "lat",
            "lon",
        ]
    ].reset_index(drop=True)
