# src/normet/io/ukaq.py
"""UK air-quality network adapter (AURN, AQE, SAQN, WAQN, NI, LMAM).

Fetches hourly measurements and station metadata for UK air-quality
networks from two complementary backends behind one interface
(``source=`` on both :func:`list_ukaq_stations` and
:func:`fetch_ukaq_measurements`):

============  ==========================  ==================================
              ``source="aurn_live"``      ``source="aurn"`` (+ 5 more networks)
============  ==========================  ==================================
networks      AURN only                   AURN, AQE, SAQN, WAQN, NI, LMAM
backend       SOS REST API (JSON)         openair ``.RData`` archives
freshness     near real-time              whole calendar years, published
              (rolling window)            with a lag once a year is complete
site_type /   not available (SOS has      populated from network metadata
date range    no station classification)
============  ==========================  ==================================

Use ``aurn_live`` for the last few days/weeks of AURN; use an archive
source for anything historical, multi-network, or multi-year. This used
to be two separate modules (``ukaq`` for the archives, ``defra`` -- now
folded in below -- for the live API), briefly with ``defra`` marked
deprecated on the theory that its backend had gone permanently offline.
That theory was wrong: checked again 2026-08-05, ``sos-ukair`` answers
normally. The two were merged instead, once it was clear they are
genuinely complementary rather than one superseding the other.

Data sources
------------
Each archive network publishes ``{CODE}_{year}.RData`` (gzip-compressed R
serialisation) under its own base URL, plus a single metadata archive.
These are the same files openair's ``importUKAQ()`` reads, so results are
directly comparable with the R workflow. Parsing uses the pure-Python
``rdata`` package -- **no R installation is required**.

``aurn_live`` talks to DEFRA's UK-AIR Sensor Observation Service (52°North
Timeseries API, EIONET pollutant codes) at
``https://uk-air.defra.gov.uk/sos-ukair/api/v1/``. Its ``id``/``label``
fields are only a numeric station id and a free-text "site-pollutant"
description; the human station code (e.g. ``"MAN3"`` for Manchester
Piccadilly) is scraped separately from the AURN network-info page and
matched onto it by site name -- see ``_aurn_live_site_codes`` below.
Because that page and classification are AURN-specific, ``site_type`` and
``start_date``/``end_date`` are always ``NaN`` for ``aurn_live`` rows;
everything else lines up with the archive schema so the two can be
concatenated.
"""

from __future__ import annotations

import functools
import gzip
import html
import re
import warnings
from collections.abc import Iterable
from typing import Any

import pandas as pd

from ..utils._lazy import require
from ..utils.logging import get_logger
from ._http import get_json, request_with_retry

log = get_logger(__name__)

__all__ = [
    "UKAQ_SOURCES",
    "fetch_ukaq_measurements",
    "list_ukaq_stations",
]

#: Per-network base URL for hourly data and the metadata archive.
#:
#: ``local`` (LMAM) is DEFRA's locally-managed automatic monitoring
#: collection; it is included for completeness but its coverage is patchier
#: than the five statutory networks.
UKAQ_SOURCES: dict[str, dict[str, str]] = {
    "aurn": {
        "data": "https://uk-air.defra.gov.uk/openair/R_data/",
        "meta": "https://uk-air.defra.gov.uk/openair/R_data/AURN_metadata.RData",
    },
    "aqe": {
        "data": "https://airqualityengland.co.uk/assets/openair/R_data/",
        "meta": "https://airqualityengland.co.uk/assets/openair/R_data/AQE_metadata.RData",
    },
    "saqn": {
        "data": "https://www.scottishairquality.scot/openair/R_data/",
        "meta": "https://www.scottishairquality.scot/openair/R_data/SCOT_metadata.RData",
    },
    "waqn": {
        "data": "https://airquality.gov.wales/sites/default/files/openair/R_data/",
        "meta": (
            "https://airquality.gov.wales/sites/default/files/openair/R_data/WAQ_metadata.RData"
        ),
    },
    "ni": {
        "data": "https://www.airqualityni.co.uk/openair/R_data/",
        "meta": "https://www.airqualityni.co.uk/openair/R_data/NI_metadata.RData",
    },
    "local": {
        "data": "https://uk-air.defra.gov.uk/openair/LMAM/R_data/",
        "meta": "https://uk-air.defra.gov.uk/openair/LMAM/R_data/LMAM_metadata.RData",
    },
}

# openair's metadata column names, applied so a caller can move between the
# R and Python workflows without relearning the schema.
_META_RENAME = {
    "site_id": "code",
    "site_name": "site",
    "location_type": "site_type",
    "parameter": "variable",
}

# Metadata is small (a few thousand rows), static over a session, and needed
# by every fetch that asks for meta=True. Cached per source for the process
# lifetime rather than refetched per site-year.
_META_CACHE: dict[str, pd.DataFrame] = {}


#: Sources handled by the live SOS API instead of an ``.RData`` archive.
#: Kept separate from ``UKAQ_SOURCES`` because that dict's values are
#: archive/metadata URL pairs, a shape ``aurn_live`` does not have.
_LIVE_SOURCES = frozenset({"aurn_live"})


def _check_source(source: str) -> str:
    src = source.lower().strip()
    if src not in UKAQ_SOURCES and src not in _LIVE_SOURCES:
        valid = ", ".join((*UKAQ_SOURCES, *_LIVE_SOURCES))
        raise ValueError(f"Unknown source '{source}'. Valid sources: {valid}")
    return src


# ---------------------------------------------------------------------------
# aurn_live: DEFRA UK-AIR Sensor Observation Service (folded in from the
# former normet.io.defra). See the module docstring for why this backend
# exists alongside the archives above rather than being redundant with them.
# ---------------------------------------------------------------------------

_AURN_LIVE_API_BASE = "https://uk-air.defra.gov.uk/sos-ukair/api/v1"
# The SOS API has no short site codes (only a numeric internal id and a long
# descriptive label); the official AURN codes (e.g. "MAN3" for Manchester
# Piccadilly) live in the <select id="site_id"> on this page instead.
_AURN_NETWORK_INFO_URL = "https://uk-air.defra.gov.uk/networks/network-info"

# EIONET pollutant vocabulary codes the SOS API keys phenomena on. Named to
# match the archive sources' own column names (e.g. "NOXasNO2", not DEFRA's
# "NOX") so a `pollutant=` filter means the same string on both sources.
_AURN_LIVE_POLLUTANT_CODES: dict[str, int] = {
    "PM2.5": 6001,
    "PM10": 5,
    "NO2": 8,
    "NOXasNO2": 9,
    "NO": 20,
    "O3": 7,
    "SO2": 1,
    "CO": 10,
    "BENZENE": 24,
}


_AURN_LIVE_POLLUTANT_CODES_LOWER = {k.lower(): v for k, v in _AURN_LIVE_POLLUTANT_CODES.items()}


def _aurn_live_resolve_pollutant_code(pollutant: str | int) -> int:
    if isinstance(pollutant, int):
        return pollutant
    # Case-insensitive on the whole key, not `.upper()`: "NOXasNO2" (mixed
    # case, to match the archive sources' own column name) does not equal
    # its own `.upper()`, so that transform alone would never match itself.
    code = _AURN_LIVE_POLLUTANT_CODES_LOWER.get(pollutant.lower())
    if code is None:
        valid = ", ".join(_AURN_LIVE_POLLUTANT_CODES)
        raise ValueError(f"Unknown pollutant '{pollutant}'. Known: {valid}")
    return code


def _aurn_live_request(url: str, params: dict[str, Any] | None = None, retries: int = 3) -> Any:
    """GET a SOS JSON endpoint with retry/backoff/429 handling."""
    return get_json(url, params=params or {}, retries=retries, source="UK-AQ (aurn_live)")


@functools.lru_cache(maxsize=1)
def _aurn_live_site_codes() -> dict[str, str]:
    """Official AURN short site codes, keyed by site name.

    e.g. ``{"Manchester Piccadilly": "MAN3", "London Marylebone Road": "MY1"}``
    -- the codes used throughout UK-AIR/openair/saqgetr and by the archive
    sources' own ``code`` column, distinct from the SOS API's internal
    numeric station id. Scraped from the ``<select id="site_id">`` on
    UK-AIR's public AURN network-info page (there is no JSON endpoint for
    this). Cached for the process lifetime -- the list is static enough
    that one re-fetch per session is plenty.

    Returns
    -------
    dict[str, str]
        Site name -> AURN code. Empty (with a logged warning) if the page
        layout changes and the codes can't be parsed, so callers should
        treat a missing/blank code as "unknown" rather than fail outright.
    """
    try:
        resp = request_with_retry(
            _AURN_NETWORK_INFO_URL, params={"view": "aurn"}, source="UK-AQ (aurn_live)"
        )
        match = re.search(r'<select id="site_id"[^>]*>(.*?)</select>', resp.text, re.S)
        if not match:
            raise ValueError("could not find the #site_id <select> on the network-info page")
        options = re.findall(r'<option value="([^"]*)"[^>]*>([^<]*)</option>', match.group(1))
        codes = {html.unescape(name).strip(): code for code, name in options if code}
        log.info("Fetched %d AURN site codes from UK-AIR.", len(codes))
        return codes
    except Exception as e:
        log.warning("Could not fetch AURN site codes (%s) -- the 'code' column will be blank.", e)
        return {}


def _aurn_live_split_label(label: str) -> str:
    """``'Manchester Piccadilly-Nitrogen dioxide (air)'`` -> ``'Manchester Piccadilly'``.

    The SOS API's station label is always ``{site name}-{pollutant
    description}``; this is the one place that split happens, so every
    aurn_live row (station list or measurement) resolves the same site name
    from it.
    """
    return str(label).rsplit("-", 1)[0].strip()


def _aurn_live_list_stations(
    *,
    pollutant: str | Iterable[str] | None,
    site_type: str | Iterable[str] | None,
    all_variables: bool,
) -> pd.DataFrame:
    if site_type is not None:
        raise ValueError(
            "site_type filtering is not supported for source='aurn_live': the SOS "
            "API has no station classification. Use an archive source instead."
        )

    name_to_code = _aurn_live_site_codes()
    lowered_codes = {name.lower(): code for name, code in name_to_code.items()}

    if pollutant is not None or all_variables:
        wanted = (
            list(_AURN_LIVE_POLLUTANT_CODES)
            if pollutant is None
            else ([pollutant] if isinstance(pollutant, str) else list(pollutant))
        )
        rows: list[dict[str, Any]] = []
        for pol in wanted:
            code = _aurn_live_resolve_pollutant_code(pol)
            timeseries_list = _aurn_live_request(
                f"{_AURN_LIVE_API_BASE}/timeseries", {"phenomenon": str(code), "limit": 5000}
            )
            for ts in timeseries_list:
                props = (ts.get("station") or {}).get("properties") or {}
                geom = (ts.get("station") or {}).get("geometry") or {}
                coords = geom.get("coordinates", [None, None])
                label = props.get("label", ts.get("label", ""))
                site = _aurn_live_split_label(label)
                rows.append(
                    {
                        "code": lowered_codes.get(site.lower()),
                        "site": site,
                        "site_type": pd.NA,
                        "latitude": coords[0],
                        "longitude": coords[1],
                        "start_date": pd.NaT,
                        "end_date": pd.NaT,
                        "network": "aurn_live",
                        "variable": pol,
                    }
                )
        out = pd.DataFrame(rows)
        if not all_variables:
            out = out.drop(columns="variable").drop_duplicates(subset="site")
        return out.reset_index(drop=True)

    stations_raw = _aurn_live_request(f"{_AURN_LIVE_API_BASE}/stations", {"limit": 5000})
    rows = []
    seen: set[str] = set()
    for s in stations_raw:
        props = s["properties"]
        geom = s.get("geometry", {})
        coords = geom.get("coordinates", [None, None])
        site = _aurn_live_split_label(props["label"])
        if site in seen:
            continue
        seen.add(site)
        rows.append(
            {
                "code": lowered_codes.get(site.lower()),
                "site": site,
                "site_type": pd.NA,
                "latitude": coords[0],
                "longitude": coords[1],
                "start_date": pd.NaT,
                "end_date": pd.NaT,
                "network": "aurn_live",
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def _aurn_live_fetch_measurements(
    codes: list[str],
    years: list[int],
    *,
    pollutant: str | Iterable[str] | None,
    meta: bool,
    on_missing: str,
) -> pd.DataFrame:
    name_to_code = _aurn_live_site_codes()
    code_to_name = {c.lower(): n for n, c in name_to_code.items()}

    wanted_sites: dict[str, str] = {}  # requested code -> site name
    for code in codes:
        name = code_to_name.get(code.lower())
        if name is None:
            msg = f"aurn_live: unknown AURN code '{code}' (not in the network-info site list)"
            if on_missing == "raise":
                raise RuntimeError(msg)
            if on_missing == "warn":
                log.warning(msg)
            continue
        wanted_sites[code] = name

    if not wanted_sites:
        log.warning("aurn_live: no requested codes could be resolved to a site name.")
        return pd.DataFrame()

    pollutants = (
        list(_AURN_LIVE_POLLUTANT_CODES)
        if pollutant is None
        else ([pollutant] if isinstance(pollutant, str) else list(pollutant))
    )
    if pollutant is None:
        log.info(
            "aurn_live: pollutant=None -- fetching all %d known species sequentially "
            "from the live API; this is slower than an archive fetch.",
            len(pollutants),
        )

    long_rows: list[dict[str, Any]] = []
    for pol in pollutants:
        pcode = _aurn_live_resolve_pollutant_code(pol)
        timeseries_list = _aurn_live_request(
            f"{_AURN_LIVE_API_BASE}/timeseries", {"phenomenon": str(pcode), "limit": 5000}
        )
        # site name (lowercased) -> timeseries id, for this pollutant only.
        ts_by_site = {}
        for ts in timeseries_list:
            props = (ts.get("station") or {}).get("properties") or {}
            label = props.get("label", ts.get("label", ""))
            ts_by_site[_aurn_live_split_label(label).lower()] = ts["id"]

        for code, name in wanted_sites.items():
            ts_id = ts_by_site.get(name.lower())
            if ts_id is None:
                msg = f"aurn_live: no '{pol}' timeseries found for {code} ({name})"
                if on_missing == "raise":
                    raise RuntimeError(msg)
                if on_missing == "warn":
                    log.warning(msg)
                continue

            for yr in years:
                timespan = f"{yr}-01-01T00:00:00Z/{yr}-12-31T23:59:59Z"
                try:
                    data = _aurn_live_request(
                        f"{_AURN_LIVE_API_BASE}/timeseries/{ts_id}/getData",
                        {"timespan": timespan},
                    )
                except Exception as e:
                    msg = f"aurn_live: fetching {pol} for {code} in {yr} failed ({e})"
                    if on_missing == "raise":
                        raise RuntimeError(msg) from e
                    if on_missing == "warn":
                        log.warning(msg)
                    continue
                for v in data.get("values") or []:
                    ts_ms, val = v.get("timestamp"), v.get("value")
                    if ts_ms is None or val is None:
                        continue
                    long_rows.append(
                        {
                            "date": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
                            "code": code,
                            "site": name,
                            "variable": pol,
                            "value": float(val),
                        }
                    )

    if not long_rows:
        log.warning("aurn_live: no data fetched for any of %s.", list(wanted_sites))
        return pd.DataFrame()

    long_df = pd.DataFrame(long_rows)
    out = long_df.pivot_table(
        index=["date", "code", "site"], columns="variable", values="value", aggfunc="first"
    ).reset_index()
    out.columns.name = None
    out["network"] = "aurn_live"

    if meta:
        stations = _aurn_live_list_stations(pollutant=None, site_type=None, all_variables=False)
        cols = [c for c in ("code", "site_type", "latitude", "longitude") if c in stations.columns]
        out = out.merge(stations[cols], on="code", how="left")

    return out.sort_values(["code", "date"]).reset_index(drop=True)


def _read_rdata(url: str, *, timeout: float = 120.0) -> dict[str, Any]:
    """Download and parse one ``.RData`` archive into ``{name: object}``.

    The archives are gzip-compressed R serialisations. ``rdata`` reads the
    uncompressed bytes directly, so nothing is written to disk.
    """
    rdata = require(
        "rdata",
        hint="pip install rdata  (pure-Python .RData reader; no R needed)",
    )
    resp = request_with_retry(url, timeout=timeout, source="UK-AQ")
    payload = resp.content
    # Served gzipped, but do not assume it: a network could switch to plain
    # serialisation, and requests may already have transparently decoded a
    # Content-Encoding: gzip response. Sniff the magic bytes instead.
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    with warnings.catch_warnings():
        # rdata warns that it has no constructor for R's POSIXct class and
        # returns the underlying numeric instead. That is exactly what we
        # want -- _to_datetime below converts it -- so the warning is noise.
        warnings.simplefilter("ignore", UserWarning)
        parsed = rdata.parser.parse_data(payload)
        converted = rdata.conversion.convert(parsed)
    return dict(converted)


def _to_datetime(col: pd.Series) -> pd.Series:
    """Coerce an R ``POSIXct`` column to a UTC-aware pandas datetime.

    ``rdata`` has no POSIXct constructor and hands back the raw numeric
    (seconds since the Unix epoch), but a future version may convert it
    properly, so both forms are handled.
    """
    if pd.api.types.is_numeric_dtype(col):
        return pd.to_datetime(col, unit="s", utc=True)
    out = pd.to_datetime(col, utc=True)
    return out


def list_ukaq_stations(
    source: str = "aurn",
    *,
    pollutant: str | Iterable[str] | None = None,
    site_type: str | Iterable[str] | None = None,
    all_variables: bool = False,
) -> pd.DataFrame:
    """
    List monitoring stations for a UK network.

    Parameters
    ----------
    source : str, default "aurn"
        One of ``aurn``, ``aqe``, ``saqn``, ``waqn``, ``ni``, ``local``
        (archives, whole years) or ``aurn_live`` (SOS API, rolling recent
        window -- see the module docstring). ``site_type`` is not supported
        for ``aurn_live``, and its ``site_type``/``start_date``/``end_date``
        are always ``NaN``: the SOS API does not carry them.
    pollutant : str or iterable of str, optional
        Keep only stations measuring these species, matched against the
        metadata ``variable`` column (e.g. ``"NOx"``, ``["NO2", "PM2.5"]``).
    site_type : str or iterable of str, optional
        Keep only these classifications, e.g. ``"Urban Traffic"``,
        ``["Rural Background", "Suburban Background"]``. Not supported for
        ``source="aurn_live"``.
    all_variables : bool, default False
        If ``False`` (the default) return one row per station, dropping the
        per-species columns. If ``True`` return the raw one-row-per
        station-species metadata, which is what you need to know *which*
        species a station reports and over what period.

    Returns
    -------
    pandas.DataFrame
        Columns include ``code``, ``site``, ``site_type``, ``latitude``,
        ``longitude``, ``start_date``, ``end_date``, plus ``variable`` when
        ``all_variables=True``. A ``network`` column is always added so
        frames from several sources can be concatenated unambiguously.

    Notes
    -----
    Station codes are unique **within** a network but not across networks,
    and several AURN stations are mirrored into the devolved networks under
    the same code and coordinates (e.g. ``BUSH``, ``ESK``, ``AH``,
    ``PEMB``). Deduplicate on coordinates, not codes, when combining
    sources.

    Examples
    --------
    >>> rural = list_ukaq_stations("aurn", site_type="Rural Background")
    >>> scot = list_ukaq_stations("saqn", pollutant="NOx")
    >>> recent = list_ukaq_stations("aurn_live", pollutant="NO2")
    """
    src = _check_source(source)
    if src in _LIVE_SOURCES:
        return _aurn_live_list_stations(
            pollutant=pollutant, site_type=site_type, all_variables=all_variables
        )
    if src not in _META_CACHE:
        objs = _read_rdata(UKAQ_SOURCES[src]["meta"])
        frames = [v for v in objs.values() if isinstance(v, pd.DataFrame)]
        if not frames:
            raise RuntimeError(f"no data frame found in {src} metadata archive")
        # Every network ships a single object named "metadata"; take the
        # largest frame rather than the name in case one renames it.
        meta = max(frames, key=len).rename(columns=_META_RENAME)
        meta["network"] = src
        _META_CACHE[src] = meta
    out = _META_CACHE[src].copy()

    if pollutant is not None and "variable" in out.columns:
        wanted = {pollutant} if isinstance(pollutant, str) else set(pollutant)
        lowered = {w.lower() for w in wanted}
        out = out[out["variable"].astype(str).str.lower().isin(lowered)]
    if site_type is not None and "site_type" in out.columns:
        wanted = {site_type} if isinstance(site_type, str) else set(site_type)
        lowered = {w.lower() for w in wanted}
        out = out[out["site_type"].astype(str).str.lower().isin(lowered)]

    if not all_variables:
        drop = [c for c in ("variable", "Parameter_name", "ratified_to") if c in out.columns]
        out = out.drop(columns=drop).drop_duplicates(subset="code")

    return out.reset_index(drop=True)


def fetch_ukaq_measurements(
    site: str | Iterable[str],
    year: int | Iterable[int],
    *,
    source: str = "aurn",
    pollutant: str | Iterable[str] | None = None,
    meta: bool = False,
    on_missing: str = "warn",
) -> pd.DataFrame:
    """
    Fetch hourly measurements for one or more UK network stations.

    Parameters
    ----------
    site : str or iterable of str
        Station code(s), e.g. ``"MAN3"`` or ``["MAN3", "GLAZ"]``. Case
        insensitive; the archives are keyed on upper-case codes.
    year : int or iterable of int
        Calendar year(s). The archives are stored one file per station-year,
        so this cannot be a partial range -- slice the result if you need
        one.
    source : str, default "aurn"
        One of ``aurn``, ``aqe``, ``saqn``, ``waqn``, ``ni``, ``local``
        (archives) or ``aurn_live`` (SOS API -- see the module docstring).
        For ``aurn_live``, ``year`` still selects whole calendar year(s) (Jan
        1 00:00 UTC to Dec 31 23:59:59 UTC), even though the live API itself
        can serve an arbitrary range; this keeps the two sources' contract
        identical. ``meta=True`` on ``aurn_live`` rows only ever fills in
        ``latitude``/``longitude`` -- ``site_type`` stays ``NaN``.
    pollutant : str or iterable of str, optional
        Keep only these measurement columns, alongside ``date``, ``code``,
        ``site`` and ``network``, which are always retained. Names are
        matched in full but case-insensitively, so NOx is ``"NOXasNO2"``
        (or ``"noxasno2"``), not ``"nox"``. Anything unmatched is logged
        with the list of columns that were available. If ``None``, every
        reported species is returned -- for ``aurn_live`` this means one
        live discovery + fetch per known pollutant, sequentially, so it is
        markedly slower than an archive fetch.
    meta : bool, default False
        Join ``site_type``, ``latitude`` and ``longitude`` from the
        network's metadata archive.
    on_missing : {"warn", "raise", "ignore"}, default "warn"
        What to do when a station-year archive does not exist (a station
        that had not opened yet, or reports no data that year), or -- for
        ``aurn_live`` -- when a requested code has no matching site name or
        a site/pollutant/year has no live timeseries. The default logs and
        skips, so one gap does not abort a long fetch.

    Returns
    -------
    pandas.DataFrame
        Wide format, one row per hour per station: ``date`` (UTC-aware),
        ``code``, ``site``, one column per measured species, plus
        ``network`` and -- with ``meta=True`` -- ``site_type``,
        ``latitude``, ``longitude``. Sorted by ``(code, date)``. Empty if
        nothing could be fetched.

    Notes
    -----
    Concentrations are mass units (ug m-3), as published. NOx is reported
    as ``NOXasNO2``, matching openair and the AURN archive.

    ``aurn_live`` passes the SOS API's values through raw: its most recent
    few hours are typically ``-99`` (DEFRA's own sentinel for a reading not
    yet ratified/QC'd), not a fetch failure -- expect it at the tail of
    almost every ``aurn_live`` pull and filter it out downstream if needed.
    The archive sources do not have this because by the time a year's
    ``.RData`` is published, ratification is already done.

    Examples
    --------
    >>> df = fetch_ukaq_measurements("MAN1", range(2018, 2023), source="aqe")
    >>> gm = fetch_ukaq_measurements(
    ...     ["GLAZ", "LB"], [2020], source="aurn",
    ...     pollutant="NOXasNO2", meta=True,
    ... )
    >>> recent = fetch_ukaq_measurements(
    ...     "MAN3", 2026, source="aurn_live", pollutant="NO2",
    ... )
    """
    src = _check_source(source)
    if on_missing not in {"warn", "raise", "ignore"}:
        raise ValueError("on_missing must be one of 'warn', 'raise', 'ignore'")

    sites = [site] if isinstance(site, str) else list(site)
    codes = [str(s).upper().strip() for s in sites]
    years = [year] if isinstance(year, int) else [int(y) for y in year]
    if not codes or not years:
        raise ValueError("both `site` and `year` must be non-empty")

    if src in _LIVE_SOURCES:
        return _aurn_live_fetch_measurements(
            codes, years, pollutant=pollutant, meta=meta, on_missing=on_missing
        )

    base = UKAQ_SOURCES[src]["data"]
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for code in codes:
        for yr in years:
            url = f"{base}{code}_{yr}.RData"
            try:
                objs = _read_rdata(url)
            except Exception as e:
                missing.append(f"{code}_{yr}")
                if on_missing == "raise":
                    raise RuntimeError(f"could not fetch {url}: {e}") from e
                if on_missing == "warn":
                    log.warning("No %s data for %s in %d (%s).", src.upper(), code, yr, e)
                continue

            # Each archive holds the hourly frame under "{CODE}_{year}" plus
            # pre-aggregated "_24hour_mean"/"_daily_mean" companions. Select
            # the hourly one by exact name; falling back to "largest frame"
            # would silently pick an aggregate if the naming ever changed.
            key = f"{code}_{yr}"
            df = objs.get(key)
            if not isinstance(df, pd.DataFrame):
                candidates = [
                    v
                    for k, v in objs.items()
                    if isinstance(v, pd.DataFrame) and not k.endswith(("_mean", "_24hour_mean"))
                ]
                if not candidates:
                    missing.append(key)
                    continue
                df = max(candidates, key=len)

            df = df.copy()
            if "date" in df.columns:
                df["date"] = _to_datetime(df["date"])
            frames.append(df)

    if not frames:
        log.warning("No %s data fetched for any of %s.", src.upper(), codes)
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["network"] = src

    if pollutant is not None:
        wanted = {pollutant} if isinstance(pollutant, str) else set(pollutant)
        lowered = {w.lower() for w in wanted}
        keep_always = {"date", "code", "site", "network"}
        cols = [c for c in out.columns if c in keep_always or c.lower() in lowered]
        unmatched = lowered - {c.lower() for c in out.columns}
        if unmatched:
            log.warning(
                "Requested pollutant(s) not present in %s data: %s. Available: %s",
                src.upper(),
                sorted(unmatched),
                sorted(c for c in out.columns if c not in keep_always),
            )
        out = out[cols]

    if meta:
        stations = list_ukaq_stations(src)
        cols = [c for c in ("code", "site_type", "latitude", "longitude") if c in stations.columns]
        out = out.merge(stations[cols], on="code", how="left")

    sort_cols = [c for c in ("code", "date") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    if missing:
        log.info("%d station-year archive(s) unavailable: %s", len(missing), ", ".join(missing))
    return out
