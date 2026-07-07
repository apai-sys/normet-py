# src/normet/io/defra.py
"""
UK AURN (Automatic Urban and Rural Network) air-quality data adapter.

Fetches hourly pollutant measurements from DEFRA's UK-AIR Sensor Observation
Service (SOS) REST API (52°North Timeseries API).

API endpoint:
    https://uk-air.defra.gov.uk/sos-ukair/api/v1/

The service uses EIONET pollutant codes (the same as the EEA).  Stations are
identified by internal numeric IDs; each station-pollutant pair is a separate
"timeseries" resource.

Returns tidy long-format DataFrames compatible with ``normet`` pipelines.
"""

from __future__ import annotations

import functools
import html
import re
from collections.abc import Iterable
from typing import Any

import pandas as pd

from ..utils.logging import get_logger
from ._http import get_json, request_with_retry

log = get_logger(__name__)

__all__ = [
    "AURN_POLLUTANT_CODES",
    "fetch_aurn_measurements",
    "fetch_aurn_site_codes",
    "list_aurn_stations",
]

_API_BASE = "https://uk-air.defra.gov.uk/sos-ukair/api/v1"
# The SOS API above has no short site codes (only a numeric internal id and a
# long descriptive label); the official AURN codes (e.g. "MAN3" for
# Manchester Piccadilly) live in the <select id="site_id"> on this page.
_NETWORK_INFO_URL = "https://uk-air.defra.gov.uk/networks/network-info"

AURN_POLLUTANT_CODES: dict[str, int] = {
    "PM2.5": 6001,
    "PM10": 5,
    "NO2": 8,
    "NOX": 9,
    "NO": 20,
    "O3": 7,
    "SO2": 1,
    "CO": 10,
    "BENZENE": 24,
}


def _resolve_pollutant_code(pollutant: str | int) -> int:
    if isinstance(pollutant, int):
        return pollutant
    code = AURN_POLLUTANT_CODES.get(pollutant.upper())
    if code is None:
        valid = ", ".join(AURN_POLLUTANT_CODES)
        raise ValueError(f"Unknown pollutant '{pollutant}'. Known: {valid}")
    return code


def _request(url: str, params: dict[str, Any] | None = None, retries: int = 3) -> Any:
    """GET a JSON endpoint with retry/backoff/429 handling."""
    return get_json(url, params=params or {}, retries=retries, source="DEFRA")


@functools.lru_cache(maxsize=1)
def fetch_aurn_site_codes() -> dict[str, str]:
    """Official AURN short site codes, keyed by site name.

    e.g. ``{"Manchester Piccadilly": "MAN3", "London Marylebone Road": "MY1"}``
    — the codes used throughout UK-AIR/openair/saqgetr, distinct from the SOS
    API's internal numeric station id. Scraped from the ``<select
    id="site_id">`` on UK-AIR's public AURN network-info page (there is no
    JSON endpoint for this). Cached for the process lifetime — the list is
    static enough that a fresh GUI session re-fetching it once is plenty.

    Returns
    -------
    dict[str, str]
        Site name -> AURN code. Empty (with a logged warning) if the page
        layout changes and the codes can't be parsed, so callers should treat
        a missing/blank code as "unknown" rather than fail outright.
    """
    try:
        resp = request_with_retry(_NETWORK_INFO_URL, params={"view": "aurn"}, source="DEFRA")
        match = re.search(r'<select id="site_id"[^>]*>(.*?)</select>', resp.text, re.S)
        if not match:
            raise ValueError("could not find the #site_id <select> on the network-info page")
        options = re.findall(r'<option value="([^"]*)"[^>]*>([^<]*)</option>', match.group(1))
        codes = {html.unescape(name).strip(): code for code, name in options if code}
        log.info("Fetched %d AURN site codes from UK-AIR.", len(codes))
        return codes
    except Exception as e:
        log.warning("Could not fetch AURN site codes (%s) — the 'abbr' column will be blank.", e)
        return {}


def list_aurn_stations(
    *,
    pollutant: str | int | None = None,
    limit: int = 5000,
) -> pd.DataFrame:
    """
    List AURN monitoring stations, optionally filtered by pollutant.

    Parameters
    ----------
    pollutant : str or int, optional
        Pollutant name (e.g., ``"PM2.5"``, ``"NO2"``) or EIONET code.
        If ``None``, returns all stations.
    limit : int, default 5000
        Maximum number of stations.

    Returns
    -------
    pandas.DataFrame
        Columns: ``id``, ``label``, ``lat``, ``lon``.
    """
    if pollutant is not None:
        code = _resolve_pollutant_code(pollutant)
        timeseries_list = _request(
            f"{_API_BASE}/timeseries", {"phenomenon": str(code), "limit": limit}
        )
    else:
        stations_raw = _request(f"{_API_BASE}/stations", {"limit": limit})
        rows = []
        for s in stations_raw:
            props = s["properties"]
            geom = s.get("geometry", {})
            coords = geom.get("coordinates", [None, None])
            rows.append(
                {
                    "id": props["id"],
                    "label": props["label"],
                    "lat": coords[0],
                    "lon": coords[1],
                }
            )
        return pd.DataFrame(rows)

    rows = []
    for ts in timeseries_list:
        station_info = ts.get("station", {})
        props = station_info.get("properties", {})
        geom = station_info.get("geometry", {})
        coords = geom.get("coordinates", [None, None])
        rows.append(
            {
                "id": props.get("id"),
                "label": props.get("label", ts.get("label", "")),
                "timeseries_id": ts["id"],
                "lat": coords[0],
                "lon": coords[1],
            }
        )
    return pd.DataFrame(rows)


def fetch_aurn_measurements(
    *,
    station: str | int | Iterable[str | int] | None = None,
    pollutant: str = "PM2.5",
    date_from: str | pd.Timestamp,
    date_to: str | pd.Timestamp,
    station_label: str | None = None,
) -> pd.DataFrame:
    """
    Fetch hourly AURN measurements as a long-format DataFrame.

    Parameters
    ----------
    station : int, str, or iterable, optional
        Station ID(s) or label substring(s).  If ``None``, returns data for
        **all** stations measuring the given pollutant (potentially large).
    pollutant : str, default "PM2.5"
        Pollutant name; one of ``PM2.5``, ``PM10``, ``NO2``, ``NOX``, ``NO``,
        ``O3``, ``SO2``, ``CO``, ``BENZENE``.
    date_from, date_to : str or Timestamp
        Inclusive UTC date range.
    station_label : str, optional
        Convenience filter: only fetch data for stations whose label contains
        this substring (case-insensitive).  Overrides ``station`` filtering
        if both are given.

    Returns
    -------
    pandas.DataFrame
        Columns: ``date`` (UTC), ``site`` (station label), ``station_id``,
        ``pollutant``, ``value``, ``unit``, ``lat``, ``lon``.
        Sorted by ``(site, date)``.
    """
    code = _resolve_pollutant_code(pollutant)
    df_from = pd.to_datetime(date_from, utc=True)
    df_to = pd.to_datetime(date_to, utc=True)

    timespan = f"{df_from.strftime('%Y-%m-%dT%H:%M:%SZ')}/{df_to.strftime('%Y-%m-%dT%H:%M:%SZ')}"

    # Discover timeseries for this pollutant
    all_ts = _request(f"{_API_BASE}/timeseries", {"phenomenon": str(code), "limit": 5000})

    # Resolve station filter
    ids_to_fetch: list[str] = []
    if station_label:
        pattern = station_label.lower()
        for ts in all_ts:
            props = (ts.get("station") or {}).get("properties") or {}
            label = (props.get("label") or ts.get("label") or "").lower()
            if pattern in label:
                ids_to_fetch.append(ts["id"])
    elif station is not None:
        # Scalars include numpy integers (e.g. ids taken from a stations DataFrame).
        scalar = isinstance(station, str) or not isinstance(station, Iterable)
        # mypy can't narrow `station` through the `scalar` bool flag (it stays
        # the full `str | int | Iterable[str | int]` union in both branches).
        station_items: list[str | int] = (
            [station] if scalar else list(station)  # type: ignore
        )
        station_ids = {
            s if isinstance(s, str) else str(int(s) if hasattr(s, "__int__") else s)
            for s in station_items
        }
        for ts in all_ts:
            props = (ts.get("station") or {}).get("properties") or {}
            ts_id = str(props.get("id", ""))
            ts_label = props.get("label", "")
            if ts_id in station_ids or any(sid in ts_label for sid in station_ids):
                ids_to_fetch.append(ts["id"])
    else:
        # Station is None — fetch ALL
        ids_to_fetch = [ts["id"] for ts in all_ts]

    if not ids_to_fetch:
        log.warning("No matching timeseries found for pollutant %s", pollutant)
        return pd.DataFrame()

    # Build a lookup from timeseries_id -> metadata
    ts_lookup: dict[str, dict[str, Any]] = {}
    for ts in all_ts:
        tid = ts["id"]
        if tid in ids_to_fetch or not (station or station_label):
            props = (ts.get("station") or {}).get("properties") or {}
            geom = (ts.get("station") or {}).get("geometry") or {}
            coords = geom.get("coordinates", [None, None])
            ts_lookup[tid] = {
                "label": props.get("label", ts.get("label", tid)),
                "station_id": props.get("id", tid),
                "lat": coords[0],
                "lon": coords[1],
            }

    rows: list[dict[str, Any]] = []

    for i, ts_id in enumerate(ids_to_fetch):
        try:
            data = _request(f"{_API_BASE}/timeseries/{ts_id}/getData", {"timespan": timespan})
        except Exception as e:
            log.warning("Failed to fetch timeseries %s: %s", ts_id, e)
            continue

        vals = data.get("values") or []
        meta = ts_lookup.get(ts_id, {})
        site_label = meta.get("label", ts_id)
        station_id = meta.get("station_id", ts_id)
        lat = meta.get("lat")
        lon = meta.get("lon")

        for v in vals:
            ts_ms = v.get("timestamp")
            val = v.get("value")
            if ts_ms is None or val is None:
                continue
            rows.append(
                {
                    "date": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
                    "site": site_label,
                    "station_id": station_id,
                    "pollutant": pollutant,
                    "value": float(val),
                    "unit": "ug.m-3",
                    "lat": lat,
                    "lon": lon,
                }
            )

        if (i + 1) % 50 == 0:
            log.info("Fetched %d/%d timeseries for %s", i + 1, len(ids_to_fetch), pollutant)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(["site", "date"]).reset_index(drop=True)
    return out
