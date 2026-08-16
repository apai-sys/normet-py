# src/normet/io/openaq.py
"""
OpenAQ air-quality data adapter.

Pulls hourly measurements from the OpenAQ v3 API into a long-format DataFrame
ready for ``normet`` pipelines. Requires an API key (free tier) supplied via
the ``OPENAQ_API_KEY`` environment variable or ``api_key=`` argument.

API docs: https://docs.openaq.org/

This is a thin wrapper: it does pagination, basic retry, and parses the JSON
response into a tidy ``pandas.DataFrame``. No mocking or stubbing — calls hit
the live OpenAQ servers.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import pandas as pd

from ..utils.logging import get_logger
from ._http import get_json

log = get_logger(__name__)

__all__ = ["fetch_openaq_measurements", "openaq_locations", "openaq_sensors"]

_BASE = "https://api.openaq.org/v3"

# OpenAQ v3 identifies pollutants by numeric `parameters_id`, not by slug.
# Verified against this package's own mocked test fixtures (parameter id 2 ==
# "pm25"); other common ids per the OpenAQ v3 /parameters reference.
_PARAMETER_IDS = {
    "pm25": 2,
    "pm10": 1,
    "o3": 10,
    "co": 8,
    "no2": 7,
    "so2": 9,
}


def _resolve_parameter_id(parameter: str | int) -> int:
    """Map a pollutant slug (``"pm25"``) or raw numeric id to the OpenAQ v3 id."""
    if isinstance(parameter, int):
        return parameter
    try:
        return _PARAMETER_IDS[parameter.lower()]
    except KeyError:
        try:
            return int(parameter)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Unknown OpenAQ parameter {parameter!r}; pass a known slug "
                f"({sorted(_PARAMETER_IDS)}) or a numeric parameters_id."
            ) from e


def _resolve_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("OPENAQ_API_KEY")
    if not key:
        raise RuntimeError(
            "OpenAQ requires an API key. Pass `api_key=` or set OPENAQ_API_KEY. "
            "Register a free key at https://explore.openaq.org/register."
        )
    return key


def _get(
    url: str, params: dict[str, Any], headers: dict[str, str], retries: int = 3
) -> dict[str, Any]:
    """Fetch a JSON page from the OpenAQ API with retry/backoff/429 handling."""
    return get_json(url, params=params, headers=headers, retries=retries, source="OpenAQ")


def openaq_locations(
    *,
    country: str | None = None,
    city: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    parameter: str | None = None,
    limit: int = 1000,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    List OpenAQ monitoring locations matching the given filters.

    Parameters
    ----------
    country : str, optional
        ISO-3166 alpha-2 country code (e.g., ``"GB"``, ``"US"``, ``"CN"``).
    city : str, optional
        City name; case-insensitive substring match.
    bbox : (min_lon, min_lat, max_lon, max_lat), optional
        Geographic bounding box.
    parameter : str, optional
        Pollutant filter (e.g., ``"pm25"``, ``"no2"``).
    limit : int, default 1000
        Maximum locations returned (server-side cap may apply).
    api_key : str, optional
        Overrides ``OPENAQ_API_KEY``.

    Returns
    -------
    pandas.DataFrame
        Columns include ``id``, ``name``, ``city``, ``country``, ``lat``, ``lon``,
        ``parameters``, ``last_updated``.
    """
    headers = {"X-API-Key": _resolve_key(api_key)}

    params: dict[str, Any] = {"limit": int(limit)}
    if country:
        params["iso"] = country
    if city:
        params["city"] = city
    if parameter:
        params["parameters_id"] = _resolve_parameter_id(parameter)
    if bbox:
        params["bbox"] = ",".join(f"{float(x):.4f}" for x in bbox)

    data = _get(f"{_BASE}/locations", params, headers)
    rows: list[dict[str, Any]] = []
    for r in data.get("results", []) or []:
        coords = r.get("coordinates") or {}
        params_list = [s.get("parameter", {}).get("name") for s in (r.get("sensors") or [])]
        sensors_parsed = []
        for s in r.get("sensors") or []:
            param_info = s.get("parameter") or {}
            sensors_parsed.append(
                {
                    "id": s.get("id"),
                    "name": s.get("name"),
                    "parameter_id": param_info.get("id"),
                    "parameter_name": param_info.get("name"),
                    "parameter_units": param_info.get("units"),
                }
            )
        rows.append(
            {
                "id": r.get("id"),
                "name": r.get("name"),
                "city": (r.get("locality") or {})
                if isinstance(r.get("locality"), dict)
                else r.get("locality"),
                "country": (r.get("country") or {}).get("code"),
                "lat": coords.get("latitude"),
                "lon": coords.get("longitude"),
                "parameters": [p for p in params_list if p],
                "sensors": sensors_parsed,
                "provider": (r.get("provider") or {}).get("name"),
                "owner": (r.get("owner") or {}).get("name"),
                "last_updated": r.get("datetimeLast", {}).get("utc"),
            }
        )
    return pd.DataFrame(rows)


def _resolve_sensor(
    loc: int, parameter_id: int, headers: dict[str, str]
) -> tuple[int | None, float | None, float | None]:
    """Look up the sensor id (and site lat/lon) for `parameter_id` at `loc`.

    OpenAQ v3 has no `/locations/{id}/measurements` endpoint (a 404 in
    practice) -- measurements are only served per-sensor, via
    `/sensors/{sensors_id}/measurements`. Each location exposes multiple
    single-parameter sensors, so the sensor id has to be resolved first.
    """
    data = _get(f"{_BASE}/locations/{loc}", {}, headers)
    results = data.get("results") or []
    if not results:
        return None, None, None
    r = results[0]
    coords = r.get("coordinates") or {}
    for s in r.get("sensors") or []:
        if (s.get("parameter") or {}).get("id") == parameter_id:
            return s.get("id"), coords.get("latitude"), coords.get("longitude")
    return None, coords.get("latitude"), coords.get("longitude")


def fetch_openaq_measurements(
    *,
    location_id: int | Iterable[int],
    parameter: str,
    date_from: str | pd.Timestamp,
    date_to: str | pd.Timestamp,
    page_limit: int = 1000,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Pull hourly measurements from OpenAQ v3.

    Parameters
    ----------
    location_id : int or iterable of int
        OpenAQ location identifier(s). Use :func:`openaq_locations` to discover.
    parameter : str
        Pollutant slug (e.g., ``"pm25"``, ``"no2"``, ``"o3"``, ``"so2"``, ``"co"``)
        or a raw numeric ``parameters_id``.
    date_from, date_to : str or Timestamp
        Inclusive UTC date range; parseable by :func:`pandas.to_datetime`.
    page_limit : int, default 1000
        Server page size; the function paginates automatically.
    api_key : str, optional
        Overrides ``OPENAQ_API_KEY``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``date`` (UTC), ``site`` (location id), ``parameter``,
        ``value``, ``unit``, ``lat``, ``lon``. Sorted by ``(site, date)``.
        A location with no sensor for `parameter` is silently skipped (logged
        at debug level) rather than raising, so a batch pull over many
        stations does not abort on the first station that lacks the pollutant.
    """
    headers = {"X-API-Key": _resolve_key(api_key)}
    locs = [location_id] if isinstance(location_id, int) else list(location_id)
    parameter_id = _resolve_parameter_id(parameter)

    df_from = pd.to_datetime(date_from, utc=True)
    df_to = pd.to_datetime(date_to, utc=True)

    rows: list[dict[str, Any]] = []
    for loc in locs:
        loc = int(loc)
        sensor_id, lat, lon = _resolve_sensor(loc, parameter_id, headers)
        if sensor_id is None:
            log.debug(
                "OpenAQ location %s has no sensor for parameter id %s; skipping.", loc, parameter_id
            )
            continue

        page = 1
        while True:
            params = {
                "datetime_from": df_from.isoformat(),
                "datetime_to": df_to.isoformat(),
                "limit": int(page_limit),
                "page": page,
            }
            data = _get(f"{_BASE}/sensors/{sensor_id}/measurements", params, headers)
            chunk = data.get("results", []) or []
            if not chunk:
                break
            for r in chunk:
                period = r.get("period") or {}
                start = period.get("datetimeFrom") or {}
                coords = r.get("coordinates") or {}
                rows.append(
                    {
                        "date": start.get("utc"),
                        "site": loc,
                        "parameter": (r.get("parameter") or {}).get("name") or parameter,
                        "value": r.get("value"),
                        "unit": (r.get("parameter") or {}).get("units"),
                        "lat": coords.get("latitude")
                        if coords.get("latitude") is not None
                        else lat,
                        "lon": coords.get("longitude")
                        if coords.get("longitude") is not None
                        else lon,
                    }
                )
            if len(chunk) < page_limit:
                break
            page += 1

    out = pd.DataFrame(rows)
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
        out = out.sort_values(["site", "date"]).reset_index(drop=True)
    return out


def openaq_sensors(
    *,
    location_id: int,
    limit: int = 100,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    List sensors active at a given OpenAQ monitoring location.

    Parameters
    ----------
    location_id : int
        OpenAQ location identifier.
    limit : int, default 100
        Maximum sensors returned.
    api_key : str, optional
        Overrides ``OPENAQ_API_KEY``.

    Returns
    -------
    pandas.DataFrame
        Columns include ``id``, ``name``, ``parameter_id``, ``parameter_name``,
        ``parameter_units``, ``parameter_display_name``, ``datetime_first``,
        ``datetime_last``.
    """
    headers = {"X-API-Key": _resolve_key(api_key)}
    params = {"limit": int(limit)}

    data = _get(f"{_BASE}/locations/{int(location_id)}/sensors", params, headers)
    rows: list[dict[str, Any]] = []
    for r in data.get("results", []) or []:
        param = r.get("parameter") or {}
        rows.append(
            {
                "id": r.get("id"),
                "name": r.get("name"),
                "parameter_id": param.get("id"),
                "parameter_name": param.get("name"),
                "parameter_units": param.get("units"),
                "parameter_display_name": param.get("displayName"),
                "datetime_first": r.get("datetimeFirst", {}).get("utc"),
                "datetime_last": r.get("datetimeLast", {}).get("utc"),
            }
        )
    return pd.DataFrame(rows)
