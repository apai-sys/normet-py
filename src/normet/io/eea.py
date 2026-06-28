# src/normet/io/eea.py
"""
European Environment Agency (EEA) air-quality download adapter.

Uses the EEA's "Discomap" download-by-URL service. Given a country, station,
pollutant and year range, this function emits a tidy long-format DataFrame
ready for ``normet``.

EEA download portal:
    https://eeadmz1-downloads-webapp.azurewebsites.net/

The discovery endpoint returns a *list of CSV URLs* per (country, pollutant,
year, station). We fetch each CSV and concatenate.

Pollutant codes (subset):
    PM2.5 → 6001, PM10 → 5, NO2 → 8, O3 → 7, SO2 → 1, CO → 10
"""

from __future__ import annotations

import io
from collections.abc import Iterable

import pandas as pd

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

_DISCOVERY = (
    "https://fme.discomap.eea.europa.eu/fmedatastreaming/AirQualityDownload/AQData_Extract.fmw"
)


def _resolve_pollutant_code(pollutant: str | int) -> int:
    if isinstance(pollutant, int):
        return pollutant
    code = EEA_POLLUTANT_CODES.get(pollutant.upper())
    if code is None:
        valid = ", ".join(EEA_POLLUTANT_CODES)
        raise ValueError(f"Unknown pollutant '{pollutant}'. Known: {valid}")
    return code


def fetch_eea_data(
    *,
    country: str,
    pollutant: str | int,
    year_from: int,
    year_to: int,
    station: str | None = None,
    source: str = "All",
    output: str = "TEXT",
    keep_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Download air-quality CSV files from the EEA and concatenate them.

    Parameters
    ----------
    country : str
        ISO-2 country code (e.g., ``"GB"``, ``"FR"``, ``"DE"``, ``"ES"``).
    pollutant : str or int
        Pollutant code or name. Names supported: ``PM2.5``, ``PM10``,
        ``NO2``, ``O3``, ``SO2``, ``CO``. Integer codes also accepted.
    year_from, year_to : int
        Inclusive year range.
    station : str, optional
        Specific station code (e.g., ``"GB0001A"``). If ``None``, all
        stations in ``country`` are returned.
    source : {"All", "E1a", "E2a"}, default "All"
        EEA dataflow source.
    output : str, default "TEXT"
        Format requested from the EEA service.
    timezone : str, default "Europe%2FBrussels"
        URL-encoded timezone string forwarded to the discovery service.
    keep_columns : iterable of str, optional
        Restrict the returned DataFrame to these columns; defaults to a
        useful subset (see Returns).

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame with at least:
        ``[date, site, country, pollutant, value, unit, lat, lon]``.
    """
    code = _resolve_pollutant_code(pollutant)

    params = {
        "CountryCode": country.upper(),
        "CityName": "",
        "Pollutant": code,
        "Year_from": int(year_from),
        "Year_to": int(year_to),
        "Station": station or "",
        "Samplingpoint": "",
        "Source": source,
        "Output": output,
        "UpdateDate": "",
        "TimeCoverage": "Year",
        "TimeZone": "Europe/Brussels",
    }
    log.info("Querying EEA discovery for %s %s %d-%d", country, pollutant, year_from, year_to)
    resp = request_with_retry(_DISCOVERY, params=params, timeout=60, source="EEA")
    csv_urls = [u.strip() for u in resp.text.splitlines() if u.strip().startswith("http")]
    if not csv_urls:
        log.warning("EEA returned no CSV URLs for the given query.")
        return pd.DataFrame()

    pieces: list[pd.DataFrame] = []
    for i, url in enumerate(csv_urls, 1):
        try:
            r = request_with_retry(url, timeout=60, source="EEA")
            df = pd.read_csv(io.BytesIO(r.content), low_memory=False)
            pieces.append(df)
            if i % 25 == 0:
                log.info("EEA: %d/%d CSVs", i, len(csv_urls))
        except Exception as e:
            log.warning("EEA CSV fetch failed (%s): %s", url, e)

    if not pieces:
        return pd.DataFrame()

    raw = pd.concat(pieces, ignore_index=True)
    # Friendly canonical column names (EEA uses verbose headers like
    # "DatetimeBegin", "Concentration", etc.).
    rename = {
        "DatetimeBegin": "date",
        "AirQualityStation": "site",
        "Concentration": "value",
        "UnitOfMeasurement": "unit",
        "Latitude": "lat",
        "Longitude": "lon",
        "Pollutant": "pollutant",
        "CountryCode": "country",
    }
    raw = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns})
    if "date" in raw.columns:
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce", utc=True)

    default_keep = ["date", "site", "country", "pollutant", "value", "unit", "lat", "lon"]
    cols = list(keep_columns) if keep_columns else [c for c in default_keep if c in raw.columns]
    if not cols:
        return raw
    return (
        raw[cols].sort_values(["site", "date"]).reset_index(drop=True)
        if "site" in raw.columns
        else raw[cols]
    )
