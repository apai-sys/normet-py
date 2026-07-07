# src/normet/io/openmeteo.py
"""Open-Meteo historical-weather adapter (keyless ERA5-derived meteorology).

The `Open-Meteo archive API <https://open-meteo.com/en/docs/historical-weather-api>`_
serves hourly reanalysis data (ERA5 / ERA5-Land blend) for any coordinate
without registration or an API key, which makes it the friction-free
alternative to the Copernicus CDS for assembling deweathering predictors.
Output columns follow normet's ERA5 naming and units (``t2m``/``d2m`` in K,
``sp`` in Pa, ``ssrd`` in J m⁻², ``tcc`` 0–1, ``tp`` in m, plus derived
``u10``/``v10`` from wind speed and direction), so the result merges directly
with AURN / OpenAQ measurements.

Data by `Open-Meteo.com <https://open-meteo.com/>`_ (CC BY 4.0), based on
Copernicus ERA5.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from ._http import request_with_retry

log = get_logger(__name__)

__all__ = ["OPENMETEO_HOURLY_DEFAULT", "fetch_openmeteo_timeseries"]

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

#: Open-Meteo hourly fields fetched by default → normet/ERA5 column mapping.
#: ``boundary_layer_height`` is deliberately absent — the archive API does not
#: backfill it (all-NaN), only the forecast API serves it.
OPENMETEO_HOURLY_DEFAULT: dict[str, str] = {
    "temperature_2m": "t2m",
    "dew_point_2m": "d2m",
    "relative_humidity_2m": "rh2m",
    "surface_pressure": "sp",
    "cloud_cover": "tcc",
    "precipitation": "tp",
    "shortwave_radiation": "ssrd",
    "wind_speed_10m": "ws",
    "wind_direction_10m": "wd",
}


def _to_era5_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Open-Meteo native units to ERA5 conventions in place."""
    if "t2m" in df:
        df["t2m"] = df["t2m"] + 273.15  # °C → K
    if "d2m" in df:
        df["d2m"] = df["d2m"] + 273.15  # °C → K
    if "sp" in df:
        df["sp"] = df["sp"] * 100.0  # hPa → Pa
    if "tcc" in df:
        df["tcc"] = df["tcc"] / 100.0  # % → fraction
    if "tp" in df:
        df["tp"] = df["tp"] / 1000.0  # mm → m
    if "ssrd" in df:
        df["ssrd"] = df["ssrd"] * 3600.0  # W m⁻² (hour mean) → J m⁻² per hour
    if "ws" in df and "wd" in df:
        from ..utils.featureeng import wind_to_uv

        u, v = wind_to_uv(df["ws"], df["wd"])
        df["u10"] = u
        df["v10"] = v
    return df


def _coerce_sites(
    sites: pd.DataFrame | Mapping[str, tuple[float, float]],
    site_col: str,
    lat_col: str,
    lon_col: str,
) -> list[tuple[str, float, float]]:
    if isinstance(sites, Mapping):
        return [(str(name), float(lat), float(lon)) for name, (lat, lon) in sites.items()]
    missing = [c for c in (site_col, lat_col, lon_col) if c not in sites.columns]
    if missing:
        raise ValueError(f"`sites` DataFrame is missing columns: {missing}")
    dedup = sites.drop_duplicates(subset=[site_col])
    return [(str(r[site_col]), float(r[lat_col]), float(r[lon_col])) for _, r in dedup.iterrows()]


def fetch_openmeteo_timeseries(
    *,
    sites: pd.DataFrame | Mapping[str, tuple[float, float]],
    date_from: str | pd.Timestamp,
    date_to: str | pd.Timestamp,
    variables: Sequence[str] | None = None,
    site_col: str = "site",
    lat_col: str = "lat",
    lon_col: str = "lon",
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Fetch hourly ERA5-derived meteorology from Open-Meteo (no API key).

    Parameters
    ----------
    sites : DataFrame or Mapping
        Site locations: a DataFrame with ``[site_col, lat_col, lon_col]``
        columns, or a mapping ``{"site name": (lat, lon)}``.
    date_from, date_to : str or Timestamp
        Inclusive date range (UTC).
    variables : sequence of str, optional
        Open-Meteo hourly field names. Defaults to
        :data:`OPENMETEO_HOURLY_DEFAULT` (a standard deweathering set).
        Unknown names are passed through and keep their Open-Meteo name
        and units.
    site_col, lat_col, lon_col : str
        Column names when ``sites`` is a DataFrame.
    timeout : float, default 60
        Per-request timeout in seconds.

    Returns
    -------
    pandas.DataFrame
        Long format ``[site, date, lat, lon, <met columns…>]`` with naive UTC
        timestamps, one row per site-hour, ERA5-style names and units
        (``t2m``/``d2m`` K, ``sp`` Pa, ``ssrd`` J m⁻², ``tcc`` 0–1, ``tp`` m,
        ``ws``/``wd``/``u10``/``v10`` m s⁻¹ / degrees).

    Notes
    -----
    Data by Open-Meteo.com (CC BY 4.0), based on Copernicus ERA5. The archive
    lags real time by a few days; requests inside that lag return NaNs.
    """
    fields = list(variables) if variables is not None else list(OPENMETEO_HOURLY_DEFAULT)
    start = pd.to_datetime(date_from).strftime("%Y-%m-%d")
    end = pd.to_datetime(date_to).strftime("%Y-%m-%d")

    frames: list[pd.DataFrame] = []
    for name, lat, lon in _coerce_sites(sites, site_col, lat_col, lon_col):
        log.info("Open-Meteo: fetching %s (%.4f, %.4f) %s → %s", name, lat, lon, start, end)
        payload = request_with_retry(
            _ARCHIVE_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": start,
                "end_date": end,
                "hourly": ",".join(fields),
                "wind_speed_unit": "ms",
                "timezone": "UTC",
            },
            timeout=timeout,
        ).json()
        hourly = payload.get("hourly") or {}
        if not hourly.get("time"):
            log.warning("Open-Meteo returned no data for site %s", name)
            continue
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(hourly["time"]),
                **{
                    OPENMETEO_HOURLY_DEFAULT.get(f, f): np.asarray(
                        hourly.get(f, [np.nan] * len(hourly["time"])), dtype=float
                    )
                    for f in fields
                },
            }
        )
        df = _to_era5_units(df)
        df.insert(0, "site", name)
        df["lat"] = lat
        df["lon"] = lon
        frames.append(df)

    if not frames:
        raise RuntimeError("Open-Meteo returned no data for any site.")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["site", "date"]).reset_index(drop=True)
