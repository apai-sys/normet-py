# src/normet/io/era5.py
"""ERA5 reanalysis adapter via the Climate Data Store (CDS).

Wraps :mod:`cdsapi` to download pre-interpolated single-point time-series from
the ``reanalysis-era5-single-levels-timeseries`` collection and ingest them
into a long-format DataFrame compatible with the rest of ``normet``.

Requires:
  * A free CDS account and ``~/.cdsapirc`` with ``url`` and ``key``.
  * ``pip install cdsapi``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils._lazy import require
from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["ERA5_AQ_VARIABLES_DEFAULT", "fetch_era5_timeseries"]

# Common surface variables for air-quality modelling (CDS variable names).
ERA5_AQ_VARIABLES_DEFAULT: list[str] = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "boundary_layer_height",
    "total_cloud_cover",
    "total_precipitation",
    "surface_solar_radiation_downwards",
]


def _coerce_sites(
    sites: pd.DataFrame | Mapping[str, tuple[float, float]] | Sequence[Mapping[str, Any]],
    *,
    site_col: str = "site",
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """Normalise the ``sites`` argument into a DataFrame with ``[site_col,lat_col,lon_col]``."""
    if isinstance(sites, pd.DataFrame):
        missing = [c for c in (site_col, lat_col, lon_col) if c not in sites.columns]
        if missing:
            raise ValueError(f"sites DataFrame missing columns: {missing}")
        return sites[[site_col, lat_col, lon_col]].copy()
    if isinstance(sites, Mapping):
        rows = []
        for name, (la, lo) in sites.items():
            rows.append({site_col: name, lat_col: float(la), lon_col: float(lo)})
        return pd.DataFrame(rows)
    if isinstance(sites, Sequence):
        return pd.DataFrame(list(sites))
    raise TypeError(f"Unsupported sites type: {type(sites).__name__}")


def fetch_era5_timeseries(
    *,
    sites: pd.DataFrame | Mapping[str, tuple[float, float]],
    date_from: str | pd.Timestamp,
    date_to: str | pd.Timestamp,
    variables: Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
    site_col: str = "site",
    lat_col: str = "lat",
    lon_col: str = "lon",
    date_col: str = "date",
    cds_url: str | None = None,
    cds_key: str | None = None,
) -> pd.DataFrame:
    """Query the Copernicus CDS API for single-point meteorological time-series.

    Aggregates results into a long-format DataFrame compatible with normet.
    Uses the 'reanalysis-era5-single-levels-timeseries' dataset, which
    downloads pre-interpolated station CSVs directly from the CDS servers,
    saving bandwidth and bypassing xarray / NetCDF binary dependencies.

    Parameters
    ----------
    sites : DataFrame, Mapping
        Site locations. A DataFrame must contain columns [site_col, lat_col, lon_col].
        A mapping must look like {"site_name": (lat, lon)}.
    date_from, date_to : str | Timestamp
        Inclusive date range.
    variables : sequence of str, optional
        CDS API variable names (long names). Defaults to a standard AQ predictor list.
    cache_dir : str | Path, optional
        Directory to cache downloaded CSVs. If provided, reuses existing CSVs.
    site_col : str, default "site"
    lat_col : str, default "lat"
    lon_col : str, default "lon"
    date_col : str, default "date"
    cds_url, cds_key : str, optional
        Override the CDS endpoint/key instead of reading ``~/.cdsapirc``.
        Useful when that file is configured for a different Copernicus
        service (e.g. the Atmosphere Data Store, which does not host ERA5).

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame: ``[site, date, lat, lon, <variables...>]``.

    Notes
    -----
    Uses the ``reanalysis-era5-single-levels-timeseries`` dataset on the
    Climate Data Store (https://cds.climate.copernicus.eu), which returns
    short variable names directly (``t2m``, ``u10``, ``v10``, ``d2m``,
    ``sp``, ``blh``, ``tcc``, ``tp``, ``ssrd``, ...) — no renaming needed.
    The CDS API wraps the result CSV in a zip archive; this function
    extracts it transparently.
    """
    cdsapi = require("cdsapi", hint="pip install cdsapi")

    # 1. Expand/Standardise sites
    sites_df = _coerce_sites(sites, site_col=site_col, lat_col=lat_col, lon_col=lon_col)

    # 2. Standardise date range as the single "from/to" string this API expects
    d_from = pd.to_datetime(date_from).strftime("%Y-%m-%d")
    d_to = pd.to_datetime(date_to).strftime("%Y-%m-%d")
    date_range = f"{d_from}/{d_to}"

    cds_vars = list(variables or ERA5_AQ_VARIABLES_DEFAULT)

    client = None
    results = []

    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
    else:
        cache_path = None

    for _, row in sites_df.iterrows():
        s_name = str(row[site_col])
        s_lat = float(row[lat_col])
        s_lon = float(row[lon_col])

        if cache_path:
            filename = f"era5_timeseries_{s_name}_{d_from}_{d_to}.csv"
            site_target = cache_path / filename
        else:
            site_target = None

        if site_target and site_target.exists():
            log.info("Reusing cached ERA5 timeseries CSV for site %s: %s", s_name, site_target)
            df_site = pd.read_csv(site_target)
        else:
            if client is None:
                client = (
                    cdsapi.Client(url=cds_url, key=cds_key)
                    if (cds_url or cds_key)
                    else cdsapi.Client()
                )

            request = {
                "variable": cds_vars,
                "location": {"longitude": s_lon, "latitude": s_lat},
                "date": [date_range],
                "data_format": "csv",
            }

            log.info(
                "Submitting CDS point timeseries request for site %s (lat=%f, lon=%f)",
                s_name,
                s_lat,
                s_lon,
            )

            import tempfile
            import zipfile

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_zip = str(Path(tmpdir) / "era5.zip")
                client.retrieve("reanalysis-era5-single-levels-timeseries", request, tmp_zip)
                if zipfile.is_zipfile(tmp_zip):
                    with zipfile.ZipFile(tmp_zip) as zf:
                        members = zf.namelist()
                        # Prefer an explicit .csv member; otherwise fall back to
                        # the sole member (the CDS response is a single CSV, but
                        # don't depend on its name carrying a .csv extension).
                        csv_members = [n for n in members if n.lower().endswith(".csv")]
                        if csv_members:
                            member = csv_members[0]
                        elif len(members) == 1:
                            member = members[0]
                        else:
                            raise RuntimeError(
                                f"CDS response zip has no single CSV member: {members}"
                            )
                        with zf.open(member) as f:
                            df_site = pd.read_csv(f)
                else:
                    df_site = pd.read_csv(tmp_zip)

            if site_target:
                df_site.to_csv(site_target, index=False)

        # Clean and rename columns in df_site (this endpoint already returns
        # short variable names like t2m/u10/v10 — only the time/coord columns
        # need renaming).
        rename_cols = {}
        for col in df_site.columns:
            col_lower = col.lower()
            if col_lower in {"date", "time", "timestamp", "valid_time"}:
                rename_cols[col] = date_col
            elif col_lower in {"lat", "latitude"}:
                rename_cols[col] = lat_col
            elif col_lower in {"lon", "longitude"}:
                rename_cols[col] = lon_col

        df_site = df_site.rename(columns=rename_cols)

        # Add site identifier and coordinate values to match expectation
        df_site[site_col] = s_name
        if lat_col not in df_site.columns:
            df_site[lat_col] = s_lat
        if lon_col not in df_site.columns:
            df_site[lon_col] = s_lon

        # Standardise date type
        if date_col in df_site.columns:
            df_site[date_col] = pd.to_datetime(df_site[date_col])

        # Keep id/coordinate/time columns plus whatever variable columns the
        # API returned (already short names, e.g. t2m/u10/v10 — no mapping
        # from the long `cds_vars` request names is needed).
        meta_cols = [site_col, date_col, lat_col, lon_col]
        df_site = df_site[meta_cols + [c for c in df_site.columns if c not in meta_cols]]

        results.append(df_site)

    if not results:
        return pd.DataFrame()

    df_all = pd.concat(results, ignore_index=True)
    sort_cols = [c for c in (site_col, date_col) if c in df_all.columns]
    if sort_cols:
        df_all = df_all.sort_values(sort_cols).reset_index(drop=True)

    return df_all
