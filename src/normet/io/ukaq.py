# src/normet/io/ukaq.py
"""UK air-quality network adapter (AURN, AQE, SAQN, WAQN, NI, LMAM).

Fetches hourly measurements and station metadata for **all six** UK
air-quality networks from the openair-format ``.RData`` archives each
network publishes.

Why this exists alongside :mod:`normet.io.defra`
------------------------------------------------
``defra.py`` talks to DEFRA's UK-AIR Sensor Observation Service and is
**AURN-only**: its site-code lookup hard-codes ``params={"view": "aurn"}``,
and the other network views on that endpoint return HTTP 200 with no site
list at all. AURN is roughly 210 of the ~1500 UK stations that report
hourly NOx, so restricting to it excludes the entire local-authority
estate -- Air Quality England, the Scottish and Welsh networks, and
Northern Ireland -- along with most rural and suburban background sites.

The two adapters are complementary rather than redundant:

============  ==========================  ===========================
              ``defra.fetch_aurn_*``      ``ukaq.fetch_ukaq_*``
============  ==========================  ===========================
networks      AURN only                   all six
backend       SOS REST API (JSON)         openair ``.RData`` archives
shape         long (one row per reading)  wide (one column per species)
granularity   arbitrary date range        whole calendar years
============  ==========================  ===========================

Use ``defra`` for a narrow date slice of AURN; use this for anything
multi-network, multi-species, or multi-year.

Data sources
------------
Each network publishes ``{CODE}_{year}.RData`` (gzip-compressed R
serialisation) under its own base URL, plus a single metadata archive.
These are the same files openair's ``importUKAQ()`` reads, so results are
directly comparable with the R workflow.

Parsing uses the pure-Python ``rdata`` package -- **no R installation is
required**.
"""

from __future__ import annotations

import gzip
import warnings
from collections.abc import Iterable
from typing import Any

import pandas as pd

from ..utils._lazy import require
from ..utils.logging import get_logger
from ._http import request_with_retry

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


def _check_source(source: str) -> str:
    src = source.lower().strip()
    if src not in UKAQ_SOURCES:
        valid = ", ".join(UKAQ_SOURCES)
        raise ValueError(f"Unknown source '{source}'. Valid sources: {valid}")
    return src


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
        One of ``aurn``, ``aqe``, ``saqn``, ``waqn``, ``ni``, ``local``.
    pollutant : str or iterable of str, optional
        Keep only stations measuring these species, matched against the
        metadata ``variable`` column (e.g. ``"NOx"``, ``["NO2", "PM2.5"]``).
    site_type : str or iterable of str, optional
        Keep only these classifications, e.g. ``"Urban Traffic"``,
        ``["Rural Background", "Suburban Background"]``.
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
    """
    src = _check_source(source)
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
        One of ``aurn``, ``aqe``, ``saqn``, ``waqn``, ``ni``, ``local``.
    pollutant : str or iterable of str, optional
        Keep only these measurement columns, alongside ``date``, ``code``,
        ``site`` and ``network``, which are always retained. Names are
        matched in full but case-insensitively, so NOx is ``"NOXasNO2"``
        (or ``"noxasno2"``), not ``"nox"``. Anything unmatched is logged
        with the list of columns that were available. If ``None``, every
        reported species is returned.
    meta : bool, default False
        Join ``site_type``, ``latitude`` and ``longitude`` from the
        network's metadata archive.
    on_missing : {"warn", "raise", "ignore"}, default "warn"
        What to do when a station-year archive does not exist (a station
        that had not opened yet, or reports no data that year). The default
        logs and skips, so one gap does not abort a long fetch.

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

    Examples
    --------
    >>> df = fetch_ukaq_measurements("MAN1", range(2018, 2023), source="aqe")
    >>> gm = fetch_ukaq_measurements(
    ...     ["GLAZ", "LB"], [2020], source="aurn",
    ...     pollutant="NOXasNO2", meta=True,
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
