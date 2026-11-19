# src/normet/causal/spatial.py
"""Spatial diagnostics for the donor pool of a synthetic-control fit.

A synthetic control borrows its counterfactual from a weighted pool of donor
units, and the weights say nothing about *where* those donors sit relative to
the treated unit. For a policy that operates through local or regional
processes -- a city's emissions regime, a river basin's meteorology -- a
donor pool clustered on one side of the treated site is a different object
from one that surrounds it, and the two can carry the same weights. This
module supplies the reading: each donor's distance and compass bearing from
the treated unit, and the pool's centroid offset and sector coverage.

It is a diagnostic, not a fitter: nothing here changes the weights, and no
distance or bearing is fed back into the fit.

The ``haversine`` and ``bearing`` primitives are the pure-numpy cores of the
pairwise functions the (since-removed) ``normet.physics`` graph builder used;
they operate on scalars or arrays and carry no dependency beyond numpy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["bearing", "donor_spatial_summary", "haversine", "spatial_summary"]

#: Compass labels for the 8-sector default, in bearing order (N at 0 degrees).
_SECTOR_LABELS_8 = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def haversine(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> np.ndarray:
    """Great-circle distance in km between (lat1, lon1) and (lat2, lon2).

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : array-like
        Coordinates in degrees. Scalars or arrays of matching shape; the
        operation is elementwise and broadcasts like numpy.

    Returns
    -------
    numpy.ndarray
        Distance in km (a 0-d array for scalar inputs). Uses an Earth radius
        of 6371 km, so values are accurate to the ~0.3 % that a spherical
        Earth is off from the WGS-84 ellipsoid -- far tighter than any
        station-network spacing this is meant for.
    """
    r1, r2 = np.asarray(lat1, dtype=float), np.asarray(lat2, dtype=float)
    l1, l2 = np.asarray(lon1, dtype=float), np.asarray(lon2, dtype=float)
    R = 6371.0
    phi1, phi2 = np.radians(r1), np.radians(r2)
    dphi = np.radians(r2 - r1)
    dlambda = np.radians(l2 - l1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def bearing(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> np.ndarray:
    """Initial great-circle bearing in degrees from (lat1, lon1) to (lat2, lon2).

    0 degrees is true north, 90 east, 180 south, 270 west; the result is
    always in [0, 360). This is the forward (initial) azimuth -- over long
    arcs the great circle curves away from it, which is the wrong precision
    to promise for a compass reading, so it is the right one to ask for.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : array-like
        Coordinates in degrees; elementwise, broadcasts like numpy.

    Returns
    -------
    numpy.ndarray
        Bearing in degrees, in [0, 360).
    """
    r1, r2 = np.radians(np.asarray(lat1, dtype=float)), np.radians(np.asarray(lat2, dtype=float))
    l1, l2 = np.radians(np.asarray(lon1, dtype=float)), np.radians(np.asarray(lon2, dtype=float))
    dlambda = l2 - l1
    y = np.sin(dlambda) * np.cos(r2)
    x = np.cos(r1) * np.sin(r2) - np.sin(r1) * np.cos(r2) * np.cos(dlambda)
    theta = np.arctan2(y, x)
    return np.degrees((theta + 2.0 * np.pi) % (2.0 * np.pi))


def _unit_coordinates(
    df: pd.DataFrame, *, unit_col: str, lat_col: str, lon_col: str
) -> pd.DataFrame:
    """One (lat, lon) row per unit: the median of each unit's reported
    coordinates, so a station whose log repeats the same fix (or drifts by a
    meter) is read once. Units missing either coordinate are dropped."""
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"`df` must contain the coordinate columns {lat_col!r} and {lon_col!r}.")
    coords = df[[unit_col, lat_col, lon_col]].copy()
    coords[lat_col] = pd.to_numeric(coords[lat_col], errors="coerce")
    coords[lon_col] = pd.to_numeric(coords[lon_col], errors="coerce")
    coords = coords.dropna(subset=[lat_col, lon_col])
    if coords.empty:
        raise ValueError(f"No unit in `df` has finite {lat_col!r}/{lon_col!r} values.")
    med = coords.groupby(unit_col, as_index=True)[[lat_col, lon_col]].median()
    return med


def _sector_label(bearing_deg: float, n_sectors: int) -> str:
    if n_sectors == 8:
        idx = int(np.clip(int(bearing_deg // (360.0 / 8.0)), 0, 7))
        return _SECTOR_LABELS_8[idx]
    # A non-8 sector count is labelled by its index, which is all a custom
    # division can honestly claim.
    step = 360.0 / n_sectors
    return str(int(np.clip(int(bearing_deg // step), 0, n_sectors - 1)))


def donor_spatial_summary(
    df: pd.DataFrame,
    donors: list[str],
    treated_unit: str,
    *,
    unit_col: str = "unit",
    lat_col: str = "lat",
    lon_col: str = "lon",
    n_sectors: int = 8,
    weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Each donor's distance and compass bearing from the treated unit.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format panel carrying ``unit_col``, ``lat_col`` and ``lon_col``;
        one coordinate fix per unit is read as the median of its rows.
    donors : list of str
        The donor units to place relative to the treated one.
    treated_unit : str
        The treated unit the bearings and distances are taken from.
    unit_col, lat_col, lon_col : str
        The unit, latitude and longitude column names.
    n_sectors : int, default 8
        Compass sectors the bearing is divided into. 8 is labelled N/NE/E/...;
        any other count is labelled by index.
    weights : pandas.Series, optional
        The fit's donor weights, indexed by donor name. When given, a
        ``weight`` column is added so the table reads alongside the fit.

    Returns
    -------
    pandas.DataFrame
        One row per donor (in the given order), indexed by donor, columns:
          - ``distance_km`` -- great-circle distance to the treated unit
          - ``bearing_deg`` -- compass bearing from the treated unit (0 = N)
          - ``sector``      -- the bearing's compass sector
          - ``weight``      -- the donor's fit weight (when ``weights`` given)

        A donor with no coordinate in ``df`` gets a row of NaNs and sector
        ``"?"`` rather than being dropped: the table is a reading of the pool
        as named, and a missing fix is itself information.
    """
    coords = _unit_coordinates(df, unit_col=unit_col, lat_col=lat_col, lon_col=lon_col)
    if treated_unit not in coords.index:
        raise ValueError(f"`treated_unit` {treated_unit!r} has no coordinate in `df`.")
    tlat = float(np.asarray(coords.loc[treated_unit, lat_col]))
    tlon = float(np.asarray(coords.loc[treated_unit, lon_col]))

    rows = []
    for d in donors:
        if d in coords.index:
            dlat = float(np.asarray(coords.loc[d, lat_col]))
            dlon = float(np.asarray(coords.loc[d, lon_col]))
            dist = float(np.asarray(haversine(tlat, tlon, dlat, dlon)))
            brg = float(np.asarray(bearing(tlat, tlon, dlat, dlon)))
            rows.append(
                {
                    "donor": d,
                    "distance_km": dist,
                    "bearing_deg": brg,
                    "sector": _sector_label(brg, n_sectors),
                }
            )
        else:
            rows.append({"donor": d, "distance_km": np.nan, "bearing_deg": np.nan, "sector": "?"})
    out = pd.DataFrame(rows, columns=["donor", "distance_km", "bearing_deg", "sector"]).set_index(
        "donor"
    )
    if weights is not None:
        w = pd.Series(weights, dtype=float)
        out["weight"] = out.index.map(lambda d: float(w[d]) if d in w.index else np.nan)
    return out


def spatial_summary(
    donor_df: pd.DataFrame, *, n_donors: int | None = None, n_sectors: int = 8
) -> dict[str, Any]:
    """The pool-level reading behind a :func:`donor_spatial_summary` table.

    Parameters
    ----------
    donor_df : pandas.DataFrame
        The output of :func:`donor_spatial_summary`.
    n_donors : int, optional
        The pool's intended size (len of ``donors``). When it exceeds the
        number of placed donors, the gap is reported as missing fixes.
    n_sectors : int, default 8
        The sector count the table was built with; the coverage fraction is
        read against it, not against however many sectors the donors happened
        to span (a pool in three sectors still covers three of eight, not
        three of three).

    Returns
    -------
    dict
      - ``centroid_offset_km`` / ``centroid_bearing_deg`` -- where the pool's
        centroid sits from the treated unit (NaN when no donor is placed).
      - ``sector_coverage`` -- one (sector, count) pair per non-empty sector,
        in bearing order.
      - ``sector_fraction`` -- the non-empty sectors' share of the compass:
        how much of the surroundings the pool actually spans.
      - ``n_donors_placed`` / ``n_donors_missing``.
    """
    placed = donor_df.dropna(subset=["distance_km"])
    n_placed = int(len(placed))
    n_missing = int((n_donors if n_donors is not None else n_placed) - n_placed)

    if n_placed == 0:
        return {
            "centroid_offset_km": np.nan,
            "centroid_bearing_deg": np.nan,
            "sector_coverage": [],
            "sector_fraction": 0.0,
            "n_donors_placed": 0,
            "n_donors_missing": max(n_missing, 0),
        }

    # The centroid as an offset vector from the treated unit: each donor
    # contributes (distance, bearing) as an (east, north) vector, averaged --
    # the honest way to average directions, since a mean of angles wraps.
    rad = np.deg2rad(placed["bearing_deg"].to_numpy(dtype=float))
    d = placed["distance_km"].to_numpy(dtype=float)
    east = float(np.mean(d * np.sin(rad)))
    north = float(np.mean(d * np.cos(rad)))
    centroid_km = float(np.hypot(east, north))
    centroid_brg = float(np.degrees((np.arctan2(east, north) + 2.0 * np.pi) % (2.0 * np.pi)))

    counts = placed["sector"].value_counts()
    # Sort sectors by their smallest member's bearing (the table's order),
    # not alphabetically: N must come before S, not after.
    sectors = sorted(
        counts.index, key=lambda s: float(placed.loc[placed["sector"] == s, "bearing_deg"].min())
    )
    sector_coverage = [(s, int(counts[s])) for s in sectors]
    sector_fraction = len(sectors) / float(n_sectors) if n_sectors else np.nan

    return {
        "centroid_offset_km": centroid_km,
        "centroid_bearing_deg": centroid_brg,
        "sector_coverage": sector_coverage,
        "sector_fraction": sector_fraction,
        "n_donors_placed": n_placed,
        "n_donors_missing": max(n_missing, 0),
    }
