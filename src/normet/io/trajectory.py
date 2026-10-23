# src/normet/io/trajectory.py
"""HYSPLIT back-trajectory adapter.

Turns HYSPLIT back-trajectory output into per-receptor feature rows that can be
joined onto an air-quality panel and used as transport-aware predictors in
``normet`` pipelines (``do_all`` / ``train_model`` / ``normalise``).

Scope: this module *consumes* trajectory output — it does not run HYSPLIT.
Generate the ``tdump`` files separately (e.g. with ``pysplit``, ``splitr``, or
the HYSPLIT ``hyts_std`` executable), then point :func:`build_trajectory_features`
at them.

Workflow
--------
>>> import normet.io as nio
>>> feats = nio.build_trajectory_features(
...     "traj/tdump_*",
...     source_regions={"industrial_NE": (116.0, 39.0, 120.0, 42.0)},
...     min_hours=72,  # null out trajectories cut short by the met data
... )                                                        # doctest: +SKIP
>>> # Align backward: each hour takes the latest trajectory that has already
>>> # arrived, within one release interval. A nearest-time join would hand
>>> # hours 04-05 the 06:00 trajectory, i.e. air that has not arrived yet.
>>> df = nio.join_trajectory_features(df, feats)             # doctest: +SKIP
>>> # then pass the ``traj_*`` columns to do_all, also in variables_resample

Evaluation caveat
-----------------
Trajectories released every few hours and carried onto an hourly panel are
piecewise constant: every hour of a release window shares one identical feature
vector. Under a *random* train/test split a tree model can use that vector as a
time fingerprint and memorise neighbouring hours, inflating test R^2 (on the
bundled MY1 case a trajectory-only model scores ~0.84 under a random split but
~0.46 under a blocked one). Judge transport-aware models with a blocked split
(``split_method="month_ts"`` / ``"ts"``), or release trajectories hourly.

Quality columns
---------------
Every feature row also carries ``traj_n_endpoints`` and ``traj_age_max_h``
(how far back the trajectory actually reached). A trajectory that terminates
early -- the met data ran out, or it left the model domain -- otherwise looks
like a legitimately short-range one: its ``dist_km`` shrinks and its residence
fractions are computed over fewer points, with no flag. ``min_hours`` turns
such rows into NaN instead.
"""

from __future__ import annotations

import numbers
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterable, Mapping
from glob import glob
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ..utils import progress
from ..utils._time import to_datetime_coerced
from ..utils.logging import get_logger
from .gdas import _gdas1_file_range, gdas1_filenames

log = get_logger(__name__)

__all__ = [
    "read_trajectory_tdump",
    "trajectory_features",
    "build_trajectory_features",
    "join_trajectory_features",
    "run_back_trajectories",
    "load_source_regions",
    "ALL_DIAGNOSTICS",
]

# HYSPLIT diagnostic variables -> normet-friendly short names.
_DIAG_RENAME = {
    "mixdepth": "blh",
    "relhumid": "rh",
    "air_temp": "temp",
    "rainfall": "rainfall",
    "pressure": "pressure",
}

_BASE_COLS = [
    "traj",
    "grid",
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "fcast",
    "age",
    "lat",
    "lon",
    "height",
]


def read_trajectory_tdump(path: str | Path) -> pd.DataFrame:
    """Parse a HYSPLIT ``tdump`` trajectory file into a tidy DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to a single HYSPLIT endpoints (``tdump``) file.

    Returns
    -------
    pandas.DataFrame
        One row per trajectory endpoint with columns ``traj`` (trajectory
        index, in case the file holds several), ``datetime`` (endpoint time),
        ``age_h`` (hours from the receptor; 0 at the receptor, negative going
        back), ``lat``, ``lon``, ``height``, plus any diagnostic variables the
        run wrote (e.g. ``pressure``, ``rainfall``, ``blh`` (from ``MIXDEPTH``),
        ``rh`` (from ``RELHUMID``), ``temp`` (from ``AIR_TEMP``)).
    """
    path = Path(path)
    lines = [ln for ln in path.read_text().splitlines()]
    i = 0
    # 1) number of meteorological grids, then one info line each.
    n_met = int(lines[i].split()[0])
    i += 1 + n_met
    # 2) "<n_traj> <direction> <vert-motion>", then one starting-location line each.
    n_traj = int(lines[i].split()[0])
    i += 1 + n_traj
    # 3) "<n_var> <NAME1> <NAME2> ...": diagnostic output variables.
    parts = lines[i].split()
    n_var = int(parts[0])
    var_names = [v.lower() for v in parts[1 : 1 + n_var]]
    i += 1

    cols = _BASE_COLS + var_names
    rows = [ln.split()[: len(cols)] for ln in lines[i:] if ln.strip()]
    rows = [r for r in rows if len(r) == len(cols)]
    if not rows:
        raise ValueError(f"No trajectory data rows parsed from {path}")

    df = pd.DataFrame(rows, columns=cols).apply(pd.to_numeric, errors="coerce")

    yr = df["year"].astype(int).to_numpy()
    yr = np.where(yr < 50, 2000 + yr, np.where(yr < 100, 1900 + yr, yr))
    df["datetime"] = pd.to_datetime(
        {
            "year": yr,
            "month": df["month"].astype(int),
            "day": df["day"].astype(int),
            "hour": df["hour"].astype(int),
            "minute": df["minute"].astype(int),
        }
    )
    df["age_h"] = df["age"].astype(float)
    df = df.rename(columns={k: v for k, v in _DIAG_RENAME.items() if k in df.columns})
    return df


def _haversine_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> Any:
    """Great-circle distance in km (scalar or vectorised)."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2, dtype=float) - lat1)
    dl = np.radians(np.asarray(lon2, dtype=float) - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _bearing_deg(lat0: float, lon0: float, lat1: float, lon1: float) -> float:
    """Initial bearing (deg from North) from the receptor to the trajectory origin."""
    dl = np.radians(lon1 - lon0)
    y = np.sin(dl) * np.cos(np.radians(lat1))
    x = np.cos(np.radians(lat0)) * np.sin(np.radians(lat1)) - np.sin(np.radians(lat0)) * np.cos(
        np.radians(lat1)
    ) * np.cos(dl)
    return float((np.degrees(np.arctan2(y, x)) + 360) % 360)


def _is_bbox(region: Any) -> bool:
    """A ``(lon_min, lat_min, lon_max, lat_max)`` box, numpy scalars included."""
    return (
        isinstance(region, tuple | list)
        and len(region) == 4
        and all(isinstance(v, numbers.Real) and not isinstance(v, bool) for v in region)
    )


def _region_mask(region: Any, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Point-in-region test — a 4-tuple bbox, or a shapely geometry (polygon
    boundaries loaded from GeoJSON via :func:`load_source_regions`)."""
    if _is_bbox(region):
        xmn, ymn, xmx, ymx = (float(v) for v in region)
        return (lon >= xmn) & (lon <= xmx) & (lat >= ymn) & (lat <= ymx)
    from shapely import contains_xy

    return contains_xy(region, lon, lat)


def _overlapping_regions(source_regions: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Pairs of source regions that share area; regions that only touch do not."""
    items = list(source_regions.items())
    pairs: list[tuple[str, str]] = []
    for i, (name_a, a) in enumerate(items):
        for name_b, b in items[i + 1 :]:
            if _is_bbox(a) and _is_bbox(b):
                ax0, ay0, ax1, ay1 = (float(v) for v in a)
                bx0, by0, bx1, by1 = (float(v) for v in b)
                shared = ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1
            else:
                from shapely.geometry import box

                ga = box(*(float(v) for v in a)) if _is_bbox(a) else a
                gb = box(*(float(v) for v in b)) if _is_bbox(b) else b
                shared = ga.intersection(gb).area > 0
            if shared:
                pairs.append((str(name_a), str(name_b)))
    return pairs


def _load_regions_geojson(path: Path) -> dict[str, Any]:
    import json

    from shapely.geometry import shape

    with open(path, encoding="utf-8") as fh:
        gj = json.load(fh)
    features = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]

    out: dict[str, Any] = {}
    for i, feat in enumerate(features):
        props = feat.get("properties") or {}
        name = str(props.get("name") or props.get("id") or props.get("NAME") or f"region_{i}")
        out[name] = shape(feat["geometry"])
    return out


def _load_regions_shapefile(path: Path) -> dict[str, Any]:
    import shapefile  # pyshp
    from shapely.geometry import shape

    reader = shapefile.Reader(str(path))
    fields = [f[0] for f in reader.fields[1:]]  # skip the leading DeletionFlag
    name_col = next((f for f in ("name", "NAME", "Name", "id", "ID") if f in fields), None)

    out: dict[str, Any] = {}
    for i, srec in enumerate(reader.shapeRecords()):
        name = str(srec.record[name_col]) if name_col else f"region_{i}"
        out[name] = shape(srec.shape.__geo_interface__)
    return out


def load_source_regions(path: str | Path) -> dict[str, Any]:
    """Load named region polygons from a GeoJSON or ESRI Shapefile.

    Parameters
    ----------
    path : str or Path
        A ``.geojson``/``.json`` ``FeatureCollection``, or a ``.shp`` file
        (its sibling ``.dbf``/``.shx`` must sit alongside it). Each
        feature's polygon/multipolygon geometry becomes one region, named
        from a ``name``/``id`` (or ``NAME``/``ID``) attribute, falling back
        to ``region_<i>``.

    Returns
    -------
    dict
        ``{name: shapely.Geometry}``, ready to pass (alone or mixed with
        bbox tuples) as :func:`trajectory_features`'s ``source_regions``.

    Notes
    -----
    Needs ``shapely`` for both formats, plus ``pyshp`` for ``.shp``
    (``pip install normet[geo]`` gets both). Point-in-polygon residence time
    is exact — unlike a bounding box, it respects concave and multi-part
    boundaries, so this is the natural way to use real administrative or
    airshed boundaries (many of which ship as Shapefiles, not GeoJSON).
    """
    path = Path(path)
    if path.suffix.lower() == ".shp":
        return _load_regions_shapefile(path)
    return _load_regions_geojson(path)


def trajectory_features(
    traj: pd.DataFrame,
    *,
    source_regions: Mapping[str, tuple[float, float, float, float] | Any] | None = None,
    prefix: str = "traj_",
    min_hours: float | None = None,
) -> dict[str, float]:
    """Collapse one back-trajectory into a fixed-length feature dict.

    Parameters
    ----------
    traj : pandas.DataFrame
        A single trajectory (e.g. one ``traj`` group from
        :func:`read_trajectory_tdump`). Needs ``age_h``, ``lat``, ``lon``,
        ``height``; uses ``rainfall`` / ``blh`` if present.
    source_regions : mapping, optional
        ``{name: region}`` where each region is either a
        ``(lon_min, lat_min, lon_max, lat_max)`` bounding box, or a shapely
        polygon/multipolygon (e.g. from :func:`load_source_regions`)
        for exact point-in-polygon residence time. For each, the fraction of
        trajectory time spent inside is returned as ``{prefix}resid_{name}``.
        An endpoint inside several overlapping regions counts towards each, so
        the fractions are only shares of the trajectory when the regions do not
        overlap; :func:`build_trajectory_features` warns when they do.
    prefix : str, default ``"traj_"``
        Prefix for every feature name.
    min_hours : float, optional
        Minimum backward reach (hours) a trajectory must have to be trusted.
        A trajectory whose span is shorter -- HYSPLIT stopped early because the
        met files ran out, or it left the model domain -- has every feature
        except the two quality columns set to NaN, rather than passing off a
        truncated path as a short-range one. ``None`` (default) keeps every
        trajectory, leaving detection to ``{prefix}age_max_h``.

    Returns
    -------
    dict
        Quality descriptors ``{prefix}n_endpoints`` and ``{prefix}age_max_h``
        (hours the trajectory actually reached back), then transport
        descriptors: straight-line reach, path length, mean transport
        speed, inflow bearing, mean/min height, per-region residence
        fractions, and — only if the ``tdump`` run wrote them — along-path
        rainfall sum, mean boundary-layer height, mean relative humidity,
        mean pressure, and mean air temperature.
    """
    if traj is None or traj.empty:
        return {}
    # age_h is 0 at the receptor and negative going back, so sort descending to
    # put the receptor first and the oldest endpoint (air origin) last.
    t = traj.sort_values("age_h", ascending=False)
    lat0, lon0 = float(t["lat"].iloc[0]), float(t["lon"].iloc[0])
    latn, lonn = float(t["lat"].iloc[-1]), float(t["lon"].iloc[-1])

    step = _haversine_km(
        t["lat"].to_numpy()[:-1],
        t["lon"].to_numpy()[:-1],
        t["lat"].to_numpy()[1:],
        t["lon"].to_numpy()[1:],
    )
    path_len = float(np.nansum(step))
    span_h = float(abs(t["age_h"].iloc[-1] - t["age_h"].iloc[0]))

    f = {
        f"{prefix}n_endpoints": float(len(t)),
        f"{prefix}age_max_h": span_h,
        f"{prefix}dist_km": float(_haversine_km(lat0, lon0, latn, lonn)),
        f"{prefix}pathlen_km": path_len,
        f"{prefix}speed_kmh": path_len / span_h if span_h > 0 else np.nan,
        f"{prefix}inflow_deg": _bearing_deg(lat0, lon0, latn, lonn),
        f"{prefix}height_mean": float(t["height"].mean()),
        f"{prefix}height_min": float(t["height"].min()),
    }
    if source_regions:
        lon, lat = t["lon"].to_numpy(), t["lat"].to_numpy()
        for name, region in source_regions.items():
            inside = _region_mask(region, lon, lat)
            f[f"{prefix}resid_{name}"] = float(np.asarray(inside).mean())
    if "rainfall" in t:
        f[f"{prefix}rain_sum"] = float(t["rainfall"].sum())
    if "blh" in t:
        f[f"{prefix}blh_mean"] = float(t["blh"].mean())
    if "rh" in t:
        f[f"{prefix}rh_mean"] = float(t["rh"].mean())
    if "pressure" in t:
        f[f"{prefix}pressure_mean"] = float(t["pressure"].mean())
    if "temp" in t:
        f[f"{prefix}temp_mean"] = float(t["temp"].mean())
    if min_hours is not None and span_h < min_hours - 1e-6:
        quality = {f"{prefix}n_endpoints", f"{prefix}age_max_h"}
        f = {k: (v if k in quality else np.nan) for k, v in f.items()}
    return f


def build_trajectory_features(
    tdumps: str | Iterable[str | Path],
    *,
    source_regions: Mapping[str, tuple[float, float, float, float]] | None = None,
    prefix: str = "traj_",
    date_col: str = "date",
    min_hours: float | None = None,
) -> pd.DataFrame:
    """Build a receptor-time feature table from many HYSPLIT ``tdump`` files.

    Parameters
    ----------
    tdumps : str or iterable of paths
        A glob pattern (e.g. ``"traj/tdump_*"``) or an explicit iterable of
        ``tdump`` file paths. One back-trajectory run per receptor time is the
        typical layout; files holding multiple trajectories are split per
        ``traj`` index.
    source_regions, prefix, min_hours
        Forwarded to :func:`trajectory_features`. With ``min_hours`` set, the
        number of trajectories nulled for being truncated is logged.
    date_col : str, default ``"date"``
        Name of the index column (receptor timestamp), so the result joins
        straight onto a date-indexed panel.

    Returns
    -------
    pandas.DataFrame
        Indexed by receptor timestamp, one ``{prefix}*`` column per feature.
        Sorted by time; deduplicated on the receptor timestamp (last wins).
        Trajectories started at several heights -- a run with a column of
        starting points, or runs at different heights mixed -- each get their
        own columns, suffixed with the start height: ``traj_dist_km_500m``,
        ``traj_dist_km_1500m``.

    Raises
    ------
    ValueError
        When the trajectories start from more than one place. A feature table
        is one receptor's: build each site's on its own (and tag it with a
        ``site`` column to join several, see :func:`join_trajectory_features`).
    """
    paths = sorted(glob(tdumps)) if isinstance(tdumps, str) else [str(p) for p in tdumps]
    if not paths:
        raise ValueError(f"No tdump files matched: {tdumps!r}")
    overlaps = _overlapping_regions(source_regions) if source_regions else []
    if overlaps:
        log.warning(
            "Source regions overlap (%s): an endpoint in a shared area counts towards each "
            "of them, so their residence fractions can add up to more than 1 and are not "
            "shares of the trajectory.",
            ", ".join(f"{a} & {b}" for a, b in overlaps),
        )

    rows: list[dict[str, Any]] = []
    for p in paths:
        try:
            traj = read_trajectory_tdump(p)
        except Exception as exc:  # skip unparseable files but say so
            log.warning("Skipping trajectory file %s: %s", p, exc)
            continue
        for _, g in traj.groupby("traj"):
            # The endpoint at age 0 is where (and when) the trajectory starts.
            start = g.loc[g["age_h"].abs().idxmin()]
            feats = trajectory_features(
                g, source_regions=source_regions, prefix=prefix, min_hours=min_hours
            )
            rows.append(
                {
                    date_col: pd.Timestamp(cast(Any, start["datetime"])),
                    _START_LAT: float(start["lat"]),
                    _START_LON: float(start["lon"]),
                    _START_HEIGHT: round(float(start["height"]), 1),
                    **feats,
                }
            )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    _check_one_receptor(frame)
    frame = frame.drop(columns=[_START_LAT, _START_LON]).drop_duplicates(
        subset=[date_col, _START_HEIGHT], keep="last"
    )
    if min_hours is not None and f"{prefix}age_max_h" in frame:
        n_short = int((frame[f"{prefix}age_max_h"] < min_hours - 1e-6).sum())
        if n_short:
            log.warning(
                "%d of %d trajectories reach back < %g h (truncated); their features are NaN.",
                n_short,
                len(frame),
                min_hours,
            )
    heights = sorted(frame[_START_HEIGHT].unique())
    if len(heights) > 1:
        # Keyed on time alone, the heights would overwrite one another: one column
        # set per height instead.
        out = pd.concat(
            [
                part.drop(columns=_START_HEIGHT).set_index(date_col).add_suffix(f"_{h:g}m")
                for h, part in frame.groupby(_START_HEIGHT)
            ],
            axis=1,
        ).sort_index()
        log.info(
            "Trajectories start at %d heights (%s m): each height's features are suffixed with it.",
            len(heights),
            ", ".join(f"{h:g}" for h in heights),
        )
    else:
        out = frame.drop(columns=_START_HEIGHT).set_index(date_col).sort_index()
    log.info("Built trajectory features: %d receptors × %d columns", len(out), out.shape[1])
    return out


# Where a trajectory starts, carried alongside its features while the table is built.
_START_LAT, _START_LON, _START_HEIGHT = "__start_lat", "__start_lon", "__start_height"
# Starting points this close (degrees) are one receptor: tdumps print 3 decimals.
_SAME_PLACE_DEG = 0.01


def _check_one_receptor(frame: pd.DataFrame) -> None:
    """Raise if the trajectories in *frame* start from more than one place."""
    lat, lon = frame[_START_LAT], frame[_START_LON]
    if lat.max() - lat.min() <= _SAME_PLACE_DEG and lon.max() - lon.min() <= _SAME_PLACE_DEG:
        return
    places = frame[[_START_LAT, _START_LON]].round(2).drop_duplicates()
    shown = ", ".join(f"({a:g}, {b:g})" for a, b in places.head(4).itertuples(index=False))
    raise ValueError(
        f"The trajectories start from {len(places)} different places ({shown}"
        f"{', ...' if len(places) > 4 else ''}): build each receptor's features on its own, "
        "tagged with a 'site' column if they are to be joined together."
    )


def _naive_utc(values: Any) -> pd.Series:
    """Timestamps on HYSPLIT's clock: tz-naive UTC. A naive value is taken as UTC
    already; anything unparseable becomes NaT."""
    s = pd.Series(values).reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = to_datetime_coerced(s, utc=True)
    if s.dt.tz is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.astype("datetime64[ns]")


def _release_interval(times: pd.Series, sites: pd.Series) -> pd.Timedelta | None:
    """The commonest step between one site's successive trajectories (the smaller
    on a tie), or ``None`` when no site has two."""
    steps = (
        pd.DataFrame({"t": times.to_numpy(), "s": sites.to_numpy()})
        .sort_values(["s", "t"])
        .groupby("s")["t"]
        .diff()
        .dropna()
    )
    steps = steps[steps > pd.Timedelta(0)]
    return pd.Timedelta(steps.mode().iloc[0]) if len(steps) else None


def join_trajectory_features(
    df: pd.DataFrame,
    features: pd.DataFrame,
    *,
    date_col: str = "date",
    site_col: str = "site",
    tolerance: str | pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Join trajectory features onto a panel, each row taking the latest
    trajectory to have arrived.

    A row at time *t* takes the trajectory that arrived last at or before *t*,
    and only if it arrived less than ``tolerance`` before. A nearest-time join
    instead hands a row the trajectory arriving after it -- air that has not
    reached the receptor yet -- and, with no tolerance, carries the first and
    last trajectories over every row before and after them.

    Parameters
    ----------
    df : pandas.DataFrame
        The panel: times in a ``date_col`` column, or in a ``DatetimeIndex``.
    features : pandas.DataFrame
        Receptor-time features, e.g. from :func:`build_trajectory_features` or
        :func:`run_back_trajectories`: times in the index or a ``date_col``
        column, one row per time (per site, with a ``site_col`` column).
    date_col : str, default ``"date"``
        Name of the time column.
    site_col : str, default ``"site"``
        When both frames have it, each row takes only its own site's
        trajectories (sites are matched as text), and rows of sites
        ``features`` does not hold are left as they were -- so the features of
        several sites can be joined one batch at a time.
    tolerance : str or Timedelta, optional
        How long a trajectory stands for after it arrives (a row strictly less
        than this after it takes it). Default: the release interval, the
        commonest step between one site's successive trajectories -- every hour
        of a 6-hourly release window takes that window's trajectory, and a
        window whose trajectory is missing stays empty. With a single
        trajectory, only a row at its exact time takes it.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df``, rows in their order, with the feature columns added;
        a row no trajectory matches (or with no time) has them NaN. A feature
        column ``df`` already had is replaced; its other columns are untouched.

    Raises
    ------
    ValueError
        If ``features`` holds several sites but ``df`` has no ``site_col``, or
        several rows for one receptor time (and site).

    Notes
    -----
    Times are compared as UTC, HYSPLIT's clock: tz-aware times are converted
    and naive ones are taken to be UTC already.
    """
    if date_col in df.columns:
        when = _naive_utc(df[date_col])
    elif isinstance(df.index, pd.DatetimeIndex):
        when = _naive_utc(df.index)
    else:
        raise KeyError(f"df has no {date_col!r} column and no DatetimeIndex")
    if date_col in features.columns:
        ftimes = _naive_utc(features[date_col])
    elif isinstance(features.index, pd.DatetimeIndex) or features.index.dtype == object:
        ftimes = _naive_utc(features.index)
    else:
        raise KeyError(f"features have no {date_col!r} column and no DatetimeIndex")
    cols = [c for c in features.columns if c not in (date_col, site_col)]
    if not cols:
        raise ValueError("features hold no feature columns")

    fsite = (
        features[site_col].astype(str).reset_index(drop=True)
        if site_col in features.columns
        else pd.Series("", index=ftimes.index)
    )
    by_site = site_col in features.columns and site_col in df.columns
    feature_sites = sorted(set(fsite))
    if site_col in features.columns and not by_site and len(feature_sites) > 1:
        raise ValueError(
            f"features hold {len(feature_sites)} sites ({', '.join(feature_sites)}) but df has "
            f"no {site_col!r} column to match them on"
        )
    known = ftimes.notna()
    dup = pd.DataFrame({"t": ftimes, "s": fsite})[known].duplicated()
    if dup.any():
        first = ftimes[known][dup].iloc[0]
        raise ValueError(
            f"features hold {int(dup.sum())} more row(s) for a receptor time already given "
            f"(e.g. {first}{' at one site' if by_site else ''}); keep one per time"
        )

    if tolerance is not None:
        given = pd.Timedelta(tolerance)
        if not given > pd.Timedelta(0):
            raise ValueError(f"tolerance must be positive, got {tolerance!r}")
        tol: pd.Timedelta | None = given
    else:
        tol = _release_interval(ftimes[known], fsite[known])
    # merge_asof's tolerance is inclusive; a trajectory stands for strictly less.
    asof_tol = tol - pd.Timedelta(1, "ns") if tol is not None else pd.Timedelta(0)

    left = pd.DataFrame({"_t": when.to_numpy(), "_row": np.arange(len(df))})
    right = pd.DataFrame({"_t": ftimes.to_numpy(), "_pos": np.arange(len(features))})
    if by_site:
        left["_site"] = df[site_col].astype(str).to_numpy()
        right["_site"] = fsite.to_numpy()
    matched = pd.merge_asof(
        left[left["_t"].notna()].sort_values("_t", kind="stable"),
        right[right["_t"].notna()].sort_values("_t", kind="stable"),
        on="_t",
        by="_site" if by_site else None,
        direction="backward",
        tolerance=asof_tol,
    )
    pos = np.full(len(df), -1, dtype=np.int64)
    hit = matched["_pos"].notna().to_numpy()
    pos[matched["_row"].to_numpy()[hit]] = matched["_pos"].to_numpy()[hit].astype(np.int64)

    in_scope = (
        df[site_col].astype(str).isin(feature_sites).to_numpy()
        if by_site
        else np.ones(len(df), dtype=bool)
    )
    block = features[cols].reset_index(drop=True).reindex(pos)  # -1 -> a row of NaN
    out = df.copy()
    replaced = [c for c in cols if c in df.columns]
    for c in cols:
        # Explicit: np.where's stub return shape disagrees with to_numpy's
        # across numpy 2.x stub generations.
        values: np.ndarray = block[c].to_numpy()
        if not in_scope.all():
            prior = out[c].to_numpy() if c in replaced else np.full(len(out), np.nan)
            values = np.where(in_scope, values, prior)
        out[c] = values

    n_hit, n_scope = int((pos >= 0).sum()), int(in_scope.sum())
    gap = f"less than {_hours(tol)} before" if tol is not None else "at the same time as"
    log.info(
        "Joined %d trajectory feature column(s): %d of %d rows took a trajectory (%s).",
        len(cols),
        n_hit,
        n_scope,
        f"the latest to arrive, if {gap} the row" if tol is not None else "one arriving with it",
    )
    if replaced:
        log.info("Replaced column(s) the panel already had: %s", ", ".join(replaced))
    if site_col in df.columns and site_col not in features.columns:
        n_sites = df[site_col].nunique()
        if n_sites > 1:
            log.warning(
                "The features have no %r column, so the rows of all %d of the panel's sites "
                "took the same trajectories.",
                site_col,
                n_sites,
            )
    if by_site:
        present = set(left["_site"])
        absent = [s for s in feature_sites if s not in present]
        if absent:
            log.warning(
                "No rows of site(s) %s in the panel's %r column; their trajectories were not used.",
                ", ".join(absent),
                site_col,
            )
    if not n_hit:
        log.warning(
            "No row took a trajectory: none arrived %s any of the panel's times (%s → %s); "
            "they arrive %s → %s.",
            gap,
            when.min(),
            when.max(),
            ftimes.min(),
            ftimes.max(),
        )
    return out


def _hours(td: pd.Timedelta) -> str:
    return f"{td / pd.Timedelta(hours=1):g} h"


# GDAS1 is 3-hourly. A receptor time between a weekly file's last record and the
# next file's first one needs *both* to interpolate: probed against hyts_std with
# two adjacent daily ARL files, a start time in that gap (23:30, 23:59) failed
# with only the earlier file and ran with both. A strict "does the file's span
# overlap the window" test drops the next file there, so the window is widened by
# one record interval on each side.
_MET_RECORD_PAD_H = 3.0


def _filter_met_files(
    paths: list[str],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    *,
    pad_h: float = _MET_RECORD_PAD_H,
) -> list[str]:
    """Keep the ARL files whose span overlaps ``[window_start, window_end]``.

    The span comes from the GDAS1 filename (:func:`_gdas1_file_range`). A file whose
    name does not follow that convention is always kept -- its dates are unknown,
    so dropping it could silently truncate the trajectory. The window is widened
    by ``pad_h`` on each side (see ``_MET_RECORD_PAD_H``).
    """
    pad = pd.Timedelta(hours=pad_h)
    lo, hi = window_start - pad, window_end + pad
    kept = []
    for p in paths:
        span = _gdas1_file_range(p)
        if span is None or (span[1] >= lo and span[0] <= hi):
            kept.append(p)
    return kept


def _missing_gdas1_files(
    paths: list[str],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    *,
    pad_h: float = _MET_RECORD_PAD_H,
) -> list[str] | None:
    """The GDAS1 weekly files ``[window_start, window_end]`` needs that *paths* lack.

    ``None`` when a file's name does not say what it covers, so the coverage
    cannot be judged. The window is widened as in :func:`_filter_met_files`.
    """
    if any(_gdas1_file_range(p) is None for p in paths):
        return None
    pad = pd.Timedelta(hours=pad_h)
    have = {Path(p).name for p in paths}
    return [n for n in gdas1_filenames(window_start - pad, window_end + pad) if n not in have]


def _control_text(
    time: pd.Timestamp,
    lat: float,
    lon: float,
    height_m: float,
    hours_back: int,
    met_files: list[str],
    tdump_name: str,
    *,
    top_of_model: float,
    vert_motion: int,
) -> str:
    """Render a HYSPLIT ``CONTROL`` file for a single backward trajectory."""
    ts = pd.Timestamp(time)
    lines = [
        # start: YY MM DD HH (2-digit year), and MM only when it is not on the hour
        ts.strftime("%y %m %d %H %M" if ts.minute else "%y %m %d %H"),
        "1",  # one starting location
        f"{float(lat):.4f} {float(lon):.4f} {float(height_m):.1f}",
        str(-abs(int(hours_back))),  # negative run hours = BACKWARD
        str(int(vert_motion)),  # 0 = use met vertical velocity
        f"{float(top_of_model):.1f}",
        str(len(met_files)),  # number of met files
    ]
    for mf in met_files:  # each met file -> (dir/, file) pair
        ab = os.path.abspath(os.path.expanduser(str(mf)))
        lines.append(os.path.dirname(ab) + os.sep)
        lines.append(os.path.basename(ab))
    lines.append("./")  # output dir (cwd == work_dir)
    lines.append(tdump_name)
    return "\n".join(lines) + "\n"


# normet-friendly diagnostic name -> HYSPLIT SETUP.CFG namelist flag. These
# control which along-trajectory meteorological variables hyts_std writes
# into tdump (see trajectory_features()'s optional traj_rain_sum/blh_mean/
# rh_mean/pressure_mean/temp_mean columns).
_DIAG_SETUP_FLAGS = {
    "pressure": "tm_pres",
    "rainfall": "tm_rain",
    "blh": "tm_mixd",
    "rh": "tm_relh",
    "temp": "tm_tamb",
}
ALL_DIAGNOSTICS = tuple(_DIAG_SETUP_FLAGS)


def _setup_cfg_text(diagnostics: Iterable[str]) -> str:
    """Render a HYSPLIT ``SETUP.CFG`` &SETUP namelist toggling tdump diagnostics.

    Without this file, ``hyts_std`` falls back to its own built-in defaults
    (typically just ``PRESSURE``) — every diagnostic is written explicitly
    here (0 or 1) so the run is reproducible regardless of what those
    defaults are.
    """
    wanted = set(diagnostics)
    unknown = wanted - set(_DIAG_SETUP_FLAGS)
    if unknown:
        raise ValueError(f"Unknown diagnostic(s) {sorted(unknown)}; choose from {ALL_DIAGNOSTICS}")
    lines = ["&SETUP"]
    for name, flag in _DIAG_SETUP_FLAGS.items():
        lines.append(f"{flag} = {1 if name in wanted else 0},")
    lines.append("/")
    return "\n".join(lines) + "\n"


def _utc(ts: pd.Timestamp) -> pd.Timestamp:
    """*ts* on HYSPLIT's clock: tz-naive UTC (a naive time is taken as UTC already)."""
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts


def _run_hyts(
    exe: str, work: Path, timeout: float, stop: threading.Event | None
) -> subprocess.CompletedProcess[str] | None:
    """Run ``hyts_std`` once in *work*; ``None`` if *stop* was set while it ran.

    A stopped or timed-out run is killed rather than left to finish.
    """
    proc = subprocess.Popen(
        [exe],
        cwd=work,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + timeout
    while True:
        left = deadline - time.monotonic()
        try:
            wait = min(left, 0.2) if stop is not None else left
            out, err = proc.communicate(timeout=max(0.0, wait))
        except subprocess.TimeoutExpired:
            if stop is not None and stop.is_set():
                proc.kill()
                proc.communicate()
                return None
            if time.monotonic() >= deadline:
                proc.kill()
                proc.communicate()
                raise
            continue
        return subprocess.CompletedProcess([exe], proc.returncode, out, err)


def run_back_trajectories(
    times: Iterable[Any],
    lat: float,
    lon: float,
    *,
    met_files: str | Iterable[str],
    hysplit_exec: str | Path,
    height_m: float = 500.0,
    hours_back: int = 72,
    work_dir: str | Path | None = None,
    top_of_model: float = 10000.0,
    vert_motion: int = 0,
    diagnostics: Iterable[str] = ALL_DIAGNOSTICS,
    source_regions: Mapping[str, tuple[float, float, float, float]] | None = None,
    prefix: str = "traj_",
    min_hours: float | None = None,
    timeout: int = 600,
    stop: threading.Event | None = None,
) -> pd.DataFrame:
    """Run HYSPLIT back-trajectories for many receptor times and reduce to features.

    For each timestamp this writes a ``CONTROL`` file, runs the HYSPLIT
    ``hyts_std`` executable (one backward trajectory ending at ``(lat, lon,
    height_m)``), then passes all resulting ``tdump`` files to
    :func:`build_trajectory_features`.

    Trajectory generation is done outside ``normet`` proper — you must supply a
    built ``hyts_std`` and ARL-format met files. This helper only orchestrates
    the runs and parses the output.

    Parameters
    ----------
    times : iterable
        Receptor (arrival) timestamps; anything ``pandas.Timestamp`` accepts.
        HYSPLIT runs on UTC: tz-aware times are converted, naive ones are taken
        as UTC. Repeats are run once, and ``NaT`` is skipped.
    lat, lon : float
        Receptor location (degrees).
    met_files : str or iterable of str
        ARL-format meteorology file(s). They must collectively cover the full
        backward window (``hours_back`` before each receptor time), or the
        trajectory truncates where the data runs out. Each run is handed only
        the files that can touch its window: GDAS1 weekly files
        (``gdas1.<mmm><yy>.w<N>``) are selected by the dates in their name,
        widened by one 3-hourly record either side; any file with another name
        is always passed. A receptor time no file can touch is skipped, with a
        warning, rather than run.
    hysplit_exec : str or Path
        Path to the ``hyts_std`` executable (e.g. ``~/hysplit-5.4.2/exec/hyts_std``).
    height_m : float, default 500.0
        Receptor start height (m AGL).
    hours_back : int, default 72
        Backward duration in hours (sign ignored; always run backward).
    work_dir : str or Path, optional
        Directory for ``CONTROL`` and ``tdump_*`` files. A temp dir is created
        if omitted; the tdumps are left there for inspection/reuse. A tdump
        an earlier call left under the same name is removed before its run, so
        a failed run cannot pass the old file off as its own.
    top_of_model, vert_motion : float, int
        HYSPLIT ``CONTROL`` settings (model top in m; ``0`` = use met w).
    diagnostics : iterable of str, default all of them
        Along-trajectory meteorological variables ``hyts_std`` should write
        into ``tdump`` (subset of ``"pressure"``, ``"rainfall"``, ``"blh"``,
        ``"rh"``, ``"temp"``) — written into a ``SETUP.CFG`` alongside
        ``CONTROL``. These feed :func:`trajectory_features`'s optional
        ``traj_rain_sum``/``traj_blh_mean``/``traj_rh_mean``/
        ``traj_pressure_mean``/``traj_temp_mean`` columns; pass fewer to
        skip the ones you don't need.
    source_regions, prefix, min_hours
        Forwarded to :func:`build_trajectory_features`. Pass
        ``min_hours=hours_back`` to null trajectories that HYSPLIT ended early;
        either way, truncated runs are counted in a warning.
    timeout : int, default 600
        Per-run timeout (seconds) for ``hyts_std``.
    stop : threading.Event, optional
        Set it (from another thread) to stop: the run in progress is killed, no
        other starts, and the features of the runs already done are returned
        (an empty frame if none was).

    Returns
    -------
    pandas.DataFrame
        The :func:`build_trajectory_features` table (one row per receptor time).

    Notes
    -----
    macOS: HYSPLIT ships ``x86_64`` binaries (run under Rosetta on Apple
    Silicon) downloaded via a disk image, so they carry a Gatekeeper quarantine
    flag. If ``hyts_std`` is killed (exit 137) clear it once::

        xattr -dr com.apple.quarantine /path/to/hysplit

    Runs are sequential (each overwrites ``CONTROL``/``MESSAGE`` in ``work_dir``).
    """
    exe = os.path.abspath(os.path.expanduser(str(hysplit_exec)))
    if not os.access(exe, os.X_OK):
        raise FileNotFoundError(f"hyts_std not found or not executable: {exe}")

    mets = [met_files] if isinstance(met_files, str | Path) else list(met_files)
    mets = [os.path.abspath(os.path.expanduser(str(m))) for m in mets]
    if not mets:
        raise ValueError("No met files given")
    missing = [m for m in mets if not os.path.exists(m)]
    if missing:
        raise FileNotFoundError(f"Met file(s) not found: {missing}")

    work = Path(work_dir).expanduser() if work_dir else Path(tempfile.mkdtemp(prefix="nm_traj_"))
    work.mkdir(parents=True, exist_ok=True)

    # hyts_std requires ASCDATA.CFG (surface land-use/roughness config) in the
    # run dir, else it aborts with a header-only tdump. Stage it from the
    # install's bdyfiles/ (the trajectory falls back to default surface fields
    # if the referenced land-use data isn't co-located — fine for trajectories).
    if not (work / "ASCDATA.CFG").exists():
        ascdata = Path(exe).resolve().parents[1] / "bdyfiles" / "ASCDATA.CFG"
        if ascdata.exists():
            shutil.copy(ascdata, work / "ASCDATA.CFG")
        else:
            log.warning("ASCDATA.CFG not found at %s; hyts_std may abort (sfcinp).", ascdata)

    # Without SETUP.CFG, hyts_std falls back to its own built-in diagnostic
    # defaults (typically just PRESSURE); write it explicitly so the
    # requested tdump columns (rainfall/blh/rh/temp) are reproducible.
    (work / "SETUP.CFG").write_text(_setup_cfg_text(diagnostics))

    stamps = [_utc(pd.Timestamp(t)) for t in times]
    stamps = list(dict.fromkeys(ts for ts in stamps if not pd.isna(ts)))
    window = pd.Timedelta(hours=abs(hours_back))
    tdumps: list[str] = []
    uncovered: list[pd.Timestamp] = []
    stopped = False
    every = max(1, len(stamps) // 10)
    for i, ts in enumerate(stamps, 1):
        if stop is not None and stop.is_set():
            stopped = True
            break
        progress.report(i - 1, len(stamps))
        # Minutes in the name only off the hour, so on-the-hour names stay as they were.
        name = "tdump_" + ts.strftime("%Y%m%d%H%M" if ts.minute else "%Y%m%d%H")
        # Hand hyts_std only the files this run can touch: a multi-year archive is
        # hundreds of weekly files, and every one would otherwise be listed in --
        # and opened by -- each of the thousands of runs.
        run_mets = _filter_met_files(mets, ts - window, ts)
        if not run_mets:
            uncovered.append(ts)
            continue
        out = work / name
        out.unlink(missing_ok=True)  # else a run that fails would leave the last one's
        (work / "CONTROL").write_text(
            _control_text(
                ts,
                lat,
                lon,
                height_m,
                hours_back,
                run_mets,
                name,
                top_of_model=top_of_model,
                vert_motion=vert_motion,
            )
        )
        try:
            r = _run_hyts(exe, work, timeout, stop)
        except subprocess.TimeoutExpired:
            log.warning("hyts_std timed out (%ds) for receptor %s", timeout, ts)
            continue
        if r is None:
            stopped = True
            break

        if r.returncode == 137:  # SIGKILL — almost always macOS Gatekeeper quarantine
            raise RuntimeError(
                "hyts_std was killed (exit 137) — likely macOS Gatekeeper quarantine. "
                f"Clear it once with:\n  xattr -dr com.apple.quarantine {Path(exe).parents[1]}"
            )
        if not out.exists():
            msg = (r.stderr or r.stdout or "").strip().replace("\n", " ")[:200]
            log.warning("No tdump for %s (hyts_std rc=%s): %s", ts, r.returncode, msg)
            continue
        tdumps.append(str(out))
        if i % every == 0 and i < len(stamps):
            log.info("hyts_std: %d of %d receptor times run", i, len(stamps))

    if uncovered:
        spans = [sp for sp in map(_gdas1_file_range, mets) if sp is not None]
        log.warning(
            "%d of %d receptor times skipped: no met file covers any of their %d-h window "
            "(e.g. %s); the met files span %s → %s.",
            len(uncovered),
            len(stamps),
            abs(hours_back),
            uncovered[0],
            min(sp[0] for sp in spans),
            max(sp[1] for sp in spans),
        )
    if stopped:
        log.warning("Stopped after %d of %d receptor times.", len(tdumps), len(stamps))
        if not tdumps:
            return pd.DataFrame()
    if not tdumps:
        raise RuntimeError(
            "No trajectories produced. "
            + (
                "No met file covers the receptor times' backward windows."
                if uncovered and len(uncovered) == len(stamps)
                else "Check met coverage of the backward window, the CONTROL settings, "
                "and the hyts_std path."
            )
        )
    log.info("Ran %d back-trajectories -> %s", len(tdumps), work)
    feats = build_trajectory_features(
        tdumps, source_regions=source_regions, prefix=prefix, min_hours=min_hours
    )
    age_col = f"{prefix}age_max_h"
    if min_hours is None and age_col in feats:
        n_short = int((feats[age_col] < abs(hours_back) - 1e-6).sum())
        if n_short:
            log.warning(
                "%d of %d trajectories stopped short of hours_back=%d (met coverage or "
                "domain edge); see %s, or pass min_hours to null them.",
                n_short,
                len(feats),
                abs(hours_back),
                age_col,
            )
    return feats
