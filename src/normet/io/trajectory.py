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
... )                                                        # doctest: +SKIP
>>> df = df.join(feats).ffill(limit=8)                        # doctest: +SKIP
>>> # then pass the ``traj_*`` columns to do_all, also in variables_resample
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from glob import glob
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "read_trajectory_tdump",
    "trajectory_features",
    "build_trajectory_features",
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


def _region_mask(region: Any, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Point-in-region test — a 4-tuple bbox, or a shapely geometry (polygon
    boundaries loaded from GeoJSON via :func:`load_source_regions`)."""
    if (
        isinstance(region, (tuple, list))
        and len(region) == 4
        and all(isinstance(v, (int, float)) for v in region)
    ):
        xmn, ymn, xmx, ymx = region
        return (lon >= xmn) & (lon <= xmx) & (lat >= ymn) & (lat <= ymx)
    from shapely import contains_xy

    return contains_xy(region, lon, lat)


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
    prefix : str, default ``"traj_"``
        Prefix for every feature name.

    Returns
    -------
    dict
        Transport descriptors: straight-line reach, path length, mean transport
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
    return f


def build_trajectory_features(
    tdumps: str | Iterable[str | Path],
    *,
    source_regions: Mapping[str, tuple[float, float, float, float]] | None = None,
    prefix: str = "traj_",
    date_col: str = "date",
) -> pd.DataFrame:
    """Build a receptor-time feature table from many HYSPLIT ``tdump`` files.

    Parameters
    ----------
    tdumps : str or iterable of paths
        A glob pattern (e.g. ``"traj/tdump_*"``) or an explicit iterable of
        ``tdump`` file paths. One back-trajectory run per receptor time is the
        typical layout; files holding multiple trajectories are split per
        ``traj`` index.
    source_regions, prefix
        Forwarded to :func:`trajectory_features`.
    date_col : str, default ``"date"``
        Name of the index column (receptor timestamp), so the result joins
        straight onto a date-indexed panel.

    Returns
    -------
    pandas.DataFrame
        Indexed by receptor timestamp, one ``{prefix}*`` column per feature.
        Sorted by time; deduplicated on the receptor timestamp (last wins).
    """
    paths = sorted(glob(tdumps)) if isinstance(tdumps, str) else [str(p) for p in tdumps]
    if not paths:
        raise ValueError(f"No tdump files matched: {tdumps!r}")

    rows: list[dict[str, Any]] = []
    for p in paths:
        try:
            traj = read_trajectory_tdump(p)
        except Exception as exc:  # skip unparseable files but say so
            log.warning("Skipping trajectory file %s: %s", p, exc)
            continue
        for _, g in traj.groupby("traj"):
            receptor = g.loc[g["age_h"].abs().idxmin(), "datetime"]
            feats = trajectory_features(g, source_regions=source_regions, prefix=prefix)
            rows.append({date_col: pd.Timestamp(cast(Any, receptor)), **feats})

    if not rows:
        return pd.DataFrame()

    out = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=date_col, keep="last")
        .set_index(date_col)
        .sort_index()
    )
    log.info("Built trajectory features: %d receptors × %d columns", len(out), out.shape[1])
    return out


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
        ts.strftime("%y %m %d %H"),  # start: YY MM DD HH (2-digit year)
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
    timeout: int = 600,
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
    lat, lon : float
        Receptor location (degrees).
    met_files : str or iterable of str
        ARL-format meteorology file(s). They must collectively cover the full
        backward window (``hours_back`` before each receptor time), or the
        trajectory truncates where the data runs out.
    hysplit_exec : str or Path
        Path to the ``hyts_std`` executable (e.g. ``~/hysplit-5.4.2/exec/hyts_std``).
    height_m : float, default 500.0
        Receptor start height (m AGL).
    hours_back : int, default 72
        Backward duration in hours (sign ignored; always run backward).
    work_dir : str or Path, optional
        Directory for ``CONTROL`` and ``tdump_*`` files. A temp dir is created
        if omitted; the tdumps are left there for inspection/reuse.
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
    source_regions, prefix
        Forwarded to :func:`build_trajectory_features`.
    timeout : int, default 600
        Per-run timeout (seconds) for ``hyts_std``.

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

    tdumps: list[str] = []
    for t in times:
        ts = pd.Timestamp(t)
        name = "tdump_" + ts.strftime("%Y%m%d%H")
        (work / "CONTROL").write_text(
            _control_text(
                ts,
                lat,
                lon,
                height_m,
                hours_back,
                mets,
                name,
                top_of_model=top_of_model,
                vert_motion=vert_motion,
            )
        )
        try:
            r = subprocess.run(
                [exe],
                cwd=work,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            log.warning("hyts_std timed out (%ds) for receptor %s", timeout, ts)
            continue

        if r.returncode == 137:  # SIGKILL — almost always macOS Gatekeeper quarantine
            raise RuntimeError(
                "hyts_std was killed (exit 137) — likely macOS Gatekeeper quarantine. "
                f"Clear it once with:\n  xattr -dr com.apple.quarantine {Path(exe).parents[1]}"
            )
        out = work / name
        if not out.exists():
            msg = (r.stderr or r.stdout or "").strip().replace("\n", " ")[:200]
            log.warning("No tdump for %s (hyts_std rc=%s): %s", ts, r.returncode, msg)
            continue
        tdumps.append(str(out))

    if not tdumps:
        raise RuntimeError(
            "No trajectories produced. Check met coverage of the backward window, "
            "the CONTROL settings, and the hyts_std path."
        )
    log.info("Ran %d back-trajectories -> %s", len(tdumps), work)
    return build_trajectory_features(tdumps, source_regions=source_regions, prefix=prefix)
