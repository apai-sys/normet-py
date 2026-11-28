# src/normet/analysis/transport.py
"""Layer-2 Lagrangian transport analysis: receptor grids, footprint inversion,
trajectory clustering, and SCM spillover screening.

Builds on :mod:`normet.io.trajectory`'s physical endpoint weights (BLH coupling,
inverse mixing-depth sensitivity, exponential decay, and causal wet scavenging)
so every method below natively supports both raw observations and de-weathered
series (e.g. from ``normalise`` with local meteorology only).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..core.exceptions import ConfigError
from ..io.trajectory import (
    _endpoint_weights_flat_frame,
    _haversine_km,
    _naive_utc,
    _region_mask,
    is_trajectory_qc_column,
)
from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "ReceptorGridResult",
    "FootprintInversionResult",
    "MultisiteInversionResult",
    "TrajectoryClusterResult",
    "gridded_receptor_model",
    "wpscf",
    "wcwt",
    "build_footprint_matrix",
    "invert_source_footprints",
    "invert_multisite_footprints",
    "cluster_trajectories",
    "screen_donor_transport",
]

_KM_PER_DEG_LAT = 111.195
_HORIZON_SUFFIX_RE = re.compile(r"_\d+(?:\.\d+)?_\d+(?:\.\d+)?h$")


@dataclass
class ReceptorGridResult:
    """Output of gridded trajectory receptor models (:func:`wpscf`, :func:`wcwt`)."""

    method: str
    lon_edges: np.ndarray
    lat_edges: np.ndarray
    lon_centers: np.ndarray
    lat_centers: np.ndarray
    field: np.ndarray
    weighted: np.ndarray
    weight_matrix: np.ndarray
    exposure: np.ndarray
    count: np.ndarray
    receptor_lat: float
    receptor_lon: float
    threshold: float | None
    n_matched_trajectories: int

    def to_dataframe(self) -> pd.DataFrame:
        """Return non-empty grid cells as a long DataFrame sorted by weighted intensity."""
        rows: list[dict[str, Any]] = []
        valid_i, valid_j = np.nonzero(self.count > 0)
        for i, j in zip(valid_i, valid_j, strict=True):
            rows.append(
                {
                    "lat": float(self.lat_centers[i]),
                    "lon": float(self.lon_centers[j]),
                    "value": float(self.field[i, j]),
                    "weighted": float(self.weighted[i, j]),
                    "weight": float(self.weight_matrix[i, j]),
                    "exposure": float(self.exposure[i, j]),
                    "count": int(self.count[i, j]),
                }
            )
        cols = ["lat", "lon", "value", "weighted", "weight", "exposure", "count"]
        if not rows:
            return pd.DataFrame(columns=cols)
        return (
            pd.DataFrame(rows, columns=cols)
            .sort_values(["weighted", "exposure"], ascending=[False, False], na_position="last")
            .reset_index(drop=True)
        )


@dataclass
class FootprintInversionResult:
    """Result of regularized source-region footprint inversion (:func:`invert_source_footprints`)."""

    intercept: float
    regions: pd.DataFrame
    contributions: pd.DataFrame
    alpha: float
    r2: float
    rmse: float
    n_obs: int
    rolling_beta: pd.DataFrame | None = None


@dataclass
class MultisiteInversionResult:
    """Result of multi-receptor joint footprint inversion (:func:`invert_multisite_footprints`)."""

    intercepts: dict[str, float]
    regions: pd.DataFrame
    site_metrics: pd.DataFrame
    contributions: pd.DataFrame
    alpha: float
    spatial_alpha: float
    r2: float
    rmse: float
    n_obs: int


@dataclass
class TrajectoryClusterResult:
    """Result of hierarchical back-trajectory clustering (:func:`cluster_trajectories`)."""

    labels: pd.Series
    centroids: pd.DataFrame
    summary: pd.DataFrame
    tsv: pd.DataFrame
    n_clusters: int


def _resolve_series_frame(
    series: pd.Series | pd.DataFrame,
    *,
    value_col: str | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Normalise a target Series/DataFrame into tidy ``[_t, _y]`` sorted by UTC time.

    When *value_col* is not given, a DataFrame's de-weathered estimate is
    preferred over its raw observations: ``normalised`` (from
    :func:`~normet.analysis.normalise`), then ``emi_total`` (from
    :func:`~normet.analysis.decom_met`), then ``value``, and only then
    ``observed``. A frame that carries both columns (the usual output of both
    functions) otherwise mapped raw concentrations while the caller believed
    it was mapping the de-weathered signal.
    """
    if isinstance(series, pd.Series):
        if isinstance(series.index, pd.DatetimeIndex):
            times = _naive_utc(series.index)
        else:
            raise TypeError("Target Series must have a DatetimeIndex.")
        vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    elif isinstance(series, pd.DataFrame):
        if date_col in series.columns:
            times = _naive_utc(series[date_col])
        elif isinstance(series.index, pd.DatetimeIndex):
            times = _naive_utc(series.index)
        else:
            raise KeyError(f"series has no {date_col!r} column and no DatetimeIndex.")
        col = value_col
        if col is None:
            for cand in ("normalised", "emi_total", "value", "observed"):
                if cand in series.columns and pd.api.types.is_numeric_dtype(series[cand]):
                    col = cand
                    break
        if col is None:
            num_cols = [
                c
                for c in series.columns
                if c != date_col and pd.api.types.is_numeric_dtype(series[c])
            ]
            if len(num_cols) == 1:
                col = num_cols[0]
            else:
                raise ValueError(
                    "Could not infer `value_col` uniquely; pass it explicitly "
                    f"(numeric columns: {num_cols[:8]})."
                )
        if col not in series.columns:
            raise KeyError(f"Column {col!r} not found in series.")
        vals = pd.to_numeric(series[col], errors="coerce").to_numpy(dtype=float)
    else:
        raise TypeError(f"series must be a Series or DataFrame, got {type(series).__name__}.")

    df = pd.DataFrame({"_t": times.to_numpy(), "_y": vals})
    df = df[df["_t"].notna() & np.isfinite(df["_y"])]
    if df.empty:
        raise ValueError("No finite dated observations in `series`.")
    return df.groupby("_t", as_index=False)["_y"].mean().sort_values("_t").reset_index(drop=True)


def _normalise_endpoints_frame(endpoints: pd.DataFrame) -> pd.DataFrame:
    """Validate an endpoints DataFrame and attach canonical grouping columns."""
    if not isinstance(endpoints, pd.DataFrame) or endpoints.empty:
        raise ValueError("`endpoints` must be a non-empty DataFrame.")
    work = endpoints.copy()
    if "receptor_date" not in work.columns:
        if "date" in work.columns:
            work["receptor_date"] = work["date"]
        else:
            raise KeyError("`endpoints` must contain 'receptor_date' (or 'date').")
    for req in ("age_h", "lat", "lon", "height"):
        if req not in work.columns:
            raise KeyError(f"`endpoints` missing required column {req!r}.")
    if "start_height" not in work.columns:
        work["start_height"] = 0.0
    if "traj_id" not in work.columns:
        work["traj_id"] = 1
    work["_t"] = _naive_utc(work["receptor_date"]).to_numpy()
    work = work[work["_t"].notna()].copy()
    if work.empty:
        raise ValueError("`endpoints` contains no valid receptor timestamps.")
    work["start_height"] = pd.to_numeric(work["start_height"], errors="coerce").fillna(0.0)
    work["traj_id"] = pd.to_numeric(work["traj_id"], errors="coerce").fillna(1).astype(int)
    return work


def _trajectory_codes(frame: pd.DataFrame) -> np.ndarray:
    """First-appearance integer code per row, one code per trajectory.

    The values are irrelevant to endpoint weights (only the equality
    classes are), and ``ngroup`` gives a cheap int64 array for the flat
    kernel without building a per-group DataFrame. The three keys must be
    non-null: ``_normalise_endpoints_frame`` fills ``start_height`` and
    ``traj_id`` and drops NaT ``_t``, and pandas encodes a null key as NaN,
    which would otherwise cast to code 0 and contaminate the first
    trajectory (the per-group loops skip such rows instead).
    """
    codes = frame.groupby(["_t", "start_height", "traj_id"], sort=False).ngroup()
    if codes.isna().any():
        raise ValueError("trajectory grouping keys must be non-null")
    return codes.to_numpy(dtype=np.int64)


def _match_timestamps(
    target_times: pd.Series,
    ref_df: pd.DataFrame,
    *,
    tolerance: str | pd.Timedelta | None = None,
) -> np.ndarray:
    """Match UTC timestamps in *target_times* to ``ref_df['_t', '_y']`` (NaN when unmatched)."""
    tol = pd.Timedelta(0) if tolerance is None else pd.Timedelta(tolerance)
    if tol < pd.Timedelta(0):
        raise ValueError(f"tolerance must be non-negative, got {tolerance!r}")
    left = pd.DataFrame({"_t": target_times.to_numpy(), "_pos": np.arange(len(target_times))})
    matched = pd.merge_asof(
        left.sort_values("_t", kind="stable"),
        ref_df[["_t", "_y"]].sort_values("_t", kind="stable"),
        on="_t",
        direction="nearest",
        tolerance=tol,
    )
    out = np.full(len(target_times), np.nan, dtype=float)
    out[matched["_pos"].to_numpy(dtype=int)] = matched["_y"].to_numpy(dtype=float)
    return out


def _polissar_weights(count: np.ndarray) -> np.ndarray:
    """Polissar/PSCF confidence weights from unique-trajectory counts per cell.

    Published tiers are 1.0, 0.7, 0.42 and 0.05, at *strictly* more than
    3x, 1.5x and 1x the mean non-empty count (Polissar et al., 1999; PSCF
    literature). Empty cells get NaN. Earlier code used 0.4 for the third
    tier and 0.2 for the fourth with non-strict boundaries, which inflated
    the sparsest cells fourfold -- exactly the cells the scheme exists to
    down-weight.
    """
    non_empty = count >= 1
    if not non_empty.any():
        return np.full(count.shape, np.nan, dtype=float)
    mean_n = float(count[non_empty].mean())
    return np.where(
        count > 3.0 * mean_n,
        1.0,
        np.where(
            count > 1.5 * mean_n,
            0.7,
            np.where(count > mean_n, 0.42, np.where(count >= 1, 0.05, np.nan)),
        ),
    )


def gridded_receptor_model(
    endpoints: pd.DataFrame,
    series: pd.Series | pd.DataFrame,
    *,
    method: str = "wcwt",
    value_col: str | None = None,
    date_col: str = "date",
    tolerance: str | pd.Timedelta | None = None,
    grid_res: float = 0.5,
    lon_edges: Sequence[float] | np.ndarray | None = None,
    lat_edges: Sequence[float] | np.ndarray | None = None,
    pad_deg: float = 0.5,
    threshold: float | None = None,
    threshold_quantile: float = 0.75,
    blh_coupled: bool = True,
    blh_factor: float = 1.0,
    min_blh_m: float = 100.0,
    fallback_height_m: float | None = 1500.0,
    decay_hours: float | None = None,
    scavenging_coeff: float | None = None,
    exclude_receptor: bool = True,
    min_dist_km: float = 0.0,
    min_count: int = 1,
    weighting: str = "polissar",
) -> ReceptorGridResult:
    """Physical BLH-weighted WPSCF / WCWT gridded trajectory receptor model.

    Maps either raw observations or *de-weathered* local-control concentrations
    onto geographic grid cells along back-trajectories weighted by
    :func:`~normet.io.trajectory_endpoint_weights`, applying Polissar
    confidence down-weighting based on **unique trajectories** visiting each
    cell (preventing slow stagnant loops from inflating cell confidence).

    Parameters
    ----------
    endpoints : pandas.DataFrame
        Output of :func:`~normet.io.read_trajectory_endpoints`.
    series : pandas.Series or pandas.DataFrame
        Target (or de-weathered) series indexed by datetime or carrying
        ``date_col`` + ``value_col``. Without ``value_col``, a frame holding
        both ``normalised``/``emi_total`` and ``observed`` uses the
        de-weathered estimate (see :func:`_resolve_series_frame`).
    method : {"wcwt", "wpscf"}, default "wcwt"
        ``"wcwt"`` computes Concentration Weighted Trajectory field;
        ``"wpscf"`` computes Potential Source Contribution Function field above
        ``threshold`` (or ``threshold_quantile``).
    grid_res : float, default 0.5
        Grid spacing (degrees) when ``lon_edges``/``lat_edges`` are omitted.
    blh_coupled, blh_factor, min_blh_m, fallback_height_m, decay_hours, scavenging_coeff, exclude_receptor :
        Physical exposure parameters forwarded to
        :func:`~normet.io.trajectory_endpoint_weights`.
    min_dist_km : float, default 0.0
        Exclude endpoints closer than ``min_dist_km`` to the receptor to
        suppress near-source trailing artefacts.
    min_count : int, default 1
        Minimum distinct trajectories visiting a cell to retain ``weighted``.
    weighting : {"polissar", "none"}, default "polissar"
        Polissar confidence step-weight using unique trajectory counts per
        cell: 1.0 above 3x the mean count, then 0.7, 0.42 and 0.05, with
        strict ``>`` boundaries.

    Returns
    -------
    ReceptorGridResult
    """
    meth = method.lower()
    if meth in ("pscf", "wpscf"):
        meth = "wpscf"
    elif meth in ("cwt", "wcwt"):
        meth = "wcwt"
    else:
        raise ValueError(f"method must be 'wcwt' or 'wpscf', got {method!r}")
    if weighting not in ("polissar", "none"):
        raise ValueError(f"weighting must be 'polissar' or 'none', got {weighting!r}")
    if float(grid_res) <= 0.0:
        raise ValueError(f"grid_res must be positive, got {grid_res}")
    if not (0.0 < float(threshold_quantile) < 1.0):
        raise ValueError(f"threshold_quantile must lie in (0, 1), got {threshold_quantile}")

    ep = _normalise_endpoints_frame(endpoints)
    obs_df = _resolve_series_frame(series, value_col=value_col, date_col=date_col)

    grp_cols = ["_t", "start_height", "traj_id"]
    keys_df = ep[grp_cols].drop_duplicates().sort_values(grp_cols).reset_index(drop=True)
    keys_df["_y"] = _match_timestamps(keys_df["_t"], obs_df, tolerance=tolerance)
    keys_df = keys_df[np.isfinite(keys_df["_y"])].reset_index(drop=True)
    if keys_df.empty:
        raise ValueError("No trajectory receptor timestamps matched `series` within `tolerance`.")
    keys_df["_traj_idx"] = np.arange(len(keys_df), dtype=np.int64)

    ep_m = ep.merge(keys_df, on=grp_cols, how="inner")
    matched_y = keys_df["_y"].to_numpy(dtype=float)
    thr: float | None = None
    if meth == "wpscf":
        thr = (
            float(threshold)
            if threshold is not None
            else float(np.nanquantile(matched_y, float(threshold_quantile)))
        )

    anchor_i = int(
        np.nanargmin(np.abs(pd.to_numeric(ep_m["age_h"], errors="coerce").to_numpy(dtype=float)))
    )
    rec_lat = (
        float(ep_m["start_lat"].median())
        if "start_lat" in ep_m.columns
        else float(pd.to_numeric(ep_m["lat"], errors="coerce").to_numpy(dtype=float)[anchor_i])
    )
    rec_lon = (
        float(ep_m["start_lon"].median())
        if "start_lon" in ep_m.columns
        else float(pd.to_numeric(ep_m["lon"], errors="coerce").to_numpy(dtype=float)[anchor_i])
    )

    lon_ep = pd.to_numeric(ep_m["lon"], errors="coerce").to_numpy(dtype=float)
    lat_ep = pd.to_numeric(ep_m["lat"], errors="coerce").to_numpy(dtype=float)
    fin = np.isfinite(lon_ep) & np.isfinite(lat_ep)
    if not fin.any():
        raise ValueError("No finite trajectory coordinates available.")

    res_deg = float(grid_res)
    x_edges: np.ndarray[Any, Any]
    if lon_edges is None:
        lo_x = np.floor((float(np.min(lon_ep[fin])) - float(pad_deg)) / res_deg) * res_deg
        hi_x = np.ceil((float(np.max(lon_ep[fin])) + float(pad_deg)) / res_deg) * res_deg
        x_edges = np.arange(lo_x, hi_x + 0.5 * res_deg, res_deg, dtype=float)
        if x_edges.size < 2:
            x_edges = np.array([lo_x, lo_x + res_deg], dtype=float)
    else:
        x_edges = np.asarray(lon_edges, dtype=float)
    y_edges: np.ndarray[Any, Any]
    if lat_edges is None:
        lo_y = max(
            -90.0, np.floor((float(np.min(lat_ep[fin])) - float(pad_deg)) / res_deg) * res_deg
        )
        hi_y = min(90.0, np.ceil((float(np.max(lat_ep[fin])) + float(pad_deg)) / res_deg) * res_deg)
        y_edges = np.arange(lo_y, hi_y + 0.5 * res_deg, res_deg, dtype=float)
        if y_edges.size < 2:
            y_edges = np.array([lo_y, lo_y + res_deg], dtype=float)
    else:
        y_edges = np.asarray(lat_edges, dtype=float)

    n_lon = len(x_edges) - 1
    n_lat = len(y_edges) - 1
    if n_lon < 1 or n_lat < 1:
        raise ValueError("lon_edges and lat_edges must each have at least 2 boundary points.")

    # One flat pass over the matched table instead of a pandas extraction
    # per trajectory (the extraction was ~2/3 of the weight loop's cost).
    # ``merge`` returns a positional index and ``_traj_idx`` is already the
    # trajectory code, so the result aligns with ep_m's rows.
    ep_m = ep_m.reset_index(drop=True)
    w_all: np.ndarray[Any, Any] = _endpoint_weights_flat_frame(
        ep_m,
        ep_m["_traj_idx"].to_numpy(dtype=np.int64),
        blh_coupled=blh_coupled,
        blh_factor=blh_factor,
        min_blh_m=min_blh_m,
        fallback_height_m=fallback_height_m,
        decay_hours=decay_hours,
        scavenging_coeff=scavenging_coeff,
        exclude_receptor=exclude_receptor,
    )

    if float(min_dist_km) > 0.0:
        d_rec = np.asarray(_haversine_km(rec_lat, rec_lon, lat_ep, lon_ep), dtype=float)
        w_all = np.where(d_rec >= float(min_dist_km), w_all, 0.0)

    li = np.searchsorted(y_edges, lat_ep, side="right") - 1
    lj = np.searchsorted(x_edges, lon_ep, side="right") - 1
    keep = fin & (w_all > 0.0) & (li >= 0) & (li < n_lat) & (lj >= 0) & (lj < n_lon)

    exposure = np.zeros((n_lat, n_lon), dtype=float)
    num = np.zeros((n_lat, n_lon), dtype=float)
    count = np.zeros((n_lat, n_lon), dtype=np.int64)

    if keep.any():
        cell_flat = li[keep].astype(np.int64) * n_lon + lj[keep].astype(np.int64)
        w_k = w_all[keep]
        y_k = ep_m["_y"].to_numpy(dtype=float)[keep]
        t_k = ep_m["_traj_idx"].to_numpy(dtype=np.int64)[keep]

        np.add.at(exposure.ravel(), cell_flat, w_k)
        if meth == "wpscf":
            assert thr is not None
            hi_mask = y_k > thr
            if hi_mask.any():
                np.add.at(num.ravel(), cell_flat[hi_mask], w_k[hi_mask])
        else:
            np.add.at(num.ravel(), cell_flat, w_k * y_k)

        n_cells = np.int64(n_lat * n_lon)
        pair_codes = np.unique(t_k * n_cells + cell_flat)
        np.add.at(count.ravel(), pair_codes % n_cells, 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        field = np.where(exposure > 0.0, num / exposure, np.nan)

    non_empty = count >= 1
    if weighting == "polissar" and non_empty.any():
        w_mat = _polissar_weights(count)
    else:
        w_mat = np.where(non_empty, 1.0, np.nan)

    weighted = np.where(count >= int(min_count), field * w_mat, np.nan)
    return ReceptorGridResult(
        method=meth,
        lon_edges=x_edges,
        lat_edges=y_edges,
        lon_centers=0.5 * (x_edges[:-1] + x_edges[1:]),
        lat_centers=0.5 * (y_edges[:-1] + y_edges[1:]),
        field=field,
        weighted=weighted,
        weight_matrix=w_mat,
        exposure=exposure,
        count=count,
        receptor_lat=rec_lat,
        receptor_lon=rec_lon,
        threshold=thr,
        n_matched_trajectories=len(keys_df),
    )


def wpscf(
    endpoints: pd.DataFrame, series: pd.Series | pd.DataFrame, **kwargs: Any
) -> ReceptorGridResult:
    """Weighted Potential Source Contribution Function (:func:`gridded_receptor_model`)."""
    kwargs.pop("method", None)
    return gridded_receptor_model(endpoints, series, method="wpscf", **kwargs)


def wcwt(
    endpoints: pd.DataFrame, series: pd.Series | pd.DataFrame, **kwargs: Any
) -> ReceptorGridResult:
    """Weighted Concentration Weighted Trajectory field (:func:`gridded_receptor_model`)."""
    kwargs.pop("method", None)
    return gridded_receptor_model(endpoints, series, method="wcwt", **kwargs)


def build_footprint_matrix(
    endpoints: pd.DataFrame,
    source_regions: Mapping[str, tuple[float, float, float, float] | Any],
    *,
    blh_coupled: bool = True,
    blh_factor: float = 1.0,
    min_blh_m: float = 100.0,
    fallback_height_m: float | None = 1500.0,
    decay_hours: float | None = None,
    scavenging_coeff: float | None = None,
    exclude_receptor: bool = True,
) -> pd.DataFrame:
    """Aggregate per-endpoint physical weights into a ``(time × region)`` footprint matrix.

    Multiple release heights at the same ``receptor_date`` are averaged into an
    ensemble surface footprint per region.
    """
    if not source_regions:
        raise ValueError("`source_regions` must be a non-empty mapping.")
    ep = _normalise_endpoints_frame(endpoints).reset_index(drop=True)
    reg_names = [str(k) for k in source_regions]

    # Weight and convert every endpoint once; the per-group loop below then
    # only slices, so each region's ``np.sum`` keeps its element order.
    w_all = _endpoint_weights_flat_frame(
        ep,
        _trajectory_codes(ep),
        blh_coupled=blh_coupled,
        blh_factor=blh_factor,
        min_blh_m=min_blh_m,
        fallback_height_m=fallback_height_m,
        decay_hours=decay_hours,
        scavenging_coeff=scavenging_coeff,
        exclude_receptor=exclude_receptor,
    )
    lon_all = pd.to_numeric(ep["lon"], errors="coerce").to_numpy(dtype=float)
    lat_all = pd.to_numeric(ep["lat"], errors="coerce").to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
    for (t_stamp, start_h, _tid), g in ep.groupby(["_t", "start_height", "traj_id"], sort=True):
        pos = g.index.to_numpy()
        w = w_all[pos]
        lon = lon_all[pos]
        lat = lat_all[pos]
        rec: dict[str, Any] = {"date": t_stamp, "_h": start_h}
        for name, region in source_regions.items():
            # Per-trajectory masks: a table-wide mask would cost
            # n_rows x n_regions bytes for no correctness gain.
            inside = np.asarray(_region_mask(region, lon, lat), dtype=bool)
            rec[str(name)] = float(w[inside].sum())
        rows.append(rec)

    df = pd.DataFrame(rows)
    out = df.groupby("date", as_index=True)[reg_names].mean().sort_index()
    out.index = pd.DatetimeIndex(out.index, name="date")
    return out


def _solve_centered_ridge(
    F: np.ndarray, y: np.ndarray, alpha: float, non_negative: bool
) -> tuple[float, np.ndarray]:
    """Fit ``y ~ beta0 + F @ beta`` with unpenalized intercept ``beta0``."""
    f_mean = F.mean(axis=0)
    y_mean = float(y.mean())
    Fc = F - f_mean
    yc = y - y_mean
    p = F.shape[1]
    if non_negative:
        if alpha > 0.0:
            A = np.vstack([Fc, np.sqrt(float(alpha)) * np.eye(p)])
            b = np.concatenate([yc, np.zeros(p)])
        else:
            A, b = Fc, yc
        from scipy import optimize as sp_opt

        beta, _ = sp_opt.nnls(A, b)
    else:
        gram = Fc.T @ Fc + (float(alpha) + 1e-12) * np.eye(p)
        rhs = Fc.T @ yc
        try:
            beta = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            beta, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
    beta0 = y_mean - float(f_mean @ beta)
    return beta0, np.asarray(beta, dtype=float)


def _select_alpha_blocked_cv(
    F: np.ndarray, y: np.ndarray, alphas: np.ndarray, non_negative: bool
) -> float:
    """Pick ``alpha`` minimizing MSE across contiguous time-ordered CV folds."""
    n, p = F.shape
    n_folds = min(5, max(2, n // max(8, p + 2)))
    if n < 2 * n_folds:
        return float(alphas[len(alphas) // 2])
    splits = np.array_split(np.arange(n), n_folds)
    best_alpha = float(alphas[0])
    best_mse = np.inf
    for a in alphas:
        sq_errs: list[np.ndarray] = []
        for val_idx in splits:
            if val_idx.size == 0:
                continue
            mask = np.ones(n, dtype=bool)
            mask[val_idx] = False
            if mask.sum() <= p:
                continue
            b0, b = _solve_centered_ridge(F[mask], y[mask], float(a), non_negative)
            pred = b0 + F[val_idx] @ b
            sq_errs.append((y[val_idx] - pred) ** 2)
        if not sq_errs:
            continue
        mse = float(np.mean(np.concatenate(sq_errs)))
        if mse < best_mse:
            best_mse = mse
            best_alpha = float(a)
    return best_alpha


def invert_source_footprints(
    footprints: pd.DataFrame,
    series: pd.Series | pd.DataFrame,
    *,
    value_col: str | None = None,
    date_col: str = "date",
    region_cols: Sequence[str] | None = None,
    alpha: float | Sequence[float] | str = "auto",
    non_negative: bool = True,
    n_bootstrap: int = 200,
    block_length: int | None = None,
    seed: int = 7_654_321,
    window_days: float | None = None,
    step_days: float | None = None,
    min_window_obs: int = 20,
) -> FootprintInversionResult:
    """Invert source-region unit intensities and contributions from physical footprints.

    Solves :math:`y(t) \\approx \\beta_0 + \\sum_g \\beta_g F_g(t)` (with
    non-negative constraint :math:`\\beta_g \\ge 0` by default via
    :func:`scipy.optimize.nnls` and contiguous time-blocked CV selection for
    ridge penalty ``alpha``). Uncertainties are computed via moving-block
    bootstrap preserving temporal autocorrelation. When ``window_days`` is set,
    also estimates sliding-window :math:`\\beta_g(t)` to track changes in upwind
    source intensity over time.
    """
    if not isinstance(footprints, pd.DataFrame) or footprints.empty:
        raise ValueError("`footprints` must be a non-empty DataFrame.")

    fp = footprints.copy()
    if date_col in fp.columns:
        fp_times = _naive_utc(fp[date_col])
        fp = fp.drop(columns=[date_col])
    elif isinstance(fp.index, pd.DatetimeIndex):
        fp_times = _naive_utc(fp.index)
    else:
        raise KeyError(f"`footprints` has no {date_col!r} column and no DatetimeIndex.")
    fp.index = pd.DatetimeIndex(fp_times.to_numpy(), name="date")
    keep = np.asarray(fp_times.notna().to_numpy(), dtype=bool)
    fp = fp.loc[keep].sort_index()

    if region_cols is not None:
        r_cols = [str(c) for c in region_cols]
        missing = [c for c in r_cols if c not in fp.columns]
        if missing:
            raise KeyError(f"region_cols absent from `footprints`: {missing}")
    else:
        fp_pref = [c for c in fp.columns if str(c).startswith("traj_footprint_")]
        if fp_pref:
            base_fp = [c for c in fp_pref if not _HORIZON_SUFFIX_RE.search(str(c))]
            r_cols = base_fp or fp_pref
        else:
            r_cols = [
                str(c)
                for c in fp.columns
                if pd.api.types.is_numeric_dtype(fp[c]) and not is_trajectory_qc_column(str(c))
            ]
    if not r_cols:
        raise ValueError("No region footprint columns found in `footprints`.")

    stripped = [
        c[len("traj_footprint_") :] if c.startswith("traj_footprint_") else c for c in r_cols
    ]
    display_names = stripped if len(set(stripped)) == len(stripped) else list(r_cols)

    obs_df = _resolve_series_frame(series, value_col=value_col, date_col=date_col)
    fp_num = fp[r_cols].apply(pd.to_numeric, errors="coerce")
    fp_grp = fp_num.groupby(level=0).mean()
    obs_s = pd.Series(obs_df["_y"].to_numpy(), index=pd.DatetimeIndex(obs_df["_t"]))
    joined = fp_grp.join(obs_s.rename("_y"), how="inner").dropna()
    if len(joined) < max(3, len(r_cols) + 1):
        raise ValueError(
            f"Need at least {max(3, len(r_cols) + 1)} finite aligned rows for inversion, "
            f"got {len(joined)}."
        )

    F = joined[r_cols].to_numpy(dtype=float)
    y = joined["_y"].to_numpy(dtype=float)
    n, p = F.shape

    if isinstance(alpha, str):
        if alpha.lower() != "auto":
            raise ValueError(f"alpha string must be 'auto', got {alpha!r}")
        scale = max(float(np.mean(np.var(F, axis=0))), 1e-6)
        chosen_alpha = _select_alpha_blocked_cv(F, y, scale * np.logspace(-3, 3, 13), non_negative)
    elif isinstance(alpha, Sequence):
        cand = np.asarray([float(a) for a in alpha], dtype=float)
        if cand.size == 0 or np.any(cand < 0.0):
            raise ValueError("alpha sequence must contain non-negative floats.")
        chosen_alpha = _select_alpha_blocked_cv(F, y, cand, non_negative)
    else:
        chosen_alpha = float(alpha)
        if chosen_alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")

    beta0, beta = _solve_centered_ridge(F, y, chosen_alpha, non_negative)
    pred = beta0 + F @ beta
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid * resid)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - np.sum(resid * resid) / ss_tot) if ss_tot > 0.0 else float("nan")

    ci_lo = np.full(p, np.nan, dtype=float)
    ci_hi = np.full(p, np.nan, dtype=float)
    if int(n_bootstrap) > 0 and n >= 4:
        L = min(
            n, max(1, int(block_length) if block_length is not None else round(n ** (1.0 / 3.0)))
        )
        max_start = max(1, n - L + 1)
        n_draws = int(np.ceil(n / L))
        rng = np.random.default_rng(int(seed))
        boot_betas = np.empty((int(n_bootstrap), p), dtype=float)
        for b_i in range(int(n_bootstrap)):
            starts = rng.integers(0, max_start, size=n_draws)
            idx = np.concatenate([np.arange(s, s + L) for s in starts])[:n]
            _, b_hat = _solve_centered_ridge(F[idx], y[idx], chosen_alpha, non_negative)
            boot_betas[b_i] = b_hat
        ci_lo = np.percentile(boot_betas, 2.5, axis=0)
        ci_hi = np.percentile(boot_betas, 97.5, axis=0)

    mean_fp = F.mean(axis=0)
    mean_contrib = beta * mean_fp
    total_tr = float(np.sum(mean_contrib))
    mean_y = float(np.mean(y))
    reg_rows = []
    for j, name in enumerate(display_names):
        reg_rows.append(
            {
                "region": name,
                "beta": float(beta[j]),
                "beta_ci_low": float(ci_lo[j]),
                "beta_ci_high": float(ci_hi[j]),
                "mean_footprint": float(mean_fp[j]),
                "mean_contribution": float(mean_contrib[j]),
                "share_of_transport": float(mean_contrib[j] / total_tr)
                if total_tr != 0.0
                else np.nan,
                "share_of_total": float(mean_contrib[j] / mean_y) if mean_y != 0.0 else np.nan,
            }
        )
    regions_df = pd.DataFrame(reg_rows)

    contrib_data: dict[str, Any] = {
        "observed": y,
        "predicted": pred,
        "background": np.full(n, beta0, dtype=float),
    }
    comp_mat = F * beta[None, :]
    for j, name in enumerate(display_names):
        contrib_data[name] = comp_mat[:, j]
    contrib_data["transport_total"] = comp_mat.sum(axis=1)
    contrib_data["residual"] = resid
    contrib_df = pd.DataFrame(contrib_data, index=joined.index)

    rolling_df: pd.DataFrame | None = None
    if window_days is not None:
        win_td = pd.Timedelta(days=float(window_days))
        step_td = pd.Timedelta(
            days=float(step_days) if step_days is not None else max(1.0, float(window_days) / 4.0)
        )
        t_idx = joined.index
        t_cur = t_idx[0]
        t_last = t_idx[-1]
        win_rows: list[dict[str, Any]] = []
        while t_cur + win_td <= t_last + pd.Timedelta(seconds=1):
            w_end = t_cur + win_td
            w_mask = (t_idx >= t_cur) & (t_idx < w_end)
            if int(w_mask.sum()) >= int(min_window_obs) and np.max(np.std(F[w_mask], axis=0)) > 0.0:
                Fw, yw = F[w_mask], y[w_mask]
                b0_w, b_w = _solve_centered_ridge(Fw, yw, chosen_alpha, non_negative)
                pr_w = b0_w + Fw @ b_w
                ss_t = float(np.sum((yw - yw.mean()) ** 2))
                r2_w = float(1.0 - np.sum((yw - pr_w) ** 2) / ss_t) if ss_t > 0.0 else float("nan")
                row_w: dict[str, Any] = {
                    "date": t_cur + win_td / 2,
                    "n_obs": int(w_mask.sum()),
                    "intercept": float(b0_w),
                    "r2": r2_w,
                }
                for j, name in enumerate(display_names):
                    row_w[name] = float(b_w[j])
                win_rows.append(row_w)
            t_cur = t_cur + step_td
        if win_rows:
            rolling_df = pd.DataFrame(win_rows).set_index("date").sort_index()

    return FootprintInversionResult(
        intercept=float(beta0),
        regions=regions_df,
        contributions=contrib_df,
        alpha=float(chosen_alpha),
        r2=r2,
        rmse=rmse,
        n_obs=n,
        rolling_beta=rolling_df,
    )


def _solve_joint_centered(
    Fc: np.ndarray,
    yc: np.ndarray,
    alpha: float,
    non_negative: bool,
    spatial_reg: np.ndarray | None = None,
) -> np.ndarray:
    """Solve centered multi-receptor system ``Fc @ beta ~ yc`` with ridge + spatial penalty."""
    p = Fc.shape[1]
    if non_negative:
        blocks_A = [Fc]
        blocks_b = [yc]
        if alpha > 0.0:
            blocks_A.append(np.sqrt(float(alpha)) * np.eye(p))
            blocks_b.append(np.zeros(p))
        if spatial_reg is not None and spatial_reg.size > 0:
            blocks_A.append(spatial_reg)
            blocks_b.append(np.zeros(spatial_reg.shape[0]))
        A = np.vstack(blocks_A) if len(blocks_A) > 1 else blocks_A[0]
        b = np.concatenate(blocks_b) if len(blocks_b) > 1 else blocks_b[0]
        from scipy import optimize as sp_opt

        beta, _ = sp_opt.nnls(A, b)
        return np.asarray(beta, dtype=float)

    gram = Fc.T @ Fc + (float(alpha) + 1e-12) * np.eye(p)
    if spatial_reg is not None and spatial_reg.size > 0:
        gram = gram + spatial_reg.T @ spatial_reg
    rhs = Fc.T @ yc
    try:
        beta = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        beta, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
    return np.asarray(beta, dtype=float)


def _build_spatial_reg_factor(
    region_names: Sequence[str],
    display_names: Sequence[str],
    region_coords: Mapping[str, tuple[float, float]] | pd.DataFrame | None,
    spatial_alpha: float,
    spatial_scale_km: float,
) -> np.ndarray | None:
    """Return ``sqrt(spatial_alpha) * B`` where ``B^T B = L`` (Haversine graph Laplacian)."""
    if region_coords is None or float(spatial_alpha) <= 0.0:
        return None
    coord_map: dict[str, tuple[float, float]] = {}
    if isinstance(region_coords, pd.DataFrame):
        name_col = "region" if "region" in region_coords.columns else region_coords.columns[0]
        for _, row in region_coords.iterrows():
            coord_map[str(row[name_col])] = (float(row["lat"]), float(row["lon"]))
    else:
        for k, v in region_coords.items():
            coord_map[str(k)] = (float(v[0]), float(v[1]))

    p = len(region_names)
    coords: list[tuple[float, float] | None] = []
    for raw_n, disp_n in zip(region_names, display_names, strict=True):
        coords.append(coord_map.get(disp_n) or coord_map.get(raw_n))

    W = np.zeros((p, p), dtype=float)
    scale = max(float(spatial_scale_km), 1e-3)
    for i in range(p):
        ci = coords[i]
        if ci is None:
            continue
        for j in range(i + 1, p):
            cj = coords[j]
            if cj is None:
                continue
            d_km = float(_haversine_km(ci[0], ci[1], cj[0], cj[1]))
            w_ij = float(np.exp(-d_km / scale))
            W[i, j] = W[j, i] = w_ij
    if float(W.sum()) <= 0.0:
        return None
    L = np.diag(W.sum(axis=1)) - W
    eigvals, eigvecs = np.linalg.eigh(L)
    keep = eigvals > 1e-12
    if not np.any(keep):
        return None
    return np.sqrt(float(spatial_alpha) * eigvals[keep])[:, None] * eigvecs[:, keep].T


def invert_multisite_footprints(
    footprints_by_site: Mapping[str, pd.DataFrame],
    series_by_site: Mapping[str, pd.Series | pd.DataFrame],
    *,
    value_col: str | None = None,
    date_col: str = "date",
    region_cols: Sequence[str] | None = None,
    region_coords: Mapping[str, tuple[float, float]] | pd.DataFrame | None = None,
    alpha: float | Sequence[float] | str = "auto",
    spatial_alpha: float = 0.0,
    spatial_scale_km: float = 100.0,
    non_negative: bool = True,
    n_bootstrap: int = 200,
    block_length: int | None = None,
    seed: int = 7_654_321,
) -> MultisiteInversionResult:
    """Joint multi-receptor Bayesian/regularized source-footprint inversion.

    Estimates shared regional source intensities :math:`\\beta_g \\ge 0` across
    multiple receptor stations while allowing each station its own local
    background intercept :math:`\\beta_{0, s}`:

    .. math::
        y_s(t) \\approx \\beta_{0, s} + \\sum_g \\beta_g F_{s, g}(t)

    Optionally enforces spatial smoothness across adjacent source regions via
    a Haversine graph Laplacian penalty ``spatial_alpha * beta^T L beta``.
    """
    if not footprints_by_site or not series_by_site:
        raise ValueError("`footprints_by_site` and `series_by_site` must be non-empty.")
    if float(spatial_alpha) < 0.0:
        raise ValueError(f"`spatial_alpha` must be non-negative, got {spatial_alpha}")

    sites = [str(s) for s in footprints_by_site if str(s) in series_by_site]
    if not sites:
        raise ValueError(
            "No overlapping site keys between `footprints_by_site` and `series_by_site`."
        )

    r_cols: list[str] | None = [str(c) for c in region_cols] if region_cols is not None else None
    site_data: list[tuple[str, pd.DatetimeIndex, np.ndarray, np.ndarray]] = []

    for site in sites:
        fp_raw = footprints_by_site[site]
        if not isinstance(fp_raw, pd.DataFrame) or fp_raw.empty:
            continue
        fp = fp_raw.copy()
        if date_col in fp.columns:
            fp_times = _naive_utc(fp[date_col])
            fp = fp.drop(columns=[date_col])
        elif isinstance(fp.index, pd.DatetimeIndex):
            fp_times = _naive_utc(fp.index)
        else:
            raise KeyError(
                f"`footprints_by_site[{site!r}]` has no {date_col!r} column or DatetimeIndex."
            )
        fp.index = pd.DatetimeIndex(fp_times.to_numpy(), name="date")
        fp = fp.loc[np.asarray(fp_times.notna().to_numpy(), dtype=bool)].sort_index()

        if r_cols is None:
            fp_pref = [c for c in fp.columns if str(c).startswith("traj_footprint_")]
            if fp_pref:
                base_fp = [c for c in fp_pref if not _HORIZON_SUFFIX_RE.search(str(c))]
                r_cols = base_fp or fp_pref
            else:
                r_cols = [
                    str(c)
                    for c in fp.columns
                    if pd.api.types.is_numeric_dtype(fp[c]) and not is_trajectory_qc_column(str(c))
                ]
        missing_r = [c for c in r_cols if c not in fp.columns]
        if missing_r:
            raise KeyError(f"region_cols absent from site {site!r}: {missing_r}")

        obs_df = _resolve_series_frame(series_by_site[site], value_col=value_col, date_col=date_col)
        fp_num = fp[r_cols].apply(pd.to_numeric, errors="coerce").groupby(level=0).mean()
        obs_s = pd.Series(obs_df["_y"].to_numpy(), index=pd.DatetimeIndex(obs_df["_t"]))
        joined = fp_num.join(obs_s.rename("_y"), how="inner").dropna()
        if joined.empty:
            continue
        site_data.append(
            (
                site,
                pd.DatetimeIndex(joined.index),
                joined[r_cols].to_numpy(dtype=float),
                joined["_y"].to_numpy(dtype=float),
            )
        )

    if not site_data or r_cols is None or not r_cols:
        raise ValueError("No aligned multi-receptor observations available for inversion.")

    p = len(r_cols)
    stripped = [
        c[len("traj_footprint_") :] if c.startswith("traj_footprint_") else c for c in r_cols
    ]
    display_names = stripped if len(set(stripped)) == len(stripped) else list(r_cols)
    spatial_reg = _build_spatial_reg_factor(
        r_cols, display_names, region_coords, float(spatial_alpha), float(spatial_scale_km)
    )

    Fc_parts: list[np.ndarray] = []
    yc_parts: list[np.ndarray] = []
    t_parts: list[pd.DatetimeIndex] = []
    site_parts: list[np.ndarray] = []
    F_raw_parts: list[np.ndarray] = []
    y_raw_parts: list[np.ndarray] = []
    site_means: dict[str, tuple[np.ndarray, float]] = {}

    for site, idx_s, F_s, y_s in site_data:
        f_mu = F_s.mean(axis=0)
        y_mu = float(y_s.mean())
        site_means[site] = (f_mu, y_mu)
        Fc_parts.append(F_s - f_mu)
        yc_parts.append(y_s - y_mu)
        t_parts.append(idx_s)
        site_parts.append(np.full(len(y_s), site, dtype=object))
        F_raw_parts.append(F_s)
        y_raw_parts.append(y_s)

    Fc_all = np.vstack(Fc_parts)
    yc_all = np.concatenate(yc_parts)
    F_all = np.vstack(F_raw_parts)
    y_all = np.concatenate(y_raw_parts)
    t_all = np.concatenate([t.to_numpy() for t in t_parts])
    s_all = np.concatenate(site_parts)
    n_tot = len(y_all)
    if n_tot < max(3, p + 1):
        raise ValueError(f"Need at least {max(3, p + 1)} total aligned rows, got {n_tot}.")

    # Sort stacked observations chronologically so blocked CV holds out contiguous time windows.
    ord_t = np.argsort(t_all, kind="stable")
    Fc_ord = Fc_all[ord_t]
    yc_ord = yc_all[ord_t]

    if isinstance(alpha, str):
        if alpha.lower() != "auto":
            raise ValueError(f"alpha string must be 'auto', got {alpha!r}")
        scale = max(float(np.mean(np.var(Fc_all, axis=0))), 1e-6)
        cand_alphas = scale * np.logspace(-3, 3, 13)
    elif isinstance(alpha, Sequence):
        cand_alphas = np.asarray([float(a) for a in alpha], dtype=float)
        if cand_alphas.size == 0 or np.any(cand_alphas < 0.0):
            raise ValueError("alpha sequence must contain non-negative floats.")
    else:
        cand_alphas = np.array([float(alpha)], dtype=float)
        if cand_alphas[0] < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")

    if len(cand_alphas) == 1:
        chosen_alpha = float(cand_alphas[0])
    else:
        n_folds = min(5, max(2, n_tot // max(8, p + 2)))
        if n_tot < 2 * n_folds:
            chosen_alpha = float(cand_alphas[len(cand_alphas) // 2])
        else:
            splits = np.array_split(np.arange(n_tot), n_folds)
            chosen_alpha = float(cand_alphas[0])
            best_mse = np.inf
            for a in cand_alphas:
                sq_errs: list[np.ndarray] = []
                for val_idx in splits:
                    if val_idx.size == 0:
                        continue
                    tr_mask = np.ones(n_tot, dtype=bool)
                    tr_mask[val_idx] = False
                    if tr_mask.sum() <= p:
                        continue
                    b_cv = _solve_joint_centered(
                        Fc_ord[tr_mask], yc_ord[tr_mask], float(a), non_negative, spatial_reg
                    )
                    sq_errs.append((yc_ord[val_idx] - Fc_ord[val_idx] @ b_cv) ** 2)
                if sq_errs:
                    mse = float(np.mean(np.concatenate(sq_errs)))
                    if mse < best_mse:
                        best_mse = mse
                        chosen_alpha = float(a)

    beta = _solve_joint_centered(Fc_all, yc_all, chosen_alpha, non_negative, spatial_reg)
    intercepts = {
        site: float(y_mu - float(f_mu @ beta)) for site, (f_mu, y_mu) in site_means.items()
    }

    pred_all = np.empty(n_tot, dtype=float)
    b0_vec = np.empty(n_tot, dtype=float)
    site_metric_rows: list[dict[str, Any]] = []
    cursor = 0
    for site, _idx_s, F_s, y_s in site_data:
        m_s = len(y_s)
        b0_s = intercepts[site]
        p_s = b0_s + F_s @ beta
        r_s = y_s - p_s
        rmse_s = float(np.sqrt(np.mean(r_s * r_s)))
        ss_t_s = float(np.sum((y_s - y_s.mean()) ** 2))
        r2_s = float(1.0 - np.sum(r_s * r_s) / ss_t_s) if ss_t_s > 0.0 else float("nan")
        pred_all[cursor : cursor + m_s] = p_s
        b0_vec[cursor : cursor + m_s] = b0_s
        site_metric_rows.append(
            {
                "site": site,
                "n_obs": m_s,
                "intercept": b0_s,
                "rmse": rmse_s,
                "r2": r2_s,
            }
        )
        cursor += m_s

    resid_all = y_all - pred_all
    rmse_all = float(np.sqrt(np.mean(resid_all * resid_all)))
    ss_tot_all = float(np.sum((y_all - y_all.mean()) ** 2))
    r2_all = (
        float(1.0 - np.sum(resid_all * resid_all) / ss_tot_all)
        if ss_tot_all > 0.0
        else float("nan")
    )

    ci_lo = np.full(p, np.nan, dtype=float)
    ci_hi = np.full(p, np.nan, dtype=float)
    if int(n_bootstrap) > 0 and n_tot >= 4:
        L = min(
            n_tot,
            max(1, int(block_length) if block_length is not None else round(n_tot ** (1.0 / 3.0))),
        )
        max_start = max(1, n_tot - L + 1)
        n_draws = int(np.ceil(n_tot / L))
        rng = np.random.default_rng(int(seed))
        boot_betas = np.empty((int(n_bootstrap), p), dtype=float)
        for b_i in range(int(n_bootstrap)):
            starts = rng.integers(0, max_start, size=n_draws)
            idx_b = np.concatenate([np.arange(s, s + L) for s in starts])[:n_tot]
            boot_betas[b_i] = _solve_joint_centered(
                Fc_ord[idx_b], yc_ord[idx_b], chosen_alpha, non_negative, spatial_reg
            )
        ci_lo = np.percentile(boot_betas, 2.5, axis=0)
        ci_hi = np.percentile(boot_betas, 97.5, axis=0)

    mean_fp = F_all.mean(axis=0)
    mean_contrib = beta * mean_fp
    total_tr = float(np.sum(mean_contrib))
    mean_y = float(np.mean(y_all))
    reg_rows = [
        {
            "region": name,
            "beta": float(beta[j]),
            "beta_ci_low": float(ci_lo[j]),
            "beta_ci_high": float(ci_hi[j]),
            "mean_footprint": float(mean_fp[j]),
            "mean_contribution": float(mean_contrib[j]),
            "share_of_transport": float(mean_contrib[j] / total_tr) if total_tr != 0.0 else np.nan,
            "share_of_total": float(mean_contrib[j] / mean_y) if mean_y != 0.0 else np.nan,
        }
        for j, name in enumerate(display_names)
    ]

    contrib_data: dict[str, Any] = {
        "site": s_all,
        "date": pd.DatetimeIndex(t_all),
        "observed": y_all,
        "predicted": pred_all,
        "background": b0_vec,
    }
    comp_mat = F_all * beta[None, :]
    for j, name in enumerate(display_names):
        contrib_data[name] = comp_mat[:, j]
    contrib_data["transport_total"] = comp_mat.sum(axis=1)
    contrib_data["residual"] = resid_all
    contrib_df = pd.DataFrame(contrib_data).sort_values(["site", "date"]).reset_index(drop=True)

    return MultisiteInversionResult(
        intercepts=intercepts,
        regions=pd.DataFrame(reg_rows),
        site_metrics=pd.DataFrame(site_metric_rows),
        contributions=contrib_df,
        alpha=float(chosen_alpha),
        spatial_alpha=float(spatial_alpha),
        r2=r2_all,
        rmse=rmse_all,
        n_obs=n_tot,
    )


def _auto_tsv_elbow(tsv: np.ndarray) -> int:
    """Pick the elbow k >= 2 maximising perpendicular distance to the (1, max_k) chord."""
    m = len(tsv)
    if m <= 2:
        return m
    x = np.linspace(0.0, 1.0, m)
    y_span = float(tsv[0] - tsv[-1])
    if y_span <= 0.0:
        return min(2, m)
    y = (tsv - tsv[-1]) / y_span
    # Distance from each point (x_k, y_k) to line connecting (0, 1) -> (1, 0): (1 - x_k - y_k) / sqrt(2)
    dist = np.abs(1.0 - x - y)
    # Exclude endpoints (0 and m-1); pick k in 2..m-1 (1-based) if possible.
    if m > 2:
        best_idx = int(np.argmax(dist[1:-1])) + 1
        return best_idx + 1
    return 2


def _angle_distance_matrix(dx: np.ndarray, dy: np.ndarray, backend: str = "python") -> np.ndarray:
    """Sirois-Bottenheim mean directional-angle distances between trajectories.

    Each trajectory pair is scored by the mean angle between its step
    vectors from the receptor, normalised by pi (Sirois & Bottenheim 1995):
    0 for parallel paths, 1 for opposite ones. A step whose receptor
    distance product is at or below 1e-9 is stagnant and carries no
    direction, so it is skipped for that pair; a pair with no valid step
    scores exactly 0.

    ``dx``/``dy`` are the resampled tangent-plane coordinates (rows are
    trajectories, the first column is the receptor); steps are taken from
    the second column on. The pair loop is quadratic and performs a handful
    of small numpy calls per pair, so ``backend="rust"`` hands it to the
    optional ``normet_rust`` extension. The norms are computed here with
    ``np.hypot`` and passed in, and the per-pair mean keeps numpy's
    summation order, so the matrix is bit-identical on the platforms tested
    (numpy may dispatch SIMD ``arccos`` on some x86 builds, so parity is
    pinned per target); the default path is untouched.
    """
    vx = dx[:, 1:]
    vy = dy[:, 1:]
    norms = np.hypot(vx, vy)
    n_traj = int(vx.shape[0])
    if backend == "rust":
        from ..utils._rust import rust_core

        core = rust_core()
        flat = core.angle_distances(
            np.ascontiguousarray(vx, dtype=float),
            np.ascontiguousarray(vy, dtype=float),
            np.ascontiguousarray(norms, dtype=float),
        )
        return np.asarray(flat, dtype=float).reshape(n_traj, n_traj)

    dmat = np.zeros((n_traj, n_traj), dtype=float)
    for i in range(n_traj):
        for j in range(i + 1, n_traj):
            denom = norms[i] * norms[j]
            valid = denom > 1e-9
            if valid.any():
                cos_t = np.clip(
                    (vx[i, valid] * vx[j, valid] + vy[i, valid] * vy[j, valid]) / denom[valid],
                    -1.0,
                    1.0,
                )
                d = float(np.mean(np.arccos(cos_t) / np.pi))
            else:
                d = 0.0
            dmat[i, j] = dmat[j, i] = d
    return dmat


def cluster_trajectories(
    endpoints: pd.DataFrame,
    *,
    n_clusters: int | None = None,
    max_clusters: int = 8,
    n_points: int | None = None,
    z_scale: float = 0.0,
    metric: str = "euclidean",
    backend: str = "python",
    series: pd.Series | pd.DataFrame | None = None,
    value_col: str | None = None,
    date_col: str = "date",
    tolerance: str | pd.Timedelta | None = None,
    min_hours: float | None = None,
) -> TrajectoryClusterResult:
    """Cluster back-trajectories in local tangent-plane coordinates with TSV elbow selection.

    Resamples trajectories onto a common age grid, rotates longitudes smoothly
    across the dateline, and performs Ward hierarchical clustering on
    tangent-plane km coordinates (``metric="euclidean"``) or average-linkage
    clustering on Sirois-Bottenheim mean directional angles (``metric="angle"``).
    Clusters are numbered ``1..K`` in descending order of frequency. ``K`` is
    the number of groups ``fcluster`` actually returned: asking for more than
    the paths support (e.g. identical stagnant trajectories) yields fewer
    clusters rather than empty ones with NaN centroids.

    With ``metric="angle"``, ``backend="rust"`` computes the pairwise angle
    distances through the optional ``normet_rust`` extension; the distance
    matrix and every downstream result (labels, centroids, TSV, summary) are
    bit-identical on the platforms tested. The Euclidean path is scipy's
    Ward linkage and has no Rust backend, so it requires the default
    ``backend="python"``.
    """
    if metric not in ("euclidean", "angle"):
        raise ValueError(f"metric must be 'euclidean' or 'angle', got {metric!r}")
    if backend not in ("python", "rust"):
        raise ConfigError(f"`backend` must be 'python' or 'rust', got {backend!r}.")
    if backend == "rust" and metric != "angle":
        raise ConfigError(
            "`backend='rust'` accelerates only `metric='angle'`; the Euclidean "
            "path is scipy's Ward linkage."
        )
    if float(z_scale) < 0.0:
        raise ValueError(f"z_scale must be non-negative, got {z_scale}")
    if n_clusters is not None and int(n_clusters) < 2:
        raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")

    ep = _normalise_endpoints_frame(endpoints)
    trajs: list[pd.DataFrame] = []
    rec_times: list[pd.Timestamp] = []
    spans: list[float] = []
    counts: list[int] = []
    for (t_stamp, _sh, _tid), g in ep.groupby(["_t", "start_height", "traj_id"], sort=True):
        gs = g.sort_values("age_h", ascending=False).reset_index(drop=True)
        # A path that leaves the model domain has no coordinates on those rows
        # (HYSPLIT writes NaN). Interpolating over them spreads the NaN down
        # the whole path and scipy then refuses the distance matrix with "must
        # contain only finite values" -- an error that names neither the
        # trajectory nor the column. Drop them here, as the footprint path
        # does; a path with fewer than two located points is not a path.
        located = np.isfinite(
            pd.to_numeric(gs["lat"], errors="coerce").to_numpy(dtype=float)
        ) & np.isfinite(pd.to_numeric(gs["lon"], errors="coerce").to_numpy(dtype=float))
        if float(z_scale) > 0.0:
            located &= np.isfinite(
                pd.to_numeric(gs["height"], errors="coerce").to_numpy(dtype=float)
            )
        if not located.all():
            gs = gs[located].reset_index(drop=True)
        if len(gs) < 2:
            continue
        ages = pd.to_numeric(gs["age_h"], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(ages)):
            continue
        span = float(abs(ages[-1] - ages[0]))
        if min_hours is not None and span < float(min_hours) - 1e-6:
            continue
        trajs.append(gs)
        rec_times.append(pd.Timestamp(t_stamp))
        spans.append(span)
        counts.append(len(gs))

    n_traj = len(trajs)
    if n_traj < 2:
        raise ValueError(f"Need at least 2 multi-point trajectories to cluster, got {n_traj}.")

    common_span = min(spans)
    if common_span <= 0.0:
        raise ValueError("Trajectories have zero time span.")
    m_pts = int(n_points) if n_points is not None else max(3, min(counts))
    age_grid = np.linspace(0.0, -common_span, m_pts, dtype=float)
    abs_grid = np.abs(age_grid)

    lat0 = float(np.median([float(t["lat"].iloc[0]) for t in trajs]))
    lon0 = float(np.median([float(t["lon"].iloc[0]) for t in trajs]))
    cos_lat0 = max(float(np.cos(np.radians(lat0))), 0.1)

    dx = np.empty((n_traj, m_pts), dtype=float)
    dy = np.empty((n_traj, m_pts), dtype=float)
    h_grid = np.empty((n_traj, m_pts), dtype=float)
    for i, tr in enumerate(trajs):
        abs_a = np.abs(tr["age_h"].to_numpy(dtype=float))
        la = pd.to_numeric(tr["lat"], errors="coerce").to_numpy(dtype=float)
        lo_rel = (
            (pd.to_numeric(tr["lon"], errors="coerce").to_numpy(dtype=float) - lon0 + 180.0) % 360.0
        ) - 180.0
        lo_unw = np.degrees(np.unwrap(np.radians(lo_rel)))
        hg = pd.to_numeric(tr["height"], errors="coerce").to_numpy(dtype=float)

        order_a = np.argsort(abs_a)
        la_i = np.interp(abs_grid, abs_a[order_a], la[order_a])
        lo_i = np.interp(abs_grid, abs_a[order_a], lo_unw[order_a])
        h_i = np.interp(abs_grid, abs_a[order_a], hg[order_a])
        dy[i] = (la_i - lat0) * _KM_PER_DEG_LAT
        dx[i] = lo_i * _KM_PER_DEG_LAT * cos_lat0
        h_grid[i] = h_i

    parts = [dx, dy]
    if float(z_scale) > 0.0:
        dz = (h_grid - h_grid[:, :1]) / 1000.0 * float(z_scale)
        parts.append(dz)
    X = np.hstack(parts)

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if metric == "euclidean":
        Z = linkage(X, method="ward")
    else:
        # Sirois & Bottenheim (1995) angle distance over step vectors from receptor.
        dmat = _angle_distance_matrix(dx, dy, backend=backend)
        Z = linkage(squareform(dmat, checks=False), method="average")

    k_max_eff = min(int(max_clusters), n_traj)
    if n_clusters is not None:
        k_max_eff = max(k_max_eff, min(int(n_clusters), n_traj))
    if k_max_eff < 2:
        k_max_eff = min(2, n_traj)

    tsv_vals: list[float] = []
    for k in range(1, k_max_eff + 1):
        raw_lbl = fcluster(Z, t=k, criterion="maxclust") if k > 1 else np.ones(n_traj, dtype=int)
        sse = 0.0
        for cid in np.unique(raw_lbl):
            sub = X[raw_lbl == cid]
            sse += float(np.sum((sub - sub.mean(axis=0)) ** 2))
        tsv_vals.append(sse)
    tsv_arr = np.asarray(tsv_vals, dtype=float)
    pct_chg = np.full(len(tsv_arr), np.nan, dtype=float)
    if len(tsv_arr) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_chg[1:] = np.where(
                tsv_arr[:-1] > 0.0,
                100.0 * (tsv_arr[:-1] - tsv_arr[1:]) / tsv_arr[:-1],
                0.0,
            )
    tsv_df = pd.DataFrame(
        {"k": np.arange(1, k_max_eff + 1, dtype=int), "tsv": tsv_arr, "pct_change": pct_chg}
    )

    chosen_k = int(n_clusters) if n_clusters is not None else _auto_tsv_elbow(tsv_arr)
    chosen_k = max(1, min(chosen_k, n_traj))
    raw_lbl = (
        fcluster(Z, t=chosen_k, criterion="maxclust")
        if chosen_k > 1
        else np.ones(n_traj, dtype=int)
    )

    # Relabel the clusters fcluster actually returned, sorted by descending
    # member count (ties broken stably). fcluster's ``maxclust`` returns *at
    # most* chosen_k groups: identical (stagnant) paths cannot be split, so
    # looping to the requested k would report empty clusters with NaN
    # centroids and emit "Mean of empty slice" warnings. Report the number
    # found instead of the number asked for.
    order_c = sorted(np.unique(raw_lbl), key=lambda cid: (-int(np.sum(raw_lbl == cid)), int(cid)))
    remap = {old: new for new, old in enumerate(order_c, start=1)}
    lbl_arr = np.asarray([remap[int(c)] for c in raw_lbl], dtype=int)
    found_k = len(order_c)
    if found_k < chosen_k:
        log.info(
            "fcluster returned %d distinct cluster(s), not the %d requested: the "
            "trajectories are too similar to separate further.",
            found_k,
            chosen_k,
        )

    # Target values per trajectory if series is supplied.
    y_traj: np.ndarray | None = None
    if series is not None:
        obs_df = _resolve_series_frame(series, value_col=value_col, date_col=date_col)
        y_traj = _match_timestamps(pd.Series(rec_times), obs_df, tolerance=tolerance)

    centroid_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for k_id in range(1, found_k + 1):
        sel = lbl_arr == k_id
        cnt = int(sel.sum())
        freq = float(cnt / n_traj)
        dx_m = dx[sel].mean(axis=0)
        dy_m = dy[sel].mean(axis=0)
        h_m = h_grid[sel].mean(axis=0)
        c_lat = lat0 + dy_m / _KM_PER_DEG_LAT
        c_lon = ((lon0 + dx_m / (_KM_PER_DEG_LAT * cos_lat0) + 180.0) % 360.0) - 180.0
        for p_idx in range(m_pts):
            centroid_rows.append(
                {
                    "cluster": k_id,
                    "age_h": float(age_grid[p_idx]),
                    "lat": float(c_lat[p_idx]),
                    "lon": float(c_lon[p_idx]),
                    "height": float(h_m[p_idx]),
                    "n_trajectories": cnt,
                    "freq": freq,
                }
            )
        mean_dist = float(np.hypot(dx_m[-1], dy_m[-1]))
        mean_bear = float((np.degrees(np.arctan2(dx_m[-1], dy_m[-1])) + 360.0) % 360.0)
        s_row: dict[str, Any] = {
            "cluster": k_id,
            "n_trajectories": cnt,
            "freq": freq,
            "mean_dist_km": mean_dist,
            "mean_inflow_deg": mean_bear,
            "mean_height_m": float(h_m.mean()),
        }
        if y_traj is not None:
            yk = y_traj[sel]
            fin_y = yk[np.isfinite(yk)]
            s_row["target_mean"] = float(np.mean(fin_y)) if fin_y.size else np.nan
            s_row["target_median"] = float(np.median(fin_y)) if fin_y.size else np.nan
            s_row["target_p90"] = float(np.percentile(fin_y, 90)) if fin_y.size else np.nan
        summary_rows.append(s_row)

    # Collapse potential multi-height trajectories to dominant cluster per receptor_date.
    lbl_df = pd.DataFrame({"date": pd.DatetimeIndex(rec_times), "cluster": lbl_arr})
    labels_s = (
        lbl_df.groupby("date")["cluster"].agg(lambda s: int(s.value_counts().index[0])).sort_index()
    )
    labels_s.name = "traj_cluster"

    return TrajectoryClusterResult(
        labels=labels_s,
        centroids=pd.DataFrame(centroid_rows),
        summary=pd.DataFrame(summary_rows),
        tsv=tsv_df,
        n_clusters=found_k,
    )


def screen_donor_transport(
    endpoints: pd.DataFrame,
    treated_lat: float,
    treated_lon: float,
    donors: pd.DataFrame,
    *,
    mode: str = "donor_back",
    unit_col: str = "unit",
    lat_col: str = "lat",
    lon_col: str = "lon",
    site_col: str = "site",
    radius_km: float = 50.0,
    blh_coupled: bool = True,
    blh_factor: float = 1.0,
    min_blh_m: float = 100.0,
    fallback_height_m: float | None = 1500.0,
    max_exposure_frac: float = 0.10,
) -> pd.DataFrame:
    """Screen synthetic-control donor units for physical air-mass spillover coupling.

    - ``mode="donor_back"``: ``endpoints`` holds back-trajectories arriving at each
      donor site (identified by ``site_col``); flags donors whose incoming air
      frequently passed inside the BLH within ``radius_km`` of the treated site
      (downwind spillover violating SUTVA).
    - ``mode="treated_back"``: ``endpoints`` holds back-trajectories arriving at
      the treated site; flags donor units whose coordinates frequently lie inside
      the treated site's BLH-coupled upwind fetch (shared regional transport).
    """
    if mode not in ("donor_back", "treated_back"):
        raise ValueError(f"mode must be 'donor_back' or 'treated_back', got {mode!r}")
    if float(radius_km) <= 0.0:
        raise ValueError(f"radius_km must be positive, got {radius_km}")
    if not (0.0 <= float(max_exposure_frac) <= 1.0):
        raise ValueError(f"max_exposure_frac must be in [0, 1], got {max_exposure_frac}")
    for c in (unit_col, lat_col, lon_col):
        if c not in donors.columns:
            raise KeyError(f"`donors` missing required column {c!r}.")

    ep = _normalise_endpoints_frame(endpoints).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    rad = float(radius_km)

    if mode == "donor_back":
        if site_col not in ep.columns:
            raise KeyError(f"mode='donor_back' requires {site_col!r} column in `endpoints`.")
        ep[site_col] = ep[site_col].astype(str)
        for _, drow in donors.iterrows():
            unit = str(drow[unit_col])
            sub = ep[ep[site_col] == unit].reset_index(drop=True)
            if sub.empty:
                rows.append(
                    {
                        "unit": unit,
                        "n_trajectories": 0,
                        "hit_frac": np.nan,
                        "mean_footprint": np.nan,
                        "role": "downwind_spillover",
                        "flagged": False,
                    }
                )
                continue
            # Weight and convert this donor's table once; the group loop
            # then only slices, so each hit sum keeps its element order.
            # Computing on `sub` preserves the old semantics even if one
            # trajectory's rows were split across sites.
            w_sub = _endpoint_weights_flat_frame(
                sub,
                _trajectory_codes(sub),
                blh_coupled=blh_coupled,
                blh_factor=blh_factor,
                min_blh_m=min_blh_m,
                fallback_height_m=fallback_height_m,
                exclude_receptor=True,
            )
            # Table-wide distances are elementwise; a SIMD transcendental
            # build could in principle flip a distance sitting exactly on
            # `rad`, so parity is pinned on the tested targets.
            d_sub = np.asarray(
                _haversine_km(
                    float(treated_lat),
                    float(treated_lon),
                    pd.to_numeric(sub["lat"], errors="coerce").to_numpy(dtype=float),
                    pd.to_numeric(sub["lon"], errors="coerce").to_numpy(dtype=float),
                ),
                dtype=float,
            )
            hits = 0
            fps: list[float] = []
            for _, g in sub.groupby(["_t", "start_height", "traj_id"], sort=False):
                pos = g.index.to_numpy()
                w = w_sub[pos]
                d_km = d_sub[pos]
                hit_w = float(w[(d_km <= rad) & (w > 0.0)].sum())
                if hit_w > 0.0:
                    hits += 1
                fps.append(hit_w)
            n_t = len(fps)
            frac = float(hits / n_t)
            rows.append(
                {
                    "unit": unit,
                    "n_trajectories": n_t,
                    "hit_frac": frac,
                    "mean_footprint": float(np.mean(fps)),
                    "role": "downwind_spillover",
                    "flagged": bool(frac > float(max_exposure_frac)),
                }
            )
    else:
        traj_groups = list(ep.groupby(["_t", "start_height", "traj_id"], sort=False))
        w_all = _endpoint_weights_flat_frame(
            ep,
            _trajectory_codes(ep),
            blh_coupled=blh_coupled,
            blh_factor=blh_factor,
            min_blh_m=min_blh_m,
            fallback_height_m=fallback_height_m,
            exclude_receptor=True,
        )
        lat_all = pd.to_numeric(ep["lat"], errors="coerce").to_numpy(dtype=float)
        lon_all = pd.to_numeric(ep["lon"], errors="coerce").to_numpy(dtype=float)
        # Store row positions, not copies: weights and distances are sliced
        # per trajectory below, preserving each hit sum's element order.
        traj_pos: list[np.ndarray] = [g.index.to_numpy() for _, g in traj_groups]

        n_t = len(traj_pos)
        for _, drow in donors.iterrows():
            unit = str(drow[unit_col])
            dlat = float(drow[lat_col])
            dlon = float(drow[lon_col])
            # See the donor_back note above on table-wide distances.
            d_all = np.asarray(_haversine_km(dlat, dlon, lat_all, lon_all), dtype=float)
            hits = 0
            fps = []
            for pos in traj_pos:
                w = w_all[pos]
                d_km = d_all[pos]
                hit_w = float(w[(d_km <= rad) & (w > 0.0)].sum())
                if hit_w > 0.0:
                    hits += 1
                fps.append(hit_w)
            frac = float(hits / n_t) if n_t else np.nan
            rows.append(
                {
                    "unit": unit,
                    "n_trajectories": n_t,
                    "hit_frac": frac,
                    "mean_footprint": float(np.mean(fps)) if fps else np.nan,
                    "role": "upwind_coupled",
                    "flagged": bool(n_t > 0 and frac > float(max_exposure_frac)),
                }
            )

    cols = ["unit", "n_trajectories", "hit_frac", "mean_footprint", "role", "flagged"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return (
        pd.DataFrame(rows, columns=cols)
        .sort_values(["hit_frac", "mean_footprint"], ascending=[False, False], na_position="last")
        .reset_index(drop=True)
    )
