"""Parity tests for the flat-array endpoint-weight path.

``trajectory_endpoint_weights`` is the public per-trajectory implementation.
The transport call sites use ``_endpoint_weights_flat_frame`` to convert a
whole endpoint table once instead of once per trajectory; that changed the
weight loop from ~66 % pandas overhead to a flat numpy kernel. The contract
is bit-identity: the flat path must reproduce the scalar path element for
element (compared as ``uint64`` views, so NaN payloads and signed zeros
count), and the public functions must reproduce their pre-flat outputs
exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from normet.analysis import build_footprint_matrix, screen_donor_transport, wcwt, wpscf
from normet.analysis.transport import (
    _endpoint_weights_flat_frame,
    _normalise_endpoints_frame,
    _region_mask,
    _trajectory_codes,
)
from normet.io.trajectory import _haversine_km, trajectory_endpoint_weights

_KEYS = ["_t", "start_height", "traj_id"]


def _endpoint_corpus() -> pd.DataFrame:
    """Interleaved trajectories spanning the weight branches.

    Contains uniform and irregular ages, a duplicate age (zero step), an
    all-NaN age group, a one-point group, heights above and below the BLH,
    missing and infinite BLH, and NaN / negative / infinite rainfall.
    Rows are shuffled, so no call site may rely on contiguous groups.
    """
    rng = np.random.default_rng(11)
    times = pd.date_range("2023-03-01", periods=18, freq="6h")
    rows: list[dict[str, Any]] = []
    for i, ts in enumerate(times):
        regime = i % 6
        if regime == 0:
            ages = tuple(np.arange(0.0, -25.0, -3.0))
        elif regime == 1:
            ages = (0.0, -2.0, -5.0, -5.0, -9.0, -17.0)
        elif regime == 2:
            ages = (0.0, np.nan, -6.0)
        elif regime == 3:
            ages = (0.0,)
        elif regime == 4:
            ages = (0.0, -1.5, -4.5, -12.5)
        else:
            ages = (0.0, -6.0, -12.0)
        for k, age in enumerate(ages):
            if regime == 0:
                lat, lon, h = 51.5 + 0.1 * abs(age), 0.0, 250.0
            elif regime == 1:
                lat, lon, h = 51.5, 0.1 * abs(age), 2400.0 if k == 2 else 300.0
            elif regime == 2:
                lat, lon, h = 51.5 - 0.1 * abs(age if np.isfinite(age) else 0.0), 0.0, 600.0
            elif regime == 3:
                lat, lon, h = 51.5, 0.0, 200.0
            else:
                lat, lon, h = 51.5 + 0.05 * abs(age), 0.5, 800.0
            blh: float = 800.0 + 20.0 * float(rng.normal())
            if regime == 2 and k % 2:
                blh = np.nan
            elif regime == 4 and k == 1:
                blh = np.inf
            rain = float(rng.choice([0.0, 0.0, 0.4, 2.5, np.nan, -1.0, np.inf]))
            height = float(rng.choice([h, h, h, np.inf, np.nan])) if regime == 5 else h
            rows.append(
                {
                    "receptor_date": ts,
                    "start_lat": 51.5,
                    "start_lon": 0.0,
                    "start_height": 250.0,
                    "traj_id": 1,
                    "site": f"S{i % 4}",
                    "datetime": (ts + pd.Timedelta(hours=age)) if np.isfinite(age) else pd.NaT,
                    "age_h": age,
                    "lat": lat + 0.01 * float(rng.normal()),
                    "lon": lon + 0.01 * float(rng.normal()),
                    "height": height,
                    "blh": blh,
                    "rainfall": rain,
                }
            )
    ep = pd.DataFrame(rows)
    return ep.sample(frac=1.0, random_state=11).reset_index(drop=True)


def _reference_weights(ep: pd.DataFrame, **kw: Any) -> np.ndarray:
    """The public per-trajectory function, applied group by group."""
    out = np.empty(len(ep), dtype=float)
    for _, g in ep.groupby(_KEYS, sort=False):
        out[g.index.to_numpy()] = trajectory_endpoint_weights(g, **kw)
    return out


def _assert_weights_identical(ep: pd.DataFrame, **kw: Any) -> None:
    ep = _normalise_endpoints_frame(ep).reset_index(drop=True)
    codes = _trajectory_codes(ep)
    with np.errstate(all="ignore"):
        ref = _reference_weights(ep, **kw)
        got = _endpoint_weights_flat_frame(ep, codes, **kw)
    assert np.array_equal(ref.view(np.uint64), got.view(np.uint64))


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"blh_coupled": False},
        {"decay_hours": 9.0},
        {"scavenging_coeff": 0.03},
        {"exclude_receptor": False},
        {"fallback_height_m": None},
        {"blh_factor": 0.8, "min_blh_m": 150.0},
        {"decay_hours": 18.0, "scavenging_coeff": 0.05, "fallback_height_m": 1300.0},
    ],
)
def test_flat_weights_match_the_scalar_path(kw: dict[str, Any]) -> None:
    _assert_weights_identical(_endpoint_corpus(), **kw)


def test_flat_weights_without_blh_or_rainfall_columns() -> None:
    ep = _endpoint_corpus()
    _assert_weights_identical(ep.drop(columns=["blh"]))
    _assert_weights_identical(ep.drop(columns=["blh"]), fallback_height_m=None)
    _assert_weights_identical(ep.drop(columns=["rainfall"]), scavenging_coeff=0.05)


def test_flat_weights_empty_frame() -> None:
    empty = _normalise_endpoints_frame(_endpoint_corpus()).iloc[:0]
    got = _endpoint_weights_flat_frame(empty, np.empty(0, dtype=np.int64))
    assert got.shape == (0,)


def test_trajectory_codes_reject_null_keys() -> None:
    """A null key would cast to code 0 and contaminate the first trajectory."""
    ep = _normalise_endpoints_frame(_endpoint_corpus())
    ep.loc[ep.index[0], "start_height"] = np.nan
    with pytest.raises(ValueError, match="non-null"):
        _trajectory_codes(ep)


def test_flat_weights_validation_matches_the_scalar_path() -> None:
    ep = _normalise_endpoints_frame(_endpoint_corpus()).reset_index(drop=True)
    cases: list[tuple[dict[str, Any], str]] = [
        ({"blh_factor": 0.0}, "blh_factor must be positive"),
        ({"min_blh_m": 0.0}, "min_blh_m must be positive"),
        ({"decay_hours": 0.0}, "decay_hours must be positive"),
        ({"scavenging_coeff": -1.0}, "scavenging_coeff must be non-negative"),
    ]
    codes = _trajectory_codes(ep)
    for kw, msg in cases:
        with pytest.raises(ValueError, match=msg):
            trajectory_endpoint_weights(ep, **kw)
        with pytest.raises(ValueError, match=msg):
            _endpoint_weights_flat_frame(ep, codes, **kw)


def _assert_grid_equal(got: Any, ref: Any) -> None:
    for field in (
        "lon_edges",
        "lat_edges",
        "lon_centers",
        "lat_centers",
        "field",
        "weighted",
        "weight_matrix",
        "exposure",
        "count",
    ):
        a = np.asarray(getattr(got, field))
        b = np.asarray(getattr(ref, field))
        assert np.array_equal(a.view(np.uint64), b.view(np.uint64)), field
    pd.testing.assert_frame_equal(got.to_dataframe(), ref.to_dataframe(), check_exact=True)


def _reference_footprint(
    endpoints: pd.DataFrame,
    regions: dict[str, Any],
    **kw: Any,
) -> pd.DataFrame:
    """The pre-flat per-trajectory footprint implementation."""
    ep = _normalise_endpoints_frame(endpoints)
    reg_names = [str(k) for k in regions]
    rows: list[dict[str, Any]] = []
    for (t_stamp, start_h, _tid), g in ep.groupby(_KEYS, sort=True):
        w = trajectory_endpoint_weights(g, **kw)
        lon = pd.to_numeric(g["lon"], errors="coerce").to_numpy(dtype=float)
        lat = pd.to_numeric(g["lat"], errors="coerce").to_numpy(dtype=float)
        rec: dict[str, Any] = {"date": t_stamp, "_h": start_h}
        for name, region in regions.items():
            inside = np.asarray(_region_mask(region, lon, lat), dtype=bool)
            rec[str(name)] = float(w[inside].sum())
        rows.append(rec)
    df = pd.DataFrame(rows)
    out = df.groupby("date", as_index=True)[reg_names].mean().sort_index()
    out.index = pd.DatetimeIndex(out.index, name="date")
    return out


def _reference_screen(
    endpoints: pd.DataFrame,
    treated_lat: float,
    treated_lon: float,
    donors: pd.DataFrame,
    *,
    mode: str,
    radius_km: float,
) -> pd.DataFrame:
    """The pre-flat per-trajectory screening implementation."""
    ep = _normalise_endpoints_frame(endpoints)
    rows: list[dict[str, Any]] = []
    rad = float(radius_km)
    if mode == "donor_back":
        ep["site"] = ep["site"].astype(str)
        for _, drow in donors.iterrows():
            unit = str(drow["unit"])
            sub = ep[ep["site"] == unit]
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
            hits = 0
            fps: list[float] = []
            for _, g in sub.groupby(_KEYS, sort=False):
                w = trajectory_endpoint_weights(g, exclude_receptor=True)
                d_km = np.asarray(
                    _haversine_km(
                        float(treated_lat),
                        float(treated_lon),
                        pd.to_numeric(g["lat"], errors="coerce").to_numpy(dtype=float),
                        pd.to_numeric(g["lon"], errors="coerce").to_numpy(dtype=float),
                    ),
                    dtype=float,
                )
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
                    "flagged": bool(frac > 0.10),
                }
            )
    else:
        traj_groups = list(ep.groupby(_KEYS, sort=False))
        traj_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for _, g in traj_groups:
            w = trajectory_endpoint_weights(g, exclude_receptor=True)
            la = pd.to_numeric(g["lat"], errors="coerce").to_numpy(dtype=float)
            lo = pd.to_numeric(g["lon"], errors="coerce").to_numpy(dtype=float)
            traj_data.append((la, lo, w))
        n_t = len(traj_data)
        for _, drow in donors.iterrows():
            unit = str(drow["unit"])
            dlat = float(drow["lat"])
            dlon = float(drow["lon"])
            hits = 0
            fps = []
            for la, lo, w in traj_data:
                d_km = np.asarray(_haversine_km(dlat, dlon, la, lo), dtype=float)
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
                    "flagged": bool(n_t > 0 and frac > 0.10),
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


def _series() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-03-01", periods=18, freq="6h"),
            "pm25": 20.0 + 10.0 * np.sin(np.arange(18) / 3.0),
        }
    )


def _donors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit": ["S0", "S1", "S2", "S3"],
            "lat": [52.0, 53.0, 55.0, 50.0],
            "lon": [0.0, 1.0, 2.0, -1.0],
        }
    )


_REGIONS = {
    "north": (0.0, 51.5, 1.0, 54.0),
    "east": (0.0, 51.0, 3.0, 52.0),
    "box_wrap": (179.0, 50.0, -179.0, 54.0),
}


def test_build_footprint_matrix_matches_its_scalar_reference() -> None:
    ep = _endpoint_corpus()
    kw = {"decay_hours": 12.0, "scavenging_coeff": 0.02}
    ref = _reference_footprint(ep, _REGIONS, **kw)
    with np.errstate(all="ignore"):
        got = build_footprint_matrix(ep, _REGIONS, **kw)
    pd.testing.assert_frame_equal(got, ref, check_exact=True)


def test_gridded_models_match_the_scalar_weight_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace only the flat kernel with its scalar loop; everything else is real."""
    import normet.analysis.transport as transport

    def scalar_adapter(traj: pd.DataFrame, codes: np.ndarray, **kw: Any) -> np.ndarray:
        out = np.empty(len(traj), dtype=float)
        for _, g in traj.groupby(codes, sort=False):
            out[g.index.to_numpy()] = trajectory_endpoint_weights(g, **kw)
        return out

    ep = _endpoint_corpus()
    series = _series()
    args = {"value_col": "pm25", "grid_res": 1.0, "decay_hours": 18.0, "scavenging_coeff": 0.03}
    wpscf_args = {"value_col": "pm25", "grid_res": 1.0, "threshold": 22.0, "blh_coupled": False}

    with monkeypatch.context() as m:
        m.setattr(transport, "_endpoint_weights_flat_frame", scalar_adapter)
        ref_cwt = wcwt(ep, series, **args)
        ref_pscf = wpscf(ep, series, **wpscf_args)
    got_cwt = wcwt(ep, series, **args)
    got_pscf = wpscf(ep, series, **wpscf_args)

    _assert_grid_equal(got_cwt, ref_cwt)
    _assert_grid_equal(got_pscf, ref_pscf)


@pytest.mark.parametrize("mode", ["donor_back", "treated_back"])
def test_screen_donor_transport_matches_its_scalar_reference(mode: str) -> None:
    ep = _endpoint_corpus()
    donors = _donors()
    with np.errstate(all="ignore"):
        ref = _reference_screen(ep, 51.5, 0.0, donors, mode=mode, radius_km=120.0)
        got = screen_donor_transport(ep, 51.5, 0.0, donors, mode=mode, radius_km=120.0)
    pd.testing.assert_frame_equal(got, ref, check_exact=True)


def test_screen_donor_back_keeps_split_trajectory_subsets() -> None:
    """A trajectory key split across sites is weighted per subset, as before."""
    ep = _endpoint_corpus()
    split = pd.concat(
        [ep.iloc[: len(ep) // 2].assign(site="A"), ep.iloc[len(ep) // 2 :].assign(site="B")],
        ignore_index=True,
    )
    # Give one trajectory rows under both sites by copying a row.
    first = split.iloc[[0]].copy()
    first["site"] = "B"
    split = pd.concat([split, first], ignore_index=True)
    donors = pd.DataFrame({"unit": ["A", "B"], "lat": [52.0, 51.0], "lon": [0.0, 0.1]})
    with np.errstate(all="ignore"):
        ref = _reference_screen(split, 51.5, 0.0, donors, mode="donor_back", radius_km=150.0)
        got = screen_donor_transport(split, 51.5, 0.0, donors, mode="donor_back", radius_km=150.0)
    pd.testing.assert_frame_equal(got, ref, check_exact=True)


def test_public_functions_ignore_the_input_index() -> None:
    """Every call site resets to a positional index before slicing."""
    ep = _endpoint_corpus()
    shuffled = ep.set_axis(pd.Index(np.arange(len(ep)) * 7 + 3))
    series = _series()
    donors = _donors()
    with np.errstate(all="ignore"):
        pd.testing.assert_frame_equal(
            build_footprint_matrix(shuffled, _REGIONS),
            build_footprint_matrix(ep, _REGIONS),
            check_exact=True,
        )
        _assert_grid_equal(
            wcwt(shuffled, series, value_col="pm25", grid_res=1.0, decay_hours=12.0),
            wcwt(ep, series, value_col="pm25", grid_res=1.0, decay_hours=12.0),
        )
        for mode in ("donor_back", "treated_back"):
            pd.testing.assert_frame_equal(
                screen_donor_transport(shuffled, 51.5, 0.0, donors, mode=mode, radius_km=120.0),
                screen_donor_transport(ep, 51.5, 0.0, donors, mode=mode, radius_km=120.0),
                check_exact=True,
            )


def _random_endpoint_frame(rng: np.random.Generator) -> pd.DataFrame:
    """A small random frame covering arbitrary spacing, missing values and order."""
    rows: list[dict[str, Any]] = []
    base = pd.Timestamp("2024-02-01")
    for traj in range(int(rng.integers(1, 10))):
        ages = rng.choice([0.0, -1.0, -3.0, -4.5, -9.0, -12.0], size=int(rng.integers(1, 10)))
        if rng.random() < 0.15:
            ages[int(rng.integers(ages.size))] = np.nan
        for age in ages:
            rows.append(
                {
                    "receptor_date": base + pd.Timedelta(hours=6 * traj),
                    "start_height": 100.0,
                    "traj_id": int(rng.integers(3)),
                    "age_h": float(age),
                    "lat": 50.0 + float(rng.normal()),
                    "lon": float(rng.normal()),
                    "height": float(rng.choice([100.0, 500.0, 1500.0, 3000.0, np.inf])),
                    "blh": float(rng.choice([200.0, 800.0, 1200.0, np.nan, np.inf])),
                    "rainfall": float(rng.choice([0.0, 0.5, 2.0, np.nan, -1.0, np.inf])),
                }
            )
    frame = pd.DataFrame(rows)
    if rng.random() < 0.2:
        frame = frame.drop(columns=["blh"])
    if rng.random() < 0.2:
        frame = frame.drop(columns=["rainfall"])
    return frame.sample(frac=1.0, random_state=int(rng.integers(2**31))).reset_index(drop=True)


def test_flat_weights_randomized_parity() -> None:
    """Fuzz the flat/scalar contract, since only tests keep the copies in sync."""
    rng = np.random.default_rng(20261127)
    for _ in range(25):
        kw: dict[str, Any] = {
            "blh_coupled": bool(rng.integers(2)),
            "blh_factor": float(rng.uniform(0.5, 1.5)),
            "min_blh_m": float(rng.uniform(50.0, 200.0)),
            "fallback_height_m": None if rng.random() < 0.2 else float(rng.choice([500.0, 1500.0])),
            "decay_hours": None if rng.random() < 0.5 else float(rng.uniform(2.0, 50.0)),
            "scavenging_coeff": None if rng.random() < 0.5 else float(rng.uniform(0.0, 0.2)),
            "exclude_receptor": bool(rng.integers(2)),
        }
        _assert_weights_identical(_random_endpoint_frame(rng), **kw)
