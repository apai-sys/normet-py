"""Tests for Layer-2 Lagrangian transport analysis and stratified normalisation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import normet as nm
from normet.analysis import (
    build_footprint_matrix,
    cluster_trajectories,
    decom_emi,
    decom_met,
    decompose,
    invert_source_footprints,
    normalise,
    screen_donor_transport,
    wcwt,
    wpscf,
)
from normet.analysis.transport import _polissar_weights, _resolve_series_frame


def _make_corridor_endpoints(n_traj: int = 36, *, high_alt_south: bool = False):
    """Build synthetic trajectories from North (polluted), East (clean), or South (high-alt)."""
    rng = np.random.default_rng(42)
    lat0, lon0 = 51.5, 0.0
    times = pd.date_range("2023-01-01", periods=n_traj, freq="6h")
    rows = []
    y_rows = []
    for i, ts in enumerate(times):
        regime = i % 3  # 0: North polluted, 1: East clean, 2: South clean (or high alt)
        base_conc = {0: 80.0, 1: 15.0, 2: 20.0}[regime] + float(rng.normal(0.0, 1.5))
        y_rows.append({"date": ts, "pm25": base_conc})
        for age in (0, -3, -6, -9, -12):
            step = abs(age) / 3.0
            if regime == 0:
                la = lat0 + 0.6 * step
                lo = lon0 + float(rng.normal(0.0, 0.05))
                h = 250.0
            elif regime == 1:
                la = lat0 + float(rng.normal(0.0, 0.05))
                lo = lon0 + 0.8 * step
                h = 300.0
            else:
                la = lat0 - 0.6 * step
                lo = lon0 + float(rng.normal(0.0, 0.05))
                h = 2500.0 if high_alt_south and age != 0 else 300.0
            rows.append(
                {
                    "receptor_date": ts,
                    "start_lat": lat0,
                    "start_lon": lon0,
                    "start_height": 250.0,
                    "traj_id": 1,
                    "datetime": ts + pd.Timedelta(hours=age),
                    "age_h": float(age),
                    "lat": la,
                    "lon": lo,
                    "height": h,
                    "blh": 800.0,
                    "rainfall": 0.0,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(y_rows)


def test_wcwt_and_wpscf_recover_polluted_corridor_and_gate_blh():
    ep, series = _make_corridor_endpoints(36, high_alt_south=True)
    # Check WCWT highlights North (>53 deg) over East/South. The raw
    # concentration field, not `weighted`: the Polissar confidence weight
    # deliberately down-weights the sparsely sampled northern cells, so the
    # weighted score mixes source strength with sampling density.
    res_cwt = wcwt(ep, series, value_col="pm25", grid_res=0.8)
    df_cwt = res_cwt.to_dataframe()
    assert not df_cwt.empty
    north = df_cwt[df_cwt["lat"] > 52.5]["value"].mean()
    east = df_cwt[(df_cwt["lon"] > 1.0) & (df_cwt["lat"].between(51.0, 52.0))]["value"].mean()
    assert north > 2.5 * east

    # High-altitude Southern air (> BLH 800 m) is gated out when blh_coupled=True
    south_coupled = df_cwt[df_cwt["lat"] < 50.5]
    assert south_coupled.empty
    res_un = wcwt(ep, series, value_col="pm25", grid_res=0.8, blh_coupled=False)
    assert not res_un.to_dataframe()[res_un.to_dataframe()["lat"] < 50.5].empty

    # WPSCF selects polluted North cells cleanly
    res_pscf = wpscf(ep, series, value_col="pm25", grid_res=0.8, threshold=50.0)
    df_p = res_pscf.to_dataframe()
    assert df_p[df_p["lat"] > 52.5]["value"].mean() > 0.9
    assert df_p[df_p["lon"] > 1.0]["value"].mean() == 0.0


def test_footprint_inversion_recovers_betas_cis_and_rolling_step():
    rng = np.random.default_rng(7)
    n = 160
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    fp = pd.DataFrame(
        {
            "north": rng.uniform(0.0, 5.0, n),
            "east": rng.uniform(0.0, 5.0, n),
        },
        index=dates,
    )
    # Rolling test: first 80 days have beta_north=6.0, last 80 days drop to 2.0 (emission control!)
    beta_n = np.where(np.arange(n) < 80, 6.0, 2.0)
    y_step = 10.0 + beta_n * fp["north"] + 1.5 * fp["east"] + rng.normal(0.0, 0.25, n)
    series = pd.Series(y_step, index=dates, name="pm25")

    # Static fit on first half should accurately recover beta_0=10, north~6, east~1.5
    inv_static = invert_source_footprints(
        fp.iloc[:80], series.iloc[:80], alpha=0.01, n_bootstrap=40, seed=11
    )
    reg = inv_static.regions.set_index("region")
    assert inv_static.intercept == pytest.approx(10.0, abs=0.4)
    assert reg.loc["north", "beta"] == pytest.approx(6.0, abs=0.25)
    assert reg.loc["east", "beta"] == pytest.approx(1.5, abs=0.25)
    assert (
        reg.loc["north", "beta_ci_low"]
        <= reg.loc["north", "beta"]
        <= reg.loc["north", "beta_ci_high"]
    )
    assert inv_static.r2 > 0.98

    # Sliding-window inversion detects upstream intensity decline in second half
    inv_roll = invert_source_footprints(
        fp, series, alpha="auto", n_bootstrap=20, window_days=40, step_days=20, min_window_obs=25
    )
    rb = inv_roll.rolling_beta
    assert rb is not None and len(rb) >= 4
    assert rb["north"].iloc[0] > 4.5
    assert rb["north"].iloc[-1] < 3.0

    # Can also consume build_footprint_matrix directly
    ep, _ = _make_corridor_endpoints(12)
    regions = {"N": (-2.0, 52.0, 2.0, 56.0), "E": (0.8, 50.5, 5.0, 52.5)}
    mat = build_footprint_matrix(ep, regions)
    assert list(mat.columns) == ["N", "E"] and (mat.to_numpy() >= 0.0).all()


def test_cluster_trajectories_euclidean_angle_and_tsv():
    ep, series = _make_corridor_endpoints(36)
    res_euc = cluster_trajectories(ep, series=series, value_col="pm25")
    assert res_euc.n_clusters == 3
    assert set(res_euc.labels) == {1, 2, 3}
    # TSV decreases monotonically as k increases
    assert (np.diff(res_euc.tsv["tsv"].to_numpy()) <= 1e-8).all()
    assert {"target_mean", "target_median", "target_p90"} <= set(res_euc.summary.columns)
    assert res_euc.summary["target_mean"].max() > 70.0

    res_ang = cluster_trajectories(ep, metric="angle", n_clusters=3)
    assert res_ang.n_clusters == 3 and len(res_ang.summary) == 3


def test_screen_donor_transport_spillover_and_upwind_coupling():
    ts = pd.date_range("2023-06-01", periods=10, freq="6h")
    # Donor D_down passes near treated (51.5, 0.0) in 8 of 10 trajectories at 300 m (< BLH 800 m).
    # Donor D_hi passes at 2500 m (> BLH 800 m) -> decoupled, should NOT flag.
    # Donor D_far never passes near treated.
    ep_rows = []
    for i, t in enumerate(ts):
        for unit, lat_path, h in (
            ("D_down", 51.52 if i < 8 else 54.0, 300.0),
            ("D_hi", 51.52, 2500.0),
            ("D_far", 55.0, 300.0),
        ):
            for age in (0, -3, -6):
                ep_rows.append(
                    {
                        "receptor_date": t,
                        "site": unit,
                        "age_h": float(age),
                        "lat": lat_path + 0.02 * abs(age),
                        "lon": 0.0 + 0.02 * abs(age),
                        "height": h if age != 0 else 200.0,
                        "blh": 800.0,
                    }
                )
    donors = pd.DataFrame(
        {
            "unit": ["D_down", "D_hi", "D_far"],
            "lat": [52.0, 52.0, 55.0],
            "lon": [1.0, 1.0, 3.0],
        }
    )
    out = screen_donor_transport(
        pd.DataFrame(ep_rows), 51.5, 0.0, donors, mode="donor_back", radius_km=30.0
    )
    by_u = out.set_index("unit")
    assert by_u.loc["D_down", "hit_frac"] == pytest.approx(0.8)
    assert bool(by_u.loc["D_down", "flagged"]) is True
    assert by_u.loc["D_hi", "hit_frac"] == 0.0 and bool(by_u.loc["D_hi", "flagged"]) is False
    assert by_u.loc["D_far", "hit_frac"] == 0.0

    # Treated back-trajectory upwind screening
    ep_tr, _ = _make_corridor_endpoints(12)
    donors_tr = pd.DataFrame(
        {"unit": ["U_north", "U_west"], "lat": [52.7, 51.5], "lon": [0.0, -5.0]}
    )
    up = screen_donor_transport(ep_tr, 51.5, 0.0, donors_tr, mode="treated_back", radius_km=40.0)
    up_u = up.set_index("unit")
    assert up_u.loc["U_north", "hit_frac"] > 0.25
    assert up_u.loc["U_west", "hit_frac"] == 0.0


class _IdentityModel:
    backend = "flaml"
    feature_names_in_ = np.array(["ws", "traj_f"])
    feature_importances_ = np.ones(2)

    def predict(self, X):
        # Returns sum of encoded values so any cross-month leakage jumps by >= 900!
        return X["ws"].to_numpy(dtype=float) + X["traj_f"].to_numpy(dtype=float)


def test_stratified_normalise_and_decom_met_prevent_cross_season_swaps(tmp_path):
    """Verify stratify_by keeps resampled local/transport rows strictly within stratum
    and stays bit-identical between memory_save=False and memory_save=True."""
    dates = pd.DatetimeIndex(
        list(pd.date_range("2023-01-01", periods=30, freq="D"))
        + list(pd.date_range("2023-07-01", periods=30, freq="D"))
    )
    # Encode month cleanly inside ws and traj_f so we can prove zero cross-month leakage!
    months = dates.month.to_numpy(dtype=float)
    df = pd.DataFrame(
        {
            "date": dates,
            "value": 10.0 + months + 0.1 * np.arange(len(dates)),
            "ws": months * 100.0 + np.arange(len(dates)),
            "traj_f": months * 1000.0 + np.arange(len(dates)),
        }
    )

    model = _IdentityModel()
    # Both default and memory_save=True must give bit-identical within-month values
    n_vec = normalise(
        df,
        model,
        covariates=["ws", "traj_f"],
        n_samples=6,
        stratify_by="month",
        resample_pools={"tr": df[["date", "traj_f"]]},
        batch_size=0,
        memory_save=False,
        cache=str(tmp_path / "cache"),
        verbose=False,
    )
    n_mem = normalise(
        df,
        model,
        covariates=["ws", "traj_f"],
        n_samples=6,
        stratify_by="month",
        resample_pools={"tr": df[["date", "traj_f"]]},
        batch_size=0,
        memory_save=True,
        verbose=False,
    )
    pd.testing.assert_frame_equal(n_vec, n_mem)

    # Confirm stratify_by is part of the cache key (same cache dir, stratify_by=None differs)
    n_unstrat = normalise(
        df,
        model,
        covariates=["ws", "traj_f"],
        n_samples=6,
        stratify_by=None,
        resample_pools={"tr": df[["date", "traj_f"]]},
        batch_size=0,
        memory_save=False,
        cache=str(tmp_path / "cache"),
        verbose=False,
    )
    assert not n_vec["normalised"].equals(n_unstrat["normalised"])

    # Jan rows have ws in [100, 129] and traj_f in [1000, 1029] -> sum < 1200.
    # Jul rows have ws in [700, 759] and traj_f in [7000, 7059] -> sum > 7700.
    jan_vals = n_vec.loc["2023-01-01":"2023-01-30", "normalised"]
    jul_vals = n_vec.loc["2023-07-01":"2023-07-30", "normalised"]
    assert (jan_vals < 1200.0).all() and (jul_vals > 7700.0).all()

    # Check decom_met forwards stratify_by and obeys additive decomposition identity
    dec = decom_met(
        df,
        model,
        covariates=["ws", "traj_f"],
        groups={"local": ["ws"], "transport": ["traj_f"]},
        resample_pools={"tr": df[["date", "traj_f"]]},
        stratify_by="month",
        n_samples=6,
        verbose=False,
    )
    assert {"local", "transport", "emi_total", "met_total"} <= set(dec.columns)
    assert np.allclose(dec["observed"] - dec["emi_total"], dec["met_total"])
    assert np.allclose(
        dec["local"] + dec["transport"] + dec["met_base"] + dec["met_noise"],
        dec["met_total"],
    )

    assert hasattr(nm, "wpscf") and hasattr(nm, "invert_source_footprints")


def test_resolve_series_frame_prefers_the_de_weathered_estimate():
    """A frame carrying both observations and a de-weathered column maps the latter."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "observed": [10.0, 11.0, 12.0, 13.0, 14.0],
            "normalised": [1.0] * 5,
            "emi_total": [2.0] * 5,
            "value": [3.0] * 5,
        }
    )
    assert _resolve_series_frame(frame)["_y"].tolist() == [1.0] * 5
    assert _resolve_series_frame(frame.drop(columns="normalised"))["_y"].tolist() == [2.0] * 5
    assert (
        _resolve_series_frame(frame.drop(columns=["normalised", "emi_total"]))["_y"].tolist()
        == [3.0] * 5
    )
    assert _resolve_series_frame(frame.drop(columns=["normalised", "emi_total", "value"]))[
        "_y"
    ].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]
    # An explicit column still wins over the preference order.
    assert _resolve_series_frame(frame, value_col="observed")["_y"].tolist() == [
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
    ]


def test_wcwt_uses_the_de_weathered_column_of_a_normalise_frame():
    ep, series = _make_corridor_endpoints(12)
    frame = pd.DataFrame({"date": series["date"], "observed": series["pm25"], "normalised": 1.0})
    from_frame = wcwt(ep, frame, grid_res=1.0, weighting="none")
    normalised = pd.Series(1.0, index=pd.DatetimeIndex(frame["date"]))
    from_series = wcwt(ep, normalised, grid_res=1.0, weighting="none")
    assert np.allclose(from_frame.field, from_series.field, equal_nan=True)


def test_decom_emi_forwards_stratify_by():
    """The default emission path must honour DecomposeConfig.stratify_by."""
    dates = pd.DatetimeIndex(
        list(pd.date_range("2023-01-01", periods=20, freq="D"))
        + list(pd.date_range("2023-07-01", periods=20, freq="D"))
    )
    months = dates.month.to_numpy(dtype=float)
    df = pd.DataFrame(
        {
            "date": dates,
            "value": 10.0 + months,
            "ws": months * 100.0 + np.arange(len(dates)),
            "traj_f": months * 1000.0 + np.arange(len(dates)),
        }
    )
    model = _IdentityModel()
    kw = dict(covariates=["ws", "traj_f"], n_samples=4, verbose=False)
    unstrat = decom_emi(df, model, **kw)
    strat = decom_emi(df, model, stratify_by="month", **kw)
    assert not np.allclose(unstrat["emi_total"], strat["emi_total"])
    # `decompose(method="emission")` is the default route and must forward it too.
    strat_via_decompose = decompose(
        df, model, target="value", method="emission", stratify_by="month", **kw
    )
    pd.testing.assert_frame_equal(strat, strat_via_decompose, check_exact=False)


def test_polissar_weights_are_the_published_tiers():
    """Pin the Polissar/PSCF step weights and their strict boundaries."""
    count = np.array([[0, 1, 2], [5, 10, 20]], dtype=np.int64)
    w = _polissar_weights(count)
    # Non-empty counts are 1, 2, 5, 10, 20 -> mean 7.6.
    assert np.isnan(w[0, 0])
    assert w[0, 1] == w[0, 2] == w[1, 0] == 0.05  # at or below the mean
    assert w[1, 1] == 0.42  # > mean
    assert w[1, 2] == 0.7  # > 1.5 * mean

    # Counts [1, 2, 3] have mean 2: 3 is exactly 1.5 * mean, so the strict
    # boundary keeps it in the 0.42 tier, not the 0.7 one.
    boundary = _polissar_weights(np.array([[1, 2, 3]], dtype=np.int64))
    assert boundary.tolist() == [[0.05, 0.05, 0.42]]


def test_cluster_trajectories_reports_the_clusters_it_found():
    """Identical paths cannot be split: report 1 cluster, not empty ones."""
    times = pd.date_range("2023-01-01", periods=4, freq="D")
    rows = [
        {
            "receptor_date": t,
            "start_lat": 51.5,
            "start_lon": 0.0,
            "start_height": 100.0,
            "traj_id": 1,
            "age_h": float(a),
            "lat": 51.5 + 0.1 * abs(a),
            "lon": 0.1 * abs(a),
            "height": 500.0,
        }
        for t in times
        for a in (0, -3, -6)
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = cluster_trajectories(pd.DataFrame(rows), n_clusters=3)
    assert res.n_clusters == 1
    assert list(res.summary["cluster"]) == [1]
    assert (res.summary["n_trajectories"] == 4).all()
    assert np.isfinite(res.summary["mean_dist_km"]).all()
    assert np.isfinite(res.centroids[["lat", "lon", "height"]].to_numpy()).all()
    assert set(res.labels) == {1}
    assert not [w for w in caught if "empty slice" in str(w.message)]


def test_stratified_conditional_on_handles_a_non_unique_index():
    """A concat of sites (repeated index) must not expand the resample pool."""
    base = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=20, freq="D"),
            "value": np.arange(20.0),
            "ws": np.arange(20.0) * 10.0,
            "traj_f": np.arange(20.0),
        }
    )
    other = base.copy()
    other["date"] = pd.date_range("2023-07-01", periods=20, freq="D")
    df = pd.concat([base, other])  # index 0..19 twice, as a multisite frame
    assert not df.index.is_unique

    out = normalise(
        df,
        _IdentityModel(),
        covariates=["ws", "traj_f"],
        n_samples=3,
        stratify_by="month",
        conditional_on={"ws": lambda v: v >= 0.0},
        batch_size=0,
        verbose=False,
    )
    assert len(out) == 40
    assert np.isfinite(pd.to_numeric(out["normalised"], errors="coerce")).all()


def test_stratified_cache_key_includes_pool_dates(tmp_path):
    """Changing a pool's date mix changes its month strata, so it invalidates the cache."""
    dates = pd.DatetimeIndex(
        list(pd.date_range("2023-01-01", periods=20, freq="D"))
        + list(pd.date_range("2023-07-01", periods=20, freq="D"))
    )
    months = dates.month.to_numpy(dtype=float)
    df = pd.DataFrame(
        {
            "date": dates,
            "value": 10.0 + months,
            "ws": months * 100.0 + np.arange(len(dates)),
            "traj_f": months * 1000.0 + np.arange(len(dates)),
        }
    )
    pool_a = df[["date", "traj_f"]].copy()
    pool_b = pool_a.copy()
    # Same served traj_f values, opposite month composition: only the dates
    # (from which the month strata are derived) differ.
    pool_b["date"] = np.concatenate([dates[20:], dates[:20]]).astype("datetime64[ns]")
    assert pool_a["traj_f"].equals(pool_b["traj_f"])
    assert not pool_a["date"].equals(pool_b["date"])

    cache = str(tmp_path / "cache")
    kw = dict(
        covariates=["ws", "traj_f"],
        n_samples=4,
        stratify_by="month",
        batch_size=0,
        verbose=False,
        cache=cache,
    )
    a = normalise(df, _IdentityModel(), resample_pools={"tr": pool_a}, **kw)
    b_cached = normalise(df, _IdentityModel(), resample_pools={"tr": pool_b}, **kw)
    b_fresh = normalise(
        df, _IdentityModel(), resample_pools={"tr": pool_b}, **{**kw, "cache": None}
    )
    # Pre-fix the second call hit A's entry and returned A's stratified draws.
    pd.testing.assert_frame_equal(b_cached, b_fresh)
    assert not np.allclose(a["normalised"], b_cached["normalised"])
