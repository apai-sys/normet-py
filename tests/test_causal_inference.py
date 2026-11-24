"""Causal-inference and counterfactual methods.

Panel and placebo design, pre-trend tests, uncertainty bands, staggered
adoption, SCM estimators, conformal inference and HonestDiD bounds.
"""

from __future__ import annotations

import importlib.util

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from normet import (
    HarmonicCounterfactual,
    HonestDiDResult,
    conformal_effect_interval,
    conformal_inference,
    donut_hole_sensitivity,
    effect_bands_space,
    honest_did_bounds,
    placebo_in_space,
    placebo_in_time,
    plot_honest_did,
    pre_trend_test,
    prepare_panel,
    run_scm,
    run_scm_staggered,
    scm,
    scm_abadie,
    scm_ascm,
    scm_diagnostics,
    scm_mcnnm,
    screen_donors,
    sdid,
    uncertainty_bands,
)
from normet.causal._common import newey_west_se_mean, newey_west_se_slope

DONORS = ["D1", "D2", "D3", "D4", "D5", "D6"]
pymc_available = importlib.util.find_spec("pymc") is not None
requires_pymc = pytest.mark.skipif(not pymc_available, reason="pymc not installed")


# ---------------------------------------------------------------------------
# Layer 1: Correctness & econometric fixes
# ---------------------------------------------------------------------------
def test_prepare_panel_splits_interpolation_at_cutoff_and_enforces_max_gap():
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    cutoff = "2023-01-11"  # first 10 days pre (values ~10), last 10 post (values ~100)
    rows = []
    for unit in ("T", "D1", "D2", "D3"):
        for d in dates:
            is_post = d >= pd.Timestamp(cutoff)
            val = 100.0 if (unit == "T" and is_post) else (20.0 if is_post else 10.0)
            # Punch a boundary missing value right before cutoff on T
            if unit == "T" and d == pd.Timestamp("2023-01-10"):
                continue
            # Punch a 4-day contiguous outage in D3
            if unit == "D3" and pd.Timestamp("2023-01-04") <= d <= pd.Timestamp("2023-01-07"):
                continue
            rows.append({"date": d, "ID": unit, "value": val})
    df = pd.DataFrame(rows)

    clean = prepare_panel(
        df,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        cutoff_date=cutoff,
        treated_unit="T",
        min_coverage=0.5,
        split_interpolation=True,
    )
    t_pre_edge = clean.loc[
        (clean["ID"] == "T") & (clean["date"] == pd.Timestamp("2023-01-10")), "value"
    ].iloc[0]
    assert t_pre_edge == pytest.approx(10.0, abs=1e-6)
    assert clean.attrs["split_interpolation"] is True

    # Legacy unsplit interpolation lets post=100 bleed back into the pre-period boundary
    leaked = prepare_panel(
        df,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        cutoff_date=cutoff,
        treated_unit="T",
        min_coverage=0.5,
        split_interpolation=False,
    )
    t_pre_leaked = leaked.loc[
        (leaked["ID"] == "T") & (leaked["date"] == pd.Timestamp("2023-01-10")), "value"
    ].iloc[0]
    assert t_pre_leaked > 40.0

    # max_gap=2 refuses D3's 4-day contiguous outage (drops D3 instead of filling)
    capped = prepare_panel(
        df,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        cutoff_date=cutoff,
        treated_unit="T",
        min_coverage=0.5,
        max_gap=2,
    )
    assert "D3" not in set(capped["ID"])
    assert {"T", "D1", "D2"} == set(capped["ID"])


def test_placebo_in_time_truncates_post_cutoff_and_prevents_leakage(scm_panel, monkeypatch):
    import normet.causal.placebo as plc_mod

    cutoff = "2023-05-01"
    cutoff_ts = pd.Timestamp(cutoff)
    seen_max_dates: list[pd.Timestamp] = []
    real_run = plc_mod.run_scm

    def _spy(**kw):
        if kw["cutoff_date"] != cutoff:
            seen_max_dates.append(pd.to_datetime(kw["df"][kw["date_col"]]).max())
        return real_run(**kw)

    monkeypatch.setattr(plc_mod, "run_scm", _spy)

    out1 = placebo_in_time(
        scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date=cutoff,
        donors=DONORS,
        scm_backend="mcnnm",
        min_pre_period=30,
        placebo_every=14,
        n_cores=1,
    )
    assert seen_max_dates
    assert all(d < cutoff_ts for d in seen_max_dates)

    # Injecting massive post-cutoff contamination on donors must NOT alter any placebo-in-time path
    contaminated = scm_panel.copy()
    post_mask = pd.to_datetime(contaminated["date"]) >= cutoff_ts
    contaminated.loc[post_mask, "value"] += 5000.0
    out2 = placebo_in_time(
        contaminated,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date=cutoff,
        donors=DONORS,
        scm_backend="mcnnm",
        min_pre_period=30,
        placebo_every=14,
        n_cores=1,
    )
    pd.testing.assert_series_equal(out1["placebo_stats"], out2["placebo_stats"])


def test_placebo_in_space_pre_mspe_ratio_filtering(scm_panel):
    rng = np.random.default_rng(123)
    noisy = scm_panel.copy()
    is_d6 = (noisy["ID"] == "D6") & (pd.to_datetime(noisy["date"]) < pd.Timestamp("2023-05-01"))
    noisy.loc[is_d6, "value"] += rng.normal(0.0, 25.0, int(is_d6.sum()))

    base_out = placebo_in_space(
        noisy,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=DONORS,
        scm_backend="abadie",
        n_cores=1,
    )
    assert set(base_out.keys()) == {"treated", "placebos", "p_value", "ref_band"}

    filt = placebo_in_space(
        noisy,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=DONORS,
        scm_backend="abadie",
        max_pre_mspe_ratio=5.0,
        n_cores=1,
    )
    assert "D6" in filt["dropped_placebos"]
    assert "D6" not in filt["placebos"]
    assert filt["pre_mspe_ratios"]["D6"] > 5.0


def test_pre_trend_hac_matches_statsmodels_and_controls_autocorrelation():
    import statsmodels.api as sm

    rng = np.random.default_rng(7)
    n = 120
    # Generate strongly autocorrelated AR(1) residuals with zero true trend
    rho = 0.85
    e = np.empty(n, dtype=float)
    e[0] = rng.normal(0.0, 1.0)
    for t in range(1, n):
        e[t] = rho * e[t - 1] + rng.normal(0.0, 0.5)

    # 1. Exact parity against statsmodels OLS HAC (use_correction=True)
    idx = pd.date_range("2023-01-01", periods=n + 5, freq="D")
    frame = pd.DataFrame(
        {"observed": np.r_[e, np.zeros(5)], "synthetic": 0.0},
        index=idx,
    )
    cutoff_s = idx[n].strftime("%Y-%m-%d")
    pt_hac = pre_trend_test(frame, cutoff_s, se_type="hac", maxlags=4)
    pt_ols = pre_trend_test(frame, cutoff_s, se_type="ols")

    X_sm = sm.add_constant(np.arange(n, dtype=float))
    res_sm = sm.OLS(e, X_sm).fit(
        cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": True}, use_t=True
    )
    assert pt_hac.slope == pytest.approx(float(res_sm.params[1]), rel=1e-12)
    assert pt_hac.se_slope == pytest.approx(float(res_sm.bse[1]), rel=1e-10)
    assert pt_hac.p_slope == pytest.approx(float(res_sm.pvalues[1]), rel=1e-8)
    assert pt_hac.se_slope > pt_ols.se_slope * 1.4

    # 2. Intercept mean HAC parity
    se_mean, _ = newey_west_se_mean(e, maxlags=4)
    res_mean = sm.OLS(e, np.ones((n, 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": True}
    )
    assert se_mean == pytest.approx(float(res_mean.bse[0]), rel=1e-10)

    # 3. Empirical false-positive check over 25 AR(1) null draws
    rej_ols = 0
    rej_hac = 0
    for seed_i in range(25):
        r_i = np.random.default_rng(100 + seed_i)
        z = np.empty(n, dtype=float)
        z[0] = r_i.normal(0.0, 1.0)
        for t in range(1, n):
            z[t] = 0.85 * z[t - 1] + r_i.normal(0.0, 0.5)
        f_i = pd.DataFrame({"observed": np.r_[z, [0.0]], "synthetic": 0.0}, index=idx[: n + 1])
        if pre_trend_test(f_i, idx[n].strftime("%Y-%m-%d"), se_type="ols").p_slope < 0.05:
            rej_ols += 1
        if pre_trend_test(f_i, idx[n].strftime("%Y-%m-%d"), se_type="hac").p_slope < 0.05:
            rej_hac += 1
    assert rej_hac < rej_ols


def test_effect_bands_space_pivotal_inversion_direction():
    idx = pd.date_range("2023-01-01", periods=4, freq="D")
    treated = pd.DataFrame({"effect": np.full(4, 10.0)}, index=idx)
    # Positively biased placebos centred around +3.0 => true tau = effect - placebo ~ 7.0
    placebos = {
        "D1": pd.DataFrame({"effect": np.full(4, 2.0)}, index=idx),
        "D2": pd.DataFrame({"effect": np.full(4, 3.0)}, index=idx),
        "D3": pd.DataFrame({"effect": np.full(4, 4.0)}, index=idx),
    }
    q_bands = effect_bands_space({"treated": treated, "placebos": placebos}, level=0.90)
    assert (q_bands["lower"] < q_bands["upper"]).all()
    assert float(q_bands["upper"].iloc[0]) < 8.5  # shifted down from 10 by -q_low ~ -2.1
    assert float(q_bands["lower"].iloc[0]) == pytest.approx(
        10.0 - float(q_bands["plc_q_high"].iloc[0])
    )

    s_bands = effect_bands_space(
        {"treated": treated, "placebos": placebos}, level=0.95, method="std"
    )
    assert float(s_bands["lower"].iloc[0]) < 7.0 < float(s_bands["upper"].iloc[0])
    assert float(0.5 * (s_bands["lower"].iloc[0] + s_bands["upper"].iloc[0])) == pytest.approx(7.0)


def test_uncertainty_bands_bootstrap_rescaling_replace_and_mbb(scm_panel):
    # Replace bootstrap on the minimum three-donor pool must not collapse: every
    # draw contains all three donors, so the auto moving-block fallback supplies
    # the uncertainty.
    out_rep = uncertainty_bands(
        scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=DONORS[:3],
        scm_backend="abadie",
        method="bootstrap",
        bootstrap_mode="replace",
        B=15,
        random_state=42,
        n_cores=1,
    )
    assert out_rep["low"] is not None
    width_rep = (out_rep["high"] - out_rep["low"]).dropna()
    assert float(width_rep.max()) > 1e-6

    # Subsample rescaling at donor_frac=0.8 gives non-shrunk bands & moving-block works
    out_mbb = uncertainty_bands(
        scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=DONORS,
        scm_backend="abadie",
        method="bootstrap",
        donor_frac=1.0,
        time_block_days=10,
        B=12,
        random_state=42,
        n_cores=1,
    )
    assert (out_mbb["low"] <= out_mbb["high"] + 1e-9).all()
    post_mask = out_mbb["low"].index >= pd.Timestamp("2023-05-01")
    assert (out_mbb["high"][post_mask] > out_mbb["low"][post_mask]).any()


def test_uncertainty_bands_three_donor_auto_time_block_is_not_degenerate(scm_panel):
    # Three is the accepted minimum; the documented fallback forces a 7-day
    # moving-block resample of the pre-period, so the bands must be genuinely
    # wider than zero and must match an explicit time_block_days=7 request.
    kw = {
        "date_col": "date",
        "unit_col": "ID",
        "outcome_col": "value",
        "treated_unit": "T",
        "cutoff_date": "2023-05-01",
        "donors": DONORS[:3],
        "scm_backend": "abadie",
        "method": "bootstrap",
        "B": 12,
        "random_state": 7,
        "n_cores": 1,
    }
    auto = uncertainty_bands(scm_panel, **kw)
    explicit = uncertainty_bands(scm_panel, **kw, time_block_days=7)
    pd.testing.assert_series_equal(auto["low"], explicit["low"])
    pd.testing.assert_series_equal(auto["high"], explicit["high"])
    width = (auto["high"] - auto["low"]).dropna()
    assert float(width.max()) > 1e-6

    # Explicit replace mode on three donors uses the same fallback.
    replaced = uncertainty_bands(scm_panel, **kw, bootstrap_mode="replace")
    width_rep = (replaced["high"] - replaced["low"]).dropna()
    assert float(width_rep.max()) > 1e-6


def test_uncertainty_bands_subsample_scale_applies_with_time_blocks(scm_panel):
    # The finite-population sqrt(k / (J - k)) rescaling is part of the
    # documented contract and must not be dropped when time blocks are used.
    kw = {
        "date_col": "date",
        "unit_col": "ID",
        "outcome_col": "value",
        "treated_unit": "T",
        "cutoff_date": "2023-05-01",
        "donors": DONORS,
        "scm_backend": "abadie",
        "method": "bootstrap",
        "time_block_days": 10,
        "B": 15,
        "random_state": 42,
        "n_cores": 1,
    }
    sub = uncertainty_bands(scm_panel, **kw, donor_frac=0.8)
    full = uncertainty_bands(scm_panel, **kw, donor_frac=1.0)
    w_sub = float((sub["high"] - sub["low"]).dropna().mean())
    w_full = float((full["high"] - full["low"]).dropna().mean())
    assert w_sub > w_full


def test_run_scm_staggered_not_yet_treated_horizon_and_event_study():
    dates = pd.date_range("2023-01-01", periods=160, freq="D")
    t = np.arange(len(dates))
    rng = np.random.default_rng(9)
    donors = ["D1", "D2", "D3", "D4", "D5"]
    base = {
        d: 15.0 + 4.0 * np.sin(2 * np.pi * t / (25 + 3 * i)) + rng.normal(0, 0.2, len(t))
        for i, d in enumerate(donors)
    }
    common = 0.5 * base["D1"] + 0.5 * base["D2"]
    series = {
        "T1": common + 6.0 * (t >= 40),
        "T2": common + 6.0 * (t >= 70),
        "T3": common + 6.0 * (t >= 120),
        **base,
    }
    rows = [
        {"date": d, "ID": u, "value": float(vals[i])}
        for u, vals in series.items()
        for i, d in enumerate(dates)
    ]
    df = pd.DataFrame(rows)

    res = run_scm_staggered(
        df,
        treated_units=["T1", "T2", "T3"],
        cutoff_dates={"T1": "2023-02-10", "T2": "2023-03-12", "T3": "2023-05-01"},
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        scm_backend="abadie",
        include_not_yet_treated=True,
        horizon=20,
        max_permutations=200,
        n_cores=1,
    )
    by_u = res.set_index("unit")
    assert by_u.loc["T1", "n_post"] == 20
    assert abs(by_u.loc["mean", "effect"] - 6.0) < 0.5
    assert by_u.loc["mean", "p_value"] < 0.05

    es = res.attrs["event_study"]
    assert es.index.name == "event_time"
    assert {"att", "se", "ci_low", "ci_high", "p_value", "n_units"} <= set(es.columns)
    assert abs(float(es.loc[es.index < 0, "att"].mean())) < 0.5
    assert abs(float(es.loc[es.index >= 0, "att"].mean()) - 6.0) < 0.5


def test_run_scm_staggered_placebo_uses_the_real_slot_donor_pool(monkeypatch):
    # With include_not_yet_treated=True the real fit admits later-treated units
    # as donors; each placebo must use that same pool minus the placebo unit,
    # otherwise the permutation null belongs to a different estimator.
    import importlib

    dates = pd.date_range("2023-01-01", periods=160, freq="D")
    t = np.arange(len(dates))
    rng = np.random.default_rng(9)
    donors = ["D1", "D2", "D3", "D4", "D5"]
    base = {
        d: 15.0 + 4.0 * np.sin(2 * np.pi * t / (25 + 3 * i)) + rng.normal(0, 0.2, len(t))
        for i, d in enumerate(donors)
    }
    common = 0.5 * base["D1"] + 0.5 * base["D2"]
    series = {
        "T1": common + 6.0 * (t >= 40),
        "T2": common + 6.0 * (t >= 70),
        "T3": common + 6.0 * (t >= 120),
        **base,
    }
    df = pd.DataFrame(
        [
            {"date": d, "ID": u, "value": float(vals[i])}
            for u, vals in series.items()
            for i, d in enumerate(dates)
        ]
    )
    cutoffs = {"T1": "2023-02-10", "T2": "2023-03-12", "T3": "2023-05-01"}

    batch_mod = importlib.import_module("normet.causal.batch")
    real_run = batch_mod.run_scm
    calls: list[tuple[str, str, tuple[str, ...]]] = []

    def spy(**kwargs):
        calls.append(
            (
                str(kwargs["treated_unit"]),
                str(kwargs["cutoff_date"]),
                tuple(kwargs["donors"]),
            )
        )
        return real_run(**kwargs)

    monkeypatch.setattr(batch_mod, "run_scm", spy)
    run_scm_staggered(
        df,
        treated_units=["T1", "T2", "T3"],
        cutoff_dates=cutoffs,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        scm_backend="abadie",
        include_not_yet_treated=True,
        horizon=20,
        max_permutations=50,
        n_cores=1,
    )
    real_pools = {c: pool for unit, c, pool in calls if unit in cutoffs}
    assert {"T2", "T3"} <= set(real_pools["2023-02-10"])
    assert "T3" in set(real_pools["2023-03-12"])
    placebos = [(unit, c, pool) for unit, c, pool in calls if unit not in cutoffs]
    assert placebos
    for unit, c, pool in placebos:
        assert set(pool) == set(real_pools[c]) - {unit}


def test_staggered_event_time_uses_calendar_periods_across_shared_gap():
    # A ten-day gap shared by every unit must advance event time by ten
    # periods, not consume one row. Both units then keep the same calendar
    # alignment (event e is e days after that unit's own cutoff), and the
    # horizon counts periods rather than observed rows.
    from normet.causal.batch import _panel_step_days, _rel_event_series

    dates = pd.date_range("2023-01-01", "2023-01-31", freq="D").append(
        pd.date_range("2023-02-11", "2023-02-28", freq="D")
    )
    step = _panel_step_days(pd.Series(dates))
    assert step == pytest.approx(1.0)
    frame = pd.DataFrame({"effect": np.arange(len(dates), dtype=float)}, index=dates)
    for cutoff in ("2023-01-26", "2023-01-28"):
        rel = _rel_event_series(frame, pd.Timestamp(cutoff), step)
        lags = (dates - pd.Timestamp(cutoff)).days.to_numpy()
        assert np.array_equal(rel.index.to_numpy(), lags)

    t = np.arange(len(dates))
    donors = ["D1", "D2", "D3", "D4"]
    base = {
        d: 5.0 + j + 3.0 * np.sin(2 * np.pi * t / (30 - 4 * j) + j) for j, d in enumerate(donors)
    }
    common = 0.5 * base["D1"] + 0.5 * base["D2"]
    series = {
        "T1": common + 5.0 * (dates >= pd.Timestamp("2023-01-26")),
        "T2": common + 5.0 * (dates >= pd.Timestamp("2023-01-28")),
        **base,
    }
    df = pd.DataFrame(
        [
            {"date": d, "ID": u, "value": float(vals[i])}
            for u, vals in series.items()
            for i, d in enumerate(dates)
        ]
    )
    kw = {
        "treated_units": ["T1", "T2"],
        "cutoff_dates": {"T1": "2023-01-26", "T2": "2023-01-28"},
        "date_col": "date",
        "unit_col": "ID",
        "outcome_col": "value",
        "donors": donors,
        "scm_backend": "abadie",
        "max_permutations": 20,
        "n_cores": 1,
    }
    full = run_scm_staggered(df, **kw)
    labels = set(full.attrs["event_study"].index)
    assert {6, 7, 8, 9, 10, 11, 12, 13}.isdisjoint(labels)
    assert {14, 15, 16} <= labels

    horizon = run_scm_staggered(df, horizon=5, **kw).set_index("unit")
    assert horizon.loc["T1", "n_post"] == 5  # Jan 26-30
    assert horizon.loc["T2", "n_post"] == 4  # Jan 28-31; Feb 2 is inside the gap


# ---------------------------------------------------------------------------
# Layer 2: Top-journal estimators & conformal / sensitivity inference
# ---------------------------------------------------------------------------
def test_scm_ascm_recovers_effect_avoids_noise_leakage_and_demean(scm_panel):
    res = scm_ascm(
        scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=DONORS,
    )
    assert abs(float(res["weights"].sum()) - 1.0) < 1e-8
    post = res["synthetic"].loc[res["synthetic"].index >= pd.Timestamp("2023-05-01"), "effect"]
    assert 8.5 < float(post.mean()) < 11.5
    assert res["l2_imbalance_aug"] <= res["l2_imbalance_base"] + 1e-9

    # Routed through run_scm + pre_covariates works
    df_cov = scm_panel.copy()
    df_cov["cov1"] = df_cov["ID"].map({u: float(i) for i, u in enumerate(["T", *DONORS])})
    syn_via_run = run_scm(
        df=df_cov,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=DONORS,
        scm_backend="ascm",
        pre_covariates=["cov1"],
    )
    assert (
        8.0
        < float(syn_via_run.loc[syn_via_run.index >= pd.Timestamp("2023-05-01"), "effect"].mean())
        < 12.0
    )

    # Leak-free check on 40 random-walk donors vs pure white-noise treated unit (T_pre=12)
    rng = np.random.default_rng(3)
    dates = pd.date_range("2020-01-01", periods=18, freq="MS")
    noise_rows = [
        (d, f"D{j}", v)
        for j in range(40)
        for d, v in zip(dates, 50.0 + np.cumsum(rng.normal(0.0, 3.0, len(dates))), strict=True)
    ]
    noise_rows += [
        (d, "T", v) for d, v in zip(dates, rng.normal(50.0, 5.0, len(dates)), strict=True)
    ]
    noise_df = pd.DataFrame(noise_rows, columns=["date", "ID", "value"])
    noise_donors = [f"D{j}" for j in range(40)]

    legacy_r2 = scm_diagnostics(
        scm(
            noise_df,
            date_col="date",
            unit_col="ID",
            outcome_col="value",
            treated_unit="T",
            cutoff_date="2021-01-01",
            donors=noise_donors,
        ),
        cutoff_date="2021-01-01",
    )["pre_r2"]
    ascm_r2 = scm_diagnostics(
        scm_ascm(
            noise_df,
            date_col="date",
            unit_col="ID",
            outcome_col="value",
            treated_unit="T",
            cutoff_date="2021-01-01",
            donors=noise_donors,
        ),
        cutoff_date="2021-01-01",
    )["pre_r2"]
    assert legacy_r2 > 0.9
    assert ascm_r2 < 0.7

    # Level-shifted treated unit outside convex hull (+50) -> demean=True recovers +10 shock
    shifted = scm_panel.copy()
    shifted.loc[shifted["ID"] == "T", "value"] += 50.0
    res_dm = scm_ascm(
        shifted,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=DONORS,
        demean=True,
    )
    post_dm = res_dm["synthetic"].loc[
        res_dm["synthetic"].index >= pd.Timestamp("2023-05-01"), "effect"
    ]
    assert res_dm["intercept"] == pytest.approx(50.0, abs=3.0)
    assert 8.5 < float(post_dm.mean()) < 11.5


def test_sdid_exact_identities_level_shift_and_placebo_se(scm_panel):
    # Treated outside donor convex hull (+40 level shift) with non-stationary common drift
    shifted = scm_panel.copy()
    dates_s = pd.to_datetime(shifted["date"])
    day_idx = (dates_s - dates_s.min()).dt.days.to_numpy(dtype=float)
    shifted["value"] = shifted["value"] + 0.08 * day_idx
    shifted.loc[shifted["ID"] == "T", "value"] += 40.0

    res = sdid(
        shifted,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=DONORS,
        se_method="placebo",
    )
    w = res["weights"]
    lam = res["time_weights"]
    syn = res["synthetic"]
    assert abs(float(w.sum()) - 1.0) < 1e-8 and (w >= -1e-9).all()
    assert abs(float(lam.sum()) - 1.0) < 1e-8 and (lam >= -1e-9).all()

    pre_eff = syn.loc[lam.index, "effect"].to_numpy(dtype=float)
    assert float(np.dot(lam.to_numpy(), pre_eff)) == pytest.approx(0.0, abs=1e-9)

    post_eff = syn.loc[syn.index >= pd.Timestamp("2023-05-01"), "effect"]
    assert res["att"] == pytest.approx(float(post_eff.mean()), rel=1e-12)
    assert 8.5 < res["att"] < 11.5
    assert np.isfinite(res["att_se"]) and res["att_se"] > 0.0
    assert res["p_value"] < 0.25

    via_run = run_scm(
        df=shifted,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=DONORS,
        scm_backend="sdid",
    )
    assert (
        8.5
        < float(via_run.loc[via_run.index >= pd.Timestamp("2023-05-01"), "effect"].mean())
        < 11.5
    )


def test_scm_abadie_demean_ridge_elastic_net_and_mcnnm_block_cv(scm_panel):
    # Demeaned Abadie handles +50 level shift cleanly
    shifted = scm_panel.copy()
    shifted.loc[shifted["ID"] == "T", "value"] += 50.0
    out_dm = scm_abadie(
        shifted,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=DONORS,
        demean=True,
    )
    assert out_dm["intercept"] == pytest.approx(50.0, abs=2.0)
    post_dm = out_dm["synthetic"].loc[
        out_dm["synthetic"].index >= pd.Timestamp("2023-05-01"), "effect"
    ]
    assert 8.5 < float(post_dm.mean()) < 11.5

    # L2 penalty spreads weights over collinear donors (lowers HHI)
    w_unreg = scm_abadie(
        scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=DONORS,
    )["weights"]
    w_reg = scm_abadie(
        scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=DONORS,
        l2_penalty=1e4,
    )["weights"]
    assert float(np.sum(w_reg**2)) < float(np.sum(w_unreg**2))

    # Elastic Net (allow_negative + l1 + l2, unconstrained intercept) recovers shock
    out_en = scm_abadie(
        scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=DONORS,
        allow_negative_weights=True,
        l1_penalty=1.0,
        l2_penalty=0.5,
        sum_to_one=False,
    )
    post_en = out_en["synthetic"].loc[
        out_en["synthetic"].index >= pd.Timestamp("2023-05-01"), "effect"
    ]
    assert 7.0 < float(post_en.mean()) < 13.0

    # MC-NNM block CV selects valid lambda and recovers +10 shock
    out_mc = scm_mcnnm(
        scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=DONORS,
        cv=3,
        cv_strategy="block",
        block_len=10,
        max_iter=100,
        random_state=0,
    )
    assert out_mc["rank_lambda"] > 0
    post_mc = out_mc["synthetic"].loc[
        out_mc["synthetic"].index >= pd.Timestamp("2023-05-01"), "effect"
    ]
    assert 7.0 < float(post_mc.mean()) < 13.0


@pytest.mark.parametrize("backend", ["ascm", "abadie", "sdid"])
def test_conformal_inference_cwz_refitting_across_backends(scm_panel, backend):
    ci = conformal_inference(
        scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=DONORS,
        scm_backend=backend,
        n_perm=40,
        grid_points=13,
    )
    assert 8.0 < ci["att"] < 12.0
    assert ci["low"] <= ci["att"] <= ci["high"]
    assert ci["p_value"] < 0.05
    assert ci["pointwise"] is not None
    assert {"effect", "lower", "upper"} <= set(ci["pointwise"].columns)

    # Null panel (remove +10 shock) should not reject tau = 0
    null_panel = scm_panel.copy()
    is_t_post = (null_panel["ID"] == "T") & (
        pd.to_datetime(null_panel["date"]) >= pd.Timestamp("2023-05-01")
    )
    null_panel.loc[is_t_post, "value"] -= 10.0
    ci_null = conformal_inference(
        null_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=DONORS,
        scm_backend=backend,
        n_perm=40,
        grid_points=13,
    )
    assert ci_null["p_value"] > 0.2
    assert ci_null["low"] <= 0.0 <= ci_null["high"]


def test_conformal_inference_refits_with_tau_shifted_outcomes(scm_panel, monkeypatch):
    # ``refit: True`` is a literal in the result dict, so it cannot detect a
    # broken refit path. Spy on the backend instead: each candidate tau must
    # reach run_scm as a different treated post-outcome vector on the symmetric
    # full-horizon cutoff.
    import importlib

    run_scm_mod = importlib.import_module("normet.causal.run_scm")
    real_run = run_scm_mod.run_scm
    seen: list[tuple[str, np.ndarray]] = []

    def spy(**kwargs):
        df_arg = kwargs["df"]
        is_post = (df_arg["ID"] == "T") & (
            pd.to_datetime(df_arg["date"]) >= pd.Timestamp("2023-05-01")
        )
        seen.append(
            (str(kwargs["cutoff_date"]), df_arg.loc[is_post, "value"].to_numpy(dtype=float))
        )
        return real_run(**kwargs)

    monkeypatch.setattr(run_scm_mod, "run_scm", spy)
    conformal_inference(
        scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=DONORS,
        scm_backend="scm",
        n_perm=12,
        grid_points=11,
    )
    original = [v for c, v in seen if c == "2023-05-01"]
    refits = [v for c, v in seen if c != "2023-05-01"]
    assert original and len(refits) >= 2
    assert any(not np.allclose(v, original[0]) for v in refits)


def test_conformal_inference_sdid_treats_none_zeta_omega_as_auto(scm_panel):
    # zeta_omega=None is sdid's documented default ("auto"); forwarding it must
    # give the same fit as omitting it, not float(None).
    kw = {
        "date_col": "date",
        "unit_col": "ID",
        "outcome_col": "value",
        "treated_unit": "T",
        "cutoff_date": "2023-05-01",
        "donors": DONORS,
        "scm_backend": "sdid",
        "n_perm": 8,
        "grid_points": 9,
    }
    absent = conformal_inference(scm_panel, **kw)
    explicit = conformal_inference(scm_panel, **kw, zeta_omega=None)
    assert explicit["att"] == pytest.approx(absent["att"], rel=1e-12)
    assert explicit["low"] == pytest.approx(absent["low"], rel=1e-12)
    assert explicit["high"] == pytest.approx(absent["high"], rel=1e-12)
    assert explicit["p_value"] == pytest.approx(absent["p_value"], rel=1e-12)


def test_honest_did_bounds_all_methods_and_plot(scm_panel):
    res = scm_ascm(
        scm_panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-05-01",
        donors=DONORS,
    )
    for method in ("relative_magnitude", "level_magnitude", "smoothness"):
        hd = honest_did_bounds(res, "2023-05-01", method=method)
        assert isinstance(hd, HonestDiDResult)
        tbl = hd.table
        assert np.all(np.diff(tbl["ci_high"] - tbl["ci_low"]) >= -1e-12)
        assert np.all(tbl["ci_low"] <= tbl["id_low"] + 1e-12)
        assert np.all(tbl["ci_high"] >= tbl["id_high"] - 1e-12)
        assert np.isfinite(hd.breakdown_m) and hd.breakdown_m > 0.0
        assert hd.breakdown_m <= hd.breakdown_m_id + 1e-9

    # Null shock should break down immediately (breakdown_m == 0.0)
    syn_null = res["synthetic"].copy()
    post_idx = syn_null.index >= pd.Timestamp("2023-05-01")
    syn_null.loc[post_idx, "observed"] = syn_null.loc[post_idx, "synthetic"]
    syn_null["effect"] = syn_null["observed"] - syn_null["synthetic"]
    hd_null = honest_did_bounds(syn_null, "2023-05-01")
    assert hd_null.breakdown_m == pytest.approx(0.0)

    hd_rm = honest_did_bounds(res, "2023-05-01", method="relative_magnitude")
    ax = plot_honest_did(hd_rm)
    plt.close(ax.figure)
    assert ax.get_ylabel() == "ATT bound"


# ---------------------------------------------------------------------------
# Layer 3: Domain & probabilistic enhancements
# ---------------------------------------------------------------------------
def test_screen_donors_and_donut_hole_sensitivity():
    rng = np.random.default_rng(11)
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    # Coordinates around treated T (51.5, 0.0):
    # - NEAR is ~11 km North (< 25 km donut)
    # - DW is ~80 km East (clean distance, but incoming trajectories pass right over T!)
    # - D1..D4 are clean distant donors in N, S, W, NW (90-140 km)
    coords = {
        "T": (51.5, 0.0),
        "NEAR": (51.6, 0.0),
        "DW": (51.5, 1.15),
        "D1": (52.4, 0.0),
        "D2": (50.6, 0.0),
        "D3": (51.5, -1.4),
        "D4": (52.2, -1.2),
    }
    common = 20.0 + 3.0 * np.sin(np.linspace(0, 6 * np.pi, len(dates)))
    rows = []
    for u, (la, lo) in coords.items():
        for i, d in enumerate(dates):
            v = common[i] + rng.normal(0.0, 0.2)
            if u == "T" and i >= 40:
                v += 8.0
            if u == "DW" and i >= 40:
                v += 5.0  # spillover distortion if DW isn't screened out
            rows.append({"date": d, "unit": u, "value": v, "lat": la, "lon": lo})
    df = pd.DataFrame(rows)

    # Synthetic back-trajectories arriving at DW and passing directly over T at 300 m inside 1000 m BLH
    ep_rows = []
    for td in pd.date_range("2023-01-05", periods=5, freq="5D"):
        for age, (la, lo) in ((0.0, (51.5, 1.15)), (-6.0, (51.5, 0.02)), (-12.0, (51.5, -0.8))):
            ep_rows.append(
                {
                    "site": "DW",
                    "receptor_date": td,
                    "age_h": age,
                    "lat": la,
                    "lon": lo,
                    "height": 300.0,
                    "blh": 1000.0,
                    "start_height": 100.0,
                    "traj_id": 1,
                }
            )
        # D1..D4 come cleanly from far away
        for u, (la0, lo0) in (
            ("D1", (52.4, 0.0)),
            ("D2", (50.6, 0.0)),
            ("D3", (51.5, -1.4)),
            ("D4", (52.2, -1.2)),
        ):
            for age in (0.0, -6.0, -12.0):
                ep_rows.append(
                    {
                        "site": u,
                        "receptor_date": td,
                        "age_h": age,
                        "lat": la0 + 0.1 * abs(age),
                        "lon": lo0 - 0.1 * abs(age),
                        "height": 300.0,
                        "blh": 1000.0,
                        "start_height": 100.0,
                        "traj_id": 1,
                    }
                )
    ep = pd.DataFrame(ep_rows)

    donors = ["NEAR", "DW", "D1", "D2", "D3", "D4"]
    scr = screen_donors(
        df,
        donors,
        "T",
        unit_col="unit",
        min_distance_km=25.0,
        endpoints=ep,
        transport_mode="donor_back",
        transport_radius_km=40.0,
        max_exposure_frac=0.10,
    )
    assert "within_donut" in scr["reasons"]["NEAR"]
    assert "transport_spillover" in scr["reasons"]["DW"]
    assert scr["kept_donors"] == ["D1", "D2", "D3", "D4"]

    sens = donut_hole_sensitivity(
        df,
        date_col="date",
        unit_col="unit",
        outcome_col="value",
        treated_unit="T",
        cutoff_date="2023-02-10",
        donors=donors,
        radii_km=(0.0, 25.0),
        endpoints=ep,
        transport_radius_km=40.0,
        scm_backend="ascm",
    )
    assert set(sens["spec"]) == {"all", "donut>=25km", "transport_screened"}
    assert "transport_screened" in sens.attrs["fits"]
    att_clean = float(sens.loc[sens["spec"] == "transport_screened", "att"].iloc[0])
    assert 6.5 < att_clean < 9.5


@requires_pymc
def test_bayesian_scm_intercept_ar1_predictive_and_diagnostics():
    rng = np.random.default_rng(14)
    dates = pd.date_range("2023-01-01", periods=35, freq="D")
    t = np.arange(len(dates), dtype=float)
    d1 = 20.0 + 4.0 * np.sin(t / 4.0) + rng.normal(0.0, 0.3, len(dates))
    d2 = 22.0 + 4.0 * np.cos(t / 5.0) + rng.normal(0.0, 0.3, len(dates))
    d3 = 18.0 + 3.0 * np.sin(t / 3.0) + rng.normal(0.0, 0.3, len(dates))
    err = np.empty(len(dates), dtype=float)
    err[0] = rng.normal(0.0, 1.2)
    for i in range(1, len(dates)):
        err[i] = 0.6 * err[i - 1] + rng.normal(0.0, 0.9)
    y_t = 15.0 + 0.5 * d1 + 0.5 * d2 + err + np.where(t >= 25, 8.0, 0.0)
    df = pd.concat(
        [
            pd.DataFrame({"date": dates, "ID": u, "value": arr})
            for u, arr in (("T", y_t), ("D1", d1), ("D2", d2), ("D3", d3))
        ],
        ignore_index=True,
    )

    from normet.causal import bayesian_scm

    res = bayesian_scm(
        df,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        treated_unit="T",
        cutoff_date=dates[25].strftime("%Y-%m-%d"),
        include_intercept=True,
        ar_order=1,
        predictive_band=True,
        draws=40,
        tune=60,
        chains=2,
        n_cores=1,
        progressbar=False,
    )
    diag = res["diagnostics"]
    assert diag["ar_order"] == 1
    assert diag["include_intercept"] is True
    assert diag["predictive_band"] is True
    assert "max_r_hat" in diag and "min_ess_bulk" in diag and "n_divergences" in diag

    syn = res["synthetic"]
    pred_w = (syn["synthetic_high"] - syn["synthetic_low"]).to_numpy()
    mean_w = (syn["synthetic_mean_high"] - syn["synthetic_mean_low"]).to_numpy()
    assert np.nanmean(pred_w) > np.nanmean(mean_w)
    post = syn.loc[syn.index >= dates[25], "effect"]
    assert 3.0 < float(post.mean()) < 13.0


def test_harmonic_counterfactual_val_split_auto_cv_covariates_and_block_bootstrap():
    idx = pd.date_range("2019-01-01", periods=24 * 400, freq="h")
    rng = np.random.default_rng(5)
    # Seasonal baseline temperature + sudden +12C anomaly during intervention (no policy cut!)
    temp = np.asarray(
        10.0
        + 6.0 * np.sin(2.0 * np.pi * idx.dayofyear / 365.25)
        + 3.0 * np.sin(2.0 * np.pi * idx.hour / 24.0)
        + rng.normal(0.0, 1.5, len(idx)),
        dtype=float,
    )
    interv_date = idx[24 * 380]
    temp[idx >= interv_date] += 12.0  # pure meteorological anomaly, ZERO policy intervention
    cov_df = pd.DataFrame({"temp": temp}, index=idx)

    y = (
        25.0
        + 6.0 * np.sin(2.0 * np.pi * idx.hour / 24.0)
        + 4.0 * np.sin(2.0 * np.pi * idx.dayofyear / 365.25)
        + 1.5 * temp
        + rng.normal(0.0, 1.0, len(idx))
    )
    series = pd.Series(y, index=idx)

    # Validation split & auto CV
    train_end = idx[24 * 360]
    hc_fit = HarmonicCounterfactual(alpha_ridge="auto", cv_folds=3)
    hc_fit.fit(
        series[idx <= train_end],
        validation_split_date=idx[24 * 300],
        covariates=cov_df.loc[idx <= train_end],
    )
    assert hc_fit.cv_scores_ is not None and np.isfinite(hc_fit.alpha_selected_)
    assert hc_fit.validation_bias_pct_ is not None and abs(hc_fit.validation_bias_pct_) < 5.0
    assert hc_fit.validation_rmse_ is not None and hc_fit.validation_rmse_ < 2.5

    # Without covariates: calendar model attributes +12C*1.5 = +18 weather anomaly to intervention!
    res_cal = HarmonicCounterfactual(alpha_ridge="auto", cv_folds=3).evaluate_intervention(
        series,
        train_end_date=train_end,
        intervention_start_date=interv_date,
    )
    assert abs(res_cal.summary_metrics["net_impact_ugm3"]) > 10.0

    # With covariates + moving-block bootstrap: net impact recovers ~0 and provides bootstrap SE
    res_cov = HarmonicCounterfactual(alpha_ridge="auto", cv_folds=3).evaluate_intervention(
        series,
        train_end_date=train_end,
        intervention_start_date=interv_date,
        covariates=cov_df,
        uncertainty="block_bootstrap",
        n_boot=25,
        block_hours=24,
    )
    assert abs(res_cov.summary_metrics["net_impact_ugm3"]) < 1.5
    assert "net_impact_se" in res_cov.summary_metrics
    assert "net_impact_hac_se" in res_cov.summary_metrics
    assert (res_cov.counterfactual_p10 <= res_cov.counterfactual_p90 + 1e-12).all()
