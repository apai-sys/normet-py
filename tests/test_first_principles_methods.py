"""First-principles methods suite: DML, control variates, Markov blocks, ICN upgrades,
photochemical Ox-NOx conservation, fused Shapley coalitions, deweathered/transport SCM,
multi-receptor inversion, and built-in synthetic ground-truth benchmarks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import normet as nm
from normet.analysis.normalise import _draw_rows_block
from normet.foundation import (
    Chronos2Estimator,
    ICNDesign,
    ICNDrawBank,
    build_icn_draw_bank,
    icn_decompose,
)
from normet.foundation.in_context import _aggregate, _design_hash


class _FakeICLPipeline:
    """Torch-free ridge ICL stand-in for testing ICN orchestration and attribution."""

    def forecast(
        self,
        target_history: np.ndarray,
        past: pd.DataFrame,
        future_met: pd.DataFrame,
        cols: list[str],
        target: str,
    ) -> np.ndarray:
        met = [c for c in cols if c in past.columns]
        Xp = past[met].to_numpy(dtype=float)
        yp = target_history.astype(float)
        good = np.isfinite(yp) & np.all(np.isfinite(Xp), axis=1)
        Xp, yp = Xp[good], yp[good]
        if Xp.shape[0] == 0 or Xp.shape[1] == 0:
            return np.full(len(future_met), float(np.nanmean(yp)) if yp.size else np.nan)
        XtX = Xp.T @ Xp + 1e-4 * np.eye(Xp.shape[1])
        w = np.linalg.solve(XtX, Xp.T @ yp)
        intercept = float(yp.mean() - w @ Xp.mean(axis=0))
        Xf = future_met[met].to_numpy(dtype=float)
        return (intercept + Xf @ w).astype(np.float64)


def _fast_lgb_cfg() -> dict[str, Any]:
    return {
        "n_trials": 2,
        "cv_folds": 2,
        "nrounds": 80,
        "early_stopping_rounds": 15,
        "deterministic": True,
    }


def test_benchmarks_all_families_and_evaluate() -> None:
    for fam in (
        "F1_additive",
        "F2_multiplicative",
        "F3_titration",
        "F4_nonstationary",
        "F5_transport",
    ):
        bm = nm.make_synthetic_benchmark(fam, n_days=45, seed=7)
        assert len(bm.df) == 45 * 24
        assert "emission_true" in bm.truth.columns
        assert bm.truth["event_mask"].any()
        perfect = pd.DataFrame(
            {"normalised": bm.truth["emission_true"].to_numpy()},
            index=bm.truth.index,
        )
        scores = nm.evaluate_benchmark(bm, perfect)
        assert scores["nrmse_pct"] < 1e-8
        assert abs(scores["bias_pct"]) < 1e-8
        assert scores["event_recovery_pct"] == pytest.approx(100.0, abs=1e-6)
        assert scores["pearson_r"] == pytest.approx(1.0, abs=1e-6)


def test_dml_normalise_prevents_event_absorption_and_boundary_steps() -> None:
    pytest.importorskip("lightgbm")
    bm = nm.make_synthetic_benchmark(
        "F1_additive",
        n_days=90,
        event_start_day=45,
        event_duration_days=10,
        event_reduction_pct=-35.0,
        trend_pct_per_year=-8.0,
        noise_sd=0.8,
        seed=11,
    )
    res_dml = nm.dml_normalise(
        bm.df,
        target=bm.target,
        covariates=bm.covariates,
        variables_resample=bm.met_covariates,
        backend="lightgbm",
        model_config=_fast_lgb_cfg(),
        cutoff_days=25.0,
        n_splits=3,
        embargo=12,
        n_samples=60,
        paired="additive",
        return_mc_se=True,
        sampling="balanced",
        seed=11,
        n_cores=1,
    )
    assert isinstance(res_dml, nm.DMLNormaliseResult)
    assert len(res_dml.normalised) == len(bm.df)
    assert "mc_se" in res_dml.normalised.columns
    assert "all" in set(res_dml.oof_metrics["fold"])

    # DML removes date_unix from tree splits, preventing boundary step jumps on level
    # and achieving strong event recovery on paired.
    score_level = nm.evaluate_benchmark(bm, res_dml.normalised, col="normalised")
    score_paired = nm.evaluate_benchmark(bm, res_dml.normalised, col="paired")
    assert score_level["wrf_linear"] > 0.70
    assert score_level["nrmse_pct"] < 25.0
    assert score_paired["event_recovery_pct"] > 75.0


def test_control_variates_reduces_mc_se_and_is_unbiased() -> None:
    pytest.importorskip("lightgbm")
    bm = nm.make_synthetic_benchmark("F1_additive", n_days=30, seed=5)
    df_prep, mdl = nm.build_model(
        bm.df,
        target=bm.target,
        backend="lightgbm",
        covariates=bm.covariates,
        model_config=_fast_lgb_cfg(),
        seed=5,
        n_cores=1,
    )
    base_res = nm.normalise(
        df_prep,
        mdl,
        covariates=bm.covariates,
        variables_resample=bm.met_covariates,
        n_samples=40,
        return_mc_se=True,
        control_variates=False,
        seed=123,
        n_cores=1,
    )
    cv_res = nm.normalise(
        df_prep,
        mdl,
        covariates=bm.covariates,
        variables_resample=bm.met_covariates,
        n_samples=40,
        return_mc_se=True,
        control_variates=True,
        seed=123,
        n_cores=1,
    )
    # Control variates must reduce MC standard error substantially while remaining unbiased.
    assert cv_res["mc_se"].median() < 0.85 * base_res["mc_se"].median()
    assert abs(float(cv_res["normalised"].mean() - base_res["normalised"].mean())) < 1.5

    # Vectorised (batch_size=0) and memory_save=True paths must agree with batch_reduce.
    cv_vec = nm.normalise(
        df_prep,
        mdl,
        covariates=bm.covariates,
        variables_resample=bm.met_covariates,
        n_samples=20,
        return_mc_se=True,
        control_variates=True,
        batch_size=0,
        seed=123,
        n_cores=1,
    )
    cv_mem = nm.normalise(
        df_prep,
        mdl,
        covariates=bm.covariates,
        variables_resample=bm.met_covariates,
        n_samples=20,
        return_mc_se=True,
        control_variates=True,
        memory_save=True,
        seed=123,
        n_cores=1,
    )
    assert np.allclose(cv_vec["normalised"].to_numpy(), cv_mem["normalised"].to_numpy(), atol=1e-10)
    assert np.allclose(cv_vec["mc_se"].to_numpy(), cv_mem["mc_se"].to_numpy(), atol=1e-10)

    # The control-variate shift must move the quantile bands too: they are
    # draws of the same adjusted distribution as `normalised`. The old code
    # built them from the raw draws, so the bands stayed centred on the
    # unadjusted level while the central line moved.
    q_base = nm.normalise(
        df_prep,
        mdl,
        covariates=bm.covariates,
        variables_resample=bm.met_covariates,
        n_samples=40,
        return_quantiles=(0.1, 0.5, 0.9),
        control_variates=False,
        seed=123,
        n_cores=1,
    )
    q_cv = nm.normalise(
        df_prep,
        mdl,
        covariates=bm.covariates,
        variables_resample=bm.met_covariates,
        n_samples=40,
        return_quantiles=(0.1, 0.5, 0.9),
        control_variates=True,
        seed=123,
        n_cores=1,
    )
    q50_shifts = np.abs(q_cv["q500"].to_numpy() - q_base["q500"].to_numpy())
    assert float(np.max(q50_shifts)) > 1e-6
    assert (q_cv["q100"].to_numpy() <= q_cv["q500"].to_numpy() + 1e-9).all()
    assert (q_cv["q500"].to_numpy() <= q_cv["q900"].to_numpy() + 1e-9).all()


def test_markov_bridge_block_resampling_reduces_seam_jumps() -> None:
    rng = np.random.default_rng(77)
    n_pool = 1000
    t = np.linspace(0.0, 20.0 * np.pi, n_pool)
    pool_vals = np.column_stack([np.sin(t), np.cos(t)]) + 0.05 * rng.standard_normal((n_pool, 2))
    mu = pool_vals.mean(axis=0)
    sd = pool_vals.std(axis=0)
    z_pool = (pool_vals - mu) / sd

    block_h = 20
    n_rows = 400
    seams = np.arange(block_h, n_rows, block_h)

    idx_ind = _draw_rows_block(0, 42, n_pool, n_rows, block_h, block_mode="independent")
    idx_mkv = _draw_rows_block(0, 42, n_pool, n_rows, block_h, z_pool=z_pool, block_mode="markov")
    jump_ind = np.linalg.norm(z_pool[idx_ind[seams]] - z_pool[idx_ind[seams - 1]], axis=1).mean()
    jump_mkv = np.linalg.norm(z_pool[idx_mkv[seams]] - z_pool[idx_mkv[seams - 1]], axis=1).mean()
    assert jump_mkv < 0.6 * jump_ind


def test_icn_surprise_weighting_persistence_and_v1_hash(tmp_path: Path) -> None:
    # Frozen ICN-v1 hash must not be perturbed by the new default surprise_tau=0.0.
    v1 = ICNDesign.v1()
    v1_no_surp = dict(v1.__dict__)
    v1_no_surp.pop("surprise_tau")
    import hashlib
    import json

    expected_hash = hashlib.sha256(
        json.dumps(v1_no_surp, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    assert _design_hash(v1) == expected_hash

    # Create step event at hour 600 where forward anchors from pre-event predict
    # low level and reverse anchors from post-event predict high level across the edge.
    rng = np.random.default_rng(13)
    n = 1200
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    w = rng.standard_normal(n)
    level_true = np.where(np.arange(n) >= 600, 30.0, 10.0)
    y = level_true + 3.0 * w + 0.05 * rng.standard_normal(n)
    df = pd.DataFrame({"y": y, "w": w}, index=idx)

    est = Chronos2Estimator()
    est.fit(df[["w"]], df["y"])
    design = ICNDesign(
        block_h=100,
        stride=50,
        direction="bidirectional",
        context_length=300,
        aggregate="mean",
        n_resamples=4,
        seed=0,
        surprise_tau=0.0,
    )
    bank = build_icn_draw_bank(est, df, "y", design=design, pipeline=_FakeICLPipeline())

    res_plain = bank.aggregate(surprise_tau=0.0)
    res_surp = bank.aggregate(surprise_tau=0.2)

    # Near the step boundary [550..650], surprise gating suppresses cross-boundary anchors,
    # keeping the step transition sharper (closer to true level).
    edge_slice = slice(550, 650)
    mae_plain = float(np.nanmean(np.abs(res_plain.level.iloc[edge_slice] - level_true[edge_slice])))
    mae_surp = float(np.nanmean(np.abs(res_surp.level.iloc[edge_slice] - level_true[edge_slice])))
    assert mae_surp < mae_plain

    # DrawBank disk persistence round-trip must be bit-identical.
    save_path = tmp_path / "icn_bank.json.gz"
    bank.save(save_path)
    loaded = ICNDrawBank.load(save_path)
    res_loaded = loaded.aggregate(surprise_tau=0.2)
    pd.testing.assert_series_equal(res_surp.level, res_loaded.level)
    pd.testing.assert_series_equal(res_surp.paired, res_loaded.paired)

    # A timezone-aware index must survive the round trip. `index_tz` was
    # written but never read back, so a Europe/London bank silently became
    # fixed-offset/UTC after loading and no longer matched the caller's frame.
    tz_name = "Europe/London"
    bank.index = bank.index.tz_localize(tz_name)
    tz_path = tmp_path / "icn_bank_tz.json.gz"
    bank.save(tz_path)
    loaded_tz = ICNDrawBank.load(tz_path)
    assert str(loaded_tz.index.tz) == tz_name
    pd.testing.assert_index_equal(loaded_tz.index, bank.index)
    pd.testing.assert_series_equal(
        loaded_tz.aggregate(surprise_tau=0.2).level,
        bank.aggregate(surprise_tau=0.2).level,
    )


def test_icn_decompose_groups_and_interactions() -> None:
    rng = np.random.default_rng(29)
    n = 800
    idx = pd.date_range("2022-05-01", periods=n, freq="h")
    w1 = rng.standard_normal(n)
    w2 = rng.standard_normal(n)
    t2m = rng.standard_normal(n)
    y = 20.0 + 4.0 * w1 - 2.5 * w2 + 1.5 * t2m + 0.05 * rng.standard_normal(n)
    df = pd.DataFrame({"y": y, "ws": w1, "blh": w2, "t2m": t2m}, index=idx)

    est = Chronos2Estimator()
    est.fit(df[["ws", "blh", "t2m"]], df["y"])
    design = ICNDesign(
        block_h=80,
        stride=80,
        direction="forward",
        context_length=240,
        aggregate="mean",
        n_resamples=4,
        seed=3,
    )
    groups = {"dynamics": ["ws", "blh"], "thermodynamics": ["t2m"]}
    res_inter = icn_decompose(
        est,
        df,
        "y",
        design=design,
        pipeline=_FakeICLPipeline(),
        groups=groups,
        return_interactions=True,
    )
    dec = res_inter.decomposition
    valid = dec["emi_total"].notna()
    assert valid.any()
    # Exact Shapley closure: emi_total + dynamics + thermodynamics == predicted
    lhs = (
        dec.loc[valid, "emi_total"] + dec.loc[valid, "dynamics"] + dec.loc[valid, "thermodynamics"]
    )
    assert np.allclose(lhs.to_numpy(), dec.loc[valid, "predicted"].to_numpy(), atol=1e-10)
    # Second-order interaction closure: main_effects + pairwise == met_explained
    m_sum = res_inter.main_effects.loc[valid].sum(axis=1)
    p_sum = res_inter.pairwise.loc[valid].sum(axis=1)
    assert np.allclose(
        (m_sum + p_sum).to_numpy(), dec.loc[valid, "met_explained"].to_numpy(), atol=1e-10
    )


def test_normalise_ox_invariant_conservation_and_f3_recovery() -> None:
    pytest.importorskip("lightgbm")
    bm = nm.make_synthetic_benchmark(
        "F3_titration",
        n_days=75,
        event_start_day=35,
        event_duration_days=12,
        event_reduction_pct=-45.0,
        trend_pct_per_year=-6.0,
        seed=19,
    )
    ox_res = nm.normalise_ox_invariant(
        bm.df,
        o3_col="O3",
        no2_col="NO2",
        no_col="NO",
        covariates=bm.covariates,
        variables_resample=bm.met_covariates,
        unit="ug_m3",
        method="dml",
        backend="lightgbm",
        model_config=_fast_lgb_cfg(),
        cutoff_days=20.0,
        n_splits=3,
        embargo=12,
        n_samples=40,
        seed=19,
        n_cores=1,
    )
    assert isinstance(ox_res, nm.ConservedOxResult)
    # Exact stoichiometric molar conservation & non-negativity
    assert ox_res.conservation_max_err < 1e-10
    sp = ox_res.species
    assert (sp["O3_normalised"] >= 0.0).all()
    assert (sp["NO2_normalised"] >= 0.0).all()
    assert (sp["NO_normalised"] >= 0.0).all()

    # During NOx reduction event, titration reallocation should boost deweathered O3
    ev_m = bm.truth["event_mask"].reindex(sp.index).to_numpy(dtype=bool)
    o3_ev = float(sp.loc[ev_m, "O3_normalised"].mean())
    o3_out = float(sp.loc[~ev_m, "O3_normalised"].mean())
    no2_ev = float(sp.loc[ev_m, "NO2_normalised"].mean())
    no2_out = float(sp.loc[~ev_m, "NO2_normalised"].mean())
    assert o3_ev > o3_out
    assert no2_ev < no2_out
    assert float(ox_res.titration_reallocation.loc[ev_m].mean()) > 0.0

    # Regime-conditioned deweathering for O3 conditioned on NOx
    rc_res = nm.normalise_regime_conditioned(
        bm.df,
        target="O3",
        precursors=["NOx"],
        covariates=bm.covariates,
        variables_resample=bm.met_covariates,
        method="dml",
        backend="lightgbm",
        model_config=_fast_lgb_cfg(),
        cutoff_days=20.0,
        n_splits=3,
        embargo=12,
        n_samples=30,
        seed=23,
        n_cores=1,
    )
    assert isinstance(rc_res, nm.RegimeConditionedResult)
    assert "precursor_regime_effect" in rc_res.normalised.columns
    assert "NOx" in rc_res.precursors.columns


def test_ode_memory_physics_features() -> None:
    bm = nm.make_synthetic_benchmark("F1_additive", n_days=10, seed=3)
    base_phys = nm.add_met_physics_features(bm.df)
    mem_phys = nm.add_met_physics_features(bm.df, include_memory=True, residence_tau_hours=12.0)
    assert "stagnation_memory" not in base_phys.columns
    assert {"stagnation_memory", "photochemical_dose", "wet_scavenging_memory"}.issubset(
        mem_phys.columns
    )
    assert np.all(np.isfinite(mem_phys["stagnation_memory"]))
    assert (mem_phys["stagnation_memory"] > 0).all()
    assert (mem_phys["photochemical_dose"] >= 0).all()
    assert (mem_phys["wet_scavenging_memory"] >= 0).all()


def test_fused_shapley_coalitions_bit_identical_to_normalise() -> None:
    pytest.importorskip("lightgbm")
    bm = nm.make_synthetic_benchmark("F1_additive", n_days=20, seed=17)
    df_sub = bm.df[["date", "value", "ws", "blh", "t2m", "hour", "weekday"]].copy()
    df_prep, mdl = nm.build_model(
        df_sub,
        target="value",
        backend="lightgbm",
        covariates=["ws", "blh", "t2m", "hour", "weekday"],
        model_config=_fast_lgb_cfg(),
        seed=17,
        n_cores=1,
    )
    # Fused path (cache=None, memory_save=False) vs un-fused path (memory_save=True)
    fused = nm.decom_met(
        df_prep,
        model=mdl,
        attribution="shapley",
        n_samples=25,
        seed=17,
        n_cores=1,
        memory_save=False,
        include_diagnostics=True,
    )
    unfused = nm.decom_met(
        df_prep,
        model=mdl,
        attribution="shapley",
        n_samples=25,
        seed=17,
        n_cores=1,
        memory_save=True,
        include_diagnostics=True,
    )
    for col in ("emi_total", "ws", "blh", "t2m", "met_explained"):
        assert np.allclose(fused[col].to_numpy(), unfused[col].to_numpy(), atol=1e-10)

    inter = nm.shapley_interactions(
        df_prep,
        model=mdl,
        n_samples=25,
        seed=17,
        n_cores=1,
    )
    lhs = inter.main_effects.sum(axis=1) + inter.pairwise.sum(axis=1)
    assert np.allclose(lhs.to_numpy(), inter.decomposition["met_explained"].to_numpy(), atol=1e-10)


def test_deweathered_scm_and_transport_adjusted_scm() -> None:
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(101)
    n_days = 60
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    cutoff = "2023-02-01"
    is_post = dates >= pd.Timestamp(cutoff)

    ws = np.clip(4.0 + rng.normal(0.0, 1.5, n_days), 0.5, 10.0)
    t2m = 10.0 + 5.0 * np.sin(2.0 * np.pi * np.arange(n_days) / 30.0) + rng.normal(0.0, 1.5, n_days)
    common_shock = 3.0 * np.sin(2.0 * np.pi * np.arange(n_days) / 14.0)

    # 4 units: treated_unit ("T") has -12.0 policy effect post-cutoff.
    # Donor "D1" is intermittently downwind with exposure F_D1(t), suffering spillover -5.0 * F_D1(t).
    # Donors "D2", "D3" are clean upwind donors.
    f_d1 = np.where(is_post, np.clip(rng.uniform(0.0, 1.5, n_days), 0.0, 1.5), 0.0)
    rows: list[pd.DataFrame] = []
    for unit, offset, tr_eff, sp_exp in (
        ("T", 5.0, -12.0, np.zeros(n_days)),
        ("D1", 2.0, 0.0, f_d1),
        ("D2", -1.0, 0.0, np.zeros(n_days)),
        ("D3", 3.0, 0.0, np.zeros(n_days)),
    ):
        met_eff = -2.0 * (ws - 4.0) + 0.5 * (t2m - 10.0)
        y = (
            30.0
            + offset
            + common_shock
            + met_eff
            + np.where(is_post, tr_eff, 0.0)
            + (-5.0 * sp_exp)
            + rng.normal(0.0, 0.4, n_days)
        )
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ID": unit,
                    "value": y,
                    "ws": ws,
                    "t2m": t2m,
                    "spillover": sp_exp,
                }
            )
        )
    panel = pd.concat(rows, ignore_index=True)

    # 1. End-to-end deweathered_scm with uncertainty propagation
    dscm = nm.deweathered_scm(
        panel,
        treated_unit="T",
        cutoff_date=cutoff,
        donors=["D1", "D2", "D3"],
        covariates=["ws", "t2m", "day_julian", "weekday"],
        variables_resample=["ws", "t2m"],
        deweather_method="dml",
        ml_backend="lightgbm",
        model_config=_fast_lgb_cfg(),
        scm_backend="ascm",
        cutoff_days=20.0,
        n_splits=3,
        embargo=2,
        n_samples=30,
        n_scm_draws=40,
        seed=101,
        n_cores=1,
    )
    assert isinstance(dscm, nm.DeweatheredSCMResult)
    assert np.allclose(
        dscm.synthetic["se_total"].to_numpy(),
        np.hypot(dscm.synthetic["se_met"].to_numpy(), dscm.synthetic["se_scm"].to_numpy()),
        atol=1e-10,
    )
    assert dscm.att < -7.0
    assert dscm.att_se_total > 0.0
    assert 0.0 <= dscm.met_variance_share <= 1.0

    # 2. Dynamic transport_adjusted_scm on clean (deweathered) panel with downwind D1 spillover
    panel_clean = panel.copy()
    panel_clean["value"] = (
        panel_clean["value"] + 2.0 * (panel_clean["ws"] - 4.0) - 0.5 * (panel_clean["t2m"] - 10.0)
    )
    tscm = nm.transport_adjusted_scm(
        panel_clean,
        treated_unit="T",
        cutoff_date=cutoff,
        donors=["D1", "D2", "D3"],
        spillover_exposure="spillover",
        scm_backend="ascm",
    )
    assert isinstance(tscm, nm.TransportSCMResult)
    assert tscm.gamma == pytest.approx(-5.0, abs=0.8)
    assert abs(tscm.att_debiased - (-12.0)) < abs(tscm.att_naive - (-12.0))


def test_invert_multisite_footprints() -> None:
    rng = np.random.default_rng(33)
    n_t = 120
    dates = pd.date_range("2023-03-01", periods=n_t, freq="h")
    true_beta = np.array([10.0, 4.0, 0.5])
    regions = ["north", "east", "south"]
    region_coords = {
        "north": (52.5, -1.5),
        "east": (52.0, -0.5),
        "south": (51.0, -1.5),
    }

    fp_by_site: dict[str, pd.DataFrame] = {}
    ser_by_site: dict[str, pd.Series] = {}
    intercepts_true = {"site_A": 12.0, "site_B": 25.0, "site_C": 8.0}

    for site, b0 in intercepts_true.items():
        F = rng.uniform(0.0, 2.0, size=(n_t, 3))
        y = b0 + F @ true_beta + rng.normal(0.0, 0.4, n_t)
        fp_by_site[site] = pd.DataFrame(
            {f"traj_footprint_{r}": F[:, i] for i, r in enumerate(regions)},
            index=dates,
        )
        ser_by_site[site] = pd.Series(y, index=dates, name="normalised")

    res = nm.invert_multisite_footprints(
        fp_by_site,
        ser_by_site,
        region_coords=region_coords,
        alpha=0.1,
        spatial_alpha=0.05,
        non_negative=True,
        n_bootstrap=30,
        seed=33,
    )
    assert isinstance(res, nm.MultisiteInversionResult)
    assert res.n_obs == 3 * n_t
    assert res.r2 > 0.95
    est_beta = res.regions.set_index("region").loc[regions, "beta"].to_numpy()
    assert np.allclose(est_beta, true_beta, atol=0.8)
    for site, b0 in intercepts_true.items():
        assert res.intercepts[site] == pytest.approx(b0, abs=1.2)


def test_ode_memory_is_grouped_and_time_aware() -> None:
    dates = pd.date_range("2023-01-01", periods=6, freq="D")
    rows: list[dict[str, Any]] = []
    for d in dates:
        rows.append({"date": d, "site": "A", "blh": 500.0, "ws": 2.0, "ssrd": 0.0, "tp": 0.0})
        rows.append({"date": d, "site": "B", "blh": 500.0, "ws": 2.0, "ssrd": 100.0, "tp": 0.0})
    # Shuffle so index order cannot accidentally separate the sites.
    panel = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)

    grouped = nm.add_met_physics_features(panel, include_memory=True, group_col="site")
    a = grouped[grouped["site"] == "A"].sort_values("date")
    b = grouped[grouped["site"] == "B"].sort_values("date")
    # Site A sees no radiation; an ungrouped EMA carries B's radiation into it.
    assert np.allclose(a["photochemical_dose"].to_numpy(), 0.0, atol=1e-12)
    assert np.allclose(b["photochemical_dose"].to_numpy(), 100.0, atol=1e-9)

    single_a = nm.add_met_physics_features(
        panel[panel["site"] == "A"].sort_values("date").drop(columns=["site"]),
        include_memory=True,
    )
    assert np.allclose(
        a["stagnation_memory"].to_numpy(), single_a["stagnation_memory"].to_numpy(), atol=1e-12
    )
    assert np.allclose(
        a["wet_scavenging_memory"].to_numpy(),
        single_a["wet_scavenging_memory"].to_numpy(),
        atol=1e-12,
    )

    # Decay is per elapsed hour, not per row: a 100 W/m^2 impulse one day
    # later is exp(-24/6) of it, versus exp(-1/6) for a one-hour step.
    cols = {"blh": 500.0, "ws": 2.0, "tp": 0.0}
    daily = nm.add_met_physics_features(
        pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=4, freq="D"),
                "ssrd": [100.0, 0, 0, 0],
                **cols,
            }
        ),
        include_memory=True,
    )
    hourly = nm.add_met_physics_features(
        pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=4, freq="h"),
                "ssrd": [100.0, 0, 0, 0],
                **cols,
            }
        ),
        include_memory=True,
    )
    assert daily["photochemical_dose"].iloc[1] == pytest.approx(100.0 * np.exp(-4.0), rel=1e-9)
    assert hourly["photochemical_dose"].iloc[1] == pytest.approx(
        100.0 * np.exp(-1.0 / 6.0), rel=1e-9
    )

    with pytest.raises(ValueError, match="not a column"):
        nm.add_met_physics_features(panel, include_memory=True, group_col="nosuch")


def test_dml_back_transform_aggregates_on_original_scale() -> None:
    pytest.importorskip("lightgbm")
    bm = nm.make_synthetic_benchmark("F2_multiplicative", n_days=60, seed=5)
    common: dict[str, Any] = {
        "target": bm.target,
        "covariates": ["ws", "blh", "t2m", "day_julian", "hour"],
        "variables_resample": ["ws", "blh", "t2m"],
        "backend": "lightgbm",
        "model_config": _fast_lgb_cfg(),
        "cutoff_days": 20.0,
        "n_splits": 3,
        "embargo": 10,
        "n_samples": 40,
        "paired": "log_ratio",
        "sampling": "balanced",
        "return_mc_se": True,
        "return_quantiles": (0.1, 0.9),
        "seed": 5,
        "n_cores": 1,
        "log_transform": True,
    }
    geo = nm.dml_normalise(bm.df, back_transform="geometric", **common)
    arith = nm.dml_normalise(bm.df, back_transform="arithmetic", **common)

    # `day_julian` is a linear day counter and must stay out of the tree's
    # split space alongside `date_unix` and `year`.
    assert "day_julian" not in arith.fold_models[0].feature_names  # type: ignore[attr-defined]
    assert "date_unix" not in arith.fold_models[0].feature_names  # type: ignore[attr-defined]

    g = geo.normalised["normalised"].to_numpy(dtype=float)
    a = arith.normalised["normalised"].to_numpy(dtype=float)
    # Averaging expm1(m + z_i) is at least expm1(m + mean z_i) by Jensen. The
    # old fold returned the log-space mean for both, so `back_transform` was
    # a no-op and the two arrays were identical.
    assert not np.allclose(a, g, atol=1e-9)
    assert (a >= g - 1e-9).all()
    assert float(np.mean(a - g)) > 1e-3

    assert (arith.normalised["q100"].to_numpy() <= arith.normalised["q900"].to_numpy()).all()
    assert np.isfinite(arith.normalised["mc_se"].to_numpy()).all()

    # Smearing is the arithmetic level multiplied by the Duan factor exactly
    # once: (level + 1) must scale by phi, per fold. The earlier combination
    # applied phi in the fold aggregation and again when adding m(t), i.e.
    # phi^2.
    smear = nm.dml_normalise(bm.df, back_transform="smearing", **common)
    phi_by_fold = {k: float(phi) for k, phi in enumerate(smear.smearing_factors)}
    phi_rows = smear.normalised["fold"].map(phi_by_fold).to_numpy(dtype=float)
    smear_vals = smear.normalised["normalised"].to_numpy(dtype=float)
    arith_vals = arith.normalised["normalised"].to_numpy(dtype=float)
    assert np.allclose(smear_vals + 1.0, phi_rows * (arith_vals + 1.0), rtol=1e-6, atol=1e-9)


def test_icn_surprise_weights_stable_under_underflow() -> None:
    items = [(10.0, 0), (10.0, 0), (20.0, 0)]
    # Every anchor's Gaussian gate underflows to 0 here. The old code then
    # replaced the peaked weights with a uniform average (13.33); the stable
    # log-space form keeps the least-surprising anchors.
    out = _aggregate(items, "mean", 0.0, surprise_errors=[30.0, 31.0, 32.0], surprise_tau=1.0)
    assert out == pytest.approx(10.0, abs=1e-6)
    out_lead = _aggregate(items, "mean", 5.0, surprise_errors=[30.0, 31.0, 32.0], surprise_tau=1.0)
    assert np.isfinite(out_lead)
    assert out_lead == pytest.approx(10.0, abs=1e-6)
    assert _aggregate(items, "mean", 0.0) == pytest.approx(float(np.mean([10.0, 10.0, 20.0])))


def test_icn_decompose_rejects_reserved_default_player_names() -> None:
    rng = np.random.default_rng(5)
    n = 400
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    w = rng.standard_normal(n)
    df = pd.DataFrame({"y": 10.0 + 2.0 * w, "observed": w}, index=idx)
    est = Chronos2Estimator()
    est.fit(df[["observed"]], df["y"])
    design = ICNDesign(
        block_h=80,
        stride=80,
        direction="forward",
        context_length=200,
        aggregate="mean",
        n_resamples=2,
        seed=0,
    )
    with pytest.raises(ValueError, match="clash"):
        icn_decompose(est, df, "y", design=design, pipeline=_FakeICLPipeline())


def test_transport_adjusted_scm_nan_weights_and_matched_donor_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import normet.causal.deweathered as dw

    rng = np.random.default_rng(7)
    n_days = 40
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    cutoff = "2023-01-21"
    post = (dates >= pd.Timestamp(cutoff)).astype(float)
    f = {
        "D1": np.where(post > 0, rng.uniform(0.0, 1.5, n_days), 0.0),
        "D2": np.where(post > 0, rng.uniform(0.0, 1.5, n_days), 0.0),
        # Large, and its outcome is missing on every third post day, so the
        # observed-donor mean and the all-donor mean of F differ.
        "D3": np.where(post > 0, 3.0, 0.0),
    }
    common_shock = 2.0 * np.sin(2.0 * np.pi * np.arange(n_days) / 9.0)
    rows: list[pd.DataFrame] = []
    for unit, offset in (("T", 0.0), ("D1", -1.0), ("D2", 2.0), ("D3", 1.0)):
        exposure = f.get(unit, np.zeros(n_days))
        y = 30.0 + offset + common_shock - 5.0 * exposure * post
        y = y + rng.normal(0.0, 0.01, n_days)
        if unit == "D3":
            y = np.where((post > 0) & (np.arange(n_days) % 3 == 0), np.nan, y)
        rows.append(pd.DataFrame({"date": dates, "ID": unit, "value": y, "expo": exposure}))
    panel = pd.concat(rows, ignore_index=True)

    def fake_fit(df_panel: pd.DataFrame, **kw: Any) -> tuple[pd.DataFrame, pd.Series, None]:
        idx = pd.DatetimeIndex(sorted(df_panel["date"].unique()))
        syn = pd.DataFrame({"synthetic": 30.0, "observed": 30.0, "effect": 0.0}, index=idx)
        # `mcnnm` reports NaN weights; the transport correction must fall back
        # to equal weights rather than filling them with 0 and switching itself off.
        nan_w = pd.Series(np.nan, index=list(kw["donors"]), dtype=float)
        return syn, nan_w, None

    monkeypatch.setattr(dw, "_fit_scm_with_weights", fake_fit)
    res = nm.transport_adjusted_scm(
        panel,
        treated_unit="T",
        cutoff_date=cutoff,
        donors=["D1", "D2", "D3"],
        spillover_exposure="expo",
        scm_backend="ascm",
    )
    assert res.gamma == pytest.approx(-5.0, abs=0.3)
    assert np.isfinite(res.weights.to_numpy(dtype=float)).all()
    assert float(np.abs(res.synthetic["spillover_adjustment"]).max()) > 0.0


def test_ox_invariant_string_dates_and_standard_log_transform() -> None:
    pytest.importorskip("lightgbm")
    bm = nm.make_synthetic_benchmark("F3_titration", n_days=30, seed=31)
    common: dict[str, Any] = {
        "o3_col": "O3",
        "no2_col": "NO2",
        "no_col": "NO",
        "covariates": bm.covariates,
        "variables_resample": bm.met_covariates,
        "method": "standard",
        "backend": "lightgbm",
        "model_config": _fast_lgb_cfg(),
        "n_samples": 12,
        "seed": 31,
        "n_cores": 1,
    }
    df_str = bm.df.copy()
    df_str["date"] = df_str["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    logged = nm.normalise_ox_invariant(df_str, log_transform=True, **common)
    reference = nm.normalise_ox_invariant(bm.df, log_transform=True, **common)
    # A string `date` column must be coerced, not used as raw group labels
    # (which left the observed columns all-NaN).
    assert logged.species["O3_observed"].notna().all()
    pd.testing.assert_series_equal(
        logged.species["O3_normalised"], reference.species["O3_normalised"]
    )

    unlogged = nm.normalise_ox_invariant(bm.df, log_transform=False, **common)
    # The standard branch must honour `log_transform` like dml/crossfit do.
    assert not np.allclose(
        reference.species["O3_normalised"].to_numpy(dtype=float),
        unlogged.species["O3_normalised"].to_numpy(dtype=float),
    )


def test_f3_benchmark_noise_and_pool_expectation_truth() -> None:
    quiet = nm.make_synthetic_benchmark("F3_titration", n_days=20, noise_sd=0.0, seed=5)
    noisy = nm.make_synthetic_benchmark("F3_titration", n_days=20, noise_sd=4.0, seed=5)
    # `noise_sd` must reach F3 (the old branch hard-coded a 0.5 ppb noise).
    assert not np.allclose(quiet.df["O3"], noisy.df["O3"])
    # Ground truth is the noise-free pool expectation, so it is unchanged by
    # the observation-noise level.
    np.testing.assert_allclose(
        quiet.truth["emission_true"].to_numpy(dtype=float),
        noisy.truth["emission_true"].to_numpy(dtype=float),
    )
    # Noise enters the conserved totals, so the species still add up exactly.
    for bm in (quiet, noisy):
        ox_rebuilt = bm.df["O3"] / 48.0000 + bm.df["NO2"] / 46.0055
        np.testing.assert_allclose(ox_rebuilt.to_numpy(dtype=float), bm.df["Ox"] / 48.0, atol=1e-9)
