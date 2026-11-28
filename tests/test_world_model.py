"""Tests for normet.world_model (AC-JEPA, SeparableWorldModel, and benchmark harness)."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd
import pytest

import normet as nm
from normet.eval.benchmarks import evaluate_benchmark, make_synthetic_benchmark
from normet.world_model import (
    LatentJEPA,
    SeparableWorldModel,
    WorldModelConfig,
    benchmark_interaction_sweep,
    benchmark_world_models,
    detect_jepa_events,
    jepa_select_donors,
)


def test_jepa_anomaly_detection_ignores_weather_and_catches_emission_burst() -> None:
    """AC-JEPA stays quiet under extreme meteorological stagnation and flags true emission bursts."""
    rng = np.random.default_rng(101)
    n_train = 24 * 35
    n_test = 24 * 25
    n_tot = n_train + n_test
    idx = pd.date_range("2024-01-01", periods=n_tot, freq="h")
    t_h = np.arange(n_tot, dtype=float)

    ws = np.clip(
        4.0 + 1.2 * np.sin(2.0 * np.pi * t_h / 24.0) + rng.normal(0.0, 0.5, n_tot), 0.4, 14.0
    )
    blh = np.clip(
        500.0 + 250.0 * np.sin(2.0 * np.pi * (t_h - 6.0) / 24.0) + rng.normal(0.0, 35.0, n_tot),
        90.0,
        1800.0,
    )
    t2m = 12.0 + 5.0 * np.sin(2.0 * np.pi * t_h / (24.0 * 30.0)) + rng.normal(0.0, 1.0, n_tot)

    # Meteorological stagnation window in test split (predictable weather spike, zero emission anomaly)
    ws[n_train + 72 : n_train + 108] = 1.4
    blh[n_train + 72 : n_train + 108] = 180.0

    met_effect = 22.0 * np.exp(-ws / 2.8) + 10.0 * (300.0 / blh) - 0.15 * t2m
    emi = 30.0 + 5.0 * np.sin(2.0 * np.pi * (t_h - 6.0) / 24.0)
    # True unmodeled emission burst later in test split under ordinary weather
    burst_sl = slice(n_train + 260, n_train + 310)
    emi_obs = emi.copy()
    emi_obs[burst_sl] += 38.0

    y = np.maximum(1.0, emi_obs + met_effect + rng.normal(0.0, 0.6, n_tot))
    df = pd.DataFrame({"value": y, "ws": ws, "blh": blh, "t2m": t2m}, index=idx)

    df_train = df.iloc[:n_train]
    df_test = df.iloc[n_train:]

    jepa = LatentJEPA(patch_length=6, stride=6, latent_dim=6, met_degree=2, seed=7).fit(
        df_train, target_cols="value", met_cols=["ws", "blh", "t2m"]
    )
    scores = jepa.anomaly_scores(df_test)
    assert len(scores) == n_test
    assert np.all(np.isfinite(scores))

    met_spike_score = float(scores.iloc[72:108].mean())
    burst_score = float(scores.iloc[260:310].mean())
    assert burst_score > 3.5
    assert burst_score > 2.0 * max(met_spike_score, 0.5)

    events = detect_jepa_events(
        df_test,
        target_cols="value",
        met_cols=["ws", "blh", "t2m"],
        jepa=jepa,
        threshold=3.0,
        min_length=6,
    )
    assert not events.empty
    assert any(row["start"] <= idx[n_train + 290] <= row["end"] for _, row in events.iterrows())

    # Also verify external ChronosEmbedder-compatible duck-typed `.embed_patches` encoder
    class _StubPatchEncoder:
        def embed_patches(self, series: np.ndarray, context_length: int = 2048) -> np.ndarray:
            arr = np.asarray(series, dtype=float)[-context_length:]
            n_p = max(8, len(arr) // 12)
            chunks = np.array_split(arr, n_p)
            return np.stack([[np.mean(c), np.std(c), c[-1] - c[0]] for c in chunks], axis=0)

    jepa_ext = LatentJEPA(latent_dim=3, encoder=_StubPatchEncoder(), seed=7).fit(
        df_train, target_cols="value", met_cols=["ws", "blh", "t2m"]
    )
    ext_scores = jepa_ext.anomaly_scores(df_test)
    assert float(ext_scores.iloc[260:310].mean()) > float(ext_scores.iloc[72:108].mean())


def test_jepa_select_donors_beats_raw_under_weather_confounding() -> None:
    """JEPA donor matching ranks emission-matched twin above weather-confounded decoy."""
    rng = np.random.default_rng(202)
    n_pre = 24 * 30
    n_post = 24 * 10
    n = n_pre + n_post
    idx = pd.date_range("2024-03-01", periods=n, freq="h")
    t_h = np.arange(n, dtype=float)
    cutoff = idx[n_pre]

    # Shared true latent emission factor for Treated & TrueDonor; opposite phase for Decoy
    e_true = (
        35.0
        + 10.0 * np.sin(2.0 * np.pi * t_h / (24.0 * 7.0))
        + 4.0 * np.sin(2.0 * np.pi * t_h / 24.0)
    )
    e_decoy = (
        35.0
        - 12.0 * np.sin(2.0 * np.pi * t_h / (24.0 * 7.0))
        + 5.0 * np.cos(2.0 * np.pi * t_h / 24.0)
    )
    e_noise = 35.0 + rng.normal(0.0, 6.0, n)

    w_a = 4.0 + 2.0 * np.sin(2.0 * np.pi * t_h / 60.0) + rng.normal(0.0, 0.3, n)
    w_b = 4.0 - 2.0 * np.sin(2.0 * np.pi * t_h / 60.0) + rng.normal(0.0, 0.3, n)  # opposite weather
    w_c = w_a + rng.normal(0.0, 0.1, n)  # same weather as treated!

    def _obs(e: np.ndarray, w: np.ndarray) -> np.ndarray:
        return np.maximum(1.0, e + 28.0 * np.exp(-w / 2.4) + rng.normal(0.0, 0.3, len(e)))

    y_tr = _obs(e_true, w_a)
    y_true_d = _obs(e_true, w_b)
    y_decoy_d = _obs(e_decoy, w_c)
    y_other_d = _obs(e_noise, w_b)

    # Inject post-cutoff corruption to confirm zero leakage from >= cutoff
    y_tr[n_pre:] -= 20.0
    y_true_d[n_pre:] += 25.0

    panel = pd.concat(
        [
            pd.DataFrame({"date": idx, "ID": "Treated", "value": y_tr, "ws": w_a}),
            pd.DataFrame({"date": idx, "ID": "TrueDonor", "value": y_true_d, "ws": w_b}),
            pd.DataFrame({"date": idx, "ID": "DecoyDonor", "value": y_decoy_d, "ws": w_c}),
            pd.DataFrame({"date": idx, "ID": "OtherDonor", "value": y_other_d, "ws": w_b}),
        ],
        ignore_index=True,
    )

    ranked = jepa_select_donors(
        panel,
        date_col="date",
        unit_col="ID",
        outcome_col="value",
        met_cols=["ws"],
        treated_unit="Treated",
        cutoff_date=cutoff,
        patch_length=6,
        stride=6,
        top_k=2,
        return_weights=True,
    )
    assert ranked.iloc[0]["unit"] == "TrueDonor"
    true_row = ranked.loc[ranked["unit"] == "TrueDonor"].iloc[0]
    decoy_row = ranked.loc[ranked["unit"] == "DecoyDonor"].iloc[0]
    assert true_row["similarity"] > decoy_row["similarity"]
    assert true_row["similarity"] > true_row["raw_similarity"]
    assert abs(float(ranked["weight"].sum()) - 1.0) < 1e-6
    assert float(true_row["weight"]) > 0.7


def test_separable_world_model_bijectivity_and_coordinate_discovery() -> None:
    """SeparableWorldModel is strictly bijective and picks raw vs log coordinates accurately."""
    bm_add = make_synthetic_benchmark("F1_additive", n_days=45, noise_sd=0.8, seed=11)
    wm_add = SeparableWorldModel(WorldModelConfig(backend="separable", seed=11)).fit(
        bm_add.df, bm_add.target, bm_add.met_covariates
    )
    Y_add = bm_add.df[[bm_add.target]].to_numpy(dtype=float)
    assert np.allclose(wm_add.decode(wm_add.encode(Y_add)), Y_add, atol=1e-8)
    assert wm_add.fit_metadata_["gammas"][0] >= 0.6

    bm_mult = make_synthetic_benchmark("F2_multiplicative", n_days=45, noise_sd=0.8, seed=11)
    wm_mult = SeparableWorldModel(WorldModelConfig(backend="separable", seed=11)).fit(
        bm_mult.df, bm_mult.target, bm_mult.met_covariates
    )
    Y_mult = bm_mult.df[[bm_mult.target]].to_numpy(dtype=float)
    assert np.allclose(wm_mult.decode(wm_mult.encode(Y_mult)), Y_mult, atol=1e-8)
    assert wm_mult.fit_metadata_["gammas"][0] <= 0.35

    res_mult = wm_mult.normalise(bm_mult.df, n_samples=60, paired=True, seed=11)
    assert (
        res_mult.separability_gain["mean_rho_lb_latent"]
        < res_mult.separability_gain["mean_rho_lb_raw"]
    )


def test_separable_world_model_discovers_ox_coupling_on_f3_titration() -> None:
    """Joint multi-species fit reduces interaction rho_lb and recovers O3 on F3_titration."""
    bm = make_synthetic_benchmark(
        "F3_titration", n_days=50, event_reduction_pct=-40.0, noise_sd=1.0, seed=19
    )
    wm = SeparableWorldModel(WorldModelConfig(backend="separable", seed=19)).fit(
        bm.df, ["O3", "NO2"], bm.met_covariates
    )
    Y_pair = bm.df[["O3", "NO2"]].to_numpy(dtype=float)
    assert np.allclose(wm.decode(wm.encode(Y_pair)), Y_pair, atol=1e-7)

    res = wm.normalise(bm.df, n_samples=80, paired=True, seed=19)
    assert res.separability_gain["rho_lb_reduction"] >= 0.0
    o3_frame = res.to_normet_frame("O3")
    scores = evaluate_benchmark(bm, o3_frame, shuffle_control=True, seed=19)
    assert scores["wrf_linear"] > 0.70
    assert scores["cf_r2"] > scores["shuffle_control_r2"]


def test_world_model_trajectory_rollout_what_if() -> None:
    """counterfactual_rollout preserves exact identity and models disturbance, wind, and precursor shifts."""
    bm = make_synthetic_benchmark(
        "F2_multiplicative",
        n_days=50,
        event_start_day=30,
        event_duration_days=5,
        event_reduction_pct=45.0,
        noise_sd=0.8,
        seed=23,
    )
    df = bm.df.copy()
    wm = SeparableWorldModel(WorldModelConfig(backend="separable", seed=23)).fit(
        df, bm.target, bm.met_covariates
    )

    # 1. Exact identity check when disturbance_scale=1.0 and no weather shift
    roll_id = wm.counterfactual_rollout(
        df, start=bm.event_start, end=bm.event_end, disturbance_scale=1.0
    )
    assert np.allclose(
        roll_id.actual[bm.target].to_numpy(),
        roll_id.counterfactual[bm.target].to_numpy(),
        atol=1e-6,
    )
    assert "latent_rho_lb" in roll_id.separability_certificate

    # 2. Disturbance ablation drops counterfactual mean below positive burst actuals
    roll_ablate = wm.counterfactual_rollout(
        df, start=bm.event_start, end=bm.event_end, disturbance_scale=0.0
    )
    assert (
        roll_ablate.summary_metrics["mean_counterfactual"]
        < roll_ablate.summary_metrics["mean_actual"]
    )

    # 3. Stronger wind dilution reduces peak & mean levels during episode
    roll_windy = wm.counterfactual_rollout(
        df, start=bm.event_start, end=bm.event_end, met_shifts={"ws": ("*", 1.8)}
    )
    assert (
        roll_windy.summary_metrics["mean_counterfactual"]
        < roll_windy.summary_metrics["mean_actual"]
    )

    # 4. Multi-species precursor scaling on F3_titration
    bm3 = make_synthetic_benchmark("F3_titration", n_days=40, seed=29)
    df3 = bm3.df.set_index("date")
    wm3 = SeparableWorldModel(WorldModelConfig(backend="separable", seed=29)).fit(
        df3, ["O3", "NO2"], bm3.met_covariates
    )
    ep_s = bm3.event_start
    ep_e = bm3.event_end
    scen = df3.loc[(df3.index >= ep_s) & (df3.index <= ep_e), bm3.met_covariates].copy()
    scen["ws"] = scen["ws"] * 1.3
    roll_sp = wm3.counterfactual_rollout(
        df3,
        start=ep_s,
        end=ep_e,
        weather_scenario=scen,
        species_scale={"NO2": 0.65},
    )
    assert roll_sp.counterfactual["NO2"].mean() < roll_sp.actual["NO2"].mean()


def test_separable_world_model_torch_backend() -> None:
    """Optional PyTorch backend runs differentiable separability training and bijective decoding."""
    pytest.importorskip("torch")
    bm = make_synthetic_benchmark("F2_multiplicative", n_days=60, noise_sd=0.8, seed=11)
    cfg = WorldModelConfig(
        backend="torch",
        device="cpu",
        torch_epochs=250,
        torch_lr=3e-2,
        torch_hidden=24,
        seed=11,
    )
    wm = SeparableWorldModel(cfg).fit(bm.df, bm.target, bm.met_covariates)
    Y = bm.df[[bm.target]].to_numpy(dtype=float)
    assert np.allclose(wm.decode(wm.encode(Y)), Y, atol=1e-6)

    res = wm.normalise(bm.df, n_samples=40, paired=True, quantiles=(0.1, 0.9), seed=11)
    assert res.separability_gain["mean_rho_lb_latent"] < res.separability_gain["mean_rho_lb_raw"]
    assert {"observed", "normalised", "factual", "q100", "q900"} <= set(res.df.columns)

    roll = wm.counterfactual_rollout(
        bm.df, start=bm.event_start, end=bm.event_end, disturbance_scale=0.0
    )
    assert len(roll.counterfactual) > 24


def test_benchmark_world_models_and_interaction_sweep() -> None:
    """Benchmark harness evaluates baselines and SeparableWorldModel across F1-F6."""
    table = benchmark_world_models(
        families=("F1_additive", "F2_multiplicative", "F3_titration", "F6_interaction_dial"),
        methods=("raw_linear", "log1p_linear", "separable_wm"),
        n_days=35,
        n_samples=40,
        seed=42,
    )
    assert len(table) == 12
    assert np.all(np.isfinite(table["cf_r2"]))
    assert np.all(np.isfinite(table["nrmse_pct"]))

    f2 = table[table["family"] == "F2_multiplicative"].set_index("method")
    assert f2.loc["separable_wm", "mean_rho_lb_latent"] <= f2.loc["raw_linear", "mean_rho_lb_raw"]

    sweep = benchmark_interaction_sweep(
        lambdas=(0.0, 0.5, 1.0),
        methods=("raw_linear", "log1p_linear", "separable_wm"),
        n_days=35,
        n_samples=40,
        seed=42,
    )
    assert len(sweep) == 9
    assert "interaction_lambda" in sweep.columns
    sw_sep = sweep[sweep["method"] == "separable_wm"].sort_values("interaction_lambda")
    # Interaction rho in raw world increases with lambda
    assert float(sw_sep.iloc[-1]["sim_rho_raw"]) > float(sw_sep.iloc[0]["sim_rho_raw"])


def test_gam_head_accuracy_gains_and_tree_head() -> None:
    """GAM quantile spline head captures nonlinear meteorology sharply beyond linear head."""
    bm_f1 = make_synthetic_benchmark("F1_additive", n_days=55, noise_sd=0.8, seed=11)
    wm_lin = SeparableWorldModel(WorldModelConfig(met_head="linear", seed=11)).fit(
        bm_f1.df, bm_f1.target, bm_f1.met_covariates
    )
    wm_gam = SeparableWorldModel(WorldModelConfig(met_head="gam", n_knots=6, seed=11)).fit(
        bm_f1.df, bm_f1.target, bm_f1.met_covariates
    )
    s_lin = evaluate_benchmark(
        bm_f1, wm_lin.normalise(bm_f1.df, n_samples=60, seed=11).to_normet_frame()
    )
    s_gam = evaluate_benchmark(
        bm_f1, wm_gam.normalise(bm_f1.df, n_samples=60, seed=11).to_normet_frame()
    )
    assert s_gam["wrf_linear"] > 0.78
    assert s_gam["cf_r2"] > 0.75
    assert s_gam["cf_r2"] > s_lin["cf_r2"] + 0.20
    assert s_gam["wrf_linear"] > s_lin["wrf_linear"] + 0.35

    bm_f2 = make_synthetic_benchmark("F2_multiplicative", n_days=55, noise_sd=0.8, seed=11)
    wm_f2 = SeparableWorldModel(WorldModelConfig(met_head="gam", n_knots=6, seed=11)).fit(
        bm_f2.df, bm_f2.target, bm_f2.met_covariates
    )
    s_f2 = evaluate_benchmark(
        bm_f2, wm_f2.normalise(bm_f2.df, n_samples=60, seed=11).to_normet_frame()
    )
    assert s_f2["wrf_linear"] > 0.90
    assert s_f2["cf_r2"] > 0.62
    assert s_f2["nrmse_pct"] < 11.0

    pytest.importorskip("sklearn")
    wm_tree = SeparableWorldModel(WorldModelConfig(met_head="tree", seed=11)).fit(
        bm_f1.df, bm_f1.target, bm_f1.met_covariates
    )
    res_tree = wm_tree.normalise(bm_f1.df, n_samples=40, seed=11)
    assert float(res_tree.df["normalised"].std()) > 0.0


def test_mix_then_warp_conservation_on_f3_titration() -> None:
    """Explicit mix_then_warp mode maintains strict invertibility and high O3 WRF."""
    bm3 = make_synthetic_benchmark("F3_titration", n_days=55, noise_sd=0.8, seed=19)
    cfg = WorldModelConfig(met_head="gam", coordinate_order="mix_then_warp", seed=19)
    wm3 = SeparableWorldModel(cfg).fit(bm3.df, ["O3", "NO2"], bm3.met_covariates)
    Y = bm3.df[["O3", "NO2"]].to_numpy(dtype=float)
    assert wm3.fit_metadata_["coordinate_order"] == "mix_then_warp"
    assert np.allclose(wm3.decode(wm3.encode(Y)), Y, atol=1e-7)

    res = wm3.normalise(bm3.df, n_samples=60, paired=True, seed=19)
    sc = evaluate_benchmark(bm3, res.to_normet_frame("O3"))
    assert sc["wrf_linear"] > 0.90
    assert sc["nrmse_pct"] < 8.0
    assert sc["cf_r2"] > 0.35
    assert sc["cf_r2"] > sc["factual_copy_r2"] + 5.0


def test_rollout_bootstrap_uq_operating_window_and_persistence(tmp_path) -> None:
    """Rollout bootstrap intervals, operating_window curve, and JSON save/load work cleanly."""
    bm = make_synthetic_benchmark("F2_multiplicative", n_days=45, noise_sd=0.8, seed=11)
    wm = SeparableWorldModel(WorldModelConfig(met_head="gam", seed=11)).fit(
        bm.df, bm.target, bm.met_covariates
    )
    roll = wm.counterfactual_rollout(
        bm.df,
        start=bm.event_start,
        end=bm.event_end,
        met_shifts={"ws": ("*", 1.4)},
        n_boot=40,
        block_hours=12,
        seed=11,
    )
    assert roll.counterfactual_p10 is not None and roll.counterfactual_p90 is not None
    assert np.all(
        roll.counterfactual_p10[bm.target].to_numpy()
        <= roll.counterfactual_p90[bm.target].to_numpy() + 1e-9
    )

    ow = wm.operating_window(bm.df, horizons=(1, 6, 12, 24))
    assert list(ow["horizon"]) == [1, 6, 12, 24]
    assert (ow["rmse"] >= 0.0).all()
    assert np.all(np.isfinite(ow["rc_score_linear"]))

    save_path = tmp_path / "wm_v1.json.gz"
    wm.save(save_path)
    loaded = SeparableWorldModel.load(save_path)
    Y = bm.df[[bm.target]].to_numpy(dtype=float)
    assert np.allclose(wm.encode(Y), loaded.encode(Y), atol=1e-12)
    pd.testing.assert_frame_equal(
        wm.normalise(bm.df, n_samples=20, seed=11).df,
        loaded.normalise(bm.df, n_samples=20, seed=11).df,
    )


def test_jepa_scm_recovers_policy_effect_under_weather_confounding() -> None:
    """jepa_scm selects true donors under local weather mismatch and recovers ATT accurately."""
    rng = np.random.default_rng(55)
    n_pre = 24 * 35
    n_post = 24 * 12
    n = n_pre + n_post
    idx = pd.date_range("2024-05-01", periods=n, freq="h")
    t_h = np.arange(n, dtype=float)
    cutoff = idx[n_pre]

    e_shared = (
        40.0
        + 10.0 * np.sin(2.0 * np.pi * t_h / (24.0 * 7.0))
        + 4.0 * np.sin(2.0 * np.pi * t_h / 24.0)
    )
    e_decoy = 40.0 - 11.0 * np.sin(2.0 * np.pi * t_h / (24.0 * 7.0))

    w_tr = np.clip(
        4.0 + 2.0 * np.sin(2.0 * np.pi * t_h / 60.0) + rng.normal(0.0, 0.3, n), 0.6, 12.0
    )
    w_don = np.clip(
        4.0 - 2.0 * np.sin(2.0 * np.pi * t_h / 60.0) + rng.normal(0.0, 0.3, n), 0.6, 12.0
    )
    w_dec = w_tr + rng.normal(0.0, 0.1, n)

    y_tr = np.maximum(1.0, e_shared + 26.0 * np.exp(-w_tr / 2.4) + rng.normal(0.0, 0.4, n))
    y_don = np.maximum(1.0, e_shared + 26.0 * np.exp(-w_don / 2.4) + rng.normal(0.0, 0.4, n))
    y_dec = np.maximum(1.0, e_decoy + 26.0 * np.exp(-w_dec / 2.4) + rng.normal(0.0, 0.4, n))

    # Policy reduces treated emissions by -12.0 post-cutoff
    y_tr[n_pre:] = np.maximum(1.0, y_tr[n_pre:] - 12.0)

    panel = pd.concat(
        [
            pd.DataFrame({"date": idx, "ID": "Treated", "value": y_tr, "ws": w_tr}),
            pd.DataFrame({"date": idx, "ID": "TrueDonor", "value": y_don, "ws": w_don}),
            pd.DataFrame({"date": idx, "ID": "DecoyDonor", "value": y_dec, "ws": w_dec}),
        ],
        ignore_index=True,
    )
    res = nm.jepa_scm(
        panel,
        met_cols=["ws"],
        treated_unit="Treated",
        cutoff_date=cutoff,
        met_head="gam",
        patch_length=6,
        top_k=1,
    )
    assert isinstance(res, nm.JEPASCMResult)
    assert res.donor_table.iloc[0]["unit"] == "TrueDonor"
    assert res.pre_rmspe < 2.0
    assert -14.5 < res.att < -9.5
    assert np.isfinite(res.att_hac_se)


def test_multisite_world_model_separates_regional_plume_from_local_spike() -> None:
    """MultisiteWorldModel filters shared regional transport and flags only the local emitter spike."""
    rng = np.random.default_rng(88)
    n = 24 * 35
    idx = pd.date_range("2024-06-01", periods=n, freq="h")
    t_h = np.arange(n, dtype=float)

    # Shared regional smoke plume affecting all 4 stations on days 15..18
    reg_plume = np.zeros(n, dtype=float)
    reg_plume[24 * 15 : 24 * 18] = 28.0

    # Purely local emission leak only at Site_A on days 25..27
    local_leak = np.zeros(n, dtype=float)
    local_leak[24 * 25 : 24 * 27] = 30.0

    frames = []
    for i, site in enumerate(("Site_A", "Site_B", "Site_C", "Site_D")):
        ws = np.clip(
            4.0 + 1.5 * np.sin(2.0 * np.pi * t_h / 72.0 + i) + rng.normal(0.0, 0.3, n), 0.5, 12.0
        )
        base = 25.0 + 3.0 * np.sin(2.0 * np.pi * t_h / 24.0) + 18.0 * np.exp(-ws / 2.5)
        y = base + reg_plume + (local_leak if site == "Site_A" else 0.0) + rng.normal(0.0, 0.5, n)
        frames.append(pd.DataFrame({"date": idx, "ID": site, "value": y, "ws": ws}))
    panel = pd.concat(frames, ignore_index=True)

    ms = nm.fit_multisite_world_model(panel, met_cols=["ws"], n_regional_factors=1)
    dec = ms.decompose()
    assert float(dec.variance_explained.iloc[0]["variance_ratio"]) > 0.50

    loc_z = ms.local_anomaly_scores()
    # Shared regional plume should NOT trigger high local z-scores on background sites
    assert float(loc_z.iloc[24 * 15 : 24 * 18][["Site_B", "Site_C", "Site_D"]].mean().mean()) < 2.5
    # Site_A local leak stands out strongly above background stations
    assert float(loc_z.iloc[24 * 25 : 24 * 27]["Site_A"].mean()) > 4.0
    assert float(loc_z.iloc[24 * 25 : 24 * 27]["Site_B"].mean()) < 2.5

    wi = ms.regional_what_if("Site_B", start=idx[24 * 15], end=idx[24 * 18 - 1], regional_scale=0.0)
    assert float(wi["counterfactual"].mean()) < float(wi["actual"].mean()) - 15.0


def test_world_model_diagnostic_plots_smoke() -> None:
    """Plotting helpers render without GUI backends and preserve valid matplotlib figures."""
    import matplotlib.pyplot as plt

    bm = make_synthetic_benchmark("F2_multiplicative", n_days=35, noise_sd=0.8, seed=11)
    wm = SeparableWorldModel(WorldModelConfig(met_head="gam", seed=11)).fit(
        bm.df, bm.target, bm.met_covariates
    )
    fig1, _ = nm.plot_separability_landscape(wm, bm.df)
    roll = wm.counterfactual_rollout(
        bm.df, start=bm.event_start, end=bm.event_end, n_boot=20, seed=11
    )
    fig2, _ = nm.plot_rollout(roll)

    jepa = LatentJEPA(patch_length=6, stride=6, seed=11).fit(bm.df, bm.target, bm.met_covariates)
    sc = jepa.anomaly_scores(bm.df)
    ev = detect_jepa_events(bm.df, bm.target, bm.met_covariates, jepa=jepa, threshold=2.5)
    fig3, _ = nm.plot_jepa_anomalies(bm.df, sc, value_col=bm.target, events=ev)
    for fig in (fig1, fig2, fig3):
        plt.close(fig)


def test_separable_world_model_calendar_basis_pinned_across_window_spans(tmp_path) -> None:
    """F1: fit/eval windows crossing 90/180d thresholds match basis width, trend scale, and legacy load."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(3)
    n = 24 * 100  # 100 days (>= 90d threshold -> 8 calendar columns)
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    t_h = np.arange(n, dtype=float)
    ws = np.clip(4.0 + 2.0 * np.sin(2.0 * np.pi * t_h / 24.0) + rng.normal(0.0, 0.4, n), 0.3, 12.0)
    blh = np.clip(
        500.0 + 200.0 * np.sin(2.0 * np.pi * (t_h - 6.0) / 24.0) + rng.normal(0.0, 30.0, n),
        80.0,
        2000.0,
    )
    a = np.maximum(
        0.5,
        20.0
        + 8.0 * np.sin(2.0 * np.pi * t_h / 24.0)
        + 15.0 * np.exp(-ws / 2.5)
        + 3.0 * np.sin(2.0 * np.pi * t_h / (24.0 * 60.0))
        + 0.002 * t_h
        + rng.normal(0.0, 0.5, n),
    )
    df = pd.DataFrame({"a": a, "ws": ws, "blh": blh}, index=idx)
    wm = SeparableWorldModel(WorldModelConfig(seed=1)).fit(df, "a", ["ws", "blh"])
    assert wm._n_cal == 8

    # 5-day subset (< 90d threshold -> would be 7 calendar columns if unpinned)
    ep = df.iloc[24 * 40 : 24 * 45]
    norm_full = wm.normalise(df, n_samples=10, seed=1)
    norm_ep = wm.normalise(ep, n_samples=10, seed=1)
    np.testing.assert_allclose(
        norm_full.df.loc[ep.index, "factual"].to_numpy(),
        norm_ep.df["factual"].to_numpy(),
        atol=1e-10,
    )

    roll_full = wm.counterfactual_rollout(
        df, start=idx[24 * 40], end=idx[24 * 45 - 1], disturbance_scale=0.0, ar_carryover=False
    )
    roll_ep = wm.counterfactual_rollout(
        ep, start=idx[24 * 40], end=idx[24 * 45 - 1], disturbance_scale=0.0, ar_carryover=False
    )
    np.testing.assert_allclose(
        roll_full.counterfactual["a"].to_numpy(),
        roll_ep.counterfactual["a"].to_numpy(),
        atol=1e-10,
    )

    audit_ep = wm.audit(ep)
    assert np.isfinite(audit_ep["separability_gain"]["mean_rc_latent"])
    ow_ep = wm.operating_window(ep, horizons=(1, 6))
    assert list(ow_ep["horizon"]) == [1, 6]
    fig, _ = nm.plot_separability_landscape(wm, ep)
    plt.close(fig)

    # Legacy archive load() without "calendar_basis" infers the 90/180d bucket from _n_cal
    legacy_path = tmp_path / "wm_legacy.json.gz"
    wm.save(legacy_path)
    with gzip.open(legacy_path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert "calendar_basis" in payload
    payload.pop("calendar_basis")
    with gzip.open(legacy_path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, allow_nan=False)

    legacy = SeparableWorldModel.load(legacy_path)
    assert legacy._cal_basis is None
    legacy_norm = legacy.normalise(ep, n_samples=10, seed=1)
    assert len(legacy_norm.df) == len(ep)
    legacy_roll = legacy.counterfactual_rollout(ep, start=idx[24 * 40], end=idx[24 * 45 - 1])
    assert len(legacy_roll.counterfactual) == len(ep)


def test_jepa_scm_survives_pre_and_full_crossing_calendar_thresholds() -> None:
    """F2: jepa_scm handles pre-cutoff (<90d) and full record (>=180d) crossing calendar buckets."""
    rng = np.random.default_rng(5)
    n_pre, n_post = 24 * 40, 24 * 150  # 40d pre (7 cal cols) vs 190d full (12 cal cols)
    n = n_pre + n_post
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    t_h = np.arange(n, dtype=float)
    cutoff = idx[n_pre]

    e = (
        40.0
        + 10.0 * np.sin(2.0 * np.pi * t_h / (24.0 * 7.0))
        + 4.0 * np.sin(2.0 * np.pi * t_h / 24.0)
    )
    w_tr = np.clip(
        4.0 + 2.0 * np.sin(2.0 * np.pi * t_h / 60.0) + rng.normal(0.0, 0.3, n), 0.6, 12.0
    )
    w_don = np.clip(
        4.0 - 2.0 * np.sin(2.0 * np.pi * t_h / 60.0) + rng.normal(0.0, 0.3, n), 0.6, 12.0
    )
    y_tr = np.maximum(1.0, e + 26.0 * np.exp(-w_tr / 2.4) + rng.normal(0.0, 0.4, n))
    y_don = np.maximum(1.0, e + 26.0 * np.exp(-w_don / 2.4) + rng.normal(0.0, 0.4, n))
    y_tr[n_pre:] = np.maximum(1.0, y_tr[n_pre:] - 12.0)

    panel = pd.concat(
        [
            pd.DataFrame({"date": idx, "ID": "Treated", "value": y_tr, "ws": w_tr}),
            pd.DataFrame({"date": idx, "ID": "Donor", "value": y_don, "ws": w_don}),
        ],
        ignore_index=True,
    )
    res = nm.jepa_scm(panel, met_cols=["ws"], treated_unit="Treated", cutoff_date=cutoff, top_k=1)
    assert res.pre_rmspe < 1.5
    assert -13.5 < res.att < -10.5


def test_jepa_anomaly_scores_cover_tail_remainder_when_not_stride_multiple() -> None:
    """F3: AC-JEPA detects a burst in the final (n - L) % stride timestamps and pins calendar basis."""
    rng = np.random.default_rng(101)
    n_train = 24 * 35
    n_tail = 245  # (245 - 6) % 6 == 5 != 0 -> last rows sit past stride grid
    n_tot = n_train + n_tail
    idx = pd.date_range("2024-01-01", periods=n_tot, freq="h")
    t_h = np.arange(n_tot, dtype=float)
    ws = np.clip(
        4.0 + 1.2 * np.sin(2.0 * np.pi * t_h / 24.0) + rng.normal(0.0, 0.5, n_tot), 0.4, 14.0
    )
    blh = np.clip(
        500.0 + 250.0 * np.sin(2.0 * np.pi * (t_h - 6.0) / 24.0) + rng.normal(0.0, 35.0, n_tot),
        90.0,
        1800.0,
    )
    y = np.maximum(
        1.0,
        30.0
        + 5.0 * np.sin(2.0 * np.pi * (t_h - 6.0) / 24.0)
        + 22.0 * np.exp(-ws / 2.8)
        + 10.0 * (300.0 / blh)
        + rng.normal(0.0, 0.6, n_tot),
    )
    df = pd.DataFrame({"value": y, "ws": ws, "blh": blh}, index=idx)

    jepa = LatentJEPA(patch_length=6, stride=6, latent_dim=6, met_degree=2, seed=7).fit(
        df.iloc[:n_train], "value", ["ws", "blh"]
    )
    frame = df.iloc[n_train:]
    base_tail = jepa.anomaly_scores(frame).iloc[-4:]

    burst = frame.copy()
    burst.iloc[-4:, burst.columns.get_loc("value")] += 45.0
    burst_scores = jepa.anomaly_scores(burst)
    assert float(burst_scores.iloc[-4:].min()) > max(10.0, float(base_tail.max()) * 2.0)

    events = detect_jepa_events(
        burst, "value", ["ws", "blh"], jepa=jepa, threshold=3.0, min_length=4
    )
    assert not events.empty
    assert events.iloc[0]["end"] == frame.index[-1]
