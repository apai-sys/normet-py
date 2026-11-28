"""Benchmark and ablation harness comparing World-Model deweathering across F1–F6."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from ..eval.benchmarks import (
    SyntheticBenchmark,
    benchmark_interaction_diagnostics,
    evaluate_benchmark,
    make_synthetic_benchmark,
)
from ..utils._config import DEFAULT_SEED
from .latent_wm import SeparableWorldModel, WorldModelConfig

__all__ = [
    "benchmark_world_models",
    "benchmark_interaction_sweep",
]


def _make_wm_for_method(method: str, *, seed: int) -> SeparableWorldModel:
    if method == "raw_linear":
        cfg = WorldModelConfig(
            backend="separable", met_head="linear", fixed_gamma=1.0, learn_mixing=False, seed=seed
        )
    elif method == "log1p_linear":
        cfg = WorldModelConfig(
            backend="separable", met_head="linear", fixed_gamma=0.0, learn_mixing=False, seed=seed
        )
    elif method == "raw_gam":
        cfg = WorldModelConfig(
            backend="separable", met_head="gam", fixed_gamma=1.0, learn_mixing=False, seed=seed
        )
    elif method == "log1p_gam":
        cfg = WorldModelConfig(
            backend="separable", met_head="gam", fixed_gamma=0.0, learn_mixing=False, seed=seed
        )
    elif method == "separable_wm":
        cfg = WorldModelConfig(
            backend="separable", met_head="gam", coordinate_order="auto", seed=seed
        )
    elif method == "torch_wm":
        cfg = WorldModelConfig(backend="torch", met_head="gam", seed=seed)
    else:
        raise ValueError(
            f"Unknown method {method!r}; expected raw_linear, log1p_linear, raw_gam, "
            "log1p_gam, separable_wm, or torch_wm."
        )
    return SeparableWorldModel(cfg)


def _eval_one_run(
    bm: SyntheticBenchmark,
    method: str,
    *,
    n_samples: int,
    seed: int,
) -> dict[str, Any]:
    wm = _make_wm_for_method(method, seed=seed)
    if bm.family == "F3_titration" and method in ("separable_wm", "torch_wm"):
        targets: list[str] = ["O3", "NO2"]
    else:
        targets = [bm.target]

    wm.fit(bm.df, targets, bm.met_covariates)
    norm_res = wm.normalise(bm.df, n_samples=n_samples, paired=True, seed=seed)
    norm_frame = norm_res.to_normet_frame(species=bm.target)
    scores = evaluate_benchmark(bm, norm_frame, shuffle_control=True, seed=seed)
    sim_diag = benchmark_interaction_diagnostics(
        bm, n_draws=min(128, max(32, n_samples)), seed=seed
    )
    gain = norm_res.separability_gain
    return {
        "family": bm.family,
        "method": method,
        "met_head": wm.fit_metadata_.get("met_head", wm.config.met_head),
        "coordinate_order": wm.fit_metadata_.get("coordinate_order", "warp_then_mix"),
        "gammas": tuple(round(float(g), 3) for g in wm.fit_metadata_["gammas"]),
        "optimal_gamma": float(wm.fit_metadata_["gammas"][0]),
        **scores,
        "sim_rho_raw": float(sim_diag["rho_raw"]),
        "sim_rho_log": float(sim_diag["rho_log"]),
        "mean_rho_lb_raw": float(gain["mean_rho_lb_raw"]),
        "mean_rho_lb_latent": float(gain["mean_rho_lb_latent"]),
        "rho_lb_reduction": float(gain["rho_lb_reduction"]),
        "mean_rc_raw": float(gain["mean_rc_raw"]),
        "mean_rc_latent": float(gain["mean_rc_latent"]),
    }


def benchmark_world_models(
    *,
    families: Sequence[str] = (
        "F1_additive",
        "F2_multiplicative",
        "F3_titration",
        "F4_nonstationary",
        "F5_transport",
        "F6_interaction_dial",
    ),
    methods: Sequence[str] = ("raw_linear", "log1p_linear", "separable_wm"),
    n_days: int = 45,
    n_samples: int = 80,
    noise_sd: float = 1.2,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Compare coordinate-fixed baselines against :class:`SeparableWorldModel` across F1–F6."""
    rows: list[dict[str, Any]] = []
    for fam in families:
        bm = make_synthetic_benchmark(
            fam,
            n_days=int(n_days),
            noise_sd=float(noise_sd),
            interaction_lambda=0.6 if fam == "F6_interaction_dial" else 0.5,
            seed=int(seed),
        )
        for method in methods:
            rows.append(_eval_one_run(bm, method, n_samples=int(n_samples), seed=int(seed)))
    return pd.DataFrame(rows)


def benchmark_interaction_sweep(
    *,
    lambdas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    floor_clip_quantile: float = 0.0,
    n_days: int = 45,
    n_samples: int = 80,
    methods: Sequence[str] = ("raw_linear", "log1p_linear", "separable_wm"),
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Sweep ``F6_interaction_dial`` coupling ``interaction_lambda`` and evaluate coordinate discovery."""
    rows: list[dict[str, Any]] = []
    for lam in lambdas:
        bm = make_synthetic_benchmark(
            "F6_interaction_dial",
            n_days=int(n_days),
            interaction_lambda=float(lam),
            floor_clip_quantile=float(floor_clip_quantile),
            noise_sd=1.0,
            seed=int(seed),
        )
        for method in methods:
            row = _eval_one_run(bm, method, n_samples=int(n_samples), seed=int(seed))
            row["interaction_lambda"] = float(lam)
            row["floor_clip_quantile"] = float(floor_clip_quantile)
            rows.append(row)
    out = pd.DataFrame(rows)
    lead_cols = ["interaction_lambda", "floor_clip_quantile", "family", "method"]
    rest_cols = [c for c in out.columns if c not in lead_cols]
    return out[[*lead_cols, *rest_cols]]
