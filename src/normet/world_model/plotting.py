"""Publication-grade diagnostic visualisations for ``normet.world_model``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..viz import style as S
from ..viz.style import styled
from ._common import (
    _day_interleaved_folds,
    _ensure_datetime_frame,
    _fast_oof_separability,
    _shifted_power,
    _standardize_fit,
)
from .latent_wm import SeparableWorldModel, TrajectoryRolloutResult

__all__ = [
    "plot_separability_landscape",
    "plot_rollout",
    "plot_jepa_anomalies",
]


@styled
def plot_separability_landscape(
    wm: SeparableWorldModel,
    df: pd.DataFrame,
    *,
    species: str | None = None,
    gammas: Sequence[float] | None = None,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Plot factual RC-score and interaction rho_lb across shifted-power coordinate gamma."""
    import matplotlib.pyplot as plt

    wm._require_fitted()
    df_s = _ensure_datetime_frame(df)
    sp = wm._species[0] if species is None else species
    if sp not in wm._species:
        raise KeyError(f"Species {sp!r} not in fitted model {wm._species}.")
    sp_idx = wm._species.index(sp)

    Y = pd.to_numeric(df_s[sp], errors="coerce").to_numpy(dtype=float)
    X_met, _ = wm._build_met_features(df_s, fit_state=False)
    X_cal = wm._calendar_basis(pd.DatetimeIndex(df_s.index))
    fin = np.isfinite(Y)
    if X_met.shape[1] > 0:
        fin &= np.all(np.isfinite(X_met), axis=1)
    Yv, Xmv, Xcv = Y[fin], X_met[fin], X_cal[fin]
    folds = _day_interleaved_folds(len(Yv), block_len=24, n_splits=3)

    g_grid = np.asarray(gammas if gammas is not None else np.linspace(0.0, 1.0, 9), dtype=float)
    rhos: list[float] = []
    rcs: list[float] = []
    for g in g_grid:
        z = _shifted_power(Yv, float(g))[:, None]
        zn, _, _ = _standardize_fit(z)
        rc, rho, _, _ = _fast_oof_separability(zn, Xmv, Xcv, folds)
        rcs.append(rc)
        rhos.append(rho)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
    else:
        fig = ax.figure

    ax.plot(g_grid, rhos, marker="o", color=S.MODELLED, label=r"Interaction $\rho_{\mathrm{lb}}$")
    ax.plot(g_grid, rcs, marker="s", linestyle="--", color=S.OBSERVED, label="RC-score (linear)")
    g_opt = float(wm._gammas[sp_idx])
    ax.axvline(
        g_opt,
        color=S.SECONDARY,
        linestyle=":",
        linewidth=1.4,
        label=rf"Selected $\gamma^*={g_opt:.2f}$",
    )
    ax.set_xlabel(r"Shifted Box–Cox $\gamma$ (0 = log1p, 1 = linear)")
    ax.set_ylabel("Factual violation score (lower = more separable)")
    ax.set_title(f"Separability landscape — {S.formula(sp)}")
    ax.legend(frameon=False)
    return fig, ax


@styled
def plot_rollout(
    rollout: TrajectoryRolloutResult,
    *,
    species: str | None = None,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Plot actual vs trajectory-level What-If counterfactual rollout with optional uncertainty band."""
    import matplotlib.pyplot as plt

    sp = rollout.species[0] if species is None else species
    if sp not in rollout.species:
        raise KeyError(f"Species {sp!r} not in rollout species {rollout.species}.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.6, 3.6))
    else:
        fig = ax.figure

    x = rollout.actual.index
    y_act = rollout.actual[sp].to_numpy(dtype=float)
    y_cf = rollout.counterfactual[sp].to_numpy(dtype=float)

    ax.plot(x, y_act, color=S.OBSERVED, linewidth=1.4, label="Actual (observed)")
    ax.plot(
        x, y_cf, color=S.MODELLED, linewidth=1.5, linestyle="--", label="Counterfactual rollout"
    )
    if rollout.counterfactual_p10 is not None and rollout.counterfactual_p90 is not None:
        lo = rollout.counterfactual_p10[sp].to_numpy(dtype=float)
        hi = rollout.counterfactual_p90[sp].to_numpy(dtype=float)
        ax.fill_between(x, lo, hi, color=S.MODELLED, alpha=0.18, label="80% bootstrap band")

    cert = rollout.separability_certificate
    badge = (
        f"latent ρ={cert.get('latent_rho_lb', float('nan')):.2f}, "
        f"RC={cert.get('latent_rc_score', float('nan')):.2f}"
    )
    ax.set_title(f"Trajectory What-If rollout — {S.formula(sp)} ({badge})")
    ax.set_ylabel(S.formula(sp))
    S.format_time_axis(ax)
    ax.legend(frameon=False)
    return fig, ax


@styled
def plot_jepa_anomalies(
    df: pd.DataFrame,
    scores: pd.Series,
    *,
    value_col: str = "value",
    events: pd.DataFrame | None = None,
    threshold: float = 3.0,
    axes: Sequence[Any] | None = None,
) -> tuple[Any, Any]:
    """Two-panel plot of observed series and meteorology-invariant AC-JEPA anomaly scores."""
    import matplotlib.pyplot as plt

    df_s = _ensure_datetime_frame(df)
    if value_col not in df_s.columns:
        raise KeyError(f"Column {value_col!r} not found in `df`.")
    sc = (
        scores.reindex(df_s.index)
        if isinstance(scores.index, pd.DatetimeIndex)
        else pd.Series(scores.to_numpy(dtype=float), index=df_s.index)
    )

    if axes is None:
        fig, ax_arr = plt.subplots(2, 1, figsize=(8.0, 4.8), sharex=True)
    else:
        if len(axes) != 2:
            raise ValueError("`axes` must be a pair of (top_ax, bottom_ax).")
        ax_arr = list(axes)
        fig = ax_arr[0].figure

    ax_top, ax_bot = ax_arr[0], ax_arr[1]
    ax_top.plot(
        df_s.index, pd.to_numeric(df_s[value_col], errors="coerce"), color=S.OBSERVED, linewidth=1.2
    )
    ax_top.set_ylabel(S.formula(value_col))
    ax_top.set_title("AC-JEPA meteorology-invariant anomaly detection")

    ax_bot.plot(
        sc.index,
        sc.to_numpy(dtype=float),
        color=S.MODELLED,
        linewidth=1.2,
        label="Innovation z-score",
    )
    ax_bot.axhline(
        float(threshold),
        color=S.REFERENCE,
        linestyle=":",
        linewidth=1.1,
        label=f"Threshold ({threshold:.1f})",
    )
    ax_bot.set_ylabel("Innovation z")
    ax_bot.legend(frameon=False)

    if events is not None and not events.empty:
        for _, row in events.iterrows():
            ax_top.axvspan(row["start"], row["end"], color=S.MODELLED, alpha=0.14)
            ax_bot.axvspan(row["start"], row["end"], color=S.MODELLED, alpha=0.14)

    S.outer_ticks(ax_arr)
    S.format_time_axis(ax_bot)
    return fig, ax_arr
