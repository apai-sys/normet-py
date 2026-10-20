# src/normet/plotting.py
"""
Matplotlib-based plotting helpers for ``normet`` outputs.

All public functions accept an optional ``ax``/``axes`` keyword so they can be
composed into larger dashboards. None require optional dependencies — pure
matplotlib + numpy. Each draws in :mod:`normet.style`'s journal style (the window's
too): 7–8.5 pt text, thin axes with ticks outward, frameless legends, concise date
ticks and Okabe–Ito colours.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from . import style as S
from .style import styled
from .utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "polar_plot",
    "pdp_grid",
    "decomposition_stack",
    "scm_dashboard",
    "normalise_plot",
    "plot_bayesian_scm",
    "time_series_plot",
]


# ---------------------------------------------------------------------------
# Polar / wind-rose-style plots
# ---------------------------------------------------------------------------
@styled
def polar_plot(
    df: pd.DataFrame,
    *,
    target: str,
    ws_col: str = "ws",
    wd_col: str = "wd",
    statistic: str = "mean",
    n_bins_ws: int = 8,
    n_bins_wd: int = 36,
    cmap: str = "viridis",
    title: str | None = None,
    ax: Any | None = None,
) -> Any:
    """
    Wind-direction × wind-speed concentration polar plot ("openair" style).

    For each (wind direction, wind speed) bin, aggregate ``target`` with the
    given statistic (mean/median/max/percentile_X). Plot as a polar pcolor.

    Parameters
    ----------
    df : pandas.DataFrame
    target : str
        Column to aggregate (e.g., concentration).
    ws_col, wd_col : str
        Wind speed and wind direction columns (degrees from north).
    statistic : {"mean","median","max","sum"} or "p<NN>" for percentile
    n_bins_ws : int, default 8
        Number of speed bins (linear from 0 to 99th percentile of ws).
    n_bins_wd : int, default 36
        Number of direction bins (10° each by default).
    cmap : str, default "viridis"
    title, ax : standard matplotlib args.

    Returns
    -------
    matplotlib.axes._subplots.PolarAxesSubplot
    """
    import matplotlib.pyplot as plt

    for col in (target, ws_col, wd_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in df.")

    mask = df[[target, ws_col, wd_col]].notna().all(axis=1)
    sub = df.loc[mask, [target, ws_col, wd_col]].copy()
    if sub.empty:
        raise ValueError("No non-null rows for polar plot.")

    # Bin edges
    ws_max = float(np.nanpercentile(sub[ws_col], 99))
    ws_edges = np.linspace(0.0, ws_max if ws_max > 0 else 1.0, n_bins_ws + 1)
    wd_edges = np.linspace(0.0, 360.0, n_bins_wd + 1)
    sub["_ws_bin"] = pd.cut(sub[ws_col], bins=ws_edges, include_lowest=True)  # type: ignore[call-overload]
    sub["_wd_bin"] = pd.cut(sub[wd_col] % 360.0, bins=wd_edges, include_lowest=True)  # type: ignore[call-overload]

    if statistic.startswith("p") and statistic[1:].isdigit():
        q = float(statistic[1:]) / 100.0
        grid = sub.groupby(["_wd_bin", "_ws_bin"], observed=True)[target].quantile(q).unstack()
    elif statistic in ("mean", "median", "max", "min", "sum", "std"):
        grid = sub.groupby(["_wd_bin", "_ws_bin"], observed=True)[target].agg(statistic).unstack()
    else:
        raise ValueError(f"Unsupported statistic: {statistic}")

    Z = grid.to_numpy(dtype=float)
    theta = np.deg2rad(0.5 * (wd_edges[:-1] + wd_edges[1:]))
    r = 0.5 * (ws_edges[:-1] + ws_edges[1:])
    Theta, R = np.meshgrid(theta, r, indexing="ij")

    if ax is None:
        fig, ax = plt.subplots(
            subplot_kw={"projection": "polar"}, figsize=(4.2, 3.6), layout="constrained"
        )
    ax.set_theta_zero_location("N")  # type: ignore[union-attr]
    ax.set_theta_direction(-1)  # type: ignore[union-attr]  # clockwise, as a compass reads
    mesh = ax.pcolormesh(Theta, R, Z, cmap=cmap, shading="auto", rasterized=True)
    ax.set_thetagrids(  # type: ignore[union-attr]
        [0, 90, 180, 270], ["N", "E", "S", "W"]
    )
    ax.tick_params(axis="y", labelsize=S.SMALL_SIZE)
    ax.grid(True, color="white", lw=0.4, alpha=0.6)
    ax.spines["polar"].set_linewidth(S.RC["axes.linewidth"])
    ax.set_rlabel_position(112.5)  # type: ignore[union-attr]
    if ax.figure is not None:
        S.colorbar(
            ax.figure, mesh, ax, label=f"{statistic} {S.formula(target)}", fraction=0.05, pad=0.1
        )
    ax.set_title(title or f"{S.formula(target)} by wind direction and speed")
    return ax


# ---------------------------------------------------------------------------
# PDP grid
# ---------------------------------------------------------------------------
@styled
def pdp_grid(
    pdp_df: pd.DataFrame,
    *,
    cols: int = 3,
    sharey: bool = False,
    figsize_per: tuple[float, float] = (3.2, 2.2),
    title: str | None = None,
    ylabel: str | None = "Partial dependence",
) -> Any:
    """
    Faceted grid of partial-dependence curves from :func:`normet.pdp`.

    Parameters
    ----------
    pdp_df : pandas.DataFrame
        Output of :func:`normet.pdp` with columns ``[variable, value, pdp_mean,
        pdp_std]``.
    cols : int, default 3
        Number of columns in the grid.
    sharey : bool, default False
        Share the y-axis across panels.
    figsize_per : (float, float), default (3.2, 2.2)
        Figure size per panel.
    title : str, optional
        A title over the grid.
    ylabel : str, optional
        The prediction's label, once, beside the grid.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    for col in ("variable", "value", "pdp_mean"):
        if col not in pdp_df.columns:
            raise ValueError(f"`pdp_df` missing column '{col}'.")

    variables = list(pdp_df["variable"].drop_duplicates())
    n = len(variables)
    if n == 0:
        raise ValueError("No variables in pdp_df.")
    cols = max(1, min(cols, n))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per[0] * cols, figsize_per[1] * rows),
        sharey=sharey,
        squeeze=False,
        layout="constrained",
    )

    for k, var in enumerate(variables):
        ax = axes[k // cols, k % cols]
        sub = pdp_df[pdp_df["variable"] == var].sort_values("value")
        if "pdp_std" in sub.columns and sub["pdp_std"].notna().any():
            lo = sub["pdp_mean"] - sub["pdp_std"]
            hi = sub["pdp_mean"] + sub["pdp_std"]
            ax.fill_between(sub["value"], lo, hi, color=S.COMPONENT, alpha=0.15, lw=0)
        ax.plot(sub["value"], sub["pdp_mean"], lw=1.4, color=S.COMPONENT)
        ax.set_xlabel(str(var))
        ax.margins(x=0)
        S.grid(ax)

    # Hide unused axes
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].set_visible(False)

    if ylabel:
        fig.supylabel(ylabel)
    if title:
        fig.suptitle(title)
    return fig


# ---------------------------------------------------------------------------
# Decomposition stack plot
# ---------------------------------------------------------------------------
@styled
def decomposition_stack(
    decomp_df: pd.DataFrame,
    *,
    observed_col: str = "observed",
    exclude: Iterable[str] = ("observed", "model_pred", "residual", "base"),
    cmap: str = "tab20",
    title: str | None = None,
    ax: Any | None = None,
) -> Any:
    """
    Stacked-area visualisation of a decomposition output.

    Suitable for the nested-normalisation
    decompositions in :mod:`normet.analysis.decomposition`.

    Parameters
    ----------
    decomp_df : pandas.DataFrame
        Indexed by date, with an ``observed`` column and one column per
        feature contribution.
    observed_col : str, default "observed"
        Overlay the observed series as a black line on top of the stack.
    exclude : iterable of str
        Column names to skip (defaults handle the SHAP convention).
    cmap : str, default "tab20"
        Kept for compatibility; the contributions take the style's palette.
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if not provided.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the stacked-area plot.
    """
    import matplotlib.pyplot as plt

    if observed_col not in decomp_df.columns:
        raise ValueError(f"`{observed_col}` not in decomp_df.")
    contrib_cols = [c for c in decomp_df.columns if c not in set(exclude)]
    if not contrib_cols:
        raise ValueError("No contribution columns found to stack.")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3), layout="constrained")
    idx = decomp_df.index
    Y = decomp_df[contrib_cols].to_numpy(dtype=float).T
    colours = [S.PALETTE[i % len(S.PALETTE)] for i in range(len(contrib_cols))]
    ax.stackplot(idx, Y, labels=contrib_cols, colors=colours, alpha=0.85, lw=0)
    ax.plot(idx, decomp_df[observed_col].to_numpy(), color=S.INK, lw=0.8, label=observed_col)
    ax.set_title(title or "Decomposition")
    ax.set_ylabel("Contribution")
    if isinstance(idx, pd.DatetimeIndex):
        S.format_time_axis(ax)
    ax.margins(x=0)
    S.grid(ax)
    S.legend(ax, loc="upper right", ncols=min(4, len(contrib_cols) + 1))
    return ax


# ---------------------------------------------------------------------------
# SCM dashboard
# ---------------------------------------------------------------------------
@styled
def scm_dashboard(
    scm_result: Any,
    *,
    cutoff_date: str,
    diagnostics: dict | None = None,
    title: str = "SCM dashboard",
    ylabel: str = "Outcome",
) -> Any:
    """Three-panel summary of a synthetic-control fit.

    Panels:

    1. Observed vs. synthetic series with a cutoff line.
    2. Estimated effect path.
    3. Top-k donor weights (if available).

    Parameters
    ----------
    scm_result : dict or DataFrame
        Output of any SCM backend.
    cutoff_date : str
        Cutoff line.
    diagnostics : dict, optional
        Output of :func:`scm_diagnostics`. If given, the donor-weights panel
        uses its ``top_donors`` summary.
    title : str
    ylabel : str, default "Outcome"
        The outcome's name, on the top panel's y axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if isinstance(scm_result, pd.DataFrame):
        synth, weights = scm_result, None
    else:
        synth = scm_result.get("synthetic") if hasattr(scm_result, "get") else None  # type: ignore[assignment]
        weights = scm_result.get("weights") if hasattr(scm_result, "get") else None
    if synth is None or "effect" not in synth.columns:
        raise ValueError("`scm_result` must have a synthetic DataFrame with 'effect'.")

    fig = plt.figure(figsize=(8, 6), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 2, 2])
    a1 = fig.add_subplot(gs[0])
    a2 = fig.add_subplot(gs[1], sharex=a1)
    a3 = fig.add_subplot(gs[2])
    cutoff_ts = pd.to_datetime(cutoff_date)
    dated = isinstance(synth.index, pd.DatetimeIndex)

    a1.plot(synth.index, synth["observed"], label="observed", lw=1.2, color=S.OBSERVED)
    a1.plot(synth.index, synth["synthetic"], label="synthetic", lw=1.2, ls="--", color=S.MODELLED)
    S.cutoff_line(a1, cutoff_ts)
    a1.set_title(title)
    a1.set_ylabel(ylabel)
    S.legend(a1)
    S.grid(a1)

    a2.plot(synth.index, synth["effect"], color=S.MODELLED, lw=1.2)
    S.zero_line(a2)
    S.cutoff_line(a2, cutoff_ts)
    a2.set_ylabel("Effect (observed − synthetic)")
    S.grid(a2)
    if dated:
        S.format_time_axis(a2)
    S.outer_ticks([a1, a2])

    if diagnostics and diagnostics.get("top_donors"):
        td = diagnostics["top_donors"]
        names = [str(n) for n, _ in td][::-1]
        vals = [float(v) for _, v in td][::-1]
        a3.barh(names, vals, color=S.NEUTRAL, height=0.6)
        a3.set_title(
            f"Top donors  |  HHI={diagnostics.get('hhi', float('nan')):.3f}  "
            f"|  effective_N={diagnostics.get('effective_n_donors', float('nan')):.1f}"
        )
        a3.set_xlabel("weight")
        S.grid(a3, axis="x")
    elif weights is not None and len(weights) > 0:
        w = weights.sort_values(ascending=False).head(10)[::-1]
        a3.barh([str(i) for i in w.index], w.values, color=S.NEUTRAL, height=0.6)
        a3.set_title("Top donor weights")
        a3.set_xlabel("weight")
        S.grid(a3, axis="x")
    else:
        a3.text(0.5, 0.5, "No donor weights available", ha="center", va="center", color=S.NEUTRAL)
        a3.axis("off")
    return fig


# ---------------------------------------------------------------------------
# Normalisation result plot (#26)
# ---------------------------------------------------------------------------
@styled
def normalise_plot(
    result_df: pd.DataFrame,
    *,
    observed_col: str = "observed",
    normalised_col: str = "normalised",
    ci_low: str | None = None,
    ci_high: str | None = None,
    resample: str | None = None,
    title: str | None = None,
    ylabel: str = "Concentration",
    ax: Any | None = None,
) -> Any:
    """
    Plot observed vs. normalised (deweathered) time series.

    Designed for the output of :func:`normet.normalise` and
    :func:`normet.do_all` (which include ``observed`` and ``normalised``
    columns). Optionally overlays a shaded credible / quantile band.

    Parameters
    ----------
    result_df : pandas.DataFrame
        Indexed by ``date``, with at least ``observed`` and ``normalised``
        columns (e.g. the output of :func:`normet.normalise`).
    observed_col, normalised_col : str
        Column names for the observed and deweathered series.
    ci_low, ci_high : str, optional
        Column names for the lower and upper confidence/quantile band
        (e.g. ``"q025"`` and ``"q975"`` from ``return_quantiles``).
    resample : str, optional
        Pandas resample rule (e.g. ``"D"`` for daily, ``"W"`` for weekly).
        If given, the series are resampled to this frequency before plotting.
    title : str, optional
    ylabel : str, default "Concentration"
    ax : matplotlib axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    for col in (observed_col, normalised_col):
        if col not in result_df.columns:
            raise ValueError(f"Column '{col}' not found in result_df.")

    df = result_df.copy()
    if resample:
        df = df.resample(resample).mean()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3), layout="constrained")

    # A dense record (hourly over months) in thin lines; a short or averaged one fuller.
    dense = len(df) > 1500
    ax.plot(
        df.index,
        df[observed_col],
        color=S.OBSERVED,
        lw=0.5 if dense else 1.0,
        alpha=0.55 if dense else 0.8,
        label="Observed",
    )
    if ci_low and ci_high and ci_low in df.columns and ci_high in df.columns:
        ax.fill_between(
            df.index,
            df[ci_low],
            df[ci_high],
            color=S.MODELLED,
            alpha=0.18,
            lw=0,
            label="Uncertainty band",
        )
    ax.plot(
        df.index,
        df[normalised_col],
        color=S.MODELLED,
        lw=0.8 if dense else 1.4,
        label="Normalised (deweathered)",
    )

    ax.set_title(title or "Observed vs. Normalised concentration")
    ax.set_ylabel(ylabel)
    if isinstance(df.index, pd.DatetimeIndex):
        S.format_time_axis(ax)
    ax.margins(x=0)
    S.grid(ax)
    S.legend(ax, loc="upper right", ncols=3)
    return ax


# ---------------------------------------------------------------------------
# Bayesian SCM posterior band plot (#29)
# ---------------------------------------------------------------------------
@styled
def plot_bayesian_scm(
    result: Any,
    *,
    cutoff_date: str,
    ci_level: float = 0.95,
    title: str = "Bayesian SCM",
    ax: Any | None = None,
) -> Any:
    """
    Plot a Bayesian SCM fit with posterior credible bands.

    Parameters
    ----------
    result : dict
        Output of :func:`normet.causal.bayesian_scm.bayesian_scm`. Must
        contain a ``"synthetic"`` DataFrame with columns
        ``[observed, synthetic, synthetic_low, synthetic_high,
        effect, effect_low, effect_high]``.
    cutoff_date : str
        Treatment cutoff (drawn as a vertical dashed line).
    ci_level : float, default 0.95
        Displayed in the legend label (cosmetic only).
    title : str, default "Bayesian SCM"
    ax : matplotlib axes, optional
        If given, draw into this axes (only first panel). Otherwise a new
        2-row figure is created and returned.

    Returns
    -------
    matplotlib.figure.Figure
        The figure with two panels (observed/synthetic and effect path).
    """
    import matplotlib.pyplot as plt

    if isinstance(result, pd.DataFrame):
        synth = result
    else:
        synth = result.get("synthetic") if hasattr(result, "get") else None  # type: ignore[assignment]

    if synth is None or not isinstance(synth, pd.DataFrame):
        raise ValueError("`result['synthetic']` must be a pandas DataFrame.")
    for col in ("observed", "synthetic", "effect"):
        if col not in synth.columns:
            raise ValueError(f"'synthetic' DataFrame is missing column '{col}'.")

    cutoff_ts = pd.to_datetime(cutoff_date)
    pct = int(ci_level * 100)

    if ax is not None:
        fig = ax.figure
        a1 = ax
        a2 = None
    else:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, layout="constrained")
    dated = isinstance(synth.index, pd.DatetimeIndex)

    # Panel 1: observed vs. synthetic
    if "synthetic_low" in synth.columns and "synthetic_high" in synth.columns:
        a1.fill_between(
            synth.index,
            synth["synthetic_low"],
            synth["synthetic_high"],
            color=S.MODELLED,
            alpha=0.18,
            lw=0,
            label=f"{pct}% credible band",
        )
    a1.plot(synth.index, synth["observed"], lw=1.2, label="Observed", color=S.OBSERVED)
    a1.plot(
        synth.index,
        synth["synthetic"],
        lw=1.2,
        ls="--",
        label="Synthetic (posterior mean)",
        color=S.MODELLED,
    )
    S.cutoff_line(a1, cutoff_ts)
    a1.set_title(title)
    a1.set_ylabel("Outcome")
    S.legend(a1)
    S.grid(a1)
    if dated:
        S.format_time_axis(a1)

    # Panel 2: effect path
    if a2 is not None:
        if "effect_low" in synth.columns and "effect_high" in synth.columns:
            a2.fill_between(
                synth.index,
                synth["effect_low"],
                synth["effect_high"],
                color=S.MODELLED,
                alpha=0.18,
                lw=0,
            )
        a2.plot(synth.index, synth["effect"], color=S.MODELLED, lw=1.2, label="Effect")
        S.zero_line(a2)
        S.cutoff_line(a2, cutoff_ts)
        a2.set_ylabel("Effect (observed − synthetic)")
        S.grid(a2)
        if dated:
            S.format_time_axis(a2)
        S.outer_ticks([a1, a2])
    return fig


# ---------------------------------------------------------------------------
# Generic time series plot (#26)
# ---------------------------------------------------------------------------
@styled
def time_series_plot(
    df: pd.DataFrame,
    target: str,
    *,
    ci_low: str | None = None,
    ci_high: str | None = None,
    resample: str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    color: str = S.OBSERVED,
    ax: Any | None = None,
) -> Any:
    """
    Plot a generic time series with optional confidence/uncertainty bands.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame indexed by datetime (or having a DatetimeIndex).
    target : str
        Column name to plot.
    ci_low, ci_high : str, optional
        Column names for the lower and upper bounds of the confidence band.
    resample : str, optional
        Pandas resample rule (e.g. 'D' for daily, 'W' for weekly, 'MS' for month start).
        If given, resamples using the mean.
    title : str, optional
    ylabel : str, optional
    color : str, default the style's blue (#0072B2)
        Line and shade color.
    ax : matplotlib axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if target not in df.columns:
        raise ValueError(f"Column '{target}' not found in DataFrame.")

    df_plot = df.copy()
    if resample:
        df_plot = df_plot.resample(resample).mean()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3), layout="constrained")

    if ci_low and ci_high and ci_low in df_plot.columns and ci_high in df_plot.columns:
        ax.fill_between(
            df_plot.index,
            df_plot[ci_low],
            df_plot[ci_high],
            color=color,
            alpha=0.18,
            lw=0,
            label="Uncertainty band",
        )
    ax.plot(
        df_plot.index,
        df_plot[target],
        color=color,
        lw=0.6 if len(df_plot) > 1500 else 1.2,
        label=S.formula(target),
    )

    ax.set_title(title or f"Time series of {S.formula(target)}")
    ax.set_ylabel(ylabel or S.formula(target))
    if isinstance(df_plot.index, pd.DatetimeIndex):
        S.format_time_axis(ax)
    ax.margins(x=0)
    S.grid(ax)
    S.legend(ax, loc="upper right")
    return ax
