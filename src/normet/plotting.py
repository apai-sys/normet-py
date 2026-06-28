# src/normet/plotting.py
"""
Matplotlib-based plotting helpers for ``normet`` outputs.

All public functions accept an optional ``ax``/``axes`` keyword so they can be
composed into larger dashboards. None require optional dependencies — pure
matplotlib + numpy.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

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
def polar_plot(
    df: pd.DataFrame,
    *,
    value: str,
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

    For each (wind direction, wind speed) bin, aggregate ``value`` with the
    given statistic (mean/median/max/percentile_X). Plot as a polar pcolor.

    Parameters
    ----------
    df : pandas.DataFrame
    value : str
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

    for col in (value, ws_col, wd_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in df.")

    mask = df[[value, ws_col, wd_col]].notna().all(axis=1)
    sub = df.loc[mask, [value, ws_col, wd_col]].copy()
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
        grid = sub.groupby(["_wd_bin", "_ws_bin"], observed=True)[value].quantile(q).unstack()
    elif statistic in ("mean", "median", "max", "min", "sum", "std"):
        grid = sub.groupby(["_wd_bin", "_ws_bin"], observed=True)[value].agg(statistic).unstack()
    else:
        raise ValueError(f"Unsupported statistic: {statistic}")

    Z = grid.to_numpy(dtype=float)
    theta = np.deg2rad(0.5 * (wd_edges[:-1] + wd_edges[1:]))
    r = 0.5 * (ws_edges[:-1] + ws_edges[1:])
    Theta, R = np.meshgrid(theta, r, indexing="ij")

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.set_theta_zero_location("N")  # type: ignore[union-attr]
    ax.set_theta_direction(-1)  # type: ignore[union-attr]
    mesh = ax.pcolormesh(Theta, R, Z, cmap=cmap, shading="auto")
    cbar = ax.figure.colorbar(mesh, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label(f"{statistic}({value})")
    ax.set_title(title or f"{value} by wind direction × speed")
    try:
        ax.figure.tight_layout()  # type: ignore[union-attr]
    except Exception:
        pass
    return ax


# ---------------------------------------------------------------------------
# PDP grid
# ---------------------------------------------------------------------------
def pdp_grid(
    pdp_df: pd.DataFrame,
    *,
    cols: int = 3,
    sharey: bool = False,
    figsize_per: tuple[float, float] = (4.0, 2.8),
    title: str | None = None,
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
    figsize_per : (float, float), default (4.0, 2.8)
        Figure size per panel.

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
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per[0] * cols, figsize_per[1] * rows),
        sharey=sharey,
        squeeze=False,
    )

    for k, var in enumerate(variables):
        ax = axes[k // cols, k % cols]
        sub = pdp_df[pdp_df["variable"] == var].sort_values("value")
        ax.plot(sub["value"], sub["pdp_mean"], lw=1.8, color="C0")
        if "pdp_std" in sub.columns and sub["pdp_std"].notna().any():
            lo = sub["pdp_mean"] - sub["pdp_std"]
            hi = sub["pdp_mean"] + sub["pdp_std"]
            ax.fill_between(sub["value"], lo, hi, alpha=0.18, color="C0")
        ax.set_title(var)
        ax.grid(alpha=0.2)

    # Hide unused axes
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].set_visible(False)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Decomposition stack plot
# ---------------------------------------------------------------------------
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

    Suitable for the leave-one-out
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
        Matplotlib colormap name.
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
        _, ax = plt.subplots(figsize=(11, 4))
    idx = decomp_df.index
    Y = decomp_df[contrib_cols].to_numpy(dtype=float).T
    ax.stackplot(idx, Y, labels=contrib_cols, alpha=0.85, cmap=cmap)
    ax.plot(idx, decomp_df[observed_col].to_numpy(), color="black", lw=1.2, label=observed_col)
    ax.set_title(title or "Decomposition")
    ax.set_ylabel("Contribution")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", ncol=2, fontsize=8, frameon=False)
    return ax


# ---------------------------------------------------------------------------
# SCM dashboard
# ---------------------------------------------------------------------------
def scm_dashboard(
    scm_result: Any,
    *,
    cutoff_date: str,
    diagnostics: dict | None = None,
    title: str = "SCM dashboard",
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

    fig, (a1, a2, a3) = plt.subplots(
        3, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [3, 2, 2]}
    )
    cutoff_ts = pd.to_datetime(cutoff_date)

    a1.plot(synth.index, synth["observed"], label="observed", lw=1.5)
    a1.plot(synth.index, synth["synthetic"], label="synthetic", lw=1.5, ls="--")
    a1.axvline(cutoff_ts, color="k", ls=":", lw=1)
    a1.set_title(title)
    a1.set_ylabel("Outcome")
    a1.legend(frameon=False)
    a1.grid(alpha=0.2)

    a2.plot(synth.index, synth["effect"], color="tab:red", lw=1.5)
    a2.axhline(0.0, color="k", lw=0.5)
    a2.axvline(cutoff_ts, color="k", ls=":", lw=1)
    a2.set_ylabel("Effect (observed - synthetic)")
    a2.grid(alpha=0.2)

    if diagnostics and diagnostics.get("top_donors"):
        td = diagnostics["top_donors"]
        names = [str(n) for n, _ in td][::-1]
        vals = [float(v) for _, v in td][::-1]
        a3.barh(names, vals, color="tab:blue")
        a3.set_title(
            f"Top donors  |  HHI={diagnostics.get('hhi', float('nan')):.3f}  "
            f"|  effective_N={diagnostics.get('effective_n_donors', float('nan')):.1f}"
        )
        a3.set_xlabel("weight")
    elif weights is not None and len(weights) > 0:
        w = weights.sort_values(ascending=False).head(10)[::-1]
        a3.barh([str(i) for i in w.index], w.values, color="tab:blue")
        a3.set_title("Top donor weights")
        a3.set_xlabel("weight")
    else:
        a3.text(0.5, 0.5, "No donor weights available", ha="center", va="center")
        a3.axis("off")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Normalisation result plot (#26)
# ---------------------------------------------------------------------------
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
        _, ax = plt.subplots(figsize=(11, 4))

    ax.plot(df.index, df[observed_col], color="#2c7bb6", lw=1.2, alpha=0.7, label="Observed")
    ax.plot(df.index, df[normalised_col], color="#d7191c", lw=1.8, label="Normalised (deweathered)")

    if ci_low and ci_high and ci_low in df.columns and ci_high in df.columns:
        ax.fill_between(
            df.index,
            df[ci_low],
            df[ci_high],
            color="#d7191c",
            alpha=0.15,
            label="Uncertainty band",
        )

    ax.set_title(title or "Observed vs. Normalised concentration")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    return ax


# ---------------------------------------------------------------------------
# Bayesian SCM posterior band plot (#29)
# ---------------------------------------------------------------------------
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
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Panel 1: observed vs. synthetic
    a1.plot(synth.index, synth["observed"], lw=1.5, label="Observed", color="#2c7bb6")
    a1.plot(
        synth.index,
        synth["synthetic"],
        lw=1.5,
        ls="--",
        label="Synthetic (posterior mean)",
        color="#d7191c",
    )
    if "synthetic_low" in synth.columns and "synthetic_high" in synth.columns:
        a1.fill_between(
            synth.index,
            synth["synthetic_low"],
            synth["synthetic_high"],
            color="#d7191c",
            alpha=0.15,
            label=f"{pct}% credible band",
        )
    a1.axvline(cutoff_ts, color="k", ls=":", lw=1, label="Cutoff")
    a1.set_title(title)
    a1.set_ylabel("Outcome")
    a1.legend(frameon=False, fontsize=9)
    a1.grid(alpha=0.2)

    # Panel 2: effect path
    if a2 is not None:
        a2.plot(synth.index, synth["effect"], color="tab:orange", lw=1.5, label="Effect")
        if "effect_low" in synth.columns and "effect_high" in synth.columns:
            a2.fill_between(
                synth.index,
                synth["effect_low"],
                synth["effect_high"],
                color="tab:orange",
                alpha=0.2,
            )
        a2.axhline(0.0, color="k", lw=0.5)
        a2.axvline(cutoff_ts, color="k", ls=":", lw=1)
        a2.set_ylabel("Effect (obs \u2212 synthetic)")
        a2.set_xlabel("Date")
        a2.grid(alpha=0.2)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Generic time series plot (#26)
# ---------------------------------------------------------------------------
def time_series_plot(
    df: pd.DataFrame,
    value: str,
    *,
    ci_low: str | None = None,
    ci_high: str | None = None,
    resample: str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    color: str = "#2c7bb6",
    ax: Any | None = None,
) -> Any:
    """
    Plot a generic time series with optional confidence/uncertainty bands.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame indexed by datetime (or having a DatetimeIndex).
    value : str
        Column name to plot.
    ci_low, ci_high : str, optional
        Column names for the lower and upper bounds of the confidence band.
    resample : str, optional
        Pandas resample rule (e.g. 'D' for daily, 'W' for weekly, 'MS' for month start).
        If given, resamples using the mean.
    title : str, optional
    ylabel : str, optional
    color : str, default "#2c7bb6"
        Line and shade color.
    ax : matplotlib axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if value not in df.columns:
        raise ValueError(f"Column '{value}' not found in DataFrame.")

    df_plot = df.copy()
    if resample:
        df_plot = df_plot.resample(resample).mean()

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4))

    ax.plot(df_plot.index, df_plot[value], color=color, lw=1.5, label=value)

    if ci_low and ci_high and ci_low in df_plot.columns and ci_high in df_plot.columns:
        ax.fill_between(
            df_plot.index,
            df_plot[ci_low],
            df_plot[ci_high],
            color=color,
            alpha=0.15,
            label="Uncertainty band",
        )

    ax.set_title(title or f"Time series of {value}")
    ax.set_ylabel(ylabel or value)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    return ax
