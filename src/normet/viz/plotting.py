# src/normet/plotting.py
"""
Matplotlib-based plotting helpers for ``normet`` outputs.

All public functions accept an optional ``ax``/``axes`` keyword so they can be
composed into larger dashboards. None require optional dependencies — pure
matplotlib + numpy. Each draws in :mod:`normet.viz.style`'s journal style (the window's
too): 7–8.5 pt text, thin axes with ticks outward, frameless legends, concise date
ticks and Okabe–Ito colours.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from . import style as S
from .style import styled

log = get_logger(__name__)

__all__ = [
    "polar_plot",
    "wind_rose",
    "pollution_rose",
    "pdp_grid",
    "plot_attention_attribution",
    "plot_imputation",
    "decomposition_stack",
    "scm_dashboard",
    "normalise_plot",
    "plot_bayesian_scm",
    "time_series_plot",
    "cycle_profiles",
    "profile_axes",
    "plot_receptor_grid",
    "plot_trajectory_clusters",
    "plot_footprint_inversion",
    "plot_shapley_interactions",
    "plot_honest_did",
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

    # `observed=True` keeps only the bins that occur, so the grid shrinks
    # wherever a direction sector or a speed class is empty (a 10-degree wind
    # record, a week of hours, a calm-speed gap) and the mesh no longer matches
    # the theta x r centres. Reindex to the full categorical levels and let the
    # empty bins draw as gaps.
    grid = grid.reindex(
        index=pd.Index(sub["_wd_bin"].cat.categories),
        columns=pd.Index(sub["_ws_bin"].cat.categories),
    )
    Z = grid.to_numpy(dtype=float)
    Z[~np.isfinite(Z)] = np.nan
    theta = np.deg2rad(0.5 * (wd_edges[:-1] + wd_edges[1:]))
    r = 0.5 * (ws_edges[:-1] + ws_edges[1:])
    Theta, R = np.meshgrid(theta, r, indexing="ij")

    own_fig = ax is None
    fig = None
    try:
        if ax is None:
            fig, ax = plt.subplots(
                subplot_kw={"projection": "polar"}, figsize=(4.2, 3.6), layout="constrained"
            )
        ax.set_theta_zero_location("N")  # type: ignore[union-attr]
        ax.set_theta_direction(-1)  # type: ignore[union-attr]  # clockwise, as a compass reads
        # `shading="nearest"` takes the array at the bin centres, so the
        # reindexed grid's shape is the mesh's shape; the NaNs leave holes.
        mesh = ax.pcolormesh(Theta, R, Z, cmap=cmap, shading="nearest", rasterized=True)
        ax.set_thetagrids(  # type: ignore[union-attr]
            [0, 90, 180, 270], ["N", "E", "S", "W"]
        )
        ax.tick_params(axis="y", labelsize=S.SMALL_SIZE)
        ax.grid(True, color="white", lw=0.4, alpha=0.6)
        ax.spines["polar"].set_linewidth(S.RC["axes.linewidth"])
        ax.set_rlabel_position(112.5)  # type: ignore[union-attr]
        if ax.figure is not None:
            S.colorbar(
                ax.figure,
                mesh,
                ax,
                label=f"{statistic} {S.formula(target)}",
                fraction=0.05,
                pad=0.1,
            )
        ax.set_title(title or f"{S.formula(target)} by wind direction and speed")
    except Exception:
        # A figure made here is this call's to keep alive only if it returns:
        # pyplot holds every figure until closed, so a failed draw would
        # otherwise leave one behind for the rest of the session.
        if own_fig and fig is not None:
            plt.close(fig)
        raise
    return ax


def _rose(
    ax: Any,
    direction: pd.Series,
    value: pd.Series,
    edges: np.ndarray,
    unit: str,
    n_sectors: int,
    cmap: str,
) -> None:
    """Stacked bars by direction sector, as openair's roses: a sector's length the share
    of the rows from it, cut into the classes of *value* between *edges*."""
    import matplotlib.pyplot as plt

    width = 360.0 / n_sectors
    sector = np.floor(((direction.to_numpy(dtype=float) % 360.0) + width / 2) / width) % n_sectors
    klass = np.clip(
        np.searchsorted(edges, value.to_numpy(dtype=float), side="right") - 1, 0, len(edges) - 2
    )
    counts = np.zeros((n_sectors, len(edges) - 1))
    np.add.at(counts, (sector.astype(int), klass.astype(int)), 1)
    share = 100.0 * counts / max(1, counts.sum())
    theta = np.deg2rad(np.arange(n_sectors) * width)
    colors = plt.get_cmap(cmap)(np.linspace(0.15, 0.95, len(edges) - 1))
    bottom = np.zeros(n_sectors)
    for k in range(len(edges) - 1):
        hi = f"{edges[k + 1]:.3g}" if np.isfinite(edges[k + 1]) else ""
        label = f"{edges[k]:.3g}–{hi} {unit}".strip() if hi else f"≥ {edges[k]:.3g} {unit}".strip()
        ax.bar(
            theta,
            share[:, k],
            width=np.deg2rad(width) * 0.9,
            bottom=bottom,
            color=colors[k],
            edgecolor="white",
            lw=0.3,
            label=label,
        )
        bottom += share[:, k]
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise, as a compass reads
    ax.set_thetagrids([0, 90, 180, 270], ["N", "E", "S", "W"])
    from matplotlib.ticker import MaxNLocator

    ax.yaxis.set_major_locator(MaxNLocator(4))  # four rings read; every 2 % crowds them
    ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:.0f}%")
    ax.tick_params(axis="y", labelsize=S.SMALL_SIZE)
    ax.set_rlabel_position(112.5)
    ax.spines["polar"].set_linewidth(S.RC["axes.linewidth"])
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1.0),
        fontsize=S.SMALL_SIZE,
        frameon=False,
        handlelength=1.0,
    )


def _class_edges(values: pd.Series, breaks: Iterable[float] | None, n: int = 5) -> np.ndarray:
    """Class edges for a rose: *breaks* given, else *n* classes from 0 at round steps up
    to the 95th percentile, the last open-ended."""
    # Explicit: np.asarray's and np.arange's stub return shapes disagree
    # across numpy 2.x stub generations.
    edges: np.ndarray
    if breaks is not None:
        edges = np.asarray(sorted(float(b) for b in breaks))
    else:
        top = float(np.nanpercentile(values, 95)) if len(values) else 1.0
        step = float(10 ** np.floor(np.log10(max(top / n, 1e-9))))
        for mult in (1, 2, 2.5, 5, 10):
            if step * mult * n >= top:
                step *= mult
                break
        edges = np.arange(n) * step
    return np.append(edges, np.inf)


@styled
def wind_rose(
    df: pd.DataFrame,
    *,
    ws_col: str = "ws",
    wd_col: str = "wd",
    n_sectors: int = 16,
    breaks: Iterable[float] | None = None,
    cmap: str = "viridis",
    title: str | None = None,
    ax: Any | None = None,
) -> Any:
    """A wind rose (openair's ``windRose``): how often the wind blows from each direction,
    each sector's bar cut into wind speed classes (``breaks``, m/s; five round classes by
    default).

    Returns
    -------
    matplotlib.axes.Axes (polar)
    """
    import matplotlib.pyplot as plt

    for col in (ws_col, wd_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in df.")
    sub = df[[ws_col, wd_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.empty:
        raise ValueError("No rows with both wind speed and direction.")
    if ax is None:
        _fig, ax = plt.subplots(
            subplot_kw={"projection": "polar"}, figsize=(4.4, 3.6), layout="constrained"
        )
    edges = _class_edges(sub[ws_col], breaks)
    _rose(ax, sub[wd_col], sub[ws_col], edges, "m/s", n_sectors, cmap)
    ax.set_title(title or "Wind rose")
    return ax


@styled
def pollution_rose(
    df: pd.DataFrame,
    *,
    target: str,
    wd_col: str = "wd",
    n_sectors: int = 16,
    breaks: Iterable[float] | None = None,
    cmap: str = "magma_r",
    title: str | None = None,
    ax: Any | None = None,
) -> Any:
    """A pollution rose (openair's ``pollutionRose``): how often the wind blows from each
    direction, each sector's bar cut into classes of ``target``'s concentration
    (``breaks``; five round classes by default) -- where the high values come from.

    Returns
    -------
    matplotlib.axes.Axes (polar)
    """
    import matplotlib.pyplot as plt

    for col in (target, wd_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in df.")
    sub = df[[target, wd_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.empty:
        raise ValueError(f"No rows with both {target} and wind direction.")
    if ax is None:
        _fig, ax = plt.subplots(
            subplot_kw={"projection": "polar"}, figsize=(4.4, 3.6), layout="constrained"
        )
    edges = _class_edges(sub[target], breaks)
    _rose(ax, sub[wd_col], sub[target], edges, "", n_sectors, cmap)
    ax.set_title(title or f"{S.formula(target)} by wind direction")
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
    data: pd.DataFrame | None = None,
) -> Any:
    """
    Faceted grid of partial-dependence curves from :func:`normet.pdp`.

    Where ``data`` (the table the model was trained on) is given, each panel marks the
    variable's deciles along its x axis, as scikit-learn's display does: a curve where
    the data are few -- its ends -- is read with that in mind. A variable of a few whole
    values (an hour, a weekday) is drawn as steps with a point at each value.

    Parameters
    ----------
    pdp_df : pandas.DataFrame
        Output of :func:`normet.pdp` with columns ``[variable, value, pdp_mean]``
        and optional uncertainty columns ``[pdp_lower, pdp_upper]`` (native
        quantile envelope) or ``pdp_std`` (symmetric standard-deviation band).
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
    data : pandas.DataFrame, optional
        The training table, for each variable's deciles.

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
        x = pd.to_numeric(sub["value"], errors="coerce").to_numpy(dtype=float)
        # A few whole values -- an hour, a weekday: steps, a point on each.
        discrete = (
            len(x) <= 31 and bool(np.all(np.isfinite(x))) and bool(np.allclose(x, np.round(x)))
        )
        step = "steps-mid" if discrete else "default"
        has_quantile_band = (
            "pdp_lower" in sub.columns
            and "pdp_upper" in sub.columns
            and bool((sub["pdp_lower"].notna() & sub["pdp_upper"].notna()).any())
        )
        if has_quantile_band:
            lo = pd.to_numeric(sub["pdp_lower"], errors="coerce")
            hi = pd.to_numeric(sub["pdp_upper"], errors="coerce")
            ax.fill_between(
                x, lo, hi, color=S.COMPONENT, alpha=0.15, lw=0, step="mid" if discrete else None
            )
        elif "pdp_std" in sub.columns and sub["pdp_std"].notna().any():
            lo = sub["pdp_mean"] - sub["pdp_std"]
            hi = sub["pdp_mean"] + sub["pdp_std"]
            ax.fill_between(
                x, lo, hi, color=S.COMPONENT, alpha=0.15, lw=0, step="mid" if discrete else None
            )
        ax.plot(
            x,
            sub["pdp_mean"],
            lw=1.4,
            color=S.COMPONENT,
            drawstyle=step,
            marker="o" if discrete else None,
            ms=2.2,
        )
        if data is not None and var in data.columns:
            _decile_rug(ax, data[var])
        ax.set_xlabel(str(var))
        ax.margins(x=0.02 if discrete else 0)
        S.grid(ax)

    # Hide unused axes
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].set_visible(False)

    if ylabel:
        fig.supylabel(ylabel)
    if title:
        fig.suptitle(title)
    return fig


def _decile_rug(ax: Any, values: pd.Series) -> None:
    """The variable's deciles as short ticks along the bottom of ``ax`` -- where the
    data lie, so a curve where they are few is read with that in mind."""
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return
    deciles = np.unique(np.percentile(v, np.arange(10, 100, 10)))
    ax.plot(
        deciles,
        np.zeros_like(deciles),
        "|",
        transform=ax.get_xaxis_transform(),
        color=S.INK,
        ms=6,
        mew=0.7,
        clip_on=False,
        zorder=3,
    )


# ---------------------------------------------------------------------------
# Decomposition stack plot
# ---------------------------------------------------------------------------
@styled
def decomposition_stack(
    decomp_df: pd.DataFrame,
    *,
    observed_col: str = "observed",
    exclude: Iterable[str] | None = None,
    cmap: str = "tab20",
    title: str | None = None,
    ax: Any | None = None,
) -> Any:
    """
    Stacked-area visualisation of a decomposition output.

    Suitable for the nested-normalisation
    decompositions in :mod:`normet.analysis.decomposition`. Automatically
    switches to a diverging positive/negative stacked layout when components
    take both positive and negative values (e.g. meteorological anomalies).

    Parameters
    ----------
    decomp_df : pandas.DataFrame
        Indexed by date, with an ``observed`` column and one column per
        feature contribution.
    observed_col : str, default "observed"
        Overlay the observed series as a black line on top of the stack.
    exclude : iterable of str, optional
        Column names to skip. When omitted, summary/diagnostic columns
        (``observed``, ``predicted``, ``met_total``, ``met_explained``,
        ``residual``, ``*_se``, ``*_factor``; plus ``emi_total`` for emission
        decompositions) are excluded automatically so totals and components
        are never double-stacked.
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

    if exclude is not None:
        excl = set(exclude)
        contrib_cols = [
            c
            for c in decomp_df.columns
            if c not in excl and pd.api.types.is_numeric_dtype(decomp_df[c])
        ]
    else:
        excl = {
            "observed",
            "model_pred",
            "predicted",
            "residual",
            "base",
            "met_total",
            "met_explained",
            observed_col,
        }
        is_emission_frame = (
            "emi_total" in decomp_df.columns and "met_total" not in decomp_df.columns
        )
        if is_emission_frame:
            excl.add("emi_total")
        contrib_cols = [
            c
            for c in decomp_df.columns
            if c not in excl
            and not str(c).endswith(("_se", "_ci_low", "_ci_high", "_factor"))
            and not (is_emission_frame and (str(c).endswith("_main") or ":" in str(c)))
            and pd.api.types.is_numeric_dtype(decomp_df[c])
        ]

    if not contrib_cols:
        raise ValueError("No contribution columns found to stack.")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3), layout="constrained")
    idx = decomp_df.index
    Y = decomp_df[contrib_cols].to_numpy(dtype=float).T
    Y_clean = np.where(np.isfinite(Y), Y, 0.0)
    colours = [S.PALETTE[i % len(S.PALETTE)] for i in range(len(contrib_cols))]

    if float(np.min(Y_clean)) < 0.0:
        pos_cum = np.zeros(len(idx), dtype=float)
        neg_cum = np.zeros(len(idx), dtype=float)
        for col, row, colour in zip(contrib_cols, Y_clean, colours, strict=True):
            pos = np.clip(row, 0.0, None)
            neg = np.clip(row, None, 0.0)
            next_pos = pos_cum + pos
            next_neg = neg_cum + neg
            has_pos = bool(np.any(pos > 0.0))
            if has_pos or not np.any(neg < 0.0):
                ax.fill_between(idx, pos_cum, next_pos, color=colour, alpha=0.85, lw=0, label=col)
            if np.any(neg < 0.0):
                ax.fill_between(
                    idx,
                    next_neg,
                    neg_cum,
                    color=colour,
                    alpha=0.85,
                    lw=0,
                    label=None if has_pos else col,
                )
            pos_cum = next_pos
            neg_cum = next_neg
        S.zero_line(ax)
    else:
        ax.stackplot(idx, Y_clean, labels=contrib_cols, colors=colours, alpha=0.85, lw=0)

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
        # Only the numeric series can be averaged: a frame carrying the run's
        # own labels (a "set" column, a station name, a wind sector) would
        # otherwise raise in pandas 2.x, whose Resampler.mean is not
        # numeric_only by default.
        df = df.resample(resample).mean(numeric_only=True)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3), layout="constrained")

    # A dense record (hourly over months) in thin lines; a short or averaged one fuller;
    # either broken where the record has a gap.
    dense = len(df) > 1500
    S.line(
        ax,
        df.index,
        df[observed_col],
        color=S.OBSERVED,
        alpha=0.55 if dense else 0.8,
        label="Observed",
    )
    if ci_low and ci_high and ci_low in df.columns and ci_high in df.columns:
        S.band(ax, df.index, df[ci_low], df[ci_high], color=S.MODELLED, label="Uncertainty band")
    S.line(
        ax,
        df.index,
        df[normalised_col],
        emphasis=True,
        color=S.MODELLED,
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
        # As in normalise_plot: average the numeric columns only, so a label
        # column beside the series does not raise in pandas 2.x.
        df_plot = df_plot.resample(resample).mean(numeric_only=True)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3), layout="constrained")

    if ci_low and ci_high and ci_low in df_plot.columns and ci_high in df_plot.columns:
        S.band(
            ax,
            df_plot.index,
            df_plot[ci_low],
            df_plot[ci_high],
            color=color,
            label="Uncertainty band",
        )
    S.line(ax, df_plot.index, df_plot[target], color=color, label=S.formula(target))

    ax.set_title(title or f"Time series of {S.formula(target)}")
    ax.set_ylabel(ylabel or S.formula(target))
    if isinstance(df_plot.index, pd.DatetimeIndex):
        S.format_time_axis(ax)
    ax.margins(x=0)
    S.grid(ax)
    S.legend(ax, loc="upper right")
    return ax


# ---------------------------------------------------------------------------
# Diurnal, weekly and seasonal profiles (openair's timeVariation)
# ---------------------------------------------------------------------------

#: The cycles a record can be averaged over: (name, how a timestamp is binned, the
#: bins' tick labels, the axis label).
CYCLES: dict[str, tuple[str, list[int], list[str] | None, str]] = {
    "hour": ("hour", list(range(24)), None, "Hour of day"),
    "weekday": (
        "dayofweek",
        list(range(7)),
        ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "Day of week",
    ),
    "month": (
        "month",
        list(range(1, 13)),
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "Month",
    ),
}


def usable_cycles(index: Any) -> list[str]:
    """The cycles a record of this time index can show: the hour of day only when it
    is finer than daily, the day of the week when finer than weekly, the month when it
    spans two months or more."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return []
    idx = index.sort_values()
    # ``asi8`` counts in the index's own resolution, which pandas 2.x preserves
    # (a ``datetime64[s]`` column from Parquet stays seconds). Without the ns
    # conversion a daily s-resolution record reads as 8.6e-5 s apart and is
    # offered an "hour of day" cycle it does not have.
    step = float(np.median(np.diff(idx.as_unit("ns").asi8))) / 1e9
    span = float((idx[-1] - idx[0]).total_seconds())
    out = []
    if step < 86400 * 0.99:
        out.append("hour")
    if step < 7 * 86400 * 0.99:
        out.append("weekday")
    if span >= 2 * 30 * 86400:
        out.append("month")
    return out


def profile_axes(
    ax: Any,
    series: pd.Series,
    cycle: str,
    *,
    color: str = S.OBSERVED,
    label: str | None = None,
    band: str = "ci",
    emphasis: bool = False,
) -> pd.DataFrame:
    """One series' mean over a cycle (``hour``, ``weekday`` or ``month``) on ``ax``,
    with a band: ``ci`` the mean's 95 % confidence interval, ``iqr`` the 25th-75th
    percentiles, or ``none``. Returns the profile drawn (mean, low, high per bin)."""
    attr, bins, names, xlabel = CYCLES[cycle]
    s = pd.Series(np.asarray(series, dtype=float), index=series.index).dropna()
    key = getattr(s.index, attr)
    g = s.groupby(key)
    prof = pd.DataFrame({"mean": g.mean(), "n": g.size()})
    if band == "iqr":
        prof["low"], prof["high"] = g.quantile(0.25), g.quantile(0.75)
    elif band == "ci":
        se = g.std() / np.sqrt(prof["n"].clip(lower=1))
        prof["low"], prof["high"] = prof["mean"] - 1.96 * se, prof["mean"] + 1.96 * se
    prof = prof.reindex(bins)
    x = np.asarray(bins)
    if band in ("ci", "iqr"):
        ax.fill_between(x, prof["low"], prof["high"], color=color, alpha=0.18, lw=0)
    ax.plot(
        x,
        prof["mean"],
        color=color,
        lw=1.4 if emphasis else 1.1,
        marker="o",
        ms=2.2,
        label=label,
    )
    ax.set_xlabel(xlabel)
    if cycle == "weekday" and names is not None:
        ax.set_xticks(x, names)
    elif cycle == "month":
        ax.set_xticks(x, [str(m) for m in bins])  # 1-12: read alike in any language
    else:
        ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlim(x[0] - 0.4, x[-1] + 0.4)
    S.grid(ax)
    return prof


@styled
def cycle_profiles(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    labels: Iterable[str] | None = None,
    colors: Iterable[str] | None = None,
    cycles: Iterable[str] | None = None,
    band: str = "ci",
    ylabel: str | None = None,
    axes: Any | None = None,
) -> Any:
    """The mean of each column over the hour of the day, the day of the week and the
    month -- a record's diurnal, weekly and seasonal cycles side by side, as openair's
    ``timeVariation`` draws them -- with each mean's 95 % confidence interval (``band``:
    ``ci``, ``iqr`` or ``none``).

    Parameters
    ----------
    df : pandas.DataFrame
        Indexed by time (or with a ``date`` column).
    columns : iterable of str
        The series to average, drawn in ``colors`` (the style's palette by default).
    labels : iterable of str, optional
        Their legend entries (the column names by default).
    cycles : iterable of str, optional
        Which of ``hour``, ``weekday``, ``month`` to draw; by default those the record
        can show (:func:`usable_cycles`).
    ylabel : str, optional
        The left panel's y label.
    axes : sequence of Axes, optional
        Draw into these, one per cycle; a new figure otherwise.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    d = df.set_index(pd.to_datetime(df["date"])) if "date" in df.columns else df
    cols = list(columns)
    names = list(labels) if labels is not None else cols
    cs = list(colors) if colors is not None else S.PALETTE
    which = list(cycles) if cycles is not None else usable_cycles(d.index)
    if not which:
        raise ValueError("No cycle to draw: the record needs a time index.")
    if axes is None:
        fig, axes = plt.subplots(
            1,
            len(which),
            figsize=(2.6 * len(which) + 0.4, 2.4),
            layout="constrained",
            squeeze=False,
        )
        axes = list(axes.ravel())
    else:
        axes = list(axes)
        fig = axes[0].figure
    for ax, cycle in zip(axes, which, strict=False):
        for i, (col, name) in enumerate(zip(cols, names, strict=False)):
            profile_axes(
                ax,
                d[col],
                cycle,
                color=cs[i % len(cs)],
                label=name,
                band=band,
                emphasis=i == len(cols) - 1,
            )
    if ylabel:
        axes[0].set_ylabel(ylabel)
    if len(cols) > 1:
        S.legend(axes[0], loc="best")
    return fig


# ---------------------------------------------------------------------------
# Chronos-2 interpretability & imputation figures
# ---------------------------------------------------------------------------
@styled
def plot_attention_attribution(
    attribution: dict[str, Any],
    *,
    top_k: int = 12,
    title: str | None = None,
    figsize: tuple[float, float] = (9.5, 3.6),
) -> Any:
    """Two-panel visualisation of Chronos-2 white-box attention attribution.

    Left panel ranks the top variables by forecast-patch attention share coloured
    by role (``target``, ``known_future``, ``past_only``). Right panel shows how
    attention is distributed across historical context patches / timestamps.

    Parameters
    ----------
    attribution : dict
        Result returned by :func:`normet.chronos_attention_attribution`.
    top_k : int, default 12
        Maximum number of variables to show in the left bar chart.
    title : str, optional
        Figure suptitle.
    figsize : (float, float), default (9.5, 3.6)
        Output figure dimensions in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not isinstance(attribution, dict):
        raise ValueError("`attribution` must be a dict returned by chronos_attention_attribution.")
    for key in ("group_weights", "time_weights"):
        if key not in attribution:
            raise ValueError(f"`attribution` missing key '{key}'.")

    gw_all = attribution["group_weights"]
    if not isinstance(gw_all, pd.DataFrame) or gw_all.empty:
        raise ValueError("`attribution['group_weights']` must be a non-empty DataFrame.")
    for col in ("variable", "role", "share"):
        if col not in gw_all.columns:
            raise ValueError(f"`group_weights` missing column '{col}'.")

    gw = gw_all.sort_values("share", ascending=False).head(max(1, int(top_k))).iloc[::-1]
    role_colors = {
        "target": S.OBSERVED,
        "known_future": S.COMPONENT,
        "past_only": S.MODELLED,
    }
    role_labels = {
        "target": "Target (self)",
        "known_future": "Known future",
        "past_only": "Past only",
    }
    colors = [role_colors.get(str(r), S.NEUTRAL) for r in gw["role"]]

    fig, (ax_g, ax_t) = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    ax_g.barh(
        gw["variable"].astype(str),
        gw["share"].to_numpy(dtype=float) * 100.0,
        color=colors,
        height=0.65,
    )
    ax_g.set_xlabel("Forecast-patch attention share (%)")
    ax_g.tick_params(axis="y", length=0)
    S.grid(ax_g, axis="x")

    present_roles = [
        r for r in ("target", "known_future", "past_only") if r in set(gw["role"].astype(str))
    ]
    if len(present_roles) > 1:
        handles = [Patch(facecolor=role_colors[r], label=role_labels[r]) for r in present_roles]
        ax_g.legend(handles=handles, loc="lower right", frameon=False)

    if "target_self_share" in attribution and "covariate_share" in attribution:
        self_pct = 100.0 * float(attribution["target_self_share"])
        cov_pct = 100.0 * float(attribution["covariate_share"])
        ax_g.set_title(f"Variable attribution (self {self_pct:.1f}%, cov {cov_pct:.1f}%)")
    else:
        ax_g.set_title("Variable attribution")

    tw = attribution["time_weights"]
    if isinstance(tw, pd.DataFrame) and not tw.empty and "share" in tw.columns:
        y = tw["share"].to_numpy(dtype=float) * 100.0
        has_dates = "date" in tw.columns and bool(tw["date"].notna().any())
        x_t: np.ndarray[Any, Any]
        if has_dates:
            x_t = pd.to_datetime(tw["date"]).to_numpy()
            S.format_time_axis(ax_t)
        elif "patch_idx" in tw.columns:
            x_t = pd.to_numeric(tw["patch_idx"], errors="coerce").to_numpy(dtype=float)
            ax_t.set_xlabel("Context patch")
        else:
            x_t = np.arange(len(tw), dtype=float)
            ax_t.set_xlabel("Context patch")
        ax_t.fill_between(x_t, 0.0, y, color=S.COMPONENT, alpha=0.15, lw=0)
        ax_t.plot(x_t, y, color=S.COMPONENT, lw=1.3)
        ax_t.set_ylabel("Temporal attention share (%)")
        ax_t.set_title("Context lag profile")
        ax_t.margins(x=0)
        S.grid(ax_t)
    else:
        ax_t.text(0.5, 0.5, "No context patches", ha="center", va="center", color=S.NEUTRAL)
        ax_t.axis("off")

    if title:
        fig.suptitle(title)
    return fig


@styled
def plot_imputation(
    impute_df: pd.DataFrame,
    *,
    title: str | None = None,
    ylabel: str = "Value",
    ax: Any | None = None,
) -> Any:
    """Plot observed values alongside Chronos-2 imputed gaps and uncertainty bands.

    Parameters
    ----------
    impute_df : pandas.DataFrame
        Output of :func:`normet.chronos_impute` containing ``[observed, imputed,
        is_imputed]`` and optional ``[lower, upper]`` interval bounds.
    title : str, optional
        Panel title.
    ylabel : str, default "Value"
        Y-axis label.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw onto; creates a new figure when omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    for col in ("observed", "imputed", "is_imputed"):
        if col not in impute_df.columns:
            raise ValueError(f"`impute_df` missing column '{col}'.")

    x_raw = pd.to_datetime(impute_df["date"]) if "date" in impute_df.columns else impute_df.index
    x_vals = x_raw.to_numpy()
    is_dt = isinstance(x_raw, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(x_raw)

    obs = pd.to_numeric(impute_df["observed"], errors="coerce").to_numpy(dtype=float)
    imp = pd.to_numeric(impute_df["imputed"], errors="coerce").to_numpy(dtype=float)
    mask = impute_df["is_imputed"].fillna(False).to_numpy(dtype=bool)

    has_band = (
        "lower" in impute_df.columns
        and "upper" in impute_df.columns
        and bool(mask.any())
        and bool(
            (
                pd.to_numeric(impute_df.loc[mask, "lower"], errors="coerce").notna()
                & pd.to_numeric(impute_df.loc[mask, "upper"], errors="coerce").notna()
            ).any()
        )
    )
    lo = (
        pd.to_numeric(impute_df["lower"], errors="coerce").to_numpy(dtype=float)
        if has_band
        else None
    )
    hi = (
        pd.to_numeric(impute_df["upper"], errors="coerce").to_numpy(dtype=float)
        if has_band
        else None
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
    else:
        fig = ax.figure

    ax.plot(x_vals, obs, lw=0.9, color=S.OBSERVED, label="Observed")

    n = len(mask)
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, flagged in enumerate(mask):
        if flagged and not in_run:
            start = i
            in_run = True
        elif not flagged and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, n))

    band_labeled = False
    for idx_r, (s, e) in enumerate(runs):
        seg_s = max(0, s - 1)
        seg_e = min(n, e + 1)
        y_seg = np.where(np.isfinite(imp[seg_s:seg_e]), imp[seg_s:seg_e], obs[seg_s:seg_e])
        single = (e - s) == 1
        if single:
            ax.axvline(x_vals[s], color=S.MODELLED, alpha=0.22, lw=1.0)
        else:
            ax.axvspan(x_vals[s], x_vals[e - 1], color=S.MODELLED, alpha=0.08, lw=0)
        plot_kw: dict[str, Any] = {
            "lw": 1.3,
            "color": S.MODELLED,
            "marker": "o" if single else None,
            "ms": 3.2,
        }
        if idx_r == 0:
            plot_kw["label"] = "Imputed"
        ax.plot(x_vals[seg_s:seg_e], y_seg, **plot_kw)
        if lo is not None and hi is not None:
            if single:
                if np.isfinite(lo[s]) and np.isfinite(hi[s]):
                    v_kw: dict[str, Any] = {
                        "color": S.MODELLED,
                        "alpha": 0.45,
                        "lw": 2.0,
                    }
                    if not band_labeled:
                        v_kw["label"] = "Predictive interval"
                        band_labeled = True
                    ax.vlines(x_vals[s], lo[s], hi[s], **v_kw)
            else:
                fb_kw: dict[str, Any] = {
                    "color": S.MODELLED,
                    "alpha": 0.22,
                    "lw": 0,
                }
                if not band_labeled:
                    fb_kw["label"] = "Predictive interval"
                    band_labeled = True
                ax.fill_between(x_vals[s:e], lo[s:e], hi[s:e], **fb_kw)

    if is_dt:
        S.format_time_axis(ax)
    ax.set_ylabel(ylabel)
    ax.margins(x=0)
    S.grid(ax)
    S.legend(ax, loc="best")
    if title:
        ax.set_title(title)
    return fig


# Cross-regional transport & trajectory receptor plots
# ---------------------------------------------------------------------------
@styled
def plot_receptor_grid(
    grid: Any,
    *,
    weighted: bool = True,
    cmap: str = "magma_r",
    regions: dict[str, Any] | None = None,
    title: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Visualise a gridded trajectory receptor map (:func:`normet.wpscf` / :func:`normet.wcwt`).

    Parameters
    ----------
    grid : ReceptorGridResult
        Output from :func:`normet.wpscf`, :func:`normet.wcwt`, or
        :func:`normet.gridded_receptor_model`.
    weighted : bool, default True
        Show Polissar-weighted field (``grid.weighted``) if True, else raw
        ``grid.field``.
    cmap : str, default "magma_r"
        Matplotlib colormap name.
    regions : mapping, optional
        Optional ``{name: region}`` source bounding boxes or Shapely geometries
        to outline on top of the receptor grid.
    title : str, optional
        Panel title.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw onto; creates a new figure when omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from ..io.trajectory import _is_bbox, _lon_segments

    Z = np.asarray(grid.weighted if weighted else grid.field, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 4.4), layout="constrained")
    else:
        fig = ax.figure

    mesh = ax.pcolormesh(
        grid.lon_edges,
        grid.lat_edges,
        Z,
        cmap=cmap,
        shading="flat",
        rasterized=True,
    )
    cbar_label = "Weighted probability" if grid.method == "wpscf" else "Weighted concentration"
    S.colorbar(fig, mesh, ax, label=cbar_label)

    ax.plot(
        grid.receptor_lon,
        grid.receptor_lat,
        marker="*",
        ms=10,
        mfc="white",
        mec=S.INK,
        mew=0.9,
        zorder=5,
        label="Receptor",
    )

    if regions:
        for name, region in regions.items():
            if _is_bbox(region):
                x0, y0, x1, y1 = (float(v) for v in region)
                for seg_idx, (lo, hi) in enumerate(_lon_segments(x0, x1)):
                    rect = Rectangle(
                        (lo, y0),
                        hi - lo,
                        y1 - y0,
                        fill=False,
                        ec=S.INK,
                        lw=1.0,
                        ls="--",
                        zorder=4,
                    )
                    ax.add_patch(rect)
                    if seg_idx == 0:
                        ax.text(
                            0.5 * (lo + hi),
                            0.5 * (y0 + y1),
                            str(name),
                            ha="center",
                            va="center",
                            fontsize=S.SMALL_SIZE,
                            color=S.INK,
                            zorder=4,
                        )
            else:
                geoms = getattr(region, "geoms", [region])
                for geom in geoms:
                    if hasattr(geom, "exterior"):
                        xs, ys = geom.exterior.xy
                        ax.plot(xs, ys, color=S.INK, lw=1.0, ls="--", zorder=4)
                c_pt = getattr(region, "centroid", None)
                if c_pt is not None:
                    ax.text(
                        float(c_pt.x),
                        float(c_pt.y),
                        str(name),
                        ha="center",
                        va="center",
                        fontsize=S.SMALL_SIZE,
                        color=S.INK,
                        zorder=4,
                    )

    mean_lat = float(np.mean(grid.lat_centers)) if len(grid.lat_centers) else 0.0
    ax.set_aspect(1.0 / max(float(np.cos(np.radians(mean_lat))), 0.15))
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    if title:
        ax.set_title(title)
    else:
        tag = f"({grid.n_matched_trajectories} trajectories)"
        ax.set_title(f"{grid.method.upper()} transport source map {tag}")
    S.grid(ax)
    return fig


@styled
def plot_trajectory_clusters(
    clusters: Any,
    *,
    show_tsv: bool = False,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Visualise clustered back-trajectory centroids alongside height or TSV curve.

    Parameters
    ----------
    clusters : TrajectoryClusterResult
        Output of :func:`normet.cluster_trajectories`.
    show_tsv : bool, default False
        If True, the right panel shows Total Spatial Variance (TSV) vs *k* to
        inspect the cluster-count elbow; otherwise shows mean transport height
        vs backward age by cluster.
    title : str, optional
        Figure suptitle.
    figsize : (float, float), optional
        Figure size in inches; defaults to ``(9.4, 3.8)``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, (ax_map, ax_r) = plt.subplots(1, 2, figsize=figsize or (9.4, 3.8), layout="constrained")
    cents = clusters.centroids
    summ = clusters.summary.set_index("cluster")
    has_target = "target_mean" in summ.columns

    for idx_c, cid in enumerate(sorted(cents["cluster"].unique())):
        sub = cents[cents["cluster"] == cid].sort_values("age_h", ascending=False)
        color = S.PALETTE[idx_c % len(S.PALETTE)]
        freq = float(sub["freq"].iloc[0])
        lw = 1.2 + 2.0 * freq
        lbl = f"C{int(cid)} ({100.0 * freq:.1f}%)"
        if has_target and np.isfinite(summ.loc[cid, "target_mean"]):
            lbl += f" [{summ.loc[cid, 'target_mean']:.1f}]"
        ax_map.plot(sub["lon"], sub["lat"], color=color, lw=lw, label=lbl)
        ax_map.plot(sub["lon"].iloc[-1], sub["lat"].iloc[-1], "o", ms=4, color=color)
        if not show_tsv:
            ax_r.plot(sub["age_h"], sub["height"], color=color, lw=1.4, label=f"C{int(cid)}")

    lat0 = (
        float(cents[cents["age_h"] == 0.0]["lat"].median())
        if (cents["age_h"] == 0.0).any()
        else float(cents["lat"].iloc[0])
    )
    lon0 = (
        float(cents[cents["age_h"] == 0.0]["lon"].median())
        if (cents["age_h"] == 0.0).any()
        else float(cents["lon"].iloc[0])
    )
    ax_map.plot(lon0, lat0, marker="*", ms=10, mfc="white", mec=S.INK, mew=0.9, zorder=5)
    ax_map.set_aspect(1.0 / max(float(np.cos(np.radians(lat0))), 0.15))
    ax_map.set_xlabel("Longitude (°)")
    ax_map.set_ylabel("Latitude (°)")
    ax_map.set_title(f"Cluster centroids (k={clusters.n_clusters})")
    S.grid(ax_map)
    S.legend(ax_map, loc="best")

    if show_tsv:
        tsv = clusters.tsv
        ax_r.plot(tsv["k"], tsv["tsv"], marker="o", ms=4, color=S.COMPONENT, lw=1.5)
        ax_r.axvline(clusters.n_clusters, color=S.INK, lw=0.9, ls="--")
        ax_r.set_xlabel("Number of clusters (k)")
        ax_r.set_ylabel("Total spatial variance")
        ax_r.set_title("TSV elbow diagnostic")
        S.grid(ax_r)
    else:
        ax_r.set_xlabel("Trajectory age (h)")
        ax_r.set_ylabel("Altitude (m AGL)")
        ax_r.set_title("Vertical transport profile")
        S.grid(ax_r)

    if title:
        fig.suptitle(title)
    return fig


@styled
def plot_footprint_inversion(
    inv: Any,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (9.2, 3.6),
) -> Any:
    """Visualise region source-footprint inversion estimates and time series.

    Left panel ranks regions by mean concentration contribution (with block-
    bootstrap 95% confidence intervals when available). Right panel shows either
    sliding-window :math:`\\beta_g(t)` trends (when ``inv.rolling_beta`` is
    present) or observed vs. predicted + background series.

    Parameters
    ----------
    inv : FootprintInversionResult
        Output of :func:`normet.invert_source_footprints`.
    title : str, optional
        Figure suptitle.
    figsize : (float, float), default (9.2, 3.6)
        Output figure dimensions in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    reg = inv.regions.sort_values("mean_contribution", ascending=True).reset_index(drop=True)
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize, layout="constrained")

    y_pos = np.arange(len(reg))
    contrib = reg["mean_contribution"].to_numpy(dtype=float)
    bars = ax_l.barh(y_pos, contrib, color=S.COMPONENT, height=0.62)
    ax_l.set_yticks(y_pos, reg["region"].astype(str))
    if (
        "beta_ci_low" in reg.columns
        and "beta_ci_high" in reg.columns
        and reg["beta_ci_low"].notna().any()
    ):
        fp = reg["mean_footprint"].to_numpy(dtype=float)
        c_lo = np.maximum(0.0, contrib - reg["beta_ci_low"].to_numpy(dtype=float) * fp)
        c_hi = np.maximum(0.0, reg["beta_ci_high"].to_numpy(dtype=float) * fp - contrib)
        ax_l.errorbar(
            contrib,
            y_pos,
            xerr=np.vstack([c_lo, c_hi]),
            fmt="none",
            ecolor=S.INK,
            elinewidth=1.0,
            capsize=2.5,
        )
    for bar, share in zip(bars, reg["share_of_transport"], strict=True):
        if np.isfinite(share):
            ax_l.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2.0,
                f" {100.0 * float(share):.1f}%",
                va="center",
                ha="left",
                fontsize=S.SMALL_SIZE,
                color=S.INK,
            )
    ax_l.set_xlabel("Mean contribution")
    ax_l.set_title(f"Region contributions (R²={inv.r2:.2f}, RMSE={inv.rmse:.2f})")
    ax_l.margins(x=0.18)
    S.grid(ax_l, axis="x")

    if inv.rolling_beta is not None and not inv.rolling_beta.empty:
        rb = inv.rolling_beta
        rcols = [c for c in rb.columns if c not in ("n_obs", "intercept", "r2")]
        for i, col in enumerate(rcols):
            ax_r.plot(rb.index, rb[col], lw=1.4, color=S.PALETTE[i % len(S.PALETTE)], label=col)
        ax_r.set_ylabel("Unit footprint intensity (beta)")
        ax_r.set_title("Rolling upwind source intensity")
    else:
        cdf = inv.contributions
        ax_r.plot(
            cdf.index, cdf["observed"], lw=0.9, color=S.OBSERVED, alpha=0.65, label="Observed"
        )
        ax_r.plot(cdf.index, cdf["predicted"], lw=1.2, color=S.MODELLED, label="Predicted")
        ax_r.plot(
            cdf.index, cdf["background"], lw=1.0, ls="--", color=S.NEUTRAL, label="Background"
        )
        ax_r.set_ylabel("Concentration")
        ax_r.set_title("Footprint fit & background")
    S.format_time_axis(ax_r)
    S.grid(ax_r)
    S.legend(ax_r, loc="best")

    if title:
        fig.suptitle(title)
    return fig


@styled
def plot_shapley_interactions(
    result: Any,
    *,
    stat: str = "std",
    title: str | None = None,
    cmap: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Visualise main effects (diagonal) and pairwise Shapley interactions (off-diagonal).

    Parameters
    ----------
    result : ShapleyInteractionResult
        Output of :func:`normet.shapley_interactions`.
    stat : {"std", "mean"}, default "std"
        Show temporal standard deviation (``std_matrix``) or signed mean
        (``mean_matrix``).
    title : str, optional
        Panel title.
    cmap : str, optional
        Colormap name; defaults to ``"viridis"`` for ``stat="std"`` and
        ``"RdBu_r"`` for ``stat="mean"``.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw onto; creates a new figure when omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if stat not in ("std", "mean"):
        raise ValueError(f"`stat` must be 'std' or 'mean', got {stat!r}.")
    mat = result.std_matrix if stat == "std" else result.mean_matrix
    arr = mat.to_numpy(dtype=float)
    labels = [str(c) for c in mat.columns]
    n = len(labels)

    if ax is None:
        side = max(4.2, 0.72 * n + 1.8)
        fig, ax = plt.subplots(figsize=(side, max(3.8, side - 0.3)), layout="constrained")
    else:
        fig = ax.figure

    vmax = float(np.nanmax(np.abs(arr))) if arr.size else 1.0
    vmax = max(vmax, 1e-12)
    if stat == "mean":
        eff_cmap = cmap or "RdBu_r"
        im = ax.imshow(arr, cmap=eff_cmap, vmin=-vmax, vmax=vmax, aspect="equal")
        cbar_lbl = "Mean effect / interaction"
    else:
        eff_cmap = cmap or "viridis"
        im = ax.imshow(arr, cmap=eff_cmap, vmin=0.0, vmax=vmax, aspect="equal")
        cbar_lbl = "Temporal σ"

    S.colorbar(fig, im, ax, label=cbar_lbl)
    ax.set_xticks(np.arange(n), labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(n), labels)
    for i in range(n):
        for j in range(n):
            val = arr[i, j]
            norm_v = abs(val) / vmax if stat == "mean" else val / vmax
            txt_color = "white" if norm_v > 0.6 else S.INK
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=S.SMALL_SIZE,
                color=txt_color,
            )
    ax.set_title(title or f"Shapley interactions ({stat}; diag = main effect)")
    return fig


@styled
def plot_honest_did(
    res: Any,
    *,
    title: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Visualise Rambachan & Roth (2023) sensitivity bounds across ``M``.

    Parameters
    ----------
    res : HonestDiDResult or pandas.DataFrame
        Output of :func:`normet.honest_did_bounds` (or its ``.table`` DataFrame).
    title : str, optional
        Panel title.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw onto; creates a new figure when omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if hasattr(res, "table") and isinstance(res.table, pd.DataFrame):
        df = res.table.copy()
        method = str(getattr(res, "method", "relative_magnitude"))
        ci_level = float(getattr(res, "ci_level", 0.95))
        breakdown_m = float(getattr(res, "breakdown_m", np.nan))
    elif isinstance(res, pd.DataFrame):
        df = res.copy()
        method = "relative_magnitude"
        ci_level = 0.95
        breakdown_m = float("nan")
    else:
        raise ValueError("Expected an HonestDiDResult or a DataFrame with M/id/ci columns.")

    required = {"M", "id_low", "id_high", "ci_low", "ci_high"}
    if not required <= set(df.columns):
        raise ValueError(
            f"Missing required HonestDiD columns: {sorted(required - set(df.columns))}"
        )

    df = df.sort_values("M").reset_index(drop=True)
    ms = df["M"].to_numpy(dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 3.4), layout="constrained")

    ci_lo = df["ci_low"].to_numpy(dtype=float)
    ci_hi = df["ci_high"].to_numpy(dtype=float)
    id_lo = df["id_low"].to_numpy(dtype=float)
    id_hi = df["id_high"].to_numpy(dtype=float)
    mid = 0.5 * (id_lo + id_hi)

    span = float(np.ptp(ms)) if ms.size > 1 else 1.0
    cap = 0.018 * max(span, 1e-6)

    ax.vlines(
        ms,
        ci_lo,
        ci_hi,
        color=S.MODELLED,
        lw=1.4,
        label=f"{int(round(ci_level * 100))}% robust CI",
    )
    ax.hlines(ci_lo, ms - cap, ms + cap, color=S.MODELLED, lw=1.2)
    ax.hlines(ci_hi, ms - cap, ms + cap, color=S.MODELLED, lw=1.2)
    ax.vlines(ms, id_lo, id_hi, color=S.INK, lw=4.0, alpha=0.85, label="Identified set")
    ax.scatter(ms, mid, s=18, color="white", edgecolor=S.INK, linewidth=0.9, zorder=4)

    S.zero_line(ax)
    if (
        np.isfinite(breakdown_m)
        and breakdown_m > 0.0
        and float(ms.min()) <= breakdown_m <= float(ms.max())
    ):
        ax.axvline(
            breakdown_m,
            ls=":",
            lw=1.1,
            color=S.NEUTRAL,
            label=f"Breakdown M* = {breakdown_m:.2g}",
        )

    xlabel_map = {
        "relative_magnitude": "M (relative magnitude)",
        "level_magnitude": "M (level magnitude)",
        "smoothness": "M (smoothness bound)",
    }
    ax.set_xlabel(xlabel_map.get(method, "M"))
    ax.set_ylabel("ATT bound")
    default_title = f"HonestDiD sensitivity ({method.replace('_', ' ')})"
    if np.isfinite(breakdown_m):
        default_title = f"{default_title} — M* = {breakdown_m:.2g}"
    ax.set_title(title or default_title)
    ax.margins(x=0.06)
    S.grid(ax)
    S.legend(ax)
    return ax
