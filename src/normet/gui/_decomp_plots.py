"""Decomposition figures for the GUI, matching each method's semantics.

* **emission** (``decom_emi``): the meteorologically normalised series is
  split into time scales::

      emi_total = emi_base + date_unix (trend) + day_julian (seasonal)
                  + weekday (weekly) + hour (diurnal) + emi_noise

  Shown as STL-style stacked component panels (shared x, one component per
  panel, σ annotated) so the eye can compare the scales directly.

* **meteorology** (``decom_met``): the met-driven part of the observations::

      met_total = observed − emi_total
                = met_base + Σ per-met-variable contribution + met_noise

  Shown as the total meteorological influence, the strongest individual
  contributions, and an influence ranking (σ of every contribution).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from normet import style as S

OBS_COLOR = S.OBSERVED
NORM_COLOR = S.MODELLED
COMP_COLOR = S.COMPONENT
NOISE_COLOR = S.NEUTRAL

#: emission component columns → display labels, in decomposition order.
EMI_COMPONENTS = [
    ("date_unix", "Trend (long-term)"),
    ("day_julian", "Seasonal (annual)"),
    ("weekday", "Weekly cycle"),
    ("hour", "Diurnal cycle"),
    ("emi_noise", "Short-term (noise)"),
]

_META_COLS = {
    "date",
    "observed",
    "emi_total",
    "emi_base",
    "emi_noise",
    "met_total",
    "met_base",
    "met_noise",
}
_EMI_COMP_COLS = {c for c, _ in EMI_COMPONENTS}


def indexed_result(res: pd.DataFrame) -> pd.DataFrame:
    return res.set_index("date") if "date" in res.columns else res


def met_contribution_columns(res: pd.DataFrame) -> list[str]:
    """Per-met-variable contribution columns of a ``decom_met`` result,
    in the importance order the decomposition used."""
    return [
        c
        for c in res.columns
        if c not in _META_COLS and c not in _EMI_COMP_COLS and pd.api.types.is_numeric_dtype(res[c])
    ]


def _time_series_axes(axes: list[Any], index: Any) -> None:
    """Panels stacked on one time axis: concise dates on the bottom one only, no margin
    either side of the record, a faint grid."""
    for ax in axes:
        ax.margins(x=0)
        S.grid(ax)
    if isinstance(index, pd.DatetimeIndex):
        S.format_time_axis(axes[-1])
    S.outer_ticks(axes)


def _lw(d: pd.DataFrame) -> float:
    """A line's width for a record this long: hairlines for months of hours."""
    return 0.5 if len(d) > 1500 else 0.9


#: Emission components read better as a cycle than as a series: the component's value
#: averaged by the hour of the day, by the day of the week.
_CYCLE_OF = {"hour": "hour", "weekday": "weekday"}


@S.styled
def emission_figure(
    res: pd.DataFrame, target: str = "", cycles_from: pd.DataFrame | None = None
) -> Any:
    """STL-style stack: normalised series on top, one time-scale per panel. The diurnal
    and weekly components, where the record is fine enough to show them, are drawn as
    their cycles beside the stack -- the mean by hour of day and by day of week, with the
    spread between the 25th and 75th percentiles -- rather than as months of oscillation
    no eye can read. ``cycles_from``: the record the cycles are averaged from, where
    ``res`` is shown at a coarser resolution (a day's means have no hours)."""
    import matplotlib.pyplot as plt

    from normet.plotting import profile_axes, usable_cycles

    d = indexed_result(res)
    full = indexed_result(cycles_from) if cycles_from is not None else d
    comps = [(col, label) for col, label in EMI_COMPONENTS if col in d.columns]
    can = set(usable_cycles(full.index))
    cyclic = [(c, lbl) for c, lbl in comps if _CYCLE_OF.get(c) in can]
    stacked = [(c, lbl) for c, lbl in comps if (c, lbl) not in cyclic]
    n = 1 + len(stacked)
    fig = plt.figure(figsize=(8.5 if cyclic else 8, 1.15 * n + 0.6), layout="constrained")
    gs = fig.add_gridspec(
        n,
        2 if cyclic else 1,
        height_ratios=[1.6] + [1] * len(stacked),
        width_ratios=[3, 1] if cyclic else None,
    )
    axes = [fig.add_subplot(gs[0, 0])]
    for i in range(1, n):
        axes.append(fig.add_subplot(gs[i, 0], sharex=axes[0]))
    lw = _lw(d)
    name = S.formula(target)

    ax = axes[0]
    if "observed" in d.columns:
        S.line(ax, d.index, d["observed"], lw=lw, color=OBS_COLOR, alpha=0.45, label="observed")
    S.line(
        ax, d.index, d["emi_total"], lw=lw + 0.2, color=NORM_COLOR, label="normalised (emi_total)"
    )
    base = float(d["emi_base"].iloc[0]) if "emi_base" in d.columns else float("nan")
    ax.set_title(
        f"Meteorologically normalised {name} — decomposed into time scales (baseline = {base:.1f})"
    )
    ax.set_ylabel(name)
    S.legend(ax, loc="upper right", ncols=2)

    for ax, (col, label) in zip(axes[1:], stacked, strict=False):
        series = d[col]
        if col == "date_unix" and "emi_base" in d.columns:
            series = series + d["emi_base"]  # trend shown as a level, not an anomaly
            S.line(ax, d.index, series, lw=1.0, color=COMP_COLOR)
        else:
            color = NOISE_COLOR if col == "emi_noise" else COMP_COLOR
            S.line(ax, d.index, series, lw=lw, color=color)
            S.zero_line(ax)
        # σ of the record as it is: a coarser view's means would understate it.
        ax.set_title(f"{label}   σ = {full[col].std():.2f}")

    _time_series_axes(axes, d.index)
    fig.align_ylabels(axes)

    # The cycles, one panel each down the right-hand column.
    if cyclic:
        side = gs[:, 1].subgridspec(len(cyclic), 1)
        for i, (col, label) in enumerate(cyclic):
            axc = fig.add_subplot(side[i, 0])
            profile_axes(axc, full[col], _CYCLE_OF[col], color=COMP_COLOR, band="iqr")
            S.zero_line(axc)
            axc.set_title(f"{label}   σ = {full[col].std():.2f}")
    return fig


@S.styled
def meteorology_figure(
    res: pd.DataFrame, target: str = "", top_k: int = 5, cycles_from: pd.DataFrame | None = None
) -> Any:
    """Total met influence + strongest contributions + influence ranking. ``cycles_from``:
    the record the σ are read from, where ``res`` is shown at a coarser resolution."""
    import matplotlib.pyplot as plt

    d = indexed_result(res)
    full = indexed_result(cycles_from) if cycles_from is not None else d
    contribs = met_contribution_columns(d)
    sigma = {c: float(full[c].std()) for c in contribs}
    if "met_noise" in full.columns:
        sigma["met_noise"] = float(full["met_noise"].std())
    top = sorted(contribs, key=lambda c: sigma[c], reverse=True)[:top_k]

    n_left = 1 + len(top)
    fig = plt.figure(figsize=(8.5, 1.15 * n_left + 0.6), layout="constrained")
    gs = fig.add_gridspec(n_left, 2, width_ratios=[3, 1])
    lw = _lw(d)
    name = S.formula(target)

    ax0 = fig.add_subplot(gs[0, 0])
    S.line(ax0, d.index, d["met_total"], lw=lw, color=S.INK)
    S.zero_line(ax0)
    ax0.set_title(
        f"Meteorological influence on {name}: observed − normalised   "
        f"σ = {full['met_total'].std():.2f}"
    )
    left = [ax0]
    for i, col in enumerate(top, start=1):
        ax = fig.add_subplot(gs[i, 0], sharex=ax0)
        S.line(ax, d.index, d[col], lw=lw, color=COMP_COLOR)
        S.zero_line(ax)
        ax.set_title(f"{col}   σ = {sigma[col]:.2f}")
        left.append(ax)
    _time_series_axes(left, d.index)
    fig.align_ylabels(left)

    axr = fig.add_subplot(gs[:, 1])
    order = sorted(sigma, key=sigma.get)
    colors = [NOISE_COLOR if c == "met_noise" else COMP_COLOR for c in order]
    axr.barh(order, [sigma[c] for c in order], color=colors, height=0.6)
    axr.set_title("Influence ranking")
    axr.set_xlabel(f"σ of contribution ({name})" if name else "σ of contribution")
    S.grid(axr, axis="x")
    axr.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        axr.spines[side].set_visible(False)

    return fig
