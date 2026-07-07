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

OBS_COLOR = "#9a9a9a"
NORM_COLOR = "#d7191c"
COMP_COLOR = "#2c7bb6"
NOISE_COLOR = "#8e8e93"

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


def emission_figure(res: pd.DataFrame, target: str = "") -> Any:
    """STL-style stack: normalised series on top, one time-scale per panel."""
    import matplotlib.pyplot as plt

    d = indexed_result(res)
    comps = [(col, label) for col, label in EMI_COMPONENTS if col in d.columns]
    n = 1 + len(comps)
    fig, axes = plt.subplots(n, 1, figsize=(10.5, 1.55 * n + 0.8), sharex=True, squeeze=False)
    axes = axes.ravel()

    ax = axes[0]
    if "observed" in d.columns:
        ax.plot(d.index, d["observed"], lw=0.5, color=OBS_COLOR, alpha=0.6, label="observed")
    ax.plot(d.index, d["emi_total"], lw=1.0, color=NORM_COLOR, label="normalised (emi_total)")
    base = float(d["emi_base"].iloc[0]) if "emi_base" in d.columns else float("nan")
    ax.set_title(
        f"Meteorologically normalised {target} — decomposed into time scales "
        f"(baseline = {base:.1f})",
        fontsize=10,
        loc="left",
    )
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncols=2)
    ax.grid(alpha=0.2)

    for ax, (col, label) in zip(axes[1:], comps, strict=False):
        series = d[col]
        if col == "date_unix" and "emi_base" in d.columns:
            series = series + d["emi_base"]  # trend shown as a level, not an anomaly
            ax.plot(d.index, series, lw=1.2, color=COMP_COLOR)
        else:
            color = NOISE_COLOR if col == "emi_noise" else COMP_COLOR
            ax.plot(d.index, series, lw=0.7, color=color)
            ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{label}   σ = {d[col].std():.2f}", fontsize=9, loc="left")
        ax.grid(alpha=0.2)

    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.6)
    return fig


def meteorology_figure(res: pd.DataFrame, target: str = "", top_k: int = 5) -> Any:
    """Total met influence + strongest contributions + influence ranking."""
    import matplotlib.pyplot as plt

    d = indexed_result(res)
    contribs = met_contribution_columns(d)
    sigma = {c: float(d[c].std()) for c in contribs}
    if "met_noise" in d.columns:
        sigma["met_noise"] = float(d["met_noise"].std())
    top = sorted(contribs, key=lambda c: sigma[c], reverse=True)[:top_k]

    n_left = 1 + len(top)
    fig = plt.figure(figsize=(11.5, 1.7 * n_left + 0.8), layout="constrained")
    gs = fig.add_gridspec(n_left, 2, width_ratios=[2.5, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(d.index, d["met_total"], lw=0.7, color="#1c1c1e")
    ax0.axhline(0, color="k", lw=0.5)
    ax0.set_title(
        f"Meteorological influence on {target}: observed − normalised   "
        f"σ = {d['met_total'].std():.2f}",
        fontsize=10,
        loc="left",
    )
    ax0.grid(alpha=0.2)

    prev = ax0
    for i, col in enumerate(top, start=1):
        ax = fig.add_subplot(gs[i, 0], sharex=prev)
        ax.plot(d.index, d[col], lw=0.7, color=COMP_COLOR)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{col}   σ = {sigma[col]:.2f}", fontsize=9, loc="left")
        ax.grid(alpha=0.2)
        prev = ax

    axr = fig.add_subplot(gs[:, 1])
    order = sorted(sigma, key=sigma.get)
    colors = [NOISE_COLOR if c == "met_noise" else COMP_COLOR for c in order]
    axr.barh(order, [sigma[c] for c in order], color=colors)
    axr.set_title("Influence ranking (σ of contribution)", fontsize=9, loc="left")
    axr.grid(alpha=0.2, axis="x")

    return fig
