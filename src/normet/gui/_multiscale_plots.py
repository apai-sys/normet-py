"""Figure for the multi-scale meteorological-influence decomposition
(see :mod:`normet.gui._multiscale` for the scientific rationale).

Layout, left-to-right / top-to-bottom:

1. **Scale-space family** — ``Y_fast``, ``Y_meso``, ``Y_slow``, ``Y_inf``
   overlaid, light-to-dark as the resample pool widens; each still carries
   the data's own local/diurnal texture (they are differently-biased
   estimates of the same quantity, not low-pass-filtered smooths of it) but
   converges toward ``Y_inf`` as the window approaches the full record.
2. One panel per available **detail band** (``D_fast``/``D_meso``/``D_slow``),
   the actual wavelet-detail-coefficient analogues — each isolates the
   meteorological-influence component specific to its timescale — styled
   like the emission-decomposition panels for visual consistency.
3. An **energy-by-band** ranking (σ of each band) on the right, so the
   dominant timescale of residual meteorological influence is a one-glance
   read — the same device used by the meteorology-decomposition figure.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from normet import style as S

from ._decomp_plots import COMP_COLOR, NOISE_COLOR, _lw, _time_series_axes

#: fast -> meso -> slow: light to dark blue as the resample pool widens; inf the
#: normalised series' own colour.
SCALE_COLORS = ["#9ECAE1", "#4292C6", "#08519C", S.MODELLED]

_BAND_DEFS = [
    ("D_fast", "fast_days", "meso_days", "synoptic / sub-monthly residual"),
    ("D_meso", "meso_days", "slow_days", "intra-seasonal residual"),
    ("D_slow", "slow_days", None, "residual non-stationarity"),
]


def _band_label(key: str, hi_key: str, lo_key: str | None, note: str, r: dict) -> str:
    hi = r[hi_key]
    lo = "∞" if lo_key is None else r[lo_key]
    return f"{key} = Y_{hi} − Y_{lo}   ({note})"


def available_bands(result: dict) -> list[tuple[str, str]]:
    """(column_key, display_label) for every band that was computable."""
    out = []
    for key, hi_key, lo_key, note in _BAND_DEFS:
        series = result.get(key)
        if series is not None and len(series):
            out.append((key, _band_label(key, hi_key, lo_key, note, result)))
    return out


@S.styled
def multiscale_figure(result: dict, target: str = "") -> Any:
    import matplotlib.pyplot as plt

    bands = available_bands(result)
    n = 1 + len(bands)
    fig = plt.figure(figsize=(8.5, 1.15 * n + 0.6), layout="constrained")
    gs = fig.add_gridspec(n, 2, width_ratios=[3, 1], height_ratios=[1.6] + [1] * len(bands))

    ax0 = fig.add_subplot(gs[0, 0])
    scales = [
        ("Y_fast", f"Y_{result['fast_days']}"),
        ("Y_meso", f"Y_{result['meso_days']}"),
        ("Y_slow", f"Y_{result['slow_days']}"),
        ("Y_inf", "Y_∞ (Step 2)"),
    ]
    index = None
    for (key, label), color in zip(scales, SCALE_COLORS, strict=True):
        series = result.get(key)
        if series is not None and len(series):
            index = series.index if index is None else index
            ax0.plot(
                series.index,
                series.values,
                lw=0.9 if key == "Y_inf" else 0.7,
                color=color,
                label=label,
                zorder=3 if key == "Y_inf" else 2,
            )
    name = S.formula(target)
    ax0.set_title(f"Scale-space cascade of meteorologically normalised {name}".strip())
    ax0.set_ylabel(name)
    S.legend(ax0, loc="upper right", ncols=4)

    left = [ax0]
    sigmas: dict[str, float] = {}
    for i, (key, label) in enumerate(bands, start=1):
        s = result[key]
        sigmas[key] = float(s.std())
        ax = fig.add_subplot(gs[i, 0], sharex=ax0)
        ax.plot(s.index, s.values, lw=_lw(pd.DataFrame(index=s.index)), color=COMP_COLOR)
        S.zero_line(ax)
        ax.set_title(f"{label}   σ = {sigmas[key]:.2f}")
        left.append(ax)
    _time_series_axes(left, index if index is not None else pd.Index([]))
    fig.align_ylabels(left)

    axr = fig.add_subplot(gs[:, 1])
    if sigmas:
        order = sorted(sigmas, key=sigmas.get)
        axr.barh(order, [sigmas[k] for k in order], color=[COMP_COLOR] * len(order), height=0.6)
        axr.set_title("Energy by band (σ)")
        S.grid(axr, axis="x")
        axr.tick_params(axis="y", length=0)
        for side in ("top", "right"):
            axr.spines[side].set_visible(False)
    else:
        axr.text(
            0.5,
            0.5,
            "No band was\ncomputable",
            ha="center",
            va="center",
            color=NOISE_COLOR,
        )
        axr.axis("off")

    return fig
