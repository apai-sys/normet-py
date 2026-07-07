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

from ._decomp_plots import COMP_COLOR, NOISE_COLOR

#: fast -> meso -> slow -> inf: light to dark as the resample pool widens.
SCALE_COLORS = ["#a6cee3", "#2c7bb6", "#08306b", "#d7191c"]

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


def multiscale_figure(result: dict, target: str = "") -> Any:
    import matplotlib.pyplot as plt

    bands = available_bands(result)
    n = 1 + len(bands)
    fig = plt.figure(figsize=(11.5, 1.7 * n + 0.8), layout="constrained")
    gs = fig.add_gridspec(n, 2, width_ratios=[2.5, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    scales = [
        ("Y_fast", f"Y_{result['fast_days']}"),
        ("Y_meso", f"Y_{result['meso_days']}"),
        ("Y_slow", f"Y_{result['slow_days']}"),
        ("Y_inf", "Y_∞ (Step 2)"),
    ]
    for (key, label), color in zip(scales, SCALE_COLORS, strict=True):
        series = result.get(key)
        if series is not None and len(series):
            ax0.plot(series.index, series.values, lw=1.1, color=color, label=label)
    ax0.set_title(
        f"Scale-space cascade of meteorologically normalised {target}".strip(),
        fontsize=10,
        loc="left",
    )
    ax0.legend(loc="upper right", fontsize=8, frameon=False, ncols=4)
    ax0.grid(alpha=0.2)

    prev = ax0
    sigmas: dict[str, float] = {}
    for i, (key, label) in enumerate(bands, start=1):
        s = result[key]
        sigmas[key] = float(s.std())
        ax = fig.add_subplot(gs[i, 0], sharex=prev)
        ax.plot(s.index, s.values, lw=0.7, color=COMP_COLOR)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{label}   σ = {sigmas[key]:.2f}", fontsize=9, loc="left")
        ax.grid(alpha=0.2)
        prev = ax

    axr = fig.add_subplot(gs[:, 1])
    if sigmas:
        order = sorted(sigmas, key=sigmas.get)
        axr.barh(order, [sigmas[k] for k in order], color=[COMP_COLOR] * len(order))
        axr.set_title("Energy by band (σ)", fontsize=9, loc="left")
        axr.grid(alpha=0.2, axis="x")
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
