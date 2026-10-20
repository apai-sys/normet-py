"""A figure style of the kind journals ask for, shared by every figure normet draws.

What it sets is
what Nature Portfolio's, Science's, ACP's and ES&T's guides for figures have in common: a
sans-serif face embedded as TrueType (Type 42), 7 to 8.5 pt text, 0.6 pt axes with ticks
outward, legends without a frame, colours a colour-blind reader can tell apart (Okabe and
Ito's), and files written at 300 dpi. What it offers::

    rc()                     the style's rcParams (light)
    apply_style()            set them for the session (the window does, at start)
    styled                   a decorator: a plotting function draws in the style
    formula(text)            NO2, PM2.5, SO42- written as chemistry writes them
    format_time_axis(ax)     concise date ticks: the year once, not on every tick
    outer_ticks(axes)        stacked panels: the time axis on the bottom one only
    zero_line(ax)            a thin reference line at y = 0
    break_at_gaps(x, *ys)    a NaN put in each gap, so a line stops there
    line(ax, x, y, ...)      a time series drawn broken at its gaps, as wide as it is dense
    band(ax, x, lo, hi, ...) an interval shaded, broken at the same gaps

The figures are drawn in the light colours; the window shows one on a dark theme by
recolouring it (:mod:`normet.gui.theme`), and saves it light.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

#: Okabe and Ito's colour-blind-safe eight (less black), then five of Paul Tol's muted.
PALETTE: list[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#332288",  # indigo
    "#882255",  # wine
    "#44AA99",  # teal
    "#999933",  # olive
    "#AA4499",  # magenta
]

# What a colour stands for, the same in every figure: what
# was measured in the system blue, what a model makes of it in the system red, a third
# series in green; the palette is for categories (sources, variables, sites), one each.
OBSERVED = "#007AFF"  # a measured series
MODELLED = "#FF3B30"  # the normalised series, a synthetic control, a fit
COMPONENT = "#007AFF"  # one part of a decomposition
SECONDARY = "#34C759"  # a third series beside the two
TERTIARY = "#FF9500"  # a fourth
NEUTRAL = "#8E8E93"  # noise, the donors, what is context rather than result
INK = "#1A1A1A"  # text, and a line that is no series' (a 1:1 line, a cutoff)
REFERENCE = "#8E8E93"  # a zero line

#: The light theme's colours.
LIGHT = {
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "savefig.facecolor": "#FFFFFF",
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.edgecolor": "#262626",
    "grid.color": "#E5E5EA",
    "figure.edgecolor": "#FFFFFF",
}

RC: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.prop_cycle": None,  # set below, from PALETTE
    "axes.linewidth": 0.6,
    "axes.titlesize": 8.5,
    "axes.titlelocation": "left",
    "axes.titlepad": 4.0,
    "axes.labelsize": 8.0,
    "axes.labelpad": 3.0,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": False,
    "axes.axisbelow": True,
    "axes.formatter.useoffset": False,
    "axes.formatter.limits": (-4, 5),
    "lines.linewidth": 1.0,
    "lines.markersize": 4.0,
    "patch.linewidth": 0.6,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.pad": 2.5,
    "ytick.major.pad": 2.5,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.8,
    "legend.fontsize": 7.0,
    "legend.title_fontsize": 7.0,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.2,
    "legend.borderaxespad": 0.3,
    "figure.titlesize": 9.0,
    "figure.labelsize": 8.0,
    "figure.titleweight": "normal",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    **LIGHT,
}

#: Text a figure writes itself -- a note in an empty panel -- is set as the axis labels;
#: text that has to fit a crowded panel one step below the tick labels.
TEXT_SIZE = RC["axes.labelsize"]
SMALL_SIZE = 6.0


def rc() -> Any:
    """The style's rcParams, a dict. (Typed Any: matplotlib's newer stubs key rcParams by
    a Literal of every name, which a dict built from strings cannot be.)"""
    from cycler import cycler

    out = dict(RC)
    out["axes.prop_cycle"] = cycler(color=PALETTE)
    return out


def apply_style() -> None:
    """Set the style for the session: every figure made after it is drawn in it."""
    import matplotlib as mpl

    mpl.rcParams.update(rc())


F = TypeVar("F", bound=Callable[..., Any])


def _hold_ticks(fig: Any) -> None:
    """Keep the style's ticks on ``fig``'s axes after the style's context has closed: the
    ticks matplotlib makes later (a zoom, a pan, a resize) read the session's rcParams
    otherwise."""
    for ax in getattr(fig, "axes", []):
        for axis in ("x", "y"):
            for which in ("major", "minor"):
                ax.tick_params(
                    axis=axis,
                    which=which,
                    direction=RC[f"{axis}tick.direction"],
                    length=RC[f"{axis}tick.{which}.size"],
                    width=RC[f"{axis}tick.{which}.width"],
                    labelsize=RC[f"{axis}tick.labelsize"],
                    color=RC[f"{axis}tick.color"],
                    labelcolor=RC[f"{axis}tick.color"],
                )


def styled(fn: F) -> F:
    """A plotting function that draws in the style, whatever the session's rcParams."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        import matplotlib as mpl

        with mpl.rc_context(rc()):
            out = fn(*args, **kwargs)
            fig = out if hasattr(out, "add_subplot") else getattr(out, "figure", None)
            if fig is not None:
                _hold_ticks(fig)
        return out

    return wrapper  # type: ignore[return-value]


# -- formulas ---------------------------------------------------------------------------

#: The periodic table to plutonium, as it reads.
_TABLE = """
H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As
Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd
Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu
"""
_ELEMENTS = frozenset(_TABLE.split())
#: Names that read as formulas and are not: the thermal fractions of organic and elemental
#: carbon (OC1-OC4, EC1-EC3), pyrolysed carbon (OP) and S2, whose digits count no atoms.
_NOT_A_FORMULA = re.compile(r"(?:OC|EC|OP)\d?|S2")
_PM = re.compile(r"(?<![\w.$\\])PM(\d+(?:\.\d+)?)(?![\w$])")
_FORMULA = re.compile(r"(?<![\w.$\\])((?:[A-Z][a-z]?\d*)+)(\^?\d*[+\-−])?(?![\w$])")
_PART = re.compile(r"([A-Z][a-z]?)(\d*)")


def _formula(match: re.Match[str]) -> str:
    body, charge = match.group(1), match.group(2) or ""
    if _NOT_A_FORMULA.fullmatch(body):
        return match.group(0)
    parts = _PART.findall(body)
    if not parts or any(sym not in _ELEMENTS for sym, _ in parts):
        return match.group(0)
    charge = charge.lstrip("^").replace("−", "-")
    if charge and not charge[:-1]:
        # A charge's size written before its sign: Ca2+ is calcium's ion, one atom;
        # SO42- is sulfate's, the count's last digit its charge.
        sym, count = parts[-1]
        if len(parts) == 1 and count:
            parts, charge = [(sym, "")], count + charge
        elif len(count) > 1:
            parts[-1], charge = (sym, count[:-1]), count[-1] + charge
    if not charge and not any(count and count != "1" for _, count in parts):
        return match.group(0)  # nothing to set lower or higher: CO, NO, Na
    out = "".join(sym + (f"_{{{count}}}" if count else "") for sym, count in parts)
    if charge:
        out += f"^{{{charge}}}"
    return f"$\\mathregular{{{out}}}$"


def formula(text: Any) -> str:
    """``text`` with each chemical formula in it written as chemistry writes it -- its
    counts below, its charge above, in the text's own face: NO2, SO2, O3, PM2.5, PM10,
    NH4+, SO42-. What only reads as one is left: CO and NO (nothing to set lower), OC1 and
    EC2 (thermal fractions), and any name that is not all elements (``ws``, ``blh``,
    ``atmos_pres``). A text holding mathtext already is left as it is."""
    text = "" if text is None else str(text)
    if not text or "$" in text:
        return text
    text = _PM.sub(lambda m: f"$\\mathregular{{PM_{{{m.group(1)}}}}}$", text)
    return _FORMULA.sub(_formula, text)


_SUB = str.maketrans("0123456789+-−=()", "₀₁₂₃₄₅₆₇₈₉₊₋₋₌₍₎")
_SUP = str.maketrans("0123456789+-−=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾")
_SCRIPT = re.compile(r"([_^])(?:\{([^{}]*)\}|(\S))")


def _plain_math(body: str) -> str:
    body = re.sub(r"\\mathregular\{(.*)\}", r"\1", body)
    body = _SCRIPT.sub(
        lambda m: (m.group(2) if m.group(2) is not None else m.group(3)).translate(
            _SUB if m.group(1) == "_" else _SUP
        ),
        body,
    )
    return body.replace("{", "").replace("}", "")


def plain(text: str) -> str:
    """``text`` with its mathtext written out in Unicode's sub- and superscripts:
    $\\mathregular{NO_{2}}$ as NO₂, $\\mathregular{SO_{4}^{2-}}$ as SO₄²⁻. A text in
    another script is drawn so: mathtext sets the whole text in its own face, which has
    no Chinese, while plain text falls back to a face that has it."""
    return re.sub(r"\$([^$]*)\$", lambda m: _plain_math(m.group(1)), text or "")


# -- axes -------------------------------------------------------------------------------


def format_time_axis(ax: Any, label: str | None = None) -> None:
    """Concise date ticks on ``ax``'s x axis: each tick written as briefly as its level
    needs (a month as Feb, a day as 05, an hour as 06:00), the year once, as the axis'
    offset, not on every tick."""
    import matplotlib.dates as mdates

    locator = mdates.AutoDateLocator(minticks=3, maxticks=9)  # type: ignore[no-untyped-call]
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator)  # type: ignore[no-untyped-call]
    )
    if label is not None:
        ax.set_xlabel(label)


def outer_ticks(axes: Iterable[Any]) -> None:
    """Panels stacked on one x axis: the tick labels and the x label on the bottom panel
    only, the others keeping their ticks."""
    axes = [ax for ax in axes if ax.get_visible()]
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")
        ax.xaxis.offsetText.set_visible(False)


def zero_line(ax: Any) -> None:
    """A thin reference line at y = 0, behind the data."""
    ax.axhline(0.0, color=REFERENCE, lw=0.6, zorder=0.9)


def cutoff_line(ax: Any, at: Any) -> None:
    """A dotted vertical line at an intervention's date."""
    ax.axvline(at, color=INK, ls=(0, (2, 2)), lw=0.7, zorder=1.5)


def grid(ax: Any, axis: str = "both") -> None:
    """A faint grid under the data."""
    ax.grid(True, axis=axis, color=LIGHT["grid.color"], lw=0.5)
    ax.set_axisbelow(True)


def legend(ax: Any, **kwargs: Any) -> Any:
    """A legend without a frame, in the style's size; ``best`` unless placed."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    kwargs.setdefault("loc", "best")
    return ax.legend(handles, labels, **kwargs)


def colorbar(fig: Any, mappable: Any, ax: Any, label: str | None = None, **kwargs: Any) -> Any:
    """A slim colour bar: as tall as the axes it reads, its outline and ticks as the axes'."""
    kwargs.setdefault("fraction", 0.04)
    kwargs.setdefault("pad", 0.03)
    kwargs.setdefault("aspect", 30)
    cbar = fig.colorbar(mappable, ax=ax, **kwargs)
    cbar.outline.set_linewidth(RC["axes.linewidth"])
    cbar.ax.tick_params(width=RC["xtick.major.width"], length=RC["xtick.major.size"])
    if label:
        cbar.set_label(label)
    return cbar


# -- series -----------------------------------------------------------------------------

#: An interval this many times a record's median spacing is a gap in it, where a series
#: is drawn broken rather than joined across: an hourly record's missing
#: days are, a day-to-day wobble in a sampler's schedule is not.
GAP_SPACINGS = 5.0


def break_at_gaps(x: Any, *ys: Any) -> tuple[Any, tuple[Any, ...]]:
    """``x`` and each of ``ys`` with a NaN row put in every gap -- an interval over
    GAP_SPACINGS times the median -- so a line or a band stops at the gap instead of
    drawing a straight run across it. Returns them unchanged where there is no gap, or
    where ``x`` is not ordered."""
    import numpy as np
    import pandas as pd

    if len(x) < 3:
        return x, ys
    tz = None
    try:
        if isinstance(x, pd.DatetimeIndex) or np.issubdtype(np.asarray(x).dtype, np.datetime64):
            idx = pd.DatetimeIndex(x)
            tz = idx.tz
            # UTC nanoseconds, a time zone or not.
            t = idx.as_unit("ns").to_numpy(dtype="datetime64[ns]").view(np.int64)
            is_time = True
        else:
            t = np.asarray(x).astype(float)
            is_time = False
    except (TypeError, ValueError):
        return x, ys
    step = np.diff(t)
    if np.any(step < 0) or not np.any(step > 0):
        return x, ys
    gaps = np.flatnonzero(step > GAP_SPACINGS * np.median(step[step > 0]))
    if not len(gaps):
        return x, ys
    mid = (t[gaps] + t[gaps + 1]) / 2.0
    t_new = np.insert(t.astype(float), gaps + 1, mid)
    x_new: Any
    if is_time:
        x_new = pd.DatetimeIndex(t_new.astype(np.int64))
        if tz is not None:
            x_new = x_new.tz_localize("UTC").tz_convert(tz)
    else:
        x_new = t_new
    out = tuple(np.insert(np.asarray(y, dtype=float), gaps + 1, np.nan, axis=0) for y in ys)
    return x_new, out


def series_width(n: int, *, emphasis: bool = False) -> float:
    """A line's width for a series of ``n`` points: hairlines for months of hours, fuller
    for a short or averaged series; the series a figure is about a step wider."""
    base = 0.5 if n > 1500 else 0.9 if n > 300 else 1.2
    return base + (0.3 if emphasis else 0.0)


def line(ax: Any, x: Any, y: Any, *, emphasis: bool = False, **kwargs: Any) -> Any:
    """A series drawn broken at its gaps, as wide as it is dense (``lw`` overrides)."""
    xs, (ys,) = break_at_gaps(x, y)
    kwargs.setdefault("lw", series_width(len(xs), emphasis=emphasis))
    return ax.plot(xs, ys, **kwargs)


def band(ax: Any, x: Any, lo: Any, hi: Any, **kwargs: Any) -> Any:
    """An interval shaded between ``lo`` and ``hi``, broken at the gaps a line breaks at."""
    xs, (los, his) = break_at_gaps(x, lo, hi)
    kwargs.setdefault("lw", 0)
    kwargs.setdefault("alpha", 0.18)
    return ax.fill_between(xs, los, his, **kwargs)
