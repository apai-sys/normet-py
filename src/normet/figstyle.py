"""Figures for print: laid out at the size a journal prints them, in the lettering its
guide asks for.

On screen a figure is drawn at the size its tab gives it -- 12 in wide, say -- and a
journal prints it at a column's width: scaled down, its 7 pt tick labels come out at 4.
:func:`to_print` lays a figure out again at a print size -- each text at the size the
preset gives its kind (a title, an axis label, a tick label, a legend, a note), the
margins solved anew and the ticks chosen anew for the axes' new length: drawn at the size,
not scaled to it -- and returns what puts the figure back. :func:`preflight` reads what
the publisher's guide asks of the figure as drawn: its size, its smallest text and its
smallest sub- or superscript, its face, its panels. :func:`save_for_print` writes it at
that size exactly, its fonts embedded.

A preset holds what its publisher's own page says, with the page it is from; where a page
is silent -- Copernicus on text size, Elsevier and ACS on panel labels -- normet's own
choice stands, and the preset's notes say which.

    PRESETS                  the journals' guides, by key
    to_print(fig, preset)    the figure at a print size; returns what puts it back
    preflight(fig, preset)   what the guide asks of the figure, met or not
    save_for_print(fig, p)   the file, at the figure's size exactly, fonts embedded
    label_panels(fig)        a b c at the panels' top left
"""

from __future__ import annotations

import functools
import itertools
import re
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .style import formula, plain

MM = 25.4  # mm in an inch
POINTS_PER_MM = 72 / MM
#: matplotlib's mathtext draws a sub- or superscript at this share of its text's size.
SCRIPT_SHRINK = 0.7


@dataclass(frozen=True)
class Preset:
    """What a publisher's guide asks of a figure. Sizes in mm and pt; None where the
    guide says nothing."""

    name: str
    source: str
    widths: dict[str, float]  # the columns a figure fills, by name: mm
    min_width: float | None = None
    max_width: float | None = None
    max_height: float | None = None
    text: float = 8.0  # axis labels, legends, notes
    ticks: float = 7.0
    title: float = 8.0
    min_text: float | None = None
    max_text: float | None = None  # panel labels aside
    min_script: float | None = None  # a sub- or superscript
    label: str = "a"  # "a", "(a)", "A" or "(A)"
    label_bold: bool = True
    label_size: float = 8.0
    max_panels: int | None = None
    dpi: int = 300  # a raster's resolution
    formats: tuple[str, ...] = ("pdf", "eps", "svg", "tiff", "png")
    fonts: tuple[str, ...] = ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans")
    notes: tuple[str, ...] = field(default_factory=tuple)


PRESETS: dict[str, Preset] = {
    "elsevier": Preset(
        name="Elsevier",
        source="https://www.elsevier.com/about/policies-and-standards/author/"
        "artwork-and-media-instructions/artwork-sizing",
        widths={"single column": 90.0, "1.5 columns": 140.0, "double column": 190.0},
        min_width=30.0,
        max_width=190.0,
        text=7.0,
        ticks=7.0,
        title=7.0,
        min_text=7.0,
        min_script=6.0,
        label="(a)",
        label_bold=False,
        label_size=7.0,
        notes=(
            "About 7 pt for normal text, 6 pt or larger for a sub- or superscript: the "
            "guide calls it a rule of thumb.",
            "The guide sets no panel labels: (a) is normet's choice.",
        ),
    ),
    "nature": Preset(
        name="Nature",
        source="https://research-figure-guide.nature.com/figures/"
        "preparing-figures-our-specifications/",
        widths={"single column": 89.0, "double column": 183.0},
        max_width=183.0,
        max_height=170.0,
        text=7.0,
        ticks=6.0,
        title=7.0,
        min_text=5.0,
        max_text=7.0,
        min_script=5.0,
        label="a",
        label_bold=True,
        label_size=8.0,
        formats=("pdf", "eps", "svg"),
        notes=("Vector files with editable layers; no JPEG, TIFF or PNG for a main figure.",),
    ),
    "copernicus": Preset(
        name="Copernicus (ACP, AMT, ...)",
        source="https://www.atmospheric-chemistry-and-physics.net/submission.html",
        widths={"least width": 80.0},
        min_width=80.0,
        label="(a)",
        label_bold=False,
        label_size=8.0,
        formats=("pdf", "eps", "png", "tiff"),
        notes=(
            "Panel labels are required, lower case in brackets: (a), (b).",
            "One font family, sans-serif.",
            "The page sets no text sizes: normet's own, 8 pt and 7 pt for ticks, stand.",
        ),
    ),
    "acs": Preset(
        name="ACS (ES&T)",
        source="https://researcher-resources.acs.org/publish/author_guidelines?coden=esthag",
        widths={"single column": 240 / POINTS_PER_MM, "double column": 504 / POINTS_PER_MM},
        max_width=504 / POINTS_PER_MM,
        max_height=660 / POINTS_PER_MM,
        text=8.0,
        ticks=8.0,
        title=8.0,
        min_text=8.0,
        min_script=4.5,
        label="(a)",
        label_bold=False,
        label_size=8.0,
        max_panels=4,
        formats=("tiff", "pdf", "eps", "png"),
        notes=(
            "A double-column figure is 300 to 504 pt wide (105.8 to 177.8 mm).",
            "The 660 pt depth includes the caption, 12 pt a line.",
            "Fonts no smaller than 8 pt (submission checklist); 4.5 pt in print.",
            "1200 dpi for black-and-white line art, 600 for greyscale, 300 for colour.",
            "The guide sets no panel labels: (a) is normet's choice.",
        ),
    ),
}


def has_script(text: str) -> bool:
    """Whether a text sets some of itself lower or higher (a mathtext _ or ^)."""
    return bool(re.search(r"\$[^$]*[_^][^$]*\$", text or ""))


# -- panels -----------------------------------------------------------------------------


def panel_axes(fig: Any) -> list[Any]:
    """The figure's panels in reading order: its axes but colour bars, an axis off -- a
    legend's own, or a cell left blank -- and a twin drawn over another. The order is by
    the panels' bottom edges, the highest first, ties left to right: a grid's rows read
    top to bottom, and a panel stacked under another reads before one running the
    figure's height beside them."""
    seen: list[tuple[float, ...]] = []
    panels: list[Any] = []
    for ax in fig.axes:
        # matplotlib labels a colour bar it places itself "<colorbar>"; one a figure places
        # on its own cax carries the Colorbar on the axis alone. A legend's key or a cell
        # left blank has its axis off and is no panel either.
        if (
            not ax.get_visible()
            or not ax.axison
            or ax.get_label() == "<colorbar>"
            or getattr(ax, "_colorbar", None) is not None
        ):
            continue
        box = tuple(round(v, 3) for v in ax.get_position().bounds)
        if box in seen:
            continue  # a twin: the same panel
        seen.append(box)
        panels.append(ax)
    return sorted(
        panels, key=lambda a: (-round(a.get_position().y0, 2), round(a.get_position().x0, 2))
    )


def panel_letters(n: int, style: str = "a", start: str = "a") -> list[str]:
    """The labels of ``n`` panels: a b c, (a) (b) (c), A B C or (A) (B) (C), from ``start``."""
    first = ord(start.lower()) - ord("a")
    letters = [chr(ord("a") + first + i) for i in range(n)]
    if style.strip("()").isupper():
        letters = [s.upper() for s in letters]
    return [f"({s})" if style.startswith("(") else s for s in letters]


#: A panel label's artist id, so the checks tell it from the figure's own text.
PANEL_LABEL = "normet-panel-label"


@contextmanager
def _quiet_layout() -> Iterator[None]:
    """matplotlib's layout engines warn where they cannot make a margin large enough --
    while a figure is being raised to the height its panels need, every step but the last.
    The checks say what such a figure lacks (:func:`preflight`); the warning would only
    repeat."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Tight layout not applied")
        warnings.filterwarnings("ignore", message="constrained_layout not applied")
        yield


def _renderer(fig: Any) -> Any:
    with _quiet_layout():
        fig.draw_without_rendering()
    canvas = fig.canvas
    return canvas.get_renderer() if hasattr(canvas, "get_renderer") else fig._get_renderer()


def label_panels(
    fig: Any,
    style: str = "a",
    *,
    bold: bool = True,
    size: float = 8.0,
    start: str = "a",
    face: str | None = None,
) -> list[Any]:
    """Letter the figure's panels (:func:`panel_axes`) at their top left, beyond what each
    draws on its left -- its tick labels and its axis label -- so a column of panels has
    its letters in a line. Returns the labels, each an artist of its panel."""
    from matplotlib.transforms import ScaledTranslation

    panels = panel_axes(fig)
    renderer = _renderer(fig)
    # A column of panels has its letters in a line: each as far out as the column's
    # furthest-reaching tick labels.
    reach: dict[float, float] = {}
    for ax in panels:
        col = round(ax.get_position().x0, 2)
        here = ax.get_window_extent(renderer).x0 - ax.get_tightbbox(renderer).x0
        reach[col] = max(reach.get(col, 0.0), here)
    out = []
    for ax, letter in zip(panels, panel_letters(len(panels), style, start), strict=True):
        far = reach[round(ax.get_position().x0, 2)]
        shift = ScaledTranslation(-far / fig.dpi, 2 / 72, fig.dpi_scale_trans)
        t = ax.text(
            0,
            1,
            letter,
            transform=ax.transAxes + shift,
            ha="left",
            va="bottom",
            fontsize=size,
            fontweight="bold" if bold else "normal",
            clip_on=False,
        )
        if face:
            t.set_fontfamily([face, "sans-serif"])
        t.set_gid(PANEL_LABEL)
        out.append(t)
    return out


#: Blank height, mm, a figure laid out at the height it needs is lowered past: the
#: height its panels' shape leaves unfilled.
TRIM_MM = 5.0
#: The margin, mm, kept round what is drawn where the blank is cut.
PAD_MM = 2.0

#: The least height a panel's data are drawn in, mm: a figure laid out lower than this
#: has panels too low to read, and :func:`to_print` raises it until each has as much.
MIN_PANEL = 12.0


def panel_grid(fig: Any) -> tuple[int, int]:
    """(rows, columns) of the figure's panels."""
    panels = panel_axes(fig)
    return (
        len({round(a.get_position().y0, 2) for a in panels}),
        len({round(a.get_position().x0, 2) for a in panels}),
    )


def suggest_width(fig: Any, preset: Preset) -> float:
    """The width a figure is printed at, mm: the preset's narrowest column for one column
    of panels, its widest for panels side by side -- and where its guide gives only a
    least width (Copernicus), that much for each column of panels."""
    widths = sorted(preset.widths.values())
    cols = panel_grid(fig)[1]
    if cols <= 1:
        return widths[0]
    if len(widths) > 1:
        return widths[-1]
    wide = widths[0] * cols
    return min(wide, preset.max_width) if preset.max_width else wide


def available_face(names: Sequence[str]) -> str | None:
    """The first of ``names`` this machine has a font for."""
    from matplotlib import font_manager as fm

    for name in names:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            return name
        except ValueError:
            continue
    return None


def _points(size: Any) -> float:
    from matplotlib.font_manager import FontProperties

    return float(FontProperties(size=size).get_size_in_points())


def _bounded(size: float, preset: Preset, script: bool = False) -> float:
    """A text's size within the preset's least and greatest; one with a sub- or
    superscript as large as reaching the preset's least script asks, where the greatest
    allows it."""
    if preset.min_text is not None:
        size = max(size, preset.min_text)
    if script and preset.min_script is not None:
        size = max(size, preset.min_script / SCRIPT_SHRINK)
    if preset.max_text is not None:
        size = min(size, preset.max_text)
    return size


def _wrapped(formatter: Any, change: Callable[[str], str]) -> Any:
    """``formatter``, its labels passed through ``change`` -- an axis's labels are written
    by its formatter at every draw (set_xticklabels's too), so that is where they change."""
    from matplotlib.ticker import Formatter

    class Changed(Formatter):  # type: ignore[misc]
        def __call__(self, x: Any, pos: Any = None) -> str:
            return change(formatter(x, pos))

        def format_ticks(self, values: Any) -> list[str]:
            return [change(s) for s in formatter.format_ticks(values)]

        def get_offset(self) -> str:
            return str(formatter.get_offset())

        def set_locs(self, locs: Any) -> None:
            formatter.set_locs(locs)

        def format_data_short(self, value: Any) -> str:
            return str(formatter.format_data_short(value))

    return Changed()


def _ticks_back(axis: Any, formatter: Any, kw: tuple[dict[str, Any], dict[str, Any]]) -> None:
    """An axis's formatter and tick settings as they were; its ticks made again from them."""
    axis.set_major_formatter(formatter)
    for now, was in zip((axis._major_tick_kw, axis._minor_tick_kw), kw, strict=True):
        now.clear()
        now.update(was)
    axis.reset_ticks()


def set_legend_columns(leg: Any, ncols: int) -> None:
    """``leg``'s entries in ``ncols`` columns, laid out again. matplotlib's own
    ``set_ncols`` only notes the number: the entries were packed into columns when the
    legend was made, and stay so. Here they are taken out of their columns, in order, and
    packed into as many anew, each column as the first was."""
    import numpy as np
    from matplotlib.offsetbox import VPacker

    box: Any = getattr(leg, "_legend_handle_box", None)
    columns = list(box.get_children()) if box is not None else []
    ncols = max(1, int(ncols))
    leg._ncols = ncols
    if not columns:
        return
    items = [item for col in columns for item in col.get_children()]
    first = columns[0]
    box._children = [
        VPacker(pad=0, sep=first.sep, align=first.align, children=list(part))
        for part in np.array_split(np.array(items, dtype=object), min(ncols, len(items)))
        if len(part)
    ]
    leg.stale = True


def _legend_fits(fig: Any, preset: Preset, undo: list[Callable[[], Any]]) -> None:
    """Each legend's entries in as few columns as keep it within the figure's width: one
    laid out in a row across a wide panel runs past a column's edge once the figure is
    printed in it (a legend of four, set in a row over a 12 in series, at 89 mm)."""
    from matplotlib.legend import Legend

    renderer = _renderer(fig)
    room = fig.bbox.width
    for leg in fig.findobj(Legend):
        ncols0 = getattr(leg, "_ncols", 1)
        ncols = ncols0
        while ncols > 1 and leg.get_window_extent(renderer).width > room * 0.98:
            ncols -= 1
            set_legend_columns(leg, ncols)
            renderer = _renderer(fig)
        if ncols != ncols0:
            undo.append(functools.partial(set_legend_columns, leg, ncols0))


def to_print(
    fig: Any,
    preset: Preset,
    width: float | None = None,
    height: float | None = None,
    *,
    labels: str | None = None,
    label_bold: bool | None = None,
    start: str = "a",
    formulas: bool = True,
    title: bool = False,
) -> Callable[[], None]:
    """Lay ``fig`` out at ``width`` mm (:func:`suggest_width`'s where None) by ``height``
    mm -- where None the figure's own proportion, raised until each panel has MIN_PANEL mm
    for its data, within the preset's greatest height -- in the preset's lettering, and
    return what puts it back as it was. The titles -- the figure's own and each panel's --
    are a caption's, and are left off but with ``title``.

    Titles, axis labels, tick labels, legends and the figure's own titles take the
    preset's sizes; any other text (a note, a bar's value) the size it had beside the axis
    labels'; none falls below the preset's least or above its greatest. Every text is set
    in the preset's face where this machine has it (Arial, Helvetica, ...), the faces it
    had kept behind it for the characters that face lacks. With ``formulas``, the chemical
    formulas in the text and the tick labels are written as chemistry writes them
    (:func:`normet.style.formula`); with ``labels`` ("a", "(a)", "A", "(A)") the panels are
    lettered. A legend wider than the figure is given fewer columns.
    """
    import matplotlib as mpl
    from matplotlib.layout_engine import PlaceHolderLayoutEngine
    from matplotlib.text import Text

    undo: list[Callable[[], Any]] = []
    w0, h0 = fig.get_size_inches()
    engine0 = fig.get_layout_engine()
    w = (width or suggest_width(fig, preset)) / MM
    h = height / MM if height else w * h0 / w0
    if preset.max_height and not height:
        h = min(h, preset.max_height / MM)
    face = available_face(preset.fonts)
    # Titles, the figure's own and each panel's alike, are a caption's, as a rule: one
    # switch leaves them all off, or draws them all.
    texts = [getattr(fig, "_suptitle", None)]
    for ax in fig.axes:
        texts += [ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)]
    for t in texts:
        if t is None or not t.get_text().strip() or t.get_visible() == title:
            continue
        t.set_visible(title)
        undo.append(functools.partial(t.set_visible, not title))
    base = _points(mpl.rcParams["axes.labelsize"])

    def keep(t: Any) -> None:
        size, family, text = t.get_fontsize(), list(t.get_fontfamily()), t.get_text()
        undo.append(lambda: (t.set_fontsize(size), t.set_fontfamily(family), t.set_text(text)))

    def faced(families: list[str]) -> list[str]:
        return ([face] + [f for f in families if f != face]) if face else families

    role: dict[int, float] = {}
    for ax in fig.axes:
        for t in (ax.title, getattr(ax, "_left_title", None), getattr(ax, "_right_title", None)):
            if t is not None:
                role[id(t)] = preset.title
        role[id(ax.xaxis.label)] = role[id(ax.yaxis.label)] = preset.text
        role[id(ax.xaxis.offsetText)] = role[id(ax.yaxis.offsetText)] = preset.ticks
    for leg in [ax.get_legend() for ax in fig.axes] + list(fig.legends):
        if leg is not None:
            for t in list(leg.get_texts()) + [leg.get_title()]:
                role[id(t)] = preset.text
    for name, size in (
        ("_suptitle", preset.title),
        ("_supxlabel", preset.text),
        ("_supylabel", preset.text),
    ):
        t = getattr(fig, name, None)
        if t is not None:
            role[id(t)] = size
    ticks: set[int] = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                ticks.update((id(tick.label1), id(tick.label2)))

    for t in fig.findobj(Text):
        if id(t) in ticks:
            continue
        keep(t)
        text = formula(t.get_text()) if formulas else t.get_text()
        size = role.get(id(t), t.get_fontsize() * preset.text / base)
        t.set_text(text)
        t.set_fontsize(_bounded(size, preset, has_script(text)))
        t.set_fontfamily(faced(list(t.get_fontfamily())))

    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            shown = axis.get_ticklabels()
            family0 = list(shown[0].get_fontfamily()) if shown else ["sans-serif"]
            major = axis.get_major_formatter()
            # The axis's own tick settings put back as they were, not a size read off its
            # labels: an axis that set none takes the session's at its next ticks, and one
            # given a size here would keep it (a profile panel's ticks came back at 10 pt).
            kw = (dict(axis._major_tick_kw), dict(axis._minor_tick_kw))
            undo.append(functools.partial(_ticks_back, axis, major, kw))
            script = False
            if formulas:
                axis.set_major_formatter(_wrapped(major, formula))
                script = any(has_script(formula(t.get_text())) for t in shown)
            axis.set_tick_params(
                which="both",
                labelsize=_bounded(preset.ticks, preset, script),
                labelfontfamily=faced(family0),
            )

    fig.set_size_inches(w, h, forward=False)
    undo.append(lambda: fig.set_size_inches(w0, h0, forward=False))
    if isinstance(engine0, (type(None), PlaceHolderLayoutEngine)):
        fig.set_layout_engine("tight")
        undo.append(lambda: fig.set_layout_engine(engine0))
    _legend_fits(fig, preset, undo)
    if not height:
        # Raised, a step at a time, until the lowest panel has room for its data.
        cap = preset.max_height / MM if preset.max_height else 4 * w
        for _ in range(6):
            renderer = _renderer(fig)
            low = min(
                (a.get_window_extent(renderer).height / fig.dpi * MM for a in panel_axes(fig)),
                default=MIN_PANEL,
            )
            if low >= MIN_PANEL - 0.5 or h >= cap:
                break
            h = min(cap, h + (MIN_PANEL - low) * panel_grid(fig)[0] / MM)
            fig.set_size_inches(w, h, forward=False)
        # Lowered to what is drawn: panels held to a shape of their own (a rose, a map)
        # fill the width and leave the rest of the height blank, above and below.
        # Lowered no further than leaves what is drawn as it was: a panel that would
        # shrink with the figure has no blank to give.
        drawn = fig.get_tightbbox(_renderer(fig)).height  # inches
        for _ in range(3):
            if h - drawn <= TRIM_MM / MM:
                break
            was = h
            h = max(drawn + PAD_MM / MM, MIN_PANEL / MM)
            fig.set_size_inches(w, h, forward=False)
            now = fig.get_tightbbox(_renderer(fig)).height
            if now < drawn - 0.5 / MM:
                h = was
                fig.set_size_inches(w, h, forward=False)
                break
            drawn = now
    if labels:
        made = label_panels(
            fig,
            labels,
            bold=preset.label_bold if label_bold is None else label_bold,
            size=preset.label_size,
            start=start,
            face=face,
        )
        undo.append(lambda: [t.remove() for t in made])

    def put_back() -> None:
        for step in reversed(undo):
            step()

    return put_back


# -- what the guide asks, read from the figure as drawn ----------------------------------


@dataclass(frozen=True)
class Finding:
    """One thing a publisher's guide asks of a figure, as the figure has it."""

    ok: bool
    about: str  # "width", "height", "text", "script", "largest text", "face", "panels"
    said: str


def drawn_texts(fig: Any) -> list[Any]:
    """The texts a draw of ``fig`` puts on the page: those shown, in the axes' view, not
    empty -- a tick outside the view is not drawn, nor is its label."""
    from matplotlib.text import Text

    drawn: list[Any] = []
    draw = Text.draw

    def record(self: Any, renderer: Any) -> Any:
        if self.get_visible() and self.get_text().strip():
            drawn.append(self)
        return draw(self, renderer)

    Text.draw = record  # type: ignore[method-assign]
    try:
        with _quiet_layout():
            fig.draw_without_rendering()
    finally:
        Text.draw = draw  # type: ignore[method-assign]
    return drawn


def _tick_clashes(fig: Any, drawn: list[Any]) -> list[str]:
    """The axes whose drawn tick labels run into one another, as "the x axis of panel b"."""
    renderer = _renderer(fig)
    shown = {id(t) for t in drawn}
    panels = panel_axes(fig)
    out = []
    for ax, name in zip(panels, panel_letters(len(panels)), strict=True):
        for axis, which in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            boxes = sorted(
                (
                    t.get_window_extent(renderer)
                    for t in axis.get_ticklabels(which="both")
                    if id(t) in shown
                ),
                key=(lambda b: b.x0) if which == "x" else (lambda b: b.y0),
            )
            if any(a.overlaps(b.shrunk(0.98, 0.98)) for a, b in itertools.pairwise(boxes)):
                out.append(f"the {which} axis of panel {name}")
    return out


def _face(t: Any) -> str:
    from matplotlib import font_manager as fm

    return str(fm.get_font(fm.findfont(t.get_fontproperties())).family_name)


def preflight(fig: Any, preset: Preset) -> list[Finding]:
    """What ``preset``'s guide asks of ``fig``, read from the figure as it is drawn: its
    width and height, its smallest text and its smallest sub- or superscript (drawn at
    SCRIPT_SHRINK of its text), its largest text but the panel labels, the faces its
    text is drawn in, its panels."""
    out: list[Finding] = []
    w, h = (float(v) * MM for v in fig.get_size_inches())
    widths = ", ".join(f"{k} {v:.0f} mm" for k, v in preset.widths.items())
    if preset.max_width is not None and w > preset.max_width + 0.5:
        out.append(
            Finding(
                False,
                "width",
                f"{w:.0f} mm wide: wider than {preset.name}'s "
                f"{preset.max_width:.0f} mm ({widths}).",
            )
        )
    elif preset.min_width is not None and w < preset.min_width - 0.5:
        out.append(
            Finding(
                False,
                "width",
                f"{w:.0f} mm wide: narrower than {preset.name}'s least, {preset.min_width:.0f} mm.",
            )
        )
    else:
        out.append(Finding(True, "width", f"{w:.0f} mm wide ({widths})."))
    if preset.max_height is not None:
        out.append(
            Finding(
                h <= preset.max_height + 0.5,
                "height",
                f"{h:.0f} mm high; at most {preset.max_height:.0f} mm.",
            )
        )
    else:
        out.append(Finding(True, "height", f"{h:.0f} mm high."))

    drawn = drawn_texts(fig)
    own = [t for t in drawn if t.get_gid() != PANEL_LABEL]
    if own:
        small = min(own, key=lambda t: t.get_fontsize())
        size = small.get_fontsize()
        least = preset.min_text
        out.append(
            Finding(
                least is None or size >= least - 0.05,
                "text",
                f"Smallest text {size:.1f} pt ({plain(small.get_text())[:30]!r})"
                + (f"; at least {least:g} pt." if least is not None else "."),
            )
        )
        scripts = [t for t in own if has_script(t.get_text())]
        if scripts:
            s = min(scripts, key=lambda t: t.get_fontsize())
            size = s.get_fontsize() * SCRIPT_SHRINK
            least = preset.min_script if preset.min_script is not None else preset.min_text
            out.append(
                Finding(
                    least is None or size >= least - 0.05,
                    "script",
                    f"Smallest sub- or superscript {size:.1f} pt "
                    f"({plain(s.get_text())[:30]!r})"
                    + (f"; at least {least:g} pt." if least is not None else "."),
                )
            )
        if preset.max_text is not None:
            big = max(own, key=lambda t: t.get_fontsize())
            size = big.get_fontsize()
            out.append(
                Finding(
                    size <= preset.max_text + 0.05,
                    "largest text",
                    f"Largest text {size:.1f} pt ({plain(big.get_text())[:30]!r}); at "
                    f"most {preset.max_text:g} pt, the panel labels aside.",
                )
            )
        faces = sorted({_face(t) for t in drawn})
        wanted = [f for f in preset.fonts if f in ("Arial", "Helvetica")]
        out.append(
            Finding(
                any(f in faces for f in wanted) or not wanted,
                "face",
                "Drawn in "
                + ", ".join(faces)
                + "."
                + (
                    ""
                    if any(f in faces for f in wanted)
                    else f" {preset.name} asks for {' or '.join(wanted)}."
                ),
            )
        )
    clashes = _tick_clashes(fig, drawn)
    out.append(
        Finding(
            not clashes,
            "tick labels",
            "Tick labels overlap on " + "; ".join(clashes) + ": a wider figure, or fewer ticks."
            if clashes
            else "No tick labels overlap.",
        )
    )
    n = len(panel_axes(fig))
    if preset.max_panels is not None:
        out.append(
            Finding(
                n <= preset.max_panels,
                "panels",
                f"{n} panel(s); {preset.name} asks for {preset.max_panels} or fewer.",
            )
        )
    return out


def save_for_print(
    fig: Any,
    path: Any,
    preset: Preset | None = None,
    *,
    fmt: str | None = None,
    dpi: int | None = None,
) -> Any:
    """Write ``fig`` at its size exactly -- not cropped to its ink -- its fonts embedded
    as TrueType (Type 42) in PDF and EPS and its text kept as text in SVG, editable; a
    TIFF or PNG at ``dpi`` (the preset's, else 300). ``path`` a file's path, or a
    file-like object with ``fmt`` given."""
    import os

    import matplotlib as mpl

    if fmt is None:
        fmt = os.path.splitext(str(path))[1].lstrip(".") or "pdf"
    fmt = fmt.lower()
    kw: dict[str, Any] = (
        {"pil_kwargs": {"compression": "tiff_lzw"}} if fmt in ("tif", "tiff") else {}
    )
    with (
        mpl.rc_context(
            {
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "svg.fonttype": "none",
                "savefig.bbox": "standard",
            }
        ),
        _quiet_layout(),
    ):
        fig.savefig(
            path,
            format=fmt,
            dpi=dpi or (preset.dpi if preset else 300),
            pad_inches=0,
            facecolor=fig.get_facecolor(),
            **kw,
        )
    return path
