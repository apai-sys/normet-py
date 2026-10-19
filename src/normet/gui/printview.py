"""A figure for print: a tab's figure laid out at the width a journal prints it,
in the lettering asked for, shown at that size and saved at the resolution
chosen. Adapted from the cume app's print view.

A figure on screen is drawn at the tab's size; printed in a 89 mm column its
lettering shrinks with it. Here the figure is set to the printed size and its
text to a size in points, so what is saved is what the page shows. While the
dialog is open the figure itself is on the dialog's canvas, at its printed
size, and can be finished there:

- the legend set in a corner or dragged where it reads best, its text sized,
  its entries put in columns, its frame shown or not;
- the data lines' width and marker size set (a line drawn without markers takes
  round ones when the size is raised);
- the panels lettered (a, b, c …);
- any text -- a title, an axis label, a note -- edited where it stands by
  double-clicking it, and a tick's numbers sized and turned the same way.

Nothing is kept: every change is recorded as it is made and undone when the
dialog closes, and the figure goes back to its tab as it was.

The widths offered are the column widths the publishers' artwork guides give;
check the journal's own current guide before submitting.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Callable, Iterator
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .theme import apply_figure_theme, light_figure

MM_PER_INCH = 25.4

#: Column widths in mm, as the publishers' artwork guides give them.
PRESETS: dict[str, float | None] = {
    "Nature: single column (89 mm)": 89.0,
    "Nature: double column (183 mm)": 183.0,
    "Elsevier: single column (90 mm)": 90.0,
    "Elsevier: 1.5 column (140 mm)": 140.0,
    "Elsevier: double column (190 mm)": 190.0,
    "ACS: single column (3.25 in)": 3.25 * MM_PER_INCH,
    "ACS: double column (7 in)": 7.0 * MM_PER_INCH,
    "Custom": None,
}

#: The resolutions offered, in dots per inch.
DPIS = (300, 600, 1200)

#: What a figure can be saved as, by extension.
FORMATS = "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tif *.tiff);;EPS (*.eps)"

#: Where the legend can be put, besides dragging it; None leaves it as drawn.
PLACES: tuple[tuple[str, str | None], ...] = (
    ("As drawn", None),
    ("Best", "best"),
    ("Upper right", "upper right"),
    ("Upper left", "upper left"),
    ("Lower right", "lower right"),
    ("Lower left", "lower left"),
    ("Outside, right", "outside right"),
)

#: Panel letters: what the menu shows, and the first letter as written.
LETTERS: tuple[tuple[str, str | None], ...] = (
    ("None", None),
    ("a  b  c", "a"),
    ("(a)  (b)  (c)", "(a)"),
    ("A  B  C", "A"),
    ("(A)  (B)  (C)", "(A)"),
)

#: Logical pixels to the inch the dialog shows the figure at: its printed size.
SCREEN_DPI = 96.0


# -- the layout, as a function ------------------------------------------------------------
@contextlib.contextmanager
def print_layout(fig: Any, width_mm: float, height_mm: float, font_pt: float) -> Iterator[None]:
    """For the time being: *fig* at the printed size, its text at *font_pt*
    (titles a point larger), laid out again to fit, in its light colours.

    Everything changed is put back on leaving: the size, every text's size,
    the layout engine and the colours.
    """
    with light_figure(fig), _Undo() as undo:
        _lay_out(fig, width_mm, height_mm, font_pt, undo)
        yield


def render(fig: Any, width_mm: float, height_mm: float, font_pt: float, dpi: float) -> bytes:
    """*fig* as a PNG at the printed size, as :func:`save` writes it."""
    buf = io.BytesIO()
    save(fig, buf, width_mm, height_mm, font_pt, dpi, fmt="png")
    return buf.getvalue()


def save(
    fig: Any,
    target: Any,
    width_mm: float,
    height_mm: float,
    font_pt: float,
    dpi: float,
    fmt: str | None = None,
) -> None:
    """Write *fig* at the printed size: the whole page, not cropped to its ink, so
    the file is exactly *width_mm* wide."""
    with print_layout(fig, width_mm, height_mm, font_pt):
        _write(fig, target, dpi, fmt)


def _write(fig: Any, target: Any, dpi: float, fmt: str | None = None) -> None:
    import matplotlib as mpl

    with mpl.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(target, dpi=dpi, format=fmt)


# -- what a layout changes, recorded to be undone -------------------------------------------
class _Undo:
    """Each change made as ``undo.set(getter, setter, value)``, put back in reverse
    order by :meth:`revert` (and on leaving a ``with``)."""

    def __init__(self) -> None:
        self._steps: list[Callable[[], Any]] = []

    def __enter__(self) -> _Undo:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.revert()

    def set(self, get: Callable[[], Any], put: Callable[[Any], Any], value: Any) -> None:
        was = get()
        put(value)
        self._steps.append(lambda: put(was))

    def then(self, step: Callable[[], Any]) -> None:
        """A step of its own to run when reverting (a legend made, taken away)."""
        self._steps.append(step)

    def revert(self) -> None:
        while self._steps:
            with contextlib.suppress(Exception):  # an artist already gone
                self._steps.pop()()


def _titles(fig: Any) -> set[int]:
    out = {id(ax.title) for ax in fig.get_axes()}
    if getattr(fig, "_suptitle", None) is not None:
        out.add(id(fig._suptitle))
    return out


def _lay_out(fig: Any, width_mm: float, height_mm: float, font_pt: float, undo: _Undo) -> None:
    """The printed size and lettering, and a layout that fits them."""
    from matplotlib.text import Text

    undo.set(
        lambda: tuple(fig.get_size_inches()),
        lambda s: fig.set_size_inches(*s, forward=False),
        (width_mm / MM_PER_INCH, height_mm / MM_PER_INCH),
    )
    titles = _titles(fig)
    for t in fig.findobj(Text):
        if t.get_text():
            undo.set(t.get_fontsize, t.set_fontsize, font_pt + 1 if id(t) in titles else font_pt)
    with contextlib.suppress(RuntimeError):  # a colour bar's engine stays as it is
        undo.set(fig.get_layout_engine, fig.set_layout_engine, "constrained")


def panel_axes(fig: Any) -> list[Any]:
    """The figure's panels, in order: its axes less colour bars and twins (an axes
    standing exactly where an earlier one stands)."""
    out: list[Any] = []
    seen: list[tuple[float, ...]] = []
    for ax in fig.get_axes():
        if ax.get_label() == "<colorbar>" or hasattr(ax, "_colorbar"):
            continue
        box = tuple(round(v, 4) for v in ax.get_position().bounds)
        if box in seen:
            continue
        seen.append(box)
        out.append(ax)
    return out


def letter(first: str, i: int) -> str:
    """The *i*-th panel letter in the style of *first*: ``a``, ``(a)``, ``A``, ``(A)``."""
    ch = chr(ord("A" if first.strip("()").isupper() else "a") + i)
    return f"({ch})" if first.startswith("(") else ch


def is_data_line(line: Any) -> bool:
    """A line drawn from data, as against a reference line (two points, a 1:1 or a
    zero line) or a grid line: more than two points, or a marker of its own."""
    marker = line.get_marker()
    return len(line.get_xdata()) > 2 or marker not in (None, "None", "", " ")


# -- the dialog -----------------------------------------------------------------------------
class _TextChoices(QDialog):
    """A text's own choices, asked at the text: its letters and its size."""

    def __init__(self, text: str, size: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Text")
        self.text = QLineEdit(text)
        self.size = QDoubleSpinBox()
        self.size.setRange(4, 24)
        self.size.setSingleStep(0.5)
        self.size.setSuffix(" pt")
        self.size.setValue(size)
        form = QFormLayout(self)
        form.addRow("Text", self.text)
        form.addRow("Size", self.size)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)


class _TickChoices(QDialog):
    """An axis's numbers, asked at a tick: their size and how they stand."""

    def __init__(self, size: float, rotation: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tick labels")
        self.size = QDoubleSpinBox()
        self.size.setRange(4, 24)
        self.size.setSingleStep(0.5)
        self.size.setSuffix(" pt")
        self.size.setValue(size)
        self.rotation = QSpinBox()
        self.rotation.setRange(-90, 90)
        self.rotation.setSingleStep(15)
        self.rotation.setSuffix("°")
        self.rotation.setValue(int(rotation if rotation <= 90 else rotation - 360))
        form = QFormLayout(self)
        form.addRow("Size", self.size)
        form.addRow("Rotation", self.rotation)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)


class PrintDialog(QDialog):
    """The figure on *canvas* -- a tab's -- laid out for print while the dialog is open."""

    def __init__(self, canvas: Any, parent: QWidget | None = None) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        super().__init__(parent)
        self.setWindowTitle("Export Figure for Publication")
        self.tab_canvas, self.fig = canvas, canvas.figure
        fig = self.fig
        w_in, h_in = fig.get_size_inches()
        # Its own proportions, within reason: a strip drawn to fill a wide tab
        # would print a column width wide and a couple of lines high.
        self._aspect = min(1.2, max(0.35, h_in / w_in if w_in else 0.6))
        # What the dialog itself changes for its whole life, put back by done().
        self._dark = bool(getattr(fig, "_normet_dark", False))
        self._kept = (tuple(fig.get_size_inches()), fig.dpi, getattr(fig, "_original_dpi", fig.dpi))
        apply_figure_theme(fig, False)
        self._undo = _Undo()  # what the last layout pass changed
        #: Edits made at the figure: text -> (letters, size); axis -> (size, rotation);
        #: an axes -> where its legend was dragged to.
        self.text_edits: dict[Any, tuple[str, float]] = {}
        self.tick_edits: dict[Any, tuple[float, float]] = {}
        self.legend_at: dict[Any, Any] = {}
        self._dragging: Any = None

        # the controls
        self.preset = QComboBox()
        self.preset.addItems(list(PRESETS))
        self.width = QDoubleSpinBox()
        self.width.setRange(20, 500)
        self.width.setSuffix(" mm")
        self.height = QDoubleSpinBox()
        self.height.setRange(20, 500)
        self.height.setSuffix(" mm")
        self.font = QDoubleSpinBox()
        self.font.setRange(4, 14)
        self.font.setSingleStep(0.5)
        self.font.setValue(7.0)
        self.font.setSuffix(" pt")
        self.font.setToolTip("Tick labels, axis labels and legends; titles are a point larger.")
        self.dpi = QComboBox()
        self.dpi.addItems([str(d) for d in DPIS])
        self.dpi.setToolTip(
            "Dots per inch of a raster file (PNG, TIFF); PDF, SVG and EPS are vector."
        )
        page = QFormLayout()
        page.addRow("Journal", self.preset)
        page.addRow("Width", self.width)
        page.addRow("Height", self.height)
        page.addRow("Lettering", self.font)
        page.addRow("Resolution (dpi)", self.dpi)

        self.legend_place = QComboBox()
        self.legend_place.addItems([name for name, _ in PLACES])
        self.legend_size = QDoubleSpinBox()
        self.legend_size.setRange(0, 14)
        self.legend_size.setSingleStep(0.5)
        self.legend_size.setSpecialValueText("as lettering")
        self.legend_size.setSuffix(" pt")
        self.legend_cols = QSpinBox()
        self.legend_cols.setRange(1, 8)
        self.legend_frame = QCheckBox("Frame")
        legend_box = QGroupBox("Legend")
        lf = QFormLayout(legend_box)
        lf.addRow("Where", self.legend_place)
        lf.addRow("Text", self.legend_size)
        lf.addRow("Columns", self.legend_cols)
        lf.addRow("", self.legend_frame)
        legends = self._legends()
        legend_box.setEnabled(bool(legends))
        if legends:
            self.legend_cols.setValue(max(1, getattr(legends[0][1], "_ncols", 1)))
            self.legend_frame.setChecked(legends[0][1].get_frame_on())

        self.line_width = QDoubleSpinBox()
        self.line_width.setRange(0, 5)
        self.line_width.setSingleStep(0.25)
        self.line_width.setSpecialValueText("as drawn")
        self.line_width.setSuffix(" pt")
        self.marker_size = QDoubleSpinBox()
        self.marker_size.setRange(0, 12)
        self.marker_size.setSingleStep(0.5)
        self.marker_size.setSpecialValueText("as drawn")
        self.marker_size.setSuffix(" pt")
        self.letters = QComboBox()
        self.letters.addItems([name for name, _ in LETTERS])
        marks = QGroupBox("Lines and panels")
        mf = QFormLayout(marks)
        mf.addRow("Line width", self.line_width)
        mf.addRow("Marker size", self.marker_size)
        mf.addRow("Panel letters", self.letters)
        self.letters.setEnabled(len(panel_axes(fig)) > 1)

        note = QLabel(
            "Drag the legend where it reads best; double-click a title, an axis label or a "
            "tick's numbers to edit it. Widths are those the publishers' artwork guides give: "
            "check the journal's own guide before submitting. Saved on a white page, at "
            "exactly this width; the tab's figure is left as it was."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: palette(placeholder-text); font-size: 11px;")
        save_btn = QPushButton("Save…")
        save_btn.setStyleSheet("font-weight: 600;")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self._save)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)

        left = QVBoxLayout()
        left.addLayout(page)
        left.addWidget(legend_box)
        left.addWidget(marks)
        left.addWidget(note)
        left.addStretch(1)
        row = QHBoxLayout()
        row.addWidget(close_btn)
        row.addWidget(save_btn)
        left.addLayout(row)

        # the figure itself, on the dialog's canvas for as long as it is open
        fig.set_dpi(SCREEN_DPI)
        fig._original_dpi = SCREEN_DPI
        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.preview = QWidget()  # the canvas centred on a plain ground
        pv = QVBoxLayout(self.preview)
        pv.addWidget(self.canvas, 0, Qt.AlignmentFlag.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidget(self.preview)
        scroll.setWidgetResizable(True)
        scroll.setMinimumSize(560, 440)
        self.size_label = QLabel()
        self.size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right = QVBoxLayout()
        right.addWidget(scroll, 1)
        right.addWidget(self.size_label)

        lay = QHBoxLayout(self)
        lay.addLayout(left)
        lay.addLayout(right, 1)

        self.canvas.mpl_connect("button_press_event", self._pressed)
        self.canvas.mpl_connect("motion_notify_event", self._moved)
        self.canvas.mpl_connect("button_release_event", self._released)
        self.preset.currentTextChanged.connect(self._preset_changed)
        self.width.valueChanged.connect(self._keep_aspect)
        for w in (self.height, self.font, self.legend_size, self.line_width, self.marker_size):
            w.valueChanged.connect(self.apply)
        self.legend_cols.valueChanged.connect(self.apply)
        self.legend_frame.toggled.connect(self.apply)
        self.legend_place.currentTextChanged.connect(self._place_changed)
        self.letters.currentTextChanged.connect(self.apply)
        self._preset_changed(self.preset.currentText())

    # -- settings ------------------------------------------------------------------------
    def settings(self) -> tuple[float, float, float, int]:
        return (
            self.width.value(),
            self.height.value(),
            self.font.value(),
            int(self.dpi.currentText()),
        )

    def _preset_changed(self, name: str) -> None:
        width = PRESETS.get(name)
        self.width.setEnabled(width is None)
        if width is not None:
            self.width.setValue(round(width, 1))
        self._keep_aspect(self.width.value())

    def _keep_aspect(self, width: float) -> None:
        self.height.blockSignals(True)
        self.height.setValue(round(width * self._aspect, 1))
        self.height.blockSignals(False)
        self.apply()

    def _place_changed(self, _name: str) -> None:
        self.legend_at.clear()  # a corner chosen: the dragged places are let go
        self.apply()

    # -- the layout pass -----------------------------------------------------------------
    def _legends(self) -> list[tuple[Any, Any]]:
        """Every axes' legend, with its axes."""
        return [(ax, ax.get_legend()) for ax in self.fig.get_axes() if ax.get_legend() is not None]

    def apply(self) -> None:
        """The figure laid out again from the controls and the edits made at it: what
        the last pass changed undone first, so each pass starts from the figure as
        drawn."""
        self._undo.revert()
        undo, fig = self._undo, self.fig
        w, h, pt, _ = self.settings()
        _lay_out(fig, w, h, pt, undo)
        self._style_lines(undo)
        self._style_legends(undo, pt)
        self._letter_panels(undo, pt)
        for t, (text, size) in self.text_edits.items():
            undo.set(t.get_text, t.set_text, text)
            undo.set(t.get_fontsize, t.set_fontsize, size)
        for axis, (size, rotation) in self.tick_edits.items():
            # Label by label, as the lettering is set: the axis's own tick settings
            # are left alone, so nothing of them outlives the dialog.
            for lbl in axis.get_ticklabels(which="both"):
                undo.set(lbl.get_fontsize, lbl.set_fontsize, size)
                undo.set(lbl.get_rotation, lbl.set_rotation, rotation)
        self._show()

    def _style_lines(self, undo: _Undo) -> None:
        lw, ms = self.line_width.value(), self.marker_size.value()
        if not lw and not ms:
            return
        for ax in self.fig.get_axes():
            for line in ax.get_lines():
                if not is_data_line(line):
                    continue
                if lw:
                    undo.set(line.get_linewidth, line.set_linewidth, lw)
                if ms:
                    if line.get_marker() in (None, "None", "", " "):
                        undo.set(line.get_marker, line.set_marker, "o")
                    undo.set(line.get_markersize, line.set_markersize, ms)

    def _style_legends(self, undo: _Undo, pt: float) -> None:
        """Each legend made again from its axes' handles -- so its keys take the lines'
        new widths and markers -- with the place, text size, columns and frame asked."""
        place = dict(PLACES)[self.legend_place.currentText()]
        size = self.legend_size.value() or pt
        for ax, old in self._legends():
            handles, labels = ax.get_legend_handles_labels()
            if not handles:
                handles, labels = old.legend_handles, [t.get_text() for t in old.get_texts()]
            loc: Any = self.legend_at.get(ax) or place or old._loc
            kw: dict[str, Any] = {}
            if loc == "outside right":
                loc, kw["bbox_to_anchor"] = "center left", (1.02, 0.5)
            title = old.get_title().get_text()
            new = ax.legend(
                handles,
                labels,
                loc=loc,
                ncols=self.legend_cols.value(),
                frameon=self.legend_frame.isChecked(),
                fontsize=size,
                title=title or None,
                title_fontsize=size,
                **kw,
            )
            new.set_draggable(True, update="loc")
            undo.then(lambda a=ax, o=old, n=new: (n.remove(), setattr(a, "legend_", o)))

    def _letter_panels(self, undo: _Undo, pt: float) -> None:
        first = dict(LETTERS)[self.letters.currentText()]
        if first is None:
            return
        for i, ax in enumerate(panel_axes(self.fig)):
            t = ax.text(
                -0.02,
                1.02,
                letter(first, i),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=pt + 1,
                fontweight="bold",
            )
            undo.then(t.remove)

    def _show(self) -> None:
        """The canvas at the figure's printed size, on this screen's logical inch."""
        w_in, h_in = self.fig.get_size_inches()
        self.canvas.setFixedSize(round(w_in * SCREEN_DPI), round(h_in * SCREEN_DPI))
        # The canvas, resized, sets the figure's size from its own pixels: put back
        # exactly, so the layout and the file are at the size asked.
        self.fig.set_size_inches(w_in, h_in, forward=False)
        self.canvas.draw_idle()
        w, h, _, _ = self.settings()
        self.size_label.setText(f"{w:.0f} × {h:.0f} mm, shown at its printed size")

    # -- edits at the figure -------------------------------------------------------------
    def _legend_under(self, event: Any) -> tuple[Any, Any] | None:
        for ax, leg in self._legends():
            if leg.contains(event)[0]:
                return ax, leg
        return None

    def _text_under(self, event: Any) -> Any:
        from matplotlib.text import Text

        for t in self.fig.findobj(Text):
            if t.get_text() and t.get_visible() and t.contains(event)[0]:
                return t
        return None

    def _axis_under(self, event: Any) -> Any:
        for ax in self.fig.get_axes():
            for axis in (ax.xaxis, ax.yaxis):
                if any(
                    lbl.contains(event)[0] for lbl in axis.get_ticklabels() if lbl.get_visible()
                ):
                    return axis
        return None

    def _pressed(self, event: Any) -> None:
        if event.dblclick:
            self.edit_at(event)
            return
        found = self._legend_under(event)
        self._dragging = found[0] if found else None

    def _moved(self, event: Any) -> None:
        pass  # matplotlib's own DraggableLegend moves it

    def _released(self, event: Any) -> None:
        """A legend dropped: where it now stands kept for the next layout pass."""
        ax, self._dragging = self._dragging, None
        if ax is not None and ax.get_legend() is not None:
            self.legend_at[ax] = ax.get_legend()._loc

    def edit_at(self, event: Any) -> None:
        """A double-click: the tick numbers or the text under it edited where it stands."""
        axis = self._axis_under(event)
        if axis is not None:
            labels = [lbl for lbl in axis.get_ticklabels() if lbl.get_text()]
            size = labels[0].get_fontsize() if labels else self.font.value()
            rotation = labels[0].get_rotation() if labels else 0.0
            ask = _TickChoices(size, rotation, self)
            if ask.exec():
                self.set_ticks(axis, ask.size.value(), ask.rotation.value())
            return
        t = self._text_under(event)
        if t is None or self._legend_under(event) is not None:
            return
        ask = _TextChoices(t.get_text(), t.get_fontsize(), self)
        if ask.exec():
            self.set_text(t, ask.text.text(), ask.size.value())

    def set_text(self, t: Any, text: str, size: float) -> None:
        """A text's letters and size, kept while the dialog is open."""
        self.text_edits[t] = (text, size)
        self.apply()

    def set_ticks(self, axis: Any, size: float, rotation: float) -> None:
        """An axis's tick numbers' size and rotation, kept while the dialog is open."""
        self.tick_edits[axis] = (size, rotation)
        self.apply()

    # -- saving and closing --------------------------------------------------------------
    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save figure", "figure.png", FORMATS)
        if path:
            self.save_to(path)
            self.accept()

    def save_to(self, target: Any, fmt: str | None = None) -> None:
        """The figure as shown, at exactly its printed size and the resolution chosen."""
        w, h, _, dpi = self.settings()
        self.fig.set_size_inches(w / MM_PER_INCH, h / MM_PER_INCH, forward=False)
        _write(self.fig, target, dpi, fmt)

    def done(self, result: int) -> None:
        """Closed: every change undone, and the figure back on its tab as it was."""
        self._undo.revert()
        size, dpi, original = self._kept
        fig = self.fig
        fig.set_dpi(dpi)
        fig._original_dpi = original
        fig.set_size_inches(*size, forward=False)
        fig.set_canvas(self.tab_canvas)
        apply_figure_theme(fig, self._dark)
        self.tab_canvas.draw_idle()
        super().done(result)
