"""A figure for print: a tab's figure laid out at the size a journal prints it, in the
lettering its guide asks for (:mod:`normet.figstyle`), shown at that size -- fitted to
the window as it opens, or its actual size on the screen -- beside what the guide asks of
it, read from the figure as drawn; saved as it is shown, a raster file at the resolution
chosen. Ported from cume (``cume_qt.printview``).

- the journal (Elsevier, Nature, Copernicus, ACS), its column widths or a width of one's
  own, and the height the figure needs or one set;
- the panels lettered -- a b c, (a) (b) (c), A B C -- bold or not, from any letter (a
  figure that is panels c and d of a composite starts at c);
- the chemical formulas written as chemistry writes them (SO₄²⁻, PM₂.₅), and the
  figure's and the panels' titles left to the caption, or drawn;
- the legend set in a corner or dragged where it reads best, its text sized, its
  entries in columns, its frame shown or not; a colour bar dragged, or put on another
  side of its panel;
- the data lines' width and marker size (a time series takes round markers when the
  size is raised);
- any text, a tick's numbers -- their size and how they stand, a tick's own text -- the
  legend or a colour bar edited where it stands, by double-clicking it.

The figure's texts are in the report's language (File → Report Language) while it is
here. Closed, the figure goes back to its tab as it was. The journal, width, panel
letters, formulas and resolution chosen are kept for the next figure.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from normet import figstyle as FS

from .theme import apply_figure_theme

#: Panel labels as the menu offers them; None for none.
LABELS: tuple[tuple[str, str | None], ...] = (
    ("None", None),
    ("a  b  c", "a"),
    ("(a)  (b)  (c)", "(a)"),
    ("A  B  C", "A"),
    ("(A)  (B)  (C)", "(A)"),
)
#: File types by extension, as the save dialog names them.
KINDS = {
    "pdf": "PDF (*.pdf)",
    "eps": "EPS (*.eps)",
    "svg": "SVG (*.svg)",
    "tiff": "TIFF (*.tiff *.tif)",
    "png": "PNG (*.png)",
}
#: Where the legend can be put, besides dragging it; None leaves it as drawn.
PLACES: tuple[tuple[str, str | None], ...] = (
    ("As drawn", None),
    ("Best", "best"),
    ("Upper right", "upper right"),
    ("Upper left", "upper left"),
    ("Lower right", "lower right"),
    ("Lower left", "lower left"),
)
#: Where the dialog's choices are kept for the next figure (QSettings, as JSON).
PREF = "figure_print"


def _prefs() -> Any:
    from PySide6.QtCore import QSettings

    return QSettings("apai-sys", "normet-gui")


def load_settings() -> dict[str, Any]:
    """The choices last made in the dialog; {} where none are kept."""
    try:
        got = json.loads(str(_prefs().value(PREF, "") or "{}"))
    except ValueError:
        return {}
    return got if isinstance(got, dict) else {}


def keep_settings(settings: dict[str, Any]) -> None:
    _prefs().setValue(PREF, json.dumps(settings))


def _call_later(obj: Any, f: Any, ms: int = 0) -> None:
    """``f()`` from the event loop, unless ``obj`` has been deleted by then."""
    from shiboken6 import isValid

    QTimer.singleShot(ms, lambda: f() if isValid(obj) else None)


def _choices_dialog(dlg: QDialog, form: QFormLayout) -> None:
    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    top = QVBoxLayout(dlg)
    top.addLayout(form)
    top.addWidget(buttons)


def _size_box(size: float) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(4.0, 14.0)
    box.setDecimals(1)
    box.setSingleStep(0.5)
    box.setSuffix(" pt")
    box.setValue(size)
    return box


class _LegendChoices(QDialog):
    """The legend's two own choices -- where it sits, its entries' columns -- asked at
    the legend itself, the side panel's own offers made again."""

    def __init__(self, parent: QWidget, pos: str | None, ncols: int) -> None:
        super().__init__(parent)
        self.setWindowTitle("Legend")
        self.place = QComboBox()
        for text, where in PLACES:
            self.place.addItem(text, where)
        self.place.setCurrentIndex(
            next((i for i in range(self.place.count()) if self.place.itemData(i) == pos), 0)
        )
        self.columns = QSpinBox()
        self.columns.setRange(1, 6)
        self.columns.setValue(ncols)
        form = QFormLayout()
        form.addRow("Where", self.place)
        form.addRow("Columns", self.columns)
        _choices_dialog(self, form)

    def chosen(self) -> tuple[str | None, int]:
        return self.place.currentData(), self.columns.value()


class _LabelChoices(QDialog):
    """A text's own choices -- its letters, its size -- asked at the text itself: an
    axis's own label, a title's, a note's; an axis's offset text only the size."""

    def __init__(self, parent: QWidget, text: str, size: float, with_text: bool = True) -> None:
        super().__init__(parent)
        self.with_text = with_text
        self.setWindowTitle("Label" if with_text else "Tick labels")
        self.text = QLineEdit(text)
        self.size = _size_box(size)
        form = QFormLayout()
        if with_text:
            form.addRow("Text", self.text)
        form.addRow("Size", self.size)
        _choices_dialog(self, form)

    def chosen(self) -> tuple[str | None, float]:
        return (self.text.text() if self.with_text else None), self.size.value()


class _TickChoices(QDialog):
    """A tick's own choices -- the one's letters, their size, how they stand -- asked at
    the tick itself: its text is that tick's own, the size and the rotation the axis's
    every tick's (a column of numbers, one size, one angle)."""

    def __init__(self, parent: QWidget, text: str, size: float, rotation: float) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tick labels")
        self.text = QLineEdit(text)
        self.size = _size_box(size)
        self.rotation = QDoubleSpinBox()
        self.rotation.setRange(-180.0, 180.0)
        self.rotation.setDecimals(0)
        self.rotation.setSingleStep(15.0)
        self.rotation.setSuffix("°")
        self.rotation.setValue(rotation)
        form = QFormLayout()
        form.addRow("Text", self.text)
        form.addRow("Size", self.size)
        form.addRow("Rotation", self.rotation)
        _choices_dialog(self, form)

    def chosen(self) -> tuple[str, float, float]:
        return self.text.text(), self.size.value(), self.rotation.value()


def print_canvas(fig: Any) -> Any:
    """A canvas showing ``fig`` at the figure's own size: resized, it draws more or fewer
    dots to the inch, where matplotlib's would make the figure the widget's whole pixels
    -- 350 px at 100 to the inch, 88.9 mm of a column 89 mm wide."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    class PrintCanvas(FigureCanvasQTAgg):
        def resizeEvent(self, event: Any) -> None:  # noqa: N802
            w_in = self.figure.get_size_inches()[0]
            w = event.size().width() * self.device_pixel_ratio
            if w > 0 and w_in > 0:
                self.figure._set_dpi(w / w_in, forward=False)
            QWidget.resizeEvent(self, event)
            self.draw_idle()

    return PrintCanvas(fig)


def _quiet(widget: Any, setter: Any, value: Any) -> None:
    widget.blockSignals(True)
    try:
        setter(value)
    finally:
        widget.blockSignals(False)


def _renamed(formatter: Any, said: dict[float, str]) -> Any:
    """``formatter``, the tick values in ``said`` written by the texts asked for them, the
    rest as before: an axis's labels are written by its formatter at every draw, so that
    is where a tick's own text lives."""
    from matplotlib.ticker import Formatter

    class Renamed(Formatter):
        def __call__(self, x: Any, pos: Any = None) -> str:
            here = said.get(float(x))
            return here if here is not None else formatter(x, pos)

        def format_ticks(self, values: Any) -> list[str]:
            return [
                said.get(float(v), s)
                for v, s in zip(values, formatter.format_ticks(values), strict=False)
            ]

        def get_offset(self) -> str:
            return str(formatter.get_offset())

        def set_locs(self, locs: Any) -> None:
            formatter.set_locs(locs)

        def format_data_short(self, value: Any) -> str:
            return str(formatter.format_data_short(value))

    made = Renamed()
    made.said = said  # the texts, added to as more ticks are asked
    made.under = formatter  # what it wraps, restored on closing
    made._normet_renamed = True
    return made


class PrintDialog(QDialog):
    """The figure on ``canvas`` -- a tab's -- laid out for print while the dialog is open.

    ``settings`` are the choices last made (journal, width, panel labels, bold, formulas,
    the resolution), :func:`load_settings` where None; the legend's own choices start
    from the figure and its preset, and each change is kept for the next figure
    (``on_settings``, :func:`keep_settings` where None)."""

    def __init__(
        self,
        canvas: Any,
        parent: QWidget | None = None,
        settings: dict[str, Any] | None = None,
        on_settings: Any = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Figure for Print")
        self._home, self.fig = canvas, canvas.figure
        # A figure for a paper is in the report's language (File → Report Language), not
        # the window's: its texts are written so while it is here, and the tab's canvas
        # writes them back in the window's language when it next draws.
        from ._shell import report_language
        from .uilang import figure_texts

        figure_texts(self.fig, lang=report_language())
        self._on_settings = keep_settings if on_settings is None else on_settings
        self._put_back: Any = None
        self._label_edits: dict[Any, tuple[str, float]] = {}  # a text's, as last asked
        self._label_was: dict[Any, str] = {}  # the text it had before, put back
        self._letter_edits: dict[int, tuple[str, float]] = {}  # a panel's letter
        self._tick_edits: dict[Any, float] = {}  # a tick axis's size, laid over each pass
        self._tick_rots: dict[Any, float] = {}  # a tick axis's rotation, ditto
        self._tick_rot_was: dict[Any, float] = {}  # the rotation it had, put back
        self._tick_said: dict[Any, dict[float, str]] = {}  # single ticks' own texts
        self._bar_edits: dict[Any, dict[str, Any]] = {}  # where a bar stood, put back
        self._bar_drags: dict[Any, dict[str, Any]] = {}  # a dragged bar's own locator
        self._drag: Any = None  # the legend or bar under the press, if any
        self._dark = bool(getattr(self.fig, "_normet_dark", False))
        self._size0 = tuple(self.fig.get_size_inches())
        self._dpi0 = self.fig.dpi
        self._orig0 = getattr(self.fig, "_original_dpi", self.fig.dpi)
        apply_figure_theme(self.fig, False)  # a white page while it is here
        # Where and how each legend is, put back on closing.
        self._legends = [
            (leg, leg._loc, leg._bbox_to_anchor, getattr(leg, "_ncols", 1), leg.get_frame_on())
            for leg in self._all_legends()
        ]
        s = dict(load_settings() if settings is None else settings)
        # The lines' own strokes, kept to put back when the dialog closes. The data lines
        # are those with more than two points (a reference line has two) or a marker of
        # their own; a figure of nothing but two-point lines has those as its data.
        from matplotlib.lines import Line2D

        seen: set[int] = set()
        axes_lines = []
        for ax in self.fig.axes:
            for line in ax.get_lines():
                if id(line) not in seen:
                    seen.add(id(line))
                    axes_lines.append(line)
        data_lines = [line for line in axes_lines if self._is_data_line(line)]
        if not data_lines:
            data_lines = list(axes_lines)
        data_ids = {id(line) for line in data_lines}
        self._line_styles = [
            (
                line,
                line.get_linewidth(),
                line.get_markersize(),
                line.get_marker(),
                id(line) in data_ids,
            )
            for line in axes_lines
        ]
        for leg in self._all_legends():  # a legend's handles are its own copies
            for line in getattr(leg, "legend_handles", ()):
                if isinstance(line, Line2D) and id(line) not in seen:
                    seen.add(id(line))
                    self._line_styles.append(
                        (line, line.get_linewidth(), line.get_markersize(), line.get_marker(), True)
                    )
        self._data_lines = data_lines
        self._line_asked = False
        self._marker_asked = False
        widths = [line.get_linewidth() for line in data_lines if line.get_linewidth() > 0]
        marks = [
            line.get_markersize()
            for line in data_lines
            if str(line.get_marker()).lower() not in ("none", "", " ")
        ]
        line_w = Counter(round(float(v), 2) for v in widths).most_common(1)
        marker_s = Counter(round(float(v), 2) for v in marks).most_common(1)
        self.line_width = QDoubleSpinBox()
        self.line_width.setRange(0.0, 8.0)
        self.line_width.setDecimals(1)
        self.line_width.setSingleStep(0.1)
        self.line_width.setSuffix(" pt")
        self.line_width.setValue(line_w[0][0] if line_w else 1.5)
        self.line_width.setToolTip(
            "The data lines' width, in points. Reference lines and grid lines keep theirs."
        )
        self.line_width.setEnabled(bool(data_lines))
        self.marker_size = QDoubleSpinBox()
        self.marker_size.setRange(0.0, 16.0)
        self.marker_size.setDecimals(1)
        self.marker_size.setSingleStep(0.5)
        self.marker_size.setSuffix(" pt")
        self.marker_size.setValue(marker_s[0][0] if marker_s else 0.0)
        self.marker_size.setToolTip(
            "The data lines' marker size, in points. Above 0, a line drawn without "
            "markers -- a time series, say -- gets round ones."
        )
        self.marker_size.setEnabled(bool(data_lines))

        self.preset_cb = QComboBox()
        for key, preset in FS.PRESETS.items():
            self.preset_cb.addItem(preset.name, key)
        self.width_cb = QComboBox()
        self.width_mm = QDoubleSpinBox()
        self.width_mm.setRange(20, 400)
        self.width_mm.setDecimals(1)
        self.width_mm.setSuffix(" mm")
        self.height_auto = QCheckBox("As the figure needs")
        self.height_auto.setChecked(True)
        self.height_mm = QDoubleSpinBox()
        self.height_mm.setRange(20, 400)
        self.height_mm.setDecimals(1)
        self.height_mm.setSuffix(" mm")
        self.labels_cb = QComboBox()
        for text, style in LABELS:
            self.labels_cb.addItem(text, style)
        self.bold_chk = QCheckBox("Bold")
        # A figure that is panels c and d of a composite starts at c.
        self.start_cb = QComboBox()
        for letter in "abcdefghijkl":
            self.start_cb.addItem(f"from {letter}", letter)
        self.formula_chk = QCheckBox("Formulas as chemistry writes them: SO₄²⁻, PM₂.₅")
        self.title_chk = QCheckBox("Titles: the figure's and the panels'")
        self.title_chk.setToolTip(
            "Left off unless asked for: a journal's figure carries its titles in its "
            "caption -- the figure's own and each panel's; checked, all are drawn."
        )
        self.zoom_cb = QComboBox()
        self.zoom_cb.addItem("Fit the window", "fit")
        self.zoom_cb.addItem("Actual size", "actual")
        self.legend_cb = QComboBox()
        for text, pos in PLACES:
            self.legend_cb.addItem(text, pos)
        self.legend_cb.setToolTip(
            "Where the legend sits: as drawn or in a corner. A dragged one stays where it "
            "was dragged until set here again."
        )
        self.legend_size = _size_box(7.0)
        self.legend_size.setToolTip(
            "The legend's own text, within the guide's sizes; the checks below say where it stands."
        )
        self.legend_ncol = QSpinBox()
        self.legend_ncol.setRange(1, 6)
        self.legend_ncol.setToolTip("The legend's entries in this many columns.")
        self.legend_frame = QCheckBox("Frame")
        self.legend_frame.setToolTip("A box and background behind the legend.")
        self.dpi_cb = QComboBox()
        for dpi in (300, 600, 1200):
            self.dpi_cb.addItem(f"{dpi} dpi", dpi)
        self.dpi_cb.setToolTip(
            "A TIFF's or PNG's resolution -- journals commonly ask 300 for colour, 600 for "
            "grey, 1200 for line art. PDF, EPS and SVG are vector files: resolution does "
            "not apply."
        )

        form = QFormLayout()
        form.addRow("Journal", self.preset_cb)
        row = QHBoxLayout()
        row.addWidget(self.width_cb, 1)
        row.addWidget(self.width_mm)
        form.addRow("Width", row)
        row = QHBoxLayout()
        row.addWidget(self.height_auto, 1)
        row.addWidget(self.height_mm)
        form.addRow("Height", row)
        row = QHBoxLayout()
        row.addWidget(self.labels_cb, 1)
        row.addWidget(self.start_cb)
        row.addWidget(self.bold_chk)
        form.addRow("Panel labels", row)
        if data_lines:
            form.addRow("Line width", self.line_width)
            form.addRow("Marker size", self.marker_size)
        row = QHBoxLayout()
        row.addWidget(self.legend_cb, 1)
        row.addWidget(self.legend_size)
        row.addWidget(self.legend_ncol)
        row.addWidget(self.legend_frame)
        row.setContentsMargins(0, 0, 0, 0)
        legend_row = QWidget()
        legend_row.setLayout(row)
        form.addRow("Legend", legend_row)
        self.legend_row = legend_row
        form.addRow("", self.formula_chk)
        form.addRow("", self.title_chk)
        form.addRow("Raster resolution", self.dpi_cb)
        form.addRow("View", self.zoom_cb)
        self.source = QLabel()
        self.source.setOpenExternalLinks(True)
        self.source.setWordWrap(True)
        self.source.setTextFormat(Qt.TextFormat.RichText)

        self.checks = QListWidget()
        self.checks.setWordWrap(True)
        save = QPushButton("Save…")
        save.setDefault(True)
        save.clicked.connect(lambda: self._save())  # clicked hands a slot its checked
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        side = QVBoxLayout()
        side.addLayout(form)
        side.addWidget(self.source)
        side.addWidget(QLabel("What the journal's guide asks, read from the figure:"))
        side.addWidget(self.checks, 1)
        hint = QLabel(
            "Double-click a label or a tick's numbers to size them, the legend or a colour "
            "bar to edit it; drag a legend or a colour bar where it reads best."
        )
        hint.setStyleSheet("color: gray;")
        hint.setWordWrap(True)
        side.addWidget(hint)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(close)
        buttons.addWidget(save)
        side.addLayout(buttons)
        panel = QWidget()
        panel.setLayout(side)
        panel.setFixedWidth(420)

        self.fig.set_dpi(self._screen_dpi())
        self.canvas = print_canvas(self.fig)
        self.canvas.mpl_connect("button_press_event", self._pressed)
        self.canvas.mpl_connect("motion_notify_event", self._moved)
        self.canvas.mpl_connect("button_release_event", self._released)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.canvas)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pal = self.scroll.palette()
        pal.setColor(self.scroll.backgroundRole(), QColor("#8e8e93"))
        self.scroll.setPalette(pal)
        top = QHBoxLayout(self)
        top.addWidget(panel)
        top.addWidget(self.scroll, 1)

        key = s.get("preset") if s.get("preset") in FS.PRESETS else "elsevier"
        self.preset_cb.setCurrentIndex(list(FS.PRESETS).index(key))
        self._fill_widths(s.get("width"))
        preset = self.preset()
        self.labels_cb.setCurrentIndex(
            next((i for i, (_, st) in enumerate(LABELS) if st == s.get("labels", preset.label)), 0)
        )
        self.bold_chk.setChecked(bool(s.get("bold", preset.label_bold)))
        self.formula_chk.setChecked(bool(s.get("formulas", True)))
        # The legend's own choices start from the figure and the preset: a figure has a
        # legend or not, its own columns and frame, the preset's text size.
        first = self._legends[0][0] if self._legends else None
        self.legend_size.setValue(preset.text)
        self.legend_ncol.setValue(getattr(first, "_ncols", 1) if first is not None else 1)
        self.legend_frame.setChecked(first.get_frame_on() if first is not None else False)
        # The whole row goes where there is no legend, its "Legend" label with it.
        form.setRowVisible(legend_row, first is not None)
        self.dpi_cb.setCurrentIndex(
            next(
                (i for i in range(self.dpi_cb.count()) if self.dpi_cb.itemData(i) == s.get("dpi")),
                0,
            )
        )
        self.preset_cb.currentIndexChanged.connect(self._preset_changed)
        self.width_cb.currentIndexChanged.connect(self._width_chosen)
        for w in (self.width_mm, self.height_mm):
            w.valueChanged.connect(self._apply)
        for c in (self.height_auto, self.bold_chk, self.formula_chk, self.title_chk):
            c.toggled.connect(self._apply)
        for b in (self.labels_cb, self.start_cb, self.zoom_cb):
            b.currentIndexChanged.connect(self._apply)
        self.legend_cb.currentIndexChanged.connect(self._legend_placed)
        self.legend_size.valueChanged.connect(self._apply)
        self.legend_ncol.valueChanged.connect(self._apply)
        self.legend_frame.toggled.connect(self._apply)
        self.line_width.valueChanged.connect(self._line_changed)
        self.marker_size.valueChanged.connect(self._marker_changed)
        self.dpi_cb.currentIndexChanged.connect(self._settings_changed)
        screen = QGuiApplication.primaryScreen()
        room = screen.availableGeometry() if screen is not None else None
        self.resize(
            min(1300, room.width() - 80) if room else 1200,
            min(860, room.height() - 80) if room else 800,
        )
        self._apply()

    # -- settings -------------------------------------------------------------------

    def preset(self) -> FS.Preset:
        return FS.PRESETS[self.preset_cb.currentData()]

    def settings(self) -> dict[str, Any]:
        """The choices kept for the next figure; the first letter is the figure's own."""
        return {
            "preset": self.preset_cb.currentData(),
            "width": self.width_mm.value(),
            "labels": self.labels_cb.currentData(),
            "bold": self.bold_chk.isChecked(),
            "formulas": self.formula_chk.isChecked(),
            "dpi": self.dpi_cb.currentData(),
        }

    def _settings_changed(self, *_: Any) -> None:
        """A choice that changes no pixel of the figure -- the resolution -- kept for the
        next as the others are."""
        if self._on_settings is not None:
            self._on_settings(self.settings())

    def _line_changed(self, *_: Any) -> None:
        """A line width asked for: every data line takes it; the figure as drawn is left
        alone until then."""
        self._line_asked = True
        self._apply()

    def _marker_changed(self, *_: Any) -> None:
        """A marker size asked for: a line drawn without markers gets them -- a time
        series, whose own figure has none -- rather than only the lines that had them."""
        self._marker_asked = True
        self._apply()

    def _fill_widths(self, width: float | None = None) -> None:
        """The preset's columns, and Other width; the one a width is, else the width
        the figure's panels suggest (``suggest_width``)."""
        preset = self.preset()
        self.width_cb.blockSignals(True)
        self.width_cb.clear()
        for name, mm in preset.widths.items():
            self.width_cb.addItem(f"{name}, {mm:.0f} mm", mm)
        self.width_cb.addItem("Other width", None)
        self.width_cb.blockSignals(False)
        want = width if width else FS.suggest_width(self.fig, preset)
        index = next(
            (
                i
                for i in range(self.width_cb.count() - 1)
                if abs(self.width_cb.itemData(i) - want) < 0.05
            ),
            self.width_cb.count() - 1,
        )
        _quiet(self.width_cb, self.width_cb.setCurrentIndex, index)
        _quiet(self.width_mm, self.width_mm.setValue, want)
        self.width_mm.setEnabled(index == self.width_cb.count() - 1)

    def _preset_changed(self, *_: Any) -> None:
        preset = self.preset()
        self._fill_widths()
        _quiet(
            self.labels_cb,
            self.labels_cb.setCurrentIndex,
            next(i for i, (_, st) in enumerate(LABELS) if st == preset.label),
        )
        _quiet(self.bold_chk, self.bold_chk.setChecked, preset.label_bold)
        _quiet(self.legend_size, self.legend_size.setValue, preset.text)
        self._apply()

    def _width_chosen(self, *_: Any) -> None:
        mm = self.width_cb.currentData()
        self.width_mm.setEnabled(mm is None)
        if mm is not None:
            _quiet(self.width_mm, self.width_mm.setValue, mm)
        self._apply()

    # -- the figure -----------------------------------------------------------------

    def _all_legends(self) -> list[Any]:
        """Every legend the figure draws -- an axes' own, the figure's own, and one kept
        on an axes' artist list (``ax.add_artist``, the way a second legend is kept beside
        the first)."""
        from matplotlib.legend import Legend

        out: list[Any] = []
        for leg in self.fig.findobj(Legend):
            if all(leg is not seen for seen in out):
                out.append(leg)
        return out

    @staticmethod
    def _is_data_line(line: Any) -> bool:
        """A line the figure draws data with, as against a reference line (two points)
        or a grid line (not in an axes' own list): more than two points, or a marker of
        its own."""
        try:
            many = len(line.get_xdata()) > 2
        except Exception:  # noqa: BLE001 - a line with no data to measure
            many = False
        return many or str(line.get_marker()).lower() not in ("none", "", " ")

    def _legend_placed(self, *_: Any) -> None:
        """The legends where the dialog says: a corner set, or back where drawn (a
        dragged one is left where it was dragged until As drawn)."""
        pos = self.legend_cb.currentData()
        for leg, loc, anchor, _, _ in self._legends:
            if pos is None:
                leg._loc, leg._bbox_to_anchor = loc, anchor
            else:
                leg.set_loc(pos)
                leg.set_bbox_to_anchor(None)
            leg.stale = True
        self._apply()

    def _legend_styled(self) -> None:
        """The legend's own choices, laid over the layout pass: its text's size, its
        entries' columns, its frame."""
        for leg in self._all_legends():
            for t in list(leg.get_texts()) + [leg.get_title()]:
                t.set_fontsize(self.legend_size.value())
            FS.set_legend_columns(leg, self.legend_ncol.value())
            leg.set_frame_on(self.legend_frame.isChecked())
            leg.stale = True

    def _line_styled(self) -> None:
        """The data lines -- and a legend's handles, which are its own copies -- at the
        width and marker size asked for; a line drawn without markers takes round ones
        when the size is raised. Reference lines keep their stroke."""
        if not self._data_lines or not (self._line_asked or self._marker_asked):
            return
        lw, ms = self.line_width.value(), self.marker_size.value()
        for line, _, _, marker, data in self._line_styles:
            if not data:
                continue
            if self._line_asked:
                line.set_linewidth(lw)
            if self._marker_asked:
                if str(marker).lower() in ("none", "", " "):
                    line.set_marker("o" if ms > 0 else "None")
                line.set_markersize(ms)
            line.stale = True

    # -- double-click editing --------------------------------------------------------

    def _double_clicked(self, event: Any) -> None:
        """A double-click on a text, a tick's numbers, the legend or a colour bar edits it
        where it stands: the text's own letters and size, a tick's own text with its
        axis's numbers' size and rotation, the legend's place and columns, the bar's
        side."""
        if event.button != 1 or not getattr(event, "dblclick", False):
            return
        found = self._what_at(event.x, event.y)
        if found is None:
            return
        kind, target = found[0], found[1]
        if kind == "label":
            ask = _LabelChoices(self, target.get_text(), target.get_fontsize())
            if ask.exec() == QDialog.DialogCode.Accepted:
                said, size = ask.chosen()
                if said != target.get_text() or abs(size - target.get_fontsize()) > 1e-9:
                    self._set_label(target, said or "", size)
        elif kind == "tick":
            axis, tick = found[1], found[2]
            if tick is None:  # the offset text: a size, as before
                ask = _LabelChoices(self, "", self._tick_size(axis), with_text=False)
                if ask.exec() == QDialog.DialogCode.Accepted:
                    self._set_tick_size(axis, ask.chosen()[1])
            else:
                tick_ask = _TickChoices(
                    self, tick.label1.get_text(), self._tick_size(axis), self._tick_rot(axis)
                )
                if tick_ask.exec() == QDialog.DialogCode.Accepted:
                    said, size, rotation = tick_ask.chosen()
                    self._set_tick(axis, tick, said, size, rotation)
        elif kind == "legend":
            leg_ask = _LegendChoices(self, self.legend_cb.currentData(), self.legend_ncol.value())
            if leg_ask.exec() == QDialog.DialogCode.Accepted:
                self._set_legend(*leg_ask.chosen())
        else:
            names = ["Where drawn", "Right", "Left", "Top", "Bottom"]
            sides = [None, "right", "left", "top", "bottom"]
            side0 = self._bar_side(target)
            if side0 is None:
                side0 = (self._bar_drags.get(target) or {}).get("side")
            here = sides.index(side0) if side0 in sides else 0
            said, ok = QInputDialog.getItem(
                self, "Colour bar", "Where it stands:", names, here, False
            )
            if ok:
                self._set_bar(target, sides[names.index(said)])

    def _holds(self, what: Any, x: float, y: float) -> bool:
        try:
            box = what.get_window_extent(self.canvas.get_renderer())
        except Exception:  # noqa: BLE001 - an artist not drawn yet
            return False
        return bool(box.x0 - 2 <= x <= box.x1 + 2 and box.y0 - 2 <= y <= box.y1 + 2)

    def _legend_at(self, x: float, y: float) -> Any:
        """The legend under (x, y) in the canvas's pixels, the topmost first: one dragged
        onto another stays reachable, the one that covers it taking the press."""
        for leg in reversed(self._all_legends()):
            if leg.get_visible() and self._holds(leg, x, y):
                return leg
        return None

    def _bar_at(self, x: float, y: float) -> Any:
        """The colour bar's axes under (x, y), if any."""
        for ax in self.fig.axes:
            if self._is_bar(ax) and ax.get_visible() and self._holds(ax, x, y):
                return ax
        return None

    def _what_at(self, x: float, y: float) -> tuple[Any, ...] | None:
        """What stands under (x, y), in the canvas's pixels: ("legend", the legend),
        ("label", a text) -- an axis's own label, a title, a note -- ("tick", the axis,
        the tick under it or None for the offset text) or ("bar", the colour bar's axes).
        The legend is read first, as it is drawn over what it stands on; a text before a
        tick and a bar, so a bar's own label is its text; the legend's own texts and the
        ticks' are their owner's to edit."""
        from matplotlib.text import Text

        leg = self._legend_at(x, y)
        if leg is not None:
            return ("legend", leg)
        own = set()  # texts their owners' dialogs ask for
        for leg in self._all_legends():
            own.update(id(t) for t in leg.findobj(Text))
        for ax in self.fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                own.add(id(axis.offsetText))
                for tick in axis.get_major_ticks():
                    own.add(id(tick.label1))
                    own.add(id(tick.label2))
        for text in reversed(self.fig.findobj(Text)):
            if id(text) in own or not text.get_visible():
                continue
            if getattr(text, "axes", None) is not None and not text.axes.get_visible():
                continue
            if text.get_text().strip() and self._holds(text, x, y):
                return ("label", text)
        for ax in self.fig.axes:
            if not ax.get_visible():
                continue
            for axis in (ax.xaxis, ax.yaxis):
                offset = axis.get_offset_text()
                if offset.get_text() and self._holds(offset, x, y):
                    return ("tick", axis, None)
                for tick in axis.get_major_ticks():
                    if tick.label1.get_text() and self._holds(tick.label1, x, y):
                        return ("tick", axis, tick)
        cax = self._bar_at(x, y)
        if cax is not None:
            return ("bar", cax)
        return None

    @staticmethod
    def _is_bar(ax: Any) -> bool:
        """A colour bar's axes, as ``figstyle.panel_axes`` reads one."""
        return bool(ax.get_label() == "<colorbar>" or getattr(ax, "_colorbar", None) is not None)

    def _set_label(self, text: Any, said: str, size: float) -> None:
        """The label's text and its size; a panel's letter is kept by its panel too, as
        every pass makes the letters anew; the dialog closing puts the text it had back."""
        self._label_was.setdefault(text, text.get_text())
        self._label_edits[text] = (said, size)
        if text.get_gid() == FS.PANEL_LABEL and text.axes is not None:
            self._letter_edits[id(text.axes)] = (said, size)
        text.set_text(said)
        text.set_fontsize(size)
        text.stale = True
        self.canvas.draw_idle()

    @staticmethod
    def _tick_size(axis: Any) -> float:
        """The size of a tick axis's numbers as drawn: any of its labels tells it."""
        for label in axis.get_ticklabels():
            return float(label.get_fontsize())
        return 7.0  # no ticks to read it from: the style's

    def _set_tick_size(self, axis: Any, size: float) -> None:
        """A tick axis's numbers sized; kept while the dialog is open and laid over each
        layout pass."""
        self._tick_edits[axis] = size
        axis.set_tick_params(labelsize=size)
        self.canvas.draw_idle()

    @staticmethod
    def _tick_rot(axis: Any) -> float:
        """How a tick axis's numbers stand, in degrees: any of its labels tells it."""
        for label in axis.get_ticklabels():
            return float(label.get_rotation())
        return 0.0

    def _set_tick(self, axis: Any, tick: Any, said: str, size: float, rotation: float) -> None:
        """A tick axis's numbers: the size and how they stand, its every label's, and a
        text of its own for the tick double-clicked. All are kept while the dialog is
        open and laid over each layout pass: the size by the pass's put-back, the
        rotation here, the texts in the axis's formatter -- matplotlib writes the labels
        from it at every draw."""
        self._tick_edits[axis] = size
        if abs(rotation - self._tick_rot(axis)) > 1e-9:
            self._tick_rot_was.setdefault(axis, self._tick_rot(axis))
            self._tick_rots[axis] = rotation
            axis.set_tick_params(labelrotation=rotation)
        if said is not None and tick is not None and said != tick.label1.get_text():
            self._tick_said.setdefault(axis, {})[float(tick.get_loc())] = said
            self._tick_renamed(axis)
        axis.set_tick_params(labelsize=size)
        self.canvas.draw_idle()

    def _tick_renamed(self, axis: Any) -> None:
        """The axis's formatter told the ticks' own texts: wrapped with them -- again
        after every pass, which puts the formatter it found back."""
        said = self._tick_said.get(axis)
        if not said:
            return
        current = axis.get_major_formatter()
        if getattr(current, "_normet_renamed", False):
            current.said.update(said)
        else:
            current = _renamed(current, dict(said))
        axis.set_major_formatter(current)

    def _set_legend(self, pos: str | None, ncols: int) -> None:
        """The legend where the double-click asked, the panel's own choices moving with
        it."""
        index = next(
            (i for i in range(self.legend_cb.count()) if self.legend_cb.itemData(i) == pos), 0
        )
        _quiet(self.legend_cb, self.legend_cb.setCurrentIndex, index)
        _quiet(self.legend_ncol, self.legend_ncol.setValue, ncols)
        self._legend_placed()

    def _bar_side(self, cax: Any) -> str | None:
        """Which side of its parent a colour bar stands on: "right", "left", "top",
        "bottom", or None where it cannot be told."""
        cb = getattr(cax, "_colorbar", None)
        parent = cb.mappable.axes if cb is not None else None
        if parent is None or parent is cax:
            return None
        c, p = cax.get_position(), parent.get_position()
        if c.x0 >= p.x1 - 1e-9:
            return "right"
        if c.x1 <= p.x0 + 1e-9:
            return "left"
        if c.y0 >= p.y1 - 1e-9:
            return "top"
        if c.y1 <= p.y0 + 1e-9:
            return "bottom"
        return None

    def _set_bar(self, cax: Any, side: str | None) -> None:
        """The colour bar to another side of its parent -- or back to the side it was
        drawn on, None asking that: the bar taken off and one matplotlib places put in
        its stead, its label with it. Closing the dialog puts one back on the side it
        was drawn."""
        cb = getattr(cax, "_colorbar", None)
        if cb is None:
            return
        mappable, parent = cb.mappable, cb.mappable.axes
        if parent is None or parent is cax:
            return
        was = self._bar_edits.get(mappable)
        if was is None:
            side0 = self._bar_side(cax)
            if side0 is None:  # dragged: told by its own record
                side0 = (self._bar_drags.get(cax) or {}).get("side")
            if side is None or side == side0:
                self._bar_undrag(cax)  # nothing asked: back where drawn
                return
            was = self._bar_edits[mappable] = {
                "parent": parent,
                "side": side0,
                "opts": self._bar_made_opts(cax, parent, side0),
                "label": cb.ax.get_ylabel() or cb.ax.get_xlabel(),
            }
        side = was["side"] if side is None else side
        if side == self._bar_side(cax) and cax not in self._bar_drags:
            return  # already there
        self._bar_undrag(cax)  # a dragged one is made afresh
        cb.remove()
        made = self.fig.colorbar(mappable, ax=parent, location=side, **was["opts"])
        if was["label"]:
            made.set_label(was["label"])
        self.canvas.draw_idle()

    @staticmethod
    def _bar_made_opts(cax: Any, parent: Any, side: str | None) -> dict[str, float]:
        """How the bar was made, as far as it tells: matplotlib's own note of a bar it
        placed, else measured off where it stands."""
        info = dict(getattr(cax, "_colorbar_info", None) or {})
        # anchor/panchor are not carried: matplotlib resolves them for the side the bar
        # stood on, and they misplace one moved to another side.
        keys = ("fraction", "pad", "shrink", "aspect")
        told = {k: info[k] for k in keys if k in info}
        if told:
            return told
        c, p = cax.get_position(), parent.get_position()
        if side in ("right", "left") and p.width:
            frac = c.width / p.width
            pad = ((c.x0 - p.x1) if side == "right" else (p.x0 - c.x1)) / p.width
            shrink = c.height / p.height if p.height else None
        elif side in ("top", "bottom") and p.height:
            frac = c.height / p.height
            pad = ((c.y0 - p.y1) if side == "top" else (p.y0 - c.y1)) / p.height
            shrink = c.width / p.width if p.width else None
        else:
            return {}
        return {
            k: v
            for k, v in (("fraction", frac), ("pad", pad), ("shrink", shrink))
            if v is not None and v > 0
        }

    # -- dragging --------------------------------------------------------------------

    def _pressed(self, event: Any) -> None:
        """A press: a double-click edits what stands under it; otherwise a legend or a
        colour bar under it is readied for dragging -- by the dialog's own hand, so a
        legend a twin axis draws over moves all the same."""
        if event.button != 1 or event.x is None or event.y is None:
            return
        if getattr(event, "dblclick", False):
            self._double_clicked(event)
            return
        leg = self._legend_at(event.x, event.y)
        if leg is not None:
            self._drag = ["legend", leg, event.x, event.y, False]
            return
        cax = self._bar_at(event.x, event.y)
        if cax is not None:
            self._drag = ["bar", cax, event.x, event.y, False]

    def _moved(self, event: Any) -> None:
        """A dragged legend or bar follows the mouse -- a little way in first, so a click
        that wanders a pixel does not move it -- and the canvas redraws."""
        drag = self._drag
        if drag is None or event.x is None or event.y is None:
            return
        if not drag[4]:
            if abs(event.x - drag[2]) <= 3 and abs(event.y - drag[3]) <= 3:
                return
            drag[4] = True
            if drag[0] == "legend":
                if drag[1]._draggable is None:  # made draggable after the press
                    drag[1].set_draggable(True)
                drag[1]._draggable.save_offset()
            else:
                cax = drag[1]
                self._bar_drags[cax] = {
                    "locator": cax.get_axes_locator(),
                    "pos": cax.get_position().frozen(),
                    "side": self._bar_side(cax),
                }
        if drag[0] == "legend":
            drag[1]._draggable.update_offset(event.x - drag[2], event.y - drag[3])
        else:
            self._pin_bar(drag[1], event.x - drag[2], event.y - drag[3])
        self.canvas.draw_idle()

    def _released(self, event: Any) -> None:
        """The drag ended: the legend's own finalizer sets its loc, the bar stays pinned
        where it was dropped; a press that only clicked leaves both as they were."""
        drag, self._drag = self._drag, None
        if drag is None or not drag[4] or drag[0] != "legend":
            return
        if drag[1]._draggable is not None:
            drag[1]._draggable.finalize_offset()
        self.canvas.draw_idle()

    def _pin_bar(self, cax: Any, dx: float, dy: float) -> None:
        """The bar under the press moved as the mouse moved: pinned where it is dropped
        -- in the figure's own fractions, kept within the figure."""
        from matplotlib.transforms import Bbox

        x0, y0, w, h = self._bar_drags[cax]["pos"].bounds
        bbox = self.fig.bbox
        box = Bbox.from_bounds(
            min(max(x0 + dx / bbox.width, 0.0), 1.0 - w),
            min(max(y0 + dy / bbox.height, 0.0), 1.0 - h),
            w,
            h,
        )
        cax.set_axes_locator(lambda ax, renderer, box=box: box)

    def _bar_undrag(self, cax: Any) -> None:
        """A dragged bar back as it was laid out."""
        rec = self._bar_drags.pop(cax, None)
        if rec is None:
            return
        cax.set_axes_locator(rec["locator"])
        if rec["locator"] is None:
            cax.set_position(rec["pos"])
        cax.stale = True
        self.canvas.draw_idle()

    def _screen_dpi(self) -> float:
        """Logical pixels to a physical inch on this screen: a figure shown at this many
        dots to its inch is shown at its size."""
        screen = self.screen() or QGuiApplication.primaryScreen()
        dpi = float(screen.physicalDotsPerInch()) if screen is not None else 0.0
        return dpi if dpi > 20 else 96.0

    def _apply(self, *_: Any) -> None:
        if self._put_back is not None:
            self._put_back()
        preset = self.preset()
        # The journal's name a link of its own: the words around it are translated as
        # they stand, the name is not.
        self.source.setText(
            f"The guide: <a href='{preset.source}'>{preset.name}</a>"
            + "".join(f"<br>{note}" for note in preset.notes)
        )
        self.height_mm.setEnabled(not self.height_auto.isChecked())
        self._put_back = FS.to_print(
            self.fig,
            preset,
            self.width_mm.value(),
            None if self.height_auto.isChecked() else self.height_mm.value(),
            labels=self.labels_cb.currentData(),
            label_bold=self.bold_chk.isChecked(),
            start=self.start_cb.currentData(),
            formulas=self.formula_chk.isChecked(),
            title=self.title_chk.isChecked(),
        )
        self.start_cb.setEnabled(self.labels_cb.currentData() is not None)
        for leg in self._all_legends():
            leg.set_draggable(True)
        self._legend_styled()
        self._line_styled()
        for text, (said, size) in self._label_edits.items():  # over the layout pass's own
            text.set_text(said)
            text.set_fontsize(size)
            text.stale = True
        for ax in FS.panel_axes(self.fig):  # the letters each pass makes anew, told again
            edit = self._letter_edits.get(id(ax))
            if edit is None:
                continue
            for t in getattr(ax, "texts", ()):
                if t.get_gid() == FS.PANEL_LABEL:
                    t.set_text(edit[0])
                    t.set_fontsize(edit[1])
                    t.stale = True
        for axis, size in self._tick_edits.items():
            axis.set_tick_params(labelsize=size)
        for axis, rotation in self._tick_rots.items():
            axis.set_tick_params(labelrotation=rotation)
        for axis in self._tick_said:
            self._tick_renamed(axis)
        w_in, h_in = self.fig.get_size_inches()
        if self.height_auto.isChecked():
            _quiet(self.height_mm, self.height_mm.setValue, h_in * FS.MM)
        self._show(w_in, h_in)
        self._show_checks(FS.preflight(self.fig, preset))
        if self._on_settings is not None:
            self._on_settings(self.settings())

    def _show(self, w_in: float, h_in: float) -> None:
        """The canvas the figure's size at the screen's dots to the inch -- or as many as
        fit the window -- the figure's own size in inches kept."""
        dpi = self._screen_dpi()
        if self.zoom_cb.currentData() == "fit":
            room = self.scroll.viewport().size()
            dpi = max(20.0, min((room.width() - 24) / w_in, (room.height() - 24) / h_in))
        ratio = self.canvas.device_pixel_ratio
        self.fig._original_dpi = dpi
        self.fig._set_dpi(dpi * ratio, forward=False)
        self.canvas.setFixedSize(round(w_in * dpi), round(h_in * dpi))
        self.fig.set_size_inches(w_in, h_in, forward=False)
        self.canvas.draw_idle()

    def showEvent(self, event: Any) -> None:  # noqa: N802
        """Shown: the fitted view is computed again once the layout has its real room."""
        super().showEvent(event)
        _call_later(self, self._refit)

    def _refit(self) -> None:
        if self.zoom_cb.currentData() == "fit" and self._put_back is not None:
            self._show(*self.fig.get_size_inches())

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refit()

    def _show_checks(self, found: list[FS.Finding]) -> None:
        from .uilang import tr  # a list's items are not translated as set

        self.checks.clear()
        for f in found:
            item = QListWidgetItem(("✓  " if f.ok else "⚠  ") + tr(f.said))
            if not f.ok:
                item.setForeground(QColor("#c0392b"))
            self.checks.addItem(item)

    @property
    def findings(self) -> list[FS.Finding]:
        """What the guide asks of the figure as it is now laid out."""
        return FS.preflight(self.fig, self.preset())

    def _save(self, path: str | None = None) -> str | None:
        """Save the figure as shown, in a format the journal takes; returns the path."""
        preset = self.preset()
        kinds = [k for k in preset.formats if k in KINDS]
        if path is None:
            path, chosen = QFileDialog.getSaveFileName(
                self,
                "Save figure for print",
                f"figure.{kinds[0]}",
                ";;".join(KINDS[k] for k in kinds),
            )
            if not path:
                return None
            if not os.path.splitext(path)[1]:
                names = [KINDS[k] for k in kinds]
                path += "." + (kinds[names.index(chosen)] if chosen in names else kinds[0])
        return self.save_to(path)

    def save_to(self, target: Any, fmt: str | None = None) -> Any:
        """The figure as shown, at exactly its printed size and the resolution chosen."""
        return FS.save_for_print(
            self.fig, target, self.preset(), fmt=fmt, dpi=self.dpi_cb.currentData()
        )

    def done(self, result: int) -> None:
        if self._put_back is not None:
            self._put_back()
            self._put_back = None
        for line, lw, ms, marker, _ in self._line_styles:
            line.set_linewidth(lw)
            line.set_markersize(ms)
            line.set_marker(marker)
            line.stale = True
        self._drag = None
        for cax in list(self._bar_drags):
            self._bar_undrag(cax)
        for leg, loc, anchor, ncols, frame in self._legends:
            leg.set_draggable(False)
            leg._loc, leg._bbox_to_anchor = loc, anchor
            FS.set_legend_columns(leg, ncols)
            leg.set_frame_on(frame)
            leg.stale = True
        for text, was in self._label_was.items():
            text.set_text(was)
            text.stale = True
        for axis, rotation in self._tick_rot_was.items():
            axis.set_tick_params(labelrotation=rotation)
        for axis in list(self._tick_said):
            formatter = axis.get_major_formatter()
            if getattr(formatter, "_normet_renamed", False):
                axis.set_major_formatter(formatter.under)
        for mappable, was in self._bar_edits.items():
            if mappable.axes is None:
                continue
            for ax in [
                a
                for a in self.fig.axes
                if getattr(a, "_colorbar", None) is not None and a._colorbar.mappable is mappable
            ]:
                ax._colorbar.remove()
            if was["side"] is not None:
                back = self.fig.colorbar(
                    mappable, ax=was["parent"], location=was["side"], **was["opts"]
                )
                if was["label"]:
                    back.set_label(was["label"])
        fig, home = self.fig, self._home
        fig._original_dpi = self._orig0
        fig._set_dpi(self._dpi0, forward=False)
        fig.set_size_inches(*self._size0, forward=False)
        fig.set_canvas(home)
        apply_figure_theme(fig, self._dark)
        home.draw_idle()
        super().done(result)
