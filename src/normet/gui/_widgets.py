"""Shared widgets for the normet Qt GUI.

Design language: a scrollable parameter
panel on the left with grouped, numbered workflow steps and prominent
"▶ Run" buttons; result tabs on the right rendered on a :class:`CanvasTab`
— a matplotlib figure with a navigation toolbar, optionally topped by a
verdict banner (green / amber / red), free-text notes, a compact table and
an export button.

Spin/combo/list boxes normally grab the scroll wheel even without focus,
which fights the surrounding scroll area — scrolling the panel silently
changes whatever value happens to be under the cursor.  The ``NoWheel*``
variants require a click (focus) first and otherwise let the wheel event
bubble up, so hovering-and-scrolling scrolls the panel, not the widget.

Figures follow the window's light/dark appearance (:mod:`.theme`); what the
toolbar saves is always in the figure's own light colours.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

import pandas as pd
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import (
    QAbstractAnimation,
    QEvent,
    QEasingCurve,
    QRect,
    QPropertyAnimation,
    QSize,
    Qt,
    QTimer,
)
from PySide6.QtGui import QAction, QColor, QIcon, QImage, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLayout,
    QListWidget,
    QMenu,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)

from .theme import apply_figure_theme, is_dark_mode, light_figure, muted_color, theme_manager

#: Traffic-light palette for verdict banners, per theme: the saturated Apple system
#: colours read on a light window; their lighter dark-mode variants on a dark one.
VERDICT_COLOR = {"ok": "#34c759", "warn": "#ff9500", "error": "#ff3b30"}
VERDICT_COLOR_DARK = {"ok": "#30d158", "warn": "#ff9f0a", "error": "#ff453a"}


def verdict_color(level: str, dark: bool | None = None) -> str:
    """The verdict-banner colour for *level* ("ok", "warn", "error") in the window's theme."""
    if dark is None:
        dark = is_dark_mode()
    table = VERDICT_COLOR_DARK if dark else VERDICT_COLOR
    return table.get(level, "palette(text)")


def hint_style(small: bool = False) -> str:
    """The muted secondary-hint / file-label style for the window's current theme
    (see theme.muted_color); ask again after a theme change."""
    size = "font-size:11px;" if small else ""
    return f"color:{muted_color()};{size}"

RUN_BUTTON_STYLE = "font-weight:600;padding:8px;"

#: Checkable "filter" button — unmistakably highlighted while pressed/active,
#: for one-click select-all/deselect-all toggles (e.g. "Time variables").
TOGGLE_BUTTON_STYLE = """
QPushButton:checked {
    background-color: #2c7bb6;
    color: white;
    font-weight: 600;
    border: 1px solid #1f5a86;
}
"""


def wrap_row(layout: QLayout) -> QWidget:
    """A layout as a widget, for a form row that holds several controls."""
    w = QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    w.setLayout(layout)
    return w


class _NoWheelMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoWheelSpinBox(_NoWheelMixin, QSpinBox):
    pass


class NoWheelDoubleSpinBox(_NoWheelMixin, QDoubleSpinBox):
    pass


class NoWheelComboBox(_NoWheelMixin, QComboBox):
    pass


class NoWheelListWidget(_NoWheelMixin, QListWidget):
    pass


def run_button(text: str, tooltip: str = "") -> QPushButton:
    """A prominent workflow action button ("▶  Run …")."""
    btn = QPushButton(text)
    btn.setStyleSheet(RUN_BUTTON_STYLE)
    if tooltip:
        btn.setToolTip(tooltip)
    return btn


class SortItem(QTableWidgetItem):
    """A cell shown as text and sorted by a value of its own -- a number where the text
    is one, so 10 comes after 9 -- with the cells that have none last."""

    def __init__(self, text: str, key: Any = None):
        super().__init__(text)
        self.key = key

    def __lt__(self, other: Any) -> bool:
        a, b = self.key, getattr(other, "key", None)
        if a is None or b is None:
            return a is not None and b is None
        try:
            return a < b
        except TypeError:
            return str(a) < str(b)


def fill_sortable(table: QTableWidget, header: list[str], rows: list[list[Any]]) -> None:
    """Put `rows` in `table` and let it be sorted by any column; a cell may be a plain
    value or a (text, sort-key) pair, and numeric text sorts by its number."""
    table.setSortingEnabled(False)
    table.setColumnCount(len(header))
    table.setHorizontalHeaderLabels(header)
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            if isinstance(cell, tuple):
                text, key = cell
            else:
                text, key = str(cell), cell
                if isinstance(key, str):
                    try:
                        key = float(key)
                    except ValueError:
                        key = None
            table.setItem(r, c, SortItem(str(text), key))
    table.resizeColumnsToContents()
    # Sortable, but left in the order given until a header is clicked.
    table.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
    table.setSortingEnabled(True)


def copy_table_rows(table: QTableWidget, rows: list[int]) -> None:
    """The picked rows of a table as TSV on the clipboard."""
    lines = []
    for r in rows:
        lines.append(
            "\t".join(
                table.item(r, c).text() if table.item(r, c) else ""
                for c in range(table.columnCount())
            )
        )
    QApplication.clipboard().setText("\n".join(lines))


def table_context_menu(table: QTableWidget, pos) -> None:
    """Right-click a table row: copy it as TSV."""
    row = table.rowAt(pos.y())
    if row < 0:
        return
    table.selectRow(row)
    menu = QMenu(table)
    menu.addAction("Copy row", lambda: copy_table_rows(table, [row]))
    menu.exec(table.mapToGlobal(pos))


def toggle_button(text: str, tooltip: str = "") -> QPushButton:
    """A checkable select-all/deselect-all filter button, clearly highlighted
    while active (one click ticks the matching items, the next click
    unticks them)."""
    btn = QPushButton(text)
    btn.setCheckable(True)
    btn.setStyleSheet(TOGGLE_BUTTON_STYLE)
    if tooltip:
        btn.setToolTip(tooltip)
    return btn


def app_icon_path() -> str:
    """The app's icon, a 256 px PNG shipped in the package (and so in the app)."""
    from importlib.resources import files

    return str(files("normet.gui").joinpath("assets", "normet.png"))


def hint_label(text: str, small: bool = True) -> QLabel:
    """A muted, word-wrapped helper label, recoloured when the theme changes."""
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    _apply_hint_style(lbl, small)
    try:
        theme_manager().themeChanged.connect(lambda dark: _apply_hint_style(lbl, small, dark))
    except RuntimeError:  # no application yet: the colour asked for stands
        pass
    return lbl


def _apply_hint_style(lbl: QLabel, small: bool, dark: bool | None = None) -> None:
    size = "font-size:11px;" if small else ""
    lbl.setStyleSheet(f"color:{muted_color(dark)};{size}")


def _style_verdict(lbl: QLabel, level: str, dark: bool | None = None) -> None:
    lbl.setStyleSheet(f"color:{verdict_color(level, dark)};font-weight:600;padding:4px;")


def as_figure(obj: Any):
    """Resolve a matplotlib Figure from a plotting helper's return value.

    normet's plotting functions variously return a Figure, an Axes, or draw on
    the current pyplot figure and return None.
    """
    import matplotlib.pyplot as plt

    if obj is None:
        return plt.gcf()
    if hasattr(obj, "add_subplot"):  # Figure
        return obj
    fig = getattr(obj, "figure", None)  # Axes (or anything carrying one)
    if fig is None and hasattr(obj, "get_figure"):
        fig = obj.get_figure()
    return fig if fig is not None else plt.gcf()


@functools.cache
def _canvas_class() -> type:
    """FullWidthCanvas, made on first use: matplotlib's Qt canvas is imported with it."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from shiboken6 import isValid

    class FullWidthCanvas(FigureCanvasQTAgg):
        """A canvas that draws only what is seen, and covers its whole widget.

        A canvas on a tab nobody has opened has nothing to show, and drawing it costs as
        much as drawing the one on screen: after each step every tab's figure was drawn,
        the hidden ones too. So a hidden canvas keeps its draw until it is shown, and
        draws once then.

        On resize matplotlib sets the figure to width/dpi inches, and for some widths
        that round trip lands just under the integer in binary floating point (9.04 x
        100 comes back 903.9999999999999), so Agg renders one column short and the
        widget's right-most pixel keeps whatever was there. A ten-millionth of an inch
        makes the product land on the integer."""

        _draw_held = False

        def draw_idle(self) -> None:
            if not self.isVisible():
                self._draw_held = True
                return
            super().draw_idle()

        def showEvent(self, event: Any) -> None:  # noqa: N802
            super().showEvent(event)
            if self._draw_held:
                self._draw_held = False
                super().draw_idle()

        def _draw_idle(self) -> None:
            # A draw queued for a canvas deleted meanwhile (its figure made again).
            if not isValid(self):
                return
            super()._draw_idle()

        def resizeEvent(self, event: Any) -> None:  # noqa: N802
            super().resizeEvent(event)
            ratio = self.device_pixel_ratio
            w, h = self.width() * ratio, self.height() * ratio
            if w > 0 and h > 0:
                dpi = self.figure.dpi
                self.figure.set_size_inches(w / dpi + 1e-9, h / dpi + 1e-9, forward=False)

    return FullWidthCanvas


def adaptive_layout(fig: Any) -> None:
    """Solve the figure's margins at every draw, not once at its design size.

    A one-off ``tight_layout()`` fixes the margins as fractions of the size the figure
    was made at; on a canvas of another shape they no longer leave the room its labels
    need -- titles run into the panel above, tick labels are cut. A layout engine
    measures the texts at each draw. A figure that chose its own engine keeps it."""
    from matplotlib.layout_engine import PlaceHolderLayoutEngine

    if not isinstance(fig.get_layout_engine(), (type(None), PlaceHolderLayoutEngine)):
        return
    try:
        fig.set_layout_engine("constrained", h_pad=0.04, w_pad=0.04, hspace=0.02, wspace=0.02)
    except Exception:  # noqa: BLE001 - keep the margins it was made with
        pass


class ThemedToolbar(NavigationToolbar2QT):
    """The figure toolbar: icons
    16 pt and tinted for the theme, the bar in the window's own colours, Save
    writing the figure in its light colours, and *For print…* -- the figure at
    a journal's column width and lettering -- before the pointer's read-out."""

    #: The icons' size, in points: the size of the window's own small buttons.
    #: Matplotlib's 24, with its padding, made the bar the tallest control.
    ICON = 16

    def __init__(self, canvas: Any, parent: QWidget | None = None) -> None:
        super().__init__(canvas, parent)
        self.print_action = QAction("For print…", self)
        self.print_action.setToolTip(
            "Lay the figure out at a journal's column width and lettering, and save it."
        )
        # `triggered` hands a slot its checked flag: not a figure to print.
        self.print_action.triggered.connect(lambda: self.export_for_print())
        label = getattr(self, "locLabel", None)
        where = next(
            (a for a in self.actions() if label is not None and self.widgetForAction(a) is label),
            None,
        )
        if where is not None:
            self.insertAction(where, self.print_action)
        else:
            self.addAction(self.print_action)
        # The figure's right-click menu: copy it, save it, or lay it out for print --
        # the toolbar's three figure actions, reachable from the plot itself.
        canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        canvas.customContextMenuRequested.connect(self._canvas_menu)
        self._made_in: tuple[str, ...] | None = None
        theme_manager().themeChanged.connect(self._on_theme_changed)
        self.apply_theme(theme_manager().dark)

    def _render_icon(self, name: str, dark: bool) -> QIcon:
        """Matplotlib's icon *name*, tinted the theme's ink, at 1x and 2x."""
        from matplotlib import cbook

        try:
            regular = cbook._get_data_path("images", name)
        except Exception:
            return QIcon()
        large = regular.with_name(regular.name.replace(".png", "_large.png"))
        tint = QColor("#F2F2F7" if dark else "#1C1C1E")
        icon = QIcon()
        for path, dpr in ((regular, 1.0), (large, 2.0)):
            if not path.exists():
                continue
            src = QImage(str(path)).convertToFormat(QImage.Format.Format_ARGB32)
            if src.isNull():
                continue
            coloured = QImage(src.size(), QImage.Format.Format_ARGB32)
            coloured.fill(Qt.GlobalColor.transparent)
            p = QPainter(coloured)
            p.drawImage(0, 0, src)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
            p.fillRect(coloured.rect(), tint)
            p.end()
            pm = QPixmap.fromImage(coloured)
            pm.setDevicePixelRatio(dpr)
            icon.addPixmap(pm)
        return icon

    def _icon(self, name: str) -> QIcon:  # matplotlib's hook for each tool's icon
        return self._render_icon(name, dark=theme_manager().dark)

    @staticmethod
    def _palette_key() -> tuple[str, ...]:
        pal = QApplication.palette()
        return tuple(
            pal.color(role).name()
            for role in (
                QPalette.ColorRole.Window,
                QPalette.ColorRole.WindowText,
                QPalette.ColorRole.Highlight,
            )
        )

    def apply_theme(self, dark: bool) -> None:
        """Icons tinted for the theme; the bar in the application palette's window
        colour, its hover, pressed and dividing shades mixed from that and the
        text's -- not a colour of its own, which stood out from the window."""
        for _text, _tip, image, callback in self.toolitems:
            if image and callback in self._actions:
                self._actions[callback].setIcon(self._render_icon(image + ".png", dark))
        self.setIconSize(QSize(self.ICON, self.ICON))
        pal = QApplication.palette()
        ground = pal.color(QPalette.ColorRole.Window)
        ink = pal.color(QPalette.ColorRole.WindowText)

        def mix(t: float) -> str:
            return QColor.fromRgbF(
                ground.redF() + (ink.redF() - ground.redF()) * t,
                ground.greenF() + (ink.greenF() - ground.greenF()) * t,
                ground.blueF() + (ink.blueF() - ground.blueF()) * t,
            ).name()

        accent = pal.color(QPalette.ColorRole.Highlight).name()
        self._made_in = self._palette_key()
        self.setStyleSheet(
            f"""
            QToolBar {{ background: {ground.name()}; border: none; padding: 0px 2px;
                        spacing: 0px; }}
            QToolButton {{ background: transparent; border: 1px solid transparent;
                           border-radius: 4px; padding: 1px; margin: 0px; }}
            QToolButton:hover {{ background: {mix(0.08)}; }}
            QToolButton:pressed, QToolButton:checked {{ background: {mix(0.15)};
                                                       border-color: {accent}; }}
            QToolBar::separator {{ width: 1px; background: {mix(0.18)}; margin: 3px 3px; }}
            QLabel {{ color: {mix(0.45)}; font-size: 11px; padding-right: 4px; }}
            """
        )

    def _on_theme_changed(self, dark: bool) -> None:
        try:
            self.apply_theme(dark)
        except (RuntimeError, AttributeError):  # the bar deleted underneath
            pass

    def changeEvent(self, event: Any) -> None:  # noqa: N802
        """Made again when the palette's colours change: a theme can be announced
        before the palette has changed, and the bar made then is in the old one."""
        super().changeEvent(event)
        if (
            event.type() == QEvent.Type.PaletteChange
            and self._made_in is not None
            and self._palette_key() != self._made_in
        ):
            self.apply_theme(theme_manager().dark)

    def export_for_print(self) -> None:
        from .printview import PrintDialog

        PrintDialog(self.canvas, self.window()).exec()

    def _canvas_menu(self, pos: Any) -> None:
        """The figure's right-click menu: the toolbar's figure actions, from the plot.
        Lambdas: `triggered` hands a slot its checked, which the slots would read as an
        argument (the print_action's click learned the same)."""
        menu = QMenu(self.canvas)
        menu.addAction("Copy image", lambda: self.copy_figure())
        menu.addAction("Save image…", lambda: self.save_figure())
        menu.addSeparator()
        menu.addAction("For print…", lambda: self.export_for_print())
        menu.exec(self.canvas.mapToGlobal(pos))

    def copy_figure(self) -> None:
        """The figure's bitmap to the clipboard, at the screen's resolution."""
        fig = self.canvas.figure
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = bytes(fig.canvas.buffer_rgba())
        img = QImage(buf, w, h, QImage.Format.Format_RGBA8888).copy()
        QApplication.clipboard().setImage(img)

    def save_figure(self, *args: Any) -> Any:
        with light_figure(self.canvas.figure):
            return super().save_figure(*args)


class FlowLayout(QLayout):
    """A row of widgets that wraps onto further rows when the width runs out."""

    def __init__(
        self, parent: QWidget | None = None, spacing: int = 8, center: bool = False
    ) -> None:
        super().__init__(parent)
        self._items: list[Any] = []
        self._gap = spacing
        # Each widget centred in the height of its line's controls; one laid out with
        # Qt.AlignTop (setAlignment) hangs from their top instead -- a box taller than
        # the buttons beside it, its top on theirs rather than above them.
        self._center = center

    def addItem(self, item: Any) -> None:  # noqa: N802
        self._items.append(item)

    def addWidget(self, w: QWidget) -> None:  # noqa: N802
        self.addItem(QWidgetItem(w))
        self.addChildWidget(w)
        self.invalidate()

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, i: int) -> Any:  # noqa: N802
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i: int) -> Any:  # noqa: N802
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._place(QRect(0, 0, width, 0), move=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._place(rect, move=True)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _place(self, rect: QRect, move: bool) -> int:
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, row_h = r.x(), r.y(), 0
        lines: list[list[tuple[Any, int, int]]] = [[]]
        for item in self._items:
            hint = item.sizeHint()
            if x + hint.width() > r.right() + 1 and row_h > 0:
                x, y, row_h = r.x(), y + row_h + self._gap, 0
                lines.append([])
            lines[-1].append((item, x, y))
            x += hint.width() + self._gap
            row_h = max(row_h, hint.height())
        if move:
            for line in lines:
                hangs = [it.alignment() & Qt.AlignmentFlag.AlignTop for it, _x, _y in line]
                tall = max(
                    (
                        it.sizeHint().height()
                        for (it, _x, _y), h in zip(line, hangs, strict=True)
                        if not h
                    ),
                    default=0,
                )
                for (item, ix, iy), hang in zip(line, hangs, strict=True):
                    hint = item.sizeHint()
                    dy = (tall - hint.height()) // 2 if self._center and not hang else 0
                    item.setGeometry(QRect(ix, iy + dy, hint.width(), hint.height()))
        return y + row_h - rect.y() + m.bottom()


#: What a time-series figure can be drawn at: (label, pandas rule, seconds).
#: ``None`` is the series as recorded.
RESOLUTIONS: tuple[tuple[str, str | None, float], ...] = (
    ("As recorded", None, 0.0),
    ("Hourly", "h", 3600.0),
    ("Daily", "D", 86400.0),
    ("Weekly", "W", 7 * 86400.0),
    ("Monthly", "MS", 30.4 * 86400.0),
    ("Yearly", "YS", 365.25 * 86400.0),
)


def _time_index(df: pd.DataFrame) -> pd.DatetimeIndex | None:
    """The frame's timestamps: its index, or its ``date`` column."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    if "date" in df.columns:
        idx = pd.DatetimeIndex(pd.to_datetime(df["date"], errors="coerce"))
        return idx
    return None


def resample_frame(df: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    """*df*'s numeric columns averaged to *rule* (a pandas offset alias).

    A frame with a ``date`` column comes back with one, a frame indexed by
    time comes back indexed by time; ``rule=None`` (or a frame with no
    timestamps) returns *df* itself. Empty periods stay as NaN, so the lines
    break across gaps instead of bridging them.
    """
    idx = _time_index(df)
    if rule is None or idx is None:
        return df
    by_date = not isinstance(df.index, pd.DatetimeIndex)
    d = df.drop(columns="date") if by_date else df
    d = d.select_dtypes("number").set_axis(idx, axis=0)
    d = d[d.index.notna()]
    out = d.resample(rule).mean()
    if by_date:
        out = out.rename_axis("date").reset_index()
    return out


def _step_seconds(idx: pd.DatetimeIndex) -> float:
    """The record's usual time step, in seconds (0 when it cannot tell)."""
    idx = idx[idx.notna()]
    if len(idx) < 2:
        return 0.0
    step = pd.Series(idx).sort_values().diff().dropna().median()
    return float(step.total_seconds()) if pd.notna(step) else 0.0


def _follow_language(tab: CanvasTab) -> None:
    """Have *tab* redraw its figure when the window's language changes; held
    weakly, so a closed window's tabs are not kept alive by it."""
    import weakref

    from . import uilang

    ref = weakref.ref(tab)

    def changed(_code: str) -> None:
        t = ref()
        if t is None:
            return
        try:
            t._relanguage()
        except RuntimeError:  # its Qt side deleted already
            pass

    uilang.on_change(changed)


class CanvasTab(QWidget):
    """A result tab: verdict banner + notes + table + matplotlib figure.

    All content is rebuilt on each ``show_*`` call; the previous figure is
    closed so long GUI sessions do not accumulate matplotlib state. The figure
    is shown in the window's light or dark colours and follows a change.
    """

    def __init__(self, placeholder: str = "Run an analysis to see results here.") -> None:
        super().__init__()
        self._placeholder = placeholder
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = None
        self._verdict: QLabel | None = None
        self._verdict_level: str | None = None
        #: The time-series result being shown (see :meth:`show_timeseries`).
        self._series: tuple[Callable[[pd.DataFrame], Any], pd.DataFrame, dict[str, Any]] | None
        self._series = None
        #: Several results the tab can show, one at a time (see :meth:`show_views`):
        #: the selector's label, each result by name, and the one shown.
        self._views: tuple[str, dict[str, tuple[Any, ...]], str] | None = None
        #: The resolution chosen on this tab (a RESOLUTIONS label), or None
        #: for the default each result asks for.
        self._resolution: str | None = None
        #: Controls of the tab's owner on the Resolution row, before it and at its right
        #: end: kept, not rebuilt, as the row is at each draw.
        self.bar_lead: QWidget | None = None
        self.bar_tail: list[QWidget] = []
        #: Of those, the ones taller than the row's buttons: their top on the buttons'.
        self.bar_top: list[QWidget] = []
        #: The pandas rule the time series on show is averaged to (None: as recorded).
        self.rule: str | None = None
        self._show_placeholder()
        theme_manager().themeChanged.connect(self._retheme)
        _follow_language(self)

    def _relanguage(self) -> None:
        """The figure's texts written again in the window's language (.uilang)."""
        if self.canvas is not None:
            from .uilang import figure_texts

            figure_texts(self.canvas.figure)
            self.canvas.draw_idle()

    def _retheme(self, dark: bool) -> None:
        if self._verdict is not None:
            _style_verdict(self._verdict, self._verdict_level, dark)
        if self.canvas is not None:
            apply_figure_theme(self.canvas.figure, dark)
            self.canvas.draw_idle()

    # ------------------------------------------------------------------ util
    def _show_placeholder(self) -> None:
        lbl = QLabel(self._placeholder)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color:{muted_color()};")
        self._layout.addWidget(lbl)

    def clear(self) -> None:
        import matplotlib.pyplot as plt

        going = {self._layout.itemAt(i).widget() for i in range(self._layout.count())}
        for kept in (self.bar_lead, *self.bar_tail):
            if kept is not None and kept.parentWidget() in going:
                kept.hide()  # out of the row before it goes (not the row made to follow it)
                kept.setParent(self)
        while self._layout.count():
            w = self._layout.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        if self.canvas is not None:
            plt.close(self.canvas.figure)
        self.canvas = None
        self._series = None
        self._views = None
        self._verdict = None
        self._verdict_level = None

    # --------------------------------------------------------------- content
    def show_message(self, *lines: str) -> None:
        """Replace the tab content with informational text."""
        self.clear()
        for text in lines:
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setStyleSheet("padding:6px;")
            self._layout.addWidget(lbl)
        self._layout.addStretch(1)

    def show_figure(self, fig: Any, top_widget: QWidget | None = None) -> None:
        self.show_result(fig=fig, top_widget=top_widget)

    def show_timeseries(
        self,
        render: Callable[[pd.DataFrame], Any],
        data: pd.DataFrame,
        *,
        default: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Show a time-series result with a *Resolution* selector above it.

        *render* draws a figure from a frame like *data* (indexed by time, or
        with a ``date`` column); it is called again with *data* averaged to
        the chosen resolution whenever that changes. *default* is the
        RESOLUTIONS label used until one is chosen on this tab; *kwargs* go
        to :meth:`show_result` (verdict, lines, ...) and are kept as they are.
        """
        self._series = (render, data, {"_default": default, **kwargs})
        self._views = None
        self._draw_series()

    def show_views(
        self,
        views: dict[str, tuple[Callable[[pd.DataFrame], Any], pd.DataFrame, dict[str, Any]]],
        current: str,
        label: str = "Method",
    ) -> None:
        """Several time-series results on one tab, each ``(render, data, kwargs)`` as
        :meth:`show_timeseries` takes them; a selector named *label* beside
        *Resolution* picks the one drawn, and the resolution holds across them."""
        render, data, kwargs = views[current]
        self._series = (render, data, {"_default": None, **kwargs})
        self._views = (label, views, current)
        self._draw_series()

    def _usable_resolutions(self, data: pd.DataFrame) -> list[tuple[str, str | None, bool]]:
        """Each resolution, and whether it is coarser than the record's step
        and still leaves at least two points to draw."""
        idx = _time_index(data)
        step = _step_seconds(idx) if idx is not None else 0.0
        span = (
            float((idx.max() - idx.min()).total_seconds()) if idx is not None and len(idx) else 0.0
        )
        out = []
        for label, rule, secs in RESOLUTIONS:
            ok = rule is None or (idx is not None and secs > step * 1.01 and span >= 2 * secs)
            out.append((label, rule, ok))
        return out

    #: A finer-than-daily record of more points than this opens as daily means: months
    #: of hours drawn as they are are a solid band no line can be read in.
    DENSE_POINTS = 2000

    def _dense_default(self, data: pd.DataFrame) -> str:
        """The resolution a result opens at: daily means for a dense sub-daily record,
        else as recorded. The selector changes it, and keeps what is chosen."""
        idx = _time_index(data)
        if idx is not None and len(idx) > self.DENSE_POINTS and _step_seconds(idx) < 86400:
            return "Daily"
        return "As recorded"

    def _draw_series(self) -> None:
        if self._series is None:
            return
        render, data, opts = self._series
        kwargs = {k: v for k, v in opts.items() if k != "_default"}
        options = self._usable_resolutions(data)
        usable = {label: rule for label, rule, ok in options if ok}
        label = self._resolution or opts.get("_default") or self._dense_default(data)
        if label not in usable:
            label = "As recorded"

        bar = QWidget()
        groups: list[QWidget] = []  # each kept whole where the row wraps

        def group(*widgets: QWidget) -> QWidget:
            host = QWidget()
            h = QHBoxLayout(host)
            h.setContentsMargins(0, 0, 0, 0)
            for w in widgets:
                h.addWidget(w)
            groups.append(host)
            return host

        views = self._views
        if views is not None and len(views[1]) > 1:
            title, results, shown = views
            pick = NoWheelComboBox()
            pick.addItems(list(results))
            pick.setCurrentText(shown)
            pick.setToolTip("Which of the results to draw.")

            def picked(key: str) -> None:
                render_, data_, kwargs_ = results[key]
                self._series = (render_, data_, {"_default": None, **kwargs_})
                self._views = (title, results, key)
                QTimer.singleShot(0, self._draw_series)

            pick.currentTextChanged.connect(picked)
            group(QLabel(title), pick)
        combo = NoWheelComboBox()
        for name, _rule, ok in options:
            combo.addItem(name)
            if not ok:
                item = combo.model().item(combo.count() - 1)
                item.setEnabled(False)
                item.setToolTip("Not coarser than the record's own time step, or too few points.")
        combo.setCurrentText(label)
        combo.setToolTip(
            "Draw the figure at this time step: each period's mean of every series\n"
            "(bands too). Only the figure changes; the result and its export do not."
        )
        if _time_index(data) is not None:
            # Only a result with a time axis has periods to average; for the rest
            # (a fold table) a selector with one usable entry is noise.
            group(QLabel("Resolution"), combo)
        if self.bar_lead is not None or self.bar_tail:
            # The owner's controls too: a row that wraps where the window is narrow,
            # rather than one that holds the window wide.
            flow = FlowLayout(bar, spacing=12, center=True)
            flow.setContentsMargins(4, 2, 4, 0)
            for w in [self.bar_lead, *groups, *self.bar_tail]:
                if w is not None:
                    flow.addWidget(w)
                    if w in self.bar_top:
                        flow.setAlignment(w, Qt.AlignmentFlag.AlignTop)
                    w.show()
        else:
            row = QHBoxLayout(bar)
            row.setContentsMargins(4, 2, 4, 0)
            for i, w in enumerate(groups):
                if i:
                    row.addSpacing(12)
                row.addWidget(w)
            row.addStretch(1)

        def chosen(name: str) -> None:
            self._resolution = name
            # Rebuilt after the signal returns: the combo is part of what goes.
            QTimer.singleShot(0, self._draw_series)

        combo.currentTextChanged.connect(chosen)

        try:
            self.rule = usable[label]
            fig = render(resample_frame(data, usable[label]))
        except Exception:  # noqa: BLE001 - fall back to the series as recorded
            if usable[label] is None:
                raise
            import logging

            logging.getLogger(__name__).exception("Figure at %s failed", label)
            self._resolution = "As recorded"
            self.rule = None
            fig = render(data)
            combo.blockSignals(True)
            combo.setCurrentText("As recorded")
            combo.blockSignals(False)
        series = self._series
        self.show_result(fig, top_widget=bar, **kwargs)
        self._series, self._views = series, views

    def show_result(
        self,
        fig: Any = None,
        *,
        verdict: tuple[str, str] | None = None,
        lines: list[str] | None = None,
        header: list[str] | None = None,
        rows: list[list[Any]] | None = None,
        export_fn: Any = None,
        top_widget: QWidget | None = None,
        own_words: bool = False,
    ) -> None:
        """Show a verdict banner + notes + compact table + figure.

        Parameters
        ----------
        fig
            Figure/Axes/None (resolved via :func:`as_figure` unless None).
        verdict
            ``(level, text)`` with level in {"ok", "warn", "error"}.
        lines
            Plain informational lines shown under the verdict.
        header, rows
            A compact table (e.g. diagnostics) shown above the figure.
        own_words
            The table's cells are the window's words (a subset's name), not the data's:
            translated as the window is (``uilang.translate_cells``).
        export_fn
            Callback for a "⬇ Export data (CSV)" button.
        top_widget
            Arbitrary control row inserted at the very top (e.g. selectors).
        """
        self.clear()
        if top_widget is not None:
            self._layout.addWidget(top_widget)
        if export_fn is not None:
            btn = QPushButton("⬇  Export data (CSV)")
            btn.setMaximumWidth(190)
            btn.clicked.connect(export_fn)
            self._layout.addWidget(btn)
        if verdict is not None:
            level, text = verdict
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            _style_verdict(lbl, level)
            self._verdict, self._verdict_level = lbl, level
            self._layout.addWidget(lbl)
        for text in lines or []:
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            self._layout.addWidget(lbl)
        if header and rows:
            table = QTableWidget()
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            fill_sortable(table, list(header), [list(row) for row in rows])
            if own_words:
                from .uilang import translate_cells

                translate_cells(table)
            table.setMaximumHeight(160)
            table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            table.customContextMenuRequested.connect(table_context_menu)
            self._layout.addWidget(table)
        if fig is not None:
            fig = as_figure(fig)
            adaptive_layout(fig)
            apply_figure_theme(fig, theme_manager().dark)
            from .uilang import figure_texts

            figure_texts(fig)  # in the window's language; the English kept to go back
            self.canvas = _canvas_class()(fig)
            self._layout.addWidget(ThemedToolbar(self.canvas, self))
            self._layout.addWidget(self.canvas, 1)
            # The figure's mouse read-out, at its lower left as the Data tab's: what is
            # under the cursor (a component's value at a time, a contribution's σ ...).
            from .figure_interact import readout as _fig_readout

            hover = QLabel("")
            hover.setStyleSheet(hint_style())
            hover.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
            self._layout.addWidget(hover)
            self.canvas.mpl_connect(
                "motion_notify_event", lambda e: hover.setText(_fig_readout(fig, e))
            )
            self.canvas.mpl_connect("figure_leave_event", lambda _e: hover.setText(""))
            self.canvas.draw_idle()  # now if the tab is in view, else when it comes into view
        else:
            self._layout.addStretch(1)


class SmoothProgress(QWidget):
    """A run's progress: a thin rounded bar in the system's accent colour.

    A worker reports in steps -- the base runs settle together near the end of their
    fit, each error estimate starts at its own place -- and a bar set to each value
    jumped from one to the next (the user: it "always jumps"). This one moves to each
    value over a fraction of a second, never goes back within a run, creeps on while a
    step reports nothing (to at most CREEP points past the last value reported, slowing
    as it nears them), fills when the run ends and fades out a moment later. Faded out
    it keeps its place, so nothing beside it in the status bar moves.

    value() is the percentage last reported, as the QProgressBar's was (the Assistant
    reads it)."""

    SCALE = 10  # the bar's units per percent: it moves in tenths of one
    CREEP = 8.0  # how far past the last report the bar may creep, in percent

    def __init__(self, parent=None, height: int = 6):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addStretch(1)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100 * self.SCALE)
        self.bar.setValue(0)  # a new QProgressBar's is -1
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(height)
        lay.addWidget(self.bar)
        lay.addStretch(1)
        self.setMinimumWidth(120)
        self.setMaximumWidth(260)
        self._shade = QGraphicsOpacityEffect(self.bar)
        self._shade.setOpacity(0.0)
        self.bar.setGraphicsEffect(self._shade)
        self._moving = QPropertyAnimation(self.bar, b"value", self)
        self._moving.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fading = QPropertyAnimation(self._shade, b"opacity", self)
        self._fading.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._fading.finished.connect(self._faded)
        self._creep = QTimer(self)
        self._creep.setInterval(100)
        self._creep.timeout.connect(self._step)
        self._hold = QTimer(self)
        self._hold.setSingleShot(True)
        self._hold.timeout.connect(lambda: self._fade_to(0.0, 450))
        self._pct, self._at, self._running = 0, 0.0, False
        self.restyle()

    def value(self) -> int:
        return int(self._pct)

    def running(self) -> bool:
        return self._running

    def start(self):
        """A run begins: the bar fades in, empty -- or empties quickly, where the run
        before it has only just ended."""
        if self._running:
            return
        self._hold.stop()
        self._moving.stop()
        self._pct, self._at, self._running = 0, time.monotonic(), True
        if self._shade.opacity() > 0.0 and self.bar.value() > 0:
            self._animate(self.bar.value(), 0, 200)
        else:
            self.bar.setValue(0)
        self._fade_to(1.0, 180)
        self._creep.start()

    def report(self, pct):
        """The worker's percentage: moved to, not jumped to, and never back within a run."""
        if not self._running:
            self.start()
        pct = max(0, min(100, int(pct)))
        if pct <= self._pct:
            return
        self._pct, self._at = pct, time.monotonic()
        cur, end = self.bar.value(), pct * self.SCALE
        if end > cur:
            self._animate(cur, end, int(min(900, 220 + 0.6 * (end - cur))))

    def finish(self, done: bool = True):
        """The run has ended: the bar fills if it was done, and fades out a moment later.
        Nothing happens where no run was under way."""
        if not self._running:
            return
        self._running = False
        self._creep.stop()
        if done:
            self.report_end()
            self._hold.start(900 + self._moving.duration())
        else:
            self._hold.start(250)

    def report_end(self):
        self._pct = 100
        cur, end = self.bar.value(), 100 * self.SCALE
        if end > cur:
            self._animate(cur, end, int(min(700, 220 + 0.6 * (end - cur))))

    def _animate(self, start: int, end: int, ms: int):
        self._moving.stop()
        self._moving.setStartValue(start)
        self._moving.setEndValue(end)
        self._moving.setDuration(ms)
        self._moving.start()

    def _step(self):
        """A step reporting nothing: on toward a little past its last report, slower as
        it nears it, and no further."""
        if (
            self._moving.state() == QAbstractAnimation.State.Running
            or time.monotonic() - self._at < 0.8
        ):
            return
        cap = int((self._pct + min(self.CREEP, (100 - self._pct) * 0.3)) * self.SCALE)
        cur = self.bar.value()
        if cur < cap:
            self.bar.setValue(cur + max(1, (cap - cur) // 40))

    def _fade_to(self, opacity: float, ms: int):
        self._fading.stop()
        self._fading.setStartValue(self._shade.opacity())
        self._fading.setEndValue(opacity)
        self._fading.setDuration(ms)
        self._fading.start()

    def _faded(self):
        if not self._running and self._shade.opacity() <= 0.01:
            self.bar.setValue(0)
            self._pct = 0

    def restyle(self):
        """The bar's colours from the application's palette: the accent, on a track of the
        text's colour faintly mixed into the window's."""
        pal = QApplication.palette()
        role = QPalette.ColorRole
        accent = (
            pal.color(role.Accent)
            if hasattr(role, "Accent")
            else pal.color(role.Highlight)
        )
        ground, ink = pal.color(role.Window), pal.color(role.WindowText)
        t = 0.16 if ground.lightnessF() < 0.5 else 0.12
        track = QColor.fromRgbF(
            *(
                a + (b - a) * t
                for a, b in zip(
                    (ground.redF(), ground.greenF(), ground.blueF()),
                    (ink.redF(), ink.greenF(), ink.blueF()),
                )
            )
        )
        r = self.bar.height() // 2
        self.bar.setStyleSheet(
            f"QProgressBar {{ border: none; border-radius: {r}px; background: {track.name()}; }}"
            f"QProgressBar::chunk {{ border-radius: {r}px; background: {accent.name()}; }}"
        )

    def changeEvent(self, e):
        # Light or dark: this widget has no style sheet of its own, so it hears of it.
        super().changeEvent(e)
        if e.type() == QEvent.Type.PaletteChange:
            self.restyle()
