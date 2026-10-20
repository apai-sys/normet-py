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
from collections.abc import Callable
from typing import Any

import pandas as pd
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import QEvent, QRect, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QIcon, QImage, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QListWidget,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)

from .theme import apply_figure_theme, light_figure, theme_manager

#: Traffic-light palette for verdict banners.
VERDICT_COLOR = {"ok": "#34c759", "warn": "#ff9500", "error": "#ff3b30"}

#: Muted colour for secondary hints / file labels.
HINT_STYLE = "color:#8e8e93;"
HINT_STYLE_SMALL = "color:#8e8e93;font-size:11px;"

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


def hint_label(text: str, small: bool = True) -> QLabel:
    """A muted, word-wrapped helper label."""
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(HINT_STYLE_SMALL if small else HINT_STYLE)
    return lbl


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

    def save_figure(self, *args: Any) -> Any:
        with light_figure(self.canvas.figure):
            return super().save_figure(*args)


class FlowLayout(QLayout):
    """A row of widgets that wraps onto further rows when the width runs out."""

    def __init__(self, parent: QWidget | None = None, spacing: int = 8) -> None:
        super().__init__(parent)
        self._items: list[Any] = []
        self._gap = spacing

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
        for item in self._items:
            hint = item.sizeHint()
            if x + hint.width() > r.right() + 1 and row_h > 0:
                x, y, row_h = r.x(), y + row_h + self._gap, 0
            if move:
                item.setGeometry(QRect(x, y, hint.width(), hint.height()))
            x += hint.width() + self._gap
            row_h = max(row_h, hint.height())
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
        #: The time-series result being shown (see :meth:`show_timeseries`).
        self._series: tuple[Callable[[pd.DataFrame], Any], pd.DataFrame, dict[str, Any]] | None
        self._series = None
        #: The resolution chosen on this tab (a RESOLUTIONS label), or None
        #: for the default each result asks for.
        self._resolution: str | None = None
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
        if self.canvas is not None:
            apply_figure_theme(self.canvas.figure, dark)
            self.canvas.draw_idle()

    # ------------------------------------------------------------------ util
    def _show_placeholder(self) -> None:
        lbl = QLabel(self._placeholder)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(HINT_STYLE)
        self._layout.addWidget(lbl)

    def clear(self) -> None:
        import matplotlib.pyplot as plt

        while self._layout.count():
            w = self._layout.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        if self.canvas is not None:
            plt.close(self.canvas.figure)
        self.canvas = None
        self._series = None

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
        row = QHBoxLayout(bar)
        row.setContentsMargins(4, 2, 4, 0)
        row.addWidget(QLabel("Resolution"))
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
        row.addWidget(combo)
        row.addStretch(1)

        def chosen(name: str) -> None:
            self._resolution = name
            # Rebuilt after the signal returns: the combo is part of what goes.
            QTimer.singleShot(0, self._draw_series)

        combo.currentTextChanged.connect(chosen)

        try:
            fig = render(resample_frame(data, usable[label]))
        except Exception:  # noqa: BLE001 - fall back to the series as recorded
            if usable[label] is None:
                raise
            import logging

            logging.getLogger(__name__).exception("Figure at %s failed", label)
            self._resolution = "As recorded"
            fig = render(data)
            combo.blockSignals(True)
            combo.setCurrentText("As recorded")
            combo.blockSignals(False)
        series = self._series
        self.show_result(fig, top_widget=bar, **kwargs)
        self._series = series

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
            lbl.setStyleSheet(
                f"color:{VERDICT_COLOR.get(level, 'palette(text)')};font-weight:600;padding:4px;"
            )
            self._layout.addWidget(lbl)
        for text in lines or []:
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            self._layout.addWidget(lbl)
        if header and rows:
            table = QTableWidget(len(rows), len(header))
            table.setHorizontalHeaderLabels(header)
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            for r, row in enumerate(rows):
                for c, val in enumerate(row):
                    table.setItem(r, c, QTableWidgetItem(str(val)))
            if own_words:
                from .uilang import translate_cells

                translate_cells(table)
            table.setMaximumHeight(160)
            table.resizeColumnsToContents()
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
            self.canvas.draw_idle()  # now if the tab is in view, else when it comes into view
        else:
            self._layout.addStretch(1)
