"""Shared widgets for the normet Qt GUI.

Design language (borrowed from the cume desktop app): a scrollable parameter
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

from typing import Any

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import QEvent, QRect, QSize, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
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

#: Traffic-light palette for verdict banners (matches the cume app).
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


class ThemedToolbar(NavigationToolbar2QT):
    """The figure toolbar, its Save writing the figure in its light colours, and
    an Export for Publication… that lays it out at a journal's column width."""

    def __init__(self, canvas: Any, parent: QWidget | None = None) -> None:
        super().__init__(canvas, parent)
        self.addSeparator()
        act = self.addAction("Export for Publication…", self.export_for_print)
        act.setToolTip("Lay the figure out at a journal's column width and lettering, and save it.")

    def changeEvent(self, event: Any) -> None:  # noqa: N802
        """Light or dark: matplotlib draws the icons once, in the palette of the moment
        (white on a dark window); drawn again, they stay visible after a switch."""
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            for _text, _tip, image, callback in self.toolitems:
                action = self._actions.get(callback) if callback else None
                if action is not None:
                    action.setIcon(self._icon(image + ".png"))

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
        self._show_placeholder()
        theme_manager().themeChanged.connect(self._retheme)

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
        export_fn
            Callback for a "⬇ Export data (CSV)" button.
        top_widget
            Arbitrary control row inserted at the very top (e.g. selectors).
        """
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

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
            table.setMaximumHeight(160)
            table.resizeColumnsToContents()
            self._layout.addWidget(table)
        if fig is not None:
            fig = as_figure(fig)
            apply_figure_theme(fig, theme_manager().dark)
            self.canvas = FigureCanvasQTAgg(fig)
            self._layout.addWidget(ThemedToolbar(self.canvas, self))
            self._layout.addWidget(self.canvas, 1)
            self.canvas.draw()
        else:
            self._layout.addStretch(1)
