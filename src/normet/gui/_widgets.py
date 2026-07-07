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
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QListWidget,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

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


class CanvasTab(QWidget):
    """A result tab: verdict banner + notes + table + matplotlib figure.

    All content is rebuilt on each ``show_*`` call; the previous figure is
    closed so long GUI sessions do not accumulate matplotlib state.
    """

    def __init__(self, placeholder: str = "Run an analysis to see results here.") -> None:
        super().__init__()
        self._placeholder = placeholder
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = None
        self._show_placeholder()

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
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )

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
                f"color:{VERDICT_COLOR.get(level, '#1c1c1e')};font-weight:600;padding:4px;"
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
            self.canvas = FigureCanvasQTAgg(fig)
            self._layout.addWidget(NavigationToolbar2QT(self.canvas, self))
            self._layout.addWidget(self.canvas, 1)
            self.canvas.draw()
        else:
            self._layout.addStretch(1)
