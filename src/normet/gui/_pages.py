"""A window's left panel as pages, one per step, chosen from a bar above them. The
main window and the SCM Studio both use it.

A page holds its boxes; the first box of each goes untitled, the page's tab naming it.
The panel is as wide as its widest page needs, not a set number: a larger system font
asks for more, and a form squeezed below its least draws its controls as slivers.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTabBar,
    QVBoxLayout,
    QWidget,
)


class StepPanel(QWidget):
    """``pages``: (tab name, its boxes), in the order the work runs in."""

    def __init__(
        self,
        pages: Sequence[tuple[str, Sequence[QGroupBox]]],
        least: int = 300,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.names = tuple(name for name, _ in pages)
        self.stack = QStackedWidget()
        for _name, boxes in pages:
            page = QWidget()
            pl = QVBoxLayout(page)
            pl.setContentsMargins(6, 6, 6, 6)
            for i, box in enumerate(boxes):
                if i == 0:
                    box.setTitle("")  # the page's tab names it; a second box keeps its own
                pl.addWidget(box)
            pl.addStretch(1)
            self.stack.addWidget(page)

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.stack)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        self.bar = QTabBar()
        self.bar.setExpanding(True)
        self.bar.setDocumentMode(True)
        self.bar.setUsesScrollButtons(False)
        self.bar.setStyleSheet("QTabBar::tab { padding: 4px 4px; }")
        for name in self.names:
            self.bar.addTab(name)
        self.bar.currentChanged.connect(self.show_index)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.bar)
        lay.addWidget(self.scroll, 1)

        for i in range(self.stack.count()):
            self.stack.widget(i).setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
            )
        need = max(
            max(self.stack.widget(i).minimumSizeHint().width() for i in range(self.stack.count())),
            self.bar.minimumSizeHint().width(),
        )
        bar_width = self.scroll.verticalScrollBar().sizeHint().width() or 15
        #: The width the panel opens at, and the least it is let go to: a
        #: minimum the splitter starts at, not a fixed width -- fixed, the splitter's
        #: wider slot was left as a gap between the panel and the results.
        self.least_width = max(need + bar_width + 4, least)
        self.setMinimumWidth(self.least_width)
        self.show_index(0)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt's
        return QSize(self.least_width, super().sizeHint().height())

    def show_index(self, i: int) -> None:
        """Page ``i`` in view: the others take no room, so the scroll is the page's own
        length, not the longest page's."""
        for k in range(self.stack.count()):
            policy = QSizePolicy.Policy.Preferred if k == i else QSizePolicy.Policy.Ignored
            self.stack.widget(k).setSizePolicy(policy, policy)
        self.stack.setCurrentIndex(i)
        self.stack.adjustSize()
        self.stack.updateGeometry()
        if self.bar.currentIndex() != i:
            self.bar.blockSignals(True)
            self.bar.setCurrentIndex(i)
            self.bar.blockSignals(False)
        self.scroll.verticalScrollBar().setValue(0)

    def show_page(self, name: str) -> None:
        """The page named ``name`` in view."""
        self.show_index(self.names.index(name))
