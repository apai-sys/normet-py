"""Notice bars: a line above the results for what wants the user's attention, with the
step that answers it on a button and a × to put it away -- where a dialog would stop the
window until it was answered, or a status-bar message would be gone with the next one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

#: A notice's kind and the colour of its bar: the verdict banners' own.
KINDS = {"info": "#0a84ff", "warn": "#ff9500", "fail": "#ff3b30"}

Action = tuple[str, Callable[[], object]]


class NoticeBar(QFrame):
    """One notice: its text, its buttons, and × to close it. A button runs its step and
    closes the notice, which has then been answered."""

    def __init__(
        self,
        key: str,
        text: str,
        actions: Sequence[Action] = (),
        kind: str = "info",
        on_close: Callable[[str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.key, self.kind = key, kind
        self._on_close = on_close
        self.setObjectName("notice")
        c = QColor(KINDS.get(kind, KINDS["info"]))
        # A tint of its colour over the window's own background, light or dark alike.
        self.setStyleSheet(
            f"#notice {{ background: rgba({c.red()}, {c.green()}, {c.blue()}, 34);"
            f" border-left: 3px solid {c.name()}; }}"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 4, 4, 4)
        lay.setSpacing(8)
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.label, 1)
        self.buttons: list[QPushButton] = []
        for name, run in actions:
            b = QPushButton(name)
            b.clicked.connect(lambda _=False, run=run: (self.close_notice(), run()))
            lay.addWidget(b)
            self.buttons.append(b)
        self.close_btn = QPushButton("×")
        self.close_btn.setFlat(True)
        self.close_btn.setFixedWidth(28)
        self.close_btn.setToolTip("Close this notice")
        self.close_btn.clicked.connect(self.close_notice)
        lay.addWidget(self.close_btn)

    def close_notice(self) -> None:
        self.hide()
        if self._on_close is not None:
            self._on_close(self.key)


class NoticeStack(QWidget):
    """The window's notices, newest first, one per key: a notice under a key already
    shown takes its place. One closed is not shown again while its text is the same --
    the same step asked for again says nothing new -- but is when the text changes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bars: dict[str, NoticeBar] = {}
        self._closed: dict[str, str] = {}  # key -> the text last closed under it
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(4, 4, 4, 0)
        self._lay.setSpacing(4)
        self.hide()

    def notify(
        self, key: str, text: str, actions: Sequence[Action] = (), kind: str = "info"
    ) -> NoticeBar | None:
        if self._closed.get(key) == text:
            self.clear(key)
            return None
        self.clear(key)
        bar = NoticeBar(key, text, actions, kind, on_close=self._close)
        self._bars[key] = bar
        self._lay.insertWidget(0, bar)
        self.show()
        return bar

    def _close(self, key: str) -> None:
        bar = self._bars.get(key)
        if bar is not None:
            self._closed[key] = bar.label.text()
        self.clear(key)

    def clear(self, key: str) -> None:
        bar = self._bars.pop(key, None)
        if bar is not None:
            bar.hide()
            self._lay.removeWidget(bar)
            bar.deleteLater()
        self.setVisible(bool(self._bars))

    def get(self, key: str) -> NoticeBar | None:
        return self._bars.get(key)

    def keys(self) -> list[str]:
        return list(self._bars)
