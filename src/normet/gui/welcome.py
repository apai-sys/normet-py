"""What the Data tab shows before a table is open: the ways in.

A title and the main buttons (open a CSV, load the example), and under them
cards adapted from the cume desktop app's welcome: the examples, the tables
opened lately with their folder, size and date, and where else data comes
from (the UK data studio, HYSPLIT trajectories).
"""

from __future__ import annotations

import datetime as _dt
import os
from collections.abc import Callable, Sequence

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ._widgets import FlowLayout

#: The most recent tables a card lists; File → Open Recent holds them all.
RECENT_SHOWN = 5
CARD_WIDTH = 270


def _size(n: float) -> str:
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


def file_line(path: str) -> str:
    """A table's folder, size and date last changed, in a line."""
    try:
        st = os.stat(path)
    except OSError:
        return os.path.dirname(path)
    day = _dt.date.fromtimestamp(st.st_mtime).isoformat()
    return f"{os.path.basename(os.path.dirname(path))} · {_size(st.st_size)} · {day}"


class _Paragraph(QLabel):
    """A wrapped label that asks for :attr:`WIDEST` px and can make do with
    :attr:`NARROWEST`: laid out centred, it gets its asked width when there is
    room and is squeezed (and wraps onto more lines) when there is not."""

    WIDEST, NARROWEST = 520, 240

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self.WIDEST, self.heightForWidth(self.WIDEST))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(self.NARROWEST, super().minimumSizeHint().height())


class _CardsHost(QWidget):
    """Holds the cards' flow: asks for one row of all of them, makes do
    with one card's width (the cards then wrap onto more rows)."""

    wanted = 0

    def sizeHint(self) -> QSize:  # noqa: N802
        w = max(self.wanted, self.minimumSizeHint().width())
        return QSize(w, self.heightForWidth(w))


class Card(QFrame):
    """A titled box of rows: each a link-like button and, under it, a line in grey."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("welcomeCard")
        self.setStyleSheet(
            "#welcomeCard { border: 1px solid palette(mid); border-radius: 8px;"
            " background: palette(base); }"
        )
        self.setFixedWidth(CARD_WIDTH)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(14, 12, 14, 12)
        self._lay.setSpacing(2)
        head = QLabel(title)
        head.setStyleSheet("font-weight: 600; font-size: 14px;")
        self._lay.addWidget(head)
        self._lay.addSpacing(4)
        self.rows: list[tuple[QPushButton, QLabel]] = []
        self._lay.addStretch(1)

    def add(
        self,
        text: str,
        line: str,
        run: Callable[[], object],
        tip: str = "",
        enabled: bool = True,
    ) -> QPushButton:
        b = QPushButton(text)
        b.setFlat(True)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setStyleSheet(
            "QPushButton { text-align: left; padding: 2px 0; color: palette(link); }"
            " QPushButton:disabled { color: palette(placeholder-text); }"
        )
        b.setToolTip(tip or text)
        b.setEnabled(enabled)
        b.clicked.connect(lambda _=False: run())
        sub = QLabel()
        sub.setStyleSheet("color: palette(placeholder-text); font-size: 11px;")
        sub.setToolTip(line)
        # One line each, cut short with … where too long, the whole in the tooltip.
        inner = CARD_WIDTH - 28
        b.setText(b.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, inner))
        sub.ensurePolished()
        sub.setText(sub.fontMetrics().elidedText(line, Qt.TextElideMode.ElideRight, inner))
        n = self._lay.count() - 1  # before the stretch
        self._lay.insertWidget(n, b)
        self._lay.insertWidget(n + 1, sub)
        self._lay.insertSpacing(n + 2, 4)
        self.rows.append((b, sub))
        return b


class Welcome(QWidget):
    """The welcome: a title, the main buttons and the cards.

    *actions* maps a name to what it does: ``open_csv``, ``open_path(path)``,
    ``load_example``, ``scm_example``, ``data_studio``, ``transport_studio`` and,
    optionally, ``assistant``.
    """

    def __init__(self, actions: dict[str, Callable[..., object]], parent: QWidget | None = None):
        super().__init__(parent)
        self._actions = actions
        outer = QVBoxLayout(self)
        outer.addStretch(1)
        title = QLabel("Start with a time series")
        title.setStyleSheet("font-size:18px; font-weight:600;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(title)
        said = _Paragraph(
            "Open a CSV with a 'date' column, a pollutant and its meteorology — or drop one "
            "onto the window."
        )
        said.setWordWrap(True)
        said.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 520 px wide where there is room, narrower (and taller) where there is
        # not: a fixed width held the whole window at least ~1050 px wide.
        self._said = said
        said.setMinimumHeight(said.heightForWidth(_Paragraph.WIDEST))
        outer.addWidget(said, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addSpacing(8)
        row = QHBoxLayout()
        row.addStretch(1)
        self.open_btn = QPushButton("Open CSV…")
        self.open_btn.setStyleSheet("font-weight:600; padding:8px 16px;")
        self.open_btn.clicked.connect(lambda: actions["open_csv"]())
        self.example_btn = QPushButton("Load example")
        self.example_btn.setStyleSheet("padding:8px 16px;")
        self.example_btn.clicked.connect(lambda: actions["load_example"]())
        row.addWidget(self.open_btn)
        row.addWidget(self.example_btn)
        if "assistant" in actions:
            self.ask_btn = QPushButton("Ask the Assistant")
            self.ask_btn.setStyleSheet("padding:8px 16px;")
            self.ask_btn.clicked.connect(lambda: actions["assistant"]())
            row.addWidget(self.ask_btn)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addSpacing(20)
        self._cards_host = _CardsHost()
        self._row = FlowLayout(self._cards_host, spacing=12)
        self._row.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._cards_host, 0, Qt.AlignmentFlag.AlignHCenter)
        outer.addStretch(2)
        self.cards: dict[str, Card] = {}

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Room for the paragraph's lines at the width it now gets."""
        m = self.layout().contentsMargins()
        w = max(
            _Paragraph.NARROWEST,
            min(_Paragraph.WIDEST, event.size().width() - m.left() - m.right()),
        )
        self._said.setMinimumHeight(self._said.heightForWidth(w))
        super().resizeEvent(event)

    def refresh(self, recent: Sequence[str]) -> None:
        """The cards again, from what there is now: *recent* is the tables'
        paths, newest first; a card with nothing to list is left out."""
        while self._row.count():
            w = self._row.takeAt(0).widget()
            if w is not None:
                w.hide()  # now: deleteLater leaves it drawn until the loop turns
                w.deleteLater()
        a = self._actions
        self.cards = {}
        card = Card("Examples")
        card.add(
            "De-weathering",
            "A year of hourly PM2.5 with ERA5-style meteorology",
            a["load_example"],
            tip="Synthetic single-site data: train, normalise, decompose, rolling, PDP.",
        )
        card.add(
            "Synthetic control",
            "A weekly panel of units, one treated",
            a["scm_example"],
            tip="Opens the SCM Studio with its example panel loaded.",
        )
        self.cards["examples"] = card
        if recent:
            card = Card("Recent tables")
            for p in recent[:RECENT_SHOWN]:
                card.add(
                    os.path.basename(p),
                    file_line(p),
                    lambda p=p: a["open_path"](p),
                    tip=p,
                    enabled=os.path.exists(p),
                )
            self.cards["recent"] = card
        card = Card("Get data")
        card.add(
            "UK air quality + meteorology",
            "AURN, AQE, SAQN, WAQN, NI, LMAM with ERA5 (no API key)",
            a["data_studio"],
        )
        card.add(
            "Transport predictors",
            "HYSPLIT back-trajectories joined onto your table",
            a["transport_studio"],
        )
        self.cards["data"] = card
        for c in self.cards.values():
            self._row.addWidget(c)
        self._row.invalidate()
        width = len(self.cards) * CARD_WIDTH + (len(self.cards) - 1) * 12
        # Asked for, not required: a minimum width here held the window wide.
        self._cards_host.wanted = min(width, 3 * CARD_WIDTH + 24)
        self._cards_host.updateGeometry()
