"""The command palette (Ctrl+K, ⌘K on a Mac): every menu command and every
result tab, found by typing a few letters of its name.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHeaderView,
    QLineEdit,
    QMainWindow,
    QMenu,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


@dataclass
class Command:
    text: str  # as its menu item or tab names it
    where: str  # its menu's path, "File › Open Recent", or "Tab"
    run: Callable[[], Any]
    enabled: bool = True
    shortcut: str = ""
    checked: bool | None = None  # a checkable item's state; None for the others


def _clean(text: str) -> str:
    return text.replace("&&", "\0").replace("&", "").replace("\0", "&")


def menu_commands(window: QMainWindow, skip: Iterable[Any] = ()) -> list[Command]:
    """Every item of the window's menus, submenus walked into, separators left out.

    Each menu's ``aboutToShow`` is sent first, as opening it would, so a menu
    that sets its items on or off then is read as it would be shown.
    """
    skip = set(skip)
    out: list[Command] = []
    # Each submenu found from its own action, not through QAction.menu(): PySide6
    # hands that menu to Python to own and deletes it once the reference drops.
    menus = {m.menuAction(): m for m in window.findChildren(QMenu)}

    def walk(menu: QMenu, path: list[str]) -> None:
        menu.aboutToShow.emit()
        for act in menu.actions():
            if act.isSeparator() or not act.isVisible() or act in skip:
                continue
            sub = menus.get(act)
            if sub is not None:
                walk(sub, [*path, _clean(act.text())])
                continue
            out.append(
                Command(
                    _clean(act.text()),
                    " › ".join(path),
                    act.trigger,
                    act.isEnabled(),
                    act.shortcut().toString(QKeySequence.SequenceFormat.NativeText),
                    act.isChecked() if act.isCheckable() else None,
                )
            )

    for act in window.menuBar().actions():
        if act in menus and act.isVisible():
            walk(menus[act], [_clean(act.text())])
    return out


def tab_commands(tabs: QTabWidget) -> list[Command]:
    """Every tab of *tabs*, as a command that shows it."""
    out = []
    for i in range(tabs.count()):
        w = tabs.widget(i)
        out.append(
            Command(
                tabs.tabText(i),
                "Tab",
                lambda w=w: tabs.setCurrentWidget(w),
                tabs.isTabEnabled(i),
            )
        )
    return out


def matches(commands: list[Command], query: str) -> list[Command]:
    """The commands holding every word of *query* (any case) in their name or place.

    Those whose name starts with the query come first, then those with every
    word in the name, then the rest — each group in the menus' own order, the
    ones not available now last.
    """
    words = query.lower().split()
    if not words:
        return list(commands)
    q = " ".join(words)
    ranked = []
    for n, c in enumerate(commands):
        name, where = c.text.lower(), c.where.lower()
        if not all(w in name or w in where for w in words):
            continue
        rank = 0 if name.startswith(q) else 1 if all(w in name for w in words) else 2
        ranked.append((not c.enabled, rank, n, c))
    return [c for *_, c in sorted(ranked, key=lambda r: r[:3])]


class CommandPalette(QDialog):
    """A search field over *commands*, the list under it narrowing as one types.

    Enter (or a click) closes the palette and runs the one picked; Esc closes
    it. Commands not available now are listed greyed and cannot be picked.
    """

    def __init__(self, commands: list[Command], parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("commandPalette")
        self.setStyleSheet("#commandPalette { border: 1px solid palette(mid); }")
        self._commands = commands
        self.chosen: Command | None = None
        self._most: int | None = None  # the tallest the window leaves it
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        self.field = QLineEdit()
        self.field.setPlaceholderText("Find a command or a tab")
        self.field.setClearButtonEnabled(True)
        self.field.textChanged.connect(self._fill)
        self.field.installEventFilter(self)
        self.list = QTreeWidget()
        self.list.setColumnCount(3)
        self.list.setHeaderHidden(True)
        self.list.setRootIsDecorated(False)
        self.list.setUniformRowHeights(True)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        head = self.list.header()
        head.setStretchLastSection(False)
        head.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        head.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.list.itemActivated.connect(lambda _i, _c: self._run())
        self.list.itemClicked.connect(lambda _i, _c: self._run())
        lay.addWidget(self.field)
        lay.addWidget(self.list)
        self._fill("")

    def shown(self) -> list[Command]:
        return [
            self.list.topLevelItem(i).data(0, Qt.ItemDataRole.UserRole)
            for i in range(self.list.topLevelItemCount())
        ]

    def _fill(self, query: str) -> None:
        self.list.clear()
        first = None
        for c in matches(self._commands, query):
            mark = "" if c.checked is None else ("✓ " if c.checked else "    ")
            item = QTreeWidgetItem([mark + c.text, c.where, c.shortcut])
            item.setData(0, Qt.ItemDataRole.UserRole, c)
            item.setTextAlignment(2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            for col in (1, 2):
                item.setForeground(col, self.palette().placeholderText())
            if not c.enabled:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            elif first is None:
                first = item
            self.list.addTopLevelItem(item)
        if first is not None:
            self.list.setCurrentItem(first)
        self._fit()

    def _fit(self) -> None:
        """As tall as the rows listed, up to the most the window leaves it."""
        rows = self.list.topLevelItemCount()
        row = self.list.sizeHintForRow(0) if rows else 0
        frame = 2 * self.list.frameWidth()
        self.list.setFixedHeight(frame + min(rows, 14) * max(row, 1) + (4 if rows else 0))
        self.adjustSize()
        if self._most is not None:
            self.setMaximumHeight(self._most)
        self.resize(self.width(), self.sizeHint().height())

    def _step(self, by: int) -> None:
        n = self.list.topLevelItemCount()
        if not n:
            return
        k = self.list.indexOfTopLevelItem(self.list.currentItem())
        for _ in range(n):
            k = (k + by) % n if k >= 0 else 0
            item = self.list.topLevelItem(k)
            if item.flags() & Qt.ItemFlag.ItemIsEnabled:
                self.list.setCurrentItem(item)
                return

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self.field and event.type() == QEvent.Type.KeyPress:
            key = event.key()  # type: ignore[attr-defined]
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                self._step(1 if key == Qt.Key.Key_Down else -1)
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._run()
                return True
        return super().eventFilter(obj, event)

    def _run(self) -> None:
        item = self.list.currentItem()
        if item is None or not item.flags() & Qt.ItemFlag.ItemIsEnabled:
            return
        self.chosen = item.data(0, Qt.ItemDataRole.UserRole)
        self.accept()

    def place(self, window: QWidget) -> None:
        """Over the top of *window*, centred, as wide as reads well."""
        g = window.geometry()
        w = min(620, max(360, g.width() - 80))
        self.setFixedWidth(w)
        self._most = max(160, g.height() - 120)
        self._fit()
        top = window.mapToGlobal(window.rect().topLeft())
        self.move(top.x() + (g.width() - w) // 2, top.y() + 60)


def open_palette(window: QMainWindow, skip: Iterable[Any] = ()) -> CommandPalette:
    """The palette over *window*, its commands read now: the menus as they
    stand, then the tabs of ``window.tabs`` if it has them.

    The command picked runs once the palette has closed, so a dialog it opens
    is not a child of the popup.
    """
    from shiboken6 import isValid

    commands = menu_commands(window, skip)
    tabs = getattr(window, "tabs", None)
    if isinstance(tabs, QTabWidget):
        commands += tab_commands(tabs)
    dlg = CommandPalette(commands, window)
    dlg.place(window)

    def done(result: int) -> None:
        chosen = dlg.chosen
        if result and chosen is not None:
            QTimer.singleShot(0, lambda: chosen.run() if isValid(window) else None)
        dlg.deleteLater()

    dlg.finished.connect(done)
    dlg.open()
    dlg.field.setFocus()
    return dlg


__all__ = ["Command", "CommandPalette", "matches", "menu_commands", "open_palette", "tab_commands"]
