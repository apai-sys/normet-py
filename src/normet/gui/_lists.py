"""Tick lists: read, fill and clear a ``QListWidget`` of checkable items.

The feature list, the resample list, the estimator list and the PDP list all
behave the same way, and :func:`parse_groups` turns the Decompose box's
``"local: t2m, blh; wind: u10, v10"`` into the mapping the library wants.

They live here rather than in :mod:`.main_window` so the per-step modules can
use them once the window is split. The names keep their leading underscore:
they are private to this package, and the window and the tests import them by
these names.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidget, QListWidgetItem


def _checked_items(widget: QListWidget) -> list[str]:
    return [
        widget.item(i).text()
        for i in range(widget.count())
        if widget.item(i).checkState() == Qt.CheckState.Checked
    ]


def _fill_checklist(widget: QListWidget, names: list[str], checked: set[str]) -> None:
    widget.clear()
    for name in names:
        item = QListWidgetItem(name)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if name in checked else Qt.CheckState.Unchecked)
        widget.addItem(item)


def _set_all(widget: QListWidget, checked: bool) -> None:
    state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
    for i in range(widget.count()):
        widget.item(i).setCheckState(state)


#: The tooltip of an All button that ticks and unticks by turns.
ALL_TIP = "Tick every one; once all are ticked, untick every one."


def _toggle_all(widget: QListWidget) -> None:
    """Every item ticked -- or, where every one is ticked already, every one unticked: one
    All button does what an All and a None did."""
    every = widget.count() > 0 and all(
        widget.item(i).checkState() == Qt.CheckState.Checked for i in range(widget.count())
    )
    _set_all(widget, not every)


def parse_groups(text: str, variables: list[str]) -> dict[str, list[str]] | None:
    """``"local: t2m, blh; wind: u10, v10"`` as ``{"local": [...], "wind": [...]}``;
    None for an empty box. Raises ValueError, in words for the user, on a group
    without a name or a variable not trained on."""
    text = text.strip()
    if not text:
        return None
    groups: dict[str, list[str]] = {}
    for part in filter(None, (p.strip() for p in text.split(";"))):
        name, colon, members = part.partition(":")
        name = name.strip()
        if not colon or not name:
            raise ValueError(f"write each group as name: var, var — not {part!r}")
        names = [m.strip() for m in members.split(",") if m.strip()]
        unknown = [m for m in names if m not in variables]
        if unknown:
            raise ValueError(
                f"{', '.join(unknown)} not among the variables trained on ({', '.join(variables)})"
            )
        groups[name] = names
    return groups
