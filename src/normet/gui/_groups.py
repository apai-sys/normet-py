"""The meteorological decomposition's groups, set in a table rather than typed.

Each meteorological variable the model was trained on is a row; its group is chosen
from a list -- the four the manual uses (wind; temperature and humidity; boundary layer
and pressure; sky and rain) or any name typed in -- or left empty, the variable then a
contribution of its own. *Suggest groups* puts each recognised variable in its group;
*Each alone* clears them. A group's rows are tinted in one colour, and the line under the
table says what the grouping makes.

The text form the window has always kept -- ``"Wind: ws, wd; Sky and rain: tcc, tp"``
(:func:`._lists.parse_groups`) -- is what :meth:`GroupEditor.text` gives and
:meth:`GroupEditor.setText` takes: a saved config, a session script and the tests read
and write it as before.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from normet import style as S

from ._lists import parse_groups
from ._widgets import hint_label

#: The groups offered, and the variables (lower case) each takes on Suggest groups.
PRESETS: dict[str, tuple[str, ...]] = {
    "Wind": ("ws", "wd", "u10", "v10", "u", "v", "wind", "ws_era5", "wd_era5", "wspd", "wdir"),
    "Temperature and humidity": (
        "t2m",
        "temp",
        "d2m",
        "rh",
        "rh2m",
        "temp_era5",
        "temperature",
        "humidity",
        "q",
        "dewpoint",
    ),
    "Boundary layer and pressure": ("blh", "pblh", "sp", "msl", "pressure", "atmos_pres", "slp"),
    "Sky and rain": ("tcc", "ssrd", "tp", "cloud", "radiation", "rain", "precip", "precipitation"),
}
#: What the Group column shows for a variable left alone.
ALONE = ""


def _from_start(box: QComboBox) -> None:
    """A group's name read from its start where the cell is too narrow for all of it
    (one typed in), the whole of it on the pointer."""
    edit = box.lineEdit()
    if edit is not None and not edit.hasFocus():
        edit.setCursorPosition(0)
    box.setToolTip(box.currentText())


def suggested_group(name: str) -> str:
    """The preset group a variable belongs to by its name, or ALONE."""
    key = name.lower()
    return next((g for g, members in PRESETS.items() if key in members), ALONE)


class GroupEditor(QWidget):
    """The variables of a meteorological decomposition, each with its group."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._variables: list[str] = []
        #: Each variable's group, by name -- kept for variables not (yet) in the table, so
        #: a config read before training applies once the model's variables are known.
        self._groups: dict[str, str] = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Variable", "Group"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        head = self.table.horizontalHeader()
        head.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(120)
        self.table.setMaximumHeight(220)
        self.table.setToolTip(
            "Each meteorological variable the model was trained on, and the group its "
            "contribution is counted in: pick one, type a name of your own, or leave it "
            "empty for a contribution of its own."
        )
        lay.addWidget(self.table)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.suggest_btn = QPushButton("Suggest groups")
        self.suggest_btn.setToolTip(
            "Put each recognised variable in its group: wind; temperature and humidity; "
            "boundary layer and pressure; sky and rain."
        )
        self.suggest_btn.clicked.connect(self.suggest)
        self.clear_btn = QPushButton("Each alone")
        self.clear_btn.setToolTip("No groups: each variable a contribution of its own.")
        self.clear_btn.clicked.connect(self.clear)
        row.addWidget(self.suggest_btn)
        row.addWidget(self.clear_btn)
        lay.addLayout(row)
        self.summary = hint_label("Train a model to group its meteorological variables.")
        lay.addWidget(self.summary)

    # -- the variables and their groups ---------------------------------------------
    def set_variables(self, names: list[str]) -> None:
        """The model's meteorological variables, each in the group it had."""
        self._variables = list(names)
        self.table.setRowCount(len(names))
        for r, name in enumerate(names):
            item = QTableWidgetItem(name)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(r, 0, item)
            box = QComboBox()
            box.setEditable(True)
            box.addItem(ALONE)
            box.addItems(list(PRESETS))
            box.setCurrentText(self._groups.get(name, ALONE))
            _from_start(box)
            box.currentTextChanged.connect(lambda text, name=name: self._chosen(name, text))
            box.currentTextChanged.connect(lambda _text, box=box: _from_start(box))
            self.table.setCellWidget(r, 1, box)
        self._paint()
        self.updateGeometry()  # its width: the variables' names and the groups'

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt's
        """Wide enough for the longest variable name beside the longest group's: a
        narrower table cut the group's name short, showing its end (``oundary layer and
        pressure``)."""
        hint = super().minimumSizeHint()
        t = self.table
        names = max(t.sizeHintForColumn(0), t.horizontalHeader().sectionSizeHint(0))
        probe = QComboBox()
        probe.setEditable(True)
        probe.addItems([*PRESETS, *self._groups.values()])
        probe.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        groups = probe.sizeHint().width()
        bar = t.verticalScrollBar().sizeHint().width()
        return QSize(max(hint.width(), names + groups + bar + 2 * t.frameWidth()), hint.height())

    def _chosen(self, name: str, group: str) -> None:
        group = group.strip()
        if group:
            self._groups[name] = group
        else:
            self._groups.pop(name, None)
        self._paint()

    def _box(self, row: int) -> QComboBox:
        box = self.table.cellWidget(row, 1)
        assert isinstance(box, QComboBox)
        return box

    def suggest(self) -> None:
        for r, name in enumerate(self._variables):
            self._box(r).setCurrentText(suggested_group(name))

    def clear(self) -> None:
        self._groups = {}
        for r in range(len(self._variables)):
            self._box(r).setCurrentText(ALONE)
        self._paint()

    def _in_groups(self) -> dict[str, list[str]]:
        """The groups as set, variables in the table's order; those left alone out."""
        out: dict[str, list[str]] = {}
        for name in self._variables + [v for v in self._groups if v not in self._variables]:
            group = self._groups.get(name)
            if group:
                out.setdefault(group, []).append(name)
        return out

    def _paint(self) -> None:
        """Each group's rows in a colour of its own, and the line under the table."""
        groups = list(self._in_groups())
        for r, name in enumerate(self._variables):
            group = self._groups.get(name)
            colour = QColor(S.PALETTE[groups.index(group) % len(S.PALETTE)]) if group else None
            item = self.table.item(r, 0)
            if item is not None:
                if colour is not None:
                    colour.setAlpha(60)
                    item.setBackground(colour)
                else:
                    item.setBackground(QColor(0, 0, 0, 0))
        if not self._variables:
            self.summary.setText("Train a model to group its meteorological variables.")
        elif not groups:
            self.summary.setText("No groups: each variable a contribution of its own.")
        else:
            alone = [v for v in self._variables if v not in self._groups]
            n = len(groups) + len(alone)
            self.summary.setText(
                f"{n} contributions: {len(groups)} group(s)"
                + (f" and {len(alone)} variable(s) alone" if alone else "")
                + "."
            )

    # -- the text form ----------------------------------------------------------------
    def text(self) -> str:
        """``"group: var, var; group: var"``, the form :func:`parse_groups` reads; empty
        where nothing is grouped."""
        return "; ".join(f"{g}: {', '.join(v)}" for g, v in self._in_groups().items())

    def setText(self, text: str) -> None:  # noqa: N802 - the QLineEdit it replaces
        """Groups from their text form (what :meth:`text` gives). Names the table does not
        list are kept, for :meth:`groups` to say so."""
        self._groups = {}
        for part in filter(None, (p.strip() for p in (text or "").split(";"))):
            name, _, members = part.partition(":")
            for m in members.split(","):
                if m.strip() and name.strip():
                    self._groups[m.strip()] = name.strip()
        self.set_variables(self._variables)

    def groups(self, variables: list[str]) -> dict[str, list[str]] | None:
        """The groups for a run over ``variables`` -- each variable left alone a group of
        its own, as every variable trained on must be in one -- or None where nothing is
        grouped. Raises ValueError, in words, for a variable not among them."""
        parsed = parse_groups(self.text(), variables)
        if not parsed:
            return None
        grouped = {v for members in parsed.values() for v in members}
        for v in variables:
            if v not in grouped:
                parsed[v] = [v]
        return parsed
