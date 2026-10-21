"""The Data tab: the table at a glance, its cycles, its winds and its statistics -- and
the rows left out of every step.

- Overview: every numeric column as a strip, each scaled to its own range (missing
  values grey), over the whole record; below it, one variable as measured. A span
  dragged across either is picked -- Shift (Ctrl, Cmd) adds another, as cume's samples
  are picked, and Escape clears them -- its rows listed beneath the plot, and *Exclude
  picked* leaves its rows out of every
  step -- training, normalisation, decomposition, rolling. The rows left out are greyed
  where they are, and written to the Exclude box as the lines ``exclusion_lines``
  writes: the box is the record the window, its command log and a saved session read.
- Cycles: a variable's diurnal, weekly and seasonal cycles (:func:`normet.cycle_profiles`).
- Wind: the wind rose, the pollution rose and the polar plot of a variable.
- Statistics: each column's count, missing share and spread.

The rows of the spans picked on the overview are listed under it, every column, the ones
left out marked: what a span holds is read where it is picked.

Everything but the overview reads the rows kept: what the steps see.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import QModelIndex, Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from normet import style as S

from ._widgets import CanvasTab, NoWheelComboBox
from .dataframe_model import DataFrameModel

PICK_HINT = (
    "Drag across the plot, or click it, to list the rows of a span (Shift adds another); "
    "select rows in the list and right-click to leave them out."
)

#: A press and release closer than this, in pixels, is a click rather than a drag; a
#: click on the series picks the point within _NEAR_PX of it -- as in cume.
_CLICK_PX = 4
_NEAR_PX = 10


def _adding(event: Any = None) -> bool:
    """Whether Shift, Ctrl or Cmd is held: a drag with one adds its span to the pick. A
    mouse event names them in its ``modifiers``; the keyboard's state otherwise."""
    held = {str(k).lower() for k in (getattr(event, "modifiers", None) or ())}
    held |= {k for k in str(getattr(event, "key", None) or "").lower().split("+") if k}
    if held & {"shift", "ctrl", "control", "cmd", "super", "meta"}:
        return True
    from PySide6.QtWidgets import QApplication

    keys = QApplication.keyboardModifiers()
    return any(
        keys & m
        for m in (
            Qt.KeyboardModifier.ShiftModifier,
            Qt.KeyboardModifier.ControlModifier,
            Qt.KeyboardModifier.MetaModifier,
        )
    )


#: The rows of the spans picked listed at most (a year of hours and more).
PICKED_ROWS = 20_000

#: A line of the Exclude box: one time (a date alone is its whole day) or "a .. b".
_SPAN = re.compile(r"^\s*(?P<a>[^.]+?)\s*(?:\.\.\s*(?P<b>.+?))?\s*$")
_EXCLUDED = "#8e8e93"
_PICKED = "#ff3b30"
_MISSING = "#d1d1d6"


# -- the rows left out ------------------------------------------------------------------
def parse_exclusions(text: str | Sequence[str]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """The spans the Exclude box names, a line each: ``2020-03-23 .. 2020-04-05`` (both
    ends kept out; a date alone as an end is its whole day), or one time. Blank lines
    and ``#`` comments are passed over; a line that is not a time is a ValueError
    naming it."""
    lines = text.splitlines() if isinstance(text, str) else list(text)
    out = []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = _SPAN.match(line)
        a = m.group("a") if m else line
        b = (m.group("b") if m else None) or a
        try:
            start, end = pd.Timestamp(a.strip()), pd.Timestamp(b.strip())
        except (ValueError, TypeError):
            raise ValueError(f"not a time, nor 'start .. end': {line!r}") from None
        if _date_only(b):
            end = end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        if end < start:
            start, end = end, start
        out.append((start.tz_localize(None), end.tz_localize(None)))
    return _merged(out)


def _date_only(text: str) -> bool:
    return bool(re.fullmatch(r"\s*\d{4}-\d{2}-\d{2}\s*", text))


def _merged(
    spans: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _stamp(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%d") if t == t.normalize() else t.strftime("%Y-%m-%d %H:%M")


def exclusion_lines(spans: Sequence[tuple[pd.Timestamp, pd.Timestamp]]) -> list[str]:
    """The Exclude box's lines for *spans*, merged where they meet: whole days written
    as dates."""
    out = []
    for a, b in _merged(list(spans)):
        # Whole days: from midnight to the day's last minute (a pick is to the minute).
        after = (b + pd.Timedelta(minutes=1)).floor("min")
        if a == a.normalize() and after == after.normalize():
            last = (after - pd.Timedelta(days=1)).normalize()
            out.append(_stamp(a) if last == a else f"{_stamp(a)} .. {_stamp(last)}")
        else:
            # A time at midnight written with it: a date alone as an end is its whole day.
            whole = a.strftime("%Y-%m-%d %H:%M")
            end = b.strftime("%Y-%m-%d %H:%M") if b == b.normalize() else _stamp(b)
            out.append(whole if a == b else f"{_stamp(a)} .. {end}")
    return out


def excluded_mask(
    dates: pd.Series, spans: Sequence[tuple[pd.Timestamp, pd.Timestamp]]
) -> np.ndarray:
    """Which rows of *dates* the spans leave out."""
    d = (
        pd.to_datetime(dates).dt.tz_localize(None)
        if hasattr(dates, "dt")
        else pd.to_datetime(dates)
    )
    out = np.zeros(len(d), dtype=bool)
    for a, b in spans:
        out |= ((d >= a) & (d <= b)).to_numpy()
    return out


def _subtract(
    spans: list[tuple[pd.Timestamp, pd.Timestamp]], cut: tuple[pd.Timestamp, pd.Timestamp]
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """*spans* with *cut* taken out of them (a restore): a pick is to the minute, so what
    is left starts at the minute after it."""
    out = []
    for a, b in spans:
        if b < cut[0] or a > cut[1]:
            out.append((a, b))
            continue
        if a < cut[0]:
            out.append((a, cut[0] - pd.Timedelta(seconds=1)))
        after = cut[1].floor("min") + pd.Timedelta(minutes=1)
        if b >= after:
            out.append((after, b))
    return out


def _period_name(span: tuple[pd.Timestamp, pd.Timestamp], rule: str | None) -> str:
    """The period a point averaged to *rule* stands for, as a reader names it: its hour,
    day, month or year; a week by its first and last day."""
    a, b = span
    fmt = {"h": "%Y-%m-%d %H:00", "D": "%Y-%m-%d", "MS": "%Y-%m", "YS": "%Y"}.get(rule or "")
    if fmt:
        return a.strftime(fmt)
    return f"{a:%Y-%m-%d} – {b:%Y-%m-%d}"


def _runs(dates: pd.Series, positions: np.ndarray) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """The spans of times that name the rows at *positions* of a table (*dates* its
    date column, in time order): a span for each run of rows next to each other."""
    pos = np.unique(np.asarray(positions, dtype=int))
    if not len(pos):
        return []
    breaks = np.flatnonzero(np.diff(pos) > 1)
    starts, ends = np.r_[pos[0], pos[breaks + 1]], np.r_[pos[breaks], pos[-1]]
    d = pd.to_datetime(dates)
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    return [(d.iat[a], d.iat[b]) for a, b in zip(starts, ends, strict=True)]


class _RowsModel(DataFrameModel):
    """The rows picked, those left out in grey."""

    def __init__(self, max_rows: int) -> None:
        super().__init__(max_rows=max_rows)
        self.left_out = np.zeros(0, dtype=bool)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role == Qt.ItemDataRole.ForegroundRole:
            if index.isValid() and index.row() < len(self.left_out) and self.left_out[index.row()]:
                return QColor(_EXCLUDED)
            return None
        return super().data(index, role)


def _on_rows(
    spans: list[tuple[pd.Timestamp, pd.Timestamp]], dates: pd.Series
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """*spans* (pieces a restore left) started at their first row, and one that holds a
    single row written as it -- "07:00", not "06:01 .. 07:00"; a piece with no row goes."""
    d = pd.to_datetime(dates)
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    d = np.sort(d.dropna().to_numpy())
    out = []
    for a, b in spans:
        inside = d[(d >= np.datetime64(a)) & (d <= np.datetime64(b))]
        if not len(inside):
            continue
        first, last = pd.Timestamp(inside[0]), pd.Timestamp(inside[-1])
        out.append((first, first) if first == last else (first, b))
    return out


# -- the overview's figure -----------------------------------------------------------------
def overview_figure(
    d: pd.DataFrame,
    variable: str,
    spans: Sequence[tuple[pd.Timestamp, pd.Timestamp]] = (),
    picked: Sequence[tuple[pd.Timestamp, pd.Timestamp]] = (),
) -> Any:
    """Every numeric column of *d* (indexed by time) as a strip scaled to its own range
    -- missing values grey -- above *variable* as measured; the spans left out greyed,
    the spans picked in red. The series' axes are the figure's first."""
    import matplotlib.pyplot as plt
    from matplotlib import dates as mdates

    cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
    n = max(1, len(cols))
    fig = plt.figure(figsize=(8, 2.0 + min(3.6, 0.17 * n)), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[min(3.6, 0.17 * n) + 0.3, 1.4])
    ax_s = fig.add_subplot(gs[1])  # first: the variable's line is the figure's first axes
    ax_h = fig.add_subplot(gs[0], sharex=ax_s)
    if len(d) and cols:
        values = d[cols].to_numpy(dtype=float)
        lo, hi = np.nanmin(values, axis=0), np.nanmax(values, axis=0)
        scaled = (values - lo) / np.where(hi > lo, hi - lo, 1.0)
        x = mdates.date2num(pd.DatetimeIndex(d.index).to_pydatetime())
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(_MISSING)
        ax_h.imshow(
            np.ma.masked_invalid(scaled.T),
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            extent=(x[0], x[-1] if len(x) > 1 else x[0] + 1, n - 0.5, -0.5),
        )
        ax_h.set_yticks(range(n), [S.formula(c) for c in cols])
        ax_h.tick_params(axis="y", labelsize=S.SMALL_SIZE, length=0)
        ax_h.tick_params(axis="x", labelbottom=False)
        ax_h.set_title("Every variable, each scaled to its own range (grey: missing)", loc="left")
    if variable in d.columns:
        S.line(
            ax_s, d.index, d[variable].values, lw=0.5 if len(d) > 1500 else 0.9, color=S.OBSERVED
        )
        ax_s.set_ylabel(S.formula(variable))
    for a, b in spans:
        for ax in (ax_h, ax_s):
            if a == b:  # a row of its own: a line, as a span of none would not show
                ax.axvline(a, color=_EXCLUDED, alpha=0.8, lw=1.2, zorder=3)
            else:
                ax.axvspan(a, b, color=_EXCLUDED, alpha=0.45, lw=0, zorder=3)
    for a, b in picked:
        for ax in (ax_h, ax_s):
            if a == b:
                ax.axvline(a, color=_PICKED, alpha=0.8, lw=1.2, zorder=4)
            else:
                ax.axvspan(a, b, facecolor=_PICKED, edgecolor=_PICKED, alpha=0.25, lw=1, zorder=4)
    if isinstance(d.index, pd.DatetimeIndex):
        S.format_time_axis(ax_s)
        if len(d):
            # The record's own span, fixed: the pick's selectors add an artist at x = 0,
            # which autoscaling would stretch the axis to (back to 1970).
            ax_s.set_xlim(d.index.min(), d.index.max())
    ax_s.margins(x=0)
    S.grid(ax_s)
    return fig


# -- the pages ------------------------------------------------------------------------------
class DataTab(QWidget):
    """The Data tab's pages over the table: the overview (and the rows left out), the
    cycles, the wind, the statistics. ``exclusionsChanged`` carries the Exclude box's
    lines once they have been applied."""

    exclusionsChanged = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.pages = QTabWidget()
        self.pages.setDocumentMode(True)
        lay.addWidget(self.pages, 1)
        self.df: pd.DataFrame | None = None  # every row, as loaded
        self.kept: pd.DataFrame | None = None  # the rows the steps see
        self.target = ""
        self.spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        #: The spans picked on the overview, in time order and apart.
        self.picked: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        # Overview
        ov = QWidget()
        ol = QVBoxLayout(ov)
        ol.setContentsMargins(0, 4, 0, 0)
        self.overview = CanvasTab("The table at a glance appears here once data is loaded.")
        # On the plot's Resolution row, as cume has them over its plots: the variable
        # drawn before it, the pick's buttons and the Exclude box at its right.
        lead = QWidget(self)
        hl = QHBoxLayout(lead)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(QLabel("Time series of"))
        self.var_combo = NoWheelComboBox()
        self.var_combo.setToolTip("The variable drawn below the strips, as measured.")
        self.var_combo.currentTextChanged.connect(lambda _t: self.draw_overview())
        hl.addWidget(self.var_combo)
        buttons = QWidget(self)
        bar = QHBoxLayout(buttons)
        bar.setContentsMargins(0, 0, 0, 0)
        self.exclude_btn = QPushButton("Exclude picked")
        self.exclude_btn.setToolTip(
            "Leave the rows selected in the list -- or, with none selected, every row "
            "picked -- out of every step (training, normalisation, ...): they are written "
            "to the Exclude box."
        )
        self.exclude_btn.clicked.connect(self.exclude_picked)
        self.restore_btn = QPushButton("Restore picked")
        self.restore_btn.setToolTip(
            "Put the rows selected in the list -- or, with none selected, every row "
            "picked -- back, where they are left out."
        )
        self.restore_btn.clicked.connect(self.restore_picked)
        self.clear_btn = QPushButton("Clear pick")
        self.clear_btn.setToolTip("Pick nothing (Escape on the plot).")
        self.clear_btn.clicked.connect(self.clear_pick)
        for b in (self.exclude_btn, self.restore_btn, self.clear_btn):
            bar.addWidget(b)
        box = QWidget(self)
        bar = QHBoxLayout(box)
        bar.setContentsMargins(0, 0, 0, 0)
        self.exclude_box = QPlainTextEdit()
        self.exclude_box.setPlaceholderText("Rows left out, a span a line")
        self.exclude_box.setMaximumHeight(2 * self.exclude_box.fontMetrics().lineSpacing() + 14)
        self.exclude_box.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)  # a span a line
        self.exclude_box.setToolTip(
            "Rows left out of every step, a line each: 2020-03-23 .. 2020-04-05, or one date"
        )
        # Narrow enough to stay on the row with the others in a window of usual width.
        self.exclude_box.setMinimumWidth(160)
        self.exclude_box.setMaximumWidth(180)
        bar.addWidget(self.exclude_box, 1)
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setToolTip("Leave out the rows the box names (as edited by hand).")
        self.apply_btn.clicked.connect(self.apply_box)
        bar.addWidget(self.apply_btn)
        # Groups the row wraps between, where the window is narrow.
        self.overview.bar_lead, self.overview.bar_tail = lead, [buttons, box]
        for w in (lead, buttons, box):
            w.hide()
        # At the plot's lower left, as in cume: the pick said, and what is under the
        # cursor -- the time, the variable and its value.
        plot = QWidget()
        pl = QVBoxLayout(plot)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(self.overview, 1)
        self.pick_info = QLabel(PICK_HINT)
        self.pick_info.setWordWrap(True)
        self.pick_info.setMaximumHeight(2 * self.pick_info.fontMetrics().lineSpacing() + 4)
        self.pick_info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        pl.addWidget(self.pick_info)
        self.hover = QLabel("")
        self.hover.setStyleSheet("color:#636366;")
        # As wide as the plot whatever it says, and a line tall when it says nothing.
        self.hover.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.hover.setMinimumHeight(self.hover.fontMetrics().height())
        pl.addWidget(self.hover)
        # Under the plot, the rows picked: every column, those left out in grey. Rows are
        # selected in it as in a spreadsheet (Ctrl/Cmd, Shift, a drag down the rows) and
        # left out or put back by its menu or the buttons.
        rows = QWidget()
        rl = QVBoxLayout(rows)
        rl.setContentsMargins(0, 0, 0, 0)
        self.rows_label = QLabel("")
        rl.addWidget(self.rows_label)
        self.rows_model = _RowsModel(max_rows=PICKED_ROWS)
        self.rows_view = QTableView()
        self.rows_view.setModel(self.rows_model)
        self.rows_view.setAlternatingRowColors(True)
        self.rows_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.rows_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.rows_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.rows_view.customContextMenuRequested.connect(self._rows_menu)
        self.rows_view.selectionModel().selectionChanged.connect(lambda *_: self._sync_buttons())
        rl.addWidget(self.rows_view, 1)
        self.rows_panel = rows
        self._listed = np.zeros(0, dtype=int)  # the rows listed: their places in the table
        rows.hide()
        # The plot keeps the room it needs to be read; the rows take what is left.
        plot.setMinimumHeight(360)
        rows.setMinimumHeight(120)
        self._split = split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(plot)
        split.addWidget(rows)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setChildrenCollapsible(False)
        ol.addWidget(split, 1)
        self._press: tuple[Any, ...] | None = None  # (x, y, xdata, adding) while dragging
        self._band: list[Any] = []  # the span drawn while dragging
        self._drawn_frame: dict[str, Any] = {}  # what the overview's figure shows
        self.pages.addTab(ov, "Overview")

        # Cycles
        cy = QWidget()
        cl = QVBoxLayout(cy)
        cl.setContentsMargins(0, 4, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Cycles of"))
        self.cycle_combo = NoWheelComboBox()
        self.cycle_combo.currentTextChanged.connect(lambda _t: self.draw_cycles())
        row.addWidget(self.cycle_combo)
        row.addStretch(1)
        cl.addLayout(row)
        self.cycles = CanvasTab("A variable's diurnal, weekly and seasonal cycles.")
        cl.addWidget(self.cycles, 1)
        self.pages.addTab(cy, "Cycles")

        # Wind
        wi = QWidget()
        wl = QVBoxLayout(wi)
        wl.setContentsMargins(0, 4, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Pollutant"))
        self.wind_combo = NoWheelComboBox()
        self.wind_combo.currentTextChanged.connect(lambda _t: self.draw_wind())
        row.addWidget(self.wind_combo)
        row.addStretch(1)
        wl.addLayout(row)
        self.wind = CanvasTab("The wind rose, the pollution rose and the polar plot.")
        wl.addWidget(self.wind, 1)
        self.pages.addTab(wi, "Wind")

        # Statistics
        st = QWidget()
        sl = QVBoxLayout(st)
        sl.setContentsMargins(0, 4, 0, 0)
        self.stats = QTableWidget()
        self.stats.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.stats.verticalHeader().setVisible(False)
        sl.addWidget(self.stats, 1)
        self.stats_note = QLabel("")
        sl.addWidget(self.stats_note)
        self.pages.addTab(st, "Statistics")

        self.pages.currentChanged.connect(lambda _i: self._draw_page())
        self._sync_buttons()

    # -- the data --------------------------------------------------------------------------
    def set_data(
        self,
        df: pd.DataFrame,
        kept: pd.DataFrame,
        target: str,
        spans: Sequence[tuple[pd.Timestamp, pd.Timestamp]] = (),
    ) -> None:
        """The table as loaded (*df*), the rows kept and the target; the Exclude box set
        from *spans*. Every page is drawn again (the shown one now, the others when shown)."""
        fresh = df is not self.df or target != self.target  # a new table, or a new target
        self.df, self.kept, self.target = df, kept, target
        self.spans = list(spans)
        if fresh:
            self.picked = []
            self.pick_info.setText(PICK_HINT)
            self.pick_info.setToolTip("")
        # The same table (rows left out or put back): the pick and its list kept.
        self._show_picked_rows()
        self.exclude_box.setPlainText("\n".join(exclusion_lines(self.spans)))
        numeric = [
            str(c) for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])
        ]
        first = target if target in numeric else (numeric[0] if numeric else "")
        for combo in (self.var_combo, self.cycle_combo, self.wind_combo):
            was = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(numeric)
            # The variable chosen kept while the table is the same one; the target on a new.
            combo.setCurrentText(was if was in numeric and not fresh else first)
            combo.blockSignals(False)
        self._drawn: set[int] = set()
        self._draw_page()
        self._sync_buttons()

    def _framed(self, df: pd.DataFrame | None) -> pd.DataFrame | None:
        """*df* indexed by its dates, the rows without one left out."""
        if df is None:
            return None
        if "date" not in df.columns:
            return df
        d = df.set_index("date")
        return d[d.index.notna()]

    def _draw_page(self) -> None:
        i = self.pages.currentIndex()
        if self.df is None or i in getattr(self, "_drawn", set()):
            return
        self._drawn.add(i)
        (self.draw_overview, self.draw_cycles, self.draw_wind, self.draw_stats)[i]()

    def redraw(self) -> None:
        """Every page drawn again when next shown, the shown one now."""
        self._drawn = set()
        self._draw_page()

    # -- overview --------------------------------------------------------------------------
    def draw_overview(self) -> None:
        d = self._framed(self.df)
        if d is None:
            return
        variable = self.var_combo.currentText() or self.target
        whole = d.select_dtypes("number")

        def render(frame: pd.DataFrame) -> Any:
            fig = overview_figure(frame, variable, self.spans, self.picked)
            from matplotlib import dates as mdates

            # What is drawn, for the pick and the readout: each point's time, and the
            # first and last row each stands for where the series is averaged.
            rule = self.overview.rule
            bins = None
            if rule is not None and len(whole):
                stamps = pd.Series(whole.index, index=whole.index)
                bins = stamps.resample(rule).agg(["min", "max"]).reindex(frame.index)
            self._drawn_frame = {
                "frame": frame,
                "x": mdates.date2num(pd.DatetimeIndex(frame.index).to_pydatetime())
                if len(frame)
                else np.zeros(0),
                "columns": [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])],
                "variable": variable,
                "bins": bins,
            }
            # The pick is made on the canvas the figure goes onto, once it is there --
            # and not at all where the tab has gone first (a window closed at once).
            QTimer.singleShot(0, self, self._attach_pick)
            return fig

        self.overview.show_timeseries(render, whole)

    def _attach_pick(self) -> None:
        from shiboken6 import isValid

        canvas = getattr(self.overview, "canvas", None)
        if canvas is None or not isValid(canvas) or getattr(self, "_events_on", None) is canvas:
            return
        self._events_on = canvas
        self._press, self._band = None, []
        # The toolbar's (x, y) said at the plot's lower left instead, in the data's terms.
        toolbar = getattr(canvas, "toolbar", None)
        label = getattr(toolbar, "locLabel", None)
        if toolbar is not None and label is not None and isValid(toolbar):
            for action in toolbar.actions():
                if toolbar.widgetForAction(action) is label:
                    action.setVisible(False)
        canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)  # for Escape
        for name, fn in (
            ("button_press_event", self._on_press),
            ("motion_notify_event", self._on_motion),
            ("button_release_event", self._on_release),
            ("key_press_event", self._on_key),
            ("figure_leave_event", lambda _e: self.hover.setText("")),
        ):
            canvas.mpl_connect(name, fn)

    # -- the mouse on the overview -----------------------------------------------------------
    def _axes(self) -> tuple[Any, Any] | None:
        """The overview's series and strips (the figure's first two axes)."""
        canvas = getattr(self.overview, "canvas", None)
        axes = canvas.figure.axes if canvas is not None else []
        return (axes[0], axes[1]) if len(axes) >= 2 else None

    def _on_press(self, event: Any) -> None:
        axes = self._axes()
        toolbar = getattr(event.canvas, "toolbar", None)
        if (
            axes is None
            or event.button != 1
            or event.inaxes not in axes
            or event.xdata is None
            or (toolbar is not None and toolbar.mode)
        ):
            self._press = None
            return
        self._press = (event.x, event.y, event.xdata, _adding(event))

    def _on_motion(self, event: Any) -> None:
        if self._press is None:
            self._hover(event)
            return
        axes = self._axes()
        if axes is None or event.inaxes not in axes or event.xdata is None:
            return
        from matplotlib.patches import Rectangle
        from matplotlib.transforms import blended_transform_factory

        for artist in self._band:
            artist.remove()
        x0 = self._press[2]
        lo, hi = sorted((x0, event.xdata))
        # Added as artists, not by axvspan: a patch added would rescale the plot.
        self._band = [
            ax.add_artist(
                Rectangle(
                    (lo, 0),
                    hi - lo,
                    1,
                    transform=blended_transform_factory(ax.transData, ax.transAxes),
                    facecolor=_PICKED,
                    alpha=0.2,
                    edgecolor=_PICKED,
                    lw=1,
                    ls="--",
                    zorder=6,
                )
            )
            for ax in axes
        ]
        self._press = (*self._press[:4], event.xdata)
        event.canvas.draw_idle()

    def _on_release(self, event: Any) -> None:
        if self._press is None:
            return
        px, py, x0, add = self._press[:4]
        x1 = self._press[4] if len(self._press) > 4 else x0
        self._press = None
        for artist in self._band:
            artist.remove()
        self._band = []
        if event.xdata is not None and event.inaxes in (self._axes() or ()):
            x1 = event.xdata
        if np.hypot(event.x - px, event.y - py) < _CLICK_PX:
            i = self._point_at(event)
            if i is not None:
                self._pick_point(i, add)
            return
        from matplotlib import dates as mdates

        lo, hi = sorted((x0, x1))
        a = pd.Timestamp(mdates.num2date(lo)).tz_localize(None)
        b = pd.Timestamp(mdates.num2date(hi)).tz_localize(None)
        self.pick(a, b, add=add)

    def _point_at(self, event: Any) -> int | None:
        """The point of the figure's frame a click or the cursor names: on the series the
        point within reach of it, else -- there and on the strips -- the time under it."""
        drawn = self._drawn_frame
        x = drawn.get("x")
        axes = self._axes()
        if x is None or not len(x) or axes is None or event.xdata is None:
            return None
        if event.inaxes is axes[0] and drawn["variable"] in drawn["frame"].columns:
            y = drawn["frame"][drawn["variable"]].to_numpy(dtype=float)
            ok = np.flatnonzero(np.isfinite(y))
            if ok.size:
                xy = axes[0].transData.transform(np.column_stack([x[ok], y[ok]]))
                d = np.hypot(xy[:, 0] - event.x, xy[:, 1] - event.y)
                k = int(np.argmin(d))
                if d[k] <= _NEAR_PX * (getattr(event.canvas, "device_pixel_ratio", 1) or 1):
                    return int(ok[k])
        return int(np.argmin(np.abs(x - event.xdata)))

    def _period(self, i: int) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """The first and last row point *i* of the figure stands for (None: none)."""
        drawn = self._drawn_frame
        t = pd.Timestamp(drawn["frame"].index[i])
        bins = drawn.get("bins")
        if bins is None:
            return (t, t)
        lo, hi = bins.iloc[i]
        return None if pd.isna(lo) else (pd.Timestamp(lo), pd.Timestamp(hi))

    def _pick_point(self, i: int, add: bool) -> None:
        """A click: the rows of the point under it picked -- with Shift (Ctrl, Cmd) held,
        added to the pick, or dropped from it where they are picked already."""
        span = self._period(i)
        if span is None:
            return
        a, b = span[0].floor("min"), span[1].floor("min")
        if add and any(p0 <= a and b <= p1 for p0, p1 in self.picked):
            self.picked = _subtract(self.picked, (a, b))
            self._picked_changed()
            return
        self.pick(a, b, add=add)

    def _hover(self, event: Any) -> None:
        """What is under the cursor, said at the plot's lower left."""
        axes = self._axes()
        drawn = self._drawn_frame
        if axes is None or event.inaxes not in axes or not drawn:
            self.hover.setText("")
            return
        i = self._point_at(event)
        if i is None:
            self.hover.setText("")
            return
        frame = drawn["frame"]
        if event.inaxes is axes[1] and drawn["columns"]:
            j = int(np.clip(round(event.ydata), 0, len(drawn["columns"]) - 1))
            column = drawn["columns"][j]
        else:
            column = drawn["variable"]
        if column not in frame.columns:
            self.hover.setText("")
            return
        v = frame[column].iat[i]
        value = "—" if pd.isna(v) else f"{float(v):.4g}"
        span = self._period(i)
        if drawn.get("bins") is not None:
            if span is None:
                self.hover.setText("")
                return
            self.hover.setText(f"{_period_name(span, self.overview.rule)}: mean {column} = {value}")
            return
        t = pd.Timestamp(frame.index[i])
        if any(a <= t <= b for a, b in self.spans):
            self.hover.setText(f"{_stamp(t)}: {column} = {value}, left out")
        else:
            self.hover.setText(f"{_stamp(t)}: {column} = {value}")

    # -- the pick ------------------------------------------------------------------------------
    def pick(self, a: pd.Timestamp, b: pd.Timestamp, add: bool = False) -> None:
        """The span from *a* to *b* picked -- beside those picked already, with *add* (a
        drag with Shift, Ctrl or Cmd held) -- its rows listed, and drawn on the overview."""
        # To the minute, as the Exclude box writes it: what is picked is what is read back.
        span = (min(a, b).floor("min"), max(a, b).floor("min"))
        self.picked = _merged([*self.picked, span]) if add else [span]
        self._picked_changed()

    def _picked_changed(self) -> None:
        if not self.picked:
            self.clear_pick()
            return
        n = self._show_picked_rows()
        if len(self.picked) == 1:
            ((s0, s1),) = self.picked
            self.pick_info.setText(f"Picked {_stamp(s0)} – {_stamp(s1)} ({n:,} rows)")
        else:
            lines = exclusion_lines(self.picked)
            shown = "; ".join(lines[:3]) + (f"; … {len(lines) - 3} more" if len(lines) > 3 else "")
            self.pick_info.setText(f"Picked {len(lines)} spans ({n:,} rows): {shown}")
        self.pick_info.setToolTip("\n".join(exclusion_lines(self.picked)))
        self._sync_buttons()
        self.draw_overview()

    def _on_key(self, event: Any) -> None:
        if event.key == "escape" and self.picked:
            self.clear_pick()

    def _show_picked_rows(self) -> int:
        """The picked spans' rows listed under the plot (every column, the left-out in
        grey): how many there are."""
        if self.df is None or not self.picked or "date" not in self.df.columns:
            self._listed = np.zeros(0, dtype=int)
            self.rows_panel.hide()
            return 0
        inside = excluded_mask(self.df["date"], self.picked)
        self._listed = np.flatnonzero(inside)
        rows = self.df[inside].copy()
        out = excluded_mask(rows["date"], self.spans)
        from .uilang import tr

        # Our column, not the table's: in the window's language.
        rows.insert(0, tr("Left out"), np.where(out, tr("yes"), ""))
        self.rows_model.left_out = out[:PICKED_ROWS]
        self.rows_model.set_dataframe(rows)
        self.rows_view.resizeColumnsToContents()
        shown = min(len(rows), PICKED_ROWS)
        self.rows_label.setText(
            f"{len(rows):,} rows picked"
            + (f", {int(out.sum()):,} of them left out" if out.any() else "")
            + (f" (the first {shown:,} listed)" if shown < len(rows) else "")
        )
        if len(rows) and not self.rows_panel.isVisible():
            # Shown: two thirds to the plot, a third to the rows.
            total = max(self._split.height(), 600)
            self._split.setSizes([total * 2 // 3, total - total * 2 // 3])
        self.rows_panel.setVisible(bool(len(rows)))
        self._sync_buttons()
        return len(rows)

    def selected_rows(self) -> np.ndarray:
        """The places in the table of the rows selected in the list (none: an empty one)."""
        model = self.rows_view.selectionModel()
        picked = sorted({ix.row() for ix in model.selectedRows()}) if model is not None else []
        picked = [r for r in picked if r < len(self._listed)]
        return self._listed[picked] if picked else np.zeros(0, dtype=int)

    def _acted_on(self) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """What the buttons act on: the rows selected in the list, else every row picked."""
        sel = self.selected_rows()
        if len(sel) and self.df is not None:
            return _runs(self.df["date"], sel)
        return list(self.picked)

    def _rows_menu(self, pos: Any) -> None:
        """The list's menu: the rows selected left out or put back, as in a spreadsheet."""
        n = len(self.selected_rows())
        menu = QMenu(self.rows_view)
        exclude = menu.addAction(f"Exclude {n:,} selected rows")
        exclude.setEnabled(n > 0)
        exclude.triggered.connect(self.exclude_picked)
        restore = menu.addAction(f"Restore {n:,} selected rows")
        restore.setEnabled(n > 0 and self._restorable(self._acted_on()))
        restore.triggered.connect(self.restore_picked)
        menu.addSeparator()
        menu.addAction("Select all", self.rows_view.selectAll)
        menu.exec(self.rows_view.viewport().mapToGlobal(pos))

    def clear_pick(self) -> None:
        self.picked = []
        self._listed = np.zeros(0, dtype=int)
        self.rows_panel.hide()
        self.pick_info.setText(PICK_HINT)
        self.pick_info.setToolTip("")
        self._sync_buttons()
        self.draw_overview()

    def exclude_picked(self) -> None:
        spans = self._acted_on()
        if spans:
            self._apply([*self.spans, *spans])

    def restore_picked(self) -> None:
        spans = self.spans
        for cut in self._acted_on():
            spans = _subtract(spans, cut)
        if self.df is not None and "date" in self.df.columns:
            # The pieces the restore cut written by their rows; the lines as typed kept.
            new = [s for s in spans if s not in self.spans]
            spans = [s for s in spans if s in self.spans] + _on_rows(new, self.df["date"])
        self._apply(spans)

    def apply_box(self) -> None:
        try:
            spans = parse_exclusions(self.exclude_box.toPlainText())
        except ValueError as exc:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Exclude", str(exc))
            return
        self._apply(spans)

    def _apply(self, spans: list[tuple[pd.Timestamp, pd.Timestamp]]) -> None:
        """The rows left out set to *spans*: the pick and its list kept, to go on from."""
        self.spans = _merged(spans)
        self.exclude_box.setPlainText("\n".join(exclusion_lines(self.spans)))
        self._show_picked_rows()
        self._sync_buttons()
        self.exclusionsChanged.emit(exclusion_lines(self.spans))

    def _restorable(self, spans: Sequence[tuple[pd.Timestamp, pd.Timestamp]]) -> bool:
        return any(not (b < p0 or a > p1) for a, b in self.spans for p0, p1 in spans)

    def _sync_buttons(self) -> None:
        has = self.df is not None
        self.exclude_btn.setEnabled(has and bool(self.picked))
        self.restore_btn.setEnabled(has and self._restorable(self._acted_on()))
        self.clear_btn.setEnabled(bool(self.picked))
        self.apply_btn.setEnabled(has)

    def draw_cycles(self) -> None:
        d = self._framed(self.kept)
        var = self.cycle_combo.currentText()
        if d is None or not var or var not in d.columns:
            return
        from normet.plotting import cycle_profiles, usable_cycles

        if not isinstance(d.index, pd.DatetimeIndex) or not usable_cycles(d.index):
            self.cycles.show_result(None, lines=["The record's time step shows no cycle."])
            return
        fig = cycle_profiles(d[[var]], [var], ylabel=S.formula(var))
        self.cycles.show_result(
            fig,
            lines=[
                f"Mean {var} by the hour of the day, the day of the week and the "
                "month, each with its 95 % confidence interval -- over the rows kept."
            ],
        )

    # -- wind --------------------------------------------------------------------------------
    def draw_wind(self) -> None:
        d = self._framed(self.kept)
        var = self.wind_combo.currentText()
        if d is None or not var:
            return
        from normet.plotting import polar_plot, pollution_rose, wind_rose

        from .polar_window import wind_columns

        frame, ws, wd = wind_columns(d)
        if ws is None or wd is None:
            self.wind.show_result(
                None,
                lines=["The table has no wind speed and direction (ws and wd, or u10 and v10)."],
            )
            return
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            1, 3, subplot_kw={"projection": "polar"}, figsize=(12, 3.8), layout="constrained"
        )
        wind_rose(frame, ws_col=ws, wd_col=wd, ax=axes[0])
        if var in frame.columns and var not in (ws, wd):
            pollution_rose(frame, target=var, wd_col=wd, ax=axes[1])
            polar_plot(frame, target=var, ws_col=ws, wd_col=wd, ax=axes[2])
        else:
            for ax in axes[1:]:
                ax.set_axis_off()
        self.wind.show_result(
            fig,
            lines=[
                "How often the wind blows from each direction (by speed); the same cut by "
                f"{var}'s concentration; and {var}'s mean by wind "
                "direction and speed -- over the rows kept."
            ],
        )

    # -- statistics --------------------------------------------------------------------------
    def draw_stats(self) -> None:
        d = self.kept
        if d is None:
            return
        header = ["Column", "Values", "Missing", "Mean", "SD", "Min", "25%", "Median", "75%", "Max"]
        rows = []
        for c in d.columns:
            if c == "date":
                continue
            s = d[c]
            missing = 100.0 * s.isna().mean() if len(s) else 0.0
            if pd.api.types.is_numeric_dtype(s):
                q = s.quantile([0.25, 0.5, 0.75])
                vals = [s.mean(), s.std(), s.min(), q[0.25], q[0.5], q[0.75], s.max()]
                rows.append(
                    [str(c), f"{s.notna().sum():,}", f"{missing:.1f}%", *(f"{v:.4g}" for v in vals)]
                )
            else:
                rows.append([str(c), f"{s.notna().sum():,}", f"{missing:.1f}%", *[""] * 7])
        self.stats.clear()
        self.stats.setColumnCount(len(header))
        self.stats.setHorizontalHeaderLabels(header)
        self.stats.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                item = QTableWidgetItem(text)
                if c:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                self.stats.setItem(r, c, item)
        self.stats.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        left = 0 if self.df is None else len(self.df) - len(d)
        self.stats_note.setText(
            f"{len(d):,} rows kept"
            + (f"; {left:,} left out (Overview → Exclude)." if left else ".")
        )
