"""The Data tab: the table at a glance, its cycles, its winds and its statistics -- and
the rows left out of every step.

- Overview: every numeric column as a strip, each scaled to its own range (missing
  values grey), over the whole record; below it, one variable as measured. A span
  dragged across either is picked -- its rows listed beneath the plot -- and *Exclude
  picked* leaves its rows out of every
  step -- training, normalisation, decomposition, rolling. The rows left out are greyed
  where they are, and written to the Exclude box as the lines ``exclusion_lines``
  writes: the box is the record the window, its command log and a saved session read.
- Cycles: a variable's diurnal, weekly and seasonal cycles (:func:`normet.cycle_profiles`).
- Wind: the wind rose, the pollution rose and the polar plot of a variable.
- Statistics: each column's count, missing share and spread.

The rows of a span picked on the overview are listed under it, every column, the ones
left out marked: what a span holds is read where it is picked.

Everything but the overview reads the rows kept: what the steps see.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
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

#: The rows of a span picked listed at most (a year of hours and more).
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
            out.append(_stamp(a) if a == b else f"{_stamp(a)} .. {_stamp(b)}")
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


# -- the overview's figure -----------------------------------------------------------------
def overview_figure(
    d: pd.DataFrame,
    variable: str,
    spans: Sequence[tuple[pd.Timestamp, pd.Timestamp]] = (),
    picked: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> Any:
    """Every numeric column of *d* (indexed by time) as a strip scaled to its own range
    -- missing values grey -- above *variable* as measured; the spans left out greyed,
    the span picked in red. The series' axes are the figure's first."""
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
            ax.axvspan(a, b, color=_EXCLUDED, alpha=0.45, lw=0, zorder=3)
    if picked is not None:
        for ax in (ax_h, ax_s):
            ax.axvspan(picked[0], picked[1], color=_PICKED, alpha=0.25, lw=0, zorder=4)
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
        self.picked: tuple[pd.Timestamp, pd.Timestamp] | None = None
        self._selectors: list[Any] = []

        # Overview
        ov = QWidget()
        ol = QVBoxLayout(ov)
        ol.setContentsMargins(0, 4, 0, 0)
        top = QHBoxLayout()
        top.addWidget(QLabel("Time series of"))
        self.var_combo = NoWheelComboBox()
        self.var_combo.setToolTip("The variable drawn below the strips, as measured.")
        self.var_combo.currentTextChanged.connect(lambda _t: self.draw_overview())
        top.addWidget(self.var_combo)
        top.addStretch(1)
        self.pick_info = QLabel("Drag across the plot to pick a span of rows.")
        self.pick_info.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        top.addWidget(self.pick_info, 2)
        ol.addLayout(top)
        self.overview = CanvasTab("The table at a glance appears here once data is loaded.")
        # Under the plot, the rows of the span picked: every column, the left-out marked.
        rows = QWidget()
        rl = QVBoxLayout(rows)
        rl.setContentsMargins(0, 0, 0, 0)
        self.rows_label = QLabel("")
        rl.addWidget(self.rows_label)
        self.rows_model = DataFrameModel(max_rows=PICKED_ROWS)
        self.rows_view = QTableView()
        self.rows_view.setModel(self.rows_model)
        self.rows_view.setAlternatingRowColors(True)
        rl.addWidget(self.rows_view, 1)
        self.rows_panel = rows
        rows.hide()
        # The plot keeps the room it needs to be read; the rows take what is left.
        self.overview.setMinimumHeight(360)
        rows.setMinimumHeight(120)
        self._split = split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.overview)
        split.addWidget(rows)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setChildrenCollapsible(False)
        ol.addWidget(split, 1)
        bar = QHBoxLayout()
        self.exclude_btn = QPushButton("Exclude picked")
        self.exclude_btn.setToolTip(
            "Leave the picked rows out of every step (training, normalisation, ...): "
            "they are written to the Exclude box."
        )
        self.exclude_btn.clicked.connect(self.exclude_picked)
        self.restore_btn = QPushButton("Restore picked")
        self.restore_btn.setToolTip("Put the picked rows that are left out back.")
        self.restore_btn.clicked.connect(self.restore_picked)
        self.clear_btn = QPushButton("Clear pick")
        self.clear_btn.clicked.connect(self.clear_pick)
        for b in (self.exclude_btn, self.restore_btn, self.clear_btn):
            bar.addWidget(b)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Exclude:"), 0, Qt.AlignmentFlag.AlignTop)
        self.exclude_box = QPlainTextEdit()
        self.exclude_box.setPlaceholderText(
            "Rows left out of every step, a line each: 2020-03-23 .. 2020-04-05, or one date"
        )
        self.exclude_box.setMaximumHeight(2 * self.exclude_box.fontMetrics().lineSpacing() + 14)
        bar.addWidget(self.exclude_box, 1)
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setToolTip("Leave out the rows the box names (as edited by hand).")
        self.apply_btn.clicked.connect(self.apply_box)
        bar.addWidget(self.apply_btn, 0, Qt.AlignmentFlag.AlignTop)
        ol.addLayout(bar)
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
        self.picked = None
        self.rows_panel.hide()
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

        def render(frame: pd.DataFrame) -> Any:
            fig = overview_figure(frame, variable, self.spans, self.picked)
            # The pick is made on the canvas the figure goes onto, once it is there.
            QTimer.singleShot(0, self._attach_selectors)
            return fig

        self.overview.show_timeseries(render, d.select_dtypes("number"))

    def _attach_selectors(self) -> None:
        from matplotlib import dates as mdates
        from matplotlib.widgets import SpanSelector

        canvas = getattr(self.overview, "canvas", None)
        if canvas is None:
            return
        self._selectors = []
        for ax in canvas.figure.axes:

            def picked(lo: float, hi: float) -> None:
                if hi <= lo:
                    return
                a = pd.Timestamp(mdates.num2date(lo)).tz_localize(None)
                b = pd.Timestamp(mdates.num2date(hi)).tz_localize(None)
                self.pick(a, b)

            self._selectors.append(
                SpanSelector(
                    ax,
                    picked,
                    "horizontal",
                    useblit=False,
                    props={"facecolor": _PICKED, "alpha": 0.25},
                    interactive=False,
                )
            )

    def pick(self, a: pd.Timestamp, b: pd.Timestamp) -> None:
        """The span from *a* to *b* picked: said, and drawn on the overview."""
        # To the minute, as the Exclude box writes it: what is picked is what is read back.
        self.picked = (min(a, b).floor("min"), max(a, b).floor("min"))
        n = self._show_picked_rows()
        self.pick_info.setText(
            f"Picked {_stamp(self.picked[0])} – {_stamp(self.picked[1])} ({n:,} rows)"
        )
        self._sync_buttons()
        self.draw_overview()

    def _show_picked_rows(self) -> int:
        """The picked span's rows listed under the plot (every column, the left-out marked):
        how many there are."""
        if self.df is None or self.picked is None or "date" not in self.df.columns:
            self.rows_panel.hide()
            return 0
        inside = excluded_mask(self.df["date"], [self.picked])
        rows = self.df[inside].copy()
        out = excluded_mask(rows["date"], self.spans)
        rows.insert(0, "Left out", np.where(out, "yes", ""))
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
        return len(rows)

    def clear_pick(self) -> None:
        self.picked = None
        self.rows_panel.hide()
        self.pick_info.setText("Drag across the plot to pick a span of rows.")
        self._sync_buttons()
        self.draw_overview()

    def exclude_picked(self) -> None:
        if self.picked is None:
            return
        self._apply([*self.spans, self.picked])

    def restore_picked(self) -> None:
        if self.picked is None:
            return
        self._apply(_subtract(self.spans, self.picked))

    def apply_box(self) -> None:
        try:
            spans = parse_exclusions(self.exclude_box.toPlainText())
        except ValueError as exc:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Exclude", str(exc))
            return
        self._apply(spans)

    def _apply(self, spans: list[tuple[pd.Timestamp, pd.Timestamp]]) -> None:
        self.spans = _merged(spans)
        self.picked = None
        self.rows_panel.hide()
        self.pick_info.setText("Drag across the plot to pick a span of rows.")
        self.exclude_box.setPlainText("\n".join(exclusion_lines(self.spans)))
        self._sync_buttons()
        self.exclusionsChanged.emit(exclusion_lines(self.spans))

    def _sync_buttons(self) -> None:
        has = self.df is not None
        self.exclude_btn.setEnabled(has and self.picked is not None)
        self.restore_btn.setEnabled(
            has
            and self.picked is not None
            and any(not (b < self.picked[0] or a > self.picked[1]) for a, b in self.spans)
        )
        self.clear_btn.setEnabled(self.picked is not None)
        self.apply_btn.setEnabled(has)

    # -- cycles ------------------------------------------------------------------------------
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
