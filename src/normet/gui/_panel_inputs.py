"""The Data and Columns boxes: loading a table and choosing what to model.

The left panel's Data page, with everything that reacts to it -- the table loader
(:func:`normet.io.read_time_series`: text, Excel, Parquet, Feather or JSON, its times
however written, put on UTC from the time zone chosen), the examples (``._examples``,
offered under File → Load Example and on the welcome), the
drag-and-drop handlers and the target series preview.
A mixin of :class:`~.main_window.MainWindow`: the widgets it creates are
attributes of the window, which is how the run handlers reach them.
"""

from __future__ import annotations

import logging
import os

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from normet.io.tables import TABLE_SUFFIXES

from . import _settings, commandlog
from ._constants import MET_DEFAULTS
from ._lists import ALL_TIP, _checked_items, _fill_checklist, _toggle_all
from ._widgets import NoWheelComboBox, hint_label

log = logging.getLogger(__name__)

#: What Open Data… offers, by kind.
TABLE_FILTER = (
    "Tables (*.csv *.txt *.tsv *.dat *.gz *.xlsx *.xlsm *.xls *.parquet *.feather *.json);;"
    "CSV and text (*.csv *.txt *.tsv *.dat *.gz);;Excel (*.xlsx *.xlsm *.xls);;"
    "Parquet, Feather, JSON (*.parquet *.feather *.json);;All files (*)"
)

#: The clocks a table can be kept on: (what the list shows, the IANA zone). The
#: meteorology joined to a table -- ERA5, Open-Meteo, HYSPLIT -- is on UTC.
ZONES: tuple[tuple[str, str], ...] = (
    ("UTC (as recorded)", "UTC"),
    ("UK — Europe/London (GMT/BST)", "Europe/London"),
    ("Central Europe — Europe/Paris (CET/CEST)", "Europe/Paris"),
    ("China — Asia/Shanghai (UTC+8)", "Asia/Shanghai"),
    ("Hong Kong — Asia/Hong_Kong (UTC+8)", "Asia/Hong_Kong"),
    ("Taiwan — Asia/Taipei (UTC+8)", "Asia/Taipei"),
    ("Japan — Asia/Tokyo (UTC+9)", "Asia/Tokyo"),
    ("Korea — Asia/Seoul (UTC+9)", "Asia/Seoul"),
    ("India — Asia/Kolkata (UTC+5:30)", "Asia/Kolkata"),
    ("US Eastern — America/New_York", "America/New_York"),
    ("US Central — America/Chicago", "America/Chicago"),
    ("US Mountain — America/Denver", "America/Denver"),
    ("US Pacific — America/Los_Angeles", "America/Los_Angeles"),
    ("Australia East — Australia/Sydney", "Australia/Sydney"),
    ("UTC+8, no summer time — Etc/GMT-8", "Etc/GMT-8"),
    ("UTC−5, no summer time — Etc/GMT+5", "Etc/GMT+5"),
)
ZONE_PREF = "data_time_zone"


class InputsMixin(QMainWindow):
    def _build_data_box(self) -> QGroupBox:
        """Data: open a table, and the clock its times are kept on. The examples are under
        File → Load Example and on the welcome."""
        from PySide6.QtCore import QSettings

        box = QGroupBox("Data")
        dv = QVBoxLayout(box)
        self.file_btn = QPushButton("Open Data…")
        self.file_btn.setToolTip(
            "Load a time series: CSV or other delimited text, Excel, Parquet, Feather or "
            "JSON, with its times in a column (date, datetime, timestamp, time …) or in "
            "year, month, day and hour columns, written in any of the usual ways. You can "
            "also drop a file onto the window."
        )
        self.file_btn.clicked.connect(self._open_csv)
        dv.addWidget(self.file_btn)
        self.file_label = hint_label("No data loaded", small=False)
        dv.addWidget(self.file_label)
        zone_row = QHBoxLayout()
        zone_row.setContentsMargins(0, 0, 0, 0)
        zone_row.addWidget(QLabel("Time zone"))
        self.zone_combo = NoWheelComboBox()
        # Its width from the panel, not from its longest zone: the list opens wider.
        self.zone_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.zone_combo.setMinimumContentsLength(10)
        for text, zone in ZONES:
            self.zone_combo.addItem(text, zone)
        self.zone_combo.setToolTip(
            "The clock the analysis keeps, and its figures and tables show: the site's local "
            "time, or UTC. A table's times written without an offset are taken to be on it; "
            "times written with one (+08:00) are put on it. The hour and weekday the model "
            "learns from are then the site's own, so a rush hour stays at its hour through "
            "summer time. Meteorology and trajectories fetched on UTC (the Data and "
            "Transport Studios) are put on this clock as they are joined."
        )
        kept = str(QSettings("apai-sys", "normet-gui").value(ZONE_PREF, "UTC"))
        self.zone_combo.setCurrentIndex(max(0, self.zone_combo.findData(kept)))
        self.zone_combo.currentIndexChanged.connect(self._zone_changed)
        zone_row.addWidget(self.zone_combo, 1)
        dv.addLayout(zone_row)
        self.dates_label = hint_label("")
        self.dates_label.setVisible(False)
        dv.addWidget(self.dates_label)
        #: The file the table on screen was read from, read again when the zone changes.
        self._data_path: str | None = None
        return box

    def _zone(self) -> str:
        return str(self.zone_combo.currentData() or "UTC")

    def _zone_changed(self, _index: int) -> None:
        """Another clock chosen: kept for the next session, and the open file read again
        on it (the model and results go with the old times)."""
        from PySide6.QtCore import QSettings

        QSettings("apai-sys", "normet-gui").setValue(ZONE_PREF, self._zone())
        if self._data_path is not None and self.df_raw is not None:
            self.load_csv(self._data_path)

    def _build_columns_box(self) -> QGroupBox:
        """Columns: the target series and the predictors ticked for the model."""
        box = QGroupBox("Columns")
        cv = QVBoxLayout(box)
        cv.addWidget(QLabel("Target (pollutant)"))
        self.target_combo = NoWheelComboBox()
        self.target_combo.setToolTip("The variable to model and normalise.")
        self.target_combo.currentTextChanged.connect(self._target_changed)
        cv.addWidget(self.target_combo)
        cv.addWidget(QLabel("Features (predictors)"))
        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.feature_list.setToolTip(
            "Tick the meteorological (and other) predictors for the model."
        )
        self.feature_list.setMinimumHeight(140)
        self.feature_list.setMaximumHeight(200)
        cv.addWidget(self.feature_list)
        btn_row = QHBoxLayout()
        b_all = QPushButton("All")
        b_all.setToolTip(ALL_TIP)
        b_met = QPushButton("Met only")
        b_met.setToolTip("Tick only the recognised meteorological variables.")
        b_all.clicked.connect(lambda: _toggle_all(self.feature_list))
        b_met.clicked.connect(self._check_met_only)
        btn_row.addWidget(b_all)
        btn_row.addWidget(b_met)
        cv.addLayout(btn_row)
        cv.addWidget(
            hint_label(
                "Time features (date_unix, day_julian, weekday, hour) are added automatically."
            )
        )
        return box

    def _open_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open data", "", TABLE_FILTER)
        if path:
            self.load_csv(path, remember=True)

    def load_example(self, key: str = "synthetic") -> None:
        """An example (``._examples.EXAMPLES``), opened with its target and predictors set
        as its source sets them."""
        from ._examples import EXAMPLES

        ex = EXAMPLES[key]
        df = ex.load()
        self._data_path = None
        self._ingest(df, ex.title)
        self._show_dates(None)
        if ex.target in df.columns:
            self.target_combo.setCurrentText(ex.target)
        if ex.features is not None:
            numeric = [self.feature_list.item(i).text() for i in range(self.feature_list.count())]
            _fill_checklist(self.feature_list, numeric, set(ex.features))
        self._log_command(commandlog.load_example(ex))
        self.statusBar().showMessage(
            f"{ex.title}: set up as {ex.source} — check the target and features, then "
            "▶ Train model."
        )

    def load_csv(self, path: str, remember: bool = False) -> None:
        """Load the table at *path* (any file :func:`normet.io.read_time_series` reads)
        into the Data tab, its times on UTC from the time zone chosen, and populate the
        column selectors."""
        from normet.io import read_time_series

        try:
            df = read_time_series(path, zone=self._zone())
        except ValueError as exc:
            QMessageBox.warning(
                self,
                "No times found",
                f"{exc}\n\nFor synthetic-control panel data, use the SCM Studio instead "
                "(Analysis → Synthetic Control (SCM) Studio).",
            )
            return
        except Exception as exc:  # noqa: BLE001 - a file pandas cannot read, said as it is
            QMessageBox.critical(self, "Failed to read the file", str(exc))
            return
        self._data_path = path
        self._ingest(df, os.path.basename(path))
        self._show_dates(df)
        self._log_command(commandlog.open_table(os.path.abspath(path), self._zone()))
        if remember:
            _settings.add_recent(path)
            self._rebuild_recent_menu()
        log.info("Loaded %s (%d rows): times %s", path, len(df), df.attrs.get("normet_dates"))

    def _show_dates(self, df: pd.DataFrame | None) -> None:
        """How the table's times were read, under the file's name -- and, where some could
        not be, how many."""
        if df is None:
            self.dates_label.setVisible(False)
            return
        how = df.attrs.get("normet_dates", "")
        bad = int(df["date"].isna().sum())
        parts = ["Times", *how.split(" · ")] if how else []
        if bad:
            parts.append(f"{bad:,} could not be read and are left empty")
        text = " · ".join(parts)
        self.dates_label.setText(text)
        self.dates_label.setVisible(bool(text))

    def _ingest(self, df: pd.DataFrame, label: str) -> None:
        self.df_raw = self.df_loaded = df
        self.exclusions = []
        self.df_prep = None
        self.model = None
        self.chronos_check = None
        self.trained_features = []
        self.results.clear()
        self._multiscale_raw = None
        self.tab_model.show_message("Train a model to see its quality here.")

        # The clock the times are on: every figure and table after this one keeps it.
        span = f"{df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d} ({self._zone()})"
        self.file_label.setText(f"{label}\n{len(df):,} rows × {df.shape[1]} columns")
        self.file_label.setStyleSheet("")

        n_nan = int(df.isna().sum().sum())
        self.data_summary.setText(
            f"{len(df):,} rows × {df.shape[1]} columns   |   date range: {span}"
            f"   |   missing values: {n_nan:,}"
        )

        numeric = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItems(numeric)
        pollutant = next(
            (
                c
                for c in numeric
                if c.upper() in {"PM2.5", "PM25", "PM10", "NO2", "O3", "SO2", "CO", "NOX"}
            ),
            numeric[0] if numeric else "",
        )
        if pollutant:
            self.target_combo.setCurrentText(pollutant)
        self.target_combo.blockSignals(False)

        met = {c for c in numeric if c.lower() in MET_DEFAULTS or c in MET_DEFAULTS}
        _fill_checklist(self.feature_list, numeric, met or set(numeric))
        self.pdp_vars.clear()
        self._reset_pdp_toggle_buttons()

        self._sync_enabled()
        self._sync_welcome()
        self._draw_data_preview()
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage(
            "Data loaded — check the target and features, then ▶ Train model."
        )

    def _target_changed(self, _text: str) -> None:
        if self.df_raw is not None:
            self._draw_data_preview()

    def _check_met_only(self) -> None:
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            is_met = item.text().lower() in MET_DEFAULTS or item.text() in MET_DEFAULTS
            item.setCheckState(Qt.CheckState.Checked if is_met else Qt.CheckState.Unchecked)

    def _draw_data_preview(self) -> None:
        """The Data tab over the table as loaded and the rows kept: the overview (daily
        means when dense), the cycles, the winds, the statistics."""
        if self.df_loaded is None:
            return
        from ._data_tab import parse_exclusions

        self.data_tab.set_data(
            self.df_loaded,
            self.df_raw,
            self.target_combo.currentText(),
            parse_exclusions(self.exclusions),
        )

    def _set_exclusions(self, lines: list[str]) -> None:
        """The rows the Exclude box names left out of every step: df_raw is the table as
        loaded without them. Logged as the line that leaves them out; a model already
        trained was trained on the rows before, which the window says."""
        from ._data_tab import excluded_mask, parse_exclusions

        lines = list(lines)
        if self.df_loaded is None or lines == self.exclusions:
            return
        mask = excluded_mask(self.df_loaded["date"], parse_exclusions(lines))
        self.exclusions = lines
        self.df_raw = (
            self.df_loaded[~mask].reset_index(drop=True) if mask.any() else self.df_loaded
        )
        self._log_command(commandlog.exclude(lines))
        self._say_rows()
        if self.model is not None:
            self.notify(
                "The rows left out changed: the model and its results were made with the "
                "rows before. Train again to use them.",
                key="exclusions-changed",
                kind="warn",
                actions=(("Train", self._run_train),),
            )
        self._draw_data_preview()
        self._sync_enabled()

    def _say_rows(self) -> None:
        """The data summary: the rows kept, and those left out where any are."""
        df = self.df_loaded
        if df is None:
            return
        span = f"{df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d} ({self._zone()})"
        n_nan = int(df.isna().sum().sum())
        left = len(df) - len(self.df_raw)
        self.data_summary.setText(
            f"{len(df):,} rows × {df.shape[1]} columns   |   date range: {span}"
            f"   |   missing values: {n_nan:,}"
            + (f"   |   left out: {left:,} rows" if left else "")
        )

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        md = event.mimeData()
        if md.hasUrls() and any(
            u.toLocalFile().lower().endswith(TABLE_SUFFIXES) for u in md.urls()
        ):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        for u in event.mimeData().urls():
            p = u.toLocalFile()
            if p.lower().endswith(TABLE_SUFFIXES):
                self.load_csv(p, remember=True)
                break

    def _selected_features(self) -> list[str]:
        target = self.target_combo.currentText()
        return [f for f in _checked_items(self.feature_list) if f != target]
