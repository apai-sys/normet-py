"""The Data and Columns boxes: loading a table and choosing what to model.

The first two steps of the left panel, with everything that reacts to them --
the CSV loader, the drag-and-drop handlers and the target series preview.
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

from normet.utils._time import to_datetime_coerced

from . import _settings, commandlog
from ._constants import MET_DEFAULTS
from ._lists import _checked_items, _fill_checklist, _set_all
from ._widgets import NoWheelComboBox, hint_label

log = logging.getLogger(__name__)


class InputsMixin(QMainWindow):
    def _build_data_box(self) -> QGroupBox:
        """Data: open a CSV, or load the bundled example."""
        box = QGroupBox("Data")
        dv = QVBoxLayout(box)
        file_row = QHBoxLayout()
        file_row.setContentsMargins(0, 0, 0, 0)
        self.file_btn = QPushButton("Open CSV…")
        self.file_btn.setToolTip(
            "Load a time series CSV with a 'date' column.\nYou can also drag && drop a file onto the window."
        )
        self.file_btn.clicked.connect(self._open_csv)
        self.example_btn = QPushButton("Load example")
        self.example_btn.setToolTip(
            "Generate a one-year synthetic air-quality + meteorology dataset\nso you can try every step immediately."
        )
        self.example_btn.clicked.connect(self.load_example)
        file_row.addWidget(self.file_btn)
        file_row.addWidget(self.example_btn)
        dv.addLayout(file_row)
        self.file_label = hint_label("No data loaded", small=False)
        dv.addWidget(self.file_label)
        return box

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
        b_none = QPushButton("None")
        b_met = QPushButton("Met only")
        b_met.setToolTip("Tick only the recognised meteorological variables.")
        b_all.clicked.connect(lambda: _set_all(self.feature_list, True))
        b_none.clicked.connect(lambda: _set_all(self.feature_list, False))
        b_met.clicked.connect(self._check_met_only)
        btn_row.addWidget(b_all)
        btn_row.addWidget(b_none)
        btn_row.addWidget(b_met)
        cv.addLayout(btn_row)
        cv.addWidget(
            hint_label(
                "Time features (date_unix, day_julian, weekday, hour) are added automatically."
            )
        )
        return box

    def _open_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open time-series CSV", "", "CSV files (*.csv *.csv.gz)"
        )
        if path:
            self.load_csv(path, remember=True)

    def load_example(self) -> None:
        from ._examples import make_deweather_example

        df = make_deweather_example()
        self._ingest(df, "Example data (synthetic, 1 year hourly)")
        self._log_command(commandlog.load_example())
        self.statusBar().showMessage(
            "Example loaded — target PM2.5 with met features preselected. Click ▶ Train model."
        )

    def load_csv(self, path: str, remember: bool = False) -> None:
        """Load *path* into the Data tab and populate the column selectors."""
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to read CSV", str(exc))
            return

        date_col = next((c for c in df.columns if c.lower() == "date"), None)
        if date_col is None:
            QMessageBox.warning(
                self,
                "No date column",
                "The CSV must contain a 'date' column with timestamps.\n\n"
                "For synthetic-control panel data, use the SCM Studio instead "
                "(Analysis → Synthetic Control).",
            )
            return
        df[date_col] = to_datetime_coerced(df[date_col])
        df = df.rename(columns={date_col: "date"})

        self._ingest(df, os.path.basename(path))
        self._log_command(commandlog.open_csv(os.path.abspath(path), date_col))
        if remember:
            _settings.add_recent(path)
            self._rebuild_recent_menu()
        log.info("Loaded %s (%d rows)", path, len(df))

    def _ingest(self, df: pd.DataFrame, label: str) -> None:
        self.df_raw = df
        self.df_prep = None
        self.model = None
        self.chronos_check = None
        self.trained_features = []
        self.results.clear()
        self.tab_model.show_message("Train a model (Step 1) to see its quality here.")

        span = f"{df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d}"
        self.file_label.setText(f"{label}\n{len(df):,} rows × {df.shape[1]} columns")
        self.file_label.setStyleSheet("")
        self.table_model.set_dataframe(df)

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
            "Data loaded — check the target and features, then ▶ Train model (Step 1)."
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
        """Quick look at the chosen target series (daily means when dense)."""
        target = self.target_combo.currentText()
        if self.df_raw is None or not target or target not in self.df_raw.columns:
            return
        import matplotlib.pyplot as plt

        s = self.df_raw.set_index("date")[target].dropna()
        # A date that did not parse has no place on the axis; with a time zone, one
        # NaT also stops matplotlib converting the whole axis.
        s = s[s.index.notna()]
        note = ""
        if len(s) > 20_000:
            s = s.resample("D").mean()
            note = " (daily means)"
        fig, ax = plt.subplots(figsize=(9, 2.4))
        ax.plot(s.index, s.values, lw=0.7, color="#2c7bb6")
        ax.set_title(f"{target}{note}", fontsize=10, loc="left")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        self.data_preview.show_figure(fig)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        md = event.mimeData()
        if md.hasUrls() and any(
            u.toLocalFile().lower().endswith((".csv", ".csv.gz")) for u in md.urls()
        ):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        for u in event.mimeData().urls():
            p = u.toLocalFile()
            if p.lower().endswith((".csv", ".csv.gz")):
                self.load_csv(p, remember=True)
                break

    def _selected_features(self) -> list[str]:
        target = self.target_combo.currentText()
        return [f for f in _checked_items(self.feature_list) if f != target]
