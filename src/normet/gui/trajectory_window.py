"""Transport Studio — HYSPLIT back-trajectory (transport-aware) features.

Turns HYSPLIT ``tdump`` back-trajectory output into per-receptor transport
descriptors (inflow direction, transport distance/speed, residence time over
named source regions, along-path rainfall/boundary-layer height) that join
straight onto a date-indexed panel as extra predictors — see
:mod:`normet.io.trajectory`.

Two sections, in the same design language as the Data/SCM Studios:

1. **Build features from tdump files** — parse existing HYSPLIT output. No
   external binary required. An optional site name tags the result with a
   'site' column, so batches from several receptors (processed one at a
   time) can all be sent to the main window and matched up correctly.
2. **Run HYSPLIT (advanced)** — orchestrate ``hyts_std`` itself for one or
   more receptor sites, at a configurable time resolution, optionally
   downloading GDAS1 meteorology first. Needs a local HYSPLIT install; most
   users will only need section 1.

Source regions (shared by both sections) accept either a manual bounding
box, or exact polygon boundaries loaded from a GeoJSON/Shapefile
(``normet[geo]``) — see :func:`normet.io.trajectory.load_source_regions`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QDate, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDateEdit,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ._widgets import CanvasTab, NoWheelSpinBox, hint_label, run_button
from .workers import TaskRunner

log = logging.getLogger(__name__)


class SourceRegionTable(QTableWidget):
    """Editable ``{name: (lon_min, lat_min, lon_max, lat_max)}`` table."""

    _COLS = ["name", "lon_min", "lat_min", "lon_max", "lat_max"]

    def __init__(self) -> None:
        super().__init__(0, len(self._COLS))
        self.setHorizontalHeaderLabels(self._COLS)
        self.setToolTip(
            "Named bounding boxes; for each, the fraction of trajectory time\n"
            "spent inside is added as a traj_resid_<name> feature (a\n"
            "residence-time-over-source-region proxy)."
        )
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def add_row(self, name: str = "", box: tuple[float, float, float, float] | None = None) -> None:
        r = self.rowCount()
        self.insertRow(r)
        values = [name, *(box or (0.0, 0.0, 0.0, 0.0))]
        for c, v in enumerate(values):
            self.setItem(r, c, QTableWidgetItem(str(v)))

    def remove_selected(self) -> None:
        for idx in sorted({i.row() for i in self.selectedIndexes()}, reverse=True):
            self.removeRow(idx)

    def regions(self) -> dict[str, tuple[float, float, float, float]]:
        out: dict[str, tuple[float, float, float, float]] = {}
        for r in range(self.rowCount()):
            name_item = self.item(r, 0)
            name = name_item.text().strip() if name_item else ""
            if not name:
                continue
            try:
                box = tuple(float(self.item(r, c).text()) for c in range(1, 5))
            except (AttributeError, ValueError):
                continue
            out[name] = box  # type: ignore[assignment]
        return out


class RegionFileTable(QTableWidget):
    """Editable ``[(name, path), ...]`` list of loaded GeoJSON/Shapefile region files.

    ``name`` is user-editable (defaults to the file stem) and is only used
    when a file contains a single region — files with several named
    features keep their own per-feature names, prefixed with this one to
    avoid collisions across multiple loaded files (see ``_merged_regions``).
    """

    _COLS = ["name", "path"]

    def __init__(self) -> None:
        super().__init__(0, len(self._COLS))
        self.setHorizontalHeaderLabels(self._COLS)
        self.setToolTip(
            "Loaded region files. 'name' is used directly for single-region\n"
            "files (e.g. one administrative boundary); multi-region files\n"
            "keep their own per-feature names, prefixed with this one."
        )
        self.horizontalHeader().setStretchLastSection(True)
        self.setColumnWidth(0, 90)

    def add_row(self, name: str, path: str) -> None:
        r = self.rowCount()
        self.insertRow(r)
        self.setItem(r, 0, QTableWidgetItem(name))
        path_item = QTableWidgetItem(path)
        path_item.setFlags(path_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(r, 1, path_item)

    def remove_selected(self) -> None:
        for idx in sorted({i.row() for i in self.selectedIndexes()}, reverse=True):
            self.removeRow(idx)

    def entries(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for r in range(self.rowCount()):
            name_item, path_item = self.item(r, 0), self.item(r, 1)
            path = path_item.text().strip() if path_item else ""
            if not path:
                continue
            name = (name_item.text().strip() if name_item else "") or Path(path).stem
            out.append((name, path))
        return out


class ReceptorTable(QTableWidget):
    """Editable list of ``(name, lat, lon)`` receptor sites for HYSPLIT runs."""

    _COLS = ["name", "lat", "lon"]

    def __init__(self) -> None:
        super().__init__(0, len(self._COLS))
        self.setHorizontalHeaderLabels(self._COLS)
        self.setToolTip(
            "One row per receptor site. hyts_std runs once per (site, time),\n"
            "so results are tagged with a 'site' column when there is more\n"
            "than one row."
        )
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def add_row(self, name: str = "", lat: float = 0.0, lon: float = 0.0) -> None:
        r = self.rowCount()
        self.insertRow(r)
        for c, v in enumerate((name, lat, lon)):
            self.setItem(r, c, QTableWidgetItem(str(v)))

    def remove_selected(self) -> None:
        for idx in sorted({i.row() for i in self.selectedIndexes()}, reverse=True):
            self.removeRow(idx)

    def receptors(self) -> list[tuple[str, float, float]]:
        out: list[tuple[str, float, float]] = []
        for r in range(self.rowCount()):
            name_item = self.item(r, 0)
            name = name_item.text().strip() if name_item else ""
            if not name:
                continue
            try:
                lat = float(self.item(r, 1).text())
                lon = float(self.item(r, 2).text())
            except (AttributeError, ValueError):
                continue
            out.append((name, lat, lon))
        return out


class TrajectoryWindow(QMainWindow):
    """'Transport Studio': build (and optionally run) HYSPLIT trajectory features."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Normet — Transport Studio (HYSPLIT trajectories)")
        self.resize(1240, 820)
        self._main = parent
        self.features: pd.DataFrame | None = None
        self._tdump_paths: list[str] = []
        self._gdas_paths: list[str] = []

        self.runner = TaskRunner(self)
        self.runner.started.connect(self._task_started)
        self.runner.finished.connect(self._task_finished)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_panel())
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 4, 4, 0)
        rv.addWidget(self._build_action_bar())
        rv.addWidget(self._build_tabs(), 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 860])
        self.setCentralWidget(splitter)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setMaximumWidth(220)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._abandon)
        self.statusBar().addPermanentWidget(self.cancel_btn)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage(
            "Add tdump files (or run HYSPLIT below), then ▶ Build trajectory features."
        )
        self._sync_enabled()

    # ------------------------------------------------------------- left panel
    def _build_panel(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)

        # ---- Section 1: existing tdump files
        tdump_box = QGroupBox("1 · Trajectory files (tdump)")
        tv = QVBoxLayout(tdump_box)
        tv.addWidget(
            hint_label(
                "HYSPLIT back-trajectory output — generate separately (hyts_std,\n"
                "pysplit, splitr) or with section 2 below."
            )
        )
        self.tdump_list = QListWidget()
        self.tdump_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tdump_list.setMaximumHeight(120)
        tv.addWidget(self.tdump_list)
        btn_row = QHBoxLayout()
        b_add = QPushButton("Add files…")
        b_add.clicked.connect(self._add_tdump_files)
        b_remove = QPushButton("Remove selected")
        b_remove.clicked.connect(self._remove_tdump_selected)
        btn_row.addWidget(b_add)
        btn_row.addWidget(b_remove)
        tv.addLayout(btn_row)
        self.build_site_edit = QLineEdit()
        self.build_site_edit.setPlaceholderText("e.g. MY1 — leave blank for single-site data")
        site_form = QFormLayout()
        site_form.addRow("Site name (optional)", self.build_site_edit)
        tv.addLayout(site_form)
        tv.addWidget(
            hint_label(
                "If these tdump files are all for one receptor site, name it\n"
                "here — the result gets a 'site' column, so batches from\n"
                "several sites (processed one at a time) can all be sent to\n"
                "the main window and matched up correctly."
            )
        )
        v.addWidget(tdump_box)

        # ---- Source regions (shared by build + run)
        region_box = QGroupBox("Source regions (optional)")
        rv2 = QVBoxLayout(region_box)
        self.region_table = SourceRegionTable()
        self.region_table.setMaximumHeight(120)
        rv2.addWidget(self.region_table)
        rbtn_row = QHBoxLayout()
        b_radd = QPushButton("Add row")
        b_radd.clicked.connect(lambda: self.region_table.add_row())
        b_rdel = QPushButton("Remove row")
        b_rdel.clicked.connect(self.region_table.remove_selected)
        rbtn_row.addWidget(b_radd)
        rbtn_row.addWidget(b_rdel)
        rv2.addLayout(rbtn_row)
        self.region_file_table = RegionFileTable()
        self.region_file_table.setMaximumHeight(90)
        rv2.addWidget(self.region_file_table)
        rfbtn_row = QHBoxLayout()
        b_load_region = QPushButton("Load region file(s)…")
        b_load_region.setToolTip(
            "Exact point-in-polygon residence time instead of a bounding\n"
            "box — use real administrative or airshed boundaries. Pick\n"
            "several at once, or click again to add more; rename a row by\n"
            "double-clicking its 'name' cell."
        )
        b_load_region.clicked.connect(self._load_region_file)
        b_rdel_file = QPushButton("Remove row")
        b_rdel_file.clicked.connect(self.region_file_table.remove_selected)
        rfbtn_row.addWidget(b_load_region)
        rfbtn_row.addWidget(b_rdel_file)
        rv2.addLayout(rfbtn_row)
        self.prefix_edit = QLineEdit("traj_")
        form = QFormLayout()
        form.addRow("Column prefix", self.prefix_edit)
        rv2.addLayout(form)
        v.addWidget(region_box)

        self.build_btn = run_button(
            "▶  Build trajectory features",
            "Parse the tdump files above into one feature row per receptor\n"
            "time — transport distance/speed, inflow direction, residence\n"
            "time over the source regions, along-path rainfall/BLH.",
        )
        self.build_btn.clicked.connect(self._run_build)
        v.addWidget(self.build_btn)

        # ---- Section 2: run HYSPLIT (advanced)
        run_box = QGroupBox("2 · Run HYSPLIT (advanced, optional)")
        rf = QFormLayout(run_box)
        rf.addRow(
            hint_label(
                "Needs a local HYSPLIT install (hyts_std) and ARL-format\n"
                "meteorology. Most users only need section 1 above."
            )
        )
        self.hysplit_exec_edit = QLineEdit()
        exec_row = QHBoxLayout()
        exec_row.addWidget(self.hysplit_exec_edit)
        b_exec = QPushButton("Browse…")
        b_exec.clicked.connect(self._browse_hysplit_exec)
        exec_row.addWidget(b_exec)
        rf.addRow("hyts_std", self._wrap_row(exec_row))

        self.gdas_dir_edit = QLineEdit(os.path.expanduser("~/normet_gdas1"))
        gdas_row = QHBoxLayout()
        gdas_row.addWidget(self.gdas_dir_edit)
        b_gdas_dir = QPushButton("Browse…")
        b_gdas_dir.clicked.connect(self._browse_gdas_dir)
        gdas_row.addWidget(b_gdas_dir)
        rf.addRow("GDAS1 cache dir", self._wrap_row(gdas_row))

        today = QDate.currentDate()
        self.gdas_from = QDateEdit(today.addDays(-10))
        self.gdas_to = QDateEdit(today.addDays(-3))
        for de in (self.gdas_from, self.gdas_to):
            de.setCalendarPopup(True)
            de.setDisplayFormat("yyyy-MM-dd")
        gdas_date_row = QHBoxLayout()
        gdas_date_row.addWidget(self.gdas_from)
        gdas_date_row.addWidget(QLabel("→"))
        gdas_date_row.addWidget(self.gdas_to)
        rf.addRow("GDAS1 range", self._wrap_row(gdas_date_row))
        self.gdas_btn = QPushButton("⬇  Download GDAS1 for this range")
        self.gdas_btn.clicked.connect(self._run_download_gdas)
        rf.addRow(self.gdas_btn)
        self.gdas_hint = hint_label("No GDAS1 files downloaded yet.")
        rf.addRow(self.gdas_hint)

        rf.addRow(QLabel("Receptor sites"))
        self.receptor_table = ReceptorTable()
        self.receptor_table.setMaximumHeight(120)
        rf.addRow(self.receptor_table)
        recv_btn_row = QHBoxLayout()
        b_radd2 = QPushButton("Add row")
        b_radd2.clicked.connect(lambda: self.receptor_table.add_row())
        b_rdel2 = QPushButton("Remove row")
        b_rdel2.clicked.connect(self.receptor_table.remove_selected)
        recv_btn_row.addWidget(b_radd2)
        recv_btn_row.addWidget(b_rdel2)
        rf.addRow("", self._wrap_row(recv_btn_row))
        rf.addRow(
            hint_label(
                "One or more (name, lat, lon) rows. With more than one, the\n"
                "combined result is tagged with a 'site' column."
            )
        )
        self.height_spin = NoWheelSpinBox()
        self.height_spin.setRange(1, 20_000)
        self.height_spin.setValue(500)
        self.height_spin.setSuffix(" m agl")
        rf.addRow("Height", self.height_spin)
        self.hours_back_spin = NoWheelSpinBox()
        self.hours_back_spin.setRange(1, 240)
        self.hours_back_spin.setValue(72)
        self.hours_back_spin.setSuffix(" h")
        rf.addRow("Hours back", self.hours_back_spin)
        self.time_res_spin = NoWheelSpinBox()
        self.time_res_spin.setRange(1, 720)
        self.time_res_spin.setValue(1)
        self.time_res_spin.setSuffix(" h")
        self.time_res_spin.setToolTip(
            "Time resolution of the back-trajectory runs: compute one\n"
            "trajectory every N hours across the main window's date range\n"
            "(1 = hourly, the usual choice; raise it to cut runtime)."
        )
        rf.addRow("Compute every", self.time_res_spin)

        self.diag_checks: dict[str, QCheckBox] = {}
        diag_names = [
            ("pressure", "Pressure"),
            ("rainfall", "Rainfall"),
            ("blh", "BLH"),
            ("rh", "RH"),
            ("temp", "Temp"),
        ]
        diag_col = QVBoxLayout()
        diag_col.setContentsMargins(0, 0, 0, 0)
        for chunk_start in range(0, len(diag_names), 3):
            diag_row = QHBoxLayout()
            for name, label in diag_names[chunk_start : chunk_start + 3]:
                cb = QCheckBox(label)
                cb.setChecked(True)
                self.diag_checks[name] = cb
                diag_row.addWidget(cb)
            diag_row.addStretch(1)
            diag_col.addLayout(diag_row)
        rf.addRow("Record diagnostics", self._wrap_row(diag_col))
        rf.addRow(
            hint_label(
                "Along-trajectory meteorology hyts_std writes into tdump —\n"
                "feeds traj_rain_sum/blh_mean/rh_mean/pressure_mean/temp_mean."
            )
        )

        self.times_source = QLabel("Receptor times: main window's date range")
        self.times_source.setWordWrap(True)
        rf.addRow(self.times_source)

        self.run_btn = run_button(
            "▶  Run back-trajectories",
            "Runs hyts_std once per (site, time) — from the main window's\n"
            "date range at the chosen resolution — then builds features\n"
            "from the resulting tdump files.",
        )
        self.run_btn.clicked.connect(self._run_hysplit)
        rf.addRow(self.run_btn)
        v.addWidget(run_box)

        v.addStretch(1)
        self._run_buttons = [self.build_btn, self.gdas_btn, self.run_btn]

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(400)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    @staticmethod
    def _wrap_row(layout: QHBoxLayout | QVBoxLayout) -> QWidget:
        w = QWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)
        return w

    def _build_action_bar(self) -> QWidget:
        bar = QWidget()
        h = QHBoxLayout(bar)
        h.setContentsMargins(4, 0, 4, 0)
        self.save_btn = QPushButton("💾  Save features CSV…")
        self.save_btn.clicked.connect(self._save_csv)
        self.save_btn.setEnabled(False)
        self.send_btn = QPushButton("⬆  Send to main window")
        self.send_btn.setToolTip(
            "Join the traj_* columns onto the main window's data by date\n"
            "(nearest-hour match), ready to tick as predictors in Step 1."
        )
        self.send_btn.clicked.connect(self._send_to_main)
        self.send_btn.setEnabled(False)
        h.addWidget(self.save_btn)
        h.addWidget(self.send_btn)
        h.addStretch(1)
        return bar

    def _build_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()
        self.tab_features = CanvasTab(
            "Build trajectory features (left) to preview the transport\ndescriptors here."
        )
        self.tabs.addTab(self.tab_features, "① Features")
        return self.tabs

    # ---------------------------------------------------------------- data
    def offer_main_data(self, df: pd.DataFrame) -> None:
        """Called by the main window so receptor times can default to its dates."""
        self._main_df = df
        if len(df):
            span = f"{df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d}"
            self.times_source.setText(f"Receptor times: main window's date range ({span})")

    def _add_tdump_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add tdump files", "", "All files (*)")
        for p in paths:
            if p not in self._tdump_paths:
                self._tdump_paths.append(p)
                self.tdump_list.addItem(os.path.basename(p))
        self._sync_enabled()
        # macOS returns focus to the parent (main) window after the file
        # dialog closes; keep this window in front.
        self.raise_()
        self.activateWindow()

    def _remove_tdump_selected(self) -> None:
        for row in sorted({i.row() for i in self.tdump_list.selectedIndexes()}, reverse=True):
            self.tdump_list.takeItem(row)
            del self._tdump_paths[row]
        self._sync_enabled()

    def _browse_hysplit_exec(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Locate hyts_std", "", "All files (*)")
        if path:
            self.hysplit_exec_edit.setText(path)
        self.raise_()
        self.activateWindow()

    def _browse_gdas_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "GDAS1 cache directory")
        if path:
            self.gdas_dir_edit.setText(path)
        self.raise_()
        self.activateWindow()

    def _load_region_file(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load source region file(s)",
            "",
            "Region files (*.geojson *.json *.shp);;All files (*)",
        )
        for path in paths:
            try:
                from normet import load_source_regions

                names = list(load_source_regions(path))
            except Exception as exc:
                QMessageBox.critical(self, "Failed to load region file", f"{path}:\n{exc}")
                continue
            default_name = Path(path).stem if len(names) == 1 else names[0] if names else "region"
            self.region_file_table.add_row(default_name, path)
            self.statusBar().showMessage(
                f"{os.path.basename(path)}: {len(names)} region(s) — {', '.join(names)}"
            )
        self.raise_()
        self.activateWindow()

    def _merged_regions(self) -> dict | None:
        """Bbox rows from the table, plus any loaded GeoJSON/Shapefile polygons.

        A single-region file uses its table row's (editable) name directly;
        a multi-region file keeps its own per-feature names, prefixed with
        the row's name so several loaded files can't collide.
        """
        regions: dict = dict(self.region_table.regions())
        if self.region_file_table.rowCount():
            from normet import load_source_regions

            for name, path in self.region_file_table.entries():
                loaded = load_source_regions(path)
                if len(loaded) == 1:
                    regions[name] = next(iter(loaded.values()))
                else:
                    for feat_name, geom in loaded.items():
                        regions[f"{name}_{feat_name}"] = geom
        return regions or None

    # ----------------------------------------------------------------- sync
    def _sync_enabled(self) -> None:
        busy = self.runner.busy
        self.build_btn.setEnabled(not busy and bool(self._tdump_paths))
        self.gdas_btn.setEnabled(not busy)
        self.run_btn.setEnabled(not busy and bool(self.hysplit_exec_edit.text().strip()))
        has_features = self.features is not None
        self.save_btn.setEnabled(has_features)
        self.send_btn.setEnabled(has_features and self._main is not None)
        self.cancel_btn.setEnabled(busy)

    def _task_started(self, name: str) -> None:
        self.progress.setRange(0, 0)
        self.statusBar().showMessage(f"Running: {name}…")
        for b in self._run_buttons:
            b.setEnabled(False)
        self.cancel_btn.setEnabled(True)

    def _task_finished(self) -> None:
        self.progress.setRange(0, 1)
        self._sync_enabled()

    def _abandon(self) -> None:
        if self.runner.busy:
            self.runner.abandon()
            self.statusBar().showMessage(
                "Task abandoned — it finishes in the background and its result is discarded."
            )

    def _show_error(self, tb: str) -> None:
        QMessageBox.critical(self, "Task failed", tb.splitlines()[-1] if tb else "Unknown error")
        log.error("%s", tb)

    # -------------------------------------------------------------- features
    def _run_build(self) -> None:
        if not self._tdump_paths:
            QMessageBox.information(self, "No tdump files", "Add at least one tdump file.")
            return
        from normet import build_trajectory_features

        self._pending_site = self.build_site_edit.text().strip()
        self.runner.submit(
            "build trajectory features",
            build_trajectory_features,
            self._on_build_result,
            self._show_error,
            list(self._tdump_paths),
            source_regions=self._merged_regions(),
            prefix=self.prefix_edit.text() or "traj_",
        )

    def _on_build_result(self, result: pd.DataFrame) -> None:
        if result is not None and not result.empty and self._pending_site:
            result = result.copy()
            result.insert(0, "site", self._pending_site)
        self._show_features(result)

    def _show_features(self, result: pd.DataFrame | None) -> None:
        if result is None or result.empty:
            self.tab_features.show_result(
                None,
                verdict=("warn", "No trajectory features were produced — check the inputs."),
            )
            self.features = None
            self._sync_enabled()
            return
        self.features = result
        self._draw_features(result)
        self._sync_enabled()
        n_sites = result["site"].nunique() if "site" in result.columns else 1
        site_note = f" across {n_sites} site(s)" if n_sites > 1 else ""
        self.statusBar().showMessage(
            f"Built {result.shape[1]} transport features for {len(result):,} receptor times{site_note}."
        )
        log.info("Built trajectory features: %d rows x %d cols", len(result), result.shape[1])

    def _draw_features(self, result: pd.DataFrame) -> None:
        import matplotlib.pyplot as plt

        plot_cols = [
            c for c in result.columns if c.endswith(("dist_km", "speed_kmh", "inflow_deg"))
        ]
        n = max(1, len(plot_cols))
        fig, axes = plt.subplots(n, 1, figsize=(9, 2.0 * n), sharex=True, squeeze=False)
        has_sites = "site" in result.columns and result["site"].nunique() > 1
        groups = list(result.groupby("site")) if has_sites else [(None, result)]
        cmap = plt.get_cmap("tab10")
        for ax, col in zip(axes.ravel(), plot_cols or [None], strict=False):
            if col is None:
                ax.axis("off")
                continue
            for i, (site, sub) in enumerate(groups):
                ax.plot(
                    sub.index,
                    sub[col],
                    lw=0.8,
                    color=cmap(i % 10),
                    label=site if has_sites else None,
                )
            ax.set_title(col, fontsize=9, loc="left")
            ax.grid(alpha=0.2)
            if has_sites and col == (plot_cols or [None])[0]:
                ax.legend(fontsize=7, ncols=min(4, len(groups)))
        fig.tight_layout()
        site_note = f" ({result['site'].nunique()} sites)" if has_sites else ""
        self.tab_features.show_result(
            fig,
            verdict=(
                "ok",
                f"{result.shape[1]} transport features × {len(result):,} rows{site_note}.",
            ),
            lines=[f"Columns: {', '.join(result.columns)}"],
        )

    def _save_csv(self) -> None:
        if self.features is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save trajectory features", "normet_traj_features.csv", "CSV (*.csv)"
        )
        if path:
            self.features.to_csv(path)
            self.statusBar().showMessage(f"Saved {os.path.basename(path)}")
        self.raise_()
        self.activateWindow()

    def _send_to_main(self) -> None:
        if self.features is None or self._main is None:
            return
        self._main.merge_trajectory_features(self.features)
        self._main.raise_()
        self._main.activateWindow()
        self.statusBar().showMessage("Sent to the main window — features joined onto its data.")

    # ------------------------------------------------------------------ HYSPLIT
    def _run_download_gdas(self) -> None:
        cache_dir = self.gdas_dir_edit.text().strip()
        if not cache_dir:
            QMessageBox.information(self, "No cache directory", "Pick a GDAS1 cache directory.")
            return
        from normet import fetch_gdas1

        self.runner.submit(
            "download GDAS1",
            fetch_gdas1,
            self._gdas_done,
            self._show_error,
            self.gdas_from.date().toString("yyyy-MM-dd"),
            self.gdas_to.date().toString("yyyy-MM-dd"),
            cache_dir,
            on_missing="warn",
        )

    def _gdas_done(self, paths: list[str]) -> None:
        self._gdas_paths = paths
        self.gdas_hint.setText(
            f"{len(paths)} GDAS1 file(s) ready: {', '.join(os.path.basename(p) for p in paths)}"
        )
        self.statusBar().showMessage(f"Downloaded/cached {len(paths)} GDAS1 file(s).")

    def _run_hysplit(self) -> None:
        exe = self.hysplit_exec_edit.text().strip()
        if not exe:
            QMessageBox.information(
                self, "No hyts_std", "Point 'hyts_std' at your HYSPLIT install."
            )
            return
        if not self._gdas_paths:
            QMessageBox.information(
                self,
                "No meteorology",
                "Download GDAS1 files first (or point to existing ARL met files).",
            )
            return
        receptors = self.receptor_table.receptors()
        if not receptors:
            QMessageBox.information(
                self, "No receptors", "Add at least one receptor site (name, lat, lon)."
            )
            return
        main_df = getattr(self, "_main_df", None)
        if main_df is None or "date" not in main_df.columns:
            QMessageBox.information(
                self,
                "No receptor times",
                "Load data in the main window first — its dates set the receptor time range.",
            )
            return
        dates = pd.to_datetime(main_df["date"])
        step = self.time_res_spin.value()
        times = list(pd.date_range(dates.min().floor("h"), dates.max().ceil("h"), freq=f"{step}h"))
        if not times:
            QMessageBox.information(
                self, "No receptor times", "The main window's date range is empty."
            )
            return

        from normet import run_back_trajectories

        met_files = list(self._gdas_paths)
        height_m = float(self.height_spin.value())
        hours_back = self.hours_back_spin.value()
        source_regions = self._merged_regions()
        prefix = self.prefix_edit.text() or "traj_"
        diagnostics = [name for name, cb in self.diag_checks.items() if cb.isChecked()]

        def job() -> pd.DataFrame:
            frames = []
            for name, lat, lon in receptors:
                res = run_back_trajectories(
                    times,
                    lat,
                    lon,
                    met_files=met_files,
                    hysplit_exec=exe,
                    height_m=height_m,
                    hours_back=hours_back,
                    diagnostics=diagnostics,
                    source_regions=source_regions,
                    prefix=prefix,
                )
                if res is not None and not res.empty:
                    res = res.copy()
                    res.insert(0, "site", name)
                    frames.append(res)
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames) if len(frames) > 1 else frames[0]

        self.runner.submit(
            f"run HYSPLIT back-trajectories ({len(receptors)} site(s), {len(times)} time(s))",
            job,
            self._show_features,
            self._show_error,
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        self.runner.shutdown()
        super().closeEvent(event)
