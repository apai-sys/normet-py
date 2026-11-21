"""Transport Studio — HYSPLIT back-trajectory (transport-aware) features.

Turns HYSPLIT ``tdump`` back-trajectory output into per-receptor transport
descriptors (inflow direction, transport distance/speed, residence time over
named source regions, along-path rainfall/boundary-layer height) that join
straight onto a date-indexed panel as extra predictors — see
:mod:`normet.io.trajectory`.

Two pages, as the main window's and the SCM Studio's steps (._pages.StepPanel):

1. **Trajectories: build features from tdump files** — parse existing HYSPLIT output. No
   external binary required. An optional site name tags the result with a
   'site' column, so batches from several receptors (processed one at a
   time) can all be sent to the main window and matched up correctly.
2. **HYSPLIT (advanced)** — orchestrate ``hyts_std`` itself for one or
   more receptor sites, at a configurable time resolution, with GDAS1
   meteorology it downloads or ARL files already on disk. Needs a local
   HYSPLIT install; most users will only need the first page.

Source regions (on the first page, used by both) accept either a manual bounding
box, or exact polygon boundaries loaded from a GeoJSON/Shapefile
(``normet[geo]``) — see :func:`normet.io.trajectory.load_source_regions`.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

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
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from normet import style as S
from normet.io.trajectory import _missing_gdas1_files, _naive_utc

from ._clock import TaskClock
from ._widgets import (
    CanvasTab,
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
    SmoothProgress,
    hint_label,
    run_button,
    wrap_row,
)
from .workers import TaskRunner

log = logging.getLogger(__name__)

# A weekly GDAS1 file is ~570 MB: past this many, a download is asked about first.
_ASK_ABOVE_FILES = 4
_GDAS1_FILE_GB = 0.57


def _n_features(result: pd.DataFrame) -> int:
    """Feature columns in a result -- its 'site' tag is not one."""
    return result.shape[1] - int("site" in result.columns)


class SourceRegionTable(QTableWidget):
    """Editable ``{name: (lon_min, lat_min, lon_max, lat_max)}`` table."""

    _COLS = ["name", "lon_min", "lat_min", "lon_max", "lat_max"]
    #: The columns as the table heads them: in words, translated as the window is.
    _HEADS = ["Name", "Lon min", "Lat min", "Lon max", "Lat max"]

    def __init__(self) -> None:
        super().__init__(0, len(self._COLS))
        self.setHorizontalHeaderLabels(self._HEADS)
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

    def regions(
        self, *, allow_dateline_wrap: bool = False
    ) -> dict[str, tuple[float, float, float, float]]:
        """The boxes by name. A row left as added (no name, zeros) is skipped; any
        other row that is not a box raises ``ValueError`` naming it."""
        out: dict[str, tuple[float, float, float, float]] = {}
        for r in range(self.rowCount()):
            name, *nums = _cells(self, r)
            if not name and not any(_filled(v) for v in nums):
                continue
            where = f"Source region row {r + 1}" + (f" ({name})" if name else "")
            if not name:
                raise ValueError(f"{where}: give it a name.")
            try:
                lon0, lat0, lon1, lat1 = (float(v) for v in nums)
            except ValueError:
                raise ValueError(
                    f"{where}: lon_min, lat_min, lon_max and lat_max must be numbers."
                ) from None
            if not (lat0 < lat1):
                raise ValueError(
                    f"{where}: lon_min must be less than lon_max, and lat_min than lat_max."
                )
            if allow_dateline_wrap:
                if lon0 == lon1:
                    raise ValueError(f"{where}: lon_min and lon_max cannot be equal.")
            elif not (lon0 < lon1):
                raise ValueError(
                    f"{where}: lon_min must be less than lon_max, and lat_min than lat_max."
                )
            if lat0 < -90 or lat1 > 90 or lon0 < -180 or lon1 > 180:
                raise ValueError(f"{where}: latitudes run -90 to 90, longitudes -180 to 180.")
            if name in out:
                raise ValueError(f"{where}: another row is called {name!r} already.")
            out[name] = (lon0, lat0, lon1, lat1)
        return out


def _cells(table: QTableWidget, row: int) -> list[str]:
    return [
        (item.text().strip() if (item := table.item(row, c)) is not None else "")
        for c in range(table.columnCount())
    ]


def _filled(value: str) -> bool:
    """Whether a cell holds more than the 0 a new row starts with."""
    try:
        return float(value) != 0
    except ValueError:
        return bool(value)


class RegionFileTable(QTableWidget):
    """Editable ``[(name, path), ...]`` list of loaded GeoJSON/Shapefile region files.

    ``name`` is user-editable (defaults to the file stem) and is only used
    when a file contains a single region — files with several named
    features keep their own per-feature names, prefixed with this one to
    avoid collisions across multiple loaded files (see ``_merged_regions``).
    """

    _COLS = ["name", "path"]
    _HEADS = ["Name", "File"]

    def __init__(self) -> None:
        super().__init__(0, len(self._COLS))
        self.setHorizontalHeaderLabels(self._HEADS)
        self.setToolTip(
            "Loaded region files. 'Name' is used directly for single-region\n"
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
    _HEADS = ["Name", "Lat", "Lon"]

    def __init__(self) -> None:
        super().__init__(0, len(self._COLS))
        self.setHorizontalHeaderLabels(self._HEADS)
        self.setToolTip(
            "One row per receptor site. hyts_std runs once per (site, time),\n"
            "and each result is tagged with the row's name in a 'site' column."
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
        """``(name, lat, lon)`` per site. A row left as added is skipped; any other
        row that is not a place raises ``ValueError`` naming it."""
        out: list[tuple[str, float, float]] = []
        for r in range(self.rowCount()):
            name, *nums = _cells(self, r)
            if not name and not any(_filled(v) for v in nums):
                continue
            where = f"Receptor row {r + 1}" + (f" ({name})" if name else "")
            if not name:
                raise ValueError(f"{where}: give it a name; it tags the site's results.")
            try:
                lat, lon = (float(v) for v in nums)
            except ValueError:
                raise ValueError(f"{where}: lat and lon must be numbers.") from None
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError(f"{where}: lat runs -90 to 90, lon -180 to 180.")
            if any(name == n for n, _, _ in out):
                raise ValueError(f"{where}: another row is called {name!r} already.")
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
        self._last_tdump_paths: list[str] = []
        self._work_dirs: list[Path] = []
        self._gdas_paths: list[str] = []
        #: Set to stop the HYSPLIT run in progress (Cancel, or closing the window).
        self._hysplit_stop: threading.Event | None = None

        self.runner = TaskRunner(self)
        self.runner.started.connect(self._task_started)
        self.runner.finished.connect(self._task_finished)
        self.task_clock = TaskClock(self, self.runner)

        self._build_menu()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_panel())
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 4, 4, 0)
        rv.addWidget(self._build_action_bar())
        from .notices import NoticeStack

        self.notices = NoticeStack()
        rv.addWidget(self.notices)
        rv.addWidget(self._build_tabs(), 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([self.panel.least_width, 860])  # no gap: see StepPanel.least_width
        self.setCentralWidget(splitter)

        self.progress = SmoothProgress()
        self.progress.layout().setContentsMargins(0, 0, 8, 0)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._abandon)
        self.statusBar().addPermanentWidget(self.cancel_btn)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage(
            "Add tdump files (or run HYSPLIT on its page), then ▶ Build trajectory features."
        )
        self._sync_enabled()

    def _build_menu(self) -> None:
        """File: send features to the main window, close."""
        fm = self.menuBar().addMenu("&File")
        fm.addAction("Send features to main window", self._send_to_main).setShortcut("Ctrl+Return")
        fm.addSeparator()
        fm.addAction("&Close", self.close).setShortcut("Ctrl+W")

    # ------------------------------------------------------------- left panel
    def _build_panel(self) -> QWidget:
        # ---- The Trajectories page: existing tdump files, source regions, build
        tdump_box = QGroupBox()
        tv = QVBoxLayout(tdump_box)
        tv.addWidget(
            hint_label(
                "HYSPLIT back-trajectory output — generate separately (hyts_std, "
                "pysplit, splitr) or on the HYSPLIT page."
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
                "If these tdump files are all for one receptor site, name it "
                "as the main data's 'site' column does: each site's batch, "
                "built and sent one at a time, then fills only its own rows."
            )
        )
        self.build_min_check = QCheckBox("Null trajectories shorter than")
        self.build_min_spin = NoWheelSpinBox()
        self.build_min_spin.setRange(1, 480)
        self.build_min_spin.setValue(72)
        self.build_min_spin.setSuffix(" h")
        self.build_min_spin.setEnabled(False)
        self.build_min_check.toggled.connect(self.build_min_spin.setEnabled)
        self.build_min_check.setToolTip(
            "A trajectory that reaches back less than this -- the met data ran\n"
            "out, or it left the domain -- gets NaN features instead of passing\n"
            "for a short-range one (traj_age_max_h says how far each reached)."
        )
        min_row = QHBoxLayout()
        min_row.addWidget(self.build_min_check)
        min_row.addWidget(self.build_min_spin)
        min_row.addStretch(1)
        tv.addWidget(wrap_row(min_row))

        phys_form = QFormLayout()
        self.blh_coupled_check = QCheckBox("BLH-coupled footprint")
        self.blh_coupled_check.setChecked(True)
        self.blh_coupled_check.setToolTip(
            "Gate exposure by boundary-layer height (or 1500 m fallback when BLH is absent)\n"
            "and scale sensitivity as 1000 / max(BLH, 100 m)."
        )
        phys_form.addRow(self.blh_coupled_check)
        self.decay_spin = NoWheelSpinBox()
        self.decay_spin.setRange(0, 720)
        self.decay_spin.setValue(0)
        self.decay_spin.setSpecialValueText("off")
        self.decay_spin.setSuffix(" h")
        self.decay_spin.setToolTip(
            "Optional first-order chemical/deposition decay lifetime (e-folding hours)."
        )
        phys_form.addRow("Decay lifetime", self.decay_spin)
        self.scav_spin = NoWheelDoubleSpinBox()
        self.scav_spin.setRange(0.0, 2.0)
        self.scav_spin.setSingleStep(0.05)
        self.scav_spin.setValue(0.0)
        self.scav_spin.setSpecialValueText("off")
        self.scav_spin.setSuffix(" /mm")
        self.scav_spin.setToolTip(
            "Causal wet-scavenging coefficient applied to downstream rainfall strictly\n"
            "between each upstream point and the receptor."
        )
        phys_form.addRow("Wet scavenging", self.scav_spin)
        self.horizons_check = QCheckBox("Multi-scale horizons (0–12 / 12–36 / 36–72 h)")
        self.horizons_check.setToolTip(
            "Also emit separate near/regional/long-range horizon slices."
        )
        phys_form.addRow(self.horizons_check)
        tv.addLayout(phys_form)

        # ---- Source regions (shared by build + run + analysis)
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
        self.dateline_check = QCheckBox("Allow dateline wrap (lon_min > lon_max)")
        rv2.addWidget(self.dateline_check)
        self.region_file_table = RegionFileTable()
        self.region_file_table.setMaximumHeight(90)
        rv2.addWidget(self.region_file_table)
        rfbtn_row = QHBoxLayout()
        b_load_region = QPushButton("Load region file(s)…")
        b_load_region.setToolTip(
            "Exact point-in-polygon residence time instead of a bounding\n"
            "box — use real administrative or airshed boundaries. Pick\n"
            "several at once, or click again to add more; rename a row by\n"
            "double-clicking its 'Name' cell."
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

        self.build_btn = run_button(
            "▶  Build trajectory features",
            "Parse the tdump files above into one feature row per receptor\n"
            "time — transport distance/speed, inflow direction, residence\n"
            "time over the source regions, along-path rainfall/BLH.",
        )
        self.build_btn.clicked.connect(self._run_build)
        build_box = QGroupBox()
        build_box.setFlat(True)
        QVBoxLayout(build_box).addWidget(self.build_btn)

        # ---- The HYSPLIT page (advanced): run the trajectories here
        run_box = QGroupBox()
        rf = QFormLayout(run_box)
        rf.addRow(
            hint_label(
                "Needs a local HYSPLIT install (hyts_std) and ARL-format "
                "meteorology. Most users only need the Trajectories page. "
                "Its source regions go into the features built here too."
            )
        )
        self.hysplit_exec_edit = QLineEdit()
        exec_row = QHBoxLayout()
        exec_row.addWidget(self.hysplit_exec_edit)
        b_exec = QPushButton("Browse…")
        b_exec.clicked.connect(self._browse_hysplit_exec)
        exec_row.addWidget(b_exec)
        rf.addRow("hyts_std", wrap_row(exec_row))

        self.gdas_dir_edit = QLineEdit(os.path.expanduser("~/normet_gdas1"))
        gdas_row = QHBoxLayout()
        gdas_row.addWidget(self.gdas_dir_edit)
        b_gdas_dir = QPushButton("Browse…")
        b_gdas_dir.clicked.connect(self._browse_gdas_dir)
        gdas_row.addWidget(b_gdas_dir)
        rf.addRow("GDAS1 cache dir", wrap_row(gdas_row))

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
        # Its name above it, as Receptor sites' and Record diagnostics': beside it, the
        # two dates held the page wider than the panel's other page.
        rf.addRow(QLabel("GDAS1 range"))
        rf.addRow(wrap_row(gdas_date_row))
        self.gdas_btn = QPushButton("⬇  Download GDAS1 for this range")
        self.gdas_btn.clicked.connect(self._run_download_gdas)
        self.met_files_btn = QPushButton("Use existing met files…")
        self.met_files_btn.setToolTip(
            "ARL-format meteorology already on disk (GDAS1 weekly files, or\n"
            "any other ARL archive) instead of downloading."
        )
        self.met_files_btn.clicked.connect(self._choose_met_files)
        # One under the other: side by side they held the panel ~420 px wide.
        rf.addRow(self.gdas_btn)
        rf.addRow(self.met_files_btn)
        self.gdas_hint = hint_label("No met files yet: download GDAS1, or use existing ones.")
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
        rf.addRow("", wrap_row(recv_btn_row))
        rf.addRow(
            hint_label(
                "One or more (Name, Lat, Lon) rows. Each name tags its results "
                "in a 'site' column: name them as the main data's 'site' does."
            )
        )
        self.height_spin = NoWheelSpinBox()
        self.height_spin.setRange(1, 20_000)
        self.height_spin.setValue(500)
        self.height_spin.setSuffix(" m agl")
        rf.addRow("Height", self.height_spin)
        self.heights_edit = QLineEdit()
        self.heights_edit.setPlaceholderText("e.g. 100, 500, 1000 — blank uses Height")
        self.heights_edit.setToolTip(
            "Optional comma-separated start heights (m AGL) run together in one CONTROL pass."
        )
        rf.addRow("Multi-heights", self.heights_edit)
        self.workers_spin = NoWheelSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(1)
        self.workers_spin.setToolTip("Parallel worker count for HYSPLIT runs across timestamps.")
        rf.addRow("Workers", self.workers_spin)
        self.hours_back_spin = NoWheelSpinBox()
        self.hours_back_spin.setRange(1, 240)
        self.hours_back_spin.setValue(72)
        self.hours_back_spin.setSuffix(" h")
        self.hours_back_spin.valueChanged.connect(self._suggest_gdas_range)
        rf.addRow("Hours back", self.hours_back_spin)
        self.run_min_check = QCheckBox("Null trajectories cut short")
        self.run_min_check.setChecked(True)
        self.run_min_check.setToolTip(
            "A trajectory HYSPLIT ended before 'Hours back' -- the met data ran\n"
            "out, or it left the domain -- gets NaN features instead of passing\n"
            "for a short-range one (traj_age_max_h says how far each reached)."
        )
        rf.addRow("", self.run_min_check)
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
        rf.addRow(QLabel("Record diagnostics"))
        rf.addRow(wrap_row(diag_col))
        rf.addRow(
            hint_label(
                "Along-trajectory meteorology hyts_std writes into tdump — "
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

        # ---- Analysis page (WPSCF/WCWT, clustering, source footprint inversion)
        an_box = QGroupBox("Receptor & source attribution")
        af = QFormLayout(an_box)
        self.receptor_source_combo = NoWheelComboBox()
        self.receptor_source_combo.addItems(
            ["auto (de-weathered if run)", "observed (main target)", "de-weathered (normalised)"]
        )
        af.addRow("Target series", self.receptor_source_combo)
        self.receptor_method = NoWheelComboBox()
        self.receptor_method.addItems(["wcwt", "wpscf"])
        af.addRow("Receptor method", self.receptor_method)
        self.grid_res_spin = NoWheelDoubleSpinBox()
        self.grid_res_spin.setRange(0.1, 5.0)
        self.grid_res_spin.setSingleStep(0.1)
        self.grid_res_spin.setValue(0.5)
        self.grid_res_spin.setSuffix(" °")
        af.addRow("Grid resolution", self.grid_res_spin)
        self.receptor_btn = run_button(
            "▶  Run WPSCF / WCWT map",
            "Map target (or de-weathered) concentrations along BLH-weighted trajectories.",
        )
        self.receptor_btn.clicked.connect(self._run_receptor_grid)
        af.addRow(self.receptor_btn)
        self.cluster_k_spin = NoWheelSpinBox()
        self.cluster_k_spin.setRange(0, 12)
        self.cluster_k_spin.setValue(0)
        self.cluster_k_spin.setSpecialValueText("auto")
        self.cluster_k_spin.setToolTip("Cluster count k (0 = automatic TSV elbow selection).")
        af.addRow("Clusters (k)", self.cluster_k_spin)
        self.cluster_btn = run_button(
            "▶  Cluster trajectories",
            "Group back-trajectories in tangent-plane space and plot centroids.",
        )
        self.cluster_btn.clicked.connect(self._run_clusters)
        af.addRow(self.cluster_btn)
        self.inv_btn = run_button(
            "▶  Invert source footprints",
            "Fit non-negative regularized linear inversion from region footprints to target.",
        )
        self.inv_btn.clicked.connect(self._run_inversion)
        af.addRow(self.inv_btn)

        self._run_buttons = [
            self.build_btn,
            self.gdas_btn,
            self.met_files_btn,
            self.run_btn,
            self.receptor_btn,
            self.cluster_btn,
            self.inv_btn,
        ]

        # A page per step, as the main window's (._pages.StepPanel): as wide as its
        # widest page needs, and the optional HYSPLIT run out of the way.
        from ._pages import StepPanel

        self.panel = StepPanel(
            [
                ("Trajectories", (tdump_box, region_box, build_box)),
                ("HYSPLIT", (run_box,)),
                ("Analysis", (an_box,)),
            ]
        )
        return self.panel

    def _build_action_bar(self) -> QWidget:
        bar = QWidget()
        h = QHBoxLayout(bar)
        h.setContentsMargins(4, 0, 4, 0)
        self.save_btn = QPushButton("💾  Save features CSV…")
        self.save_btn.clicked.connect(self._save_csv)
        self.save_btn.setEnabled(False)
        self.send_btn = QPushButton("⬆  Send to main window")
        self.send_btn.setToolTip(
            "Join the traj_* columns onto the main window's data: each row takes\n"
            "the latest trajectory to have arrived, within one release interval\n"
            "(a row with none stays empty), ready to tick as predictors."
        )
        self.send_btn.clicked.connect(self._send_to_main)
        self.send_btn.setEnabled(False)
        h.addWidget(self.save_btn)
        h.addWidget(self.send_btn)
        h.addStretch(1)
        return bar

    def _build_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()
        # Pages on the window's own ground, as the toolbars above their figures are: a
        # framed pane drew them lighter, and each toolbar as a grey band across it.
        self.tabs.setDocumentMode(True)
        self.tab_features = CanvasTab(
            "Build trajectory features (left) to preview the transport\ndescriptors here."
        )
        self.tab_receptor = CanvasTab(
            "Run WPSCF / WCWT map (Analysis page) to map regional source probabilities."
        )
        self.tab_clusters = CanvasTab(
            "Cluster trajectories (Analysis page) to inspect air-mass pathways."
        )
        self.tab_inversion = CanvasTab(
            "Invert source footprints (Analysis page) to estimate region contributions."
        )
        self.tabs.addTab(self.tab_features, "Features")
        self.tabs.addTab(self.tab_receptor, "Receptor Map")
        self.tabs.addTab(self.tab_clusters, "Clusters")
        self.tabs.addTab(self.tab_inversion, "Footprint Inversion")
        return self.tabs

    # ---------------------------------------------------------------- data
    def offer_main_data(self, df: pd.DataFrame) -> None:
        """Called by the main window so receptor times can default to its dates."""
        self._main_df = df
        if len(df):
            span = f"{df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d}"
            self.times_source.setText(f"Receptor times: main window's date range ({span})")
        self._suggest_gdas_range()

    def _receptor_span(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """First and last receptor time (UTC), from the main window's dates."""
        df = getattr(self, "_main_df", None)
        if df is None or "date" not in df.columns:
            return None
        dates = _naive_utc(df["date"]).dropna()
        return (dates.min(), dates.max()) if len(dates) else None

    def _suggest_gdas_range(self) -> None:
        """Set the GDAS1 range to what the receptor times need: back from the first
        one by 'Hours back', up to the last."""
        span = self._receptor_span()
        if span is None:
            return
        start = span[0] - pd.Timedelta(hours=self.hours_back_spin.value())
        self.gdas_from.setDate(QDate(start.year, start.month, start.day))
        self.gdas_to.setDate(QDate(span[1].year, span[1].month, span[1].day))

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
                self.notices.notify(
                    "region-load-error",
                    f"Failed to load region file {path}: {exc}",
                    kind="error",
                )
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
        regions: dict = dict(
            self.region_table.regions(allow_dateline_wrap=self.dateline_check.isChecked())
        )
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

    def _physics_kwargs(self) -> dict[str, Any]:
        """Current Layer-1 physical weighting & horizon options."""
        decay_v = self.decay_spin.value()
        scav_v = float(self.scav_spin.value())
        return {
            "blh_coupled": self.blh_coupled_check.isChecked(),
            "decay_hours": float(decay_v) if decay_v > 0 else None,
            "scavenging_coeff": scav_v if scav_v > 0.0 else None,
            "horizons_h": ((0, 12), (12, 36), (36, 72))
            if self.horizons_check.isChecked()
            else None,
        }

    def _active_tdumps(self) -> list[str]:
        return list(self._tdump_paths) or list(self._last_tdump_paths)

    # ----------------------------------------------------------------- sync
    def _sync_enabled(self) -> None:
        busy = self.runner.busy
        self.build_btn.setEnabled(not busy and bool(self._tdump_paths))
        self.gdas_btn.setEnabled(not busy)
        self.run_btn.setEnabled(not busy and bool(self.hysplit_exec_edit.text().strip()))
        has_tdumps = bool(self._active_tdumps())
        self.receptor_btn.setEnabled(not busy and has_tdumps)
        self.cluster_btn.setEnabled(not busy and has_tdumps)
        self.inv_btn.setEnabled(not busy and has_tdumps)
        has_features = self.features is not None
        self.save_btn.setEnabled(has_features)
        self.send_btn.setEnabled(has_features and self._main is not None)
        self.cancel_btn.setEnabled(busy)

    def _task_started(self, name: str) -> None:
        for b in self._run_buttons:
            b.setEnabled(False)
        self.cancel_btn.setEnabled(True)

    def _task_finished(self) -> None:
        self._hysplit_stop = None
        self._sync_enabled()

    def _abandon(self) -> None:
        if not self.runner.busy:
            return
        stop = self._hysplit_stop
        if stop is not None:
            stop.set()  # kills the hyts_std run in progress; no further run starts
        self.runner.abandon()
        self.statusBar().showMessage(
            "Cancelled — the HYSPLIT run in progress was stopped."
            if stop is not None
            else "Task abandoned — it finishes in the background and its result is discarded."
        )

    def _show_error(self, tb: str) -> None:
        self.notices.notify(
            "task-error",
            f"Task failed: {tb.splitlines()[-1] if tb else 'Unknown error'}",
            kind="error",
        )
        log.error("%s", tb)

    # -------------------------------------------------------------- features
    def _run_build(self) -> None:
        if not self._tdump_paths:
            self.notices.notify("no-tdump", "Add at least one tdump file.")
            return
        from normet import build_trajectory_features

        ok, regions = self._regions_or_warn()
        if not ok:
            return
        self._pending_site = self.build_site_edit.text().strip()
        phys = self._physics_kwargs()
        self.runner.submit(
            "build trajectory features",
            build_trajectory_features,
            self._on_build_result,
            self._show_error,
            list(self._tdump_paths),
            source_regions=regions,
            prefix=self.prefix_edit.text() or "traj_",
            min_hours=self.build_min_spin.value() if self.build_min_check.isChecked() else None,
            **phys,
        )

    def _regions_or_warn(self) -> tuple[bool, dict | None]:
        """``(True, regions)``, or ``(False, None)`` once the user is told what is wrong."""
        try:
            return True, self._merged_regions()
        except Exception as exc:  # a bad row, or a region file that will not load
            self.notices.notify("bad-regions", f"Source regions: {exc}", kind="warn")
            return False, None

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
        n_feats = _n_features(result)
        self.statusBar().showMessage(
            f"Built {n_feats} transport features for {len(result):,} receptor times{site_note}."
        )
        log.info("Built trajectory features: %d rows x %d features", len(result), n_feats)

    def _draw_features(self, result: pd.DataFrame) -> None:
        import matplotlib.pyplot as plt

        plot_cols = [
            c
            for c in result.columns
            if c.endswith(("dist_km", "speed_kmh", "straightness", "inflow_deg"))
        ]
        n = max(1, len(plot_cols))
        fig, axes = plt.subplots(n, 1, figsize=(8, 1.5 * n + 0.4), sharex=True, squeeze=False)
        has_sites = "site" in result.columns and result["site"].nunique() > 1
        groups = list(result.groupby("site")) if has_sites else [(None, result)]
        # Few receptor times: each one marked, the line only joining them.
        marker = "o" if len(result) <= 200 else None
        shown = []
        for ax, col in zip(axes.ravel(), plot_cols or [None], strict=False):
            if col is None:
                ax.axis("off")
                continue
            for i, (site, sub) in enumerate(groups):
                ax.plot(
                    sub.index,
                    sub[col],
                    lw=0.8,
                    marker=marker,
                    ms=2.5,
                    color=S.PALETTE[i % len(S.PALETTE)],
                    label=site if has_sites else None,
                )
            ax.set_title(col)
            if col.endswith("inflow_deg"):  # a bearing: read against the compass
                ax.set_ylim(0, 360)
                ax.set_yticks([0, 90, 180, 270, 360], ["N", "E", "S", "W", "N"])
            S.grid(ax)
            if has_sites and col == (plot_cols or [None])[0]:
                S.legend(ax, loc="upper right", ncols=min(4, len(groups)))
            shown.append(ax)
        if shown and isinstance(result.index, pd.DatetimeIndex):
            S.format_time_axis(shown[-1])
        S.outer_ticks(shown)
        site_note = f" ({result['site'].nunique()} sites)" if has_sites else ""
        self.tab_features.show_result(
            fig,
            verdict=(
                "ok",
                f"{_n_features(result)} transport features × {len(result):,} rows{site_note}.",
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
        n = self._main.merge_trajectory_features(self.features)
        if not n:
            self.statusBar().showMessage("Nothing joined — the main window says why.")
            return
        self._main.raise_()
        self._main.activateWindow()
        self.statusBar().showMessage(f"Joined onto {n:,} row(s) of the main window's data.")

    # ------------------------------------------------------------------ HYSPLIT
    def _run_download_gdas(self) -> None:
        cache_dir = self.gdas_dir_edit.text().strip()
        if not cache_dir:
            self.notices.notify("no-cache", "Pick a GDAS1 cache directory.")
            return
        from normet import fetch_gdas1, gdas1_filenames

        date_from = self.gdas_from.date().toString("yyyy-MM-dd")
        date_to = self.gdas_to.date().toString("yyyy-MM-dd")
        names = gdas1_filenames(date_from, date_to)
        todo = [n for n in names if not (Path(cache_dir).expanduser() / n).exists()]
        if len(todo) > _ASK_ABOVE_FILES:
            # A several-GB download asks for a yes, not a notice.
            ans = QMessageBox.question(
                self,
                "Download GDAS1",
                f"{len(todo)} weekly GDAS1 file(s) to download, about "
                f"{_GDAS1_FILE_GB * len(todo):.0f} GB ({len(names) - len(todo)} already in "
                "the cache). Go ahead?",
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
        self.runner.submit(
            "download GDAS1",
            fetch_gdas1,
            self._gdas_done,
            self._show_error,
            date_from,
            date_to,
            cache_dir,
            on_missing="warn",
        )

    def _gdas_done(self, paths: list[str]) -> None:
        self._set_met_files(paths, "GDAS1 file(s) ready")
        self.statusBar().showMessage(f"Downloaded/cached {len(paths)} GDAS1 file(s).")

    def _choose_met_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Use existing ARL met files", self.gdas_dir_edit.text(), "All files (*)"
        )
        if paths:
            self._set_met_files(sorted(paths), "met file(s) chosen")
        self.raise_()
        self.activateWindow()

    def _set_met_files(self, paths: list[str], what: str) -> None:
        self._gdas_paths = list(paths)
        self.gdas_hint.setText(
            f"{len(paths)} {what}: {', '.join(os.path.basename(p) for p in paths)}"
        )

    def _run_hysplit(self) -> None:
        exe = self.hysplit_exec_edit.text().strip()
        if not exe:
            self.notices.notify("no-hyts", "Point 'hyts_std' at your HYSPLIT install.")
            return
        if not self._gdas_paths:
            self.notices.notify(
                "no-met",
                "Download GDAS1 files first, or use existing ARL met files.",
            )
            return
        try:
            receptors = self.receptor_table.receptors()
        except ValueError as exc:
            self.notices.notify("bad-receptors", f"Receptor sites: {exc}", kind="warn")
            return
        if not receptors:
            self.notices.notify("no-receptors", "Add at least one receptor site (name, lat, lon).")
            return
        span = self._receptor_span()
        if span is None:
            self.notices.notify(
                "no-receptor-times",
                "Load data in the main window first — its dates set the receptor time range.",
            )
            return
        step = self.time_res_spin.value()
        times = list(pd.date_range(span[0].floor("h"), span[1].ceil("h"), freq=f"{step}h"))
        hours_back = self.hours_back_spin.value()
        met_files = list(self._gdas_paths)
        missing = _missing_gdas1_files(
            met_files, times[0] - pd.Timedelta(hours=hours_back), times[-1]
        )
        if missing:
            self._suggest_gdas_range()
            shown = ", ".join(missing[:6]) + (", ..." if len(missing) > 6 else "")
            ans = QMessageBox.question(
                self,
                "Meteorology missing",
                f"The receptor times and their {hours_back}-h backward windows need "
                f"{len(missing)} GDAS1 file(s) the met files lack: {shown}.\n\n"
                "Run anyway? Times no met file covers are skipped, and trajectories that "
                "reach into a gap are cut short. (The GDAS1 range above now covers the "
                "missing weeks, to download them instead.)",
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
        ok, source_regions = self._regions_or_warn()
        if not ok:
            return

        from normet import run_back_trajectories

        height_m = float(self.height_spin.value())
        heights_text = self.heights_edit.text().strip()
        heights_m: list[float] | None = None
        if heights_text:
            try:
                heights_m = [float(x.strip()) for x in heights_text.split(",") if x.strip()]
                if not heights_m or any(h <= 0.0 for h in heights_m):
                    raise ValueError("Heights must be positive numbers.")
            except ValueError as exc:
                self.notices.notify("bad-heights", f"Multi-heights: {exc}", kind="warn")
                return
        n_workers = self.workers_spin.value()
        phys = self._physics_kwargs()
        prefix = self.prefix_edit.text() or "traj_"
        diagnostics = [name for name, cb in self.diag_checks.items() if cb.isChecked()]
        min_hours = hours_back if self.run_min_check.isChecked() else None
        stop = threading.Event()
        run_work = Path(tempfile.mkdtemp(prefix="nm_traj_gui_"))
        self._work_dirs.append(run_work)

        def job() -> pd.DataFrame:
            frames = []
            for name, lat, lon in receptors:
                if stop.is_set():
                    break
                res = run_back_trajectories(
                    times,
                    lat,
                    lon,
                    met_files=met_files,
                    hysplit_exec=exe,
                    height_m=height_m,
                    heights_m=heights_m,
                    hours_back=hours_back,
                    work_dir=run_work,
                    diagnostics=diagnostics,
                    source_regions=source_regions,
                    prefix=prefix,
                    min_hours=min_hours,
                    n_workers=n_workers,
                    stop=stop,
                    **phys,
                )
                if res is not None and not res.empty:
                    res = res.copy()
                    res.insert(0, "site", name)
                    frames.append(res)
            self._last_tdump_paths = sorted(str(p) for p in run_work.glob("tdump_*"))
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames) if len(frames) > 1 else frames[0]

        if self.runner.submit(
            f"run HYSPLIT back-trajectories ({len(receptors)} site(s), {len(times)} time(s))",
            job,
            self._show_features,
            self._show_error,
        ):
            self._hysplit_stop = stop

    # -------------------------------------------------------- Layer-2 analyses
    def _analysis_target_series(self) -> tuple[pd.Series, str] | None:
        main = self._main
        if main is None:
            return None
        mode = self.receptor_source_combo.currentText()
        norm_df = getattr(main, "results", {}).get("normalise")
        use_norm = mode.startswith("de-weathered") or (
            mode.startswith("auto")
            and isinstance(norm_df, pd.DataFrame)
            and "normalised" in norm_df.columns
        )
        if use_norm and isinstance(norm_df, pd.DataFrame) and "normalised" in norm_df.columns:
            s = pd.to_numeric(norm_df["normalised"], errors="coerce").dropna()
            label = "de-weathered normalised"
        elif getattr(main, "df_raw", None) is not None and hasattr(main, "target_combo"):
            df_raw = main.df_raw
            tgt = main.target_combo.currentText()
            if tgt not in df_raw.columns or "date" not in df_raw.columns:
                return None
            idx = pd.to_datetime(df_raw["date"], errors="coerce")
            s = pd.Series(
                pd.to_numeric(df_raw[tgt], errors="coerce").to_numpy(), index=idx, name=tgt
            )
            s = s[s.index.notna()].dropna()
            label = tgt
        else:
            return None

        if s.empty:
            return None
        if hasattr(main, "_zone"):
            from normet.io.tables import clock_to_utc

            s.index = pd.DatetimeIndex(clock_to_utc(s.index.to_series(), main._zone()))
        return s.sort_index(), label

    def _run_receptor_grid(self) -> None:
        tdumps = self._active_tdumps()
        if not tdumps:
            self.notices.notify("no-tdump", "Add tdump files or run HYSPLIT first.")
            return
        tgt_info = self._analysis_target_series()
        if tgt_info is None:
            self.notices.notify(
                "no-target-series",
                "Load data (or run normalisation) in the main window first.",
            )
            return
        ok, regions = self._regions_or_warn()
        if not ok:
            return

        series, tgt_label = tgt_info
        method = self.receptor_method.currentText()
        grid_res = float(self.grid_res_spin.value())
        min_h = self.build_min_spin.value() if self.build_min_check.isChecked() else None
        phys = self._physics_kwargs()
        phys.pop("horizons_h", None)

        def job():
            from normet import gridded_receptor_model, read_trajectory_endpoints

            ep = read_trajectory_endpoints(tdumps, min_hours=min_h)
            return gridded_receptor_model(
                ep,
                series,
                method=method,
                grid_res=grid_res,
                **phys,
            )

        self._pending_analysis_ctx = (regions, tgt_label)
        self.runner.submit(
            f"receptor map ({method.upper()})", job, self._receptor_grid_done, self._show_error
        )

    def _receptor_grid_done(self, grid: Any) -> None:
        from normet import plot_receptor_grid

        regions, tgt_label = getattr(self, "_pending_analysis_ctx", (None, ""))
        fig = plot_receptor_grid(
            grid,
            regions=regions,
            title=f"{grid.method.upper()} ({tgt_label}) — {grid.n_matched_trajectories} trajectories",
        )
        df_cells = grid.to_dataframe().head(12)
        rows = [
            [
                f"{r['lat']:.2f}",
                f"{r['lon']:.2f}",
                f"{r['weighted']:.2f}",
                f"{r['exposure']:.1f}",
                int(r["count"]),
            ]
            for _, r in df_cells.iterrows()
        ]
        self.tab_receptor.show_result(
            fig,
            verdict=(
                "ok",
                f"{grid.method.upper()} mapped {grid.n_matched_trajectories} trajectories onto {len(grid.to_dataframe())} grid cells.",
            ),
            header=["Lat", "Lon", "Weighted", "Exposure", "Trajs"],
            rows=rows,
        )
        self.tabs.setCurrentWidget(self.tab_receptor)
        self.statusBar().showMessage(f"{grid.method.upper()} receptor map done.")

    def _run_clusters(self) -> None:
        tdumps = self._active_tdumps()
        if not tdumps:
            self.notices.notify("no-tdump", "Add tdump files or run HYSPLIT first.")
            return
        k_val = self.cluster_k_spin.value()
        n_clusters = int(k_val) if k_val >= 2 else None
        min_h = self.build_min_spin.value() if self.build_min_check.isChecked() else None
        tgt_info = self._analysis_target_series()
        series = tgt_info[0] if tgt_info is not None else None

        def job():
            from normet import cluster_trajectories, read_trajectory_endpoints

            ep = read_trajectory_endpoints(tdumps, min_hours=min_h)
            return cluster_trajectories(ep, n_clusters=n_clusters, series=series)

        self.runner.submit("cluster trajectories", job, self._clusters_done, self._show_error)

    def _clusters_done(self, res: Any) -> None:
        from normet import plot_trajectory_clusters

        fig = plot_trajectory_clusters(res)
        header = ["Cluster", "Trajectories", "Share", "Dist (km)", "Inflow (°)"]
        has_t = "target_mean" in res.summary.columns
        if has_t:
            header.append("Target mean")
        rows = []
        for _, r in res.summary.iterrows():
            row = [
                f"C{int(r['cluster'])}",
                int(r["n_trajectories"]),
                f"{100.0 * r['freq']:.1f}%",
                f"{r['mean_dist_km']:.1f}",
                f"{r['mean_inflow_deg']:.0f}°",
            ]
            if has_t:
                row.append(f"{r['target_mean']:.2f}" if pd.notna(r["target_mean"]) else "—")
            rows.append(row)
        self.tab_clusters.show_result(
            fig,
            verdict=(
                "ok",
                f"Grouped {int(res.summary['n_trajectories'].sum())} trajectories into {res.n_clusters} air-mass clusters.",
            ),
            header=header,
            rows=rows,
        )
        self.tabs.setCurrentWidget(self.tab_clusters)
        self.statusBar().showMessage(f"Trajectory clustering done (k={res.n_clusters}).")

    def _run_inversion(self) -> None:
        tdumps = self._active_tdumps()
        if not tdumps:
            self.notices.notify("no-tdump", "Add tdump files or run HYSPLIT first.")
            return
        tgt_info = self._analysis_target_series()
        if tgt_info is None:
            self.notices.notify(
                "no-target-series",
                "Load data (or run normalisation) in the main window first.",
            )
            return
        ok, regions = self._regions_or_warn()
        if not ok:
            return
        if not regions:
            self.notices.notify(
                "no-regions",
                "Define at least one source region for footprint inversion.",
                kind="warn",
            )
            return

        series, tgt_label = tgt_info
        min_h = self.build_min_spin.value() if self.build_min_check.isChecked() else None
        phys = self._physics_kwargs()
        phys.pop("horizons_h", None)

        def job():
            from normet import (
                build_footprint_matrix,
                invert_source_footprints,
                read_trajectory_endpoints,
            )

            ep = read_trajectory_endpoints(tdumps, min_hours=min_h)
            fp = build_footprint_matrix(ep, regions, **phys)
            return invert_source_footprints(fp, series)

        self._pending_inv_label = tgt_label
        self.runner.submit("invert source footprints", job, self._inversion_done, self._show_error)

    def _inversion_done(self, inv: Any) -> None:
        from normet import plot_footprint_inversion

        tgt_label = getattr(self, "_pending_inv_label", "")
        fig = plot_footprint_inversion(inv, title=f"Source footprint inversion ({tgt_label})")
        rows = [
            [
                str(r["region"]),
                f"{r['beta']:.3f}",
                f"[{r['beta_ci_low']:.3f}, {r['beta_ci_high']:.3f}]"
                if pd.notna(r["beta_ci_low"])
                else "—",
                f"{r['mean_contribution']:.2f}",
                f"{100.0 * r['share_of_transport']:.1f}%"
                if pd.notna(r["share_of_transport"])
                else "—",
            ]
            for _, r in inv.regions.sort_values("mean_contribution", ascending=False).iterrows()
        ]
        self.tab_inversion.show_result(
            fig,
            verdict=(
                "ok",
                f"Background={inv.intercept:.2f}, R²={inv.r2:.3f}, RMSE={inv.rmse:.2f} (n={inv.n_obs}).",
            ),
            header=["Region", "Beta", "95% CI", "Mean contrib", "Transport share"],
            rows=rows,
        )
        self.tabs.setCurrentWidget(self.tab_inversion)
        self.statusBar().showMessage("Footprint inversion done.")

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._hysplit_stop is not None:
            self._hysplit_stop.set()
        # A stopped run ends within a fraction of a second; give it the time, so its
        # thread is not destroyed while it runs.
        self.runner.shutdown(wait_ms=5000)
        for wd in getattr(self, "_work_dirs", []):
            shutil.rmtree(wd, ignore_errors=True)
        super().closeEvent(event)
