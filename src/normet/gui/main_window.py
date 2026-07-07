"""Main window of the normet Qt GUI.

Design language: a scrollable parameter panel on the left walks the user
through the workflow top-to-bottom (Data → Columns → Train → Normalise →
Decompose → Rolling → PDP), each step with its own prominent "▶ Run"
button; results land in tabs on the right, which auto-activate when a run
finishes.  Verdict banners (green / amber / red) summarise quality at a
glance; every long computation runs on a background thread; the bottom dock
streams normet's log.

The synthetic-control workflow (panel data, treated unit vs. donors) has
genuinely different inputs, so it lives in its own window — see
:mod:`.scm_window` — opened from the toolbar button or the Analysis menu.
"""

from __future__ import annotations

import datetime
import json
import logging
import os

import pandas as pd
from PySide6.QtCore import QSettings, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import _settings
from ._widgets import (
    CanvasTab,
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
    hint_label,
    run_button,
    toggle_button,
)
from .dataframe_model import DataFrameModel
from .workers import QtLogHandler, TaskRunner

log = logging.getLogger(__name__)

TIME_VARS = ("date_unix", "day_julian", "weekday", "hour")

#: Column names auto-recognised as meteorological predictors.
MET_DEFAULTS = {
    "u10",
    "v10",
    "d2m",
    "t2m",
    "blh",
    "sp",
    "ssrd",
    "tcc",
    "tp",
    "rh2m",
    "ws",
    "wd",
    "temp",
    "rh",
    "pressure",
}

DOCS_URL = "https://github.com/apai-sys/normet-py"


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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Normet — meteorological normalisation & counterfactual toolkit")
        self.resize(1380, 900)
        self.setAcceptDrops(True)  # drag a CSV onto the window to load it

        self.df_raw: pd.DataFrame | None = None
        self.df_prep: pd.DataFrame | None = None
        self.model: object | None = None
        self.trained_features: list[str] = []
        self.results: dict[str, pd.DataFrame] = {}
        self._scm_window = None
        self._data_window = None
        self._transport_window = None

        self.runner = TaskRunner(self)
        self.runner.started.connect(self._task_started)
        self.runner.finished.connect(self._task_finished)

        self._build_central()
        self._build_log_dock()
        self._build_status_bar()
        self._build_menu()
        self._sync_enabled()

        settings = QSettings("apai-sys", "normet-gui")
        geo = settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        self.statusBar().showMessage("Open a CSV or load the example data to begin.")

    # ------------------------------------------------------------------ menu
    def _build_menu(self) -> None:
        mb = self.menuBar()

        fm = mb.addMenu("&File")
        act = fm.addAction("&Open CSV…", self._open_csv)
        act.setShortcut("Ctrl+O")
        fm.addAction("Load &Example Data", self.load_example)
        act = fm.addAction("Get &UK Data (AURN + met)…", self.open_data_studio)
        act.setShortcut("Ctrl+D")
        self.recent_menu = fm.addMenu("Open &Recent")
        self._rebuild_recent_menu()
        fm.addSeparator()
        self._export_action = fm.addAction("&Export Current Result…", self._export_current)
        self._export_action.setShortcut("Ctrl+E")
        self._export_action.setEnabled(False)
        self._report_action = fm.addAction("Export &HTML Report…", self._export_html)
        self._report_action.setEnabled(False)
        fm.addSeparator()
        fm.addAction("Save Con&fig…", self.save_config)
        fm.addAction("Load Confi&g…", self.load_config)
        fm.addSeparator()
        act = fm.addAction("&Quit", self.close)
        act.setShortcut("Ctrl+Q")

        em = mb.addMenu("&Edit")  # routes to the focused widget; native shortcuts still work
        em.addAction("Cut", lambda: self._clip("cut"))
        em.addAction("Copy", lambda: self._clip("copy"))
        em.addAction("Paste", lambda: self._clip("paste"))
        em.addAction("Select All", lambda: self._clip("selectAll"))

        vm = mb.addMenu("&View")
        vm.addAction("Run &History…", self.show_history)
        vm.addSeparator()
        vm.addAction(self._log_dock.toggleViewAction())

        am = mb.addMenu("&Analysis")
        act = am.addAction("Synthetic Control (SCM) Studio…", self.open_scm_studio)
        act.setShortcut("Ctrl+Shift+S")

        wm = mb.addMenu("&Window")
        wm.addAction("Minimize", self.showMinimized).setShortcut("Ctrl+M")
        wm.addAction("Zoom", self.showMaximized)

        hm = mb.addMenu("&Help")
        hm.addAction("Documentation", lambda: QDesktopServices.openUrl(QUrl(DOCS_URL)))
        hm.addAction("&About Normet", self._about)

    def _clip(self, method: str) -> None:
        w = self.focusWidget()
        if w is not None and hasattr(w, method):
            getattr(w, method)()

    # --------------------------------------------------------------- central
    def _build_central(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_param_panel())

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 4, 4, 0)
        rv.addWidget(self._build_action_bar())
        rv.addWidget(self._build_tabs(), 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([370, 1010])
        self.setCentralWidget(splitter)

    # --------------------------------------------------------- left panel
    def _build_param_panel(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)

        # ---- Data
        data_box = QGroupBox("Data")
        dv = QVBoxLayout(data_box)
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
        v.addWidget(data_box)

        # ---- Columns
        col_box = QGroupBox("Columns")
        cv = QVBoxLayout(col_box)
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
        v.addWidget(col_box)

        # ---- Step 1: Train
        train_box = QGroupBox("Step 1 · Train model")
        form = QFormLayout(train_box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.backend_combo = NoWheelComboBox()
        self.backend_combo.addItems(["flaml", "lightgbm"])
        self.backend_combo.setToolTip(
            "flaml = AutoML search (uses the time budget below).\nlightgbm = LightGBM with random hyperparameter search."
        )
        self.backend_combo.currentTextChanged.connect(self._train_backend_changed)
        form.addRow("Backend", self.backend_combo)
        self.split_combo = NoWheelComboBox()
        self.split_combo.addItems(["random", "ts", "season", "month"])
        self.split_combo.setToolTip("How the train/test split is drawn.")
        form.addRow("Split method", self.split_combo)
        self.fraction_spin = NoWheelDoubleSpinBox()
        self.fraction_spin.setRange(0.1, 0.95)
        self.fraction_spin.setSingleStep(0.05)
        self.fraction_spin.setValue(0.75)
        form.addRow("Training fraction", self.fraction_spin)
        self.budget_spin = NoWheelSpinBox()
        self.budget_spin.setRange(5, 36_000)
        self.budget_spin.setValue(60)
        self.budget_spin.setSuffix(" s")
        self.budget_spin.setToolTip("AutoML search time (flaml backend).")
        self._budget_label = QLabel("Time budget")
        self._budget_values = {"flaml": 60, "lightgbm": 20}
        self._prev_backend = "flaml"
        form.addRow(self._budget_label, self.budget_spin)
        # flaml only: which AutoML estimators to search over.
        self.estimator_list = QListWidget()
        self.estimator_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.estimator_list.setMaximumHeight(96)
        self.estimator_list.setToolTip(
            "AutoML estimators flaml may try (searched within the time budget).\n"
            "lgbm is the fastest and usually sufficient; tick more to widen the search."
        )
        for name, tip in [
            ("lgbm", "LightGBM (default, fast)"),
            ("xgboost", "XGBoost"),
            ("xgb_limitdepth", "XGBoost, depth-limited"),
            ("rf", "Random forest"),
            ("extra_tree", "Extra trees"),
            ("histgb", "Histogram gradient boosting (sklearn)"),
        ]:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if name == "lgbm" else Qt.CheckState.Unchecked)
            item.setToolTip(tip)
            self.estimator_list.addItem(item)
        self._estimator_label = QLabel("Estimators")
        form.addRow(self._estimator_label, self.estimator_list)
        self._train_form = form
        self.seed_spin = NoWheelSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(7_654_321)
        form.addRow("Seed", self.seed_spin)
        self.train_button = run_button(
            "▶  Train model",
            "Prepare the data and train the machine-learning model.\nEverything below needs a trained model.",
        )
        self.train_button.clicked.connect(self._run_train)
        form.addRow(self.train_button)
        v.addWidget(train_box)

        # ---- Step 2: Normalise
        norm_box = QGroupBox("Step 2 · Normalise")
        nf = QFormLayout(norm_box)
        nf.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.norm_vars = QListWidget()
        self.norm_vars.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.norm_vars.setMaximumHeight(110)
        self.norm_vars.setToolTip(
            "Which variables are resampled in the Monte-Carlo (their influence is\n"
            "averaged out of the target). Ticked = normalised away; unticked stay\n"
            "at their observed values. Lists every trained variable, including the\n"
            "auto-added time features — resampling those too removes trend and\n"
            "seasonality as well. Default (Met only) is the usual deweathering choice."
        )
        nf.addRow("Resample vars", self.norm_vars)
        nv_row = QHBoxLayout()
        b_nv_met = QPushButton("Met only")
        b_nv_met.setToolTip("Tick only the recognised meteorological variables.")
        b_nv_met.clicked.connect(lambda: self._set_norm_vars_met_only())
        b_nv_all = QPushButton("All")
        b_nv_all.clicked.connect(lambda: _set_all(self.norm_vars, True))
        nv_row.addWidget(b_nv_met)
        nv_row.addWidget(b_nv_all)
        nf.addRow("", self._wrap_row(nv_row))
        self.norm_samples = NoWheelSpinBox()
        self.norm_samples.setRange(1, 10_000)
        self.norm_samples.setValue(300)
        self.norm_samples.setToolTip(
            "Monte-Carlo resamples of the meteorological conditions.\nMore = smoother but slower."
        )
        nf.addRow("Samples", self.norm_samples)
        self.norm_cores = NoWheelSpinBox()
        self.norm_cores.setRange(0, 64)
        self.norm_cores.setValue(0)
        self.norm_cores.setSpecialValueText("auto")
        nf.addRow("CPU cores", self.norm_cores)
        self.norm_quantiles = QCheckBox("5–95 % quantile band")
        self.norm_quantiles.setToolTip("Also return the Monte-Carlo quantiles and shade the band.")
        nf.addRow(self.norm_quantiles)
        self.norm_button = run_button(
            "▶  Run normalisation",
            "Remove the meteorological signal from the target\n(counterfactual under average meteorology).",
        )
        self.norm_button.clicked.connect(self._run_normalise)
        nf.addRow(self.norm_button)
        v.addWidget(norm_box)

        # ---- Step 3: Decompose
        dec_box = QGroupBox("Step 3 · Decompose")
        df_ = QFormLayout(dec_box)
        df_.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.decom_method = NoWheelComboBox()
        self.decom_method.addItems(["emission", "meteorology"])
        self.decom_method.setToolTip(
            "emission: split the normalised series (Step 2) into time scales\n"
            "(trend / seasonal / weekly / diurnal / noise).\n"
            "meteorology: split observed − normalised into the contribution\n"
            "of each individual meteorological variable."
        )
        df_.addRow("Method", self.decom_method)
        self.decom_samples = NoWheelSpinBox()
        self.decom_samples.setRange(1, 10_000)
        self.decom_samples.setValue(300)
        df_.addRow("Samples", self.decom_samples)
        self.decom_button = run_button(
            "▶  Run decomposition",
            "Split the observed series into additive contributions\n(leave-one-out normalisation).",
        )
        self.decom_button.clicked.connect(self._run_decompose)
        df_.addRow(self.decom_button)
        v.addWidget(dec_box)

        # ---- Step 4: Rolling
        roll_box = QGroupBox("Step 4 · Rolling")
        rf = QFormLayout(roll_box)
        rf.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.roll_window = NoWheelSpinBox()
        self.roll_window.setRange(1, 365)
        self.roll_window.setValue(14)
        self.roll_window.setSuffix(" d")
        rf.addRow("Window", self.roll_window)
        self.roll_step = NoWheelSpinBox()
        self.roll_step.setRange(1, 365)
        self.roll_step.setValue(7)
        self.roll_step.setSuffix(" d")
        rf.addRow("Step", self.roll_step)
        self.roll_samples = NoWheelSpinBox()
        self.roll_samples.setRange(1, 10_000)
        self.roll_samples.setValue(100)
        rf.addRow("Samples", self.roll_samples)
        self.roll_button = run_button(
            "▶  Run rolling normalisation",
            "Windowed re-normalisation — reveals slow changes in the\n"
            "emission signal. Resamples the variables ticked in Step 2.",
        )
        self.roll_button.clicked.connect(self._run_rolling)
        rf.addRow(self.roll_button)
        v.addWidget(roll_box)

        # ---- Multi-scale meteorological-influence decomposition (builds on Rolling)
        ms_box = QGroupBox("Multi-scale decomposition")
        mf = QFormLayout(ms_box)
        mf.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        mf.addRow(
            hint_label(
                "Differences Y_fast − Y_meso − Y_slow − Y_∞ (Step 2) isolate the "
                "meteorological residual specific to each timescale — like wavelet "
                "detail coefficients or MSTL's trend/seasonal/remainder."
            )
        )
        self.ms_fast = NoWheelSpinBox()
        self.ms_fast.setRange(1, 3650)
        self.ms_fast.setValue(14)
        self.ms_fast.setSuffix(" d")
        mf.addRow("Fast window", self.ms_fast)
        self.ms_meso = NoWheelSpinBox()
        self.ms_meso.setRange(1, 3650)
        self.ms_meso.setValue(90)
        self.ms_meso.setSuffix(" d")
        mf.addRow("Meso window", self.ms_meso)
        self.ms_slow = NoWheelSpinBox()
        self.ms_slow.setRange(1, 3650)
        self.ms_slow.setValue(365)
        self.ms_slow.setSuffix(" d")
        mf.addRow("Slow window", self.ms_slow)
        self._ms_button_tip = (
            "Runs rolling normalisation at three window widths and differences\n"
            "them against each other and against Step 2's Y_∞ — needs Step 2\n"
            "(Normalise) to have been run first, and is roughly 3× the cost of\n"
            "a single Rolling run."
        )
        self.ms_button = run_button("▶  Multi-scale decomposition", self._ms_button_tip)
        self.ms_button.clicked.connect(self._run_multiscale)
        mf.addRow(self.ms_button)
        v.addWidget(ms_box)

        # ---- Step 5: PDP
        pdp_box = QGroupBox("Step 5 · Partial dependence")
        pv = QVBoxLayout(pdp_box)
        pv.addWidget(QLabel("Variables"))
        self.pdp_vars = QListWidget()
        self.pdp_vars.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.pdp_vars.setMaximumHeight(120)
        self.pdp_vars.setToolTip("Filled with the trained features after Step 1.")
        pv.addWidget(self.pdp_vars)
        grid_row = QHBoxLayout()
        grid_row.addWidget(QLabel("Grid points"))
        self.pdp_grid = NoWheelSpinBox()
        self.pdp_grid.setRange(5, 500)
        self.pdp_grid.setValue(50)
        grid_row.addWidget(self.pdp_grid)
        self.pdp_time_btn = toggle_button(
            "Time variables",
            "Toggle date_unix / day_julian / weekday / hour:\n"
            "click to tick all of them, click again to untick.",
        )
        self.pdp_time_btn.toggled.connect(lambda checked: self._toggle_pdp_vars(checked, TIME_VARS))
        self.pdp_met_btn = toggle_button(
            "Met only",
            "Toggle the recognised meteorological variables:\n"
            "click to tick all of them, click again to untick.",
        )
        self.pdp_met_btn.toggled.connect(lambda checked: self._toggle_pdp_vars(checked, None))
        grid_row.addWidget(self.pdp_time_btn)
        grid_row.addWidget(self.pdp_met_btn)
        grid_row.addStretch(1)
        pv.addLayout(grid_row)
        self.pdp_button = run_button(
            "▶  Compute PDP",
            "Partial-dependence curves: how the model's prediction\nresponds to each variable.",
        )
        self.pdp_button.clicked.connect(self._run_pdp)
        pv.addWidget(self.pdp_button)
        v.addWidget(pdp_box)

        # ---- Config row
        cfg_row = QHBoxLayout()
        b_save = QPushButton("Save config")
        b_save.clicked.connect(self.save_config)
        b_load = QPushButton("Load config")
        b_load.clicked.connect(self.load_config)
        b_hist = QPushButton("History")
        b_hist.clicked.connect(self.show_history)
        cfg_row.addWidget(b_save)
        cfg_row.addWidget(b_load)
        cfg_row.addWidget(b_hist)
        v.addLayout(cfg_row)
        v.addStretch(1)

        self._run_buttons = [
            self.train_button,
            self.norm_button,
            self.decom_button,
            self.roll_button,
            self.ms_button,
            self.pdp_button,
        ]

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(390)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll

    # ------------------------------------------------------------ action bar
    def _build_action_bar(self) -> QWidget:
        bar = QWidget()
        h = QHBoxLayout(bar)
        h.setContentsMargins(4, 0, 4, 0)
        self.exp_csv = QPushButton("⬇  Export result CSV")
        self.exp_csv.setToolTip("Export the current tab's result table as CSV.")
        self.exp_csv.clicked.connect(self._export_current)
        self.exp_csv.setEnabled(False)
        self.exp_html = QPushButton("📄  HTML report")
        self.exp_html.setToolTip(
            "Single-file HTML report of the current tab's result,\nwith provenance metadata."
        )
        self.exp_html.clicked.connect(self._export_html)
        self.exp_html.setEnabled(False)
        self.scm_btn = QPushButton("🧪  SCM Studio")
        self.scm_btn.setToolTip(
            "Synthetic-control (counterfactual) analysis on panel data\n— treated unit vs. donor pool."
        )
        self.scm_btn.clicked.connect(self.open_scm_studio)
        self.data_btn = QPushButton("🌐  Get UK data")
        self.data_btn.setToolTip(
            "Fetch UK AURN air quality + reanalysis meteorology from the\n"
            "internet and merge them into a model-ready table (no API key)."
        )
        self.data_btn.clicked.connect(self.open_data_studio)
        self.transport_btn = QPushButton("🧭  Transport Studio")
        self.transport_btn.setToolTip(
            "Build transport-aware predictors (inflow direction, transport\n"
            "distance/speed, residence time over source regions) from HYSPLIT\n"
            "back-trajectory (tdump) output, and join them onto this data."
        )
        self.transport_btn.clicked.connect(self.open_transport_studio)
        h.addWidget(self.data_btn)
        h.addWidget(self.transport_btn)
        h.addWidget(self.exp_csv)
        h.addWidget(self.exp_html)
        h.addWidget(self.scm_btn)
        h.addStretch(1)
        return bar

    # ----------------------------------------------------------------- tabs
    def _build_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()

        # Data preview tab
        data_tab = QWidget()
        dl = QVBoxLayout(data_tab)
        self.data_summary = QLabel("Open a CSV (or load the example) to begin.")
        self.data_summary.setWordWrap(True)
        dl.addWidget(self.data_summary)
        self.data_preview = CanvasTab("The target series preview appears here once data is loaded.")
        self.data_preview.setMaximumHeight(280)
        dl.addWidget(self.data_preview)
        self.table_model = DataFrameModel()
        self.table_view = QTableView()
        self.table_view.setModel(self.table_model)
        dl.addWidget(self.table_view, 1)
        self.tabs.addTab(data_tab, "① Data")

        self.tab_model = CanvasTab("Train a model (Step 1) to see its quality here.")
        self.tabs.addTab(self.tab_model, "② Model")
        self.tab_norm = CanvasTab(
            "Run Step 2 to see observed vs. meteorologically normalised series."
        )
        self.tabs.addTab(self.tab_norm, "③ Normalise")
        self.tab_decom = CanvasTab(
            "Run Step 3 to see the decomposition — emission splits the "
            "normalised series into time scales; meteorology splits "
            "observed − normalised by met variable."
        )
        self.tabs.addTab(self.tab_decom, "④ Decompose")
        self.tab_roll = CanvasTab("Run Step 4 to see the rolling normalisation.")
        self.tabs.addTab(self.tab_roll, "⑤ Rolling")
        self.tab_multiscale = CanvasTab(
            "Run Step 2 (Normalise) then Multi-scale decomposition to see "
            "how meteorological residuals split across fast/meso/slow "
            "timescales."
        )
        self.tabs.addTab(self.tab_multiscale, "Multi-scale")
        self.tab_pdp = CanvasTab("Run Step 5 to see partial-dependence curves.")
        self.tabs.addTab(self.tab_pdp, "⑥ PDP")
        return self.tabs

    # ------------------------------------------------------------- log dock
    def _build_log_dock(self) -> None:
        dock = QDockWidget("Log", self)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5_000)
        dock.setWidget(self.log_view)
        dock.setMaximumHeight(160)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self._log_dock = dock

        handler = QtLogHandler()
        handler.message.connect(self.log_view.appendPlainText)
        handler.setLevel(logging.INFO)
        # normet.gui.* loggers propagate to "normet", so one handler suffices.
        logging.getLogger("normet").addHandler(handler)
        logging.getLogger("normet").setLevel(logging.INFO)
        self._log_handler = handler

    # ------------------------------------------------------------ status bar
    def _build_status_bar(self) -> None:
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setMaximumWidth(220)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setToolTip(
            "Abandon the running task: the UI unlocks now and the result is\n"
            "discarded when the background computation eventually finishes."
        )
        self.cancel_btn.clicked.connect(self._abandon_task)
        self.statusBar().addPermanentWidget(self.cancel_btn)
        self.statusBar().addPermanentWidget(self.progress)

    # ------------------------------------------------------------ data I/O
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
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.rename(columns={date_col: "date"})

        self._ingest(df, os.path.basename(path))
        if remember:
            _settings.add_recent(path)
            self._rebuild_recent_menu()
        log.info("Loaded %s (%d rows)", path, len(df))

    def _ingest(self, df: pd.DataFrame, label: str) -> None:
        self.df_raw = df
        self.df_prep = None
        self.model = None
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
        self._draw_data_preview()
        self.tabs.setCurrentIndex(0)
        self.statusBar().showMessage(
            "Data loaded — check the target and features, then ▶ Train model (Step 1)."
        )

    def _target_changed(self, _text: str) -> None:
        if self.df_raw is not None:
            self._draw_data_preview()

    def _train_backend_changed(self, backend: str) -> None:
        """The budget row means seconds for flaml, search trials for LightGBM;
        the estimator checklist only applies to flaml's AutoML search."""
        self._budget_values[self._prev_backend] = self.budget_spin.value()
        self._prev_backend = backend
        if backend == "flaml":
            self._budget_label.setText("Time budget")
            self.budget_spin.setRange(5, 36_000)
            self.budget_spin.setSuffix(" s")
            self.budget_spin.setToolTip("AutoML search time (flaml backend).")
        else:
            self._budget_label.setText("Search trials")
            self.budget_spin.setRange(1, 500)
            self.budget_spin.setSuffix("")
            self.budget_spin.setToolTip("Random hyperparameter-search trials (LightGBM backend).")
        self.budget_spin.setValue(self._budget_values.get(backend, 20))
        is_flaml = backend == "flaml"
        self._estimator_label.setVisible(is_flaml)
        self.estimator_list.setVisible(is_flaml)

    def _selected_estimators(self) -> list[str]:
        return _checked_items(self.estimator_list)

    @staticmethod
    def _wrap_row(layout: QHBoxLayout) -> QWidget:
        w = QWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)
        return w

    def _set_norm_vars_met_only(self) -> None:
        for i in range(self.norm_vars.count()):
            item = self.norm_vars.item(i)
            is_met = item.text().lower() in MET_DEFAULTS or item.text() in MET_DEFAULTS
            item.setCheckState(Qt.CheckState.Checked if is_met else Qt.CheckState.Unchecked)

    def _selected_norm_vars(self) -> list[str]:
        return _checked_items(self.norm_vars)

    def _check_met_only(self) -> None:
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            is_met = item.text().lower() in MET_DEFAULTS or item.text() in MET_DEFAULTS
            item.setCheckState(Qt.CheckState.Checked if is_met else Qt.CheckState.Unchecked)

    def _toggle_pdp_vars(self, checked: bool, names: tuple[str, ...] | None) -> None:
        """Tick/untick a subset of Step 5's variable list.

        *names* selects an explicit subset (time variables); pass None to
        target the recognised meteorological variables instead.
        """
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.pdp_vars.count()):
            item = self.pdp_vars.item(i)
            text = item.text()
            matches = (
                text in names
                if names is not None
                else (text.lower() in MET_DEFAULTS or text in MET_DEFAULTS)
            )
            if matches:
                item.setCheckState(state)

    def _reset_pdp_toggle_buttons(self) -> None:
        """Un-press the Time-variables/Met-only toggles after the variable
        list is (re)filled, so their pressed state doesn't lie about what's
        actually ticked."""
        for btn in (self.pdp_time_btn, self.pdp_met_btn):
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)

    def _draw_data_preview(self) -> None:
        """Quick look at the chosen target series (daily means when dense)."""
        target = self.target_combo.currentText()
        if self.df_raw is None or not target or target not in self.df_raw.columns:
            return
        import matplotlib.pyplot as plt

        s = self.df_raw.set_index("date")[target].dropna()
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

    # ---- drag-and-drop + recent files ----
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

    def _rebuild_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = _settings.read_recent()
        if not recent:
            self.recent_menu.addAction("(none)").setEnabled(False)
            return
        for p in recent:
            self.recent_menu.addAction(
                os.path.basename(p),
                lambda checked=False, p=p: self.load_csv(p, remember=True),
            )
        self.recent_menu.addSeparator()
        self.recent_menu.addAction("Clear Recent", self._clear_recent)

    def _clear_recent(self) -> None:
        _settings.clear_recent()
        self._rebuild_recent_menu()

    # ------------------------------------------------------------- config
    def get_config(self) -> dict:
        return {
            "target": self.target_combo.currentText(),
            "features": _checked_items(self.feature_list),
            "backend": self.backend_combo.currentText(),
            "split_method": self.split_combo.currentText(),
            "fraction": self.fraction_spin.value(),
            "time_budget": (
                self.budget_spin.value()
                if self.backend_combo.currentText() == "flaml"
                else self._budget_values.get("flaml", 60)
            ),
            "lgb_trials": (
                self.budget_spin.value()
                if self.backend_combo.currentText() == "lightgbm"
                else self._budget_values.get("lightgbm", 20)
            ),
            "flaml_estimators": self._selected_estimators(),
            "seed": self.seed_spin.value(),
            "norm_vars": self._selected_norm_vars(),
            "norm_samples": self.norm_samples.value(),
            "norm_cores": self.norm_cores.value(),
            "norm_quantiles": self.norm_quantiles.isChecked(),
            "decom_method": self.decom_method.currentText(),
            "decom_samples": self.decom_samples.value(),
            "roll_window": self.roll_window.value(),
            "roll_step": self.roll_step.value(),
            "roll_samples": self.roll_samples.value(),
            "ms_fast": self.ms_fast.value(),
            "ms_meso": self.ms_meso.value(),
            "ms_slow": self.ms_slow.value(),
            "pdp_grid": self.pdp_grid.value(),
        }

    def apply_config(self, c: dict) -> None:
        if c.get("target"):
            self.target_combo.setCurrentText(c["target"])
        feats = set(c.get("features") or [])
        if feats:
            for i in range(self.feature_list.count()):
                item = self.feature_list.item(i)
                item.setCheckState(
                    Qt.CheckState.Checked if item.text() in feats else Qt.CheckState.Unchecked
                )
        self._budget_values["flaml"] = c.get("time_budget", 60)
        self._budget_values["lightgbm"] = c.get("lgb_trials", 20)
        self.backend_combo.setCurrentText(c.get("backend", "flaml"))
        self.split_combo.setCurrentText(c.get("split_method", "random"))
        self.fraction_spin.setValue(c.get("fraction", 0.75))
        self.budget_spin.setValue(self._budget_values.get(self.backend_combo.currentText(), 60))
        ests = set(c.get("flaml_estimators") or ["lgbm"])
        for i in range(self.estimator_list.count()):
            item = self.estimator_list.item(i)
            item.setCheckState(
                Qt.CheckState.Checked if item.text() in ests else Qt.CheckState.Unchecked
            )
        self.seed_spin.setValue(c.get("seed", 7_654_321))
        nvars = set(c.get("norm_vars") or [])
        if nvars:
            for i in range(self.norm_vars.count()):
                item = self.norm_vars.item(i)
                item.setCheckState(
                    Qt.CheckState.Checked if item.text() in nvars else Qt.CheckState.Unchecked
                )
        self.norm_samples.setValue(c.get("norm_samples", 300))
        self.norm_cores.setValue(c.get("norm_cores", 0))
        self.norm_quantiles.setChecked(c.get("norm_quantiles", False))
        self.decom_method.setCurrentText(c.get("decom_method", "emission"))
        self.decom_samples.setValue(c.get("decom_samples", 300))
        self.roll_window.setValue(c.get("roll_window", 14))
        self.roll_step.setValue(c.get("roll_step", 7))
        self.roll_samples.setValue(c.get("roll_samples", 100))
        self.ms_fast.setValue(c.get("ms_fast", 14))
        self.ms_meso.setValue(c.get("ms_meso", 90))
        self.ms_slow.setValue(c.get("ms_slow", 365))
        self.pdp_grid.setValue(c.get("pdp_grid", 50))

    def save_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save config", "normet_config.json", "JSON (*.json)"
        )
        if path:
            with open(path, "w") as f:
                json.dump(self.get_config(), f, indent=2)
            self.statusBar().showMessage(f"Saved {os.path.basename(path)}")

    def load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load config", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                self.apply_config(json.load(f))
            self.statusBar().showMessage(f"Loaded {os.path.basename(path)}")
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))

    # ------------------------------------------------------------- history
    def _log_history(self, task: str, detail: str) -> None:
        _settings.append_history(
            {
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "task": task,
                "target": self.target_combo.currentText(),
                "backend": self.backend_combo.currentText(),
                "rows": 0 if self.df_raw is None else len(self.df_raw),
                "detail": detail,
            }
        )

    def show_history(self) -> None:
        recs = _settings.read_history()
        dlg = QDialog(self)
        dlg.setWindowTitle("Run history")
        dlg.resize(760, 420)
        lay = QVBoxLayout(dlg)
        if not recs:
            lay.addWidget(QLabel("No runs logged yet."))
        else:
            cols = ["time", "task", "target", "backend", "rows", "detail"]
            tbl = QTableWidget(len(recs), len(cols))
            tbl.setHorizontalHeaderLabels(cols)
            tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            for i, rec in enumerate(reversed(recs)):  # newest first
                for j, col in enumerate(cols):
                    tbl.setItem(i, j, QTableWidgetItem(str(rec.get(col, ""))))
            tbl.resizeColumnsToContents()
            lay.addWidget(tbl)
        clr = QPushButton("Clear history")
        clr.clicked.connect(lambda: (_settings.clear_history(), dlg.accept()))
        lay.addWidget(clr)
        dlg.exec()

    # -------------------------------------------------------------- export
    _TAB_KEYS = {
        0: "data",
        1: "train",
        2: "normalise",
        3: "decompose",
        4: "rolling",
        5: "multiscale",
        6: "pdp",
    }

    def _current_result(self) -> tuple[str, pd.DataFrame | None]:
        key = self._TAB_KEYS.get(self.tabs.currentIndex(), "")
        if key == "data":
            return key, self.df_raw
        return key, self.results.get(key)

    def _export_current(self) -> None:
        key, result = self._current_result()
        if result is None:
            QMessageBox.information(
                self, "Nothing to export", "Run the current tab's analysis first."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export result", f"normet_{key}.csv", "CSV files (*.csv)"
        )
        if path:
            result.to_csv(path)
            self.statusBar().showMessage(f"Saved {os.path.basename(path)}")
            log.info("Exported %s result to %s", key, path)

    def _export_html(self) -> None:
        key, result = self._current_result()
        if result is None or key == "data":
            QMessageBox.information(
                self,
                "Nothing to report",
                "Switch to a result tab (Normalise, Decompose, …) that has been run.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save HTML report", f"normet_{key}_report.html", "HTML (*.html)"
        )
        if not path:
            return
        try:
            from normet import generate_html_report, make_run

            run = make_run(
                result,
                kind=key,
                config=self.get_config(),
                df=self.df_raw,
                model=self.model,
                seed=self.seed_spin.value(),
            )
            generate_html_report(run, path)
            self.statusBar().showMessage(f"Saved {os.path.basename(path)}")
            log.info("HTML report written to %s", path)
        except Exception as exc:
            QMessageBox.critical(self, "Report failed", str(exc))

    # ----------------------------------------------------------- helpers
    def _sync_enabled(self) -> None:
        busy = self.runner.busy
        has_data = self.df_raw is not None
        has_model = self.model is not None
        has_y_inf = "normalise" in self.results
        self.train_button.setEnabled(not busy and has_data)
        for b in (self.norm_button, self.decom_button, self.roll_button, self.pdp_button):
            b.setEnabled(not busy and has_model)
        if not has_model:
            tip = "Train a model first (Step 1)."
            for b in (self.norm_button, self.decom_button, self.roll_button, self.pdp_button):
                b.setToolTip(tip)
        self.ms_button.setEnabled(not busy and has_model and has_y_inf)
        self.ms_button.setToolTip(
            self._ms_button_tip
            if has_y_inf
            else "Run Step 2 (Normalise) first — it provides Y_∞, the full-record baseline."
        )
        has_result = bool(self.results)
        self.exp_csv.setEnabled(has_data)
        self.exp_html.setEnabled(has_result)
        self._export_action.setEnabled(has_data)
        self._report_action.setEnabled(has_result)
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

    def _abandon_task(self) -> None:
        if self.runner.busy:
            self.runner.abandon()
            self.statusBar().showMessage(
                "Task abandoned — it finishes in the background and its result is discarded."
            )

    def _check_data(self) -> bool:
        if self.df_raw is None:
            QMessageBox.information(self, "No data", "Open a CSV file first.")
            return False
        if not self._selected_features():
            QMessageBox.information(self, "No features", "Tick at least one feature.")
            return False
        return True

    def _check_model(self) -> bool:
        if self.model is None or self.df_prep is None:
            QMessageBox.information(self, "No model", "Train a model first (Step 1).")
            return False
        return True

    def _selected_features(self) -> list[str]:
        target = self.target_combo.currentText()
        return [f for f in _checked_items(self.feature_list) if f != target]

    def _show_error(self, tb: str) -> None:
        QMessageBox.critical(self, "Task failed", tb.splitlines()[-1] if tb else "Unknown error")
        self.log_view.appendPlainText(tb)

    def _n_cores(self, spin: NoWheelSpinBox) -> int | None:
        return None if spin.value() == 0 else spin.value()

    # -------------------------------------------------------------- tasks
    def _run_train(self) -> None:
        if not self._check_data():
            return
        from normet import build_model

        value = self.target_combo.currentText()
        features = self._selected_features()
        backend = self.backend_combo.currentText()
        if backend == "flaml":
            model_config = {
                "time_budget": self.budget_spin.value(),
                "estimator_list": self._selected_estimators() or ["lgbm"],
            }
        else:
            model_config = {"n_trials": self.budget_spin.value()}

        # The model must see the time features alongside the met predictors —
        # they carry the trend/seasonality that survives normalisation.
        # prepare_data() creates the columns; build_model() only trains on
        # what is listed here.
        self._pending_features = features
        self.runner.submit(
            f"train ({backend})",
            build_model,
            self._train_done,
            self._show_error,
            self.df_raw,
            value,
            backend=backend,
            feature_names=features + list(TIME_VARS),
            split_method=self.split_combo.currentText(),
            fraction=self.fraction_spin.value(),
            model_config=model_config,
            seed=self.seed_spin.value(),
            verbose=True,
        )

    def _train_done(self, result: tuple[pd.DataFrame, object]) -> None:
        self.df_prep, self.model = result
        self.trained_features = list(self._pending_features)
        _fill_checklist(
            self.pdp_vars,
            self.trained_features + list(TIME_VARS),
            set(self.trained_features[:4]),
        )
        self._reset_pdp_toggle_buttons()
        # Offer every trained variable (incl. the auto-added time features)
        # for resampling; default to Met only — the standard deweathering
        # choice, which keeps trend/seasonality in the normalised series.
        met = {f for f in self.trained_features if f.lower() in MET_DEFAULTS or f in MET_DEFAULTS}
        _fill_checklist(
            self.norm_vars,
            self.trained_features + list(TIME_VARS),
            met or set(self.trained_features),
        )
        self._render_model_tab()
        self._sync_enabled()
        self.tabs.setCurrentWidget(self.tab_model)
        self.statusBar().showMessage(
            "Model trained — check its quality, then run Step 2 (normalisation)."
        )
        log.info("Training finished")

    def _render_model_tab(self) -> None:
        """Stats table + parity plot + feature importances, with an R² verdict."""
        import matplotlib.pyplot as plt
        import numpy as np

        from normet import ml_predict, modStats

        header: list[str] | None = None
        rows: list[list] | None = None
        stats = None
        try:
            stats = modStats(self.df_prep, self.model).round(3)
            self.results["train"] = stats
            tbl = stats.reset_index() if stats.index.name or stats.index.nlevels > 1 else stats
            header = [str(c) for c in tbl.columns]
            rows = [[v for v in row] for row in tbl.itertuples(index=False)]
        except Exception:
            log.exception("modStats failed")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

        test = self.df_prep
        if "set" in self.df_prep.columns:
            test = self.df_prep[self.df_prep["set"] == "testing"]
        if len(test) == 0:
            test = self.df_prep
        obs = test["value"].to_numpy()
        pred = np.asarray(ml_predict(self.model, test))
        ax1.scatter(obs, pred, s=6, alpha=0.35, color="#2c7bb6", edgecolors="none")
        lims = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
        ax1.plot(lims, lims, color="k", lw=0.8, ls="--")
        ss_res = float(((obs - pred) ** 2).sum())
        ss_tot = float(((obs - obs.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(((obs - pred) ** 2).mean()))
        ax1.set_xlabel("Observed (test set)")
        ax1.set_ylabel("Predicted")
        ax1.set_title(f"Parity — R²={r2:.3f}, RMSE={rmse:.2f}", fontsize=10, loc="left")
        ax1.grid(alpha=0.2)

        imp = self._feature_importance()
        if imp is not None:
            names, vals = imp
            order = np.argsort(vals)[-15:]
            ax2.barh([names[i] for i in order], [vals[i] for i in order], color="#2c7bb6")
            ax2.set_title("Feature importance", fontsize=10, loc="left")
        else:
            ax2.text(
                0.5,
                0.5,
                "Feature importance not available\nfor this backend",
                ha="center",
                va="center",
            )
            ax2.axis("off")
        fig.tight_layout()

        if r2 >= 0.7:
            verdict = (
                "ok",
                f"Good fit — test R² = {r2:.2f}. The normalisation results should be reliable.",
            )
        elif r2 >= 0.5:
            verdict = (
                "warn",
                f"Moderate fit — test R² = {r2:.2f}. Consider more features or a longer time budget.",
            )
        else:
            verdict = (
                "error",
                f"Poor fit — test R² = {r2:.2f}. Normalised results will be noisy; revisit features/target.",
            )

        backend = getattr(self.model, "backend", "?")
        self.tab_model.show_result(
            fig,
            verdict=verdict,
            lines=[
                f"Backend: {backend}   |   features: {len(self.trained_features)}   |   rows: {len(self.df_prep):,}"
            ],
            header=header,
            rows=rows,
        )
        self._log_history("train", f"R2={r2:.3f}")

    def _feature_importance(self):
        """Best-effort (names, values) from the trained model, else None."""
        model = self.model
        names = self.trained_features + list(TIME_VARS)
        try:
            if hasattr(model, "feature_importance"):  # LgbModel
                vals = list(model.feature_importance())
                names = list(getattr(model, "feature_names", names))
                if len(vals) == len(names):
                    return names, vals
            automl = getattr(model, "model", model)
            est = getattr(automl, "model", automl)  # flaml AutoML.model → sklearn wrapper
            fi = getattr(est, "feature_importances_", None)
            fn = getattr(automl, "feature_names_in_", None)
            if fn is None:
                fn = getattr(est, "feature_names_in_", None)
            if fi is not None:
                fn = list(fn) if fn is not None else names
                if len(fn) == len(fi):
                    return fn, list(fi)
        except Exception:
            pass
        return None

    def _run_normalise(self) -> None:
        if not self._check_model():
            return
        resample_vars = self._selected_norm_vars()
        if not resample_vars:
            QMessageBox.information(
                self,
                "No resample variables",
                "Tick at least one variable to resample in Step 2 "
                "(usually the meteorological ones).",
            )
            return
        from normet import normalise

        quantiles = (0.05, 0.25, 0.75, 0.95) if self.norm_quantiles.isChecked() else None
        self.runner.submit(
            "normalise",
            normalise,
            self._normalise_done,
            self._show_error,
            self.df_prep,
            self.model,
            feature_names=self.trained_features + list(TIME_VARS),
            variables_resample=resample_vars,
            n_samples=self.norm_samples.value(),
            n_cores=self._n_cores(self.norm_cores),
            return_quantiles=quantiles,
            verbose=True,
        )

    def _normalise_done(self, result: pd.DataFrame) -> None:
        self.results["normalise"] = result
        from normet import normalise_plot

        qcols = sorted(c for c in result.columns if c.startswith("q") and c[1:].isdigit())
        fig = normalise_plot(
            result,
            ci_low=qcols[0] if len(qcols) >= 2 else None,
            ci_high=qcols[-1] if len(qcols) >= 2 else None,
            ylabel=self.target_combo.currentText(),
            title=f"Meteorologically normalised {self.target_combo.currentText()}",
        )
        delta = result["normalised"].mean() - result["observed"].mean()
        n_rv = len(self._selected_norm_vars())
        self.tab_norm.show_result(
            fig,
            lines=[
                f"{len(result):,} timestamps, {self.norm_samples.value()} Monte-Carlo samples, "
                f"{n_rv} resampled variables   |   mean(normalised − observed) = {delta:+.2f}"
            ],
        )
        self._sync_enabled()
        self.tabs.setCurrentWidget(self.tab_norm)
        self.statusBar().showMessage(
            "Normalisation done — the meteorologically normalised series is on tab ③."
        )
        self._log_history("normalise", f"n_samples={self.norm_samples.value()}")
        log.info("Normalisation finished (%d rows)", len(result))

    def _run_decompose(self) -> None:
        if not self._check_model():
            return
        from normet import decompose

        self.runner.submit(
            f"decompose ({self.decom_method.currentText()})",
            decompose,
            self._decompose_done,
            self._show_error,
            self.df_prep,
            self.model,
            method=self.decom_method.currentText(),
            feature_names=self.trained_features + list(TIME_VARS),
            n_samples=self.decom_samples.value(),
        )

    def _decompose_done(self, result: pd.DataFrame) -> None:
        self.results["decompose"] = result
        method = self.decom_method.currentText()
        target = self.target_combo.currentText()

        from . import _decomp_plots as DP

        try:
            if method == "emission":
                fig = DP.emission_figure(result, target=target)
                d = DP.indexed_result(result)
                comps = [(c, lbl) for c, lbl in DP.EMI_COMPONENTS if c in d.columns]
                dominant = max(comps, key=lambda cl: d[cl[0]].std())[1] if comps else "n/a"
                lines = [
                    f"emi_total = emi_base + {' + '.join(c for c, _ in comps)} — "
                    f"largest time-scale: {dominant}."
                ]
            else:
                fig = DP.meteorology_figure(result, target=target)
                d = DP.indexed_result(result)
                contribs = DP.met_contribution_columns(d)
                dominant = max(contribs, key=lambda c: d[c].std()) if contribs else "n/a"
                lines = [
                    f"met_total = observed − normalised, split across {len(contribs)} "
                    f"meteorological variables — strongest: {dominant}."
                ]
            self.tab_decom.show_result(fig, lines=lines)
        except Exception:
            log.exception("Decomposition figure failed; falling back to a line plot")
            import matplotlib.pyplot as plt

            plot_df = result.set_index("date") if "date" in result.columns else result
            fig, ax = plt.subplots(figsize=(10, 4.5))
            for column in plot_df.columns:
                if pd.api.types.is_numeric_dtype(plot_df[column]):
                    ax.plot(plot_df.index, plot_df[column], lw=0.9, label=column)
            ax.legend(loc="upper right", fontsize=8, ncols=2)
            self.tab_decom.show_result(fig)

        self.tabs.setCurrentWidget(self.tab_decom)
        self.statusBar().showMessage("Decomposition done.")
        self._log_history("decompose", method)
        log.info("Decomposition finished")

    def _run_rolling(self) -> None:
        if not self._check_model():
            return
        from normet import rolling

        self.runner.submit(
            "rolling",
            rolling,
            self._rolling_done,
            self._show_error,
            self.df_prep,
            self.model,
            feature_names=self.trained_features + list(TIME_VARS),
            variables_resample=self._selected_norm_vars() or None,
            window_days=self.roll_window.value(),
            rolling_every=self.roll_step.value(),
            n_samples=self.roll_samples.value(),
        )

    def _rolling_done(self, result: pd.DataFrame) -> None:
        self.results["rolling"] = result
        import matplotlib.pyplot as plt

        rolling_cols = [c for c in result.columns if c.startswith("rolling_")]
        mean = result[rolling_cols].mean(axis=1, skipna=True)
        overlap = result[rolling_cols].notna().sum(axis=1)
        std = result[rolling_cols].std(axis=1, skipna=True).where(overlap >= 2)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        if "observed" in result.columns:
            ax.plot(
                result.index,
                result["observed"],
                lw=0.6,
                alpha=0.35,
                color="#9a9a9a",
                label="observed",
            )
        ax.plot(mean.index, mean.values, lw=1.3, color="#2c7bb6", label="rolling mean")
        if std.notna().any():
            ax.fill_between(
                mean.index,
                mean - std,
                mean + std,
                color="#2c7bb6",
                alpha=0.2,
                label="±1 SD across overlapping windows",
            )
        ax.set_title(
            f"Rolling normalisation — mean of {len(rolling_cols)} windows "
            f"({self.roll_window.value()} d window, {self.roll_step.value()} d step)",
            fontsize=10,
            loc="left",
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        self.tab_roll.show_result(fig)
        self.tabs.setCurrentWidget(self.tab_roll)
        self.statusBar().showMessage("Rolling normalisation done.")
        self._log_history("rolling", f"{len(rolling_cols)} windows")
        log.info("Rolling finished (%d windows)", len(rolling_cols))

    def _run_multiscale(self) -> None:
        if not self._check_model():
            return
        if "normalise" not in self.results:
            QMessageBox.information(
                self,
                "Run Step 2 first",
                "Multi-scale decomposition needs Y_∞ from Step 2 (Normalise) as its "
                "full-record baseline — run Normalise, then come back here.",
            )
            return
        from normet.gui._multiscale import compute_multiscale

        y_inf = self.results["normalise"]["normalised"]
        self.runner.submit(
            "multi-scale decomposition",
            compute_multiscale,
            self._multiscale_done,
            self._show_error,
            df_prep=self.df_prep,
            model=self.model,
            feature_names=self.trained_features + list(TIME_VARS),
            variables_resample=self._selected_norm_vars() or None,
            y_inf=y_inf,
            fast_days=self.ms_fast.value(),
            meso_days=self.ms_meso.value(),
            slow_days=self.ms_slow.value(),
            n_samples=self.roll_samples.value(),
            n_cores=self._n_cores(self.norm_cores),
        )

    def _multiscale_done(self, result: dict) -> None:
        from ._multiscale_plots import available_bands, multiscale_figure

        fig = multiscale_figure(result, target=self.target_combo.currentText())
        bands = available_bands(result)
        lines = list(result.get("notes") or [])

        if bands:
            sigmas = {key: float(result[key].std()) for key, _ in bands}
            dominant = max(sigmas, key=sigmas.get)
            verdict = (
                "ok",
                f"Dominant residual timescale: {dominant} (σ={sigmas[dominant]:.2f}) — "
                f"{dict(bands)[dominant]}",
            )
        else:
            verdict = (
                "warn",
                "No band was computable — see the notes for why each scale was skipped.",
            )

        self.tab_multiscale.show_result(fig, verdict=verdict, lines=lines)
        self.tabs.setCurrentWidget(self.tab_multiscale)

        export = {k: v for k, v in result.items() if isinstance(v, pd.Series)}
        if export:
            self.results["multiscale"] = pd.DataFrame(export)

        self.statusBar().showMessage("Multi-scale decomposition done.")
        self._log_history("multiscale", f"{len(bands)}/3 bands computed")
        log.info("Multi-scale decomposition finished (%d/3 bands)", len(bands))

    def _run_pdp(self) -> None:
        if not self._check_model():
            return
        variables = _checked_items(self.pdp_vars)
        if not variables:
            QMessageBox.information(self, "No variables", "Tick at least one variable in Step 5.")
            return
        from normet import pdp

        self.runner.submit(
            "pdp",
            pdp,
            self._pdp_done,
            self._show_error,
            self.df_prep,
            self.model,
            var_list=variables,
            grid_points=self.pdp_grid.value(),
        )

    def _pdp_done(self, result: pd.DataFrame) -> None:
        self.results["pdp"] = result
        from normet import pdp_grid as _pdp_grid

        fig = _pdp_grid(result, title="Partial dependence")
        self.tab_pdp.show_result(fig)
        self.tabs.setCurrentWidget(self.tab_pdp)
        self.statusBar().showMessage("PDP done.")
        self._log_history("pdp", f"{result['variable'].nunique()} variables")
        log.info("PDP finished")

    # ---------------------------------------------------------------- SCM
    def open_scm_studio(self) -> None:
        from .scm_window import SCMWindow

        if self._scm_window is None:
            self._scm_window = SCMWindow(parent=self)
        # Offer the loaded data if it looks like a panel (a repeated-unit column).
        if self.df_raw is not None:
            self._scm_window.offer_main_data(self.df_raw)
        self._scm_window.show()
        self._scm_window.raise_()
        self._scm_window.activateWindow()

    # ---------------------------------------------------------- Data Studio
    def open_data_studio(self) -> None:
        from .data_window import DataWindow

        if self._data_window is None:
            self._data_window = DataWindow(parent=self)
        self._data_window.show()
        self._data_window.raise_()
        self._data_window.activateWindow()

    def ingest_dataframe(self, df: pd.DataFrame, label: str) -> None:
        """Load an in-memory table (e.g. from the Data Studio) as the dataset."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        self._ingest(df, label)

    # ---------------------------------------------------------- Transport Studio
    def open_transport_studio(self) -> None:
        from .trajectory_window import TrajectoryWindow

        if self._transport_window is None:
            self._transport_window = TrajectoryWindow(parent=self)
        if self.df_raw is not None:
            self._transport_window.offer_main_data(self.df_raw)
        self._transport_window.show()
        self._transport_window.raise_()
        self._transport_window.activateWindow()

    def merge_trajectory_features(self, features: pd.DataFrame) -> None:
        """Join transport-aware ``traj_*`` columns (indexed by receptor time)
        onto the loaded dataset by nearest hour, then re-select them as
        features so they show up ticked in Step 1.

        When *features* carries a ``site`` column (multi-receptor Transport
        Studio runs) and the loaded dataset has one too, the join is done
        per site so each row only picks up its own site's trajectory
        features; otherwise it falls back to a plain nearest-hour join
        (the usual single-site case).
        """
        if self.df_raw is None:
            QMessageBox.information(self, "No data", "Load a dataset in the main window first.")
            return
        traj_cols = [c for c in features.columns if c != "site"]

        if "site" in features.columns and "site" in self.df_raw.columns:
            main_sites = set(self.df_raw["site"].unique())
            feat_sites = set(features["site"].unique())
            missing = feat_sites - main_sites
            if missing:
                log.warning(
                    "Trajectory features for site(s) %s have no matching rows in the main "
                    "dataset — skipped",
                    ", ".join(sorted(missing)),
                )
            parts = []
            for site, sub_main in self.df_raw.set_index("date").groupby("site"):
                sub_feats = features[features["site"] == site].drop(columns="site")
                if sub_feats.empty:
                    parts.append(sub_main)
                    continue
                parts.append(
                    pd.merge_asof(
                        sub_main.sort_index(),
                        sub_feats.sort_index(),
                        left_index=True,
                        right_index=True,
                        direction="nearest",
                    )
                )
            merged = pd.concat(parts).sort_index()
        else:
            df = self.df_raw.set_index("date").sort_index()
            feats = features.drop(columns="site", errors="ignore").sort_index()
            merged = pd.merge_asof(
                df, feats, left_index=True, right_index=True, direction="nearest"
            )

        new_cols = [c for c in traj_cols if c not in self.df_raw.columns]
        self._ingest(
            merged.reset_index(), f"{self.file_label.text().splitlines()[0]} + transport features"
        )
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if item.text() in new_cols:
                item.setCheckState(Qt.CheckState.Checked)
        log.info("Merged %d transport feature column(s) onto the dataset", len(new_cols))

    # ------------------------------------------------------------- misc
    def _about(self) -> None:
        import normet

        QMessageBox.about(
            self,
            "About Normet",
            f"<b>normet {normet.__version__}</b><br>"
            "Normalisation, decomposition and counterfactual modelling "
            "for environmental time series.<br>"
            f'<a href="{DOCS_URL}">{DOCS_URL.removeprefix("https://")}</a>',
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        QSettings("apai-sys", "normet-gui").setValue("geometry", self.saveGeometry())
        logging.getLogger("normet").removeHandler(self._log_handler)
        self.runner.shutdown()
        if self._scm_window is not None:
            self._scm_window.close()
        if self._data_window is not None:
            self._data_window.close()
        super().closeEvent(event)
