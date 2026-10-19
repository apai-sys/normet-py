"""The window's chrome: menu, tabs, docks, status bar, palette, appearance.

Split out of :mod:`.main_window` so the class that wires the workflow is not
also the class that builds every piece of furniture around it.  The methods are
unchanged; they reach the rest of the window through
:class:`._contract.WindowContract`, which declares the attributes and the
handlers they call.
"""

from __future__ import annotations

import logging
import os

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QActionGroup, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import _settings, commandlog, theme
from ._constants import DOCS_URL
from ._contract import WindowContract
from ._widgets import CanvasTab
from .dataframe_model import DataFrameModel
from .welcome import Welcome
from .workers import QtLogHandler


class ShellMixin(WindowContract):
    def _build_menu(self) -> None:
        mb = self.menuBar()

        fm = mb.addMenu("&File")
        act = fm.addAction("&Open CSV…", self._open_csv)
        act.setShortcut("Ctrl+O")
        fm.addAction("Load &Example Data", self.load_example)
        act = fm.addAction("Get &UK Data (AQ networks + met)…", self.open_data_studio)
        act.setShortcut("Ctrl+D")
        self.recent_menu = fm.addMenu("Open &Recent")
        self._rebuild_recent_menu()
        fm.addSeparator()
        self._export_action = fm.addAction("&Export Current Result…", self._export_current)
        self._export_action.setShortcut("Ctrl+E")
        self._export_action.setEnabled(False)
        self._report_action = fm.addAction("Export &HTML Report…", self._export_html)
        self._report_action.setEnabled(False)
        fm.addAction("Save Session as &Script…", self.save_script)
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
        self._palette_action = vm.addAction("&Command Palette…", self.show_command_palette)
        self._palette_action.setShortcut("Ctrl+K")
        vm.addSeparator()
        vm.addAction("Run &History…", self.show_history)
        vm.addSeparator()
        vm.addAction(self._log_dock.toggleViewAction())
        act = self._command_dock.toggleViewAction()
        act.setText("Command &Log")
        vm.addAction(act)
        act = self.assistant.toggleViewAction()
        act.setText("&Assistant")
        act.setShortcut("Ctrl+Shift+A")
        vm.addAction(act)
        appearance = vm.addMenu("&Appearance")
        group = QActionGroup(self)
        current = theme.stored_appearance()
        for key, label in (("system", "Follow System"), ("light", "Light"), ("dark", "Dark")):
            act = appearance.addAction(label)
            act.setCheckable(True)
            act.setChecked(key == current)
            act.triggered.connect(
                lambda _checked=False, k=key: theme.theme_manager().set_appearance(k)
            )
            group.addAction(act)
        appearance.setEnabled(theme.can_set_appearance())

        am = mb.addMenu("&Analysis")
        act = am.addAction("Synthetic Control (SCM) Studio…", self.open_scm_studio)
        act.setShortcut("Ctrl+Shift+S")

        wm = mb.addMenu("&Window")
        wm.addAction("Minimize", self.showMinimized).setShortcut("Ctrl+M")
        wm.addAction("Zoom", self.showMaximized)

        hm = mb.addMenu("&Help")
        hm.addAction("Documentation", lambda: QDesktopServices.openUrl(QUrl(DOCS_URL)))
        hm.addAction("&About Normet", self._about)

    def show_command_palette(self) -> None:
        """Every menu command and every tab, found by typing (Ctrl+K)."""
        from .palette import open_palette

        open_palette(self, skip=(self._palette_action,))

    def _clip(self, method: str) -> None:
        w = self.focusWidget()
        if w is not None and hasattr(w, method):
            getattr(w, method)()

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

    def _build_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()

        # Data preview tab: the welcome until a table is open, then its preview.
        data_tab = QWidget()
        outer = QVBoxLayout(data_tab)
        outer.setContentsMargins(0, 0, 0, 0)
        self.welcome = Welcome(
            {
                "open_csv": self._open_csv,
                "open_path": lambda p: self.load_csv(p, remember=True),
                "load_example": self.load_example,
                "scm_example": self.open_scm_example,
                "data_studio": self.open_data_studio,
                "transport_studio": self.open_transport_studio,
                "assistant": self.show_assistant,
            }
        )
        outer.addWidget(self.welcome, 1)
        self._data_parts = QWidget()
        outer.addWidget(self._data_parts, 1)
        dl = QVBoxLayout(self._data_parts)
        dl.setContentsMargins(0, 0, 0, 0)
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
        self._sync_welcome()
        return self.tabs

    def _sync_welcome(self) -> None:
        """The welcome on the Data tab while no table is open; the preview once one is."""
        empty = self.df_raw is None
        if empty:
            self.welcome.refresh(_settings.read_recent())
        self.welcome.setVisible(empty)
        self._data_parts.setVisible(not empty)

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

    def _build_command_log_dock(self) -> None:
        """View → Command Log: the session as the normet calls that repeat it."""
        from PySide6.QtGui import QFontDatabase

        dock = QDockWidget("Command Log", self)
        dock.setObjectName("commandLogDock")
        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(4, 4, 4, 4)
        self.command_view = QPlainTextEdit()
        self.command_view.setReadOnly(True)
        self.command_view.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        v.addWidget(self.command_view, 1)
        row = QHBoxLayout()
        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(
            lambda: QApplication.clipboard().setText(self.command_log.script())
        )
        save_btn = QPushButton("Save as Script…")
        save_btn.clicked.connect(self.save_script)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_command_log)
        for b in (copy_btn, save_btn, clear_btn):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        dock.setWidget(body)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.tabifyDockWidget(self._log_dock, dock)
        self._log_dock.raise_()
        self._command_dock = dock
        self._refresh_command_log()

    def _refresh_command_log(self) -> None:
        self.command_view.setPlainText(self.command_log.text())
        self.command_view.verticalScrollBar().setValue(
            self.command_view.verticalScrollBar().maximum()
        )

    def _log_command(self, entry: commandlog.Entry | None) -> None:
        self.command_log.add(entry)
        self._refresh_command_log()

    def _clear_command_log(self) -> None:
        self.command_log.clear()
        self._refresh_command_log()

    def _build_assistant(self) -> None:
        """View → Assistant: a language model working in the window (:mod:`.assistant`)."""
        from .assistant import AssistantDock

        self.assistant = AssistantDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.assistant)
        self.assistant.hide()

    def show_assistant(self) -> None:
        self.assistant.show()
        self.assistant.raise_()
        self.assistant.composer.setFocus()

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

    def _rebuild_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = _settings.read_recent()
        if hasattr(self, "welcome"):
            self._sync_welcome()
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
