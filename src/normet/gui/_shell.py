"""The window's chrome: menu, tabs, docks, status bar, palette, appearance.

Split out of :mod:`.main_window` so the class that wires the workflow is not
also the class that builds every piece of furniture around it.  The methods are
unchanged; they are mixed into :class:`~.main_window.MainWindow` alongside the
panel mixins, and reach their widgets the same way the window always did.
"""

from __future__ import annotations

import logging
import os

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QActionGroup, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import _settings, commandlog, theme
from ._constants import DOCS_URL
from ._data_tab import DataTab
from ._widgets import CanvasTab, FlowLayout
from .welcome import Welcome
from .workers import QtLogHandler


def report_language_choice() -> str:
    """File -> Report Language as chosen: "window" (the default), "en" or "zh"."""
    from PySide6.QtCore import QSettings

    c = str(QSettings("apai-sys", "normet-gui").value("report_language", "window"))
    return c if c in ("window", "en", "zh") else "window"


def set_report_language(choice: str) -> None:
    from PySide6.QtCore import QSettings

    QSettings("apai-sys", "normet-gui").setValue("report_language", choice)


def report_language() -> str:
    """The language a report is written in now: the choice, else the window's."""
    choice = report_language_choice()
    if choice != "window":
        return choice
    from . import uilang

    return uilang.language()


def _menu_actions(menu) -> list:  # noqa: ANN001
    """A menu's actions, its submenus' included."""
    out = []
    for act in menu.actions():
        try:
            sub = act.menu()
        except RuntimeError:  # an action a rebuilt menu (Open Recent) has let go
            continue
        if sub is not None:
            out += _menu_actions(sub)
        elif not act.isSeparator():
            out.append(act)
    return out


#: The status bar's lights: (what has a result, its name).
STEP_LIGHTS = (
    ("data", "Data"),
    ("model", "Model"),
    ("normalise", "Normalised"),
    ("decompose", "Decomposed"),
    ("rolling", "Rolling"),
    ("multiscale", "Multi-scale"),
    ("pdp", "PDP"),
)
LIGHT_ON, LIGHT_OFF = "#34c759", "#aeaeb2"


class ShellMixin(QMainWindow):
    def _build_menu(self) -> None:
        mb = self.menuBar()

        fm = mb.addMenu("&File")
        act = fm.addAction("&Open Data…", self._open_csv)
        act.setShortcut("Ctrl+O")
        # The examples, each set up as its source sets it up (._examples), and the SCM
        # Studio's own.
        from ._examples import EXAMPLES

        ex_menu = fm.addMenu("Load &Example")
        for ex in EXAMPLES.values():
            if ex.key == "synthetic":
                ex_menu.addSeparator()
            act = ex_menu.addAction(ex.title, lambda key=ex.key: self.load_example(key))
            act.setToolTip(ex.description)
        ex_menu.addSeparator()
        act = ex_menu.addAction("Synthetic control panel (SCM Studio)", self.open_scm_example)
        act.setToolTip("Opens the SCM Studio with its example panel loaded.")
        ex_menu.setToolTipsVisible(True)
        act = fm.addAction("Get &UK Data (AQ networks + met)…", self.open_data_studio)
        act.setShortcut("Ctrl+D")
        self.recent_menu = fm.addMenu("Open &Recent")
        self._rebuild_recent_menu()
        fm.addSeparator()
        act = fm.addAction("&Save Results…", self.save_results)
        act.setShortcut("Ctrl+S")
        fm.addAction("&Load Results…", lambda: self.load_results())
        fm.addAction("Reopen &Last Session", self.reopen_last_session)
        fm.addAction("Load Session S&cript…", lambda: self.load_session_script())
        fm.addSeparator()
        fm.addAction("&Methods Paragraph…", self.show_methods_paragraph)
        self._export_action = fm.addAction("&Export Current Result…", self._export_current)
        self._export_action.setShortcut("Ctrl+E")
        self._export_action.setEnabled(False)
        self._report_action = fm.addAction("Export &HTML Report…", self._export_html)
        # The report's language: the window's unless chosen (kept between sessions).
        rl = fm.addMenu("Report &Language")
        group = QActionGroup(self)
        kept = report_language_choice()
        for code, label in (("window", "As the Window"), ("en", "English"), ("zh", "中文")):
            act = rl.addAction(label)
            act.setCheckable(True)
            act.setChecked(code == kept)
            act.triggered.connect(lambda _checked=False, c=code: set_report_language(c))
            group.addAction(act)
        self._report_action.setEnabled(False)
        fm.addAction("Save Session as &Script…", self.save_script)
        fm.addSeparator()
        fm.addAction("Save Con&fig…", self.save_config)
        fm.addAction("Load Confi&g…", self.load_config)
        fm.addAction("Run &History…", self.show_history)
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
        act = vm.addAction("Collapse / Expand &Panel", lambda: self.toggle_panel())
        act.setShortcut("Ctrl+B")
        vm.addSeparator()
        vm.addAction("&Log…", lambda: self.show_log_page(0))
        vm.addAction("Command &Log…", lambda: self.show_log_page(1))
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
        from . import uilang

        language = vm.addMenu("&Language")
        group = QActionGroup(self)
        for code, label in (("system", "Follow System"), *uilang.LANGUAGES.items()):
            act = language.addAction(label)
            act.setCheckable(True)
            act.setChecked(code == uilang.choice())
            act.triggered.connect(lambda _checked=False, c=code: uilang.set_choice(c))
            group.addAction(act)

        am = mb.addMenu("&Analysis")
        act = am.addAction("Synthetic Control (SCM) Studio…", self.open_scm_studio)
        act.setShortcut("Ctrl+Shift+S")
        am.addAction("&Polar Plot…", self.open_polar_plot)

        wm = mb.addMenu("&Window")
        wm.addAction("Minimize", self.showMinimized).setShortcut("Ctrl+M")
        wm.addAction("Zoom", self.showMaximized)

        hm = mb.addMenu("&Help")
        act = hm.addAction("&User Manual", self.open_manual)
        act.setShortcut("F1")
        hm.addAction("&Keyboard Shortcuts…", self.show_shortcuts)
        hm.addAction("Documentation", lambda: QDesktopServices.openUrl(QUrl(DOCS_URL)))
        hm.addAction("&About Normet", self._about)

    def notify(
        self,
        text: str,
        *,
        key: str | None = None,
        actions: tuple = (),
        kind: str = "info",
    ) -> None:
        """A notice above the results (.notices): what wants attention, with the step
        that answers it on a button -- where a dialog would stop the window."""
        self.notices.notify(key or text, text, actions, kind)

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
        from .notices import NoticeStack

        # What wants attention, above the results: a step to take first, with its button.
        self.notices = NoticeStack()
        rv.addWidget(self.notices)
        rv.addWidget(self._build_tabs(), 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        # The panel opens as wide as its pages need (StepPanel.least_width), so it meets
        # the results with no gap; dragged wider, it widens with the handle.
        splitter.setCollapsible(0, False)
        splitter.setSizes([self.param_panel.least_width, 1010])
        self.setCentralWidget(splitter)

    def _build_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()
        # Pages on the window's own ground, as the toolbars above their figures are: a
        # framed pane drew them lighter, and each toolbar as a grey band across it.
        self.tabs.setDocumentMode(True)

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
        self.data_summary = QLabel("Open data, or load an example from the File menu, to begin.")
        self.data_summary.setWordWrap(True)
        dl.addWidget(self.data_summary)
        # The table at a glance, its cycles, its winds, its statistics -- and the rows
        # left out of every step (._data_tab).
        self.data_tab = DataTab()
        self.data_tab.exclusionsChanged.connect(self._set_exclusions)
        #: The overview's figure, as the window has always named its data preview.
        self.data_preview = self.data_tab.overview
        dl.addWidget(self.data_tab, 1)
        self.tabs.addTab(data_tab, "Data")

        self.tab_model = CanvasTab("Train a model to see its quality here.")
        self.tabs.addTab(self.tab_model, "Model")
        self.tab_norm = CanvasTab(
            "Run the normalisation to see observed vs. meteorologically normalised series."
        )
        self.tabs.addTab(self.tab_norm, "Normalise")
        self.tab_decom = CanvasTab(
            "Run a decomposition to see it here — emission splits the "
            "normalised series into time scales; meteorology splits "
            "observed − normalised by met variable."
        )
        self.tabs.addTab(self.tab_decom, "Decompose")
        self.tab_roll = CanvasTab("Run the rolling normalisation to see it here.")
        self.tabs.addTab(self.tab_roll, "Rolling")
        self.tab_multiscale = CanvasTab(
            "Run the normalisation, then Multi-scale decomposition, to see "
            "how meteorological residuals split across fast/meso/slow "
            "timescales."
        )
        self.tabs.addTab(self.tab_multiscale, "Multi-scale")
        self.tab_pdp = CanvasTab("Compute PDP to see partial-dependence curves.")
        self.tabs.addTab(self.tab_pdp, "PDP")
        self._sync_welcome()
        return self.tabs

    def _sync_welcome(self) -> None:
        """The welcome on the Data tab while no table is open; the preview once one is."""
        empty = self.df_raw is None
        if empty:
            self.welcome.refresh(_settings.read_recent())
        self.welcome.setVisible(empty)
        self._data_parts.setVisible(not empty)

    def _log_window(self, title: str, body: QWidget) -> QDialog:
        """A window of the main window's own, opened from the View menu and left open
        beside it as the analysis goes on."""
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setModal(False)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(body)
        dlg.resize(720, 420)
        return dlg

    def _build_log(self) -> None:
        """View → Log: what the analysis has said, in a window of its own."""
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5_000)
        self._log_windows: list[QDialog] = [self._log_window("Log", self.log_view)]

        handler = QtLogHandler()
        handler.message.connect(self.log_view.appendPlainText)
        handler.setLevel(logging.INFO)
        # normet.gui.* loggers propagate to "normet", so one handler suffices.
        logging.getLogger("normet").addHandler(handler)
        logging.getLogger("normet").setLevel(logging.INFO)
        self._log_handler = handler

    def _build_command_log(self) -> None:
        """View → Command Log: the session as the normet calls that repeat it."""
        from PySide6.QtGui import QFontDatabase

        body = QWidget()
        v = QVBoxLayout(body)
        v.setContentsMargins(0, 0, 0, 0)
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
        self._log_windows.append(self._log_window("Command Log", body))
        self._refresh_command_log()

    def show_log_page(self, index: int) -> None:
        """The log (0) or the command log (1), its window opened and brought forward."""
        dlg = self._log_windows[index]
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

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

    def toggle_panel(self, show: bool | None = None) -> None:
        """Fold the parameter panel away, or bring it back (``show`` forces one)."""
        if show is None:
            show = self.param_panel.isHidden()
        self.param_panel.setVisible(show)
        self.panel_btn.setText("◀ Collapse Panel" if show else "▶ Expand Panel")

    def start_layout(self) -> None:
        """The layout the window opens in: the parameter panel
        folded away and the Assistant beside the results. Expand Panel (Ctrl+B)
        brings the panel back; View → Assistant closes the Assistant."""
        self.toggle_panel(False)
        self.show_assistant()
        width = max(self.assistant.minimumSizeHint().width(), 400)
        self.resizeDocks([self.assistant], [width], Qt.Orientation.Horizontal)

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
        # How far the analysis has come, one light per step.
        self._step_lights: dict[str, QLabel] = {}
        lights = QWidget()
        row = QHBoxLayout(lights)
        row.setContentsMargins(0, 0, 6, 0)
        row.setSpacing(8)
        for key, _name in STEP_LIGHTS:
            lbl = QLabel()
            lbl.setTextFormat(Qt.TextFormat.RichText)
            row.addWidget(lbl)
            self._step_lights[key] = lbl
        self.statusBar().addPermanentWidget(lights)
        self.statusBar().addPermanentWidget(self.cancel_btn)
        self.statusBar().addPermanentWidget(self.progress)
        self._update_step_lights()

    def _update_step_lights(self) -> None:
        """Each step's light: green once it has a result, grey until then."""
        done = {
            "data": self.df_raw is not None,
            "model": self.model is not None,
            **{key: key in self.results for key, _ in STEP_LIGHTS[2:]},
        }
        for key, name in STEP_LIGHTS:
            colour = LIGHT_ON if done[key] else LIGHT_OFF
            lbl = self._step_lights[key]
            lbl.setText(f'<span style="color:{colour}">●</span> {name}')
            lbl.setToolTip(f"{name}: {'done' if done[key] else 'not run yet'}")

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

    def open_manual(self) -> None:
        """Help -> User Manual: the PDF in the window's language (.manual)."""
        from . import uilang
        from .manual import manual_location

        QDesktopServices.openUrl(manual_location(uilang.language()))

    def show_shortcuts(self) -> None:
        """Help -> Keyboard Shortcuts: every command with a shortcut, read from the menus."""
        from PySide6.QtWidgets import QDialog, QTableWidget, QTableWidgetItem

        rows = []
        for top in self.menuBar().actions():
            menu = top.menu()
            if menu is None:
                continue
            for act in _menu_actions(menu):
                try:
                    key = act.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
                    name = act.text().replace("&", "")
                except RuntimeError:  # an action a rebuilt menu (Open Recent) has let go
                    continue
                if key:
                    rows.append((top.text().replace("&", ""), name, key))
        dlg = QDialog(self)
        dlg.setWindowTitle("Keyboard shortcuts")
        dlg.resize(520, 440)
        lay = QVBoxLayout(dlg)
        table = QTableWidget(len(rows), 3)
        table.setHorizontalHeaderLabels(["Menu", "Command", "Shortcut"])
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                table.setItem(i, j, QTableWidgetItem(value))
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(table)
        self._shortcuts_dialog = dlg
        dlg.open()

    def _about(self) -> None:
        """Help → About: the app's icon, its name and version, what it does, and where
        its documentation is."""
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import QDialog

        import normet

        from ._widgets import app_icon_path

        dlg = QDialog(self)
        dlg.setWindowTitle("About Normet")
        dlg.setMinimumWidth(380)
        v = QVBoxLayout(dlg)
        v.setContentsMargins(36, 30, 36, 30)
        v.setSpacing(14)

        icon = QLabel()
        pix = QPixmap(app_icon_path())
        if not pix.isNull():
            ratio = dlg.devicePixelRatioF()  # sharp on a Retina screen
            pix = pix.scaled(
                round(96 * ratio),
                round(96 * ratio),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            pix.setDevicePixelRatio(ratio)
            icon.setPixmap(pix)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(icon)

        title = QLabel("Normet")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 15px; font-weight: 600;")
        v.addWidget(title)

        version = QLabel(f"Version {normet.__version__}")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version.setStyleSheet("font-size: 11px; color: gray;")
        v.addWidget(version)

        what = QLabel(
            "Normalisation, decomposition and counterfactual modelling "
            "for environmental time series."
        )
        what.setWordWrap(True)
        what.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(what)

        link = QLabel(f'<a href="{DOCS_URL}">{DOCS_URL.removeprefix("https://")}</a>')
        link.setOpenExternalLinks(True)
        link.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(link)

        self._about_dialog = dlg
        dlg.open()

    def _build_action_bar(self) -> QWidget:
        bar = QWidget()
        # Wraps onto a second row when the window is narrow: in one row the
        # five buttons held the window at least ~1050 px wide.
        h = FlowLayout(bar, spacing=6)
        h.setContentsMargins(4, 0, 4, 0)
        self.panel_btn = QPushButton("◀ Collapse Panel")
        self.panel_btn.setToolTip("Fold the parameter panel away or bring it back (Ctrl+B / ⌘B).")
        self.panel_btn.clicked.connect(lambda: self.toggle_panel())
        h.addWidget(self.panel_btn)
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
            "Fetch UK air quality (AURN, AQE, SAQN, WAQN, NI, LMAM) plus\n"
            "reanalysis meteorology and merge them into a model-ready table\n"
            "(no API key)."
        )
        self.data_btn.clicked.connect(self.open_data_studio)
        self.transport_btn = QPushButton("🧭  Transport Studio")
        self.transport_btn.setToolTip(
            "Build transport-aware predictors (inflow direction, transport\n"
            "distance/speed, residence time over source regions) from HYSPLIT\n"
            "back-trajectory (tdump) output, and join them onto this data."
        )
        self.transport_btn.clicked.connect(self.open_transport_studio)
        # The studios side by side, then what is done with a result.
        h.addWidget(self.data_btn)
        h.addWidget(self.transport_btn)
        h.addWidget(self.scm_btn)
        h.addWidget(self.exp_csv)
        h.addWidget(self.exp_html)
        return bar
