"""Main window of the normet Qt GUI.

Design language: a parameter panel on the left walks the user through the
workflow a page per step (Data → Model → Normalise → Decompose → Rolling →
PDP, chosen from a bar above them), each step with its own
prominent "▶ Run" button; results land in tabs on the right, which auto-activate when a run
finishes.  Verdict banners (green / amber / red) summarise quality at a
glance; every long computation runs on a background thread; the bottom dock
streams normet's log. Before a table is open the Data tab shows the welcome
(:mod:`.welcome`); Ctrl+K (⌘K) opens the command palette (:mod:`.palette`);
the window and its figures follow the system's light/dark appearance unless
View → Appearance pins one (:mod:`.theme`).

The synthetic-control workflow (panel data, treated unit vs. donors) has
genuinely different inputs, so it lives in its own window — see
:mod:`.scm_window` — opened from the toolbar button or the Analysis menu.

The chrome around all of this — menu, tabs, docks, status bar, command palette
— is in :mod:`._shell`; the left panel is built by :mod:`._panel_inputs`,
:mod:`._panel_train` and :mod:`._panel_analyse`; the run handlers are in
:mod:`._run_train` and :mod:`._run_analyse`, the background tasks in
:mod:`._tasks`, config, history and export in :mod:`._session`, and the
windows opened from here in :mod:`._studios`. Each is a mixin, so every widget
and handler is still an attribute of the window. What is left here is the
window's state and the order it is built in.
"""

from __future__ import annotations

import logging

import pandas as pd
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QWidget,
)

from . import commandlog
from ._clock import TaskClock
from ._constants import MET_DEFAULTS, TIME_VARS  # noqa: F401 - still importable from here
from ._lists import _checked_items, _fill_checklist, parse_groups  # noqa: F401 - likewise
from ._panel_analyse import AnalysePanelMixin
from ._panel_inputs import InputsMixin
from ._panel_train import TrainPanelMixin
from ._run_analyse import AnalyseRunMixin
from ._run_train import TrainRunMixin
from ._saved import SavedResultsMixin
from ._session import SessionMixin
from ._shell import ShellMixin
from ._studios import StudiosMixin
from ._tasks import TaskMixin
from .workers import TaskRunner

log = logging.getLogger(__name__)


class MainWindow(
    InputsMixin,
    TrainPanelMixin,
    AnalysePanelMixin,
    TrainRunMixin,
    AnalyseRunMixin,
    TaskMixin,
    SessionMixin,
    SavedResultsMixin,
    StudiosMixin,
    ShellMixin,
):
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
        #: The multi-scale run as it came back (its series and notes), for a saved session.
        self._multiscale_raw: dict | None = None
        self._restoring = False  # a saved session being read back: no history, no autosave
        #: Each step's call as it ran (its plain arguments), by function name: the Methods
        #: paragraph is written from these, not from the panel as it stands.
        self._ran: dict[str, dict] = {}
        #: A session script being run: its calls and its file's name (._saved).
        self._pending_script: tuple[dict, str] = ({}, "")
        #: Chronos-2's covariate-sensitivity check of the model loaded: (numbers, verdict).
        self.chronos_check: tuple[dict, tuple[str, str]] | None = None
        self._scm_window = None
        self._data_window = None
        self._transport_window = None
        self._polar_window = None

        self.runner = TaskRunner(self)
        self.runner.started.connect(self._task_started)
        self.runner.finished.connect(self._task_finished)
        self.task_clock = TaskClock(self, self.runner)
        self.command_log = commandlog.CommandLog()
        self.runner.record = self._record_task

        self._build_central()
        self._build_log()
        self._build_command_log()
        self._build_assistant()
        self._build_status_bar()
        self._build_menu()
        self._sync_enabled()

        settings = QSettings("apai-sys", "normet-gui")
        geo = settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        self.statusBar().showMessage("Open data, or load an example from the File menu, to begin.")

    # --------------------------------------------------------- left panel
    #: The left panel's pages, each a step of the workflow: one column
    #: holding every box was four screens long, and the run a step needed was a scroll
    #: away from the settings it reads.
    PANEL_PAGES = ("Data", "Model", "Normalise", "Decompose", "Rolling", "PDP")

    def _build_param_panel(self) -> QWidget:
        """The left column: a page per workflow step, chosen from a bar above them
        (:class:`._pages.StepPanel`).

        Each box is built by its own method below, so a step's controls stay
        findable by name rather than by scrolling through four hundred lines
        of layout.  The order here is the order the workflow runs in, and
        ``_run_buttons`` collects the buttons a running task greys out.
        """
        from ._pages import StepPanel

        panel = StepPanel(
            list(
                zip(
                    self.PANEL_PAGES,
                    (
                        (self._build_data_box(), self._build_columns_box()),
                        (self._build_train_box(),),
                        (self._build_normalise_box(),),
                        (self._build_decompose_box(), self._build_multiscale_box()),
                        (self._build_rolling_box(),),
                        (self._build_pdp_box(),),
                    ),
                    strict=True,
                )
            )
        )
        self._run_buttons = [
            self.train_button,
            self.ft_button,
            self.norm_button,
            self.decom_button,
            self.roll_button,
            self.ms_button,
            self.pdp_button,
        ]
        self.panel_stack, self.panel_bar, self.param_scroll = panel.stack, panel.bar, panel.scroll
        self.param_panel = panel  # folded and unfolded by toggle_panel (._shell)
        return panel

    def _show_panel_index(self, i: int) -> None:
        self.param_panel.show_index(i)

    def show_panel_page(self, name: str) -> None:
        """The panel's page ``name``: one of PANEL_PAGES."""
        self.param_panel.show_page(name)

    def closeEvent(self, event) -> None:  # noqa: N802
        QSettings("apai-sys", "normet-gui").setValue("geometry", self.saveGeometry())
        logging.getLogger("normet").removeHandler(self._log_handler)
        self.assistant.shutdown()
        self.runner.shutdown()
        if self._scm_window is not None:
            self._scm_window.close()
        if self._data_window is not None:
            self._data_window.close()
        if self._transport_window is not None:
            self._transport_window.close()  # stops a HYSPLIT run before the app goes
        super().closeEvent(event)
