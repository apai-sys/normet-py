"""What every part of the main window may assume about the others.

``MainWindow`` is assembled from mixins -- :class:`._shell.ShellMixin` today,
one per workflow step as the split continues.  A mixin is type-checked on its
own, so a method there that reads ``self.df_raw`` or calls ``self.load_csv``
needs those declared on something it inherits.  This is that something.

It inherits ``QMainWindow`` so that a mixin *is* a window to the type checker:
without it, ``self.statusBar()`` inside a mixin would be an error, and every
mixin would have to re-declare Qt's API.  At runtime the class adds nothing.
The annotations create no attributes, and the methods below exist only under
``TYPE_CHECKING`` -- so a call to one that no mixin defines raises
``AttributeError`` instead of quietly doing nothing.

Adding a name here is the cost of sharing it across mixins; that is deliberate,
and it is also the list of coupling left to unpick if the window is ever
rebuilt as collaborating widgets rather than one object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDockWidget,
    QLabel,
    QMainWindow,
    QMenu,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableView,
    QTabWidget,
    QWidget,
)

from . import commandlog
from ._widgets import CanvasTab
from .dataframe_model import DataFrameModel
from .welcome import Welcome

if TYPE_CHECKING:
    from .assistant import AssistantDock
    from .workers import QtLogHandler, TaskRunner


class WindowContract(QMainWindow):
    # ---- what the window holds
    df_raw: pd.DataFrame | None
    runner: TaskRunner
    command_log: commandlog.CommandLog
    _log_handler: QtLogHandler

    # ---- the docks, the status bar and the assistant
    _log_dock: QDockWidget
    _command_dock: QDockWidget
    log_view: QPlainTextEdit
    command_view: QPlainTextEdit
    assistant: AssistantDock
    progress: QProgressBar
    cancel_btn: QPushButton

    # ---- the Data tab
    tabs: QTabWidget
    welcome: Welcome
    _data_parts: QWidget
    data_summary: QLabel
    data_preview: CanvasTab
    table_model: DataFrameModel
    table_view: QTableView
    tab_model: CanvasTab
    tab_norm: CanvasTab
    tab_decom: CanvasTab
    tab_roll: CanvasTab
    tab_multiscale: CanvasTab
    tab_pdp: CanvasTab

    # ---- menu entries whose enabled state follows the data
    recent_menu: QMenu
    _export_action: QAction
    _report_action: QAction
    _palette_action: QAction

    if TYPE_CHECKING:
        # Handlers the other mixins define. Declaring them here means a mixin
        # can call them and the signature is checked in one place; a change to
        # one surfaces at its declaration rather than at every call site.
        def _abandon_task(self) -> None: ...
        def _build_action_bar(self) -> QWidget: ...
        def _build_param_panel(self) -> QWidget: ...
        def _export_current(self) -> None: ...
        def _export_html(self) -> None: ...
        def _open_csv(self) -> None: ...
        def load_config(self) -> None: ...
        def load_csv(self, path: str, remember: bool = False) -> None: ...
        def load_example(self) -> None: ...
        def open_data_studio(self) -> None: ...
        def open_scm_example(self) -> None: ...
        def open_scm_studio(self) -> None: ...
        def open_transport_studio(self) -> None: ...
        def save_config(self) -> None: ...
        def save_script(self) -> None: ...
        def show_history(self) -> None: ...
