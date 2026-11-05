"""GUI extras added across the improvement rounds, offscreen: table sorting and copy-row,
window/panel-state persistence, the run-history clear confirmation, the task-finished
notification, and keyboard activation of step lights and chat tools."""

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


# ---------------------------------------------------------------- tables
def test_fill_sortable_orders_numbers_and_keeps_none_last(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QTableWidget

    from normet.gui._widgets import copy_table_rows, fill_sortable

    tbl = QTableWidget(0, 2)
    fill_sortable(tbl, ["a", "b"], [
        ["9", "x"],
        ["10", "y"],
        ["", "z"],
    ])
    assert tbl.isSortingEnabled()
    tbl.sortByColumn(0, Qt.SortOrder.AscendingOrder)
    assert tbl.item(0, 0).text() == "9"
    assert tbl.item(1, 0).text() == "10"
    assert tbl.item(2, 0).text() == ""        # no key sorts last
    copy_table_rows(tbl, [0])
    assert qapp.clipboard().text() == "9\tx"


# ---------------------------------------------------------------- window state
def test_panel_page_and_result_tab_are_remembered(qapp, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from normet.gui import _settings
    from normet.gui import main_window as mw
    from normet.gui.main_window import MainWindow

    ini = str(tmp_path / "normet-gui.ini")
    monkeypatch.setattr(_settings, "config_dir", lambda: str(tmp_path))
    monkeypatch.setattr(mw, "QSettings",
                        lambda *a: QSettings(ini, QSettings.Format.IniFormat))
    win = MainWindow()
    win.show_panel_page("Decompose")
    win.tabs.setCurrentIndex(3)
    win.close()
    s = QSettings(ini, QSettings.Format.IniFormat)
    assert s.value("panel_page") == "Decompose"
    assert s.value("result_tab") is not None
    win.deleteLater()


# ---------------------------------------------------------------- run history
def test_clear_history_asks_first(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from normet.gui import _settings
    from normet.gui._session import SessionMixin

    monkeypatch.setattr(_settings, "config_dir", lambda: str(tmp_path))
    _settings.append_history({"time": "2026-01-01 00:00", "task": "train",
                              "target": "pm25", "backend": "flaml",
                              "rows": 10, "detail": ""})
    assert _settings.read_history()

    class _Dlg:
        def accept(self):
            pass

    dlg = _Dlg()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    SessionMixin._clear_history(None, dlg)     # declined: kept
    assert _settings.read_history()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    SessionMixin._clear_history(None, dlg)     # confirmed: cleared
    assert not _settings.read_history()


# ---------------------------------------------------------------- task notify
def test_notify_done_is_silent_offscreen(qapp, tmp_path, monkeypatch):
    from normet.gui import _settings
    from normet.gui.main_window import MainWindow

    monkeypatch.setattr(_settings, "config_dir", lambda: str(tmp_path))
    win = MainWindow()
    win.runner.task_name = "train (flaml)"
    win._notify_done()                         # offscreen: no beep, no crash
    win.deleteLater()


# ---------------------------------------------------------------- keyboard
def test_step_light_opens_its_page_from_the_keyboard(qapp, tmp_path, monkeypatch):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    from normet.gui import _settings
    from normet.gui.main_window import MainWindow

    monkeypatch.setattr(_settings, "config_dir", lambda: str(tmp_path))
    win = MainWindow()
    win.show()
    light = win._step_lights["trend"]
    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return,
                   Qt.KeyboardModifier.NoModifier)
    assert win.eventFilter(light, ev)           # consumed
    assert win.panel_stack.currentIndex() >= 0
    win.deleteLater()


def test_chat_tool_toggles_from_the_keyboard(qapp):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    from normet.gui.chatview import Look, _Tool

    k = Look(1.0)
    tool = _Tool("test_call()", k)
    tool.finish("some result text", False)
    tool.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return,
                                 Qt.KeyboardModifier.NoModifier))
    assert tool.opened
    tool.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space,
                                 Qt.KeyboardModifier.NoModifier))
    assert not tool.opened
    # an errored tool never opens
    bad = _Tool("bad_call()", k)
    bad.finish("error text", True)
    bad.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return,
                                Qt.KeyboardModifier.NoModifier))
    assert not bad.opened
