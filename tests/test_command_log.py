"""The command log and the print layout (offscreen; skipped without PySide6).

The log's promise is that its script repeats the session: the test below runs the
window's steps, executes the script the log writes, and compares the results.
"""

from __future__ import annotations

import io
import os
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from normet.gui import commandlog  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _run(qapp, win, step):
    step()
    t0 = time.time()
    while win.runner.busy:
        qapp.processEvents()
        time.sleep(0.01)
        if time.time() - t0 > 300:
            raise TimeoutError
    for _ in range(20):
        qapp.processEvents()


# -- writing calls ------------------------------------------------------------------------
def test_literals_read_back():
    assert commandlog.literal(0.1 + 0.2) == "0.3"
    assert commandlog.literal(3.0) == "3"
    assert commandlog.literal(np.int64(4)) == "4"
    assert commandlog.literal(np.bool_(True)) == "True"
    assert commandlog.literal(float("nan")) == "float('nan')"
    assert commandlog.literal(float("-inf")) == "float('-inf')"
    assert commandlog.literal((0.05,)) == "(0.05,)"
    assert commandlog.literal((0.05, 0.95)) == "(0.05, 0.95)"
    assert commandlog.literal({"n": [1, 2]}) == "{'n': [1, 2]}"
    assert commandlog.literal(pd.Timestamp("2021-01-01")) == "'2021-01-01 00:00:00'"


def test_calls_wrap_under_the_bracket():
    short = commandlog.call("x", "f", ["a"], [("k", "1")])
    assert short == "x = f(a, k=1)"
    long = commandlog.call("x", "f", ["df"], [(f"k{i}", "'" + "v" * 20 + "'") for i in range(5)])
    lines = long.splitlines()
    assert len(lines) > 1 and all(len(line) <= commandlog.WIDTH for line in lines)
    assert all(line.startswith(" " * len("x = f(")) for line in lines[1:])


def test_task_names_the_window_objects():
    from normet import normalise

    df_prep, model = pd.DataFrame({"a": [1]}), object()
    entry = commandlog.task(
        normalise,
        (df_prep, model),
        {"n_samples": 5, "verbose": True},
        [(df_prep, "df_prep"), (model, "model")],
    )
    assert entry is not None and entry.code == "normalised = normalise(df_prep, model, n_samples=5)"
    assert entry.imports == ("from normet import normalise",)
    stray = commandlog.task(normalise, (pd.DataFrame(), model), {}, [(model, "model")])
    assert stray is not None and "not named here" in stray.notes[0]
    assert commandlog.task(print, (), {}, []) is None


def test_log_text_and_script():
    log = commandlog.CommandLog()
    assert "Nothing yet" in log.text()
    log.add(commandlog.load_example())
    log.add(None)
    log.add(commandlog.from_memory("the Data Studio"))
    assert len(log) == 2 and log.entries[0].when
    script = log.script(title="Test")
    assert script.startswith('"""Test.')
    assert "from normet.gui._examples import make_deweather_example" in script
    assert "#   This table was handed to the window in memory" in script
    log.clear()
    assert len(log) == 0


# -- the window ---------------------------------------------------------------------------
@pytest.fixture
def win(qapp):
    from normet.gui.main_window import MainWindow

    w = MainWindow()
    yield w
    w.close()


def test_the_script_repeats_the_session(win, qapp):
    pytest.importorskip("lightgbm")
    win.load_example()
    win.backend_combo.setCurrentText("lightgbm")
    win.budget_spin.setValue(3)
    win.norm_samples.setValue(5)
    _run(qapp, win, win._run_train)
    _run(qapp, win, win._run_normalise)
    assert [e.title for e in win.command_log.entries] == [
        "Load example data",
        "Train model",
        "Normalise",
    ]
    assert "normalised = normalise(df_prep, model" in win.command_view.toPlainText()

    ns: dict = {}
    exec(compile(win.command_log.script(), "session.py", "exec"), ns)
    pd.testing.assert_frame_equal(ns["df_prep"], win.df_prep)
    pd.testing.assert_frame_equal(ns["normalised"], win.results["normalise"])


def test_a_failed_task_is_not_logged(win, qapp):
    from normet import normalise

    win.load_example()
    errors = []
    win.runner.submit("bad", normalise, lambda r: None, errors.append, None, None)
    t0 = time.time()
    while win.runner.busy and time.time() - t0 < 60:
        qapp.processEvents()
        time.sleep(0.01)
    for _ in range(20):
        qapp.processEvents()
    assert errors and [e.title for e in win.command_log.entries] == ["Load example data"]


def test_an_opened_csv_reads_back(win, tmp_path):
    path = tmp_path / "site.csv"
    pd.DataFrame({"Date": ["2021-01-01 00:00", "2021-01-01 01:00"], "NO2": [1.0, 2.0]}).to_csv(
        path, index=False
    )
    win.load_csv(str(path))
    ns: dict = {}
    exec(compile(win.command_log.script(), "session.py", "exec"), ns)
    pd.testing.assert_frame_equal(ns["df"], win.df_raw)
    win._clear_command_log()
    assert "Nothing yet" in win.command_view.toPlainText()


def test_a_csv_across_a_daylight_saving_change_loads_and_trains(win, tmp_path):
    # Local time with its offset, +00:00 then +01:00: pandas 3 refused the column
    # at load ("Mixed timezones detected"); pandas 2 took it as objects that
    # training then refused. It is put on the window's clock (UTC here), and the logged
    # script agrees.
    from normet.utils.prepare import add_date_variables

    path = tmp_path / "bst.csv"
    path.write_text(
        "date,NO2\n2020-03-29T00:00+00:00,1\n2020-03-29T02:00+01:00,2\n2020-03-29T03:00+01:00,3\n"
    )
    win.load_csv(str(path))
    dates = win.df_raw["date"]
    assert dates.dt.tz is None and dates.dt.hour.tolist() == [0, 1, 2]  # on UTC, the clock
    assert add_date_variables(win.df_raw)["hour"].tolist() == [0, 1, 2]
    ns: dict = {}
    exec(compile(win.command_log.script(), "session.py", "exec"), ns)
    pd.testing.assert_frame_equal(ns["df"], win.df_raw)
