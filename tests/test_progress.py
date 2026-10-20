"""How far a long call has come (normet.utils.progress), and the GUI's task clock
that says it with the time so far and the time left."""

from __future__ import annotations

import os
import re
import sys
import threading
import time

import numpy as np
import pandas as pd
import pytest

from normet.utils import progress


def test_nothing_is_heard_without_a_listener():
    progress.report(1, 2)  # no listener: nothing to do, nothing raised


def test_a_part_reports_within_its_share_of_the_whole():
    heard: list[float] = []
    with progress.listening(heard.append):
        progress.report(1, 4)
        with progress.part(2, 4):  # the third quarter
            progress.report(0, 2)
            progress.report(1, 2)
            with progress.part(1, 2):  # its second half
                progress.report(1, 1)
        progress.report(5, 4)  # past the end: the whole, no more
    assert heard == pytest.approx([0.25, 0.5, 0.625, 0.75, 1.0])
    progress.report(1, 2)  # the listener is gone with its block
    assert len(heard) == 5


def test_rolling_reports_its_windows_and_keeps_a_window_inside_its_share(monkeypatch):
    rolling_mod = sys.modules["normet.analysis.rolling"]

    def normalise(df, **kwargs):
        progress.report(1, 1)  # a call inside a window says it is done
        return df.assign(normalised=df["value"])

    class Model:
        backend = "flaml"
        feature_names_in_ = np.array(["ws"])

    monkeypatch.setattr(rolling_mod, "normalise", normalise)
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({"date": dates, "value": np.arange(30.0), "ws": np.ones(30)})
    heard: list[float] = []
    with progress.listening(heard.append):
        rolling_mod.rolling(
            df=df, model=Model(), covariates=["ws"], window_days=10, rolling_every=5
        )
    # Five windows: each heard as it starts, and as its call ends.
    assert heard == pytest.approx([0.0, 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0])


def test_durations_read_at_a_glance():
    from normet.gui._clock import format_duration

    assert [format_duration(s) for s in (0.4, 42, 61, 12 * 60 + 20, 80 * 60)] == [
        "0 s",
        "42 s",
        "1 min",
        "12 min",
        "1 h 20 min",
    ]


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _until(qapp, done, timeout=10.0):
    t0 = time.time()
    while not done() and time.time() - t0 < timeout:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()


def test_the_status_bar_says_how_far_a_task_has_come_and_how_long_it_took(qapp):
    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        bar = win.statusBar()
        seen = threading.Event()

        def task():
            progress.report(1, 2)
            seen.wait(10)  # until the window has shown it
            return "Done."

        # The result goes to a method of a widget, queued on the GUI thread, as a
        # window's own handlers are (a plain function would run on the worker's).
        win.runner.submit("a test task", task, bar.showMessage, print)
        assert bar.currentMessage() == "Running: a test task… · 0 s so far"
        assert win.progress.maximum() == 0  # nothing known yet: a busy bar
        _until(qapp, lambda: "50 %" in bar.currentMessage())
        # (the seconds as a slow runner has them: the times are the clock tests' below)
        assert re.fullmatch(r"Running: a test task… · 50 % · \d+ s so far", bar.currentMessage())
        assert win.progress.maximum() == 1000 and win.progress.value() == 500
        seen.set()
        _until(qapp, lambda: not win.runner.busy)
        assert re.fullmatch(r"Done\. · took \d+ s", bar.currentMessage())
        assert win.progress.maximum() == 1

        # A task that fails writes no line of its own: the clock says it ended.
        def fails():
            raise RuntimeError("no")

        win.runner.submit("a failing task", fails, print, lambda _tb: None)
        _until(qapp, lambda: not win.runner.busy)
        assert re.fullmatch(r"Ended: a failing task · took \d+ s", bar.currentMessage())
    finally:
        win.close()


def test_once_a_result_is_in_the_clock_writes_no_more_over_its_line(qapp, monkeypatch):
    from normet.gui import _clock
    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        clock, now, bar = win.task_clock, [100.0], win.statusBar()
        monkeypatch.setattr(_clock.time, "monotonic", lambda: now[0])
        clock.start("normalise")
        now[0] += 3
        clock.settle()  # the result is in; the step writes its line
        bar.showMessage("Done.")
        now[0] += 5
        clock.report(0.9)  # a late report: no tick, no line over "Done."
        assert bar.currentMessage() == "Done."
        clock.stop()  # the time it took, not counting the wait for the end
        assert bar.currentMessage() == "Done. · took 3 s"
    finally:
        win.close()


def test_the_time_left_comes_once_there_is_enough_to_go_on(qapp, monkeypatch):
    from normet.gui import _clock
    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        clock, now = win.task_clock, [100.0]
        monkeypatch.setattr(_clock.time, "monotonic", lambda: now[0])
        clock.start("normalise")
        clock.report(0.25)
        assert "left" not in win.statusBar().currentMessage()  # 0 s in: too soon
        now[0] += 30
        clock.report(0.5)
        assert win.statusBar().currentMessage() == (
            "Running: normalise… · 50 % · 30 s so far · ~30 s left"
        )
        clock.report(0.4)  # the share never goes back
        assert "50 %" in win.statusBar().currentMessage()
        # A budget: the share is the time against it.
        clock.start("train (flaml)")
        clock.expect(60)
        now[0] += 15
        clock._show()
        assert win.statusBar().currentMessage() == (
            "Running: train (flaml)… · 25 % · 15 s so far · ~45 s left"
        )
        now[0] += 60  # past the budget: finishing, for a time no one can say
        clock._show()
        assert win.statusBar().currentMessage() == "Running: train (flaml)… · 99 % · 1 min so far"
        clock.stop()
    finally:
        win.close()
