"""The Events tab: the score strip and the event table (helpers pure), the
run wired to the library's ``detect_events``, and the box's gating (offscreen;
the scoring methods themselves are the library's own tests)."""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from normet.gui._run_analyse import (  # noqa: E402
    _events_has_clock,
    events_figure,
    events_table,
    events_verdict,
)


def _events(n: int = 2, hourly: bool = True) -> pd.DataFrame:
    """Event rows as ``detect_events`` returns them: ranked by max_score."""
    unit = pd.Timedelta(hours=6) if hourly else pd.Timedelta(days=6)
    start = pd.Timestamp("2024-06-01 03:00") if hourly else pd.Timestamp("2024-06-01")
    rows = [
        {
            "start": start + i * pd.Timedelta(days=3),
            "end": start + i * pd.Timedelta(days=3) + unit,
            "n": 6 + i,
            "max_score": 5.42 - i,
            "mean_score": 3.90 - i,
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows).sort_values("max_score", ascending=False).reset_index(drop=True)


def test_events_table_shows_windows_with_their_clock():
    header, rows = events_table(_events(), with_clock=True)
    assert header == ["Start", "End", "Steps", "Max score", "Mean score"]
    assert len(rows) == 2
    assert rows[0][0] == "2024-06-01 03:00" and rows[0][1] == "2024-06-01 09:00"
    assert rows[0][2] == "6" and rows[0][3] == "5.42" and rows[0][4] == "3.90"


def test_events_table_without_a_clock_and_from_strings():
    folds = _events(n=1)
    _h, daily = events_table(folds, with_clock=False)
    assert daily[0][0] == "2024-06-01"  # no hour on a daily stamp
    as_str = folds.copy()
    as_str["start"] = as_str["start"].astype(str)  # a bridge's string form
    _h, rows = events_table(as_str, with_clock=False)
    assert rows[0][0].startswith("2024-06-01")


def test_events_verdict_counts_windows_and_names_the_largest():
    idx = pd.date_range("2024-06-01", periods=200, freq="h")
    scores = pd.Series(np.zeros(200), index=idx)
    verdict, lines = events_verdict(
        _events(), scores, method="iqr", k=3.0, min_length=3, with_clock=True
    )
    assert verdict[0] == "warn" and "2" in verdict[1] and "flagged" in verdict[1]
    assert lines[0] == "Method iqr, threshold 3, windows of at least 3 steps."
    assert "2024-06-01 03:00" in lines[1] and "5.42" in lines[1]
    # 13 steps of the 200 flagged
    assert "6.5%" in verdict[1]


def test_events_verdict_on_a_clean_record():
    idx = pd.date_range("2024-06-01", periods=100, freq="h")
    scores = pd.Series(np.zeros(100), index=idx)
    empty = pd.DataFrame(columns=["start", "end", "n", "max_score", "mean_score"])
    verdict, lines = events_verdict(
        empty, scores, method="iqr", k=3.0, min_length=3, with_clock=False
    )
    assert verdict[0] == "ok" and verdict[1].startswith("No events")
    assert len(lines) == 1


def test_events_figure_shades_the_windows_and_draws_the_threshold():
    idx = pd.date_range("2024-06-01", periods=100, freq="h")
    scores = pd.Series(np.abs(np.sin(np.arange(100.0))), index=idx)
    fig = events_figure(scores, _events(), threshold=3.0, name="PM2.5")
    ax = fig.axes[0]
    assert len(ax.lines) == 2  # the strip and the threshold line
    assert len(ax.patches) == 2  # one shaded window each (axvspan draws a patch)
    assert "threshold 3" in [ln.get_label() for ln in ax.get_lines()]
    assert "PM2.5" in ax.get_title()
    # isolation scores against its own boundary: no line to draw
    fig2 = events_figure(scores, _events(), threshold=None, name="PM2.5")
    assert len(fig2.axes[0].lines) == 1


def test_has_clock_reads_the_record_step():
    hourly = pd.Series(0.0, index=pd.date_range("2024-06-01", periods=50, freq="h"))
    daily = pd.Series(0.0, index=pd.date_range("2024-06-01", periods=50, freq="D"))
    plain = pd.Series(np.zeros(50))
    assert _events_has_clock(hourly)
    assert not _events_has_clock(daily)
    assert not _events_has_clock(plain)


# -- the window, offscreen ----------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _wait(qapp, runner, timeout=60):
    t0 = time.time()
    while runner.busy and time.time() - t0 < timeout:
        qapp.processEvents()
        time.sleep(0.01)
    for _ in range(20):
        qapp.processEvents()


def _tab_words(tab) -> str:
    from PySide6.QtWidgets import QLabel, QTableWidget

    parts = [w.text() for w in tab.findChildren(QLabel)]
    for t in tab.findChildren(QTableWidget):
        for c in range(t.columnCount()):
            head = t.horizontalHeaderItem(c)
            if head is not None:
                parts.append(head.text())
        for r in range(t.rowCount()):
            for c in range(t.columnCount()):
                item = t.item(r, c)
                if item is not None:
                    parts.append(item.text())
    return "\n".join(parts)


def _events_win(qapp):
    """A window with data and a model: the observed target carrying one clear
    six-hour episode above 3 IQR, so the library's iqr flags exactly it."""
    from normet.gui.main_window import MainWindow

    w = MainWindow()
    w.load_example()
    rng = np.random.default_rng(4)
    n = 600
    idx = pd.date_range("2024-06-01", periods=n, freq="h")
    value = rng.normal(30.0, 4.0, n)
    value[200:206] += 40.0  # the episode
    target = w.target_combo.currentText()
    w.df_prep = pd.DataFrame({"date": idx, target: value})
    w.df_prep["set"] = "training"
    w.model = object()
    w.trained_features = ["t2m"]
    w.results.pop("decompose", None)
    return w, target, idx


def test_the_events_box_and_tab_are_wired(qapp):
    from normet.gui._session import SessionMixin

    w = None
    try:
        from normet.gui.main_window import MainWindow

        w = MainWindow()
        assert w.tabs.widget(w.tabs.indexOf(w.tab_trend) + 1) is w.tab_events
        assert SessionMixin._TAB_KEYS[w.tabs.indexOf(w.tab_events)] == "events"
        assert w.events_button.text() == "▶  Find events"
        assert w.events_k.isEnabled()
        w.events_method.setCurrentText("isolation")
        assert not w.events_k.isEnabled()  # the forest draws its own boundary
        w.events_method.setCurrentText("iqr")
        assert w.events_k.isEnabled()
    finally:
        if w is not None:
            w.close()


def test_events_button_gating(qapp):
    w, _target, _idx = _events_win(qapp)
    try:
        w.model = None
        w._sync_enabled()
        assert not w.events_button.isEnabled()
        assert "model" in w.events_button.toolTip().lower()
        if hasattr(w, "_is_chronos"):  # the R window has neither chronos nor this branch
            w.backend_combo.setCurrentText("chronos-2")
            w.model = object()
            w.events_series.setCurrentText("emission part (emi_total)")
            w._sync_enabled()
            # Like the trend button: it stays runnable (the observed target is
            # fine on chronos-2), but the emission choice says why it cannot.
            assert w.events_button.isEnabled()
            assert "chronos-2" in w.events_button.toolTip()
    finally:
        w.close()


def test_run_events_flags_the_episode_on_the_tab(qapp, monkeypatch):
    """The wiring end to end: submit, worker, render, command log. The two
    library calls are replaced by a deterministic stand-in of the same iqr
    maths — this is the window's test, and its vendored copy runs in the R
    window's CI where the bridge would shell out to an Rscript that is not
    installed there. The methods themselves are the libraries' own tests."""
    import normet

    def fake_scores(series, **_kw):
        med = series.median()
        iqr = series.quantile(0.75) - series.quantile(0.25)
        return ((series - med).abs() / iqr).rename("score")

    def fake_detect(series, *, k=3.0, min_length=1, **_kw):
        score = fake_scores(series)
        above = (score > k) & score.notna()
        rows, run = [], None
        for flag, stamp in zip(above, score.index, strict=True):
            if flag and run is None:
                run = [stamp, stamp, 0, 0.0, 0.0]
            if flag:
                run[1], run[2] = stamp, run[2] + 1
                run[3] = max(run[3], float(score.loc[stamp]))
                run[4] += float(score.loc[stamp])
            elif run is not None:
                rows.append(run)
                run = None
        if run is not None:
            rows.append(run)
        events = pd.DataFrame(rows, columns=["start", "end", "n", "max_score", "mean_score"])
        if len(events):
            events["mean_score"] = events["mean_score"] / events["n"]
            events = (
                events[events["n"] >= min_length]
                .sort_values("max_score", ascending=False)
                .reset_index(drop=True)
            )
        return events

    monkeypatch.setattr(normet, "anomaly_scores", fake_scores)
    monkeypatch.setattr(normet, "detect_events", fake_detect)
    w, _target, idx = _events_win(qapp)
    try:
        w.events_series.setCurrentText("observed target")
        w.events_method.setCurrentText("iqr")
        w.events_k.setValue(3.0)
        w.events_min_len.setValue(3)
        w._run_events()
        _wait(qapp, w.runner)
        words = _tab_words(w.tab_events)
        assert "Start" in words and "Steps" in words and "Max score" in words
        assert "flagged" in words  # the verdict
        assert str(idx[200])[:10] in words  # the episode's window is in the table
        assert "events" in w.results and len(w.results["events"]) >= 1
        # the command log replays it on the table it was run on, not a dumped series
        entry = w.command_log.entries[-1]
        assert entry.title == "Find events"
        assert "df_prep" in entry.code and "# Series" not in entry.code
    finally:
        w.close()


def test_run_events_needs_a_model_and_a_series(qapp):
    w, _target, _idx = _events_win(qapp)
    try:
        w.model = None
        w._run_events()  # the check says so; nothing runs, nothing raises
        assert not w.runner.busy
        w.model = object()
        w.events_series.setCurrentText("emission part (emi_total)")
        assert w._events_resolve() is None  # no decomposition: said on screen
    finally:
        w.close()
