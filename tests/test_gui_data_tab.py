"""The Data tab (normet.gui._data_tab): the overview, the rows left out, the cycles, the
wind, the statistics -- offscreen."""

from __future__ import annotations

import os

import pandas as pd
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_exclusion_lines_read_back_as_written():
    from normet.gui._data_tab import excluded_mask, exclusion_lines, parse_exclusions

    spans = parse_exclusions(
        "2020-03-23 .. 2020-04-05\n# a comment\n2020-05-01 06:00 .. 2020-05-01 18:00\n2020-06-02"
    )
    assert exclusion_lines(spans) == [
        "2020-03-23 .. 2020-04-05",
        "2020-05-01 06:00 .. 2020-05-01 18:00",
        "2020-06-02",
    ]
    dates = pd.Series(pd.date_range("2020-04-05 22:00", periods=4, freq="h"))
    assert list(excluded_mask(dates, spans)) == [True, True, False, False]  # the 5th whole
    # Spans that meet are one; a line that is not a time is named.
    assert len(parse_exclusions(["2020-01-01 .. 2020-01-10", "2020-01-05 .. 2020-01-20"])) == 1
    with pytest.raises(ValueError, match="not a time"):
        parse_exclusions(["soon"])


def test_a_span_picked_is_listed_left_out_and_restored(qapp):
    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        win.load_example("my1")
        tab = win.data_tab
        n = len(win.df_raw)
        tab.pick(pd.Timestamp("2020-03-23"), pd.Timestamp("2020-03-24 23:59"))
        assert tab.rows_panel.isVisibleTo(tab) and tab.rows_model.rowCount() == 48
        assert tab.rows_label.text() == "48 rows picked"
        tab.exclude_picked()
        assert win.exclusions == ["2020-03-23 .. 2020-03-24"]
        assert len(win.df_raw) == n - 48 and len(win.df_loaded) == n
        assert win.command_log.entries[-1].title == "Leave rows out"
        assert "df_kept = df[~left_out]" in win.command_log.entries[-1].code
        assert win.data_summary.text().endswith("left out: 48 rows")
        # The steps read the rows kept, and the log names them so.
        from normet.gui import commandlog

        entry = commandlog.task(
            lambda df: None, (win.df_raw,), {}, [(win.df_raw, "df_kept"), (win.df_loaded, "df")]
        )
        assert entry is None or "df_kept" in entry.code
        # A pick inside it lists its rows as left out; restored, they are back.
        tab.pick(pd.Timestamp("2020-03-24"), pd.Timestamp("2020-03-24 05:59"))
        assert "6 of them left out" in tab.rows_label.text()
        tab.restore_picked()
        assert win.exclusions == ["2020-03-23", "2020-03-24 06:00 .. 2020-03-24 23:59"]
        assert len(win.df_raw) == n - 42
        # Saved with the session: the config carries the lines, the bundle every row.
        assert win.get_config()["exclude"] == win.exclusions
        assert len(win.session_bundle()["df_raw"]) == n
    finally:
        win.close()


def test_several_spans_are_picked_with_shift_and_left_out_together(qapp):
    """As cume's samples: a drag picks a span, one with Shift (Ctrl, Cmd) adds another, a
    plain drag starts again and Escape clears; the spans picked are left out and put
    back together."""
    from types import SimpleNamespace

    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        win.load_example("my1")
        tab = win.data_tab
        n = len(win.df_raw)
        tab.pick(pd.Timestamp("2020-03-23"), pd.Timestamp("2020-03-23 23:59"))
        tab.pick(pd.Timestamp("2020-05-01"), pd.Timestamp("2020-05-01 11:59"), add=True)
        assert tab.picked == [
            (pd.Timestamp("2020-03-23"), pd.Timestamp("2020-03-23 23:59")),
            (pd.Timestamp("2020-05-01"), pd.Timestamp("2020-05-01 11:59")),
        ]
        assert tab.rows_model.rowCount() == 36 and tab.rows_label.text() == "36 rows picked"
        assert tab.pick_info.text().startswith("Picked 2 spans (36 rows): 2020-03-23; ")
        # Overlapping or touching spans are one.
        tab.pick(pd.Timestamp("2020-05-01 06:00"), pd.Timestamp("2020-05-01 17:59"), add=True)
        assert len(tab.picked) == 2 and tab.rows_model.rowCount() == 42
        tab.exclude_picked()
        assert win.exclusions == ["2020-03-23", "2020-05-01 .. 2020-05-01 17:59"]
        assert len(win.df_raw) == n - 42 and tab.picked == []
        # Both put back at once.
        tab.pick(pd.Timestamp("2020-03-23"), pd.Timestamp("2020-03-23 23:59"))
        tab.pick(pd.Timestamp("2020-05-01"), pd.Timestamp("2020-05-01 23:59"), add=True)
        assert tab.restore_btn.isEnabled()
        tab.restore_picked()
        assert win.exclusions == [] and len(win.df_raw) == n
        # A plain drag starts again; Escape clears.
        tab.pick(pd.Timestamp("2020-03-23"), pd.Timestamp("2020-03-23 23:59"))
        tab.pick(pd.Timestamp("2020-06-01"), pd.Timestamp("2020-06-01 23:59"))
        assert len(tab.picked) == 1
        tab._on_key(SimpleNamespace(key="escape"))
        assert tab.picked == [] and not tab.clear_btn.isEnabled()
    finally:
        win.close()


def test_the_other_pages_draw(qapp):
    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        win.load_example("my1")
        tab = win.data_tab
        for i in range(tab.pages.count()):
            tab.pages.setCurrentIndex(i)
            qapp.processEvents()
        assert len(tab.cycles.canvas.figure.axes) == 3  # hour, weekday, month
        assert len(tab.wind.canvas.figure.axes) >= 3  # the two roses and the polar plot
        assert tab.stats.rowCount() == len(win.df_raw.columns) - 1
        assert tab.stats_note.text().endswith("rows kept.")
    finally:
        win.close()
