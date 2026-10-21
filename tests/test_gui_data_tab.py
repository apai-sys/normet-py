"""The Data tab (normet.gui._data_tab): the overview, the rows left out, the cycles, the
wind, the statistics -- offscreen."""

from __future__ import annotations

import os

import numpy as np
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
        assert tab.pick_info.text().endswith("(48 rows)")
        tab.exclude_picked()
        # The line under the plot says it at once: the rows it holds, now left out.
        assert tab.pick_info.text().endswith("(48 rows, 48 left out)")
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
        assert tab.pick_info.text().endswith("rows, 6 left out)")
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
        assert tab.rows_model.rowCount() == 36
        assert tab.pick_info.text().startswith("Picked 2 spans (36 rows): 2020-03-23; ")
        # Overlapping or touching spans are one.
        tab.pick(pd.Timestamp("2020-05-01 06:00"), pd.Timestamp("2020-05-01 17:59"), add=True)
        assert len(tab.picked) == 2 and tab.rows_model.rowCount() == 42
        tab.exclude_picked()
        assert win.exclusions == ["2020-03-23", "2020-05-01 .. 2020-05-01 17:59"]
        # The pick and its list kept, to go on from: its rows now marked left out.
        assert len(win.df_raw) == n - 42 and len(tab.picked) == 2
        assert "(42 rows, 42 left out)" in tab.pick_info.text()
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


def test_rows_selected_in_the_list_are_left_out_and_put_back(qapp):
    """As in a spreadsheet: rows selected in the list of those picked -- a run, and rows
    apart -- left out by the list's menu or Exclude picked, and put back the same way;
    with none selected, the buttons act on every row picked."""
    from PySide6.QtCore import QItemSelection, QItemSelectionModel, QPoint, Qt
    from PySide6.QtWidgets import QMenu

    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        win.load_example("my1")
        tab = win.data_tab
        n = len(win.df_raw)
        tab.pick(pd.Timestamp("2020-03-23"), pd.Timestamp("2020-03-23 23:59"))
        assert tab.rows_model.rowCount() == 24
        model, sel = tab.rows_model, tab.rows_view.selectionModel()

        def select(*runs):
            sel.clearSelection()
            for a, b in runs:
                block = QItemSelection(model.index(a, 0), model.index(b, model.columnCount() - 1))
                sel.select(block, QItemSelectionModel.SelectionFlag.Select)

        select((0, 0), (5, 7), (23, 23))  # 00:00 alone, 05:00-07:00, 23:00
        assert len(tab.selected_rows()) == 5 and tab.exclude_btn.isEnabled()
        menus = []

        class _Menu(QMenu):
            # A subclass's exec, not one set on Qt's class: under PySide an instance
            # calls the built-in one (as cume's tests found).
            def exec(self_, *a):
                menus.append([x.text() for x in self_.actions()])

        import normet.gui._data_tab as data_tab

        view, header = tab.rows_view, tab.rows_view.verticalHeader()

        def on_cell(row):
            return view.visualRect(model.index(row, 1)).center()

        def on_number(row):
            return QPoint(header.width() // 2, header.sectionViewportPosition(row) + 2)

        saved = data_tab.QMenu
        data_tab.QMenu = _Menu
        try:
            # On a row selected, in its cells or on its number: the selection kept whole.
            tab._rows_menu(on_cell(6))
            header.customContextMenuRequested.emit(on_number(23))  # the numbers' own signal
            assert len(menus) == 2 and menus[0] == menus[1]
            assert menus[0][:2] == ["Exclude 5 selected rows", "Restore 5 selected rows"]
            # On a row not selected: that row alone, as a spreadsheet does.
            header.customContextMenuRequested.emit(on_number(12))
            listed = lambda: sorted(ix.row() for ix in sel.selectedRows())  # noqa: E731
            assert menus[-1][0] == "Exclude 1 selected rows" and listed() == [12]
            select((0, 0), (5, 7), (23, 23))
            tab._rows_menu(on_cell(15))
            assert listed() == [15]
        finally:
            data_tab.QMenu = saved
        assert header.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
        select((0, 0), (5, 7), (23, 23))
        tab.exclude_picked()
        # A row at midnight is written with its time: a date alone would be the whole day.
        assert win.exclusions == [
            "2020-03-23 00:00",
            "2020-03-23 05:00 .. 2020-03-23 07:00",
            "2020-03-23 23:00",
        ]
        assert len(win.df_raw) == n - 5
        assert list(tab.rows_model.left_out[:6]) == [True, False, False, False, False, True]
        assert tab.pick_info.text().endswith("(24 rows, 5 left out)")
        # Two of them put back: the one at 06:00, and 23:00.
        select((6, 6), (23, 23))
        assert tab.restore_btn.isEnabled()
        tab.restore_picked()
        assert win.exclusions == ["2020-03-23 00:00", "2020-03-23 05:00", "2020-03-23 07:00"]
        # None selected: the buttons act on every row picked.
        tab.rows_view.clearSelection()
        tab.exclude_picked()
        assert win.exclusions == ["2020-03-23"] and len(win.df_raw) == n - 24
        tab.restore_picked()
        assert win.exclusions == [] and len(win.df_raw) == n
    finally:
        win.close()


def test_the_exclude_box_stands_level_with_the_buttons(qapp):
    """The Exclude box, two lines tall, has its top on the buttons' (and Apply's), not
    centred above them; the variable's combo on their middle."""
    from PySide6.QtCore import QPoint

    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        win.resize(1500, 900)
        win.show()
        win.load_example("my1")
        tab = win.data_tab
        if hasattr(win, "tabs"):
            win.tabs.setCurrentWidget(tab)
        for _ in range(5):
            qapp.processEvents()
        row = tab.overview.bar_tail[0].parentWidget()

        def top(w):
            return w.mapTo(row, QPoint(0, 0)).y()

        assert tab.overview.bar_top == [tab.overview.bar_tail[1]]
        assert tab.exclude_box.height() > tab.exclude_btn.height()
        assert top(tab.exclude_box) == top(tab.exclude_btn) == top(tab.apply_btn)
        middle = top(tab.exclude_btn) + tab.exclude_btn.height() / 2
        assert abs(top(tab.var_combo) + tab.var_combo.height() / 2 - middle) <= 1
        # A span with its times, wider than the box, folded in it rather than scrolled
        # sideways under a bar that would hide the box's second line.
        box = tab.exclude_box
        box.setPlainText("2020-03-23 03:00 .. 2020-03-23 05:00")
        qapp.processEvents()
        assert not box.horizontalScrollBar().isVisible()
        layout = box.document().firstBlock().layout()
        assert layout.lineCount() == 2
        assert layout.lineAt(1).textStart() == len("2020-03-23 03:00 .. ")
        # Its two lines and no part of a third: the viewport ends where they do.
        bottom = box.contentOffset().y() + 2 * layout.lineAt(0).height()
        assert box.viewport().height() == bottom
    finally:
        win.close()


def test_the_overview_is_picked_with_the_mouse_and_read_under_it(qapp):
    """The overview's controls on its Resolution row, the pick and the readout at its
    lower left; a drag picks a span, a click the row under it, Shift adds, and the cursor
    is read: its time, the variable and the value."""
    from matplotlib import dates as mdates
    from matplotlib.backend_bases import MouseEvent

    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        win.load_example("my1")
        win.tabs.setCurrentWidget(win.data_tab) if hasattr(win, "tabs") else None
        tab = win.data_tab
        tab.overview.show()
        qapp.processEvents()
        # The variable before Resolution, the buttons and the box after it, on one row.
        buttons, box = tab.overview.bar_tail
        row = buttons.parentWidget()
        assert row is tab.overview.bar_lead.parentWidget() is box.parentWidget()
        assert row.isVisibleTo(tab.overview)
        assert tab.exclude_btn.parentWidget() is buttons and tab.apply_btn.parentWidget() is box
        # Wide, one line: the controls to the right of the variable and the resolution.
        assert tab.overview.bar_lead.y() == buttons.y() or row.width() < 1100
        tab.overview.rule = None
        # As recorded, so that a click names an hour.
        tab.var_combo.setCurrentText(tab.var_combo.currentText())
        combo = [c for c in row.findChildren(type(tab.var_combo)) if c is not tab.var_combo][0]
        combo.setCurrentText("As recorded")
        for _ in range(3):
            qapp.processEvents()
        canvas = tab.overview.canvas
        canvas.draw()
        tab._attach_pick()
        ax_s = canvas.figure.axes[0]
        frame = tab._drawn_frame["frame"]
        variable = tab._drawn_frame["variable"]

        def at(t, value=None):
            y = value if value is not None else float(np.mean(ax_s.get_ylim()))
            return ax_s.transData.transform((mdates.date2num(t), y))

        def mouse(name, xy, **kw):
            canvas.callbacks.process(name, MouseEvent(name, canvas, xy[0], xy[1], **kw))

        def close_in(t):
            # A click lands on a whole pixel, and across the months drawn an hour is less
            # than one: zoomed in on its day, as one would to click a single hour.
            day = pd.Timedelta("1D")
            ax_s.set_xlim(mdates.date2num(t - day), mdates.date2num(t + day))
            canvas.draw()

        # The cursor read at the lower left.
        t0 = pd.Timestamp("2020-03-23 05:00")
        v0 = float(frame.loc[t0, variable])
        whole = ax_s.get_xlim()
        close_in(t0)
        mouse("motion_notify_event", at(t0, v0))
        assert tab.hover.text() == f"2020-03-23 05:00: {variable} = {v0:.4g}"
        # A click on a point picks its row; with Shift another is added.
        for name in ("button_press_event", "button_release_event"):
            mouse(name, at(t0, v0), button=1)
        assert tab.picked == [(t0, t0)] and tab.rows_model.rowCount() == 1
        canvas = tab.overview.canvas  # drawn again with the pick
        canvas.draw()
        tab._attach_pick()
        ax_s = canvas.figure.axes[0]
        t1 = pd.Timestamp("2020-04-02 12:00")
        v1 = float(frame.loc[t1, variable])
        close_in(t1)
        for name in ("button_press_event", "button_release_event"):
            mouse(name, at(t1, v1), button=1, modifiers={"shift"} if "press" in name else None)
        assert tab.picked == [(t0, t0), (t1, t1)] and tab.rows_model.rowCount() == 2
        ax_s.set_xlim(whole)
        assert tab.pick_info.text().startswith("Picked 2 spans (2 rows)")
        # A right-click on the plot: the list's menu, for every row picked. On a time
        # picked, the pick kept whole; elsewhere the point under it picked first.
        from PySide6.QtWidgets import QMenu

        import normet.gui._data_tab as data_tab

        menus = []

        class _Menu(QMenu):
            def exec(self_, *a):
                menus.append([x.text() for x in self_.actions() if not x.isSeparator()])

        saved, data_tab.QMenu = data_tab.QMenu, _Menu
        try:
            canvas = tab.overview.canvas
            canvas.draw()
            tab._attach_pick()
            ax_s = canvas.figure.axes[0]
            close_in(t1)
            mouse("button_press_event", at(t1, v1), button=3)
            assert menus == [["Exclude 2 picked rows", "Restore 2 picked rows", "Clear pick"]]
            assert tab.picked == [(t0, t0), (t1, t1)]
            t2 = pd.Timestamp("2020-04-02 15:00")
            mouse("button_press_event", at(t2, float(frame.loc[t2, variable])), button=3)
            assert tab.picked == [(t2, t2)] and menus[-1][0] == "Exclude 1 picked rows"
        finally:
            data_tab.QMenu = saved
        ax_s.set_xlim(whole)
        # A drag picks the span crossed, afresh without Shift.
        canvas = tab.overview.canvas
        canvas.draw()
        tab._attach_pick()
        ax_s = canvas.figure.axes[0]
        a, b = pd.Timestamp("2020-05-01 00:20"), pd.Timestamp("2020-05-09 23:40")
        mouse("button_press_event", at(a), button=1)
        mouse("motion_notify_event", at(b))
        mouse("button_release_event", at(b), button=1)
        ((p0, p1),) = tab.picked  # to the minute the pixels name: within one of each end
        assert abs(p0 - a) < pd.Timedelta(hours=1) and abs(p1 - b) < pd.Timedelta(hours=1)
        assert abs(tab.rows_model.rowCount() - (8 * 24 + 23)) <= 2
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
