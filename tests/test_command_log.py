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
    # training then refused. It is read as UTC, and the logged script agrees.
    from normet.utils.prepare import add_date_variables

    path = tmp_path / "bst.csv"
    path.write_text(
        "date,NO2\n2020-03-29T00:00+00:00,1\n2020-03-29T02:00+01:00,2\n2020-03-29T03:00+01:00,3\n"
    )
    win.load_csv(str(path))
    dates = win.df_raw["date"]
    assert str(dates.dt.tz) == "UTC" and dates.dt.hour.tolist() == [0, 1, 2]
    assert add_date_variables(win.df_raw)["hour"].tolist() == [0, 1, 2]
    ns: dict = {}
    exec(compile(win.command_log.script(), "session.py", "exec"), ns)
    pd.testing.assert_frame_equal(ns["df"], win.df_raw)


# -- the print layout ---------------------------------------------------------------------
def _figure():
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 5))
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1], color="black")
    ax.set_title("Title")
    ax.set_xlabel("x")
    return fig, ax


def test_print_layout_is_undone():
    from normet.gui.printview import print_layout

    fig, ax = _figure()
    sizes = [t.get_fontsize() for t in (ax.title, ax.xaxis.label)]
    with print_layout(fig, 89, 60, 7):
        assert np.allclose(fig.get_size_inches(), (89 / 25.4, 60 / 25.4))
        assert ax.xaxis.label.get_fontsize() == 7 and ax.title.get_fontsize() == 8
    assert tuple(fig.get_size_inches()) == (10, 5)
    assert [t.get_fontsize() for t in (ax.title, ax.xaxis.label)] == sizes


def test_saved_at_exactly_the_printed_width_on_a_white_page(qapp):
    from PIL import Image

    from normet.gui.printview import render, save
    from normet.gui.theme import apply_figure_theme

    fig, _ = _figure()
    apply_figure_theme(fig, True)  # shown dark on the window
    png = render(fig, 89, 60, 7, dpi=300)
    img = Image.open(io.BytesIO(png)).convert("RGB")
    assert img.size == (int(89 / 25.4 * 300), int(60 / 25.4 * 300))  # matplotlib truncates
    assert img.getpixel((2, 2)) == (255, 255, 255)
    assert getattr(fig, "_normet_dark", False)  # and dark again on the window
    buf = io.BytesIO()
    save(fig, buf, 183, 90, 7, 300, fmt="pdf")
    assert buf.getvalue()[:4] == b"%PDF"


def test_the_dialog_previews_and_follows_its_preset(qapp):
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    from normet.gui._widgets import ThemedToolbar
    from normet.gui.printview import PRESETS, SCREEN_DPI, PrintDialog

    fig, _ = _figure()
    canvas = FigureCanvasQTAgg(fig)
    toolbar = ThemedToolbar(canvas)
    assert any(a.text() == "For print…" for a in toolbar.actions())
    dlg = PrintDialog(canvas)
    assert dlg.width.value() == 89.0 and not dlg.width.isEnabled()
    assert fig.canvas is dlg.canvas  # the figure itself, on the dialog's canvas
    assert dlg.canvas.width() == round(89 / 25.4 * SCREEN_DPI)
    dlg.preset.setCurrentText("Nature: double column (183 mm)")
    assert dlg.width.value() == 183.0 and round(dlg.height.value()) == round(183 * 0.5)
    assert np.allclose(fig.get_size_inches(), (183 / 25.4, dlg.height.value() / 25.4))
    dlg.preset.setCurrentText("Custom")
    assert dlg.width.isEnabled() and set(PRESETS) >= {"Custom"}
    dlg.reject()
    assert fig.canvas is canvas and tuple(fig.get_size_inches()) == (10, 5)


def _two_panels():
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 6))
    a, b = fig.subplots(2, 1)
    a.plot(range(10), np.arange(10.0), color="black", label="observed")
    a.plot(range(10), np.arange(10.0) / 2, label="normalised")
    a.axhline(0, color="k", lw=0.5)  # a reference line: two points
    a.legend(loc="lower right")
    a.set_title("Normalised")
    b.plot(range(10), np.ones(10))
    b.set_ylabel("met part")
    return fig, a, b


def _state(fig):
    """What the dialog may touch, to compare before and after."""
    from matplotlib.text import Text

    texts = [(t.get_text(), t.get_fontsize(), t.get_rotation()) for t in fig.findobj(Text)]
    lines = [
        (ln.get_linewidth(), ln.get_marker(), ln.get_markersize())
        for ax in fig.axes
        for ln in ax.get_lines()
    ]
    legends = [
        (id(ax.get_legend()), ax.get_legend()._loc) if ax.get_legend() else None for ax in fig.axes
    ]
    return tuple(fig.get_size_inches()), fig.dpi, texts, lines, legends, fig.get_layout_engine()


def test_finishing_touches_are_made_and_undone(qapp, tmp_path):
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.colors import to_hex
    from PIL import Image

    from normet.gui.printview import PrintDialog
    from normet.gui.theme import DARK, apply_figure_theme

    fig, a, b = _two_panels()
    canvas = FigureCanvasQTAgg(fig)
    apply_figure_theme(fig, True)  # shown dark on its tab
    canvas.draw()  # ticks are made at a draw: compare drawn with drawn
    before = _state(fig)
    dlg = PrintDialog(canvas)
    assert to_hex(fig.get_facecolor()) == "#ffffff"  # a white page while it is open

    dlg.legend_place.setCurrentText("Upper left")
    dlg.legend_cols.setValue(2)
    dlg.legend_frame.setChecked(True)
    leg = a.get_legend()
    assert leg._loc == 2 and leg._ncols == 2 and leg.get_frame_on()
    dlg.legend_at[a] = (0.3, 0.4)  # as a drag leaves it
    dlg.apply()
    assert a.get_legend()._loc == (0.3, 0.4)

    dlg.line_width.setValue(1.5)
    dlg.marker_size.setValue(3)
    data, half, ref = a.get_lines()
    assert data.get_linewidth() == 1.5 and data.get_marker() == "o" and data.get_markersize() == 3
    assert ref.get_linewidth() == 0.5 and ref.get_marker() in ("None", None, "")  # left alone

    dlg.letters.setCurrentText("(a)  (b)  (c)")
    assert {t.get_text() for t in a.texts} == {"(a)"} and {t.get_text() for t in b.texts} == {"(b)"}

    dlg.set_text(a.title, "Normalised NO₂", 9)
    assert a.title.get_text() == "Normalised NO₂" and a.title.get_fontsize() == 9
    dlg.set_ticks(b.yaxis, 6, 45)
    assert all(
        lbl.get_fontsize() == 6 and lbl.get_rotation() == 45 for lbl in b.yaxis.get_ticklabels()
    )

    dlg.save_to(tmp_path / "fig.png")
    img = Image.open(tmp_path / "fig.png")
    assert img.size[0] == int(89 / 25.4 * 300)

    dlg.reject()
    assert fig.canvas is canvas
    canvas.draw()
    assert _state(fig) == before
    assert to_hex(fig.get_facecolor()) == DARK["figure"]  # dark again on its tab
    assert len(a.texts) == 0 and len(b.texts) == 0  # the panel letters gone


def test_a_double_click_finds_what_it_is_on(qapp, monkeypatch):
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    from normet.gui import printview

    fig, a, b = _two_panels()
    dlg = printview.PrintDialog(FigureCanvasQTAgg(fig))
    fig.canvas.draw()

    def at(artist):
        box = artist.get_window_extent()
        return MouseEvent(
            "button_press_event",
            fig.canvas,
            (box.x0 + box.x1) / 2,
            (box.y0 + box.y1) / 2,
            dblclick=True,
        )

    monkeypatch.setattr(
        printview._TextChoices, "exec", lambda self: (self.text.setText("Edited"), True)[1]
    )
    dlg.edit_at(at(b.yaxis.label))
    assert b.yaxis.label.get_text() == "Edited"
    monkeypatch.setattr(
        printview._TickChoices, "exec", lambda self: (self.size.setValue(5), True)[1]
    )
    tick = next(lbl for lbl in a.xaxis.get_ticklabels() if lbl.get_text())
    dlg.edit_at(at(tick))
    assert a.xaxis in dlg.tick_edits and dlg.tick_edits[a.xaxis][0] == 5
    fig.canvas.draw()  # the legend made again by that pass, placed at a draw
    assert dlg._legend_under(at(a.get_legend())) is not None
    dlg.reject()
    assert b.yaxis.label.get_text() == "met part"
