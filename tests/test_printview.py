"""The figure for print in the window (normet.gui.printview, after cume's): a tab's figure
laid out at a journal's width in its lettering, finished where it stands, saved at its
size, and given back to its tab as it was."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


@pytest.fixture
def kept(monkeypatch):
    """The choices the dialog keeps for the next figure, kept here rather than in the
    machine's settings."""
    from normet.gui import printview

    store: dict = {}
    monkeypatch.setattr(printview, "load_settings", lambda: dict(store))
    monkeypatch.setattr(printview, "keep_settings", lambda s: store.update(s))
    return store


def _two_panels():
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(10, 6))
    a, b = fig.subplots(2, 1)
    a.plot(range(10), np.arange(10.0), color="black", label="observed")
    a.plot(range(10), np.arange(10.0) / 2, label="normalised NO2")
    a.plot(range(10), np.arange(10.0) / 3, label="seeded")
    a.axhline(0, color="k", lw=0.5)  # a reference line: two points
    a.legend(loc="lower right", ncols=3)
    a.set_title("Normalised")
    b.plot(range(10), np.ones(10))
    b.set_ylabel("met part")
    canvas = FigureCanvasQTAgg(fig)
    canvas.draw()  # ticks are made at a draw: compare drawn with drawn
    return fig, a, b, canvas


def _state(fig):
    """What the dialog may touch, to compare before and after."""
    from matplotlib.text import Text

    texts = [(t.get_text(), t.get_fontsize(), t.get_rotation()) for t in fig.findobj(Text)]
    lines = [
        (ln.get_linewidth(), ln.get_marker(), ln.get_markersize())
        for ax in fig.axes
        for ln in ax.get_lines()
    ]
    legends = [(ax.get_legend()._loc, ax.get_legend()._ncols) for ax in fig.axes if ax.get_legend()]
    ticks = [dict(ax.xaxis._major_tick_kw) for ax in fig.axes]
    return tuple(fig.get_size_inches()), fig.dpi, texts, lines, legends, ticks


def test_a_figure_is_laid_out_for_print_and_given_back(qapp, tmp_path, kept):
    from matplotlib.colors import to_hex
    from PIL import Image

    from normet import figstyle as FS
    from normet.gui.printview import PrintDialog
    from normet.gui.theme import DARK, apply_figure_theme

    fig, a, b, canvas = _two_panels()
    apply_figure_theme(fig, True)  # shown dark on its tab
    canvas.draw()
    before = _state(fig)
    dlg = PrintDialog(canvas, settings={"preset": "nature"})
    assert fig.canvas is dlg.canvas  # the figure itself, on the dialog's canvas
    assert to_hex(fig.get_facecolor()) == "#ffffff"  # a white page while it is here
    assert np.isclose(fig.get_size_inches()[0] * FS.MM, 89)  # one column: Nature's single
    assert not a.title.get_visible()  # a caption's
    assert {t.get_text() for t in a.texts} == {"a"} and {t.get_text() for t in b.texts} == {"b"}

    dlg.width_cb.setCurrentIndex(1)  # double column
    assert np.isclose(fig.get_size_inches()[0] * FS.MM, 183)
    dlg.labels_cb.setCurrentIndex(2)  # (a) (b) (c)
    dlg.start_cb.setCurrentIndex(2)  # panels c and d of a composite
    assert {t.get_text() for t in a.texts} == {"(c)"}

    dlg.legend_ncol.setValue(1)  # the entries laid out again in one column
    leg = a.get_legend()
    assert leg._ncols == 1 and len(leg._legend_handle_box.get_children()) == 1
    dlg.legend_cb.setCurrentIndex(3)  # upper left
    assert leg._loc == 2
    dlg.legend_frame.setChecked(True)
    assert leg.get_frame_on()

    dlg.line_width.setValue(1.5)
    dlg.marker_size.setValue(3)
    data, *_, ref = a.get_lines()
    assert data.get_linewidth() == 1.5 and data.get_marker() == "o" and data.get_markersize() == 3
    assert ref.get_linewidth() == 0.5  # a reference line keeps its stroke

    dlg._set_label(b.yaxis.label, "Met part", 6.5)
    dlg._apply()  # laid over every pass
    assert b.yaxis.label.get_text() == "Met part" and b.yaxis.label.get_fontsize() == 6.5
    dlg._set_tick(b.yaxis, None, None, 5, 45)
    assert all(t.get_rotation() == 45 for t in b.yaxis.get_ticklabels())

    found = {f.about: f.ok for f in dlg.findings}
    assert found["width"] and "text" in found and "tick labels" in found
    assert dlg.checks.count() == len(dlg.findings)

    dlg.dpi_cb.setCurrentIndex(0)
    path = dlg._save(str(tmp_path / "fig.png"))
    with Image.open(path) as img:
        assert abs(img.size[0] - round(183 / FS.MM * 300)) <= 1
    assert dlg._save(str(tmp_path / "fig.pdf")) and (tmp_path / "fig.pdf").stat().st_size

    assert kept["preset"] == "nature" and kept["width"] == 183 and kept["labels"] == "(a)"
    dlg.reject()
    assert fig.canvas is canvas
    canvas.draw()
    assert _state(fig) == before
    assert to_hex(fig.get_facecolor()) == DARK["figure"]  # dark again on its tab
    assert not a.texts and not b.texts  # the panel letters gone
    assert a.title.get_visible()


def test_the_choices_are_kept_for_the_next_figure(qapp, kept):
    from normet.figstyle import PRESETS
    from normet.gui.printview import PrintDialog

    dlg = PrintDialog(_two_panels()[3])
    dlg.preset_cb.setCurrentIndex(list(PRESETS).index("acs"))
    dlg.dpi_cb.setCurrentIndex(1)
    dlg.reject()
    again = PrintDialog(_two_panels()[3])
    assert again.preset_cb.currentData() == "acs" and again.dpi_cb.currentData() == 600
    again.reject()


def test_a_double_click_finds_what_it_is_on(qapp, kept):
    from matplotlib.backend_bases import MouseEvent

    from normet.gui.printview import PrintDialog

    fig, a, b, canvas = _two_panels()
    dlg = PrintDialog(canvas)
    dlg.canvas.draw()

    def centre(artist):
        box = artist.get_window_extent(dlg.canvas.get_renderer())
        return (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2

    assert dlg._what_at(*centre(b.yaxis.label)) == ("label", b.yaxis.label)
    assert dlg._what_at(*centre(a.get_legend()))[0] == "legend"
    tick = next(t for t in a.get_xticklabels() if t.get_text())
    kind, axis, _ = dlg._what_at(*centre(tick))
    assert kind == "tick" and axis is a.xaxis
    # A drag moves the legend; closing puts it back where it was drawn.
    x, y = centre(a.get_legend())
    press = MouseEvent("button_press_event", dlg.canvas, x, y, button=1)
    dlg._pressed(press)
    dlg._moved(MouseEvent("motion_notify_event", dlg.canvas, x - 40, y + 30))
    dlg._released(MouseEvent("button_release_event", dlg.canvas, x - 40, y + 30, button=1))
    assert a.get_legend()._loc != 4
    dlg.reject()
    assert a.get_legend()._loc == 4
