"""The GUI's appearance, command palette and welcome (offscreen; skipped without PySide6)."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _figure():
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure

    fig = Figure()
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1], color="black", label="observed")
    ax.plot([0, 1], [1, 0], color="#2c7bb6", label="normalised")
    ax.axhline(0.5, color="k", lw=0.5)
    ax.scatter([0.2, 0.8], [0.3, 0.7], c=[1.0, 2.0])  # coloured by value: left alone
    ax.bar([0.5], [0.4], color="white", edgecolor="black")
    ax.set_title("t")
    ax.grid(True)
    ax.legend()
    return fig, ax


def _colours(fig):
    from matplotlib.colors import to_hex

    out = [to_hex(fig.get_facecolor())]
    for a in fig.findobj():
        for get in ("get_color", "get_facecolor", "get_edgecolor"):
            f = getattr(a, get, None)
            if f is None:
                continue
            try:
                out.append(repr(f()))
            except Exception:
                pass
    return out


def test_dark_figure_round_trips_to_its_own_colours(qapp):
    from matplotlib.colors import to_hex

    from normet.gui.theme import DARK, apply_figure_theme, light_figure

    fig, ax = _figure()
    before = _colours(fig)

    apply_figure_theme(fig, True)
    assert to_hex(fig.get_facecolor()) == DARK["figure"]
    assert to_hex(ax.title.get_color()) == DARK["ink"]
    lines = ax.get_lines()
    assert to_hex(lines[0].get_color()) == DARK["ink"]  # black ink follows the theme
    assert to_hex(lines[1].get_color()) == "#2c7bb6"  # a series' own colour does not
    apply_figure_theme(fig, True)  # applying it again changes nothing more

    with light_figure(fig):
        assert _colours(fig) == before
    assert to_hex(fig.get_facecolor()) == DARK["figure"]

    apply_figure_theme(fig, False)
    assert _colours(fig) == before


def test_canvas_tab_follows_the_theme(qapp):
    from matplotlib.colors import to_hex

    from normet.gui._widgets import CanvasTab
    from normet.gui.theme import DARK, theme_manager

    tab = CanvasTab()
    fig, _ = _figure()
    tab.show_result(fig)
    tab._retheme(True)
    assert to_hex(tab.canvas.figure.get_facecolor()) == DARK["figure"]
    tab._retheme(False)
    assert to_hex(tab.canvas.figure.get_facecolor()) == "#ffffff"
    assert theme_manager() is theme_manager()
    tab.clear()


def test_matches_ranks_prefix_then_name_then_place():
    from normet.gui.palette import Command, matches

    cmds = [
        Command("Export HTML Report…", "File", lambda: None),
        Command("Open CSV…", "File", lambda: None),
        Command("Load Example Data", "File", lambda: None),
        Command("Example disabled", "File", lambda: None, enabled=False),
        Command("Model", "Tab", lambda: None),
    ]
    assert [c.text for c in matches(cmds, "exa")] == ["Load Example Data", "Example disabled"]
    assert [c.text for c in matches(cmds, "open")] == ["Open CSV…"]
    assert [c.text for c in matches(cmds, "tab")] == ["Model"]
    assert matches(cmds, "") == cmds


def test_palette_lists_menus_and_tabs(qapp):
    from normet.gui.main_window import MainWindow
    from normet.gui.palette import CommandPalette, menu_commands, tab_commands

    win = MainWindow()
    try:
        names = {c.text for c in menu_commands(win, skip=(win._palette_action,))}
        assert {"Open CSV…", "Load Example Data", "Run History…"} <= names
        assert "Command Palette…" not in names
        tabs = tab_commands(win.tabs)
        assert [c.text for c in tabs][:2] == ["Data", "Model"]
        tabs[1].run()
        assert win.tabs.currentWidget() is win.tab_model

        dlg = CommandPalette(menu_commands(win) + tabs, win)
        dlg.field.setText("example data")
        assert dlg.shown()[0].text == "Load Example Data"
        dlg._run()
        assert dlg.chosen is not None and dlg.chosen.text == "Load Example Data"
        dlg.deleteLater()
    finally:
        win.close()


def test_welcome_until_a_table_is_open(qapp, monkeypatch, tmp_path):
    from normet.gui import _settings
    from normet.gui.main_window import MainWindow

    csv = tmp_path / "site.csv"
    csv.write_text("date,PM2.5\n2024-01-01,1\n")
    monkeypatch.setattr(_settings, "read_recent", lambda: [str(csv)])

    win = MainWindow()
    try:
        assert not win.welcome.isHidden() and win._data_parts.isHidden()
        assert set(win.welcome.cards) == {"examples", "recent", "data"}
        (button, line) = win.welcome.cards["recent"].rows[0]
        assert button.text() == "site.csv"
        assert line.toolTip().endswith(f"24 bytes · {line.toolTip()[-10:]}")  # folder · size · date
        win.load_example()
        assert win.welcome.isHidden() and not win._data_parts.isHidden()
    finally:
        win.close()


def test_appearance_is_remembered(qapp, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from normet.gui import theme

    ini = str(tmp_path / "normet-gui.ini")
    monkeypatch.setattr(theme, "_settings", lambda: QSettings(ini, QSettings.Format.IniFormat))
    monkeypatch.setattr(theme, "apply_stored_appearance", lambda: None)
    assert theme.stored_appearance() == "system"
    theme.theme_manager().set_appearance("dark")
    assert theme.stored_appearance() == "dark"
    assert QSettings(ini, QSettings.Format.IniFormat).value("appearance") == "dark"
    with pytest.raises(ValueError):
        theme.theme_manager().set_appearance("sepia")


def test_toolbar_icons_follow_the_palette(qapp):
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from PySide6.QtGui import QColor, QPalette

    from normet.gui._widgets import ThemedToolbar

    toolbar = ThemedToolbar(FigureCanvasQTAgg(Figure()))
    home = toolbar._actions["home"]

    def ink():
        img = home.icon().pixmap(24, 24).toImage()
        pixels = [img.pixelColor(x, y) for x in range(img.width()) for y in range(img.height())]
        drawn = [c for c in pixels if c.alpha() > 128]
        return sum(c.lightness() for c in drawn) / len(drawn)

    def palette(window, text):
        pal = QPalette(toolbar.palette())
        # The toolbar's own roles (a switch of the app's theme changes every role).
        pal.setColor(QPalette.ColorRole.Button, QColor(window))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor(text))
        toolbar.setPalette(pal)

    palette("#1c1c1e", "#ffffff")
    assert ink() > 200  # light icons on a dark bar
    palette("#ececec", "#000000")
    assert ink() < 80  # and dark again on a light one
