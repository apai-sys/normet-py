"""The Resolution selector on time-series result tabs (offscreen)."""

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

    app = QApplication.instance() or QApplication([])
    yield app


def _hourly(days: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=24 * days, freq="h", name="date")
    return pd.DataFrame({"observed": np.arange(len(idx), dtype=float), "label": "x"}, index=idx)


def test_resample_frame_keeps_the_shape_it_was_given():
    from normet.gui._widgets import resample_frame

    df = _hourly(3)
    assert resample_frame(df, None) is df
    daily = resample_frame(df, "D")
    assert list(daily.columns) == ["observed"]  # numeric columns only
    assert len(daily) == 3 and daily["observed"].iloc[0] == pytest.approx(11.5)

    by_col = df.reset_index()
    out = resample_frame(by_col, "D")
    assert "date" in out.columns and len(out) == 3


def test_a_tab_redraws_at_the_chosen_resolution(qapp):
    from PySide6.QtWidgets import QComboBox

    from normet.gui._widgets import CanvasTab

    drawn: list[int] = []

    def render(d):
        import matplotlib.pyplot as plt

        drawn.append(len(d))
        fig, ax = plt.subplots()
        ax.plot(d.index, d["observed"])
        return fig

    tab = CanvasTab()
    data = _hourly(90)
    tab.show_timeseries(render, data, lines=["a note"])
    # Months of hours open as daily means: drawn as they are they are a solid band.
    assert drawn == [90]
    combo = tab.findChild(QComboBox)
    assert combo.currentText() == "Daily"
    model = combo.model()
    enabled = {combo.itemText(i): model.item(i).isEnabled() for i in range(combo.count())}
    # Hourly is the record's own step; a year is more than the record holds.
    assert enabled == {
        "As recorded": True,
        "Hourly": False,
        "Daily": True,
        "Weekly": True,
        "Monthly": True,
        "Yearly": False,
    }
    combo.setCurrentText("As recorded")
    qapp.processEvents()  # the redraw is queued
    assert drawn[-1] == len(data)
    combo = tab.findChild(QComboBox)
    assert combo.currentText() == "As recorded"
    # A new result on the tab keeps the tab's choice.
    tab.show_timeseries(render, data)
    assert drawn[-1] == len(data)
    # A short record opens as it is.
    other = CanvasTab()
    other.show_timeseries(render, _hourly(10))
    assert drawn[-1] == 240
    # Anything else shown there drops the series.
    tab.show_message("cleared")
    assert tab._series is None


def test_data_preview_offers_the_resolutions(qapp):
    from PySide6.QtWidgets import QComboBox

    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        win.load_example()
        combo = _combos(win.data_preview)["Resolution"]
        combo.setCurrentText("Monthly")
        qapp.processEvents()
        assert _combos(win.data_preview)["Resolution"].currentText() == "Monthly"
    finally:
        win.close()


def _combos(tab) -> dict[str, object]:
    """The tab's selectors by the label in front of each (in the group it stands in)."""
    from PySide6.QtWidgets import QComboBox, QLabel

    out = {}
    for combo in tab.findChildren(QComboBox):
        row, label = combo.parentWidget().layout(), None
        for i in range(row.count()):
            w = row.itemAt(i).widget()
            if w is combo:
                break
            if isinstance(w, QLabel):
                label = w.text()
        out[label] = combo
    return out


def test_a_tab_switches_between_its_results_and_keeps_the_resolution(qapp):
    from normet.gui._widgets import CanvasTab

    drawn: list[str] = []

    def render(name):
        def draw(d):
            import matplotlib.pyplot as plt

            drawn.append(f"{name}:{len(d)}")
            fig, ax = plt.subplots()
            ax.plot(d.index, d["observed"])
            return fig

        return draw

    tab = CanvasTab()
    one = {"emission": (render("emission"), _hourly(90), {})}
    tab.show_views(one, "emission")
    assert set(_combos(tab)) == {"Resolution"}  # nothing to pick between
    both = {**one, "meteorology": (render("meteorology"), _hourly(90), {"lines": ["met"]})}
    tab.show_views(both, "meteorology")
    combos = _combos(tab)
    assert list(combos) == ["Method", "Resolution"]
    assert combos["Method"].currentText() == "meteorology" and drawn[-1] == "meteorology:90"
    combos["Resolution"].setCurrentText("Weekly")
    qapp.processEvents()
    _combos(tab)["Method"].setCurrentText("emission")
    qapp.processEvents()
    combos = _combos(tab)
    assert combos["Method"].currentText() == "emission"
    assert combos["Resolution"].currentText() == "Weekly" and drawn[-1] == "emission:14"
    # A single result shown after drops the choice.
    tab.show_timeseries(render("rolling"), _hourly(90))
    assert set(_combos(tab)) == {"Resolution"}


def _decomposition(met: bool) -> pd.DataFrame:
    idx = pd.date_range("2021-01-01", periods=24 * 60, freq="h")
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"date": idx, "observed": rng.normal(30, 5, len(idx))})
    if met:
        d["met_total"] = rng.normal(0, 3, len(idx))
        d["met_base"] = 0.0
        d["t2m"] = rng.normal(0, 2, len(idx))
        d["met_noise"] = rng.normal(0, 1, len(idx))
    else:
        for c in ("emi_total", "date_unix", "day_julian", "weekday", "hour", "emi_noise"):
            d[c] = rng.normal(0, 1, len(idx))
        d["emi_base"] = 30.0
    return d


def test_both_decompositions_on_one_tab_with_a_method_to_pick(qapp):
    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        win.model = object()
        win._decompose_done(_decomposition(met=False))
        assert set(_combos(win.tab_decom)) == {"Resolution"}
        win._decompose_done(_decomposition(met=True))
        combos = _combos(win.tab_decom)
        assert combos["Method"].currentText() == "meteorology"
        assert [combos["Method"].itemText(i) for i in range(2)] == ["emission", "meteorology"]
        # Another model's decomposition does not sit beside this one's.
        win.model = object()
        win._decompose_done(_decomposition(met=False))
        assert set(_combos(win.tab_decom)) == {"Resolution"}
    finally:
        win.close()


def test_the_multiscale_tab_offers_the_resolutions(qapp):
    from normet.gui.main_window import MainWindow

    idx = pd.date_range("2020-01-01", periods=24 * 400, freq="h")
    wave = pd.Series(np.sin(np.arange(len(idx)) / 24.0), index=idx)
    result = {
        "Y_fast": wave + 3,
        "Y_meso": wave + 2,
        "Y_slow": wave + 1,
        "Y_inf": wave,
        "D_fast": wave * 0 + 1.0,
        "D_meso": wave * 0 + 1.0,
        "D_slow": wave,
        "fast_days": 14,
        "meso_days": 90,
        "slow_days": 365,
        "notes": [],
    }
    win = MainWindow()
    try:
        win._multiscale_done(result)
        combo = _combos(win.tab_multiscale)["Resolution"]
        assert combo.currentText() == "Daily"  # a dense record opens as daily means
        combo.setCurrentText("Monthly")
        qapp.processEvents()
        assert _combos(win.tab_multiscale)["Resolution"].currentText() == "Monthly"
        # The bands' σ are the record's, whatever resolution is drawn.
        titles = [ax.get_title("left") for ax in win.tab_multiscale.canvas.figure.axes]
        assert any(f"σ = {wave.std():.2f}" in t for t in titles)
        assert "multiscale" in win.results
    finally:
        win.close()


def test_the_row_centres_its_controls_and_hangs_a_tall_box_from_their_top(qapp):
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtWidgets import QWidget

    from normet.gui._widgets import FlowLayout

    host = QWidget()
    flow = FlowLayout(host, spacing=4, center=True)
    flow.setContentsMargins(0, 0, 0, 0)
    made = {}
    for name, h in (("button", 25), ("combo", 21), ("box", 44)):
        made[name] = w = QWidget()
        w.setFixedSize(60, h)
        flow.addWidget(w)
    flow.setGeometry(QRect(0, 0, 400, 50))
    # All centred on the line: the button and the combo on the box's middle.
    assert (made["box"].y(), made["button"].y(), made["combo"].y()) == (0, 9, 11)
    # The box hung from the controls' top: theirs is the button's, the combo on its middle.
    flow.setAlignment(made["box"], Qt.AlignmentFlag.AlignTop)
    flow.setGeometry(QRect(0, 0, 400, 50))
    assert (made["box"].y(), made["button"].y(), made["combo"].y()) == (0, 0, 2)
