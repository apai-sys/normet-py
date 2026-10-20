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
        combo = win.data_preview.findChild(QComboBox)
        assert combo is not None
        combo.setCurrentText("Monthly")
        qapp.processEvents()
        assert win.data_preview.findChild(QComboBox).currentText() == "Monthly"
    finally:
        win.close()
