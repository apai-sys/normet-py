"""The window's trend step: its series, its auto seasonal, its verdict, its figure
(pure; no window), and the step end to end offscreen (the runner, the Trend tab,
the light, and the Assistant's trend tool driving it as its button does)."""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from normet.analysis.trend import TrendResult, TrendSlope, TrendTest  # noqa: E402
from normet.gui import _trend as T  # noqa: E402

# -- the series ---------------------------------------------------------------------


def _frame(n: int = 48, start: str = "2024-01-01", freq: str = "h") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({"date": idx, "emi_total": np.arange(n, dtype=float), "observed": 1.0})


def test_trend_series_by_date_column():
    s = T.trend_series(_frame(10), "emi_total")
    assert isinstance(s.index, pd.DatetimeIndex) and s.name == "emi_total"
    assert list(s.index) == list(pd.date_range("2024-01-01", periods=10, freq="h"))


def test_trend_series_by_datetime_index():
    f = _frame(10).set_index("date")
    s = T.trend_series(f, "emi_total")
    assert isinstance(s.index, pd.DatetimeIndex) and len(s) == 10


def test_trend_series_without_time_stays_positional():
    f = pd.DataFrame({"emi_total": [1.0, 2.0, 3.0]})
    s = T.trend_series(f, "emi_total")
    assert not isinstance(s.index, pd.DatetimeIndex)


def test_trend_series_names_the_missing_column():
    with pytest.raises(ValueError, match="has no 'nope'"):
        T.trend_series(_frame(3), "nope")


# -- the auto's seasonal ---------------------------------------------------------------


def test_auto_seasonal_reads_the_step():
    hourly = pd.date_range("2024-01-01", periods=24 * 10, freq="h")
    assert T.auto_seasonal(hourly, len(hourly)) == (24, "")
    half = pd.date_range("2024-01-01", periods=48 * 5, freq="30min")
    assert T.auto_seasonal(half, len(half)) == (48, "")
    daily = pd.date_range("2024-01-01", periods=7 * 20, freq="D")
    assert T.auto_seasonal(daily, len(daily)) == (7, "")
    monthly = pd.date_range("2024-01-01", periods=12 * 3, freq="MS")
    assert T.auto_seasonal(monthly, len(monthly)) == (12, "")


def test_auto_seasonal_falls_back_when_the_record_is_not_whole():
    hourly = pd.date_range("2024-01-01", periods=24 * 10 + 16, freq="h")
    period, note = T.auto_seasonal(hourly, len(hourly))
    assert period is None and "16 extra" in note and "classical" in note


def test_auto_seasonal_gives_nothing_it_cannot_name():
    short = pd.date_range("2024-01-01", periods=9, freq="D")
    assert T.auto_seasonal(short, len(short)) == (None, "")
    yearly = pd.date_range("2024-01-01", periods=40, freq="YS")
    assert T.auto_seasonal(yearly, len(yearly)) == (None, "")


# -- the call ------------------------------------------------------------------------


def test_trend_call_runs_normet_trend_on_the_column():
    n = 24 * 30
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    f = pd.DataFrame(
        {"date": idx, "emi_total": (1.0 / 8760) * np.arange(n) / 24 + rng.normal(0, 0.4, n)}
    )
    res = T.trend_call(f, "emi_total", series_name="demo", n_bootstrap=120)
    assert type(res).__name__ == "TrendResult" and res.name == "demo" and res.n == n
    assert isinstance(res.summary(), pd.DataFrame)


def test_trend_call_sees_the_seasonal_period():
    n = 24 * 14
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    f = pd.DataFrame({"date": idx, "emi_total": rng.normal(0, 1.0, n) + 0.004 * np.arange(n)})
    res = T.trend_call(f, "emi_total", seasonal_period=24, n_bootstrap=120)
    assert res.test.seasonal and res.test.n_periods == 24


# -- the verdict -----------------------------------------------------------------------


def _mk(S: int, p: float, p_boot: float | None, ar: bool, acf: float = 0.5) -> TrendResult:
    """A trend result by hand: its direction is what its S and p say (a property)."""
    t = TrendTest(S=S, var_S=1.0, z=3.0, p=p, n=400)
    s = TrendSlope(slope=0.01, ci_low=0.005, ci_high=0.015, n=400, time_unit="day")
    return TrendResult(
        name="demo",
        n=400,
        n_dropped=0,
        test=t,
        slope=s,
        ar1_acf=acf,
        ar1_p=1e-6,
        ar_autocorrelated=ar,
        p_wald_boot=p_boot,
        notes=[],
    )


def test_verdict_significant_on_the_classical_test_alone():
    level, text = T.trend_verdict(_mk(100, 0.001, None, False), "demo")
    assert level == "ok"
    assert "significant increasing trend" in text and "MK p 0.001" in text


def test_verdict_no_trend():
    level, text = T.trend_verdict(_mk(0, 0.3, None, False), "demo")
    assert level == "warn" and "no significant trend" in text


def test_verdict_autocorrelated_and_not_confirmed_does_not_allow_a_quote():
    level, text = T.trend_verdict(_mk(100, 0.001, 0.4, True), "demo")
    assert level == "warn" and "does not confirm it (p 0.4)" in text and "Do not quote" in text


def test_verdict_autocorrelated_and_confirmed_quotes_the_bootstrap():
    level, text = T.trend_verdict(_mk(100, 0.001, 0.002, True), "demo")
    assert level == "ok" and "the corrected p is 0.002" in text


def test_verdict_autocorrelated_with_found_only_by_the_bootstrap():
    level, text = T.trend_verdict(_mk(100, 0.2, 0.003, True), "demo")
    assert level == "ok" and "the classical test missed" in text


def test_verdict_autocorrelated_but_no_bootstrap_says_so_plainly():
    level, text = T.trend_verdict(_mk(100, 0.001, None, True), "demo")
    assert level == "warn" and "no usable interval" in text and "do not quote" in text


# -- the figure ------------------------------------------------------------------------


def test_trend_fig_draws_the_series_and_the_line():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    s = pd.Series(0.01 * np.arange(n) + rng.normal(0, 0.3, n), index=idx)
    frame = pd.DataFrame({"date": idx, "emi_total": s.to_numpy()})
    res = T.trend_call(frame, "emi_total", series_name="demo", n_bootstrap=100)
    fig = T.trend_fig(s, res)
    try:
        assert len(fig.axes) == 1
        assert len(fig.axes[0].get_lines()) == 2  # the series and its robust line
    finally:
        plt.close(fig)


def test_trend_fig_without_enough_values_is_only_the_series():
    import matplotlib.pyplot as plt

    idx = pd.date_range("2024-01-01", periods=5, freq="h")
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
    fig = T.trend_fig(s, _mk(100, 0.001, None, False))
    try:
        assert len(fig.axes[0].get_lines()) == 1
    finally:
        plt.close(fig)


# -- the step, end to end, offscreen ----------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _wait(qapp, until, timeout=240.0) -> None:
    t0 = time.time()
    while not until():
        qapp.processEvents()
        if time.time() - t0 > timeout:
            raise TimeoutError("the window did not reach what the test waits for")
        time.sleep(0.01)


@pytest.fixture
def win(qapp):
    from normet.gui.main_window import MainWindow

    w = MainWindow()
    yield w
    w.close()


def _prepared(win, n: int = 500) -> pd.DataFrame:
    """A model in the window and a decomposition it could have made: hourly, rising."""
    rng = np.random.default_rng(3)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    eps = np.zeros(n)
    for i in range(1, n):
        eps[i] = 0.6 * eps[i - 1] + rng.normal(0, 0.3)
    emi = 0.08 * np.arange(n) / 24.0 + eps  # a rising emission part, autocorrelated noise
    win.model = object()
    win.df_prep = pd.DataFrame({"date": idx, "NO2": emi + 1.0})
    if win.target_combo.findText("NO2") < 0:
        win.target_combo.addItem("NO2")
    win.target_combo.setCurrentText("NO2")
    result = pd.DataFrame(
        {"observed": emi + 1.0, "emi_total": emi, "met_total": 1.0},
        index=pd.DatetimeIndex(idx, name="date"),
    )
    win.results["decompose"] = result
    win.tre_n_bootstrap.setValue(100)
    win.trend_after.setChecked(False)
    win._sync_enabled()
    return result


def test_the_trend_step_runs_and_says_its_verdict(win, qapp):
    from PySide6.QtWidgets import QLabel

    _prepared(win)
    win._run_trend()
    _wait(
        qapp,
        lambda: "trend" in win.results and "#34c759" in win._step_lights["trend"].text(),
    )
    res = win._trend_raw
    assert res is not None and res.test.p < 0.05  # the record it was built to trend
    labels = "\n".join(lbl.text() for lbl in win.tab_trend.findChildren(QLabel))
    assert "significant" in labels, labels[:400]
    assert "Trend" in [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert win.tabs.tabText(win.tabs.currentIndex()) == "Trend"


def test_the_assistants_trend_tool_runs_the_step(win, qapp):
    _prepared(win)
    box: dict = {}

    def call() -> None:
        box["said"] = win.assistant.tools.trend(series="emission", seasonal="auto")

    t = threading.Thread(target=call)
    t.start()
    _wait(qapp, lambda: "said" in box)
    t.join(timeout=10)
    said = box["said"]
    assert "Trend test on NO2 — emission part (normet.trend)" in said, said[:400]
    assert "Mann-Kendall" in said and "slope (per day)" in said
    assert "AR(1)" in said


def test_the_trend_tool_refuses_what_the_window_cannot_do(win, qapp):
    _prepared(win)
    box: dict = {}

    def call() -> None:
        try:
            win.assistant.tools.trend(series="nope")
        except Exception as exc:  # noqa: BLE001
            box["exc"] = exc

    t = threading.Thread(target=call)
    t.start()
    _wait(qapp, lambda: "exc" in box or not t.is_alive(), timeout=60)
    t.join(timeout=10)
    assert "series is 'emission'" in str(box["exc"])


def test_the_auto_trend_starts_when_the_decomposition_ends(win, qapp):
    _prepared(win)
    win.trend_after.setChecked(True)
    win._trend_wanted = True
    win._task_finished()  # the runner's own end: the runner is free now, the trend may start
    _wait(qapp, lambda: "trend" in win.results)
    assert win._trend_wanted is False
