"""The window's uncertainty step: its verdict (pure), the box's enable rules,
and the step end to end offscreen (the runner, the Uncertainty tab, the light,
the command log, the config round-trip and a saved session bringing the run
back)."""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# -- the verdict, pure ---------------------------------------------------------------


def _res(weather=1.0, model=2.0, interaction=0.5, n=60, target="NO2"):
    from normet.analysis.uncertainty import UncDecomposition

    idx = pd.date_range("2024-01-01", periods=n, freq="12h")
    return UncDecomposition(
        var_weather=np.full(n, weather),
        var_model=np.full(n, model),
        var_interaction=np.full(n, interaction),
        var_total=np.full(n, weather + model + interaction),
        index=idx,
        target=target,
        n_models=4,
        n_weather_draws=300,
        seed=7,
        notes=["a caveat"],
    )


def test_the_verdict_names_the_larger_share():
    from normet.gui import _uncertainty as U

    level, text = U.verdict(_res(), "NO2 (4 members x 300 met draws)")
    assert level == "ok"
    # 1/3.5, 2/3.5 and 0.5/3.5 of the median spread, in the banner's whole percents.
    assert "29%" in text and "57%" in text and "14%" in text
    assert "model form carries the larger median share" in text


def test_the_verdict_turns_when_weather_larger():
    from normet.gui import _uncertainty as U

    _, text = U.verdict(_res(weather=4.0, model=1.0), "NO2")
    assert "met resampling (weather) carries the larger median share" in text


def test_the_verdict_stays_where_the_two_stand_level():
    from normet.gui import _uncertainty as U

    _, text = U.verdict(_res(weather=2.0, model=2.0), "NO2")
    assert "stand level" in text
    assert "carries the larger" not in text


def test_the_verdict_says_so_where_nothing_varies():
    from normet.gui import _uncertainty as U

    level, text = U.verdict(_res(weather=0.0, model=0.0, interaction=0.0), "NO2")
    assert level == "warn"
    assert "no variance to attribute" in text


# -- the step, end to end, offscreen ---------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _wait(qapp, until, timeout=300.0) -> None:
    t0 = time.time()
    while not until():
        qapp.processEvents()
        if time.time() - t0 > timeout:
            raise TimeoutError("the window did not reach what the test waits for")
        time.sleep(0.01)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A home of its own: the last session is kept under ~/.normet."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


@pytest.fixture
def win(qapp):
    from normet.gui.main_window import MainWindow

    w = MainWindow()
    yield w
    w.close()


def _prepared(win, n: int = 96) -> None:
    """A model in the window and the table it was trained on: hourly, one
    meteorological driver with weekly signal in it, a diurnal target cycle and
    noise. Small enough that three ensemble members train in a few seconds."""
    import normet as nm

    rng = np.random.default_rng(11)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    t = np.arange(n) / 24.0
    temp = 10.0 + 6.0 * np.sin(2.0 * np.pi * (t / 7.0 - 0.25)) + rng.normal(0, 0.4, n)
    hour = 3.5 * np.sin(2.0 * np.pi * t)
    df = pd.DataFrame(
        {
            "date": idx,
            "NO2": 20.0 + 2.0 * temp + hour + rng.normal(0, 1.0, n),
            "temp": temp,
        }
    )
    win.df_raw = df
    if win.target_combo.findText("NO2") < 0:
        win.target_combo.addItem("NO2")
    win.target_combo.setCurrentText("NO2")
    covs = ["temp", "date_unix", "day_julian", "weekday", "hour"]
    df_prep, model = nm.build_model(
        df,
        "NO2",
        backend="lightgbm",
        covariates=covs,
        model_config={"n_trials": 2, "cv_folds": 2, "nrounds": 150},
        seed=5,
        verbose=False,
    )
    win._pending_features = ["temp"]
    win._train_done((df_prep, model))
    assert win._selected_norm_vars() == ["temp"]  # Met only after training
    win.unc_samples.setValue(40)
    win.unc_models.setValue(3)
    win.budget_spin.setValue(2)
    win._sync_enabled()


def test_the_box_greys_while_there_is_no_model(win, qapp):
    assert not win.unc_button.isEnabled()
    assert win.unc_button.toolTip() == "Train a model first."
    _prepared(win)
    assert win.unc_button.isEnabled()
    assert "Uncertainty" in [win.tabs.tabText(i) for i in range(win.tabs.count())]


def test_the_button_greys_for_a_chronos_model(win, qapp):
    from normet.gui import _chronos

    _prepared(win)
    win.backend_combo.setCurrentText(_chronos.BACKEND)  # _is_chronos looks the combo up
    win._sync_enabled()
    assert not win.unc_button.isEnabled()
    assert "chronos-2" in win.unc_button.toolTip()


def test_the_step_runs_and_says_where_the_spread_goes(win, qapp):
    from PySide6.QtWidgets import QLabel

    _prepared(win)
    win._run_unc()
    _wait(
        qapp,
        lambda: (
            "uncertainty" in win.results and "#34c759" in win._step_lights["uncertainty"].text()
        ),
    )
    raw = win._unc_raw
    assert raw is not None and raw.n_models == 3
    frame = win.results["uncertainty"]
    assert list(frame.columns) == [
        "var_weather",
        "var_model",
        "var_interaction",
        "var_total",
        "pct_weather",
        "pct_model",
        "pct_interaction",
    ]
    # The attribution's own identity, in the window's table: the three summands
    # stack to the total on every date.
    assert np.allclose(
        frame[["var_weather", "var_model", "var_interaction"]].sum(axis=1), frame["var_total"]
    )
    labels = "\n".join(lbl.text() for lbl in win.tab_unc.findChildren(QLabel))
    assert "median spread" in labels, labels[:600]
    assert "met resampling" in labels or "model form" in labels, labels[:600]
    assert win.tabs.tabText(win.tabs.currentIndex()) in ("Uncertainty", "不确定性")
    # The command log knows the call the window made.
    entry = win.command_log.entries[-1]
    assert entry.title == "Uncertainty decomposition"
    assert "decompose_uncertainty(" in entry.code


def test_the_step_says_why_without_a_model(qapp):
    from normet.gui.main_window import MainWindow

    w = MainWindow()
    try:
        w._run_unc()  # the button is grey; a call straight in still says why, in words
        assert w.notices.get("need-model") is not None
        assert "uncertainty" not in w.results
    finally:
        w.close()


def test_the_box_settings_go_into_the_config_and_come_back(win, qapp):
    _prepared(win)
    win.unc_samples.setValue(120)
    win.unc_models.setValue(7)
    win.unc_backend.setCurrentText("flaml")
    c = win.get_config()
    assert c["unc_samples"] == 120 and c["unc_models"] == 7 and c["unc_backend"] == "flaml"
    win.unc_samples.setValue(300)
    win.unc_models.setValue(4)
    win.unc_backend.setCurrentText("lightgbm")
    win.apply_config(c)
    assert win.unc_samples.value() == 120
    assert win.unc_models.value() == 7
    assert win.unc_backend.currentText() == "flaml"
    win.apply_config({})  # a config without the rows keeps the box's defaults
    assert win.unc_samples.value() == 300 and win.unc_models.value() == 4


def test_the_saved_session_brings_the_run_back(qapp, home, tmp_path, win):
    """The run comes back with the session, its tab re-shown (its figure drawn
    again) from the run as it was, not from the panel as it stands."""
    import joblib

    from normet.gui import _saved
    from normet.gui.main_window import MainWindow

    _prepared(win)
    win.unc_samples.setValue(40)
    win._run_unc()
    _wait(qapp, lambda: "uncertainty" in win.results)
    first = win._unc_raw
    path = tmp_path / "unc.normet"
    joblib.dump(win.session_bundle(), path)

    v = MainWindow()
    try:
        v.load_results(str(path))
        assert "uncertainty" in v.results
        assert v._unc_raw is not None
        assert np.allclose(v._unc_raw.var_total, first.var_total)
        assert v.tab_unc.canvas is not None  # its figure drawn again from the saved run
        assert v.unc_samples.value() == 40  # the panel's row came back with it
        from PySide6.QtWidgets import QLabel

        labels = "\n".join(lbl.text() for lbl in v.tab_unc.findChildren(QLabel))
        assert "median spread" in labels, labels[:400]
        assert _saved.last_session_path().startswith(str(home))
    finally:
        v.close()
