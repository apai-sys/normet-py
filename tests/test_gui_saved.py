"""File -> Save Results / Load Results / Reopen Last Session (offscreen)."""

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


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A home of its own: the last session is kept under ~/.normet."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


def _trained_window():
    """The example, a small LightGBM model and a normalisation, run in-process."""
    import normet as nm
    from normet.gui.main_window import MainWindow

    w = MainWindow()
    w.load_example()
    target = w.target_combo.currentText()
    feats = w._selected_features()
    covs = feats + ["date_unix", "day_julian", "weekday", "hour"]
    df_prep, model = nm.build_model(
        w.df_raw, target, backend="lightgbm", covariates=covs, model_config={"n_iter": 2}
    )
    w._pending_features = feats
    w._train_done((df_prep, model))
    norm = nm.normalise(
        df_prep, model, covariates=covs, variables_resample=feats, n_samples=5, n_cores=1
    )
    w._normalise_done(norm)
    return w


def test_a_saved_session_comes_back_as_it_was(qapp, home, tmp_path):
    import joblib

    from normet.gui import _saved
    from normet.gui.main_window import MainWindow

    w = _trained_window()
    w.decom_samples.setValue(42)
    w._log_command(None)
    path = tmp_path / "s.normet"
    joblib.dump(w.session_bundle(), path)
    w.close()

    v = MainWindow()
    try:
        v.load_results(str(path))
        assert v.model is not None and v.df_prep is not None
        assert "normalise" in v.results and "train" in v.results
        assert v.decom_samples.value() == 42
        assert v.tab_norm.canvas is not None  # its figure drawn again
        assert v.norm_button.isEnabled()
        bar = v.notices.get("restored")
        assert bar is not None and "normalise" in bar.label.text()
        lights = v._step_lights
        assert "#34c759" in lights["normalise"].text() and "#34c759" not in lights["pdp"].text()
        assert _saved.last_session_path().startswith(str(home))
    finally:
        v.close()


def test_the_last_session_is_kept_after_a_step(qapp, home):
    import time

    from normet.gui import _saved
    from normet.gui.main_window import MainWindow

    w = _trained_window()
    try:
        w.autosave_session()
        path = _saved.last_session_path()
        for _ in range(100):  # written off the GUI thread
            if os.path.exists(path):
                break
            time.sleep(0.05)
        assert os.path.exists(path)
    finally:
        w.close()
    v = MainWindow()
    try:
        for _ in range(100):
            try:
                v.reopen_last_session()
                break
            except Exception:  # the writer may still be replacing it
                time.sleep(0.05)
        assert "normalise" in v.results
    finally:
        v.close()


def test_nothing_kept_is_said_on_a_notice(qapp, home):
    from normet.gui.main_window import MainWindow

    w = MainWindow()
    try:
        w.reopen_last_session()
        bar = w.notices.get("no-session")
        assert bar is not None and "No session" in bar.label.text()
    finally:
        w.close()


SCRIPT = """\
from normet import build_model
from normet import normalise
from normet.gui._examples import make_deweather_example

# Load example data
df = make_deweather_example()

# Train model
df_prep, model = build_model(df, 'PM2.5', backend='lightgbm',
                             covariates=['t2m', 'blh', 'ws', 'wd', 'date_unix', 'day_julian', 'weekday', 'hour'],
                             split_method='random', train_fraction=0.7,
                             model_config={'n_trials': 2}, seed=7)

# Normalise
normalised = normalise(df_prep, model,
                       covariates=['t2m', 'blh', 'ws', 'wd', 'date_unix', 'day_julian', 'weekday', 'hour'],
                       variables_resample=['t2m', 'blh', 'ws', 'wd'], n_samples=5, n_cores=1)
"""


def test_a_script_is_read_without_running_it():
    from normet.gui._script import config_from_calls, read_calls

    calls = read_calls(SCRIPT)
    assert set(calls) == {"build_model", "normalise"}
    assert calls["build_model"]["args"] == [None, "PM2.5"]  # df is a name, not kept
    c = config_from_calls(calls)
    assert c["target"] == "PM2.5" and c["features"] == ["t2m", "blh", "ws", "wd"]
    assert c["backend"] == "lightgbm" and c["lgb_trials"] == 2 and c["train_fraction"] == 0.7
    assert c["norm_vars"] == ["t2m", "blh", "ws", "wd"] and c["norm_samples"] == 5


def test_a_session_script_comes_back_into_the_window(qapp, home, tmp_path):
    import time

    from normet.gui.main_window import MainWindow
    from normet.gui.methods import methods_paragraph

    path = tmp_path / "session.py"
    path.write_text(SCRIPT, encoding="utf-8")
    w = MainWindow()
    try:
        w.load_session_script(str(path), ask=False)
        t0 = time.time()
        while w.runner.busy and time.time() - t0 < 300:
            qapp.processEvents()
            time.sleep(0.02)
        for _ in range(20):
            qapp.processEvents()
        assert w.model is not None and "normalise" in w.results
        assert w.target_combo.currentText() == "PM2.5"
        assert w.backend_combo.currentText() == "lightgbm" and w.norm_samples.value() == 5
        assert w.notices.get("restored") is not None
        text = methods_paragraph(w.methods_info(), "en")
        assert "random search of 2 trials" in text and "mean of 5 predictions" in text
    finally:
        w.close()


def test_both_decompositions_come_back_with_the_session(qapp, home, tmp_path):
    """Emission and meteorology both run on the model: a saved session brings both back,
    the one run last shown, the other a Method away."""
    import joblib
    import numpy as np
    import pandas as pd
    from PySide6.QtWidgets import QComboBox

    from normet.gui.main_window import MainWindow

    w = _trained_window()
    dates = w.df_prep["date"]
    rng = np.random.default_rng(0)
    emi = pd.DataFrame({"date": dates, "observed": rng.normal(30, 5, len(dates))})
    for c in ("emi_total", "emi_base", "date_unix", "day_julian", "weekday", "hour", "emi_noise"):
        emi[c] = rng.normal(0, 1, len(dates))
    met = emi[["date", "observed"]].assign(
        met_total=rng.normal(0, 3, len(dates)), met_base=0.0, ws=0.5, met_noise=0.1
    )
    w._decompose_done(emi)
    w._decompose_done(met)
    path = tmp_path / "s.normet"
    joblib.dump(w.session_bundle(), path)
    w.close()

    v = MainWindow()
    try:
        v.load_results(str(path))
        assert set(v._decom_views) == {"emission", "meteorology"}
        assert "met_total" in v.results["decompose"].columns
        assert v.tab_decom.findChild(QComboBox).currentText() == "meteorology"
    finally:
        v.close()
