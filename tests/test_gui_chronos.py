"""The GUI's Chronos-2 features: decomposition, fine-tuning, donor similarity and
the Assistant's reading of the model's check (offscreen; the checkpoint itself is
stood in for -- what is tested is the window's wiring, the library's own tests
cover the model)."""

from __future__ import annotations

import os
import time

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


def _wait(qapp, runner, timeout=60):
    t0 = time.time()
    while runner.busy and time.time() - t0 < timeout:
        qapp.processEvents()
        time.sleep(0.01)
    for _ in range(20):
        qapp.processEvents()


class _Estimator:
    """Stands in for Chronos2Estimator where the window only passes it on."""

    model_name = "amazon/chronos-2"
    device = "cpu"
    context_length = 64


@pytest.fixture
def chronos_win(qapp):
    from normet.gui.main_window import MainWindow

    w = MainWindow()
    w.load_example()
    w.backend_combo.setCurrentText("chronos-2")
    w.df_prep = w.df_raw.rename(columns={"PM2.5": "value"})
    w.model = _Estimator()
    w.trained_features = ["t2m", "blh", "u10", "v10"]
    w._sync_enabled()
    yield w
    w.close()


def test_parse_groups():
    from normet.gui.main_window import parse_groups

    assert parse_groups("  ", ["a"]) is None
    assert parse_groups("thermal: t2m, blh; wind: u10,v10", ["t2m", "blh", "u10", "v10"]) == {
        "thermal": ["t2m", "blh"],
        "wind": ["u10", "v10"],
    }
    with pytest.raises(ValueError, match="name: var"):
        parse_groups("t2m, blh", ["t2m", "blh"])
    with pytest.raises(ValueError, match="nope not among"):
        parse_groups("x: nope", ["t2m"])


def test_step3_on_chronos(chronos_win, qapp, monkeypatch):
    import normet

    w = chronos_win
    assert w.decom_method.currentText() == "meteorology"
    assert not w.decom_method.model().item(0).isEnabled()  # emission: not on chronos-2
    assert w.decom_samples.value() == 8  # a forward pass per sample, not a tree call
    assert w.decom_button.isEnabled()
    assert not w.roll_button.isEnabled() and not w.pdp_button.isEnabled()

    seen = {}

    def fake_decompose(df, model, **kw):
        seen.update(kw, df=df, model=model)
        return pd.DataFrame(
            {
                "observed": [1.0, 2.0],
                "met_total": [0.1, 0.2],
                "thermal": [0.05, 0.1],
                "wind": [0.05, 0.1],
            },
            index=pd.date_range("2021", periods=2, name="date"),
        )

    monkeypatch.setattr(normet, "decompose", fake_decompose)
    w.decom_groups.setText("thermal: t2m, blh; wind: u10, v10")
    w._run_decompose()
    _wait(qapp, w.runner)
    assert (
        seen["backend"] == "chronos-2"
        and seen["target"] == "value"
        and seen["method"] == "meteorology"
    )
    assert seen["covariates"] == [
        "t2m",
        "blh",
        "u10",
        "v10",
    ]  # no time variables: it cannot use them
    assert seen["groups"] == {"thermal": ["t2m", "blh"], "wind": ["u10", "v10"]}
    assert "attribution" not in seen  # auto: the library's own default
    assert seen["model"] is w.model and "decompose" in w.results
    assert (
        "2 groups (shapley attribution)"
        in w.tab_decom.findChildren(__import__("PySide6.QtWidgets").QtWidgets.QLabel)[0].text()
    )

    # A bad group is said before anything runs.
    w.decom_groups.setText("thermal: t2m, rain")
    monkeypatch.setattr(
        "normet.gui.main_window.QMessageBox.warning", lambda *a: seen.update(warned=a[2])
    )
    w._run_decompose()
    assert "rain not among" in seen["warned"]

    w.backend_combo.setCurrentText("lightgbm")
    assert w.decom_method.model().item(0).isEnabled() and w.decom_samples.value() == 300


def test_fine_tune_replaces_the_model(chronos_win, qapp, monkeypatch):
    from normet.gui import _chronos

    w = chronos_win
    assert not w._ft_box.isHidden() and w.ft_button.isEnabled()
    real = _chronos.finetune
    tuned = _Estimator()
    calls = []
    monkeypatch.setattr(
        _chronos,
        "finetune",
        lambda est, df, mode, steps: (calls.append((est, mode, steps)), tuned)[1],
    )
    monkeypatch.setattr(
        _chronos,
        "sensitivity",
        lambda *a: {"mean_abs_shift": 1.0, "max_abs_shift": 2.0, "pct_of_prediction": 10.0},
    )
    w.results["normalise"] = pd.DataFrame()
    w.ft_mode.setCurrentText("full")
    w.ft_steps.setValue(7)
    before = w.model
    w._run_finetune()
    _wait(qapp, w.runner)
    assert calls == [(before, "full", 7)]
    assert w.model is tuned and "normalise" not in w.results  # made with the old weights
    assert w.chronos_check is not None and w.chronos_check[0]["pct_of_prediction"] == 10.0
    # (the stand-in is not the GUI's helper, so the log is checked against the real one)
    from normet.gui import commandlog

    entry = commandlog.task(
        real, (before, w.df_prep, "full", 7), {}, [(before, "model"), (w.df_prep, "df_prep")]
    )
    assert (
        entry.title == "Fine-tune Chronos-2"
        and entry.code == "model = finetune(model, df_prep, 'full', 7)"
    )
    w.backend_combo.setCurrentText("flaml")
    assert w._ft_box.isHidden()


def test_finetune_helper_writes_its_checkpoint_to_a_folder_it_removes():
    from normet.gui import _chronos

    seen = {}

    class Est:
        def finetune(self, df, target, *, mode, num_steps, output_dir):
            seen.update(
                dir=output_dir,
                exists=os.path.isdir(output_dir),
                target=target,
                index=type(df.index),
            )
            return "tuned"

    df = pd.DataFrame(
        {"date": pd.date_range("2021", periods=3, freq="h"), "value": [1.0, 2.0, 3.0]}
    )
    assert _chronos.finetune(Est(), df, "lora", 3) == "tuned"
    assert seen["exists"] and not os.path.exists(seen["dir"])
    assert seen["target"] == "value" and seen["index"] is pd.DatetimeIndex


def test_donor_similarity_reads_the_pre_period_alone(monkeypatch):
    import normet
    from normet.gui import _chronos

    dates = pd.date_range("2020-01-05", periods=20, freq="W")
    df = pd.concat(
        [
            pd.DataFrame({"date": dates, "ID": u, "y": np.arange(20.0) * k})
            for u, k in (("T", 1), ("A", 1), ("B", -1))
        ]
    )
    seen = {}

    def fake_embed(frame, site_col, target, *, date_col, context_length, device):
        seen.update(last=frame[date_col].max(), context=context_length)
        return {"T": np.array([1.0, 0.0]), "A": np.array([0.9, 0.1]), "B": np.array([-1.0, 0.2])}

    monkeypatch.setattr(normet, "embed_multisite", fake_embed)
    out = _chronos.donor_similarity(df, "date", "ID", "y", "T", "2020-03-15")
    assert seen["last"] < pd.Timestamp("2020-03-15") and seen["context"] == 10
    assert list(out["ID"]) == ["A", "B"] and out["similarity"].iloc[0] > 0.99
    with pytest.raises(ValueError, match="too short"):
        _chronos.donor_similarity(df, "date", "ID", "y", "T", "2020-01-20")
    with pytest.raises(ValueError, match="no data before"):
        _chronos.donor_similarity(df, "date", "ID", "y", "Z", "2020-03-15")


def test_scm_studio_ticks_the_most_similar(qapp, monkeypatch):
    from PySide6.QtCore import Qt

    from normet.gui import _chronos
    from normet.gui.main_window import MainWindow

    w = MainWindow()
    try:
        w.open_scm_example()
        s = w._scm_window
        units = [s.donor_list.item(i).text() for i in range(s.donor_list.count())]
        ranked = pd.DataFrame({"ID": units[::-1], "similarity": np.linspace(1, 0, len(units))})
        monkeypatch.setattr(_chronos, "donor_similarity", lambda *a, **k: ranked)
        s._chronos_ok = True
        s._sync_enabled()
        assert not s.sim_tick_btn.isEnabled()  # nothing ranked yet
        s._run_similarity()
        _wait(qapp, s.runner)
        assert s.tabs.currentWidget() is s.tab_sim and s.sim_tick_btn.isEnabled()
        s.sim_keep.setValue(3)
        s._tick_most_similar()
        ticked = {
            s.donor_list.item(i).text()
            for i in range(s.donor_list.count())
            if s.donor_list.item(i).checkState() == Qt.CheckState.Checked
        }
        assert ticked == set(units[::-1][:3])
        s.close()
    finally:
        w.close()


def test_the_assistant_reads_the_chronos_check(chronos_win):
    from normet.gui.assistant import chronos_reading

    w = chronos_win
    w.chronos_check = None
    assert "could not be run" in chronos_reading(w)
    w.chronos_check = (
        {"mean_abs_shift": 1.9, "max_abs_shift": 9.46, "pct_of_prediction": 12.7},
        ("ok", "Meteorology drives it."),
    )
    text = w.assistant.tools.model_fit()
    assert "zero-shot" in text and "12.7" in text and "verdict (ok): Meteorology drives it." in text
    with pytest.raises(ValueError, match="not available on chronos-2"):
        w.assistant.tools.decompose("emission")
