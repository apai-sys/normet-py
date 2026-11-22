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
    assert not w.roll_button.isEnabled() and w.pdp_button.isEnabled()

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
    labels = w.tab_decom.findChildren(__import__("PySide6.QtWidgets").QtWidgets.QLabel)
    assert any("2 groups (shapley attribution)" in lbl.text() for lbl in labels)

    # A bad group is said before anything runs.
    w.decom_groups.setText("thermal: t2m, rain")
    monkeypatch.setattr(
        "normet.gui._run_analyse.QMessageBox.warning", lambda *a: seen.update(warned=a[2])
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


def test_chronos_gui_helpers_icn_and_save_load(monkeypatch, tmp_path):
    import normet.foundation.estimator as est_mod
    from normet.foundation import Chronos2Estimator
    from normet.gui import _chronos

    levels = [0.1, 0.5, 0.9]

    class _FakePipe:
        quantiles = levels

        def predict(self, inputs, prediction_length, **kwargs):
            out = []
            for item in inputs:
                hist = np.asarray(item["target"], dtype=float)
                med = np.full(prediction_length, float(np.nanmedian(hist)))
                out.append(np.stack([med - 0.5, med, med + 0.5])[None, ...])
            return out

        def save_pretrained(self, d):
            from pathlib import Path

            p = Path(d)
            p.mkdir(parents=True, exist_ok=True)
            (p / "config.json").write_text('{"architectures": ["Chronos2Model"]}', encoding="utf-8")

    est = Chronos2Estimator(context_length=24, prediction_length=12, device="cpu")
    est._pipeline = _FakePipe()
    est._quantiles = np.asarray(levels, dtype=float)

    df_prep = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=72, freq="h"),
            "value": np.linspace(10.0, 20.0, 72),
            "ws": np.linspace(1.0, 5.0, 72),
        }
    )
    out = _chronos.deweather(
        est, df_prep, ["ws"], with_quantiles=True, n_samples=2, method="icn", estimate="paired"
    )
    assert out.index.name == "date"
    assert {"observed", "normalised", "q100", "q500", "q900"} <= set(out.columns)

    saved = _chronos.save_estimator(est, tmp_path / "gui-ckpt")
    assert (saved / "normet_estimator.json").exists()
    monkeypatch.setattr(est_mod, "resolve_device", lambda req=None: req or "cpu")
    monkeypatch.setattr(est_mod, "_load_pipeline_on", lambda name, dev, auto: (est._pipeline, dev))
    reloaded = _chronos.load_saved_estimator(saved, device="cpu")
    assert reloaded.context_length == 24


def test_chronos_gui_pdp_normalise_icn_checkpoint_and_scm_backend(
    chronos_win, qapp, monkeypatch, tmp_path
):
    """Verify Phase 5 GUI wiring: PDP on chronos-2, Normalise ICN options,
    checkpoint save/load buttons, and SCM Studio chronos backend."""
    import normet
    from normet.gui import _chronos
    from normet.gui._lists import _fill_checklist

    w = chronos_win

    # 1. PDP on chronos-2 filters out TIME_VARS and requests native anchors
    _fill_checklist(w.pdp_vars, ["t2m", "blh", "hour"], {"t2m", "hour"})
    _fill_checklist(w.norm_vars, ["t2m", "blh", "u10", "v10", "hour"], {"t2m", "blh", "u10", "v10"})
    seen_pdp = {}

    def _fake_pdp(df_prep, model, *, var_list, grid_points, target="value", n_anchors=6, **kw):
        seen_pdp.update(
            var_list=var_list, grid_points=grid_points, target=target, n_anchors=n_anchors
        )
        return pd.DataFrame(
            {
                "variable": ["t2m"] * 3,
                "value": [0.0, 5.0, 10.0],
                "pdp_mean": [12.0, 10.0, 9.0],
                "pdp_lower": [10.0, 8.0, 7.0],
                "pdp_upper": [14.0, 12.0, 11.0],
            }
        )

    monkeypatch.setattr(normet, "pdp", _fake_pdp)
    w._run_pdp()
    _wait(qapp, w.runner)
    assert seen_pdp["var_list"] == ["t2m"]
    assert seen_pdp["target"] == "value" and seen_pdp["n_anchors"] == 6
    assert "pdp" in w.results

    # 2. Normalise ICN controls: hidden on flaml, visible on chronos-2, forwards options
    assert not w._chronos_norm_box.isHidden()
    assert w.chronos_norm_method.currentText() == "deweather"
    assert not w.chronos_norm_estimate.isEnabled()
    w.chronos_norm_method.setCurrentText("icn")
    assert w.chronos_norm_estimate.isEnabled() and w.chronos_norm_log.isEnabled()
    w.chronos_norm_estimate.setCurrentText("paired")
    w.chronos_norm_log.setChecked(True)

    seen_norm = {}

    def _fake_dw(est, df_prep, met, with_q, n_samp, **kw):
        seen_norm.update(met=met, with_q=with_q, n_samp=n_samp, **kw)
        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        return pd.DataFrame({"observed": np.ones(10), "normalised": np.ones(10) * 2.0}, index=idx)

    monkeypatch.setattr(_chronos, "deweather", _fake_dw)
    w._run_normalise()
    _wait(qapp, w.runner)
    assert seen_norm["method"] == "icn"
    assert seen_norm["estimate"] == "paired"
    assert seen_norm["log_transform"] is True

    # Reset back to default deweather, switch backend away & back -> box toggles cleanly
    w.chronos_norm_method.setCurrentText("deweather")
    w.backend_combo.setCurrentText("flaml")
    assert w._chronos_norm_box.isHidden()
    w.backend_combo.setCurrentText("chronos-2")
    assert not w._chronos_norm_box.isHidden()

    # 3. Checkpoint save & load buttons inside _ft_box
    assert w.ft_save_button.isEnabled() and w.ft_load_button.isEnabled()
    save_dir = tmp_path / "gui_ckpt"
    monkeypatch.setattr(_chronos, "save_estimator", lambda est, p: save_dir)
    w._save_chronos_checkpoint(str(save_dir))
    _wait(qapp, w.runner)
    assert str(save_dir) in w.statusBar().currentMessage()

    reloaded = _Estimator()
    monkeypatch.setattr(
        _chronos,
        "load_saved_session",
        lambda df_raw, tgt, feats, folder, dev: (w.df_prep, reloaded),
    )
    monkeypatch.setattr(
        _chronos,
        "sensitivity",
        lambda *a: {"mean_abs_shift": 1.0, "max_abs_shift": 2.0, "pct_of_prediction": 9.0},
    )
    w._load_chronos_checkpoint(str(save_dir))
    _wait(qapp, w.runner)
    assert w.model is reloaded

    # 4. SCM Studio `chronos` backend option & fit dispatch
    w.open_scm_example()
    s = w._scm_window
    assert s.backend_combo.findText("chronos") >= 0
    monkeypatch.setattr(_chronos, "chronos_availability", lambda: (False, "missing foundation"))
    # Re-create an SCMWindow under chronos unavailable -> item must be disabled
    from normet.gui.scm_window import SCMWindow

    s_disabled = SCMWindow()
    try:
        idx_c = s_disabled.backend_combo.findText("chronos")
        assert idx_c >= 0 and not s_disabled.backend_combo.model().item(idx_c).isEnabled()
    finally:
        s_disabled.close()

    import normet.causal as causal_mod

    seen_scm = {}

    def _fake_run_scm(*a, **kw):
        seen_scm.update(kw)
        dates = pd.date_range("2015-01-04", periods=30, freq="W")
        return pd.DataFrame(
            {
                "observed": np.ones(30) * 10.0,
                "synthetic": np.ones(30) * 10.0,
                "effect": np.zeros(30),
            },
            index=dates,
        )

    monkeypatch.setattr(causal_mod, "run_scm", _fake_run_scm)
    s.backend_combo.setCurrentText("chronos")
    s._run_fit()
    _wait(qapp, s.runner)
    assert seen_scm.get("scm_backend") == "chronos"
    s.close()
