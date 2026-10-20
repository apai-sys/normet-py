"""File -> Methods Paragraph: written from what ran, in the window's language."""

from __future__ import annotations

import os

import pytest

from normet.gui.methods import methods_paragraph

INFO = {
    "version": "1.0.0",
    "rows": 5793,
    "start": "2020-01-01",
    "end": "2020-08-31",
    "step": "hourly",
    "step_zh": "逐时",
    "target": "NO2",
    "test": {"r2": 0.883, "rmse": 8.655, "n": 1448},
    "ran": {
        "build_model": {
            "backend": "flaml",
            "covariates": ["ws", "wd", "temp", "date_unix", "day_julian", "weekday", "hour"],
            "split_method": "random",
            "train_fraction": 0.75,
            "model_config": {"time_budget": 30, "estimator_list": ["lgbm"]},
            "seed": 7654321,
            "args": [None, "NO2"],
        },
        "normalise": {
            "variables_resample": ["ws", "wd", "temp"],
            "n_samples": 300,
            "return_quantiles": [0.05, 0.25, 0.75, 0.95],
            "args": [None, None],
        },
        "decompose": {
            "method": "meteorology",
            "n_samples": 100,
            "groups": {"wind": ["ws", "wd"], "thermo": ["temp"]},
            "args": [None, None],
        },
    },
}


def test_nothing_is_said_before_a_model():
    assert methods_paragraph({"ran": {}}) is None


def test_the_paragraph_says_what_ran_in_its_own_numbers():
    text = methods_paragraph(INFO, "en")
    assert "5,793 hourly observations of NO2, from 2020-01-01 to 2020-08-31" in text
    assert "FLAML (Wang et al., 2021) over lgbm within 30 s" in text
    assert "3 meteorological predictors (ws, wd and temp)" in text
    assert "75% of the rows, drawn at random (seed 7654321)" in text
    assert "R² = 0.88 and RMSE = 8.65 (n = 1,448)" in text
    assert "mean of 300 predictions" in text and "percentiles" in text
    assert "Shapley values among groups: wind (ws and wd) and thermo (temp)" in text
    assert "rolling" not in text.lower()  # not run, not said
    refs = text.split("References")[1]
    assert "FLAML" in refs and "Grange" in refs and "LightGBM" not in refs


def test_the_paragraph_in_chinese():
    text = methods_paragraph(INFO, "zh")
    assert "5,793 个逐时观测值" in text and "FLAML（Wang 等，2021）" in text
    assert "R² = 0.88" in text and "Shapley 值归因" in text and "参考文献" in text


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_the_window_keeps_what_ran(qapp):
    import normet as nm
    from normet.gui.main_window import MainWindow

    w = MainWindow()
    try:
        w.show_methods_paragraph()
        assert w.notices.get("need-model") is not None  # nothing ran yet
        w.load_example()
        target, feats = w.target_combo.currentText(), w._selected_features()
        kwargs = dict(
            backend="lightgbm",
            covariates=feats + ["date_unix", "day_julian", "weekday", "hour"],
            model_config={"n_trials": 2},
            split_method="random",
            train_fraction=0.75,
            seed=1,
        )
        commit = w._record_task(nm.build_model, (w.df_raw, target), kwargs)
        result = nm.build_model(w.df_raw, target, **kwargs)
        commit()
        w._pending_features = feats
        w._train_done(result)
        assert w._ran["build_model"]["model_config"] == {"n_trials": 2}
        assert w._ran["build_model"]["args"][0] is None  # the table is not kept
        text = methods_paragraph(w.methods_info(), "en")
        assert "random search of 2 trials" in text and "R² =" in text
        w.show_methods_paragraph()
        assert (
            "LightGBM"
            in w._methods_dialog.findChild(
                __import__("PySide6.QtWidgets").QtWidgets.QPlainTextEdit
            ).toPlainText()
        )
        w._methods_dialog.close()
    finally:
        w.close()


def test_the_manual_in_the_window_language_and_the_shortcuts(qapp, monkeypatch):
    from normet.gui import manual
    from normet.gui.main_window import MainWindow

    assert manual.manual_name("zh") == "normet_manual_zh.pdf"
    assert manual.manual_name("en") == "normet_manual.pdf"
    local = manual.manual_location("zh")
    assert local.isLocalFile() and local.toLocalFile().endswith("normet_manual_zh.pdf")
    monkeypatch.setattr(manual, "manual_path", lambda lang: None)
    assert manual.manual_location("en").toString().startswith("https://github.com/")

    w = MainWindow()
    try:
        w.show_shortcuts()
        from PySide6.QtWidgets import QTableWidget

        table = w._shortcuts_dialog.findChild(QTableWidget)
        commands = {table.item(i, 1).text() for i in range(table.rowCount())}
        assert {"Open Data…", "Save Results…", "User Manual", "Command Palette…"} <= commands
        w._shortcuts_dialog.close()
    finally:
        w.close()
