"""Offscreen smoke tests for the Qt GUI (skipped when PySide6 is absent).

These deliberately avoid model training / SCM fits — they only exercise
window construction, example-data ingestion and the selector logic, so the
suite stays fast and dependency-light.
"""

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


def test_main_window_example_flow(qapp):
    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        win.load_example()
        assert win.df_raw is not None and len(win.df_raw) > 1000
        assert win.target_combo.currentText() == "PM2.5"
        feats = win._selected_features()
        assert "t2m" in feats and "blh" in feats
        assert win.target_combo.currentText() not in feats
        # gating: analysis steps need a model, training only needs data
        assert win.train_button.isEnabled()
        assert not win.norm_button.isEnabled()
        # config round-trip
        cfg = win.get_config()
        win.apply_config(cfg)
        assert win.get_config() == cfg
    finally:
        win.close()


def test_train_backend_budget_row(qapp):
    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        assert win._budget_label.text() == "Time budget"
        # flaml shows the estimator sub-selection, default lgbm only
        assert not win.estimator_list.isHidden()
        assert win._selected_estimators() == ["lgbm"]
        win.backend_combo.setCurrentText("lightgbm")
        assert win._budget_label.text() == "Search trials"
        assert win.estimator_list.isHidden()  # not applicable to LightGBM
        win.budget_spin.setValue(7)
        win.backend_combo.setCurrentText("flaml")
        assert win.budget_spin.value() == 60  # flaml default remembered
        assert not win.estimator_list.isHidden()
        win.backend_combo.setCurrentText("lightgbm")
        assert win.budget_spin.value() == 7  # lightgbm value remembered
    finally:
        win.close()


def test_data_window_construction(qapp):
    from normet.gui.data_window import DEFAULT_POLLUTANTS, DataWindow, _site_name

    win = DataWindow()
    try:
        ticked = win._checked_pollutants()
        assert set(ticked) == DEFAULT_POLLUTANTS
        # no station chosen yet → fetch disabled, find enabled
        assert win.find_btn.isEnabled()
        assert not win.fetch_btn.isEnabled()
        assert not win.save_btn.isEnabled()
        assert _site_name("Manchester Piccadilly-Nitrogen dioxide (air)") == (
            "Manchester Piccadilly"
        )
        assert _site_name("Stoke-on-Trent Centre-PM10") == "Stoke-on-Trent Centre"
    finally:
        win.close()


def test_find_stations_includes_aurn_code_column(qapp, monkeypatch):
    """Each station row carries its official AURN site code (e.g. 'MAN3'),
    looked up by name; unmatched sites get a blank code, not a crash."""
    from normet.gui.data_window import _find_stations
    from normet.io import defra

    def fake_timeseries(url, params):
        assert url == f"{defra._API_BASE}/timeseries"
        return [
            {
                "id": "ts-1",
                "station": {
                    "properties": {"id": 1, "label": "Manchester Piccadilly-PM2.5"},
                    "geometry": {"coordinates": [53.48, -2.24]},
                },
                "firstValue": {"timestamp": 1704067200000},
                "lastValue": {"timestamp": 1706745600000},
            },
            {
                "id": "ts-2",
                "station": {
                    "properties": {"id": 2, "label": "Some Unlisted Site-PM2.5"},
                    "geometry": {"coordinates": [51.0, -1.0]},
                },
                "firstValue": {"timestamp": 1704067200000},
                "lastValue": {"timestamp": 1706745600000},
            },
        ]

    monkeypatch.setattr(defra, "_request", fake_timeseries)
    monkeypatch.setattr(defra, "fetch_aurn_site_codes", lambda: {"Manchester Piccadilly": "MAN3"})

    df = _find_stations(["PM2.5"])
    assert set(df["code"]) == {"MAN3", ""}
    row = df[df["site"] == "Manchester Piccadilly"].iloc[0]
    assert row["code"] == "MAN3"
    row2 = df[df["site"] == "Some Unlisted Site"].iloc[0]
    assert row2["code"] == ""


def test_train_includes_time_features(qapp):
    """The model must be trained on met + time features; resampling only met.

    Regression test: without the time features the model is a pure f(met)
    and the normalised series collapses to a flat mean.
    """
    from normet.gui.main_window import TIME_VARS, MainWindow

    win = MainWindow()
    try:
        win.load_example()
        captured = {}

        def fake_submit(name, fn, on_result, on_error, *args, **kwargs):
            captured.update(kwargs)
            return True

        win.runner.submit = fake_submit
        win._run_train()
        feats = captured["covariates"]
        assert all(tv in feats for tv in TIME_VARS)
        assert "t2m" in feats
        # the resample-candidate list must NOT offer the time features
        assert set(win._pending_features).isdisjoint(TIME_VARS)
    finally:
        win.close()


def test_norm_vars_selector(qapp):
    from normet.gui.main_window import MainWindow, _fill_checklist

    win = MainWindow()
    try:
        # simulate the post-training fill: met subset checked by default
        feats = ["t2m", "ws", "traffic_count"]
        _fill_checklist(win.norm_vars, feats, {"t2m", "ws"})
        assert win._selected_norm_vars() == ["t2m", "ws"]
        win._set_norm_vars_met_only()
        assert win._selected_norm_vars() == ["t2m", "ws"]  # traffic_count is not met
    finally:
        win.close()


def test_pdp_toggle_buttons(qapp):
    """'Time variables' / 'Met only' in Step 5: click ticks the matching
    subset, click again unticks it; the two subsets don't interfere."""
    from normet.gui.main_window import TIME_VARS, MainWindow, _checked_items, _fill_checklist

    win = MainWindow()
    try:
        # Simulate the post-Step-1 fill without a real training run.
        feats = ["t2m", "ws", "traffic_count"] + list(TIME_VARS)
        _fill_checklist(win.pdp_vars, feats, set())
        assert not win.pdp_time_btn.isChecked()
        assert not win.pdp_met_btn.isChecked()

        win.pdp_time_btn.setChecked(True)
        assert set(_checked_items(win.pdp_vars)) == set(TIME_VARS)
        win.pdp_time_btn.setChecked(False)
        assert _checked_items(win.pdp_vars) == []

        win.pdp_met_btn.setChecked(True)
        assert set(_checked_items(win.pdp_vars)) == {"t2m", "ws"}
        win.pdp_met_btn.setChecked(False)
        assert _checked_items(win.pdp_vars) == []

        # Non-overlapping subsets: both pressed together is additive.
        win.pdp_time_btn.setChecked(True)
        win.pdp_met_btn.setChecked(True)
        assert set(_checked_items(win.pdp_vars)) == {"t2m", "ws", *TIME_VARS}

        # Refilling the list (Step 1 re-run) resets both toggles.
        _fill_checklist(win.pdp_vars, feats, set())
        win._reset_pdp_toggle_buttons()
        assert not win.pdp_time_btn.isChecked()
        assert not win.pdp_met_btn.isChecked()
    finally:
        win.close()


def test_multiscale_button_gating_and_tab_mapping(qapp):
    """The Multi-scale button needs both a trained model AND Step 2's Y_inf;
    the Multi-scale tab must land at the 'multiscale' results key."""
    import pandas as pd

    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        win.load_example()
        assert not win.ms_button.isEnabled()  # no model yet

        win.model = object()  # simulate a trained model without a real run
        win.df_prep = win.df_raw
        win._sync_enabled()
        assert not win.ms_button.isEnabled(), "needs Step 2's Y_inf too"

        win.results["normalise"] = pd.DataFrame({"normalised": [1.0, 2.0]})
        win._sync_enabled()
        assert win.ms_button.isEnabled()

        assert win.tabs.tabText(5) == "Multi-scale"
        assert win._TAB_KEYS[5] == "multiscale"
        assert win.tabs.tabText(6) == "⑥ PDP"
    finally:
        win.close()


def test_rolling_config_round_trip_defaults(qapp):
    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        cfg = win.get_config()
        assert cfg["ms_fast"] == 14
        assert cfg["ms_meso"] == 90
        assert cfg["ms_slow"] == 365
        win.ms_fast.setValue(21)
        win.apply_config({"ms_fast": 30, "ms_meso": 120, "ms_slow": 400})
        assert win.ms_fast.value() == 30
        assert win.ms_meso.value() == 120
        assert win.ms_slow.value() == 400
    finally:
        win.close()


def test_scm_window_example_design(qapp):
    from normet.gui.scm_window import SCMWindow

    win = SCMWindow()
    try:
        win.load_example()
        assert win.df is not None
        assert win.treated_combo.currentText() == "2+26 cities"
        assert win.outcome_combo.currentText() == "SO2wn"
        assert win.cutoff_edit.date().toString("yyyy-MM-dd") == "2015-10-23"
        # treated unit is excluded from the donor pool
        donors = [win.donor_list.item(i).text() for i in range(win.donor_list.count())]
        assert "2+26 cities" not in donors
        assert len(donors) == 30
        design = win._design()
        assert design is not None
        assert design["treated_unit"] == "2+26 cities"
        assert len(design["donors"]) == 30
        # bayesian backend disables run_scm-based inference
        win.backend_combo.setCurrentText("bayesian")
        assert not win.placebo_space_btn.isEnabled()
        win.backend_combo.setCurrentText("scm")
        assert win.placebo_space_btn.isEnabled()
    finally:
        win.close()
