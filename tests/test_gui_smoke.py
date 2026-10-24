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


@pytest.fixture(autouse=True)
def _no_map_tiles(monkeypatch):
    """No GUI test may reach the OpenStreetMap tile server.

    The layer has its own tests (``test_basemap.py``); what these exercise is the
    window's wiring, which must not depend on the network being up -- the map
    falls back to its plain grid exactly as it does offline.
    """
    from normet.gui import _basemap

    monkeypatch.setattr(_basemap, "tile_mosaic", lambda *args, **kwargs: None)


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


def test_a_csv_of_zoned_dates_with_one_that_does_not_parse_loads(qapp, tmp_path):
    # Dates in ...Z parse tz-aware; one that did not parse is NaT, and together
    # they stopped the data preview -- and with it the whole load.
    from normet.gui.main_window import MainWindow

    path = tmp_path / "zoned.csv"
    path.write_text(
        "date,PM2.5,t2m\n"
        "2020-01-01T00:00Z,1,5\n2020-01-01T01:00Z,2,5\nnot a date,3,5\n2020-01-01T03:00Z,4,5\n"
    )
    win = MainWindow()
    try:
        win.load_csv(str(path))
        # put on the window's clock (UTC by default), as times without a zone of their own
        assert win.df_raw["date"].dt.tz is None and win.df_raw["date"].isna().sum() == 1
        (line,) = win.data_preview.canvas.figure.axes[0].get_lines()
        assert list(line.get_ydata()) == [1, 2, 4]  # the row without a date left out
        assert win.command_log.entries[-1].title == "Open data"  # the load went through
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
    from normet.gui.data_window import (
        DEFAULT_POLLUTANTS,
        EEA_NETWORK,
        NETWORKS,
        OPENAQ_NETWORK,
        DataWindow,
    )

    win = DataWindow()
    try:
        ticked = win._checked_pollutants()
        assert set(ticked) == DEFAULT_POLLUTANTS
        # no station chosen yet → fetch disabled, find enabled
        assert win.find_btn.isEnabled()
        assert not win.fetch_btn.isEnabled()
        assert not win.save_btn.isEnabled()
        # Network picker defaults to AURN and offers the six UK networks
        # plus the two non-UK families.
        assert win._provider() == ("uk", "aurn", "")
        assert win.net_combo.count() == len(NETWORKS) + 2
        assert win.net_combo.findText(EEA_NETWORK) >= 0
        assert win.net_combo.findText(OPENAQ_NETWORK) >= 0
        # The UK default needs no context fields.
        assert win.ctx_box.isHidden()
    finally:
        win.close()


def test_data_window_network_change_clears_stations(qapp):
    """A station list belongs to one network, so switching must drop it —
    otherwise Fetch would run a code from network A against network B."""
    import pandas as pd

    from normet.gui.data_window import EEA_NETWORK, OPENAQ_NETWORK, DataWindow

    win = DataWindow()
    try:
        win.stations = pd.DataFrame(
            [
                {
                    "site": "X",
                    "code": "X1",
                    "site_type": "Urban Traffic",
                    "pollutants": "NO2",
                    "n": 1,
                    "from": "2018-01-01",
                    "to": "2022-12-31",
                    "lat": 53.0,
                    "lon": -2.0,
                }
            ]
        )
        win._fill_station_table(win.stations)
        win.net_combo.setCurrentIndex(1)
        assert win.stations is None
        assert win.station_table.rowCount() == 0
        assert win._provider()[1] != "aurn"
        # The EEA row swaps in its source fields; UK rows hide them.
        # isHidden() tracks the explicit setVisible calls; isVisible() would
        # also need the window shown, which offscreen tests avoid.
        win.net_combo.setCurrentText(EEA_NETWORK)
        assert win._provider()[:1] == ("eea",)
        assert not win.country_edit.isHidden()
        assert win.openaq_key.isHidden()
        win.net_combo.setCurrentText(OPENAQ_NETWORK)
        assert not win.openaq_key.isHidden()
        win.net_combo.setCurrentIndex(0)
        assert win.ctx_box.isHidden()
    finally:
        win.close()


def test_find_stations_aggregates_metadata_per_station(monkeypatch):
    """One row per station, carrying its code, site type and the archive
    coverage window.

    Replaces an older test that exercised the DEFRA SOS path. That backend
    stopped responding (2026-07-26) and the window now reads the openair
    .RData archives through normet.io.ukaq, so a station's identity comes
    from the network metadata rather than from parsing a timeseries label.
    """
    import pandas as pd

    from normet.gui import data_window

    meta = pd.DataFrame(
        {
            "code": ["MAN3", "MAN3", "GLAZ", "OLD1"],
            "site": ["Manchester Piccadilly", "Manchester Piccadilly", "Glazebury", "Closed Site"],
            "site_type": [
                "Urban Background",
                "Urban Background",
                "Rural Background",
                "Urban Traffic",
            ],
            "variable": ["PM2.5", "NO2", "PM2.5", "PM2.5"],
            "latitude": [53.48, 53.48, 53.46, 51.0],
            "longitude": [-2.24, -2.24, -2.47, -1.0],
            "start_date": ["1995-12-18", "1995-12-18", "2004-01-26", "2001-01-01"],
            # A blank end_date means the station is still open; a real one
            # means it closed and must NOT be reported as running to today.
            "end_date": [None, None, None, "2015-06-30"],
        }
    )
    # _find_stations does `from normet import list_ukaq_stations` at call
    # time, so the attribute on the normet package is the patch target.
    # No qapp fixture here on purpose: this is a plain function, and gating
    # it on Qt would have hidden it behind this environment's broken PySide6
    # (libQt6Core Qt_6.6_PRIVATE_API) like the rest of this file.
    monkeypatch.setattr("normet.list_ukaq_stations", lambda *a, **k: meta)

    df = data_window._find_stations(data_window.PROV_UK, "aurn", "", ["PM2.5", "NO2"], None)

    assert set(df["code"]) == {"MAN3", "GLAZ", "OLD1"}
    man3 = df[df["code"] == "MAN3"].iloc[0]
    # Both requested species collapse into one station row.
    assert man3["n"] == 2
    assert man3["pollutants"] == "PM2.5, NO2"
    assert man3["site_type"] == "Urban Background"
    assert man3["from"] == "1995-12-18"
    # Still open -> coverage runs to today, not blank.
    assert man3["to"] == pd.Timestamp.today().strftime("%Y-%m-%d")
    closed = df[df["code"] == "OLD1"].iloc[0]
    assert closed["to"] == "2015-06-30"


def test_find_stations_dispatches_per_provider(monkeypatch):
    """EEA and OpenAQ finds hand their own arguments to the right adapter and
    return the shared picker columns — the fetch then reads codes off them."""
    import pandas as pd

    from normet.gui import data_window

    calls: dict = {}
    eea_df = pd.DataFrame(
        [
            {
                "site": "Berlin",
                "code": "DE0001",
                "site_type": "",
                "country": "DE",
                "pollutants": "PM2.5",
                "n": 1,
                "from": "",
                "to": "",
                "lat": 52.5,
                "lon": 13.4,
                "links": '{"PM2.5": {"url": "https://x/DE_1.parquet", "from": 2013, "to": 2024}}',
            }
        ]
    )
    monkeypatch.setattr(
        "normet.io.eea.list_eea_stations", lambda **kw: calls.update(eea=kw) or eea_df
    )
    locs = pd.DataFrame(
        [
            {
                "id": 77,
                "name": "Seoul Station",
                "provider": "KOR",
                "lat": 37.5,
                "lon": 126.9,
                "parameters": ["pm25", "o3"],
                "last_updated": "2026-01-01T00:00:00Z",
            }
        ]
    )
    monkeypatch.setattr(
        "normet.io.openaq.openaq_locations", lambda **kw: calls.update(oaq=kw) or locs
    )

    out = data_window._find_stations(data_window.PROV_EEA, None, "de", ["PM2.5", "NO2"], None)
    assert calls["eea"] == {"country": "de", "pollutants": ["PM2.5", "NO2"]}
    assert out.loc[0, "code"] == "DE0001"
    # The year bounds become full dates so the picker's snap logic is reused.
    assert out.loc[0, "from"] == "2013-01-01" and out.loc[0, "to"] == "2024-12-31"
    assert "links" not in out.columns

    out = data_window._find_stations(data_window.PROV_OPENAQ, None, "KR", ["PM2.5", "NO2"], "k")
    assert calls["oaq"]["country"] == "KR" and calls["oaq"]["api_key"] == "k"
    assert out.loc[0, "pollutants"] == "PM2.5"  # NO2 not measured at this one
    assert out.loc[0, "code"] == "77"
    assert out.loc[0, "to"] == "2026-01-01"


def _four_stations():
    import pandas as pd

    return pd.DataFrame(
        {
            "site": [
                "Manchester Piccadilly",
                "London Marylebone Road",
                "Glasgow Kerbside",
                "Belfast Centre",
            ],
            "code": ["MAN3", "MY1", "GLA4", "BEL2"],
            "site_type": ["Urban Background", "Urban Traffic", "Urban Traffic", "Urban Background"],
            "pollutants": ["PM2.5, NO2", "NO2, O3", "PM2.5", "NO2"],
            "n": [2, 2, 1, 1],
            "from": ["1995-12-18", "1998-01-01", "2000-05-01", "2001-02-01"],
            "to": ["2026-01-01", "2026-01-01", "2025-12-01", "2024-01-01"],
            "lat": [53.4779, 51.5226, 55.8642, 54.5973],
            "lon": [-2.2372, -0.1543, -4.2518, -5.9301],
        }
    )


def _click_event(map_widget, lon, lat):
    """The mouse event a click at (*lon*, *lat*) produces, as a press+release pair."""
    from types import SimpleNamespace

    from normet.gui._basemap import mercator

    x, y = map_widget.ax.transData.transform((lon, float(mercator(lat))))
    return SimpleNamespace(
        x=float(x),
        y=float(y),
        button=1,
        inaxes=map_widget.ax,
        xdata=float(lon),
        ydata=float(mercator(lat)),
        canvas=map_widget.canvas,
    )


def test_station_map_click_selects_the_station(qapp):
    """A click on a station's point is a selection, exactly like clicking its
    row: the table follows, the hint and date range update, Fetch enables."""
    from normet.gui.data_window import DataWindow

    win = DataWindow()
    win.resize(1100, 800)
    win.show()
    try:
        df = _four_stations()
        win.stations = df
        win._fill_station_table(df)
        event = _click_event(win.station_map, df.loc[2, "lon"], df.loc[2, "lat"])
        win.station_map._pressed(event)
        win.station_map._released(event)

        assert win._selected_keys() == [2]
        assert win.station_table.currentRow() == 2
        assert "Glasgow Kerbside" in win.station_hint.text()
        assert win.fetch_btn.isEnabled()
        assert win.station_map._selected == [2]  # the red ring follows the pick

        # A row click rings the same station on the map.
        win.station_table.selectRow(0)
        assert win.station_map._selected == [0]
    finally:
        win.close()


def test_station_map_picks_only_near_a_point(qapp):
    """Only a click within a few pixels of a station is a pick: the map's empty
    water must not select whichever station happens to be nearest."""
    from normet.gui._station_map import PICK_PX, StationMap

    w = StationMap()
    try:
        df = _four_stations()
        w.set_stations(df)
        hit = _click_event(w, df.loc[1, "lon"], df.loc[1, "lat"])
        assert w._nearest(hit) == 1
        away = _click_event(w, df.loc[1, "lon"] - 5.0, df.loc[1, "lat"] + 2.0)
        assert w._nearest(away) is None
        assert PICK_PX < 20
        # Hovering names the station under the pointer.
        w._moved(hit)
        assert w._tip.get_visible() and w._tip.get_text() == "London Marylebone Road (MY1)"
        w._moved(away)
        assert not w._tip.get_visible()
    finally:
        w.deleteLater()


def _event_at(widget, lon, lat):
    from types import SimpleNamespace

    from normet.gui._basemap import mercator

    x, y = widget.ax.transData.transform((lon, float(mercator(lat))))
    return SimpleNamespace(
        x=float(x),
        y=float(y),
        button=1,
        inaxes=widget.ax,
        xdata=float(lon),
        ydata=float(mercator(lat)),
        canvas=widget.canvas,
    )


def test_station_map_box_selects_the_stations_inside(qapp):
    """Dragging a box selects every station in it -- several at once, in the
    map and in the table -- and the selection drives a multi-station fetch."""
    from normet.gui.data_window import DataWindow

    win = DataWindow()
    win.resize(1100, 800)
    win.show()
    try:
        df = _four_stations()
        win.stations = df
        win._fill_station_table(df)
        # A box across the two northern stations (Glasgow 55.86 and Belfast 54.60).
        start = _event_at(win.station_map, -6.5, 54.3)
        end = _event_at(win.station_map, -3.5, 56.2)
        win.station_map._pressed(start)
        win.station_map._moved(end)
        win.station_map._released(end)

        assert win._selected_keys() == [2, 3]
        assert win.station_map._selected == [2, 3]
        assert sorted({i.row() for i in win.station_table.selectedItems()}) == [2, 3]
        assert "2 stations selected" in win.station_hint.text()
        assert win.fetch_btn.isEnabled()

        # A drag that ends on empty space selects nothing.
        empty = _event_at(win.station_map, 5.0, 40.0)
        win.station_map._pressed(empty)
        win.station_map._moved(_event_at(win.station_map, 7.0, 41.0))
        win.station_map._released(_event_at(win.station_map, 7.0, 41.0))
        assert win._selected_keys() == []
    finally:
        win.close()


def test_station_map_adds_to_the_selection_with_command(qapp, monkeypatch):
    """Command/Control-click adds a station, and clicking it again takes it out."""
    from normet.gui.data_window import DataWindow

    win = DataWindow()
    win.resize(1100, 800)
    win.show()
    try:
        df = _four_stations()
        win.stations = df
        win._fill_station_table(df)
        win.station_map._additive = lambda: True  # what holding Command/Control means
        for key in (2, 3, 2):
            event = _event_at(win.station_map, df.loc[key, "lon"], df.loc[key, "lat"])
            win.station_map._pressed(event)
            win.station_map._released(event)
        assert win._selected_keys() == [3]
        assert win.station_map._selected == [3]
    finally:
        win.close()


def test_fetch_many_stacks_stations_and_skips_failures(monkeypatch):
    """Several stations become one panel-shaped table, and one closed station
    does not take the batch down with it."""
    import pandas as pd
    import pytest

    from normet.gui import data_window as dw

    def fake(st, *args):
        if st["code"] == "BAD":
            raise RuntimeError("no data for that range")
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01 01:00", "2020-01-01 00:00"]),
                "site": [st["site"]] * 2,
                "NO2": [2.0, 1.0],
            }
        )

    monkeypatch.setattr(dw, "_fetch_and_merge", fake)
    sites = [
        {"site": "Bravo", "code": "B"},
        {"site": "Alpha", "code": "A"},
        {"site": "Gone", "code": "BAD"},
    ]
    out = dw._fetch_many(
        sites, ["NO2"], "2020-01-01", "2020-01-02", dw.MET_OPEN_METEO, dw.PROV_UK, "aurn", "", None
    )
    assert out["site"].tolist() == ["Alpha", "Alpha", "Bravo", "Bravo"]
    assert out["NO2"].tolist() == [1.0, 2.0, 1.0, 2.0]  # sorted by site, then date

    with pytest.raises(RuntimeError, match="None of the 1 selected stations"):
        dw._fetch_many(
            [{"site": "Gone", "code": "BAD"}],
            ["NO2"],
            "2020-01-01",
            "2020-01-02",
            dw.MET_OPEN_METEO,
            dw.PROV_UK,
            "aurn",
            "",
            None,
        )


def test_station_map_shows_the_filtered_list(qapp):
    """The search narrows the map to the rows the table shows, and a network
    change empties it, so no point can select a station of another network."""
    from normet.gui.data_window import EEA_NETWORK, DataWindow

    win = DataWindow()
    try:
        win.stations = _four_stations()
        win._fill_station_table(win.stations)
        assert len(win.station_map._keys) == 4
        win.search_edit.setText("glasgow")
        assert win.station_map._keys.tolist() == [2]
        assert win.station_map._labels == ["Glasgow Kerbside (GLA4)"]
        win.search_edit.clear()
        win.net_combo.setCurrentText(EEA_NETWORK)
        assert win.station_map._keys.size == 0
    finally:
        win.close()


def test_station_map_draws_the_basemap_when_available(qapp, monkeypatch):
    """With tiles the image sits behind the points at their extent and is
    credited; when they cannot be had the scatter is all there is."""
    import numpy as np

    from normet.gui import _basemap
    from normet.gui.data_window import DataWindow

    image = np.full((512, 768, 3), 240, dtype=np.uint8)
    extent = (0.0, 12.0, 40.0, 60.0)
    monkeypatch.setattr(_basemap, "tile_mosaic", lambda *args, **kwargs: (image, extent))

    win = DataWindow()
    win.resize(1100, 800)
    win.show()
    try:
        win.stations = _four_stations()
        win._fill_station_table(win.stations)
        win.station_map._load_basemap()  # what the deferred call does
        bg = win.station_map._bg
        assert bg.get_visible()
        assert tuple(bg.get_extent()) == extent
        assert win.station_map._credit.get_visible()  # attribution is a condition

        # Offline: the same search, no tiles -- the points stay, the image goes.
        monkeypatch.setattr(_basemap, "tile_mosaic", lambda *args, **kwargs: None)
        win._fill_station_table(win.stations)
        win.station_map._load_basemap()
        assert not bg.get_visible()
        assert not win.station_map._credit.get_visible()
        assert win.station_map._points.get_offsets().shape[0] == 4
    finally:
        win.close()


def test_fetch_and_merge_unknown_provider_raises():
    from normet.gui import data_window

    with pytest.raises(ValueError, match="Unknown provider"):
        data_window._fetch_and_merge(
            {"site": "x", "code": "x", "lat": 0, "lon": 0},
            ["NO2"],
            "2020-01-01",
            "2020-01-31",
            "Open-Meteo (ERA5, no key needed)",
            "nope",
            None,
            "",
            None,
        )


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


def test_all_ticks_every_one_then_unticks_every_one(qapp):
    """One All button does what an All and a None did: it ticks every item, and once
    every one is ticked, unticks every one."""
    from PySide6.QtWidgets import QListWidget

    from normet.gui._lists import _toggle_all
    from normet.gui.main_window import _checked_items, _fill_checklist

    lst = QListWidget()
    _fill_checklist(lst, ["t2m", "ws", "blh"], {"t2m"})
    _toggle_all(lst)
    assert _checked_items(lst) == ["t2m", "ws", "blh"]
    _toggle_all(lst)
    assert _checked_items(lst) == []
    _toggle_all(lst)
    assert _checked_items(lst) == ["t2m", "ws", "blh"]


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
        assert win.tabs.tabText(6) == "PDP"
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


def test_chronos_backend_hides_the_training_controls(qapp):
    """Chronos-2 is zero-shot, so budget / search space / split must disappear.

    Leaving them visible would imply the run honours them; it does not.
    """
    from normet.gui import _chronos
    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        win.load_example()
        win.show_panel_page("Model")  # the panel shows one step's page at a time
        assert win.backend_combo.currentText() == "flaml"
        assert win.budget_spin.isVisibleTo(win)
        assert win.split_combo.isVisibleTo(win)

        win.backend_combo.setCurrentText(_chronos.BACKEND)
        for widget in (
            win.budget_spin,
            win.estimator_list,
            win.split_combo,
            win.train_fraction_spin,
            win.seed_spin,
        ):
            assert not widget.isVisibleTo(win)
        assert win._chronos_hint.isVisibleTo(win)
        assert "Chronos-2" in win.train_button.text()

        # ...and come back when an AutoML backend is picked again.
        win.backend_combo.setCurrentText("lightgbm")
        assert win.budget_spin.isVisibleTo(win)
        assert win.split_combo.isVisibleTo(win)
        assert not win._chronos_hint.isVisibleTo(win)
        assert "Train model" in win.train_button.text()
    finally:
        win.close()


def test_chronos_backend_swaps_the_monte_carlo_sample_count(qapp):
    """300 samples is right for a tree ensemble and ruinous for a transformer.

    Each sample is one 2048-context forward pass under Chronos-2, so carrying the
    AutoML default across would turn "Normalise" into a multi-day run. The value
    the user set on the other side has to survive the round trip.
    """
    from normet.gui import _chronos
    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        win.load_example()
        assert win.norm_samples.value() == 300

        win.backend_combo.setCurrentText(_chronos.BACKEND)
        assert win.norm_samples.value() == _chronos.DEFAULT_SAMPLES

        # A count chosen while on chronos-2 is kept while on chronos-2...
        win.norm_samples.setValue(4)
        win.backend_combo.setCurrentText("lightgbm")
        assert win.norm_samples.value() == 300
        win.backend_combo.setCurrentText(_chronos.BACKEND)
        assert win.norm_samples.value() == 4
    finally:
        win.close()


def test_chronos_normalise_panel_declares_the_seeded_rows(qapp):
    """The first context_length rows are observed values, not a result.

    ``deweather`` seeds them because Chronos-2 has no history to condition on
    yet, so the two curves coincide there by construction. Left unsaid, a short
    record produces a plot that is mostly a copy of the input and a mean effect
    diluted toward zero by rows that could never have differed.
    """
    import types

    import numpy as np
    import pandas as pd

    from normet.gui import _chronos
    from normet.gui.main_window import MainWindow

    n, ctx = 300, 200
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    observed = pd.Series(np.linspace(10.0, 20.0, n), index=idx)
    result = pd.DataFrame({"observed": observed, "normalised": observed.copy()}, index=idx)
    result.index.name = "date"
    # Only the projected tail differs -- a clean +4.0 there, 0.0 over the seed.
    result.iloc[ctx:, result.columns.get_loc("normalised")] += 4.0

    win = MainWindow()
    try:
        win.load_example()
        win.backend_combo.setCurrentText(_chronos.BACKEND)
        win.model = types.SimpleNamespace(context_length=ctx)
        win._normalise_done(result)

        from PySide6.QtWidgets import QLabel

        text = " ".join(" ".join(lbl.text() for lbl in win.tab_norm.findChildren(QLabel)).split())
        assert f"first {ctx:,} rows" in text.lower(), text
        assert "67%" in text, text
        # Averaged over the whole frame this would be +2.67; over the tail it is +4.
        assert "+4.00" in text, text
    finally:
        win.close()


def test_chronos_option_is_greyed_out_without_the_extra(qapp):
    """The combo entry must be disabled, not raise ImportError when picked."""
    import importlib.util

    from normet.gui import _chronos
    from normet.gui.main_window import MainWindow

    have_extra = all(importlib.util.find_spec(p) is not None for p in ("chronos", "torch"))
    win = MainWindow()
    try:
        index = win.backend_combo.findText(_chronos.BACKEND)
        assert index >= 0
        item = win.backend_combo.model().item(index)
        assert item.isEnabled() is have_extra
        if not have_extra:
            assert "foundation" in item.toolTip()
    finally:
        win.close()


def test_chronos_load_requires_a_met_covariate(qapp, monkeypatch):
    """Without covariates a Chronos-2 run is a forecast, not a de-weathering."""
    from normet.gui import _chronos
    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        win.load_example()
        win.backend_combo.setCurrentText(_chronos.BACKEND)

        submitted: list[str] = []
        monkeypatch.setattr(win.runner, "submit", lambda name, *a, **k: submitted.append(name))

        # Only time features ticked -> no meteorology for the covariate channel.
        monkeypatch.setattr(win, "_selected_features", lambda: list(_time_vars()))
        win._run_train()

        # Said on a notice above the results, with the step that answers it.
        bar = win.notices.get("chronos-covariates")
        assert bar is not None and "covariate channel" in bar.label.text()
        assert [b.text() for b in bar.buttons] == ["Met only"]
        assert not submitted
    finally:
        win.close()


def _time_vars() -> tuple[str, ...]:
    from normet.gui.main_window import TIME_VARS

    return TIME_VARS


def test_chronos_backend_exposes_a_device_selector(qapp):
    """CPU works everywhere but is slow; a Mac needs mps and a workstation cuda.

    The selector only means anything for the zero-shot backend, so it appears
    with it and not before.
    """
    from normet.gui import _chronos
    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        win.load_example()
        win.show_panel_page("Model")  # the panel shows one step's page at a time
        assert not win.device_combo.isVisibleTo(win)

        win.backend_combo.setCurrentText(_chronos.BACKEND)
        assert win.device_combo.isVisibleTo(win)
        assert [win.device_combo.itemText(i) for i in range(win.device_combo.count())] == [
            "auto",
            "cpu",
            "cuda",
            "mps",
        ]

        # "auto" must reach the estimator as None, not as the string "auto".
        assert win._selected_device() is None
        win.device_combo.setCurrentText("mps")
        assert win._selected_device() == "mps"

        win.backend_combo.setCurrentText("flaml")
        assert not win.device_combo.isVisibleTo(win)
    finally:
        win.close()


def test_scm_fit_verdict_reads_the_pre_fit_only_when_a_shuffled_series_cannot_match_it(qapp):
    from normet.gui.scm_window import fit_verdict

    near_perfect = {"pre_r2": 0.999, "att": -18.14}
    level, text = fit_verdict(near_perfect, shuffled_r2=0.997)
    assert level == "warn"
    assert "Good" not in text and "Placebo in space" in text and "-18.14" in text
    assert fit_verdict(near_perfect, shuffled_r2=0.1)[0] == "ok"
    assert fit_verdict(near_perfect)[0] == "ok"  # mlscm / bayesian: no baseline
    assert fit_verdict({"pre_r2": 0.6, "att": -9.5}, shuffled_r2=0.2)[0] == "warn"
    assert fit_verdict({"pre_r2": -0.2, "att": -20.7}, shuffled_r2=-0.2)[0] == "error"


def test_shuffled_pre_r2_flags_a_design_with_more_donors_than_pre_cutoff_dates(qapp):
    import numpy as np
    import pandas as pd

    from normet.causal import BACKENDS
    from normet.gui.scm_window import NO_CHECK_R2, shuffled_pre_r2

    def design(n_donors: int) -> dict:
        # Donors on random walks over 12 pre-cutoff months; T is a mix of two.
        rng = np.random.default_rng(3)
        dates = pd.date_range("2020-01-01", periods=18, freq="MS")
        frames = [
            pd.DataFrame(
                {"date": dates, "ID": f"D{j}", "value": 50 + np.cumsum(rng.normal(0, 3, 18))}
            )
            for j in range(n_donors)
        ]
        treated = 0.5 * frames[0]["value"] + 0.5 * frames[1]["value"]
        frames.append(pd.DataFrame({"date": dates, "ID": "T", "value": treated}))
        return {
            "df": pd.concat(frames, ignore_index=True),
            "date_col": "date",
            "unit_col": "ID",
            "outcome_col": "value",
            "treated_unit": "T",
            "donors": [f"D{j}" for j in range(n_donors)],
            "cutoff_date": "2021-01-01",
        }

    assert shuffled_pre_r2(BACKENDS["scm"], design(40)) >= NO_CHECK_R2
    assert shuffled_pre_r2(BACKENDS["scm"], design(4)) < NO_CHECK_R2


def test_each_example_opens_set_up_as_its_source_sets_it(qapp):
    """File → Load Example: each example with its own target and predictors,
    and the Data box offers no example button of its own."""
    from normet.gui._examples import EXAMPLES
    from normet.gui.main_window import MainWindow, _checked_items

    win = MainWindow()
    try:
        assert not hasattr(win, "example_btn")
        win.load_example("my1")
        assert win.target_combo.currentText() == "NO2"
        assert set(_checked_items(win.feature_list)) == set(EXAMPLES["my1"].features)
        assert win.command_log.entries[-1].text().endswith("df = nm.datasets.load_my1()")
        win.load_example()  # the synthetic site
        assert win.target_combo.currentText() == "PM2.5"
    finally:
        win.close()


def test_a_table_on_local_time_keeps_its_clock_and_a_new_clock_reads_it_again(qapp, tmp_path):
    import pandas as pd

    from normet.gui.main_window import MainWindow

    path = tmp_path / "beijing.xlsx"
    pytest.importorskip("openpyxl")
    pd.DataFrame(
        {"时间": ["2020年1月1日 8时", "2020年1月1日 9时"], "PM2.5": [10.0, 12.0], "t2m": [1.0, 2.0]}
    ).to_excel(path, index=False)
    win = MainWindow()
    try:
        win.zone_combo.setCurrentIndex(win.zone_combo.findData("Asia/Shanghai"))
        win.load_csv(str(path))
        assert win.df_raw["date"].dt.hour.tolist() == [8, 9]  # shown on Beijing's clock
        assert "(Asia/Shanghai)" in win.data_summary.text()
        assert "zone='Asia/Shanghai'" in win.command_log.entries[-1].text()
        win.zone_combo.setCurrentIndex(win.zone_combo.findData("UTC"))
        assert win.df_raw["date"].dt.hour.tolist() == [8, 9]  # written without an offset
    finally:
        win.zone_combo.setCurrentIndex(win.zone_combo.findData("UTC"))
        win.close()


def test_the_group_editor_suggests_groups_and_keeps_the_text_form(qapp):
    """The decomposition's groups set in a table: Suggest groups puts each recognised
    variable in its group, one left alone is a contribution of its own, and the text
    form a config keeps goes both ways."""
    from normet.gui._groups import GroupEditor

    ed = GroupEditor()
    ed.set_variables(["ws", "wd", "temp", "blh", "traffic"])
    ed.suggest()
    assert (
        ed.text()
        == "Wind: ws, wd; Temperature and humidity: temp; Boundary layer and pressure: blh"
    )
    assert ed.groups(["ws", "wd", "temp", "blh", "traffic"])["traffic"] == ["traffic"]
    ed.clear()
    assert ed.text() == "" and ed.groups(["ws"]) is None
    ed.setText("local: temp, blh; wind: ws, wd")
    assert ed.groups(["ws", "wd", "temp", "blh"]) == {
        "local": ["temp", "blh"],
        "wind": ["ws", "wd"],
    }
    with pytest.raises(ValueError, match="rain not among"):
        ed.setText("wet: rain")
        ed.groups(["ws"])


def test_the_groups_show_only_for_the_meteorological_decomposition(qapp):
    """Groups split a meteorological decomposition; the emission one has none, so the
    table is there only when Method is meteorology."""
    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        assert win.decom_method.currentText() == "emission"
        assert win.decom_groups.isHidden() and win._decom_groups_label.isHidden()
        win.decom_method.setCurrentText("meteorology")
        assert not win.decom_groups.isHidden() and not win._decom_groups_label.isHidden()
        assert win.decom_attribution.isEnabled()
        win.decom_method.setCurrentText("emission")
        assert win.decom_groups.isHidden() and not win.decom_attribution.isEnabled()
    finally:
        win.close()


@pytest.mark.parametrize("method", ["meteorology", "emission"])
def test_the_groups_names_are_not_cut_short(qapp, method):
    """The longest group's name fits its cell beside the variables' names: the panel
    widens to the table once a model's variables are in it, whether the groups show
    then or later; a name still too long (typed in) reads from its start."""
    from PySide6.QtWidgets import QApplication, QComboBox

    from normet.gui._groups import PRESETS
    from normet.gui.main_window import MainWindow

    def settle():
        for _ in range(6):
            QApplication.sendPostedEvents()
            qapp.processEvents()

    win = MainWindow()
    try:
        win.resize(1320, 860)
        win.show()
        win.decom_method.setCurrentText(method)
        ed = win.decom_groups
        ed.set_variables(["ws", "wd", "temp", "RH", "atmos_pres", "blh", "tcc", "tp", "ssrd"])
        ed.suggest()
        win.param_panel.refit()  # as training does
        win.show_panel_page("Decompose")
        win.decom_method.setCurrentText("meteorology")
        settle()
        probe = QComboBox()
        probe.setEditable(True)
        probe.addItems(list(PRESETS))
        probe.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        assert ed.table.columnWidth(1) >= probe.sizeHint().width()
        assert ed.table.columnWidth(0) >= ed.table.sizeHintForColumn(0)
        box = ed._box(5)  # blh
        assert box.currentText() == "Boundary layer and pressure"
        assert box.toolTip() == "Boundary layer and pressure"
        box.setCurrentText("Boundary layer, pressure and everything above the canopy")
        assert box.lineEdit().cursorPosition() == 0
    finally:
        win.close()


def test_the_report_language_is_chosen_where_the_report_is_exported(qapp, monkeypatch, tmp_path):
    """As cume's: File > Export HTML Report and the HTML report button each list the
    languages, named in themselves and none ticked; the one chosen writes the report and is
    kept for the next. A Chinese window names English as English."""
    import pandas as pd
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QFileDialog

    import normet
    from normet.gui import uilang
    from normet.gui._shell import report_language
    from normet.gui.main_window import MainWindow

    prefs = QSettings("apai-sys", "normet-gui")
    kept = prefs.value("report_language")
    win = MainWindow()
    try:
        menu = win._report_action.menu()
        assert [a.text() for a in menu.actions()] == list(uilang.LANGUAGES.values())
        assert not any(a.isCheckable() for a in menu.actions())
        assert [a.text() for a in win.exp_html.menu().actions()] == list(uilang.LANGUAGES.values())
        assert uilang.tr("English", "zh") == "English" and uilang.tr("中文", "en") == "中文"

        written = []
        path = str(tmp_path / "r.html")
        monkeypatch.setattr(win, "_current_result", lambda: ("normalise", pd.DataFrame({"a": [1]})))
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (path, ""))
        )
        monkeypatch.setattr(normet, "make_run", lambda *a, **k: "run")
        monkeypatch.setattr(
            normet, "generate_html_report", lambda run, p, lang: written.append((p, lang))
        )
        menu.actions()[list(uilang.LANGUAGES).index("zh")].trigger()
        assert written == [(path, "zh")] and report_language() == "zh"
        menu.actions()[list(uilang.LANGUAGES).index("en")].trigger()
        assert written[-1] == (path, "en") and report_language() == "en"

        # A cancelled save changes nothing.
        monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: ("", "")))
        menu.actions()[list(uilang.LANGUAGES).index("zh")].trigger()
        assert len(written) == 2 and report_language() == "en"
    finally:
        win.close()
        if kept is None:
            prefs.remove("report_language")
        else:
            prefs.setValue("report_language", kept)
