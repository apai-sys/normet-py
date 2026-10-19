"""Transport Studio and the main window's join of its features (offscreen; a shell
script stands in for hyts_std)."""

from __future__ import annotations

import os
import shutil
import tempfile
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


def _wait(qapp, runner, timeout=30):
    t0 = time.time()
    while runner.busy and time.time() - t0 < timeout:
        qapp.processEvents()
        time.sleep(0.01)
    for _ in range(20):
        qapp.processEvents()


def _until(qapp, cond, timeout=10):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        qapp.processEvents()
        time.sleep(0.02)
    return cond()


@pytest.fixture
def main_win(qapp):
    from normet.gui.main_window import MainWindow

    w = MainWindow()
    yield w
    w.close()


@pytest.fixture
def studio(qapp, tmp_path, monkeypatch):
    from normet.gui.trajectory_window import TrajectoryWindow

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # run_back_trajectories' work dirs
    s = TrajectoryWindow()
    yield s
    s.close()


def _panel(start="2020-01-01", hours=48, **cols):
    n = np.arange(hours, dtype=float)
    return pd.DataFrame(
        {"date": pd.date_range(start, periods=hours, freq="h"), "PM2.5": n, "t2m": n, **cols}
    )


def _features(times, site=None):
    idx = pd.DatetimeIndex(times, name="date")
    f = pd.DataFrame(
        {"traj_dist_km": idx.hour.to_numpy(float), "traj_n_endpoints": 73.0}, index=idx
    )
    if site is not None:
        f.insert(0, "site", site)
    return f


SIX_HOURLY = ["2020-01-01 00:00", "2020-01-01 06:00", "2020-01-01 12:00", "2020-01-01 18:00"]


def _ticked(w):
    from PySide6.QtCore import Qt

    items = (w.feature_list.item(i) for i in range(w.feature_list.count()))
    return {it.text() for it in items if it.checkState() == Qt.CheckState.Checked}


def test_send_takes_the_latest_trajectory_to_have_arrived(main_win):
    w = main_win
    w._ingest(_panel(), "two days")
    # Trajectories over the first day only, every 6 h.
    assert w.merge_trajectory_features(_features(SIX_HOURLY)) == 24
    got = w.df_raw.set_index("date")["traj_dist_km"]
    assert got["2020-01-01 04:00"] == 0.0  # 00:00's: 06:00's has not arrived yet
    assert got["2020-01-01 23:00"] == 18.0
    assert got["2020-01-02 00:00":].isna().all()  # no trajectory: left empty, not carried
    assert {"traj_dist_km", "traj_n_endpoints"} <= _ticked(w)

    # Sent again, the columns are replaced rather than doubled into _x / _y.
    again = _features(SIX_HOURLY)
    again["traj_dist_km"] += 100
    assert w.merge_trajectory_features(again) == 24
    assert not [c for c in w.df_raw.columns if c.endswith(("_x", "_y"))]
    assert w.df_raw.set_index("date")["traj_dist_km"]["2020-01-01 04:00"] == 100.0


def test_send_for_a_site_the_data_does_not_have(main_win, monkeypatch):
    w = main_win
    w._ingest(_panel(site="MY1"), "MY1")
    seen = {}
    monkeypatch.setattr(
        "normet.gui.main_window.QMessageBox.warning", lambda *a: seen.update(text=a[2])
    )
    before = w.df_raw
    assert w.merge_trajectory_features(_features(SIX_HOURLY, site="Marylebone")) == 0
    assert "Marylebone" in seen["text"] and "MY1" in seen["text"]
    assert w.df_raw is before  # left as it was, not reported as merged

    # Several sites' features onto data with no site column: said, not guessed.
    w._ingest(_panel(), "one site")
    two = pd.concat([_features(SIX_HOURLY, site="A"), _features(SIX_HOURLY, site="B")])
    assert w.merge_trajectory_features(two) == 0
    assert "2 sites (A, B)" in seen["text"]


def test_send_onto_dates_with_a_gap_or_a_zone(main_win):
    # A date that did not parse used to stop the join ("Merge keys contain null
    # values"), and a zoned one (a CSV in ...Z) to fail it outright.
    w = main_win
    df = _panel(hours=6)
    df.loc[2, "date"] = pd.NaT
    w._ingest(df, "with a gap")
    assert w.merge_trajectory_features(_features(SIX_HOURLY)) == 5
    assert np.isnan(w.df_raw.loc[2, "traj_dist_km"])

    df = _panel("2020-01-01 01:00", hours=6)
    df["date"] = df["date"].dt.tz_localize("Europe/Paris")  # 01:00 here is 00:00 UTC
    w._ingest(df, "zoned")
    assert w.merge_trajectory_features(_features(SIX_HOURLY)) == 6
    assert w.df_raw["traj_dist_km"].tolist() == [0.0] * 6


def test_tables_say_which_row_is_wrong(studio):
    from PySide6.QtWidgets import QTableWidgetItem

    t = studio.region_table
    t.add_row()  # left as added: skipped, as before
    t.add_row("ind", (116.0, 39.0, 120.0, 42.0))
    assert t.regions() == {"ind": (116.0, 39.0, 120.0, 42.0)}
    t.add_row("bad", (120.0, 39.0, 116.0, 42.0))
    with pytest.raises(ValueError, match=r"row 3 \(bad\): lon_min must be less than lon_max"):
        t.regions()
    t.removeRow(2)
    t.add_row("", (1.0, 2.0, 3.0, 4.0))
    with pytest.raises(ValueError, match="row 3: give it a name"):
        t.regions()

    r = studio.receptor_table
    r.add_row()
    r.add_row("MY1", 51.52, -0.15)
    assert r.receptors() == [("MY1", 51.52, -0.15)]
    r.setItem(1, 1, QTableWidgetItem("north"))
    with pytest.raises(ValueError, match=r"Receptor row 2 \(MY1\): lat and lon must be numbers"):
        r.receptors()


def _example_tdumps(tmp_path, n=2):
    from normet.datasets import example_traj_dir

    src = sorted(example_traj_dir().glob("tdump_*"))[:n]
    return [shutil.copy(p, tmp_path / p.name) for p in src]


def test_build_says_what_is_wrong_with_a_region(studio, monkeypatch, tmp_path):
    studio._tdump_paths = [str(p) for p in _example_tdumps(tmp_path)]
    studio.region_table.add_row("bad", (5.0, 0.0, 1.0, 1.0))
    seen = {}
    monkeypatch.setattr(
        "normet.gui.trajectory_window.QMessageBox.warning", lambda *a: seen.update(text=a[2])
    )
    studio._run_build()
    assert "lon_min must be less" in seen["text"] and not studio.runner.busy


def test_build_nulls_short_trajectories_and_counts_features(studio, qapp, tmp_path):
    studio._tdump_paths = [str(p) for p in _example_tdumps(tmp_path)]
    studio.build_site_edit.setText("MY1")
    studio.build_min_check.setChecked(True)
    studio.build_min_spin.setValue(100)  # the example trajectories reach back 72 h
    studio._run_build()
    _wait(qapp, studio.runner)
    f = studio.features
    assert f["traj_dist_km"].isna().all() and (f["traj_age_max_h"] == 72).all()
    n = f.shape[1] - 1  # the site tag is not a feature
    assert studio.statusBar().currentMessage().startswith(f"Built {n} transport features")


def test_main_data_sets_the_gdas1_range(studio):
    from PySide6.QtCore import QDate

    studio.hours_back_spin.setValue(72)
    studio.offer_main_data(_panel("2020-03-10 05:00", hours=30))
    assert studio.gdas_from.date() == QDate(2020, 3, 7)
    assert studio.gdas_to.date() == QDate(2020, 3, 11)
    studio.hours_back_spin.setValue(120)
    assert studio.gdas_from.date() == QDate(2020, 3, 5)


def test_existing_met_files_can_be_used(studio, monkeypatch, tmp_path):
    met = [str(tmp_path / "gdas1.jan20.w2"), str(tmp_path / "gdas1.jan20.w1")]
    monkeypatch.setattr(
        "normet.gui.trajectory_window.QFileDialog.getOpenFileNames", lambda *a: (met, "")
    )
    studio._choose_met_files()
    assert studio._gdas_paths == sorted(met)
    assert studio.gdas_hint.text().startswith("2 met file(s) chosen: gdas1.jan20.w1")


def test_a_large_download_is_asked_about_first(studio, monkeypatch, tmp_path):
    from PySide6.QtCore import QDate
    from PySide6.QtWidgets import QMessageBox

    studio.gdas_dir_edit.setText(str(tmp_path))
    studio.gdas_from.setDate(QDate(2020, 1, 1))
    studio.gdas_to.setDate(QDate(2020, 3, 1))
    asked = {}
    monkeypatch.setattr(
        "normet.gui.trajectory_window.QMessageBox.question",
        lambda *a: (asked.update(text=a[2]), QMessageBox.StandardButton.No)[1],
    )
    studio._run_download_gdas()
    assert "weekly GDAS1 file(s) to download" in asked["text"] and not studio.runner.busy


SLOW_HYTS = """\
#!/bin/sh
touch "$FAKE_MARK"
exec sleep 30
"""


def _ready_to_run(studio, tmp_path, monkeypatch, met_names, script=SLOW_HYTS):
    exe = tmp_path / "exec" / "hyts_std"
    exe.parent.mkdir()
    exe.write_text(script)
    exe.chmod(0o755)
    monkeypatch.setenv("FAKE_MARK", str(tmp_path / "started"))
    studio.hysplit_exec_edit.setText(str(exe))
    for n in met_names:
        (tmp_path / n).write_text("")
    studio._set_met_files([str(tmp_path / n) for n in met_names], "chosen")
    studio.receptor_table.add_row("MY1", 51.5, -0.13)
    df = _panel("2020-01-10", hours=6)
    df["date"] = df["date"].dt.tz_localize("UTC")  # as the OpenAQ/UK-AIR fetchers give it
    studio.offer_main_data(df)


def test_run_asks_before_going_without_meteorology(studio, monkeypatch, tmp_path):
    from PySide6.QtCore import QDate
    from PySide6.QtWidgets import QMessageBox

    _ready_to_run(studio, tmp_path, monkeypatch, ["gdas1.jan20.w2"])  # w1 is needed too
    studio.gdas_from.setDate(QDate(2021, 1, 1))
    asked = {}
    monkeypatch.setattr(
        "normet.gui.trajectory_window.QMessageBox.question",
        lambda *a: (asked.update(text=a[2]), QMessageBox.StandardButton.No)[1],
    )
    studio._run_hysplit()
    assert "1 GDAS1 file(s) the met files lack: gdas1.jan20.w1" in asked["text"]
    assert not studio.runner.busy
    assert studio.gdas_from.date() == QDate(2020, 1, 7)  # set to cover it, to download


def test_cancel_stops_the_run_in_progress(studio, qapp, monkeypatch, tmp_path):
    _ready_to_run(studio, tmp_path, monkeypatch, ["gdas1.jan20.w1", "gdas1.jan20.w2"])
    studio._run_hysplit()
    assert studio.runner.busy and studio._hysplit_stop is not None
    assert _until(qapp, lambda: (tmp_path / "started").exists())
    thread = studio.runner._active[0]
    studio._abandon()
    assert "stopped" in studio.statusBar().currentMessage()
    # Killed, not left to run on: its thread ends at once, not after the 30 s sleep.
    t0 = time.time()
    assert _until(qapp, lambda: not studio.runner._zombies)
    assert time.time() - t0 < 10
    del thread


def test_closing_the_window_stops_the_run(studio, qapp, monkeypatch, tmp_path):
    _ready_to_run(studio, tmp_path, monkeypatch, ["gdas1.jan20.w1", "gdas1.jan20.w2"])
    studio._run_hysplit()
    assert _until(qapp, lambda: (tmp_path / "started").exists())
    thread = studio.runner._active[0]
    t0 = time.time()
    studio.close()
    assert not thread.isRunning() and time.time() - t0 < 6


def test_closing_the_main_window_closes_transport_studio(main_win):
    main_win.open_transport_studio()
    shown = main_win._transport_window
    assert shown.isVisible()
    main_win.close()
    assert not shown.isVisible()
