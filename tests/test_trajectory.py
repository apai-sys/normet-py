"""Tests for the HYSPLIT back-trajectory adapter (normet.io.trajectory)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from normet.io import trajectory as tj

# A minimal but format-correct HYSPLIT tdump: 1 met grid, 1 backward trajectory,
# 5 diagnostic vars (PRESSURE RAINFALL MIXDEPTH RELHUMID AIR_TEMP), 3 endpoints
# (age 0, -1, -2). Receptor (age 0) at (51.520, -0.130); air origin (age -2)
# at (51.000, -2.500).
TDUMP = """\
     1     1
    GDAS    20    10     1     0     0
     1 BACKWARD OMEGA
     1    20    10     1     0   51.520   -0.130    100.0
     5 PRESSURE RAINFALL MIXDEPTH RELHUMID AIR_TEMP
     1     1    20    10     1     0     0     0.0     0.0   51.520   -0.130    100.0    995.0    0.0   800.0     70.0    285.0
     1     1    20     9    30    23     0     0.0    -1.0   51.300   -1.200    300.0    980.0    0.5   600.0     75.0    283.0
     1     1    20     9    30    22     0     0.0    -2.0   51.000   -2.500    500.0    970.0    1.0   500.0     80.0    281.0
"""


def _write(tmp_path, name="tdump_2020100100"):
    p = tmp_path / name
    p.write_text(TDUMP)
    return p


def test_read_trajectory_tdump(tmp_path):
    df = tj.read_trajectory_tdump(_write(tmp_path))

    assert len(df) == 3
    assert {"age_h", "lat", "lon", "height", "datetime"}.issubset(df.columns)
    # MIXDEPTH -> blh, RELHUMID -> rh, AIR_TEMP -> temp renames; rainfall/pressure kept.
    assert {"blh", "rh", "temp", "rainfall", "pressure"}.issubset(df.columns)
    # 2-digit year decoded to 2020; receptor row is age 0.
    receptor = df.loc[df["age_h"] == 0.0, "datetime"].iloc[0]
    assert receptor == pd.Timestamp("2020-10-01 00:00")


def test_trajectory_features(tmp_path):
    df = tj.read_trajectory_tdump(_write(tmp_path))
    f = tj.trajectory_features(df, source_regions={"sw_box": (-3.0, 50.5, -1.5, 51.5)})

    # Along-path diagnostics.
    assert f["traj_blh_mean"] == 800 / 3 + 600 / 3 + 500 / 3  # (800+600+500)/3
    assert f["traj_rain_sum"] == 1.5
    assert f["traj_height_min"] == 100.0
    assert f["traj_rh_mean"] == pytest.approx((70.0 + 75.0 + 80.0) / 3)
    assert f["traj_pressure_mean"] == pytest.approx((995.0 + 980.0 + 970.0) / 3)
    assert f["traj_temp_mean"] == pytest.approx((285.0 + 283.0 + 281.0) / 3)

    # Geometry: origin is SW of the receptor -> westerly inflow sector.
    assert f["traj_dist_km"] > 100.0
    assert 200.0 < f["traj_inflow_deg"] < 290.0
    assert f["traj_pathlen_km"] >= f["traj_dist_km"]  # path >= straight line

    # Only the origin endpoint falls in the SW box -> 1 of 3 endpoints.
    assert f["traj_resid_sw_box"] == 1 / 3


def test_trajectory_quality_columns(tmp_path):
    df = tj.read_trajectory_tdump(_write(tmp_path))
    f = tj.trajectory_features(df)

    # The fixture reaches back exactly 2 h over 3 endpoints.
    assert f["traj_n_endpoints"] == 3
    assert f["traj_age_max_h"] == 2.0
    # Default (min_hours=None) keeps a short trajectory's geometry intact.
    assert np.isfinite(f["traj_dist_km"])


def test_min_hours_nulls_truncated_trajectory(tmp_path):
    df = tj.read_trajectory_tdump(_write(tmp_path))
    box = {"sw_box": (-3.0, 50.5, -1.5, 51.5)}

    # Reach (2 h) satisfies min_hours=2 -> untouched.
    ok = tj.trajectory_features(df, source_regions=box, min_hours=2)
    assert np.isfinite(ok["traj_dist_km"]) and ok["traj_resid_sw_box"] == 1 / 3

    # Asking for 72 h of a 2 h trajectory: every feature NaN except the quality
    # columns, which stay so the truncation is visible rather than silent.
    short = tj.trajectory_features(df, source_regions=box, min_hours=72)
    assert short["traj_n_endpoints"] == 3 and short["traj_age_max_h"] == 2.0
    nulled = {k for k in short if k not in ("traj_n_endpoints", "traj_age_max_h")}
    assert nulled and all(np.isnan(short[k]) for k in nulled)
    # Same columns either way, so a frame built from a mix stays rectangular.
    assert set(short) == set(ok)


def test_build_trajectory_features_warns_on_truncated(tmp_path, caplog):
    _write(tmp_path, "tdump_a")

    with caplog.at_level("WARNING"):
        out = tj.build_trajectory_features(str(tmp_path / "tdump_*"), min_hours=72)

    assert out["traj_dist_km"].isna().all()
    assert out["traj_age_max_h"].iloc[0] == 2.0
    assert any("truncated" in r.message for r in caplog.records)


def test_build_trajectory_features(tmp_path):
    _write(tmp_path, "tdump_a")
    _write(tmp_path, "tdump_b")  # same receptor time -> deduplicated

    out = tj.build_trajectory_features(
        str(tmp_path / "tdump_*"),
        source_regions={"sw_box": (-3.0, 50.5, -1.5, 51.5)},
    )

    assert out.index.name == "date"
    assert len(out) == 1  # deduplicated on receptor timestamp
    assert out.index[0] == pd.Timestamp("2020-10-01 00:00")
    assert {"traj_dist_km", "traj_inflow_deg", "traj_resid_sw_box"}.issubset(out.columns)
    assert np.isfinite(out.iloc[0]["traj_dist_km"])


def test_control_text():
    txt = tj._control_text(
        pd.Timestamp("2020-10-17 00:00"),
        40.0,
        -90.0,
        500.0,
        24,
        ["/data/oct1618.BIN"],
        "tdump_x",
        top_of_model=10000.0,
        vert_motion=0,
    )
    lines = txt.splitlines()
    assert lines[0] == "20 10 17 00"  # YY MM DD HH
    assert lines[1] == "1"  # one location
    assert lines[2] == "40.0000 -90.0000 500.0"
    assert lines[3] == "-24"  # negative run hours = backward
    assert lines[4] == "0"  # vertical motion
    assert lines[6] == "1"  # n_met
    assert lines[7].endswith(os.sep)  # met dir, trailing separator
    assert lines[8] == "oct1618.BIN"  # met filename
    assert lines[-1] == "tdump_x"  # output tdump name


def test_setup_cfg_text():
    txt = tj._setup_cfg_text(tj.ALL_DIAGNOSTICS)
    lines = txt.splitlines()
    assert lines[0] == "&SETUP"
    assert lines[-1] == "/"
    assert "tm_pres = 1," in lines
    assert "tm_rain = 1," in lines
    assert "tm_mixd = 1," in lines
    assert "tm_relh = 1," in lines
    assert "tm_tamb = 1," in lines

    # Subset -> the rest explicitly off, not just omitted.
    subset = tj._setup_cfg_text(["pressure", "rh"])
    assert "tm_pres = 1," in subset.splitlines()
    assert "tm_rain = 0," in subset.splitlines()
    assert "tm_relh = 1," in subset.splitlines()
    assert "tm_tamb = 0," in subset.splitlines()

    with pytest.raises(ValueError, match="Unknown diagnostic"):
        tj._setup_cfg_text(["bogus"])


MET = ["gdas1.jan20.w1", "gdas1.jan20.w2", "gdas1.jan20.w3", "gdas1.jan20.w4", "gdas1.jan20.w5"]


def test_filter_met_files_keeps_only_overlapping_weeks():
    ts = pd.Timestamp
    # 72 h back from 16 Jan 12:00 stays inside w3 (15-21 Jan) and w2 (8-14 Jan).
    kept = tj._filter_met_files(MET, ts("2020-01-13 12:00"), ts("2020-01-16 12:00"))
    assert kept == ["gdas1.jan20.w2", "gdas1.jan20.w3"]
    # Entirely inside one week -> one file.
    assert tj._filter_met_files(MET, ts("2020-01-16 00:00"), ts("2020-01-17 00:00")) == [
        "gdas1.jan20.w3"
    ]


def test_filter_met_files_pads_the_window_across_a_week_boundary():
    ts = pd.Timestamp
    # 22:00 on 7 Jan lies in the gap between w1's last GDAS1 record (21:00) and
    # w2's first (8 Jan 00:00); interpolating there needs w2 as well. A strict
    # overlap test would drop it and hyts_std would fail (probed against hyts_std).
    kept = tj._filter_met_files(MET, ts("2020-01-07 16:00"), ts("2020-01-07 22:00"))
    assert kept == ["gdas1.jan20.w1", "gdas1.jan20.w2"]
    # ...but not once the window is a full record interval clear of the boundary.
    kept = tj._filter_met_files(MET, ts("2020-01-07 06:00"), ts("2020-01-07 12:00"))
    assert kept == ["gdas1.jan20.w1"]


def test_filter_met_files_always_keeps_unrecognised_names():
    ts = pd.Timestamp
    paths = ["gdas1.jan20.w1", "custom_met.BIN", "gdas1.jan20.w4"]
    kept = tj._filter_met_files(paths, ts("2020-01-02"), ts("2020-01-03"))
    assert kept == ["gdas1.jan20.w1", "custom_met.BIN"]


FAKE_HYTS = """\
#!/bin/sh
# Stand-in for hyts_std: log the CONTROL it was given, then emit a canned tdump.
name=$(tail -n 1 CONTROL)
cp CONTROL "CONTROL_$name"
cp "$FAKE_TDUMP" "$name"
"""


def _fake_run(tmp_path, monkeypatch, times, met_names, **kw):
    exe = tmp_path / "exec" / "hyts_std"
    exe.parent.mkdir()
    exe.write_text(FAKE_HYTS)
    exe.chmod(0o755)
    monkeypatch.setenv("FAKE_TDUMP", str(_write(tmp_path)))
    mets = []
    for n in met_names:
        (tmp_path / n).write_text("")
        mets.append(str(tmp_path / n))
    work = tmp_path / "work"
    tj.run_back_trajectories(
        times, 51.5, -0.13, met_files=mets, hysplit_exec=exe, work_dir=work, **kw
    )
    return work


def _control_mets(work, name):
    lines = (work / f"CONTROL_{name}").read_text().splitlines()
    n_met = int(lines[6])
    return [lines[8 + 2 * i] for i in range(n_met)]  # (dir, file) pairs -> file names


def test_run_back_trajectories_passes_only_relevant_met_files(tmp_path, monkeypatch):
    work = _fake_run(
        tmp_path,
        monkeypatch,
        [pd.Timestamp("2020-01-16 12:00"), pd.Timestamp("2020-01-03 06:00")],
        MET,
        hours_back=72,
    )
    assert _control_mets(work, "tdump_2020011612") == ["gdas1.jan20.w2", "gdas1.jan20.w3"]
    assert _control_mets(work, "tdump_2020010306") == ["gdas1.jan20.w1"]


def test_run_back_trajectories_falls_back_to_all_met_files(tmp_path, monkeypatch):
    # No file's dates overlap the window -> hand hyts_std everything rather than
    # nothing, and let it report the coverage problem.
    work = _fake_run(
        tmp_path, monkeypatch, [pd.Timestamp("2021-06-01 00:00")], MET[:2], hours_back=24
    )
    assert _control_mets(work, "tdump_2021060100") == MET[:2]


def test_run_back_trajectories_warns_on_truncated(tmp_path, monkeypatch, caplog):
    # The canned tdump reaches back 2 h, the run asks for 24 -> truncated.
    with caplog.at_level("WARNING"):
        _fake_run(tmp_path, monkeypatch, [pd.Timestamp("2020-01-16 12:00")], MET, hours_back=24)
    assert any("stopped short" in r.message for r in caplog.records)


def test_run_back_trajectories_requires_executable(tmp_path):
    # Missing/non-executable hyts_std -> clear error, no HYSPLIT needed.
    with pytest.raises(FileNotFoundError):
        tj.run_back_trajectories(
            [pd.Timestamp("2020-10-17")],
            40.0,
            -90.0,
            met_files=[str(tmp_path / "oct1618.BIN")],
            hysplit_exec=str(tmp_path / "nonexistent_hyts_std"),
        )
