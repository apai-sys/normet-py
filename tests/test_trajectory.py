"""Tests for the HYSPLIT back-trajectory adapter (normet.io.trajectory)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from normet.io import trajectory as tj

# A minimal but format-correct HYSPLIT tdump: 1 met grid, 1 backward trajectory,
# 3 diagnostic vars (PRESSURE RAINFALL MIXDEPTH), 3 endpoints (age 0, -1, -2).
# Receptor (age 0) at (51.520, -0.130); air origin (age -2) at (51.000, -2.500).
TDUMP = """\
     1     1
    GDAS    20    10     1     0     0
     1 BACKWARD OMEGA
     1    20    10     1     0   51.520   -0.130    100.0
     3 PRESSURE RAINFALL MIXDEPTH
     1     1    20    10     1     0     0     0.0     0.0   51.520   -0.130    100.0    995.0    0.0   800.0
     1     1    20     9    30    23     0     0.0    -1.0   51.300   -1.200    300.0    980.0    0.5   600.0
     1     1    20     9    30    22     0     0.0    -2.0   51.000   -2.500    500.0    970.0    1.0   500.0
"""


def _write(tmp_path, name="tdump_2020100100"):
    p = tmp_path / name
    p.write_text(TDUMP)
    return p


def test_read_trajectory_tdump(tmp_path):
    df = tj.read_trajectory_tdump(_write(tmp_path))

    assert len(df) == 3
    assert {"age_h", "lat", "lon", "height", "datetime"}.issubset(df.columns)
    # MIXDEPTH -> blh, RELHUMID -> rh renames; rainfall/pressure kept.
    assert "blh" in df.columns and "rainfall" in df.columns and "pressure" in df.columns
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

    # Geometry: origin is SW of the receptor -> westerly inflow sector.
    assert f["traj_dist_km"] > 100.0
    assert 200.0 < f["traj_inflow_deg"] < 290.0
    assert f["traj_pathlen_km"] >= f["traj_dist_km"]  # path >= straight line

    # Only the origin endpoint falls in the SW box -> 1 of 3 endpoints.
    assert f["traj_resid_sw_box"] == 1 / 3


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
