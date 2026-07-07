"""Tests for the bundled example datasets."""

from __future__ import annotations

import pandas as pd

from normet import datasets
from normet.io.trajectory import read_trajectory_tdump


def test_load_my1():
    df = datasets.load_my1()
    assert df.shape == (5793, 11)
    assert list(df.columns[:2]) == ["date", "NO2"]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].min() == pd.Timestamp("2020-01-01 00:00:00")
    assert df["date"].max() == pd.Timestamp("2020-08-31 23:00:00")
    assert df["NO2"].notna().all()


def test_load_scm():
    df = datasets.load_scm()
    assert df.shape == (7378, 6)
    assert set(df.columns) == {"date", "NO2_obs", "NO2_dw", "NO2_dw_common", "code", "type"}
    assert "MY1" in set(df["code"])
    assert df["code"].nunique() == 104


def test_load_my1_pm25():
    df = datasets.load_my1_pm25()
    assert df.shape == (5856, 17)
    assert "pm25" in df.columns
    assert {"ws_era5", "wd_era5", "temp_era5", "blh", "sp", "ssrd", "tcc", "tp"} <= set(df.columns)


def test_load_traj_my1():
    df = datasets.load_traj_my1()
    assert df.shape == (976, 12)
    traj_cols = [c for c in df.columns if c.startswith("traj_")]
    assert len(traj_cols) == 11


def test_traj_joins_with_pm25():
    pm25 = datasets.load_my1_pm25()
    traj = datasets.load_traj_my1()
    merged = pm25.merge(traj, on="date", how="inner")
    assert len(merged) == len(traj)  # every 6-hourly arrival matches an hourly row


def test_example_traj_dir_readable():
    d = datasets.example_traj_dir()
    files = sorted(d.glob("tdump_*"))
    assert len(files) == 8
    traj = read_trajectory_tdump(files[0])
    assert not traj.empty
