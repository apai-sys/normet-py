"""Tests for the HYSPLIT back-trajectory adapter (normet.io.trajectory)."""

from __future__ import annotations

import os
import sys

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


def test_overlapping_regions_are_detected():
    regions = {
        "west": (-10.0, 50.0, 0.0, 55.0),
        "east": (-2.0, 50.0, 5.0, 55.0),  # shares -2..0 with west
        "north": (-10.0, 55.0, 0.0, 60.0),  # only touches west along lat 55
        "far": (20.0, 30.0, 25.0, 35.0),
    }
    assert tj._overlapping_regions(regions) == [("west", "east")]


def test_overlap_check_handles_polygons_and_numpy_boxes():
    from shapely.geometry import Polygon

    tri = Polygon([(-5.0, 50.0), (5.0, 50.0), (0.0, 58.0)])
    regions = {
        "tri": tri,
        "box32": tuple(np.float32(v) for v in (-1.0, 51.0, 1.0, 52.0)),  # inside the triangle
        "away": (20.0, 30.0, 25.0, 35.0),
    }
    assert tj._overlapping_regions(regions) == [("tri", "box32")]


def test_float32_box_gives_the_same_residence_as_a_float_box(tmp_path):
    # A box of numpy scalars used to fall through to the shapely path and crash.
    df = tj.read_trajectory_tdump(_write(tmp_path))
    box = (-3.0, 50.5, -1.5, 51.5)
    f64 = tj.trajectory_features(df, source_regions={"b": box})
    f32 = tj.trajectory_features(df, source_regions={"b": tuple(np.float32(v) for v in box)})
    assert f32["traj_resid_b"] == f64["traj_resid_b"] == 1 / 3


def test_build_trajectory_features_warns_on_overlapping_regions(tmp_path, caplog):
    _write(tmp_path, "tdump_a")
    with caplog.at_level("WARNING"):
        tj.build_trajectory_features(
            str(tmp_path / "tdump_*"),
            source_regions={"uk": (-6.0, 50.0, 2.0, 56.0), "sw": (-3.0, 50.5, -1.5, 51.5)},
        )
    assert any("overlap (uk & sw)" in r.getMessage() for r in caplog.records)

    caplog.clear()
    with caplog.at_level("WARNING"):
        tj.build_trajectory_features(
            str(tmp_path / "tdump_*"), source_regions={"sw": (-3.0, 50.5, -1.5, 51.5)}
        )
    assert not [r for r in caplog.records if "overlap" in r.getMessage()]


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


def _tdump_text(starts, n_endpoints=3):
    """A tdump holding one trajectory per ``(lat, lon, height)`` start, all arriving
    2020-10-01 00:00 and drifting south-west an hour at a time."""
    head = ["     1     1", "    GDAS    20    10     1     0     0"]
    head.append(f"{len(starts):6d} BACKWARD OMEGA")
    head += [f"    20    10     1     0 {la:8.3f} {lo:8.3f} {h:7.1f}" for la, lo, h in starts]
    head.append("     1 PRESSURE")
    rows = []
    for k, (la, lo, h) in enumerate(starts, 1):
        for a in range(n_endpoints):
            when = pd.Timestamp("2020-10-01") - pd.Timedelta(hours=a)
            rows.append(
                f"{k:6d}     1 {when:%y} {when.month:5d} {when.day:5d} {when.hour:5d}     0"
                f"     0 {-a:7.1f} {la - 0.3 * a:8.3f} {lo - 0.6 * a:8.3f} {h + 10 * a:8.1f}"
                "    990.0"
            )
    return "\n".join(head + rows) + "\n"


def test_build_gives_each_start_height_its_own_columns(tmp_path):
    # One run started at three heights used to collapse to a single row: the
    # last height's, the other two dropped as "duplicates" of the receptor time.
    (tmp_path / "tdump_x").write_text(
        _tdump_text([(51.5, -0.1, 100.0), (51.5, -0.1, 500.0), (51.5, -0.1, 1500.0)])
    )
    out = tj.build_trajectory_features(str(tmp_path / "tdump_*"))
    assert len(out) == 1 and out.index.name == "date"
    assert {"traj_dist_km_100m", "traj_dist_km_500m", "traj_dist_km_1500m"} <= set(out.columns)
    assert out["traj_height_min_100m"].iloc[0] == 100.0
    assert out["traj_height_min_1500m"].iloc[0] == 1500.0
    assert not [c for c in out.columns if c == "traj_dist_km"]

    # One height, the columns keep their plain names.
    (tmp_path / "tdump_x").write_text(_tdump_text([(51.5, -0.1, 500.0)]))
    assert "traj_dist_km" in tj.build_trajectory_features(str(tmp_path / "tdump_*")).columns


def test_build_refuses_trajectories_from_several_places(tmp_path):
    (tmp_path / "tdump_a").write_text(_tdump_text([(51.5, -0.1, 500.0)]))
    (tmp_path / "tdump_b").write_text(_tdump_text([(10.0, 20.0, 500.0)]))
    with pytest.raises(ValueError, match=r"2 different places \(\(51.5, -0.1\), \(10, 20\)\)"):
        tj.build_trajectory_features(str(tmp_path / "tdump_*"))


def _hourly(start, hours, **cols):
    return pd.DataFrame({"date": pd.date_range(start, periods=hours, freq="h"), **cols})


def _feats(times, values, site=None):
    f = pd.DataFrame({"traj_x": values}, index=pd.DatetimeIndex(times, name="date"))
    if site is not None:
        f.insert(0, "site", site)
    return f


def test_join_takes_the_latest_trajectory_to_have_arrived():
    # 6-hourly trajectories at 00, 06 and 18 (12's is missing), an hourly panel
    # from 22:00 the day before to 23:00.
    panel = _hourly("2020-01-01 22:00", 26)
    feats = _feats(["2020-01-02 00:00", "2020-01-02 06:00", "2020-01-02 18:00"], [0.0, 6.0, 18.0])
    out = tj.join_trajectory_features(panel, feats)
    got = dict(zip(out["date"].dt.strftime("%d %H"), out["traj_x"], strict=True))
    assert np.isnan(got["01 23"])  # before the first trajectory: nothing to take
    assert got["02 00"] == got["02 05"] == 0.0
    assert got["02 04"] == 0.0  # not 06:00's, which has not arrived yet
    assert got["02 06"] == got["02 11"] == 6.0
    assert np.isnan(got["02 12"]) and np.isnan(got["02 17"])  # its trajectory is missing
    assert got["02 18"] == got["02 23"] == 18.0
    assert list(out.columns) == ["date", "traj_x"] and len(out) == len(panel)

    # The interval can be given; a single trajectory alone matches its own time.
    out = tj.join_trajectory_features(panel, feats, tolerance="2h")
    assert out["traj_x"].notna().sum() == 6  # 00-01, 06-07, 18-19
    one = tj.join_trajectory_features(panel, feats.iloc[:1])
    assert one["traj_x"].notna().sum() == 1


def test_join_matches_each_site_and_keeps_the_others_rows():
    panel = pd.concat(
        [_hourly("2020-01-01", 3, site="MY1"), _hourly("2020-01-01", 3, site="KC1")],
        ignore_index=True,
    )
    t = ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"]
    out = tj.join_trajectory_features(panel, _feats(t, [1.0, 2.0, 3.0], site="MY1"))
    assert out.loc[out.site == "MY1", "traj_x"].tolist() == [1.0, 2.0, 3.0]
    assert out.loc[out.site == "KC1", "traj_x"].isna().all()

    # The next site's batch fills its own rows and leaves MY1's; a batch sent
    # again replaces its columns rather than adding traj_x_x / traj_x_y.
    out = tj.join_trajectory_features(out, _feats(t, [7.0, 8.0, 9.0], site="KC1"))
    out = tj.join_trajectory_features(out, _feats(t, [4.0, 5.0, 6.0], site="MY1"))
    assert list(out.columns) == ["date", "site", "traj_x"]
    assert out["traj_x"].tolist() == [4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


def test_join_names_a_site_the_panel_does_not_have(caplog):
    panel = _hourly("2020-01-01", 3, site="MY1")
    with caplog.at_level("WARNING"):
        out = tj.join_trajectory_features(panel, _feats(["2020-01-01"], [1.0], site="London"))
    assert out["traj_x"].isna().all()
    assert any("site(s) London" in r.getMessage() for r in caplog.records)

    # Untagged features onto several sites: every site takes them, and it is said.
    caplog.clear()
    two = pd.concat([panel, _hourly("2020-01-01", 3, site="KC1")], ignore_index=True)
    with caplog.at_level("WARNING"):
        out = tj.join_trajectory_features(two, _feats(["2020-01-01"], [1.0]))
    assert out["traj_x"].notna().sum() == 2
    assert any("all 2 of the panel's sites" in r.getMessage() for r in caplog.records)


def test_join_without_a_site_column_in_the_panel():
    panel = _hourly("2020-01-01", 2)
    out = tj.join_trajectory_features(panel, _feats(["2020-01-01"], [1.0], site="MY1"))
    assert out["traj_x"].tolist()[0] == 1.0  # one site: the panel is that site
    two = pd.concat([_feats(["2020-01-01"], [1.0], "A"), _feats(["2020-01-01"], [2.0], "B")])
    with pytest.raises(ValueError, match=r"2 sites \(A, B\) but df has no 'site' column"):
        tj.join_trajectory_features(panel, two)


def test_join_keeps_rows_without_a_time_and_compares_times_as_utc():
    feats = _feats(["2020-07-01 00:00", "2020-07-01 06:00"], [0.0, 6.0])
    panel = pd.DataFrame(
        {
            # 07:00 in London in July is 06:00 UTC: it takes 06:00's trajectory.
            "date": pd.to_datetime(
                ["2020-07-01 07:00+01:00", None, "2020-07-01 05:00+00:00"], utc=True
            ),
            "pm25": [1.0, 2.0, 3.0],
        }
    )
    out = tj.join_trajectory_features(panel, feats)
    assert out["traj_x"].tolist()[0] == 6.0 and out["traj_x"].tolist()[2] == 0.0
    assert np.isnan(out["traj_x"].iloc[1])  # NaT: kept, with nothing joined
    assert out["date"].dt.tz is not None  # the panel's own column is left as it was

    # Features with a date column (as load_traj_my1 has) onto a date-indexed panel.
    indexed = _hourly("2020-07-01", 8).set_index("date")
    out = tj.join_trajectory_features(indexed, feats.reset_index())
    assert out["traj_x"].tolist() == [0.0] * 6 + [6.0] * 2


def test_join_refuses_ambiguous_features():
    panel = _hourly("2020-01-01", 2)
    twice = _feats(["2020-01-01", "2020-01-01"], [1.0, 2.0])
    with pytest.raises(ValueError, match="keep one per time"):
        tj.join_trajectory_features(panel, twice)
    with pytest.raises(ValueError, match="tolerance must be positive"):
        tj.join_trajectory_features(panel, twice.iloc[:1], tolerance="0h")
    with pytest.raises(KeyError, match="no 'date' column"):
        tj.join_trajectory_features(panel.drop(columns="date"), twice.iloc[:1])


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

    # Off the hour, the minutes go in too (HYSPLIT's optional fifth field).
    off = tj._control_text(
        pd.Timestamp("2020-10-17 12:30"),
        40.0,
        -90.0,
        500.0,
        24,
        ["/d/m"],
        "t",
        top_of_model=10000.0,
        vert_motion=0,
    )
    assert off.splitlines()[0] == "20 10 17 12 30"


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


def _fake_run(tmp_path, monkeypatch, times, met_names, hysplit=FAKE_HYTS, result=False, **kw):
    exe = tmp_path / "exec" / "hyts_std"
    exe.parent.mkdir(parents=True)
    exe.write_text(hysplit)
    exe.chmod(0o755)
    monkeypatch.setenv("FAKE_TDUMP", str(_write(tmp_path)))
    mets = []
    for n in met_names:
        (tmp_path / n).write_text("")
        mets.append(str(tmp_path / n))
    work = tmp_path / "work"
    out = tj.run_back_trajectories(
        times, 51.5, -0.13, met_files=mets, hysplit_exec=exe, work_dir=work, **kw
    )
    return out if result else work


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


def test_run_back_trajectories_skips_times_no_met_file_covers(tmp_path, monkeypatch, caplog):
    # A receptor time whose window no file can touch is not run at all: handing
    # hyts_std every file only made each such run fail, one by one (240 of them
    # for a January panel given June's GDAS1).
    times = [pd.Timestamp("2020-01-03 06:00"), pd.Timestamp("2021-06-01 00:00")]
    with caplog.at_level("WARNING"):
        work = _fake_run(tmp_path, monkeypatch, times, MET[:2], hours_back=24)
    assert _control_mets(work, "tdump_2020010306") == ["gdas1.jan20.w1"]
    assert not (work / "CONTROL_tdump_2021060100").exists()
    assert any("1 of 2 receptor times skipped" in r.getMessage() for r in caplog.records)

    with pytest.raises(RuntimeError, match="No met file covers"):
        _fake_run(tmp_path / "b", monkeypatch, times[1:], MET[:2], hours_back=24)


def test_run_back_trajectories_removes_a_stale_tdump(tmp_path, monkeypatch):
    # A reused work_dir held a tdump of the same name from an earlier receptor;
    # a run that then failed used to pass that file off as its own.
    work = tmp_path / "work"
    work.mkdir()
    (work / "tdump_2020011612").write_text(TDUMP)
    with pytest.raises(RuntimeError, match="No trajectories produced"):
        _fake_run(
            tmp_path,
            monkeypatch,
            [pd.Timestamp("2020-01-16 12:00")],
            MET,
            hysplit="#!/bin/sh\nexit 1\n",
        )
    assert not (work / "tdump_2020011612").exists()


def test_run_back_trajectories_runs_on_utc_and_keeps_minutes(tmp_path, monkeypatch):
    times = [
        pd.Timestamp("2020-01-16 13:30", tz="Europe/Paris"),  # 12:30 UTC
        pd.Timestamp("2020-01-16 12:30"),  # the same time again, naive: run once
        pd.Timestamp("2020-01-03 06:00", tz="UTC"),
        pd.NaT,
    ]
    work = _fake_run(tmp_path, monkeypatch, times, MET, hours_back=24)
    controls = sorted(f.name for f in work.glob("CONTROL_*"))
    assert controls == ["CONTROL_tdump_2020010306", "CONTROL_tdump_202001161230"]
    lines = (work / "CONTROL_tdump_202001161230").read_text().splitlines()
    assert lines[0] == "20 01 16 12 30"
    assert _control_mets(work, "tdump_202001161230") == ["gdas1.jan20.w3"]


def test_run_back_trajectories_stops_when_asked(tmp_path, monkeypatch, caplog):
    import threading
    import time

    stop = threading.Event()

    def stop_once_running():  # once the first run has started, and not before
        t_end = time.monotonic() + 20
        while not list((tmp_path / "work").glob("CONTROL_*")) and time.monotonic() < t_end:
            time.sleep(0.02)
        stop.set()

    threading.Thread(target=stop_once_running, daemon=True).start()
    t0 = time.monotonic()
    times = [pd.Timestamp("2020-01-16 12:00"), pd.Timestamp("2020-01-16 13:00")]
    with caplog.at_level("WARNING"):
        out = _fake_run(
            tmp_path,
            monkeypatch,
            times,
            MET,
            hysplit=FAKE_HYTS.replace('cp "$FAKE_TDUMP"', "exec sleep 30; #"),
            stop=stop,
            result=True,
        )
    assert time.monotonic() - t0 < 10  # the run in progress was killed
    assert out.empty
    assert len(list((tmp_path / "work").glob("CONTROL_*"))) == 1  # the second never started
    assert any("Stopped after 0 of 2" in r.getMessage() for r in caplog.records)


def test_missing_gdas1_files():
    ts = pd.Timestamp
    have = ["/met/gdas1.jan20.w1", "/met/gdas1.jan20.w2"]
    assert tj._missing_gdas1_files(have, ts("2020-01-05"), ts("2020-01-16")) == ["gdas1.jan20.w3"]
    assert tj._missing_gdas1_files(have, ts("2020-01-02"), ts("2020-01-09")) == []
    # 22:00 on 14 Jan interpolates towards w3's first record, as in _filter_met_files.
    assert tj._missing_gdas1_files(have, ts("2020-01-12"), ts("2020-01-14 22:00")) == [
        "gdas1.jan20.w3"
    ]
    assert (
        tj._missing_gdas1_files(have + ["/met/custom.arl"], ts("2020-01-05"), ts("2020-02-01"))
        is None
    )


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


# --------------------------------------------------- geography and weighting


def test_a_dateline_crossing_box_matches_its_two_sides():
    """``lon_min > lon_max`` is a Pacific box, not an empty one.

    The old ``lon >= min & lon <= max`` test matched nothing for such a box,
    so every residence fraction over a region that crosses 180° was 0 with no
    sign of the mistake.
    """
    traj = pd.DataFrame(
        {
            "age_h": [0.0, -1.0, -2.0, -3.0],
            "lat": [0.0, 0.0, 0.0, 0.0],
            "lon": [179.0, -179.0, 179.5, -175.0],
            "height": 500.0,
        }
    )
    pac = (170.0, -10.0, -170.0, 10.0)
    f = tj.trajectory_features(traj, source_regions={"pac": pac})
    assert f["traj_resid_pac"] == 1.0

    outside = traj.copy()
    outside.loc[0, "lon"] = 0.0
    f2 = tj.trajectory_features(outside, source_regions={"pac": pac})
    assert f2["traj_resid_pac"] == pytest.approx(3 / 4)  # even spacing: 3 of 4


def test_overlap_check_handles_a_dateline_crossing_box():
    from shapely.geometry import Polygon

    poly = Polygon([(-178.0, -5.0), (-172.0, -5.0), (-172.0, 5.0), (-178.0, 5.0)])
    regions = {
        "pac": (170.0, -10.0, -170.0, 10.0),
        "west_pacific": (-175.0, -5.0, -160.0, 5.0),  # shares -175..-170
        "poly": poly,  # inside pac's -180..-170 side
        "atlantic": (-40.0, -10.0, -10.0, 10.0),
    }
    assert tj._overlapping_regions(regions) == [
        ("pac", "west_pacific"),
        ("pac", "poly"),
        ("west_pacific", "poly"),  # -175..-172 of the polygon's span
    ]


def test_residence_fractions_weight_uneven_endpoint_spacing_by_time():
    """Counts overstate a long step: weight each endpoint by the time around it."""
    ages = np.array([0.0, -1.0, -3.0, -6.0, -9.0, -12.0, -15.0, -18.0, -21.0, -24.0, -27.0, -30.0])
    lat = np.array([30.0, 30.0] + [45.0] * 10)
    traj = pd.DataFrame({"age_h": ages, "lat": lat, "lon": 110.0, "height": 500.0})
    f = tj.trajectory_features(traj, source_regions={"near": (100.0, 20.0, 120.0, 35.0)})
    # Inside until the crossing between -1 and -3 h: 2 h of the 30-h path.
    # (Counting endpoints would say 2 of 12 = 0.17.)
    assert f["traj_resid_near"] == pytest.approx(2.0 / 30.0, rel=0.02)


def test_join_uses_each_sites_own_release_interval():
    """A global modal interval under-covers a site released less often.

    Site A is 3-hourly, site B 6-hourly; the commonest step across both is
    3 h, which left B's 03:00-05:00 window empty. Each site's own interval
    lets every hour of its release window take that window's trajectory.
    """
    feat = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01 00:00",
                    "2020-01-01 03:00",
                    "2020-01-01 06:00",
                    "2020-01-01 00:00",
                    "2020-01-01 06:00",
                ]
            ),
            "site": ["A", "A", "A", "B", "B"],
            "v": [1, 2, 3, 10, 11],
        }
    )
    panel = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=8, freq="h"), "site": "B"})
    out = tj.join_trajectory_features(panel, feat)
    assert out["v"].tolist() == [10.0] * 6 + [11.0] * 2


# --------------------------------------------------- Layer-1 physics & multi-height


def test_endpoint_weights_blh_decay_and_causal_scavenging():
    """Verify BLH decoupling, 1/BLH dilution, decay, and causal rainfall ordering."""
    # Trajectory ordered arbitrarily (shuffled rows) to prove alignment back to input order.
    traj = pd.DataFrame(
        {
            "age_h": [-2.0, 0.0, -3.0, -1.0],
            "lat": [51.0, 51.5, 50.8, 51.2],
            "lon": [-2.5, -0.1, -3.2, -1.2],
            "height": [200.0, 50.0, 2000.0, 250.0],  # -3h is 2000m > BLH=800 -> decoupled
            "blh": [500.0, 600.0, 800.0, 1000.0],
            "rainfall": [10.0, 0.0, 5.0, 0.0],  # heavy rain at -2h; zero between -2h and 0
        }
    )
    w = tj.trajectory_endpoint_weights(traj, decay_hours=10.0, scavenging_coeff=0.2)
    # Row 1 (age 0) excluded -> 0; Row 2 (age -3, height 2000 > 800) decoupled -> 0.
    assert w[1] == 0.0
    assert w[2] == 0.0
    # Row 3 (age -1, height 250 <= BLH 1000, rain downstream=0): dt=1 * (1000/1000) * exp(-1/10)
    assert w[3] == pytest.approx(np.exp(-0.1))
    # Row 0 (age -2, BLH 500, rain strictly AFTER -2h is only age -1's rainfall=0):
    assert w[0] == pytest.approx(2.0 * np.exp(-0.2))

    # Now put rain strictly downstream at age -1h: age -2h is attenuated, age -1h is NOT.
    traj_down = traj.copy()
    traj_down.loc[3, "rainfall"] = 4.0  # downstream of -2h, at -1h
    w_down = tj.trajectory_endpoint_weights(traj_down, decay_hours=10.0, scavenging_coeff=0.2)
    assert w_down[3] == pytest.approx(w[3])  # age -1 not attenuated by rain at -1 itself
    assert w_down[0] == pytest.approx(w[0] * np.exp(-0.2 * 4.0))

    # Fallback when blh column absent (like bundled MY1 tdumps)
    no_blh = traj.drop(columns=["blh"])
    w_fb = tj.trajectory_endpoint_weights(no_blh, fallback_height_m=1500.0)
    assert w_fb[1] == 0.0 and w_fb[2] == 0.0
    assert w_fb[0] == 1.0 and w_fb[3] == 1.0


def test_trajectory_features_physical_columns_and_horizons(tmp_path):
    df = tj.read_trajectory_tdump(_write(tmp_path))
    box = {"sw_box": (-3.0, 50.5, -1.5, 51.5)}
    f = tj.trajectory_features(df, source_regions=box, horizons_h=((0, 1), (1, 2)))

    assert {"traj_inflow_sin", "traj_inflow_cos", "traj_straightness"} <= set(f)
    assert {"traj_mean_bearing_sin", "traj_mean_bearing_cos", "traj_blh_contact_frac"} <= set(f)
    assert f["traj_inflow_sin"] ** 2 + f["traj_inflow_cos"] ** 2 == pytest.approx(1.0)
    assert f["traj_mean_bearing_sin"] ** 2 + f["traj_mean_bearing_cos"] ** 2 == pytest.approx(1.0)
    assert 0.0 <= f["traj_straightness"] <= 1.0
    assert f["traj_rain_mm"] == pytest.approx(1.5)
    assert f["traj_blh_contact_frac"] == pytest.approx(1.0)
    # Footprint matches 1-km-BLH equivalent weight inside (-3..-1.5, 50.5..51.5) -> origin at -2h (BLH 500)
    assert f["traj_footprint_sw_box"] == pytest.approx(2.0)

    # Horizon slices conserve path length
    assert f["traj_pathlen_km_0_1h"] + f["traj_pathlen_km_1_2h"] == pytest.approx(
        f["traj_pathlen_km"]
    )
    assert f["traj_footprint_sw_box_0_1h"] == 0.0
    assert f["traj_footprint_sw_box_1_2h"] == pytest.approx(2.0)

    # QC classifier covers plain & multi-height suffixed QC columns
    assert tj.is_trajectory_qc_column("traj_n_endpoints")
    assert tj.is_trajectory_qc_column("traj_age_max_h_500m")
    assert not tj.is_trajectory_qc_column("traj_dist_km")
    assert not tj.is_trajectory_qc_column("traj_footprint_sw_box")


def test_read_trajectory_endpoints_and_parallel_multi_height_hysplit(tmp_path):
    (tmp_path / "tdump_a").write_text(_tdump_text([(51.5, -0.1, 100.0), (51.5, -0.1, 500.0)]))
    ep = tj.read_trajectory_endpoints(str(tmp_path / "tdump_*"), site="MY1")
    assert len(ep) == 6
    assert set(ep["start_height"]) == {100.0, 500.0}
    assert (ep["site"] == "MY1").all()
    assert tj.read_trajectory_endpoints(str(tmp_path / "tdump_*"), min_hours=24).empty

    # Multi-height + parallel execution (n_workers=2)
    exe = tmp_path / "exec" / "hyts_std"
    exe.parent.mkdir()
    exe.write_text(
        f"""#!{sys.executable}
from datetime import datetime, timedelta
from pathlib import Path

lines = Path("CONTROL").read_text().splitlines()
parts = [int(x) for x in lines[0].split()[:4]]
yr = 2000 + parts[0] if parts[0] < 70 else 1900 + parts[0]
t0 = datetime(yr, parts[1], parts[2], parts[3])
n_h = int(lines[1])
starts = [[float(v) for v in lines[2 + i].split()[:3]] for i in range(n_h)]
out_name = lines[-1].strip()
out = [
    "     1     1",
    f"    GDAS    {{parts[0]:2d}}    {{parts[1]:2d}}     {{parts[2]:d}}     {{parts[3]:d}}     0",
    f"     {{n_h}} BACKWARD OMEGA",
]
for la, lo, hh in starts:
    out.append(
        f"     1    {{parts[0]:2d}}    {{parts[1]:2d}}    {{parts[2]:2d}}    {{parts[3]:2d}}"
        f"   {{la:.3f}}   {{lo:.3f}}    {{hh:.1f}}"
    )
out.append("     5 PRESSURE RAINFALL MIXDEPTH RELHUMID AIR_TEMP")
deltas = [
    (0.0, 51.520, -0.130, 0.0, 995.0, 0.0, 800.0, 70.0, 285.0),
    (-1.0, 51.300, -1.200, 200.0, 980.0, 0.5, 600.0, 75.0, 283.0),
    (-2.0, 51.000, -2.500, 400.0, 970.0, 1.0, 500.0, 80.0, 281.0),
]
for tid, (_, _, h0) in enumerate(starts, start=1):
    for age_h, la, lo, dh, pres, rain, blh, rh, tmp in deltas:
        ts = t0 + timedelta(hours=int(age_h))
        yy = ts.year % 100
        out.append(
            f"     {{tid}}     1    {{yy:2d}}    {{ts.month:2d}}    {{ts.day:2d}}    {{ts.hour:2d}}"
            f"     0     0.0    {{age_h:4.1f}}   {{la:.3f}}   {{lo:.3f}}    {{h0 + dh:.1f}}"
            f"    {{pres:.1f}}    {{rain:.1f}}   {{blh:.1f}}     {{rh:.1f}}    {{tmp:.1f}}"
        )
Path(out_name).write_text("\\n".join(out) + "\\n")
"""
    )
    exe.chmod(0o755)
    mets = []
    for n in MET[:3]:
        (tmp_path / n).write_text("")
        mets.append(str(tmp_path / n))
    work = tmp_path / "par_work"
    times = [
        pd.Timestamp("2020-01-03 06:00"),
        pd.Timestamp("2020-01-10 06:00"),
        pd.Timestamp("2020-01-16 12:00"),
    ]
    out = tj.run_back_trajectories(
        times,
        51.5,
        -0.13,
        met_files=mets,
        hysplit_exec=exe,
        heights_m=(100.0, 500.0),
        work_dir=work,
        n_workers=2,
        source_regions={"sw": (-3.0, 50.5, -1.5, 51.5)},
    )
    assert len(out) == 3
    assert "traj_footprint_sw_100m" in out.columns
    assert "traj_footprint_sw_500m" in out.columns
    ctl_lines = (
        (work / "CONTROL_tdump_2020010306").read_text()
        if (work / "CONTROL_tdump_2020010306").exists()
        else ""
    )
    # Temporary .run_* dirs were cleaned up and tdump_* moved cleanly into work_dir
    assert not list(work.glob(".run_*"))
    assert len(list(work.glob("tdump_*"))) == 3
    assert ctl_lines == ""  # isolated per worker subdir
    ctrl = tj._control_text(
        times[0],
        51.5,
        -0.13,
        (100.0, 500.0),
        72,
        mets[:1],
        "tdump_x",
        top_of_model=10000.0,
        vert_motion=0,
    )
    assert ctrl.splitlines()[1:4] == ["2", "51.5000 -0.1300 100.0", "51.5000 -0.1300 500.0"]
