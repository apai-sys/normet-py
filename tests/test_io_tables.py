"""Time series from table files (normet.io.tables): the files read, the times found and
read however they are written, and a local clock put on UTC."""

from __future__ import annotations

import gzip
import importlib.util

import pandas as pd
import pytest

from normet.io.tables import (
    clock_to_utc,
    find_dates,
    on_clock,
    parse_dates,
    read_table,
    read_time_series,
    utc_to_clock,
)


def _times(df):
    return [str(t) for t in df["date"]]


@pytest.mark.parametrize(
    "text, first",
    [
        ("date,NO2\n2020-01-31 13:00,1\n2020-02-01 14:00,2\n", "2020-01-31 13:00:00"),
        ("Date;NO2\n31/01/2020 13:00;1,5\n01/02/2020 14:00;2,5\n", "2020-01-31 13:00:00"),
        ("datetime,NO2\n01/31/2020 13:00,1\n02/01/2020 14:00,2\n", "2020-01-31 13:00:00"),
        ("Time\tNO2\n31.01.2020 13:00\t1\n01.02.2020 14:00\t2\n", "2020-01-31 13:00:00"),
        ("日期,NO2\n2020年1月31日 13时,1\n2020年2月1日 14时,2\n", "2020-01-31 13:00:00"),
        ("date,time,NO2\n2020-01-31,13:00,1\n2020-02-01,14:00,2\n", "2020-01-31 13:00:00"),
        ("year,month,day,hour,NO2\n2020,1,31,13,1\n2020,2,1,14,2\n", "2020-01-31 13:00:00"),
        ("timestamp,NO2\n1580475600,1\n1580565600,2\n", "2020-01-31 13:00:00"),
        ("date,NO2\n2020013113,1\n2020020114,2\n", "2020-01-31 13:00:00"),
    ],
)
def test_the_times_are_found_and_read_however_written(tmp_path, text, first):
    path = tmp_path / "site.csv"
    path.write_text(text, encoding="utf-8")
    df = read_time_series(path)
    assert list(df.columns) == ["date", "NO2"]
    assert _times(df)[0] == first
    assert df["NO2"].dtype.kind == "f" or df["NO2"].dtype.kind == "i"
    assert df.attrs["normet_dates"]


def test_a_decimal_comma_and_a_semicolon_are_read_from_the_file(tmp_path):
    path = tmp_path / "eu.csv"
    path.write_text("date;NO2\n2020-01-01 00:00;1,5\n2020-01-01 01:00;2,25\n", encoding="cp1252")
    assert read_table(path)["NO2"].tolist() == [1.5, 2.25]


def _installed(*modules: str) -> bool:
    return any(importlib.util.find_spec(m) is not None for m in modules)


def test_excel_parquet_and_gzip_are_read(tmp_path):
    frame = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=3, freq="h"), "NO2": [1.0, 2, 3]}
    )
    with gzip.open(tmp_path / "a.csv.gz", "wt") as fh:
        frame.to_csv(fh, index=False)
    names = ["a.csv.gz"]
    # Parquet and Excel need engines of their own, not every install has them.
    if _installed("pyarrow", "fastparquet"):
        frame.to_parquet(tmp_path / "a.parquet")
        names.append("a.parquet")
    if _installed("openpyxl"):
        frame.to_excel(tmp_path / "a.xlsx", index=False)
        names.append("a.xlsx")
    for name in names:
        df = read_time_series(tmp_path / name)
        assert _times(df) == [str(t) for t in frame["date"]], name


def test_an_ambiguous_order_is_said_and_can_be_chosen():
    dates, how = parse_dates(pd.Series(["01/02/2020", "03/04/2020"]))
    assert str(dates[0].date()) == "2020-01-02" and "does not tell" in how
    dates, _ = parse_dates(pd.Series(["01/02/2020", "03/04/2020"]), dayfirst=True)
    assert str(dates[0].date()) == "2020-02-01"


def test_a_table_keeps_its_own_clock_and_utc_is_put_on_it():
    local = pd.Series(pd.to_datetime(["2020-01-01 08:00"]))
    assert on_clock(local, "Asia/Shanghai").equals(local)  # kept as it is written
    assert str(clock_to_utc(local, "Asia/Shanghai")[0]) == "2020-01-01 00:00:00"
    era5 = pd.Series(pd.to_datetime(["2020-01-01 00:00"]))
    assert str(utc_to_clock(era5, "Asia/Shanghai")[0]) == "2020-01-01 08:00:00"
    # Summer time: 01:30 on the day the clocks go forward does not exist in London.
    london = clock_to_utc(
        pd.Series(pd.to_datetime(["2020-03-29 00:30", "2020-03-29 02:30"])), "Europe/London"
    )
    assert [str(t) for t in london] == ["2020-03-29 00:30:00", "2020-03-29 01:30:00"]
    assert clock_to_utc(local, "UTC").equals(local)


def test_times_with_their_own_offset_are_put_on_the_clock(tmp_path):
    path = tmp_path / "bst.csv"
    path.write_text("date,NO2\n2020-03-29T00:00+00:00,1\n2020-03-29T02:00+01:00,2\n")
    df = read_time_series(path)
    assert df["date"].dt.tz is None and df["date"].dt.hour.tolist() == [0, 1]
    df = read_time_series(path, zone="Asia/Shanghai")
    assert df["date"].dt.hour.tolist() == [8, 9] and df.attrs["normet_zone"] == "Asia/Shanghai"


def test_no_times_is_said_in_words():
    with pytest.raises(ValueError, match="No column of times"):
        find_dates(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
