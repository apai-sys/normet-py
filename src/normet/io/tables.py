"""A time series from a table file, whatever form the file and its times take.

The window's *Open Data…* reads with :func:`read_time_series`, and a session script
repeats the same call.

    TABLE_SUFFIXES             the files read: text, Excel, Parquet, Feather, JSON
    read_table(path)           the file as a DataFrame
    find_dates(df)             its times in a column named 'date', and how they were read
    parse_dates(values)        one column of times, in whichever form they are written
    on_clock(dates, zone)      times on one clock: a table's own (local time, or UTC)
    clock_to_utc / utc_to_clock  between that clock and UTC
    read_time_series(path)     a table ready for the window, on its clock

Times. A table's times may be one column or several (a date and a time, or a year,
month, day and hour), written as text in any of the usual orders (2020-01-31,
31/01/2020, 01/31/2020, 31.01.2020, 2020年1月31日), as numbers (Unix seconds or
milliseconds, Excel's serial days, 20200131 or 2020013113), or with their own UTC
offset (2020-03-29T02:00+01:00).

Clocks. A table is analysed, and its figures drawn, on the clock it is kept on -- a
site's local time (Asia/Shanghai, Europe/London) or UTC: the hour and the weekday a
model learns from are then the site's, so a rush hour stays at its hour through summer
time. Times written with their own offset are put on that clock. The meteorology joined
to a table -- ERA5, Open-Meteo, HYSPLIT -- is on UTC, and is put on the table's clock as
it is joined (:func:`utc_to_clock`); what it is fetched for, the other way
(:func:`clock_to_utc`).
"""

from __future__ import annotations

import csv
import io
import os
import re
from typing import Any

import numpy as np
import pandas as pd

from ..utils._time import to_datetime_coerced

#: Spreadsheets, read from their first sheet.
EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")
#: Delimited text, plain or gzipped.
TEXT_SUFFIXES = (".csv", ".txt", ".tsv", ".dat")
#: Columnar files.
BINARY_SUFFIXES = (".parquet", ".feather", ".json")
#: Every file :func:`read_table` reads.
TABLE_SUFFIXES = (
    TEXT_SUFFIXES + tuple(s + ".gz" for s in TEXT_SUFFIXES) + EXCEL_SUFFIXES + BINARY_SUFFIXES
)

#: Headers a table's times go under, compared without case, spaces or punctuation.
DATE_NAMES = (
    "date",
    "datetime",
    "datetimeutc",
    "dateutc",
    "timestamp",
    "time",
    "times",
    "dt",
    "dateandtime",
    "datetimelocal",
    "starttime",
    "start",
    "readingdatetime",
    "日期",
    "时间",
    "日期时间",
)
#: A separate time-of-day column beside a date column.
TIME_NAMES = ("time", "hour", "hourofday", "timeofday", "时间", "小时")
#: Calendar parts in columns of their own.
PART_NAMES = {
    "year": ("year", "yr", "年"),
    "month": ("month", "mon", "月"),
    "day": ("day", "dayofmonth", "date", "日"),
    "hour": ("hour", "hr", "时", "小时"),
}


def _key(name: Any) -> str:
    """A header compared without case, spaces or punctuation: "Date/Time (UTC)" is
    "datetimeutc"."""
    return re.sub(r"[\s_\-/().:]+", "", str(name)).lower()


def path_suffix(path: Any) -> str:
    """The suffix a file is read by: ".csv.gz" for a gzipped CSV, else the last one."""
    name = str(path).lower()
    for s in TABLE_SUFFIXES:
        if name.endswith(s):
            return s
    return os.path.splitext(name)[1]


def read_table(path: Any, **kwargs: Any) -> pd.DataFrame:
    """A table file as a DataFrame.

    An Excel workbook (.xlsx, .xlsm, .xls) is read from its first sheet; Parquet, Feather
    and JSON (records or columns) as pandas reads them. Anything else is delimited text,
    plain or gzipped, which instruments and spreadsheets write several ways: commas,
    semicolons, tabs or bars between fields, a decimal comma where the fields are not
    comma-separated, UTF-8 with or without Excel's byte-order mark, or a Windows code page
    for a "µg" or "°C" in the header. Each is settled from the file itself. Rows that hold
    nothing in any field (lines of separators alone) are dropped. ``kwargs`` go on to the
    pandas reader.
    """
    suffix = path_suffix(path)
    if suffix in EXCEL_SUFFIXES:
        keep = {k: v for k, v in kwargs.items() if k in ("header", "usecols", "nrows", "skiprows")}
        try:
            df = pd.read_excel(path, sheet_name=0, **keep)
        except ImportError as exc:
            need = "xlrd" if suffix == ".xls" else "openpyxl"
            raise ImportError(
                f"Reading {os.path.basename(str(path))} needs the {need} package "
                f"(pip install {need}), or save the sheet as .xlsx or .csv."
            ) from exc
        df.columns = [str(c) for c in df.columns]
        return _without_blank_rows(df)
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix == ".feather":
        return pd.read_feather(path, **kwargs)
    if suffix == ".json":
        return _without_blank_rows(pd.read_json(path, **kwargs))
    import gzip

    with open(path, "rb") as fh:
        blob = fh.read()
    if suffix.endswith(".gz") or blob[:2] == b"\x1f\x8b":
        blob = gzip.decompress(blob)
    text = blob.decode("latin-1")
    for encoding in ("utf-8-sig", "gb18030", "cp1252"):
        try:
            text = blob.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    sample = "\n".join(text.splitlines()[:50])
    if "sep" not in kwargs and "delimiter" not in kwargs:
        try:
            kwargs["sep"] = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            kwargs["sep"] = r"\s+" if suffix.startswith((".dat", ".txt")) else ","
    if "decimal" not in kwargs and kwargs.get("sep") != ",":
        body = sample.splitlines()[1:]
        commas = sum(bool(re.search(r"\d,\d", line)) for line in body)
        points = sum(bool(re.search(r"\d\.\d", line)) for line in body)
        if commas > points:
            kwargs["decimal"] = ","
    return _without_blank_rows(pd.read_csv(io.StringIO(text), **kwargs))


def _without_blank_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    blank = df.apply(lambda col: col.isna() | col.astype(str).str.strip().eq("")).all(axis=1)
    return df.loc[~blank.to_numpy()].reset_index(drop=True) if blank.any() else df


# -- times ----------------------------------------------------------------------------------

_NUMERIC_DMY = re.compile(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})")
_CJK_DATE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?")
_CJK_TIME = re.compile(r"(\d{1,2})\s*[时點点]\s*(?:(\d{1,2})\s*分)?\s*(?:(\d{1,2})\s*秒)?")


def _day_first(text: pd.Series) -> bool | None:
    """Whether dates written as two numbers then the year put the day first: True where
    one of the first numbers passes 12, False where one of the second does, None where
    neither does (01/02/2020 alone could be either) -- and None for text not so written."""
    found = text.str.extract(_NUMERIC_DMY)
    found = found.dropna()
    if found.empty:
        return None
    first, second = found[0].astype(int), found[1].astype(int)
    if (first > 12).any():
        return True
    if (second > 12).any():
        return False
    return None


def _from_cjk(text: pd.Series) -> pd.Series:
    """2020年1月31日 13时05分 as 2020-01-31 13:05."""

    def one(s: str) -> str:
        s = _CJK_DATE.sub(lambda m: f"{m[1]}-{int(m[2]):02d}-{int(m[3]):02d} ", s)
        s = _CJK_TIME.sub(lambda m: f"{int(m[1]):02d}:{int(m[2] or 0):02d}:{int(m[3] or 0):02d}", s)
        return s.strip()

    return text.map(one)


def parse_dates(values: Any, *, dayfirst: bool | None = None) -> tuple[pd.Series, str]:
    """One column of times as datetimes, and how they were read, in words (short parts
    joined with " · ", each translated on its own in the window).

    Numbers are read by their size: Unix seconds (1e9 to 1e10) or milliseconds (1e12 to
    1e13), Excel serial days (20 000 to 80 000, from 1899-12-30), 20200131 and
    2020013113 as dates and hours. Text is read value by value; dates written as two
    numbers and a year (31/01/2020) take the order the column itself shows -- the day
    first where a first number passes 12, the month where a second does -- or
    ``dayfirst`` where it shows neither (by default: the day first after a dot,
    31.01.2020, the month first otherwise). Times with their own UTC offset keep it.
    """
    s = pd.Series(values)
    if pd.api.types.is_datetime64_any_dtype(s):
        return s, "as stored"
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        if lo >= 19000101 and hi <= 21001231:
            return pd.to_datetime(
                v.astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
            ), "as YYYYMMDD"
        if lo >= 1900010100 and hi <= 2100123123:
            return pd.to_datetime(
                v.astype("Int64").astype(str), format="%Y%m%d%H", errors="coerce"
            ), "as YYYYMMDDHH"
        if lo >= 1e12 and hi < 1e13:
            return pd.to_datetime(v, unit="ms", errors="coerce"), "as Unix milliseconds"
        if lo >= 1e8 and hi < 1e10:
            return pd.to_datetime(v, unit="s", errors="coerce"), "as Unix seconds"
        if lo >= 2e4 and hi <= 8e4:
            return (
                pd.to_datetime(v, unit="D", origin="1899-12-30", errors="coerce"),
                "as Excel serial days",
            )
        raise ValueError(
            f"the numbers {lo:g} to {hi:g} are not times normet can read "
            "(Unix seconds, Excel serial days, or 20200131-style dates)"
        )
    text = s.astype("string").str.strip()
    how = "as written"
    if text.str.contains("年", na=False).any():
        text = _from_cjk(text)
        how = "as written (年月日)"
    first = _day_first(text)
    if first is None:
        if dayfirst is None:
            dayfirst = bool(text.str.match(r"^\s*\d{1,2}\.\d{1,2}\.\d{2,4}").any())
        first = dayfirst
        if text.str.match(_NUMERIC_DMY).any():
            how = "day first (31/01)" if first else "month first (01/31)"
            how += " · the column does not tell the order"
    elif first:
        how = "day first (31/01)"
    else:
        how = "month first (01/31)"
    return to_datetime_coerced(text, dayfirst=first), how


def _match(columns: list[Any], names: tuple[str, ...]) -> Any:
    keys = {_key(n) for n in names}
    return next((c for c in columns if _key(c) in keys), None)


def find_dates(df: pd.DataFrame, *, dayfirst: bool | None = None) -> tuple[pd.DataFrame, str]:
    """``df`` with its times parsed in a column named 'date', first; and how they were
    found and read, in words.

    The times are a column named as times are (date, datetime, timestamp, time, 日期, ...)
    -- joined with a time-of-day column beside it where the dates have no hours -- or
    year, month, day (and hour) columns, joined; else the first text column that reads as
    times. Raises ValueError, in words for the user, where none does.
    """
    cols = list(df.columns)
    out = df.copy()
    found = _match(cols, DATE_NAMES)
    parts = {k: _match(cols, v) for k, v in PART_NAMES.items()}
    if found is not None and _key(found) in {_key(n) for n in PART_NAMES["day"]} and parts["year"]:
        found = None  # "day" beside "year" and "month" is a part, not the times
    if found is None and parts["year"] is not None and parts["month"] is not None:
        frame = pd.DataFrame(
            {
                "year": pd.to_numeric(df[parts["year"]], errors="coerce"),
                "month": pd.to_numeric(df[parts["month"]], errors="coerce"),
                "day": pd.to_numeric(df[parts["day"]], errors="coerce") if parts["day"] else 1,
                "hour": pd.to_numeric(df[parts["hour"]], errors="coerce") if parts["hour"] else 0,
            }
        )
        dates = pd.to_datetime(frame, errors="coerce")
        used = [c for c in parts.values() if c is not None]
        out = out.drop(columns=used)
        out.insert(0, "date", dates)
        return out, "joined from " + ", ".join(map(str, used))
    if found is None:
        for c in cols:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                found = c
                break
            if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
                trial, _ = parse_dates(df[c].head(200), dayfirst=dayfirst)
                if trial.notna().mean() >= 0.9:
                    found = c
                    break
    if found is None:
        raise ValueError(
            "No column of times was found: name it 'date' (or datetime, timestamp, time), "
            "or give year, month, day and hour columns."
        )
    dates, how = parse_dates(df[found], dayfirst=dayfirst)
    said = f"from '{found}' · {how}" if str(found) != "date" else how
    clock = _match([c for c in cols if c != found], TIME_NAMES)
    if clock is not None and dates.notna().any():
        times = dates.dropna()
        if bool((times == times.dt.normalize()).all()):
            if pd.api.types.is_numeric_dtype(df[clock]):
                offset = pd.to_timedelta(pd.to_numeric(df[clock], errors="coerce"), unit="h")
            else:
                # 13:00 as 13:00:00: pandas reads a time of day only with its seconds
                clock_text = df[clock].astype("string").str.strip()
                clock_text = clock_text.str.replace(r"^(\d{1,2}:\d{2})$", r"\1:00", regex=True)
                offset = pd.to_timedelta(clock_text, errors="coerce")
            dates = dates + offset
            out = out.drop(columns=[clock])
            said += f" · hours from '{clock}'"
    out = out.drop(columns=[found])
    out.insert(0, "date", dates)
    return out, said


def _utc_zone(zone: str | None) -> bool:
    return not zone or zone.upper() == "UTC"


def on_clock(dates: Any, zone: str | None) -> pd.Series:
    """Times on ``zone``'s clock (an IANA name: Asia/Shanghai, Europe/London; None for
    UTC), without a zone of their own: times written with their own offset are converted
    to it; times without one are taken to be on it already, and left as they are."""
    s = pd.Series(dates)
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = to_datetime_coerced(s)
    if getattr(s.dt, "tz", None) is None:
        return s
    return s.dt.tz_convert("UTC" if _utc_zone(zone) else zone).dt.tz_localize(None)


def _localized(s: pd.Series, zone: str) -> pd.Series:
    """Naive times taken to be on ``zone``'s clock: an hour a change to summer time skips
    moved forward, one it repeats taken in order (NaT where the order cannot tell)."""
    try:
        return s.dt.tz_localize(zone, ambiguous="infer", nonexistent="shift_forward")
    except Exception:  # noqa: BLE001 - a repeated hour the order cannot settle
        return s.dt.tz_localize(zone, ambiguous="NaT", nonexistent="shift_forward")


def clock_to_utc(dates: Any, zone: str | None) -> pd.Series:
    """Times on ``zone``'s clock, on UTC (naive both): what HYSPLIT, ERA5 and Open-Meteo
    are asked for."""
    s = on_clock(dates, zone)
    if _utc_zone(zone):
        return s
    return _localized(s, str(zone)).dt.tz_convert("UTC").dt.tz_localize(None)


def utc_to_clock(dates: Any, zone: str | None) -> pd.Series:
    """Times on UTC (naive, or with their offset), on ``zone``'s clock: meteorology and
    trajectories fetched on UTC, joined to a table kept on local time."""
    s = pd.Series(dates)
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = to_datetime_coerced(s)
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC" if _utc_zone(zone) else zone).dt.tz_localize(None)


def read_time_series(
    path: Any, zone: str | None = None, *, dayfirst: bool | None = None
) -> pd.DataFrame:
    """A table file with its times in a column named 'date', on ``zone``'s clock (the
    clock the analysis and its figures keep: the site's local time, or UTC):
    :func:`read_table`, :func:`find_dates`, then :func:`on_clock`. How the times were
    found and read is in ``df.attrs["normet_dates"]``, the clock in
    ``df.attrs["normet_zone"]``."""
    df, how = find_dates(read_table(path), dayfirst=dayfirst)
    if getattr(df["date"].dt, "tz", None) is not None:
        how += " · UTC offsets put on " + ("UTC" if _utc_zone(zone) else str(zone))
    df["date"] = on_clock(df["date"], zone)
    df.attrs["normet_dates"] = how
    df.attrs["normet_zone"] = "UTC" if _utc_zone(zone) else zone
    return df
