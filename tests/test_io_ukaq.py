"""UK air-quality network adapter — offline tests.

Every test monkeypatches ``_read_rdata``, so nothing here touches the
network. That is the only I/O boundary in the module: everything else is
frame assembly, filtering and joining, which is what these exercise.
"""

import numpy as np
import pandas as pd
import pytest
from normet.io import ukaq
from normet.io.ukaq import (
    UKAQ_SOURCES,
    _check_source,
    _to_datetime,
    fetch_ukaq_measurements,
    list_ukaq_stations,
)


@pytest.fixture(autouse=True)
def _clear_meta_cache():
    """The metadata cache is module-level; stop it leaking between tests."""
    ukaq._META_CACHE.clear()
    yield
    ukaq._META_CACHE.clear()


def _hourly(code: str, year: int, n: int = 24) -> pd.DataFrame:
    """A stand-in for one station-year archive frame.

    ``date`` is a float POSIXct, matching what ``rdata`` actually hands
    back — it has no POSIXct constructor and returns the raw numeric.
    """
    start = pd.Timestamp(f"{year}-01-01", tz="UTC").value // 10**9
    return pd.DataFrame(
        {
            "date": np.arange(start, start + n * 3600, 3600, dtype=float),
            "NO": np.linspace(1, 2, n),
            "NO2": np.linspace(10, 20, n),
            "NOXasNO2": np.linspace(11, 22, n),
            "PM10": np.linspace(5, 6, n),
            "site": [f"Site {code}"] * n,
            "code": [code] * n,
        }
    )


_META = pd.DataFrame(
    {
        "site_id": ["AAA", "AAA", "BBB", "CCC"],
        "site_name": ["Site AAA", "Site AAA", "Site BBB", "Site CCC"],
        "location_type": [
            "Urban Traffic",
            "Urban Traffic",
            "Rural Background",
            "Urban Background",
        ],
        "latitude": [53.0, 53.0, 54.0, 55.0],
        "longitude": [-2.0, -2.0, -3.0, -1.0],
        "parameter": ["NOx", "PM10", "NOx", "NOx"],
        "Parameter_name": ["Nitrogen oxides", "PM10", "Nitrogen oxides", "Nitrogen oxides"],
        "start_date": ["2010-01-01"] * 4,
        "end_date": [None] * 4,
        "ratified_to": ["2023-12-31"] * 4,
    }
)


def _patch(monkeypatch, *, present=("AAA_2020",), meta=_META):
    """Route ``_read_rdata`` to fixtures; unknown archives raise, as 404 does."""

    def fake(url, **kwargs):
        if url.endswith("_metadata.RData"):
            return {"metadata": meta.copy()}
        key = url.rsplit("/", 1)[-1].replace(".RData", "")
        if key not in present:
            raise RuntimeError(f"404 for {key}")
        code, year = key.rsplit("_", 1)
        return {
            key: _hourly(code, int(year)),
            f"{key}_daily_mean": _hourly(code, int(year), n=1),
            f"{key}_24hour_mean": _hourly(code, int(year), n=1),
        }

    monkeypatch.setattr(ukaq, "_read_rdata", fake)


# ---- constants and helpers ----


def test_sources_well_formed():
    assert set(UKAQ_SOURCES) == {"aurn", "aqe", "saqn", "waqn", "ni", "local"}
    for src, urls in UKAQ_SOURCES.items():
        assert urls["data"].startswith("https://"), src
        assert urls["data"].endswith("/"), f"{src} data base must end in / to join file names"
        assert urls["meta"].endswith("_metadata.RData"), src


def test_check_source_normalises_and_rejects():
    assert _check_source("AURN") == "aurn"
    assert _check_source("  saqn ") == "saqn"
    with pytest.raises(ValueError, match="Unknown source"):
        _check_source("kcl")


def test_to_datetime_from_posixct_numeric():
    out = _to_datetime(pd.Series([0.0, 3600.0]))
    assert str(out.dt.tz) == "UTC"
    assert out.iloc[0] == pd.Timestamp("1970-01-01", tz="UTC")
    assert out.iloc[1] == pd.Timestamp("1970-01-01 01:00", tz="UTC")


def test_to_datetime_passes_through_datetimes():
    src = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02"]))
    out = _to_datetime(src)
    assert str(out.dt.tz) == "UTC"


# ---- fetch ----


def test_fetch_single_site_year(monkeypatch):
    _patch(monkeypatch)
    df = fetch_ukaq_measurements("AAA", 2020, source="aqe")
    assert len(df) == 24
    assert df["date"].dt.tz is not None
    assert df["date"].iloc[0] == pd.Timestamp("2020-01-01", tz="UTC")
    assert (df["network"] == "aqe").all()
    assert {"NO", "NO2", "NOXasNO2", "PM10", "code", "site"} <= set(df.columns)


def test_fetch_lowercase_site_is_upcased(monkeypatch):
    _patch(monkeypatch)
    assert len(fetch_ukaq_measurements("aaa", 2020)) == 24


def test_fetch_ignores_aggregate_companions(monkeypatch):
    """The archive also holds daily/24-hour means; only hourly must be used."""
    _patch(monkeypatch)
    df = fetch_ukaq_measurements("AAA", [2020])
    assert len(df) == 24, "picked up an aggregate frame instead of the hourly one"


def test_fetch_concatenates_sites_and_years(monkeypatch):
    _patch(monkeypatch, present=("AAA_2020", "AAA_2021", "BBB_2020"))
    df = fetch_ukaq_measurements(["AAA", "BBB"], [2020, 2021], on_missing="ignore")
    assert len(df) == 72
    assert set(df["code"]) == {"AAA", "BBB"}
    # sorted by (code, date)
    assert df["code"].is_monotonic_increasing


def test_fetch_pollutant_filter_keeps_identity_columns(monkeypatch):
    _patch(monkeypatch)
    df = fetch_ukaq_measurements("AAA", 2020, pollutant="noxasno2")
    assert set(df.columns) == {"date", "code", "site", "network", "NOXasNO2"}


def test_fetch_pollutant_filter_accepts_iterable(monkeypatch):
    _patch(monkeypatch)
    df = fetch_ukaq_measurements("AAA", 2020, pollutant=["NO2", "PM10"])
    assert {"NO2", "PM10"} <= set(df.columns)
    assert "NOXasNO2" not in df.columns


def test_fetch_unknown_pollutant_warns_and_drops(monkeypatch, caplog):
    _patch(monkeypatch)
    df = fetch_ukaq_measurements("AAA", 2020, pollutant=["NO2", "SO2"])
    assert "SO2" not in df.columns
    assert "NO2" in df.columns


def test_fetch_missing_archive_warns_by_default(monkeypatch):
    _patch(monkeypatch, present=("AAA_2020",))
    df = fetch_ukaq_measurements("AAA", [2020, 2021])
    assert len(df) == 24  # 2021 skipped, 2020 kept


def test_fetch_missing_archive_can_raise(monkeypatch):
    _patch(monkeypatch, present=("AAA_2020",))
    with pytest.raises(RuntimeError, match="could not fetch"):
        fetch_ukaq_measurements("AAA", [2020, 2021], on_missing="raise")


def test_fetch_all_missing_returns_empty_frame(monkeypatch):
    _patch(monkeypatch, present=())
    df = fetch_ukaq_measurements("AAA", 2020)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_fetch_meta_join(monkeypatch):
    _patch(monkeypatch)
    df = fetch_ukaq_measurements("AAA", 2020, meta=True)
    assert (df["site_type"] == "Urban Traffic").all()
    assert (df["latitude"] == 53.0).all()
    # AAA appears twice in the metadata (NOx and PM10 rows); the join must
    # not duplicate the 24 hourly rows into 48.
    assert len(df) == 24


def test_fetch_rejects_bad_on_missing(monkeypatch):
    _patch(monkeypatch)
    with pytest.raises(ValueError, match="on_missing"):
        fetch_ukaq_measurements("AAA", 2020, on_missing="explode")


def test_fetch_rejects_empty_inputs(monkeypatch):
    _patch(monkeypatch)
    with pytest.raises(ValueError, match="non-empty"):
        fetch_ukaq_measurements([], 2020)
    with pytest.raises(ValueError, match="non-empty"):
        fetch_ukaq_measurements("AAA", [])


def test_fetch_rejects_bad_source(monkeypatch):
    _patch(monkeypatch)
    with pytest.raises(ValueError, match="Unknown source"):
        fetch_ukaq_measurements("AAA", 2020, source="nope")


# ---- station listing ----


def test_list_stations_renames_to_openair_schema(monkeypatch):
    _patch(monkeypatch)
    s = list_ukaq_stations("aqe")
    assert {"code", "site", "site_type", "latitude", "longitude", "network"} <= set(s.columns)
    assert "site_id" not in s.columns


def test_list_stations_dedups_to_one_row_per_station(monkeypatch):
    _patch(monkeypatch)
    s = list_ukaq_stations("aqe")
    assert len(s) == 3  # AAA appears twice in _META
    assert s["code"].is_unique
    assert "variable" not in s.columns


def test_list_stations_all_variables_keeps_species_rows(monkeypatch):
    _patch(monkeypatch)
    s = list_ukaq_stations("aqe", all_variables=True)
    assert len(s) == 4
    assert "variable" in s.columns


def test_list_stations_site_type_filter(monkeypatch):
    _patch(monkeypatch)
    s = list_ukaq_stations("aqe", site_type="Rural Background")
    assert list(s["code"]) == ["BBB"]


def test_list_stations_site_type_filter_is_case_insensitive(monkeypatch):
    _patch(monkeypatch)
    s = list_ukaq_stations("aqe", site_type=["rural background", "urban background"])
    assert set(s["code"]) == {"BBB", "CCC"}


def test_list_stations_pollutant_filter(monkeypatch):
    _patch(monkeypatch)
    s = list_ukaq_stations("aqe", pollutant="PM10")
    assert list(s["code"]) == ["AAA"]


def test_list_stations_caches_metadata(monkeypatch):
    calls = {"n": 0}
    real = _META

    def fake(url, **kwargs):
        calls["n"] += 1
        return {"metadata": real.copy()}

    monkeypatch.setattr(ukaq, "_read_rdata", fake)
    list_ukaq_stations("aqe")
    list_ukaq_stations("aqe")
    assert calls["n"] == 1, "metadata should be fetched once per source per process"
