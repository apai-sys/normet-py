"""Extended mock-based tests for data adapters (OpenAQ, ERA5, EEA).

Verify all fetch and download functions using robust offline mocks,
ensuring high code coverage without hitting external servers.
"""

from __future__ import annotations

import importlib.util
import io

import pandas as pd
import pytest

from normet.io import eea, era5, openaq

# Ensure optional deps are available, otherwise skip
requests_available = importlib.util.find_spec("requests") is not None
cdsapi_available = importlib.util.find_spec("cdsapi") is not None

# ---------------------------------------------------------------------------
# OpenAQ Mocks & Tests
# ---------------------------------------------------------------------------


class MockResponse:
    def __init__(self, json_data: dict, status_code: int = 200, text_data: str = "") -> None:
        self._json = json_data
        self.status_code = status_code
        self.text = text_data
        self.content = text_data.encode("utf-8")

    def json(self) -> dict:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP Error: {self.status_code}")


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_openaq_locations_mocked(monkeypatch):
    import requests

    def mock_get(url, params=None, headers=None, timeout=None):
        return MockResponse(
            json_data={
                "results": [
                    {
                        "id": 12345,
                        "name": "London West",
                        "locality": "London",
                        "country": {"code": "GB"},
                        "coordinates": {"latitude": 51.5, "longitude": -0.1},
                        "sensors": [
                            {
                                "id": 888,
                                "name": "pm25_s",
                                "parameter": {"id": 2, "name": "pm25", "units": "ug/m3"},
                            }
                        ],
                        "datetimeLast": {"utc": "2024-01-01T12:00:00Z"},
                    }
                ]
            }
        )

    monkeypatch.setattr(requests, "get", mock_get)

    df = openaq.openaq_locations(
        country="GB",
        city="London",
        bbox=(-0.2, 51.4, 0.0, 51.6),
        parameter="pm25",
        limit=5,
        api_key="mock-key",
    )

    assert len(df) == 1
    assert df.loc[0, "id"] == 12345
    assert df.loc[0, "name"] == "London West"
    assert df.loc[0, "city"] == "London"
    assert df.loc[0, "country"] == "GB"
    assert df.loc[0, "lat"] == 51.5
    assert df.loc[0, "lon"] == -0.1
    assert df.loc[0, "parameters"] == ["pm25"]
    assert len(df.loc[0, "sensors"]) == 1
    assert df.loc[0, "sensors"][0]["id"] == 888
    assert df.loc[0, "sensors"][0]["parameter_name"] == "pm25"
    assert df.loc[0, "last_updated"] == "2024-01-01T12:00:00Z"


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_openaq_locations_paging_and_cap(monkeypatch):
    """Measured v3 behaviour (2026-10-23): limit above 1000 is a hard 422, so
    the adapter caps the request at 1000 and walks the `page` cursor until a
    short page; a location that never reported has datetimeLast null."""
    import requests

    pages = {
        1: {"results": [{"id": i, "coordinates": {}} for i in range(1000)]},
        2: {
            "results": [
                {"id": 9999, "coordinates": {}, "datetimeLast": None},
            ]
        },
    }
    seen: list[tuple[int, int]] = []

    def mock_get(url, params=None, headers=None, timeout=None):
        page = (params or {}).get("page", 1)
        seen.append((page, params["limit"]))
        return MockResponse(json_data=pages[page])

    monkeypatch.setattr(requests, "get", mock_get)
    df = openaq.openaq_locations(country="GB", limit=2500, api_key="k")
    assert [s[0] for s in seen] == [1, 2]  # stopped on the short page 2
    assert all(s[1] == 1000 for s in seen)  # the 2500 request was capped
    assert len(df) == 1001
    assert df.iloc[-1]["last_updated"] is None  # datetimeLast null, not a crash


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_openaq_locations_union_parameter_ids(monkeypatch):
    """A slug union goes out as repeated parameters_id keys, resolving each
    slug to its full id family (mass first)."""
    import requests

    captured = {}

    def mock_get(url, params=None, headers=None, timeout=None):
        captured["params"] = params
        return MockResponse(json_data={"results": []})

    monkeypatch.setattr(requests, "get", mock_get)
    openaq.openaq_locations(country="GB", parameters=["pm25", "no2"], api_key="k")
    ids = captured["params"]["parameters_id"]
    assert ids[:3] == [2, 5, 7]  # pm25 then no2 mass+ppm+ppb
    assert 15 in ids  # and no2's third id


def test_openaq_parameter_id_families():
    """Each slug resolves to its unit-family ids, µg/m3 mass first (live
    /parameters, 2026-10-23), so a fetch prefers mass to ppm/ppb."""
    assert openaq._resolve_parameter_ids("no2") == (5, 7, 15)
    assert openaq._resolve_parameter_ids("pm25") == (2,)
    assert openaq._resolve_parameter_ids("o3") == (3, 10, 32)
    assert openaq._resolve_parameter_ids(42) == (42,)  # raw id passes through


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_openaq_resolve_sensor_prefers_mass_id(monkeypatch):
    """A location with both a ppm and a µg/m3 sensor for one pollutant must
    return the mass one, so values land on the same scale as UK/EEA data."""
    import requests

    def mock_get(url, params=None, headers=None, timeout=None):
        return MockResponse(
            json_data={
                "results": [
                    {
                        "coordinates": {"latitude": 1.0, "longitude": 2.0},
                        # ppm sensor first in the list; mass (id 5) must still win
                        "sensors": [
                            {"id": 10, "parameter": {"id": 7, "name": "no2"}},
                            {"id": 5, "parameter": {"id": 5, "name": "no2"}},
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr(requests, "get", mock_get)
    sid, lat, lon = openaq._resolve_sensor(1, openaq._resolve_parameter_ids("no2"), {})
    assert sid == 5


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_openaq_sensors_mocked(monkeypatch):
    import requests

    def mock_get(url, params=None, headers=None, timeout=None):
        return MockResponse(
            json_data={
                "results": [
                    {
                        "id": 999,
                        "name": "pm25_sensor",
                        "parameter": {
                            "id": 2,
                            "name": "pm25",
                            "units": "ug/m3",
                            "displayName": "PM2.5",
                        },
                        "datetimeFirst": {"utc": "2023-01-01T00:00:00Z"},
                        "datetimeLast": {"utc": "2024-01-01T00:00:00Z"},
                    }
                ]
            }
        )

    monkeypatch.setattr(requests, "get", mock_get)

    df = openaq.openaq_sensors(
        location_id=12345,
        limit=10,
        api_key="mock-key",
    )

    assert len(df) == 1
    assert df.loc[0, "id"] == 999
    assert df.loc[0, "name"] == "pm25_sensor"
    assert df.loc[0, "parameter_id"] == 2
    assert df.loc[0, "parameter_name"] == "pm25"
    assert df.loc[0, "parameter_units"] == "ug/m3"
    assert df.loc[0, "parameter_display_name"] == "PM2.5"
    assert df.loc[0, "datetime_first"] == "2023-01-01T00:00:00Z"
    assert df.loc[0, "datetime_last"] == "2024-01-01T00:00:00Z"


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_fetch_openaq_measurements_mocked(monkeypatch):
    # OpenAQ v3 has no /locations/{id}/measurements endpoint -- measurements
    # are only served per-sensor. Real calls: (1) GET /locations/{id} to
    # resolve the pm25 sensor id + site coordinates, (2) paginate
    # /sensors/{sensor_id}/measurements.
    import requests

    call_count = 0

    def mock_get(url, params=None, headers=None, timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            assert url.endswith("/locations/12345")
            return MockResponse(
                json_data={
                    "results": [
                        {
                            "id": 12345,
                            "coordinates": {"latitude": 51.5, "longitude": -0.1},
                            "sensors": [
                                {"id": 888, "parameter": {"id": 2, "name": "pm25"}},
                            ],
                        }
                    ]
                }
            )
        elif call_count == 2:
            assert url.endswith("/sensors/888/measurements")
            return MockResponse(
                json_data={
                    "results": [
                        {
                            "period": {"datetimeFrom": {"utc": "2024-01-01T00:00:00Z"}},
                            "coordinates": None,
                            "value": 15.0,
                            "parameter": {"name": "pm25", "units": "ug/m3"},
                        }
                    ]
                }
            )
        else:
            # Empty on second measurements page to terminate loop
            return MockResponse(json_data={"results": []})

    monkeypatch.setattr(requests, "get", mock_get)

    df = openaq.fetch_openaq_measurements(
        location_id=12345,
        parameter="pm25",
        date_from="2024-01-01",
        date_to="2024-01-02",
        page_limit=1,
        api_key="mock-key",
    )

    assert len(df) == 1
    assert df.loc[0, "site"] == 12345
    assert df.loc[0, "parameter"] == "pm25"
    assert df.loc[0, "value"] == 15.0
    assert df.loc[0, "unit"] == "ug/m3"
    # coordinates come from the resolved location when the measurement itself omits them
    assert df.loc[0, "lat"] == 51.5
    assert df.loc[0, "lon"] == -0.1


# ---------------------------------------------------------------------------
# EEA Mocks & Tests
# ---------------------------------------------------------------------------


class _BytesResponse:
    """Minimal response for the Parquet service: bytes and a status code."""

    def __init__(self, content: bytes) -> None:
        self.content = content
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None


def _eea_parquet_bytes() -> bytes:
    """A response frame in the service's own columns and dtypes: Decimals
    for Value, sampling-point ids for Samplingpoint, and the -99 sentinel."""
    from decimal import Decimal

    raw = pd.DataFrame(
        {
            "Samplingpoint": [
                "GB/GB_SamplingPoint_65449",
                "GB/GB_SamplingPoint_65449",
                "GB/GB_SamplingPoint_76934",
            ],
            "Pollutant": [5, 5, 5],
            "Start": ["2024-01-03 01:00:00", "2024-01-03 02:00:00", "2024-01-03 01:00:00"],
            "End": ["2024-01-03 02:00:00", "2024-01-03 03:00:00", "2024-01-03 02:00:00"],
            "Value": [Decimal("3.865"), Decimal("-99.0"), Decimal("8.696")],
            "Unit": ["ug.m-3"] * 3,
            "AggType": ["hour"] * 3,
            "Validity": [2, -1, 2],
        }
    )
    buf = io.BytesIO()
    raw.to_parquet(buf)
    return buf.getvalue()


@pytest.mark.skipif(not requests_available, reason="requests not installed")
@pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow not installed")
def test_fetch_eea_data_mocked(monkeypatch):
    """The Parquet download-service round trip, service mocked.

    Covers what the live service is relied on for: the POST query document
    (the service takes a body, not query parameters), the column mapping to
    the tidy names, the -99 sentinel becoming NaN, the numeric pollutant
    code resolved to its notation, and the client-side station filter.
    """
    import requests

    payload = _eea_parquet_bytes()
    empty_zip = b"PK\x05\x06" + b"\x00" * 18
    bodies: list = []
    urls: list = []

    def mock_post(url, params=None, json=None, headers=None, timeout=None):
        urls.append(url)
        bodies.append(json)
        # Data for January only: the service answers other empty months with
        # a 22-byte zip, which the adapter must skip rather than read.
        month = json["dateTimeStart"][:7]
        return _BytesResponse(payload if month == "2024-01" else empty_zip)

    monkeypatch.setattr(requests, "post", mock_post)

    df = eea.fetch_eea_data(
        country="gb",
        pollutant=5,  # numeric EEA code: must reach the service as "PM10"
        year_from=2024,
        year_to=2024,
        station="65449",
    )

    # One request per calendar month: the service caps a synchronous answer
    # ("Request too large..."), so a year is twelve monthly requests.
    assert len(bodies) == 12
    assert all(u.endswith("/ParquetFile/dynamic") for u in urls)
    assert bodies[0]["dateTimeStart"] == "2024-01-01T00:00:00"
    assert bodies[-1]["dateTimeEnd"] == "2024-12-31T23:59:59"
    assert bodies[0]["countries"] == ["GB"]
    assert bodies[0]["cities"] == []  # required key: "every city"
    assert bodies[0]["pollutants"] == ["PM10"]
    assert all("source" not in b for b in bodies)  # the "All" default stays unsent

    assert list(df.columns) == [
        "date",
        "site",
        "country",
        "pollutant",
        "value",
        "unit",
        "aggregation",
        "validity",
    ]
    assert len(df) == 2  # the station filter dropped the 76934 row
    assert (df["site"] == "GB/GB_SamplingPoint_65449").all()
    assert df["pollutant"].unique().tolist() == ["PM10"]
    assert df["country"].unique().tolist() == ["GB"]
    assert df["value"].isna().sum() == 1  # the -99 sentinel
    assert df["value"].dropna().iloc[0] == pytest.approx(3.865)
    assert df["date"].dt.tz is not None
    assert df["validity"].dropna().isin([2, -1]).all()


@pytest.mark.skipif(not requests_available, reason="requests not installed")
@pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow not installed")
def test_fetch_eea_data_no_rows_is_empty_frame(monkeypatch):
    """A query matching nothing answers every month with a 22-byte empty zip,
    not Parquet; reading it as Parquet would raise, so the adapter guards on
    the magic and returns an empty frame."""
    import requests

    monkeypatch.setattr(
        requests, "post", lambda url, **kw: _BytesResponse(b"PK\x05\x06" + b"\x00" * 18)
    )
    df = eea.fetch_eea_data(country="ZZ", pollutant="PM10", year_from=2024, year_to=2024)
    assert df.empty


# --- station level: ArcGIS layer + popup blobs -----------------------------


def _eea_feature(code: str, name: str, lon: float, lat: float, full: bool) -> dict:
    """One geojson feature shaped like the map service's real replies
    (GB capture, 2026-10-23): anchor text is the notation, prose titles
    carry "<" and <sup> markup, an empty href means "no blob"."""
    lines = []
    if full:
        lines.append(
            ' - Particulate matter < 2.5 µm (aerosol): <a target="_blank" '
            'href="https://blob/PM25.parquet" style="color: Blue;">PM2.5</a>, '
            "ug/m<sup>3</sup>, (2013 - 2019)"
        )
        lines.append(
            ' - Nitrogen dioxide (air): <a href="https://blob/NO2.parquet">NO2</a>, '
            "ug/m<sup>3</sup>, (2013 - 2019)"
        )
        lines.append(
            ' - Nitrogen oxides (air): <a href="https://blob/NOX.parquet">NOX as NO2</a>, '
            "ug/m<sup>3</sup>, (2015 - 2018)"
        )
        lines.append(
            ' - Benzene (air): <a href="" style="color: Blue;">C6H6</a>, '
            "ug/m<sup>3</sup>, (2014 - 2017)"
        )
        lines.append(
            ' - Benzo(a)pyrene in PM10 (aerosol): <a href="https://blob/BaP.parquet">'
            "BaP in PM10</a>, ng/m<sup>3</sup>, (2013 - 2019)"
        )
    lines.append(
        ' - Ozone (air): <a href="https://blob/O3.parquet">O3</a>, ug/m<sup>3</sup>, (2016 - 2020)'
    )
    popup = "<p> Air Quality Station:<strong>x</strong></p>" + "".join(
        f"<p>{line} </p>" for line in lines
    )
    return {
        "type": "Feature",
        "properties": {
            "AirQualityStationEoICode": code,
            "AQStationName": name,
            "CountryCode": "GB",
            "PopupInfo": popup,
        },
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
    }


def _eea_layer_response(features: list[dict]):
    import json

    body = json.dumps({"type": "FeatureCollection", "features": features})

    class _Resp:
        status_code = 200
        text = body
        content = body.encode("utf-8")

        def raise_for_status(self) -> None:
            return None

    return _Resp()


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_list_eea_stations_parses_popup_blobs(monkeypatch):
    """The popup grammar the layer really returns (GB capture, 2026-10-23):
    anchor text is the reporting notation; benzene's empty href and the
    heavy-metal species ("BaP in PM10") are skipped, while "NOX as NO2"
    normalises to the app notation and every line carries its year range."""
    import json

    import requests

    feats = [
        _eea_feature("GB0777A", "Liverpool Speke", -2.84, 53.35, True),
        _eea_feature("GB0742A", "Inverness", -4.24, 57.48, False),
    ]
    monkeypatch.setattr(requests, "get", lambda url, **kw: _eea_layer_response(feats))

    df = eea.list_eea_stations(country="GB", pollutants=["PM2.5", "O3"])
    # Sorted by n desc: Liverpool (PM2.5 + O3) first.
    assert list(df["code"]) == ["GB0777A", "GB0742A"]
    assert df["pollutants"].tolist() == ["PM2.5, O3", "O3"]
    assert df["lat"].iloc[0] == pytest.approx(53.35)

    links = json.loads(df[df["code"] == "GB0777A"].iloc[0]["links"])
    assert set(links) == {"PM2.5", "NO2", "NOXasNO2", "O3"}
    assert links["PM2.5"]["url"].endswith("PM25.parquet")
    assert (links["PM2.5"]["from"], links["PM2.5"]["to"]) == (2013, 2019)
    assert (links["NOXasNO2"]["from"], links["NOXasNO2"]["to"]) == (2015, 2018)


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_list_eea_stations_pages_past_the_layer_cap(monkeypatch):
    """One query answers at most _ARCGIS_PAGE features; the adapter must ask
    for the next offset until a short page arrives, or a big country loses
    its tail of stations."""
    import requests

    monkeypatch.setattr(eea, "_ARCGIS_PAGE", 2)
    feats = [_eea_feature(f"GB000{i}A", f"S{i}", 0.0, 50.0, False) for i in range(3)]
    offsets: list[int] = []

    def mock_get(url, params=None, **kw):
        offsets.append(params["resultOffset"])
        return _eea_layer_response(feats[params["resultOffset"] :][:2])

    monkeypatch.setattr(requests, "get", mock_get)
    df = eea.list_eea_stations(country="GB")
    assert offsets == [0, 2]
    assert len(df) == 3


def _eea_blob_bytes() -> bytes:
    """A station blob in the service's own columns: one pollutant (code 8 =
    NO2), hourly, with both "no measurement" sentinels -- -99 as the bulk
    service writes it and -999 as the per-station blobs do."""
    from decimal import Decimal

    raw = pd.DataFrame(
        {
            "Samplingpoint": ["GB/GB_SamplingPoint_61464"] * 4,
            "Pollutant": [8, 8, 8, 8],
            "Start": [
                "2013-01-01 01:00:00",
                "2013-01-01 02:00:00",
                "2013-01-01 03:00:00",
                "2013-01-01 04:00:00",
            ],
            "End": [
                "2013-01-01 02:00:00",
                "2013-01-01 03:00:00",
                "2013-01-01 04:00:00",
                "2013-01-01 05:00:00",
            ],
            "Value": [Decimal("3.2"), Decimal("-99.0"), Decimal("-999.0"), Decimal("4.4")],
            "Unit": ["ug.m-3"] * 4,
            "AggType": ["hour"] * 4,
            "Validity": [1, -1, -1, 1],
        }
    )
    buf = io.BytesIO()
    raw.to_parquet(buf)
    return buf.getvalue()


@pytest.mark.skipif(not requests_available, reason="requests not installed")
@pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow not installed")
def test_fetch_eea_station_mocked(monkeypatch):
    """Station lookup then one blob download: the whole per-station pull is
    two GETs, and the tidy frame carries the station's name and coordinates
    so the GUI can join met without another call."""
    import requests

    feats = [
        _eea_feature("GB0777A", "Liverpool Speke", -2.84, 53.35, True),
        _eea_feature("GB0742A", "Inverness", -4.24, 57.48, False),
    ]
    blob = _eea_blob_bytes()
    urls: list[str] = []

    def mock_get(url, **kw):
        urls.append(url)
        if url.endswith("/query"):
            return _eea_layer_response(feats)
        return _BytesResponse(blob)

    monkeypatch.setattr(requests, "get", mock_get)

    df = eea.fetch_eea_station(station="gb0777a", country="GB", pollutant="NO2")
    assert urls[0].endswith("/query") and len(urls) == 2  # code case-insensitive
    assert df["pollutant"].unique().tolist() == ["NO2"]
    assert df["site"].iloc[0] == "Liverpool Speke"
    assert df["value"].isna().sum() == 2  # the -99 and -999 sentinels
    assert df["lat"].iloc[0] == pytest.approx(53.35)
    assert df["date"].dt.tz is not None

    # O3-only station has no PM2.5 blob; an unknown code has no station at all.
    with pytest.raises(RuntimeError, match="no PM2.5 blob"):
        eea.fetch_eea_station(station="GB0742A", country="GB", pollutant="PM2.5")
    with pytest.raises(RuntimeError, match="No EEA station"):
        eea.fetch_eea_station(station="GB9999", country="GB", pollutant="NO2")


# ---------------------------------------------------------------------------
# ERA5 Mocks & Tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# UKAQ aurn_live (SOS API) Mocks & Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_list_ukaq_stations_aurn_live_mocked(monkeypatch):
    import requests

    from normet.io import ukaq

    monkeypatch.setattr(ukaq, "_aurn_live_site_codes", lambda: {"London N. Kensington": "MY1"})

    station_payload = [
        {
            "type": "Feature",
            "properties": {"id": "100", "label": "London N. Kensington"},
            "geometry": {"type": "Point", "coordinates": [51.521, -0.213]},
        },
        {
            "type": "Feature",
            "properties": {"id": "200", "label": "Birmingham Centre"},
            "geometry": {"type": "Point", "coordinates": [52.479, -1.906]},
        },
    ]

    def mock_get(url, params=None, headers=None, timeout=None):
        assert url == f"{ukaq._AURN_LIVE_API_BASE}/stations"
        return MockResponse(station_payload)

    monkeypatch.setattr(requests, "get", mock_get)

    df = ukaq.list_ukaq_stations("aurn_live")
    assert len(df) == 2
    assert list(df.columns) == [
        "code",
        "site",
        "site_type",
        "latitude",
        "longitude",
        "start_date",
        "end_date",
        "network",
    ]
    row = df[df["site"] == "London N. Kensington"].iloc[0]
    assert row["code"] == "MY1"
    assert row["latitude"] == 51.521
    assert row["longitude"] == -0.213
    assert row["network"] == "aurn_live"
    assert pd.isna(row["site_type"])
    birmingham = df[df["site"] == "Birmingham Centre"].iloc[0]
    assert pd.isna(birmingham["code"])  # not in the mocked site-code lookup


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_list_ukaq_stations_aurn_live_by_pollutant_mocked(monkeypatch):
    import requests

    from normet.io import ukaq

    monkeypatch.setattr(ukaq, "_aurn_live_site_codes", lambda: {"London N. Kensington": "MY1"})

    timeseries_payload = [
        {
            "id": "ts-1",
            "label": "PM2.5 London N. Kensington",
            "station": {
                "type": "Feature",
                "properties": {"id": "100", "label": "London N. Kensington"},
                "geometry": {"type": "Point", "coordinates": [51.521, -0.213]},
            },
        },
    ]

    def mock_get(url, params=None, headers=None, timeout=None):
        assert url == f"{ukaq._AURN_LIVE_API_BASE}/timeseries"
        assert params == {"phenomenon": "6001", "limit": 5000}
        return MockResponse(timeseries_payload)

    monkeypatch.setattr(requests, "get", mock_get)

    df = ukaq.list_ukaq_stations("aurn_live", pollutant="PM2.5")
    assert len(df) == 1
    assert df.loc[0, "code"] == "MY1"
    assert df.loc[0, "site"] == "London N. Kensington"
    assert df.loc[0, "network"] == "aurn_live"
    assert "variable" not in df.columns  # all_variables=False (default) drops it


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_list_ukaq_stations_aurn_live_all_variables_mocked(monkeypatch):
    import requests

    from normet.io import ukaq

    monkeypatch.setattr(ukaq, "_aurn_live_site_codes", lambda: {})

    def mock_get(url, params=None, headers=None, timeout=None):
        assert url == f"{ukaq._AURN_LIVE_API_BASE}/timeseries"
        return MockResponse(
            [
                {
                    "id": "ts-1",
                    "station": {
                        "properties": {"id": "100", "label": "London N. Kensington"},
                        "geometry": {"coordinates": [51.521, -0.213]},
                    },
                }
            ]
        )

    monkeypatch.setattr(requests, "get", mock_get)

    df = ukaq.list_ukaq_stations("aurn_live", pollutant=["NO2", "O3"], all_variables=True)
    assert set(df["variable"]) == {"NO2", "O3"}
    assert len(df) == 2  # one row per (site, pollutant) -- both requested species queried


def test_list_ukaq_stations_aurn_live_rejects_site_type():
    from normet.io import ukaq

    with pytest.raises(ValueError, match="site_type"):
        ukaq.list_ukaq_stations("aurn_live", site_type="Urban Traffic")


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_fetch_ukaq_measurements_aurn_live_mocked(monkeypatch):
    import requests

    from normet.io import ukaq

    monkeypatch.setattr(ukaq, "_aurn_live_site_codes", lambda: {"London N. Kensington": "MY1"})

    ts_discovery = [
        {
            "id": "ts-1",
            "station": {
                "properties": {"id": "100", "label": "London N. Kensington"},
                "geometry": {"coordinates": [51.521, -0.213]},
            },
        },
    ]
    ts_data = {
        "values": [
            {"timestamp": 1704067200000, "value": 12.5},
            {"timestamp": 1704070800000, "value": 14.2},
        ]
    }

    def mock_get(url, params=None, headers=None, timeout=None):
        if url == f"{ukaq._AURN_LIVE_API_BASE}/timeseries":
            return MockResponse(ts_discovery)
        if "getData" in url:
            return MockResponse(ts_data)
        return MockResponse([])

    monkeypatch.setattr(requests, "get", mock_get)

    df = ukaq.fetch_ukaq_measurements("MY1", 2024, source="aurn_live", pollutant="PM2.5")

    assert list(df.columns) == ["date", "code", "site", "PM2.5", "network"]
    assert len(df) == 2
    assert df.loc[0, "code"] == "MY1"
    assert df.loc[0, "site"] == "London N. Kensington"
    assert df.loc[0, "network"] == "aurn_live"
    assert sorted(df["PM2.5"]) == [12.5, 14.2]


def test_fetch_ukaq_measurements_aurn_live_unknown_code_warns(monkeypatch):
    from normet.io import ukaq

    monkeypatch.setattr(ukaq, "_aurn_live_site_codes", lambda: {})

    df = ukaq.fetch_ukaq_measurements("NOPE", 2024, source="aurn_live", pollutant="NO2")
    assert df.empty


def test_fetch_ukaq_measurements_aurn_live_unknown_code_raises(monkeypatch):
    from normet.io import ukaq

    monkeypatch.setattr(ukaq, "_aurn_live_site_codes", lambda: {})

    with pytest.raises(RuntimeError, match="unknown AURN code"):
        ukaq.fetch_ukaq_measurements(
            "NOPE", 2024, source="aurn_live", pollutant="NO2", on_missing="raise"
        )


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_fetch_ukaq_measurements_aurn_live_skips_none_values(monkeypatch):
    import requests

    from normet.io import ukaq

    monkeypatch.setattr(ukaq, "_aurn_live_site_codes", lambda: {"London N. Kensington": "MY1"})

    ts_discovery = [
        {
            "id": "ts-1",
            "station": {
                "properties": {"id": "100", "label": "London N. Kensington"},
                "geometry": {"coordinates": [51.521, -0.213]},
            },
        },
    ]
    ts_data = {
        "values": [
            {"timestamp": 1704067200000, "value": 10.0},
            {"timestamp": 1704070800000, "value": None},
            {"timestamp": None, "value": 12.0},
        ]
    }

    def mock_get(url, params=None, headers=None, timeout=None):
        if url == f"{ukaq._AURN_LIVE_API_BASE}/timeseries":
            return MockResponse(ts_discovery)
        if "getData" in url:
            return MockResponse(ts_data)
        return MockResponse([])

    monkeypatch.setattr(requests, "get", mock_get)

    df = ukaq.fetch_ukaq_measurements("MY1", 2024, source="aurn_live", pollutant="NO2")
    assert len(df) == 1
    assert df.loc[0, "NO2"] == 10.0


_FAKE_NETWORK_INFO_HTML = """
<html><body><form>
<select id="site_id" name="site_id" class="form-control webkit-arrow_fix">
<option value="">Select a site</option>
<option value="MAN3">Manchester Piccadilly</option>
<option value="MY1">London Marylebone Road</option>
<option value="BRS8">Bristol St Paul's</option>
</select>
</form></body></html>
"""


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_aurn_live_site_codes_mocked(monkeypatch):
    import requests

    from normet.io import ukaq

    ukaq._aurn_live_site_codes.cache_clear()

    def mock_get(url, params=None, headers=None, timeout=None):
        assert url == ukaq._AURN_NETWORK_INFO_URL
        assert params == {"view": "aurn"}
        return MockResponse({}, text_data=_FAKE_NETWORK_INFO_HTML)

    monkeypatch.setattr(requests, "get", mock_get)

    codes = ukaq._aurn_live_site_codes()
    assert codes["Manchester Piccadilly"] == "MAN3"
    assert codes["London Marylebone Road"] == "MY1"
    assert codes["Bristol St Paul's"] == "BRS8"  # plain apostrophe survives unescaped
    assert "" not in codes.values()  # the blank placeholder option is dropped

    # cached: a second call must not hit the network again
    monkeypatch.setattr(
        requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("not cached"))
    )
    assert ukaq._aurn_live_site_codes() is codes
    ukaq._aurn_live_site_codes.cache_clear()


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_aurn_live_site_codes_graceful_on_bad_html(monkeypatch):
    import requests

    from normet.io import ukaq

    ukaq._aurn_live_site_codes.cache_clear()
    monkeypatch.setattr(
        requests, "get", lambda *a, **k: MockResponse({}, text_data="<html>no select here</html>")
    )

    codes = ukaq._aurn_live_site_codes()
    assert codes == {}
    ukaq._aurn_live_site_codes.cache_clear()


@pytest.mark.skipif(not cdsapi_available, reason="cdsapi not installed")
def test_fetch_era5_timeseries_mocked(monkeypatch):
    import cdsapi

    retrieve_calls = []

    # The v3 timeseries endpoint returns short variable names directly
    # (t2m, blh, ...), a `valid_time` column, and only the requested variables.
    _long_to_short = {"2m_temperature": "t2m", "boundary_layer_height": "blh"}
    _values = {"t2m": [280.5, 281.0], "blh": [150.0, 160.0]}

    class MockTimeSeriesCDSClient:
        def retrieve(self, dataset, request, target_path):
            retrieve_calls.append((dataset, request, target_path))
            loc = request["location"]
            data = {
                "valid_time": ["2024-01-01 00:00:00", "2024-01-02 00:00:00"],
                "latitude": [loc["latitude"], loc["latitude"]],
                "longitude": [loc["longitude"], loc["longitude"]],
            }
            for v in request["variable"]:
                short = _long_to_short[v]
                data[short] = _values[short]
            pd.DataFrame(data).to_csv(target_path, index=False)

    monkeypatch.setattr(cdsapi, "Client", MockTimeSeriesCDSClient)

    # 1. Test with a DataFrame sites input
    sites_df = pd.DataFrame({"site": ["London"], "lat": [51.5], "lon": [-0.1]})

    df = era5.fetch_era5_timeseries(
        sites=sites_df,
        date_from="2024-01-01",
        date_to="2024-01-02",
        variables=["2m_temperature", "boundary_layer_height"],
    )

    assert len(df) == 2
    assert list(df.columns) == ["site", "date", "lat", "lon", "t2m", "blh"]
    assert df.loc[0, "site"] == "London"
    assert df.loc[0, "t2m"] == 280.5
    assert df.loc[1, "blh"] == 160.0
    assert len(retrieve_calls) == 1
    assert retrieve_calls[0][0] == "reanalysis-era5-single-levels-timeseries"
    assert retrieve_calls[0][1]["location"]["latitude"] == 51.5
    assert retrieve_calls[0][1]["location"]["longitude"] == -0.1

    # 2. Test with a Mapping sites input
    retrieve_calls.clear()
    sites_dict = {"Oxford": (51.75, -1.25)}

    df_dict = era5.fetch_era5_timeseries(
        sites=sites_dict,
        date_from="2024-01-01",
        date_to="2024-01-02",
        variables=["2m_temperature"],
    )
    assert len(df_dict) == 2
    assert "blh" not in df_dict.columns
    assert df_dict.loc[0, "site"] == "Oxford"
    assert df_dict.loc[0, "lat"] == 51.75
    assert df_dict.loc[0, "lon"] == -1.25
    assert len(retrieve_calls) == 1


@pytest.mark.skipif(not cdsapi_available, reason="cdsapi not installed")
def test_fetch_era5_timeseries_caching(monkeypatch, tmp_path):
    import cdsapi

    retrieve_calls = []

    class MockTimeSeriesCDSClient:
        def retrieve(self, dataset, request, target_path):
            retrieve_calls.append((dataset, request, target_path))
            loc = request["location"]
            df = pd.DataFrame(
                {
                    "valid_time": ["2024-01-01 00:00:00", "2024-01-02 00:00:00"],
                    "latitude": [loc["latitude"], loc["latitude"]],
                    "longitude": [loc["longitude"], loc["longitude"]],
                    "t2m": [280.5, 281.0],
                }
            )
            df.to_csv(target_path, index=False)

    monkeypatch.setattr(cdsapi, "Client", MockTimeSeriesCDSClient)

    sites_df = pd.DataFrame({"site": ["London"], "lat": [51.5], "lon": [-0.1]})
    cache_dir = tmp_path / "cache"

    # First call: should trigger retrieve
    df1 = era5.fetch_era5_timeseries(
        sites=sites_df,
        date_from="2024-01-01",
        date_to="2024-01-02",
        variables=["2m_temperature"],
        cache_dir=cache_dir,
    )
    assert len(df1) == 2
    assert len(retrieve_calls) == 1

    # Second call: should NOT trigger retrieve (reuses cache)
    retrieve_calls.clear()
    df2 = era5.fetch_era5_timeseries(
        sites=sites_df,
        date_from="2024-01-01",
        date_to="2024-01-02",
        variables=["2m_temperature"],
        cache_dir=cache_dir,
    )
    assert len(df2) == 2
    assert len(retrieve_calls) == 0
