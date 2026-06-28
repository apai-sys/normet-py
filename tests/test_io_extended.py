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
    import requests

    call_count = 0

    def mock_get(url, params=None, headers=None, timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MockResponse(
                json_data={
                    "results": [
                        {
                            "period": {"datetimeFrom": {"utc": "2024-01-01T00:00:00Z"}},
                            "coordinates": {"latitude": 51.5, "longitude": -0.1},
                            "value": 15.0,
                            "parameter": {"name": "pm25", "units": "ug/m3"},
                        }
                    ]
                }
            )
        else:
            # Empty on second page to terminate loop
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
    assert df.loc[0, "lat"] == 51.5
    assert df.loc[0, "lon"] == -0.1


# ---------------------------------------------------------------------------
# EEA Mocks & Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_fetch_eea_data_mocked(monkeypatch):
    import requests

    csv_data = (
        "DatetimeBegin,AirQualityStation,Concentration,UnitOfMeasurement,Latitude,Longitude,Pollutant,CountryCode\n"
        "2024-01-01 00:00:00,GB0001A,10.0,ug/m3,51.5,-0.1,PM2.5,GB\n"
    )

    def mock_get(url, params=None, headers=None, timeout=None):
        if "AQData_Extract.fmw" in url:
            # Discovery service returns URLs
            return MockResponse({}, text_data="http://mock-eea-csv.com/file.csv")
        else:
            # CSV download endpoint
            return MockResponse({}, text_data=csv_data)

    monkeypatch.setattr(requests, "get", mock_get)

    df = eea.fetch_eea_data(
        country="GB",
        pollutant="PM2.5",
        year_from=2024,
        year_to=2024,
        station="GB0001A",
    )

    assert len(df) == 1
    assert df.loc[0, "site"] == "GB0001A"
    assert df.loc[0, "value"] == 10.0
    assert df.loc[0, "unit"] == "ug/m3"
    assert df.loc[0, "lat"] == 51.5
    assert df.loc[0, "lon"] == -0.1
    assert df.loc[0, "pollutant"] == "PM2.5"
    assert df.loc[0, "country"] == "GB"


# ---------------------------------------------------------------------------
# ERA5 Mocks & Tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DEFRA / AURN Mocks & Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_list_aurn_stations_mocked(monkeypatch):
    import requests

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
        from normet.io.defra import _API_BASE

        assert url == f"{_API_BASE}/stations"
        return MockResponse(station_payload)

    monkeypatch.setattr(requests, "get", mock_get)

    from normet.io.defra import list_aurn_stations

    df = list_aurn_stations()
    assert len(df) == 2
    assert list(df.columns) == ["id", "label", "lat", "lon"]
    assert df.loc[0, "id"] == "100"
    assert df.loc[0, "label"] == "London N. Kensington"
    assert df.loc[0, "lat"] == 51.521
    assert df.loc[0, "lon"] == -0.213


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_list_aurn_stations_by_pollutant_mocked(monkeypatch):
    import requests

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
        from normet.io.defra import _API_BASE

        assert url == f"{_API_BASE}/timeseries"
        assert params == {"phenomenon": "6001", "limit": 5000}
        return MockResponse(timeseries_payload)

    monkeypatch.setattr(requests, "get", mock_get)

    from normet.io.defra import list_aurn_stations

    df = list_aurn_stations(pollutant="PM2.5")
    assert len(df) == 1
    assert "timeseries_id" in df.columns
    assert df.loc[0, "id"] == "100"
    assert df.loc[0, "label"] == "London N. Kensington"
    assert df.loc[0, "timeseries_id"] == "ts-1"


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_fetch_aurn_measurements_mocked(monkeypatch):
    import requests

    ts_discovery = [
        {
            "id": "ts-1",
            "station": {
                "type": "Feature",
                "properties": {"id": "100", "label": "London N. Kensington"},
                "geometry": {"type": "Point", "coordinates": [51.521, -0.213]},
            },
        },
    ]

    ts_data = {
        "values": [
            {"timestamp": 1704067200000, "value": 12.5},
            {"timestamp": 1704070800000, "value": 14.2},
        ]
    }

    call_log: list[str] = []

    def mock_get(url, params=None, headers=None, timeout=None):
        from normet.io.defra import _API_BASE

        call_log.append(url)
        if f"{_API_BASE}/timeseries" == url and params is not None:
            return MockResponse(ts_discovery)
        if "getData" in url:
            return MockResponse(ts_data)
        return MockResponse([])

    monkeypatch.setattr(requests, "get", mock_get)

    from normet.io.defra import fetch_aurn_measurements

    df = fetch_aurn_measurements(
        station="100",
        pollutant="PM2.5",
        date_from="2024-01-01",
        date_to="2024-01-02",
    )

    assert len(df) == 2
    assert list(df.columns) == [
        "date",
        "site",
        "station_id",
        "pollutant",
        "value",
        "unit",
        "lat",
        "lon",
    ]
    assert df.loc[0, "value"] == 12.5
    assert df.loc[1, "value"] == 14.2
    assert df.loc[0, "site"] == "London N. Kensington"
    assert df.loc[0, "station_id"] == "100"
    assert df.loc[0, "pollutant"] == "PM2.5"
    assert df.loc[0, "unit"] == "ug.m-3"


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_fetch_aurn_measurements_station_label_filter(monkeypatch):
    import requests

    ts_discovery = [
        {
            "id": "ts-a",
            "label": "PM2.5 London N. Kensington",
            "station": {
                "type": "Feature",
                "properties": {"id": "100", "label": "London N. Kensington"},
                "geometry": {"type": "Point", "coordinates": [51.521, -0.213]},
            },
        },
        {
            "id": "ts-b",
            "label": "PM2.5 Birmingham Centre",
            "station": {
                "type": "Feature",
                "properties": {"id": "200", "label": "Birmingham Centre"},
                "geometry": {"type": "Point", "coordinates": [52.479, -1.906]},
            },
        },
    ]

    ts_data_a = {
        "values": [
            {"timestamp": 1704067200000, "value": 12.5},
        ]
    }

    call_log: list[str] = []

    def mock_get(url, params=None, headers=None, timeout=None):
        from normet.io.defra import _API_BASE

        call_log.append(url)
        if f"{_API_BASE}/timeseries" == url and params is not None:
            return MockResponse(ts_discovery)
        if "timeseries/ts-a/getData" in url:
            return MockResponse(ts_data_a)
        return MockResponse({"values": []})

    monkeypatch.setattr(requests, "get", mock_get)

    from normet.io.defra import fetch_aurn_measurements

    df = fetch_aurn_measurements(
        station_label="kensington",
        pollutant="PM2.5",
        date_from="2024-01-01",
        date_to="2024-01-02",
    )

    assert len(df) == 1
    assert df.loc[0, "site"] == "London N. Kensington"
    assert df.loc[0, "value"] == 12.5


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_fetch_aurn_measurements_no_matches(monkeypatch):
    import requests

    def mock_get(url, params=None, headers=None, timeout=None):
        from normet.io.defra import _API_BASE

        if f"{_API_BASE}/timeseries" == url and params is not None:
            return MockResponse([])
        return MockResponse({"values": []})

    monkeypatch.setattr(requests, "get", mock_get)

    from normet.io.defra import fetch_aurn_measurements

    df = fetch_aurn_measurements(
        station="999",
        pollutant="PM2.5",
        date_from="2024-01-01",
        date_to="2024-01-02",
    )

    assert len(df) == 0
    assert isinstance(df, pd.DataFrame)


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_fetch_aurn_measurements_all_stations(monkeypatch):
    import requests

    ts_discovery = [
        {
            "id": "ts-1",
            "station": {
                "type": "Feature",
                "properties": {"id": "100", "label": "London N. Kensington"},
                "geometry": {"type": "Point", "coordinates": [51.521, -0.213]},
            },
        },
    ]

    ts_data = {"values": [{"timestamp": 1704067200000, "value": 8.0}]}

    def mock_get(url, params=None, headers=None, timeout=None):
        from normet.io.defra import _API_BASE

        if f"{_API_BASE}/timeseries" == url and params is not None:
            return MockResponse(ts_discovery)
        if "getData" in url:
            return MockResponse(ts_data)
        return MockResponse([])

    monkeypatch.setattr(requests, "get", mock_get)

    from normet.io.defra import fetch_aurn_measurements

    df = fetch_aurn_measurements(
        pollutant="PM2.5",
        date_from="2024-01-01",
        date_to="2024-01-02",
    )

    assert len(df) == 1
    assert df.loc[0, "value"] == 8.0


@pytest.mark.skipif(not requests_available, reason="requests not installed")
def test_fetch_aurn_measurements_skips_none_values(monkeypatch):
    import requests

    ts_discovery = [
        {
            "id": "ts-1",
            "station": {
                "type": "Feature",
                "properties": {"id": "100", "label": "London N. Kensington"},
                "geometry": {"type": "Point", "coordinates": [51.521, -0.213]},
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
        from normet.io.defra import _API_BASE

        if f"{_API_BASE}/timeseries" == url and params is not None:
            return MockResponse(ts_discovery)
        if "getData" in url:
            return MockResponse(ts_data)
        return MockResponse([])

    monkeypatch.setattr(requests, "get", mock_get)

    from normet.io.defra import fetch_aurn_measurements

    df = fetch_aurn_measurements(
        station="100",
        pollutant="PM2.5",
        date_from="2024-01-01",
        date_to="2024-01-02",
    )

    assert len(df) == 1
    assert df.loc[0, "value"] == 10.0


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
