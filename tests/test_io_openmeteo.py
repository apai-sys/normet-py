"""Offline tests for the Open-Meteo adapter (HTTP layer mocked)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from normet.io import openmeteo


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def json(self) -> dict:
        return self._payload


@pytest.fixture()
def fake_archive(monkeypatch):
    """Two hours of synthetic Open-Meteo output; captures request params."""
    calls: list[dict] = []

    payload = {
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
            "temperature_2m": [5.0, 6.0],  # °C
            "dew_point_2m": [1.0, 2.0],
            "relative_humidity_2m": [80.0, 82.0],
            "surface_pressure": [1013.0, 1014.0],  # hPa
            "cloud_cover": [50.0, 100.0],  # %
            "precipitation": [0.0, 2.0],  # mm
            "shortwave_radiation": [0.0, 100.0],  # W/m²
            "wind_speed_10m": [4.0, 0.0],  # m/s
            "wind_direction_10m": [270.0, 0.0],  # deg (from west)
        }
    }

    def fake_request(url, *, params=None, **kwargs):
        calls.append({"url": url, "params": params})
        return _FakeResponse(payload)

    monkeypatch.setattr(openmeteo, "request_with_retry", fake_request)
    return calls


def test_fetch_openmeteo_units_and_shape(fake_archive):
    df = openmeteo.fetch_openmeteo_timeseries(
        sites={"TestSite": (53.5, -2.2)},
        date_from="2024-01-01",
        date_to="2024-01-01",
    )
    assert list(df["site"].unique()) == ["TestSite"]
    assert len(df) == 2
    # ERA5 unit conventions
    assert df["t2m"].iloc[0] == pytest.approx(278.15)  # °C → K
    assert df["sp"].iloc[0] == pytest.approx(101300.0)  # hPa → Pa
    assert df["tcc"].iloc[1] == pytest.approx(1.0)  # % → fraction
    assert df["tp"].iloc[1] == pytest.approx(0.002)  # mm → m
    assert df["ssrd"].iloc[1] == pytest.approx(360000.0)  # W/m² → J/m²
    # wind from the west (270°) blows eastward → u > 0, v ≈ 0
    assert df["u10"].iloc[0] == pytest.approx(4.0, abs=1e-6)
    assert df["v10"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    # request was built for the archive endpoint with UTC hourly fields
    params = fake_archive[0]["params"]
    assert params["timezone"] == "UTC"
    assert "temperature_2m" in params["hourly"]
    assert "boundary_layer_height" in params["hourly"]  # part of the default fetch


def test_fetch_openmeteo_sites_dataframe(fake_archive):
    sites = pd.DataFrame({"site": ["A", "A"], "lat": [50.0, 50.0], "lon": [0.0, 0.0]})
    df = openmeteo.fetch_openmeteo_timeseries(
        sites=sites, date_from="2024-01-01", date_to="2024-01-01"
    )
    assert len(fake_archive) == 1  # duplicate site rows deduplicated
    assert set(df["site"]) == {"A"}
    assert np.isfinite(df["ws"]).all()
