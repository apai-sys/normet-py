"""Data adapters — offline-safe sanity tests that don't hit the network."""

import importlib.util

import pytest


def test_openaq_module_importable():
    from normet.io import openaq  # noqa: F401


def test_openaq_requires_api_key(monkeypatch):
    from normet.io.openaq import _resolve_key

    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        _resolve_key(None)
    assert _resolve_key("forced-key") == "forced-key"


def test_era5_module_importable():
    from normet.io import era5

    # Default variable list should be a non-empty list of strings
    assert era5.ERA5_AQ_VARIABLES_DEFAULT
    assert all(isinstance(v, str) for v in era5.ERA5_AQ_VARIABLES_DEFAULT)


def test_era5_coerce_sites():
    from normet.io.era5 import _coerce_sites

    out = _coerce_sites({"London": (51.5, -0.1)})
    assert list(out.columns) == ["site", "lat", "lon"]
    assert out.loc[0, "site"] == "London"
    assert out.loc[0, "lat"] == 51.5
    assert out.loc[0, "lon"] == -0.1


def test_eea_module_importable():
    from normet.io import eea

    assert eea.EEA_POLLUTANT_CODES["PM2.5"] == 6001
    assert eea._resolve_pollutant_code("NO2") == 8
    assert eea._resolve_pollutant_code(7) == 7

    with pytest.raises(ValueError):
        eea._resolve_pollutant_code("XYZ")


needs_requests = pytest.mark.skipif(
    importlib.util.find_spec("requests") is None, reason="requests not installed"
)


@needs_requests
def test_openaq_locations_signature():
    """Confirm callable; does not hit the network."""
    from normet.io.openaq import openaq_locations

    assert callable(openaq_locations)


# ---- AURN / DEFRA ----


def test_defra_module_importable():
    from normet.io import defra

    assert defra.AURN_POLLUTANT_CODES["PM2.5"] == 6001
    assert defra.AURN_POLLUTANT_CODES["NO2"] == 8
    assert defra.AURN_POLLUTANT_CODES["O3"] == 7


def test_defra_resolve_pollutant_code():
    from normet.io.defra import _resolve_pollutant_code

    assert _resolve_pollutant_code("PM2.5") == 6001
    assert _resolve_pollutant_code("no2") == 8
    assert _resolve_pollutant_code(7) == 7

    with pytest.raises(ValueError, match="Unknown pollutant"):
        _resolve_pollutant_code("XYZ")


def test_list_aurn_stations_signature():
    from normet.io.defra import list_aurn_stations

    assert callable(list_aurn_stations)


def test_fetch_aurn_measurements_signature():
    from normet.io.defra import fetch_aurn_measurements

    assert callable(fetch_aurn_measurements)
