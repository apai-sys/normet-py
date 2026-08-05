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


# ---- UKAQ (archives + aurn_live) ----


def test_ukaq_module_importable():
    from normet.io import ukaq

    assert ukaq.UKAQ_SOURCES["aurn"]["data"]
    assert callable(ukaq.list_ukaq_stations)
    assert callable(ukaq.fetch_ukaq_measurements)


def test_ukaq_aurn_live_resolve_pollutant_code():
    from normet.io.ukaq import _aurn_live_resolve_pollutant_code

    assert _aurn_live_resolve_pollutant_code("PM2.5") == 6001
    assert _aurn_live_resolve_pollutant_code("no2") == 8
    assert _aurn_live_resolve_pollutant_code("noxasno2") == 9
    assert _aurn_live_resolve_pollutant_code(7) == 7

    with pytest.raises(ValueError, match="Unknown pollutant"):
        _aurn_live_resolve_pollutant_code("XYZ")


def test_ukaq_check_source_accepts_archives_and_aurn_live():
    from normet.io.ukaq import _check_source

    assert _check_source("AURN") == "aurn"
    assert _check_source("aurn_live") == "aurn_live"
    with pytest.raises(ValueError, match="Unknown source"):
        _check_source("nope")


def test_ukaq_split_label():
    from normet.io.ukaq import _aurn_live_split_label

    assert _aurn_live_split_label("Manchester Piccadilly-Nitrogen dioxide (air)") == (
        "Manchester Piccadilly"
    )
