"""The OpenStreetMap layer under the Data Studio's station map.

No Qt and no network: the tile server is faked, and what is checked is the
geometry the map depends on -- Mercator round trips, the tile grid chosen for
a panel, the stitched image's extent, and the cache that keeps a second
search off the network.
"""

from __future__ import annotations

import io
import math

import pytest

pytest.importorskip("PIL")
pytest.importorskip("requests")

from normet.gui import _basemap


def _png(colour: tuple[int, int, int]) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (256, 256), colour).save(buf, format="PNG")
    return buf.getvalue()


def test_mercator_round_trip():
    for lat in (-60.0, -5.0, 0.0, 51.5, 53.48, 80.0):
        assert float(_basemap.inverse_mercator(float(_basemap.mercator(lat)))) == pytest.approx(
            lat, abs=1e-9
        )
    # Mercator stretches a degree of latitude away from the equator by 1/cos(lat).
    stretched = float(_basemap.mercator(54.0)) - float(_basemap.mercator(53.0))
    at_equator = float(_basemap.mercator(1.0)) - float(_basemap.mercator(0.0))
    assert stretched / at_equator == pytest.approx(1.0 / math.cos(math.radians(53.5)), rel=1e-3)


def test_tile_mosaic_stitches_and_covers_the_box(monkeypatch):
    calls: list[tuple[int, int, int]] = []

    def fake_fetch(provider, zoom, x, y, session):
        calls.append((zoom, x, y))
        return _png((x % 256, y % 256, 0))

    monkeypatch.setattr(_basemap, "_fetch", fake_fetch)
    got = _basemap.tile_mosaic(6.0, 15.0, 47.0, 55.0, width_px=770, height_px=330)
    assert got is not None
    image, (x0, x1, y0, y1) = got
    assert image.ndim == 3 and image.shape[2] == 3

    # Whole 256 px tiles stack into one image, and every one of them was fetched once.
    rows, cols = image.shape[0] // 256, image.shape[1] // 256
    assert image.shape[0] % 256 == 0 and image.shape[1] % 256 == 0
    assert rows <= _basemap.MAX_TILES_ROWS and cols <= _basemap.MAX_TILES_COLS
    assert len(calls) == rows * cols
    assert len(set(calls)) == len(calls)
    assert len({zoom for zoom, _, _ in calls}) == 1

    # The image covers the searched box, and is at least as wide as the panel shows,
    # so a wide panel crops the map rather than letterboxing it.
    assert x0 < 6.0 and x1 > 15.0
    assert y0 < float(_basemap.mercator(47.0)) and y1 > float(_basemap.mercator(55.0))
    assert cols / rows >= 770 / 330


def test_tile_mosaic_covers_a_single_point(monkeypatch):
    """One station has no box to fit: the map still needs tiles around it."""
    monkeypatch.setattr(_basemap, "_fetch", lambda *a: _png((1, 2, 3)))
    got = _basemap.tile_mosaic(13.4, 13.4, 52.5, 52.5, width_px=800, height_px=400)
    assert got is not None
    image, (x0, x1, y0, y1) = got
    assert x0 < 13.4 < x1
    assert y0 < float(_basemap.mercator(52.5)) < y1
    assert image.shape[0] % 256 == 0


def test_tile_mosaic_is_none_when_the_server_fails(monkeypatch):
    """Offline is not an error: the window draws the stations without a basemap."""

    def offline(*args, **kwargs):
        raise OSError("no route to host")

    monkeypatch.setattr(_basemap, "_fetch", offline)
    assert _basemap.tile_mosaic(6.0, 15.0, 47.0, 55.0) is None


def test_tiles_are_cached_on_disk(monkeypatch, tmp_path):
    """A second search over the same tiles must not ask the server again."""
    monkeypatch.setattr(_basemap, "CACHE_DIR", str(tmp_path))
    hits: list[str] = []

    class Resp:
        content = _png((9, 9, 9))

        def raise_for_status(self) -> None:
            return None

    class Session:
        def get(self, url, **kwargs):
            hits.append(url)
            assert "normet" in kwargs["headers"]["User-Agent"]
            return Resp()

    first = _basemap._fetch("OpenStreetMap", 7, 66, 42, Session())
    second = _basemap._fetch("OpenStreetMap", 7, 66, 42, Session())
    assert first == second
    assert len(hits) == 1
    assert (tmp_path / "osm" / "7" / "66" / "42.png").exists()


def test_zoom_follows_density(monkeypatch):
    """A city-sized box asks for a finer zoom than a continent-sized one."""
    zooms: list[int] = []

    def fake_fetch(provider, zoom, x, y, session):
        zooms.append(zoom)
        return _png((0, 0, 0))

    monkeypatch.setattr(_basemap, "_fetch", fake_fetch)
    _basemap.tile_mosaic(13.3, 13.5, 52.4, 52.6, width_px=800, height_px=400)
    city = max(zooms)
    zooms.clear()
    _basemap.tile_mosaic(-10.0, 30.0, 35.0, 60.0, width_px=800, height_px=400)
    assert city > max(zooms)


def test_extent_is_exactly_the_tiles_fetched(monkeypatch):
    """The image is drawn where its tiles belong: the extent comes from their
    row/column indices, not from the searched box (a mis-registered basemap
    would put the stations in the sea)."""
    tiles: list[tuple[int, int, int]] = []

    def fake_fetch(provider, zoom, x, y, session):
        tiles.append((zoom, x, y))
        return _png((0, 0, 0))

    monkeypatch.setattr(_basemap, "_fetch", fake_fetch)
    image, (x0, x1, y0, y1) = _basemap.tile_mosaic(
        6.0, 15.0, 47.0, 55.0, width_px=800, height_px=400
    )
    zoom = tiles[0][0]
    n = 2**zoom
    xs = [t[1] for t in tiles]
    ys = [t[2] for t in tiles]
    assert x0 == pytest.approx(min(xs) / n * 360.0 - 180.0)
    assert x1 == pytest.approx((max(xs) + 1) / n * 360.0 - 180.0)
    assert y1 == pytest.approx((1.0 - 2.0 * min(ys) / n) * 180.0)
    assert y0 == pytest.approx((1.0 - 2.0 * (max(ys) + 1) / n) * 180.0)
    assert image.shape[1] / image.shape[0] == pytest.approx(
        (max(xs) - min(xs) + 1) / (max(ys) - min(ys) + 1)
    )


def test_each_provider_asks_its_own_server_for_the_same_tile():
    """Providers differ in URL shape (Esri puts y before x), so the template is
    what keeps them honest -- and each style caches in its own folder."""
    urls = {name: _basemap._tile_url(name, 7, 66, 42) for name in _basemap.PROVIDERS}
    assert urls["OpenStreetMap"].endswith("/7/66/42.png")
    assert "/World_Light_Gray_Base/MapServer/tile/7/42/66" in urls["Esri Light Gray Canvas"]
    assert "/World_Imagery/MapServer/tile/7/42/66" in urls["Esri World Imagery"]
    assert len(set(urls.values())) == len(urls)
    slugs = {_basemap._cached_path(name, 7, 66, 42) for name in _basemap.PROVIDERS}
    assert len(slugs) == len(_basemap.PROVIDERS)
