"""Basemap tiles for the Data Studio's station map.

Three free tile styles, all no-key and all requiring the attribution the map
shows in its corner: CARTO's Positron (light) and Dark Matter, which are drawn
deliberately muted so a scatter of stations stays the most visible thing on
the map, and OpenStreetMap's standard style for more detail. The two CARTO
styles are free for non-commercial use with attribution (they ask a commercial
user for an account); their terms are stricter than OSM's, which is why the
choice is the window's and not a constant.

The map is drawn in Web Mercator: a tile is then a square of equal degrees in
both directions, and stitching a handful of them behind the station scatter
puts the points on recognisable ground. Tiles are cached under
``~/.normet/osm_tiles`` and only the ones covering the view are ever fetched.
Every failure -- offline, no ``requests``, a refused tile -- returns None, and
the map is drawn without a basemap exactly as it was before, so the studio
never depends on this service.

No new dependency: tiles are PNGs, so Pillow (matplotlib's own dependency)
and requests (already optional for the data adapters) are all it takes;
contextily, the usual shortcut, would pull rasterio in for the same handful of
images.
"""

from __future__ import annotations

import io
import math
import os
from typing import Any

import numpy as np

#: The tile styles the map offers: label -> the server, its credit, and a cache
#: folder name (two providers can serve the same z/x/y and must not share a file).
#: All are key-free; CARTO's styles were the natural light/dark pair until they
#: started answering without a key with an "API KEY REQUIRED" tile (checked
#: 2026-10-24), which is why the muted style here is Esri's canvas.
PROVIDERS: dict[str, dict[str, str]] = {
    "OpenStreetMap": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "subdomains": "",
        "credit": "© OpenStreetMap contributors",
        "slug": "osm",
        "max_zoom": "19",
    },
    "Esri Light Gray Canvas": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/"
        "World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        "subdomains": "",
        "credit": "© Esri, HERE, Garmin, © OpenStreetMap contributors",
        "slug": "esri-light",
        "max_zoom": "19",
    },
    "Esri World Imagery": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "subdomains": "",
        "credit": "© Esri, Maxar, Earthstar Geographics",
        "slug": "esri-imagery",
        "max_zoom": "19",
    },
}
#: The style that suits each appearance: the satellite image is dark enough for a
#: dark window, and the standard map carries the most detail for a light one.
DEFAULT_LIGHT = "OpenStreetMap"
DEFAULT_DARK = "Esri World Imagery"

#: Identifies the app, as the tile servers' usage policies require (no blank agents).
USER_AGENT = "normet-gui/1.0 (+https://github.com/apai-sys/normet-py)"
#: At most this many tile rows, and columns, for one view: its surroundings, not a
#: map download.
MAX_TILES_ROWS = 3
MAX_TILES_COLS = 8
#: Where downloaded tiles are kept between sessions (NORMET_TILE_CACHE overrides it,
#: for a build or a test that must not write to the real home).
CACHE_DIR = os.environ.get(
    "NORMET_TILE_CACHE", os.path.join(os.path.expanduser("~"), ".normet", "osm_tiles")
)
#: Beyond this latitude Mercator runs to infinity and the tiles stop at the pole.
MAX_LAT = 85.05112878


def mercator(lat: float | np.ndarray) -> Any:
    """Latitude in degrees -> Web Mercator y, in degrees (the tile's own units)."""
    y = np.log(np.tan(np.pi / 4.0 + np.radians(np.clip(lat, -MAX_LAT, MAX_LAT)) / 2.0))
    return np.degrees(y)


def inverse_mercator(y: float | np.ndarray) -> Any:
    """Web Mercator y (degrees) -> latitude in degrees."""
    return np.degrees(2.0 * np.arctan(np.exp(np.radians(y))) - np.pi / 2.0)


def _tile_xy(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """Fractional tile coordinates of a lon/lat at *zoom* (the slippy-map formula)."""
    n = float(2**zoom)
    lat = float(np.clip(lat, -MAX_LAT, MAX_LAT))
    r = math.radians(lat)
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(r) + 1.0 / math.cos(r)) / math.pi) / 2.0 * n
    return x, y


def _cached_path(provider: str, zoom: int, x: int, y: int) -> str:
    slug = PROVIDERS.get(provider, {}).get("slug", "osm")
    return os.path.join(CACHE_DIR, slug, str(zoom), str(x), f"{y}.png")


def _tile_url(provider: str, zoom: int, x: int, y: int) -> str:
    spec = PROVIDERS.get(provider) or PROVIDERS["OpenStreetMap"]
    url = spec["url"].replace("{z}", str(zoom)).replace("{x}", str(x)).replace("{y}", str(y))
    subs = spec.get("subdomains") or ""
    if "{s}" in url:
        url = url.replace("{s}", subs[(x + y) % len(subs)] if subs else "")
    return url


def _fetch(provider: str, zoom: int, x: int, y: int, session: Any) -> bytes:
    """One tile's PNG bytes: from the cache, else downloaded and then cached."""
    path = _cached_path(provider, zoom, x, y)
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        pass
    resp = session.get(
        _tile_url(provider, zoom, x, y),
        headers={"User-Agent": USER_AGENT},
        timeout=7.0,
    )
    resp.raise_for_status()
    data = resp.content
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.part"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except OSError:  # a read-only home: still draw this time, just do not cache
        pass
    return data


def tile_mosaic(
    lon0: float,
    lon1: float,
    lat0: float,
    lat1: float,
    *,
    width_px: float = 900.0,
    height_px: float = 400.0,
    provider: str = DEFAULT_LIGHT,
) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
    """A stitched image covering the *lon*/*lat* box, and its Mercator extent.

    The grid is widened to the panel's shape (``width_px``/``height_px``) around
    the box, so the image fills the panel without cropping the stations out or
    leaving blank bands: a wide panel shows more of the map either side.

    Returns ``(rgb, (x0, x1, y0, y1))`` in Web Mercator degrees, ready for
    ``imshow(extent=...)`` on axes drawn in :func:`mercator` coordinates; None
    when the tiles cannot be had (offline, no ``requests``/Pillow, no tile).
    """
    try:
        import requests
        from PIL import Image
    except ImportError:  # the GUI still works: the stations need no basemap
        return None
    if provider not in PROVIDERS:
        provider = DEFAULT_LIGHT

    lon0, lon1 = sorted((float(lon0), float(lon1)))
    lat0, lat1 = sorted((float(lat0), float(lat1)))
    ratio = max(float(width_px), 200.0) / max(float(height_px), 120.0)
    top_zoom = int(PROVIDERS[provider].get("max_zoom", "19"))

    zoom = min(top_zoom, 16)
    while True:
        tx_lo = math.floor(_tile_xy(lon0, lat0, zoom)[0])
        tx_hi = math.floor(_tile_xy(lon1, lat1, zoom)[0])
        ty_lo = math.floor(_tile_xy(lon0, lat1, zoom)[1])
        ty_hi = math.floor(_tile_xy(lon0, lat0, zoom)[1])
        rows, cols = ty_hi - ty_lo + 1, tx_hi - tx_lo + 1
        # Fill the panel's width from the rows the data needs: the image must be at
        # least as wide as the panel shows, and a wider image is cropped, not stretched.
        want_rows = rows
        want_cols = max(cols, math.ceil(rows * ratio))
        if want_rows <= MAX_TILES_ROWS and want_cols <= MAX_TILES_COLS:
            break
        if zoom <= 2:  # the world, whatever the box: no fewer tiles to ask for
            want_rows, want_cols = min(2, MAX_TILES_ROWS), min(2, MAX_TILES_COLS)
            break
        zoom -= 1
    want_rows = min(want_rows, MAX_TILES_ROWS)
    want_cols = min(want_cols, MAX_TILES_COLS)
    spans = (want_cols - cols, want_rows - rows)
    tx_lo -= spans[0] // 2
    ty_lo -= spans[1] // 2
    tx_hi = tx_lo + want_cols - 1
    ty_hi = ty_lo + want_rows - 1

    n = 2**zoom
    tx_lo, tx_hi = max(tx_lo, 0), min(tx_hi, n - 1)
    ty_lo, ty_hi = max(ty_lo, 0), min(ty_hi, n - 1)
    tiles: list[list[Any]] = []
    try:
        with requests.Session() as session:
            for ty in range(ty_lo, ty_hi + 1):
                row = []
                for tx in range(tx_lo, tx_hi + 1):
                    img = Image.open(io.BytesIO(_fetch(provider, zoom, tx, ty, session))).convert(
                        "RGB"
                    )
                    row.append(np.asarray(img))
                tiles.append(row)
    except Exception:  # noqa: BLE001 - offline, refused, corrupt: no basemap, no error
        return None

    # Tiles of one zoom are all 256 px, so they stack into one array without resampling.
    mosaic = np.vstack([np.hstack(row) for row in tiles])
    x0 = tx_lo / n * 360.0 - 180.0
    x1 = (tx_hi + 1) / n * 360.0 - 180.0
    y1 = (1.0 - 2.0 * ty_lo / n) * 180.0
    y0 = (1.0 - 2.0 * (ty_hi + 1) / n) * 180.0
    return mosaic, (x0, x1, y0, y1)
