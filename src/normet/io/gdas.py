# src/normet/io/gdas.py
"""GDAS1 meteorology downloader (ARL format) for HYSPLIT.

Fetches the weekly GDAS1 (1-degree) files from NOAA ARL's archive so you can
drive :func:`normet.io.run_back_trajectories` when you don't already have local
gridded met. Files are ARL-packed and large (~570 MB each), so downloads are
streamed and cached — an existing file is never re-fetched.

Archive: https://www.ready.noaa.gov/data/archives/gdas1/
Filenames: ``gdas1.<mmm><yy>.w<N>`` (e.g. ``gdas1.apr20.w1``); the week index
covers days 1-7 (w1), 8-14 (w2), 15-21 (w3), 22-28 (w4), 29-31 (w5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..utils._lazy import require
from ..utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["ARL_GDAS1_BASE_URL", "gdas1_filenames", "fetch_gdas1"]

ARL_GDAS1_BASE_URL = "https://www.ready.noaa.gov/data/archives/gdas1"

_GDAS_MONTHS = (
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
)


def _gdas1_week(day: int) -> int:
    """ARL week-of-month index: 1-7→1, 8-14→2, 15-21→3, 22-28→4, 29-31→5."""
    return min((int(day) - 1) // 7 + 1, 5)


def gdas1_filenames(date_from: Any, date_to: Any) -> list[str]:
    """Weekly GDAS1 filenames covering an (inclusive) date range.

    Parameters
    ----------
    date_from, date_to : str or Timestamp
        Inclusive date range. For HYSPLIT **back**-trajectories this must reach
        back to ``receptor_time - hours_back`` (the start of the oldest backward
        window), not just the receptor times.

    Returns
    -------
    list of str
        Unique ``gdas1.<mmm><yy>.w<N>`` names in chronological order.
    """
    d0 = pd.Timestamp(date_from).normalize()
    d1 = pd.Timestamp(date_to).normalize()
    if d1 < d0:
        d0, d1 = d1, d0
    names = [
        f"gdas1.{_GDAS_MONTHS[ts.month - 1]}{ts.year % 100:02d}.w{_gdas1_week(ts.day)}"
        for ts in pd.date_range(d0, d1, freq="D")
    ]
    return list(dict.fromkeys(names))  # ordered unique


def fetch_gdas1(
    date_from: Any,
    date_to: Any,
    cache_dir: str | Path,
    *,
    base_url: str = ARL_GDAS1_BASE_URL,
    overwrite: bool = False,
    on_missing: str = "error",
    chunk_size: int = 1 << 20,
    timeout: int = 60,
) -> list[str]:
    """Download the GDAS1 weekly ARL files covering a date range.

    Streams each weekly file (~570 MB) into ``cache_dir`` and returns the local
    paths, ready to pass as ``met_files`` to :func:`run_back_trajectories`. A
    file already present is reused (set ``overwrite=True`` to force).

    Parameters
    ----------
    date_from, date_to : str or Timestamp
        Inclusive range to cover; see :func:`gdas1_filenames` (remember to
        include the full backward window for back-trajectories).
    cache_dir : str or Path
        Directory to download into / reuse from.
    base_url : str
        Archive base URL (default :data:`ARL_GDAS1_BASE_URL`).
    overwrite : bool, default False
        Re-download even if the file already exists.
    on_missing : {"error", "warn"}, default "error"
        What to do if a weekly file returns HTTP 404 (e.g. a date outside the
        archive's coverage): raise, or warn and skip it.
    chunk_size : int, default 1 MiB
        Streaming chunk size.
    timeout : int, default 60
        Per-request connect/read timeout (seconds).

    Returns
    -------
    list of str
        Local paths of the downloaded/cached files, in chronological order.
    """
    requests = require("requests", hint="pip install requests  (or: pip install normet[data])")
    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    for name in gdas1_filenames(date_from, date_to):
        dest = cache / name
        if dest.exists() and not overwrite:
            log.info("Reusing cached GDAS1 file: %s", dest)
            paths.append(str(dest))
            continue

        url = f"{base_url.rstrip('/')}/{name}"
        log.info("Downloading GDAS1 %s", url)
        tmp = dest.with_name(dest.name + ".part")
        with requests.get(url, stream=True, timeout=timeout) as r:
            if r.status_code == 404:
                msg = f"GDAS1 file not available in archive: {url}"
                if on_missing == "warn":
                    log.warning(msg)
                    continue
                raise FileNotFoundError(msg)
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        fh.write(chunk)
        tmp.rename(dest)
        log.info("Saved %s (%.0f MB)", dest, dest.stat().st_size / 1e6)
        paths.append(str(dest))

    return paths
