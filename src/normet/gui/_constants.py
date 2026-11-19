"""Names the window and every panel it builds share.

Kept out of :mod:`.main_window` so a module split out of the window can import
them without importing the window itself.
"""

from __future__ import annotations

TIME_VARS = ("date_unix", "day_julian", "weekday", "hour")

#: Column names auto-recognised as meteorological predictors.
MET_DEFAULTS = {
    "u10",
    "v10",
    "d2m",
    "t2m",
    "blh",
    "sp",
    "ssrd",
    "tcc",
    "tp",
    "rh2m",
    "ws",
    "wd",
    "temp",
    "rh",
    "pressure",
}

DOCS_URL = "https://github.com/apai-sys/normet-py"
