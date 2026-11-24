"""Alias of :mod:`normet.data.datasets`, kept for the historical import path.

The bundled CSV.GZ assets stay in :mod:`normet.data`, so
``normet.datasets.load_my1_pm25()`` and the conda recipe's test command
keep working.
"""

from __future__ import annotations

import importlib
import sys

sys.modules[__name__] = importlib.import_module("normet.data.datasets")
