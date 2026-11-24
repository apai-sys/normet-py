"""Alias of :mod:`normet.reporting.report`, kept for the historical import path."""

from __future__ import annotations

import importlib
import sys

sys.modules[__name__] = importlib.import_module("normet.reporting.report")
