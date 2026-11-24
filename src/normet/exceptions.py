"""Alias of :mod:`normet.core.exceptions`, kept for the historical import path."""

from __future__ import annotations

import importlib
import sys

sys.modules[__name__] = importlib.import_module("normet.core.exceptions")
