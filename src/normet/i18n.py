"""Alias of :mod:`normet.core.i18n`, kept for the historical import path."""

from __future__ import annotations

import importlib
import sys

sys.modules[__name__] = importlib.import_module("normet.core.i18n")
