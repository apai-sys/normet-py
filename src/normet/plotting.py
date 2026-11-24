"""Alias of :mod:`normet.viz.plotting`, kept for the historical import path.

The figure functions moved into :mod:`normet.viz`; this module is the same
module object as the implementation, so ``from normet.plotting import ...``,
attribute access and monkeypatching all keep working.
"""

from __future__ import annotations

import importlib
import sys

sys.modules[__name__] = importlib.import_module("normet.viz.plotting")
