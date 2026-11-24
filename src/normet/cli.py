"""Alias of :mod:`normet.app.cli`, kept for the historical import path and
the ``normet`` console script (``normet.cli:main``).
"""

from __future__ import annotations

import importlib
import sys

_impl = importlib.import_module("normet.app.cli")

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
