"""Alias of :mod:`normet.app.mcp_server`, kept for the historical import path
and the ``normet-mcp`` console script (``normet.mcp_server:main``).
"""

from __future__ import annotations

import importlib
import sys

_impl = importlib.import_module("normet.app.mcp_server")

if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
