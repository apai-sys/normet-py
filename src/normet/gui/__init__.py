"""Qt (PySide6) graphical interface for normet.

Launch with ``normet-gui`` or ``python -m normet.gui``.
"""

from __future__ import annotations

__all__ = ["main"]


def main() -> int:
    """Start the normet GUI application."""
    from .app import main as _main

    return _main()
