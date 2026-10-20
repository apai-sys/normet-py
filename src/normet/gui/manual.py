"""Help -> User Manual: where the manual in a language is.

The two books (docs/manual/normet_manual.pdf and normet_manual_zh.pdf) ship inside the
installers (packaging/normet_gui.spec); a checkout of the repository has them in
docs/manual. Where neither is found -- an install from PyPI -- the PDF on GitHub is
opened instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QUrl

#: The manual on GitHub, when no copy is at hand.
ONLINE = "https://github.com/apai-sys/normet-py/raw/main/docs/manual/{book}"


def manual_name(lang: str) -> str:
    """The book in ``lang``: the Chinese one for "zh", else the English."""
    return "normet_manual_zh.pdf" if lang == "zh" else "normet_manual.pdf"


def manual_path(lang: str) -> Path | None:
    """The manual in ``lang`` on this machine: in the installer's bundle, else in the
    checkout normet runs from; None where there is none."""
    book = manual_name(lang)
    roots = []
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        roots.append(Path(bundle))
    roots.append(Path(__file__).resolve().parents[3])  # src/normet/gui -> the checkout
    for root in roots:
        path = root / "docs" / "manual" / book
        if path.is_file():
            return path
    return None


def manual_location(lang: str) -> QUrl:
    """What Help -> User Manual opens: the local PDF, else the one online."""
    path = manual_path(lang)
    if path is not None:
        return QUrl.fromLocalFile(os.fspath(path))
    return QUrl(ONLINE.format(book=manual_name(lang)))
