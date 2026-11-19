"""The catalogue engine behind the window's languages (:mod:`.uilang`).

A text is translated through a catalogue of
English templates and their translations (:mod:`.ui_zh` for Chinese). A
template is the English text with ``{n0}``, ``{n1}``, ... where a number stands
(12, 0.35, -1.2e-3) and ``{s0}``, ``{s1}``, ... where a name, a list or other
free text does (a column's name, a file's); its translation uses the same marks
in its own order. What a ``{s}`` holds is written with the catalogue's phrases
(``INNER``) where it has them and as it is otherwise. A date written with its
month's name (2023-Jul-05) is written in the language's order (``DATE_PARTS``).
A text no template matches stays English.

The engine itself lives in :mod:`normet.i18n` (the report's, ``normet.report``,
reads it too); this module keeps the window's historic import path.
"""

from __future__ import annotations

from ..i18n import (  # noqa: F401
    NOT_PROSE,
    TAG_SPLIT,
    build_catalogue,
    figure_font,
    tag_name,
    translate,
)

__all__ = [
    "NOT_PROSE",
    "TAG_SPLIT",
    "build_catalogue",
    "figure_font",
    "tag_name",
    "translate",
]
