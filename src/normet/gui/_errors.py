"""What a Load dialog says when a reader could not read the file.

pandas' and json's own errors often name no file ("No columns to parse from file",
"Expecting value: line 1 column 1"), and the window's language translates a message
box's text phrase by phrase, so a library's ``"{s0}: {s1}"`` sentence could come out
half translated with no file named at all. The first line is the window's own and
names the file; the reader's sentence keeps a line of its own, as it was written.
"""

from __future__ import annotations

import os
from typing import Any


def read_error(path: Any, exc: BaseException) -> str:
    """The message for a file the reader refused: its name, then what the reader said.

    Parameters
    ----------
    path : path-like
        The file the analyst chose.
    exc : BaseException
        What the reader raised. A sentence that starts with the path (normet's own
        checkers do) has it taken off, since the first line already gives it.

    Returns
    -------
    str
        ``"Could not read <name>.\\n<the reader's sentence>"``.
    """
    full = str(path)
    detail = str(exc)
    if detail.startswith(full):
        detail = detail[len(full) :].lstrip(": ")
    return f"Could not read {os.path.basename(full)}.\n{detail}"
