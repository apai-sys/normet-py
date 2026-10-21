"""The Assistant's carry_out: normet's API, written as code, run on the window's table.

The window's own steps (train, normalise, decompose, rolling, ...) are tools of their
own; everything else normet's API does -- an option the panel does not offer, a function
the window has no button for (detect_events, cv_score, scm_robust, fetch_ukaq_measurements,
...) -- the model writes as code, shown to the analyst on a card and run when they apply
it. This module is the language that code is in, the API it reads (``api_help``) and the
running of it (``run_code``): here Python, with normet imported. A window on another
language's normet (normet-r's) has a module of this name of its own.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import traceback
from typing import Any

import pandas as pd

#: The language carry_out's code is written in, as the system prompt says it.
LANGUAGE = "Python"
#: How the code is written, for the system prompt: what is imported and what is bound.
CONVENTIONS = (
    "Python: `import normet` is done (and pandas as pd, numpy as np); `df` is the table "
    "open (a copy), `df_prep` and `model` the prepared table and the model trained (None "
    "before training). print() what the analyst should read; a pandas DataFrame left in "
    "a variable is listed with its head. The code runs in the window's process, not in a "
    "shell: no files are written unless the code writes them."
)
#: How much of the output, and of each table, is handed back.
MAX_OUTPUT = 6000
HEAD_ROWS = 8


def api_names() -> list[str]:
    """normet's public calls (its ``__all__``), functions first."""
    import normet

    names = list(getattr(normet, "__all__", []))
    return [n for n in names if callable(getattr(normet, n, None))]


def api_help(name: str = "") -> str:
    """normet's API: with no name, its calls listed, a line each; with a name, the call's
    signature and its docstring."""
    import normet

    names = api_names()
    if not name.strip():
        lines = ["normet's API (api_help(name) shows a call in full):"]
        for n in names:
            doc = inspect.getdoc(getattr(normet, n)) or ""
            lines.append(f"- {n}: {doc.splitlines()[0] if doc else ''}")
        return "\n".join(lines)
    name = name.strip().removeprefix("normet.")
    fn = getattr(normet, name, None)
    if fn is None:
        close = [n for n in names if name.lower() in n.lower()]
        raise ValueError(
            f"normet has no {name!r}"
            + (f"; did you mean {', '.join(close[:5])}?" if close else "")
            + " api_help('') lists the calls."
        )
    try:
        sig = str(inspect.signature(fn))
    except (TypeError, ValueError):
        sig = "(...)"
    doc = inspect.getdoc(fn) or "(no docstring)"
    return _cut(f"normet.{name}{sig}\n\n{doc}")


def _cut(text: str, most: int = MAX_OUTPUT) -> str:
    return (
        text if len(text) <= most else text[:most] + f"\n… ({len(text) - most:,} more characters)"
    )


def _table_text(name: str, table: pd.DataFrame) -> str:
    head = table.head(HEAD_ROWS).to_string(max_cols=12, max_colwidth=24)
    return f"{name}: {len(table):,} rows × {table.shape[1]} columns\n{head}"


def run_code(code: str, df: pd.DataFrame | None, df_prep: Any = None, model: Any = None) -> str:
    """Run *code* with normet on the window's table and say what it made: what it
    printed, then each DataFrame it left in a variable (its size and head). An error is
    said, with its traceback's tail, rather than raised: the model reads it."""
    import numpy as np

    import normet

    ns: dict[str, Any] = {
        "__name__": "__normet_assistant__",
        "normet": normet,
        "pd": pd,
        "np": np,
        "df": None if df is None else df.copy(),
        "df_prep": df_prep,
        "model": model,
    }
    given, table = set(ns), ns["df"]
    out = io.StringIO()
    failed = ""
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            exec(compile(code, "<carry_out>", "exec"), ns)  # noqa: S102 - applied by the analyst
        except Exception:  # noqa: BLE001 -- told to the model
            failed = "".join(traceback.format_exc().splitlines(keepends=True)[-6:])
    parts = []
    printed = out.getvalue().strip()
    if printed:
        parts.append("Output:\n" + printed)
    tables = [
        _table_text(k, v)
        for k, v in ns.items()
        if isinstance(v, pd.DataFrame)
        and (k not in given or (k == "df" and v is not table))  # df: where the code replaced it
        and not k.startswith("_")
    ]
    if tables:
        parts.append("Tables made:\n" + "\n\n".join(tables))
    if failed:
        parts.append("Error:\n" + failed.strip())
    return _cut("\n\n".join(parts) or "It ran, and printed and made nothing.")
