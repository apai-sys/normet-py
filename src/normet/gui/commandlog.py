"""What the window did, written as the normet calls that do the same.

Every action that decides a result
-- opening a table, training, normalising, decomposing, a rolling run, partial
dependence -- is recorded as the call that performs it, so an analysis done by
clicking can be repeated, changed and handed on as a script.

An entry is written from the call the window actually made -- the function and
the arguments its task was started with -- and only once that task succeeded:
the log says what happened, not what the panel shows now. The tables and the
model the window passes are named as the script names them (``df``,
``df_prep``, ``model``). Where the window did something no call expresses (a
table from the Data Studio, held only in memory), the entry says so in a note
instead of a line that would quietly do something else.
``tests/test_command_log.py`` runs the script this module writes and compares
it with the window's own result.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

#: Width the generated code wraps at.
WIDTH = 88


def literal(value: Any) -> str:
    """A Python literal for a parameter value, exact enough to re-run."""
    if isinstance(value, (np.bool_, bool)):
        return repr(bool(value))
    if isinstance(value, np.integer):
        return repr(int(value))
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if value != value:
            return "float('nan')"
        if value in (float("inf"), float("-inf")):
            return f"float('{'-' if value < 0 else ''}inf')"
        # repr round-trips, but 0.30000000000000004 reads as a mistake: 12 digits
        # keep every value a person types and drop the float noise.
        short = round(value, 12)
        return repr(int(short) if short == int(short) and abs(short) < 1e15 else short)
    if isinstance(value, np.ndarray):
        return "[" + ", ".join(literal(v) for v in value.tolist()) + "]"
    if isinstance(value, list):
        return "[" + ", ".join(literal(v) for v in value) + "]"
    if isinstance(value, tuple):
        inner = ", ".join(literal(v) for v in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{literal(k)}: {literal(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return repr(str(value))
    return repr(value)


def call(
    target: str | None,
    name: str,
    positional: Sequence[str] = (),
    keywords: Sequence[tuple[str, str]] = (),
) -> str:
    """``target = name(positional, keyword=…)``, wrapped under the open bracket."""
    args = list(positional) + [f"{k}={v}" for k, v in keywords]
    head = f"{target} = {name}(" if target else f"{name}("
    one = head + ", ".join(args) + ")"
    if len(one) <= WIDTH or len(args) < 2:
        return one
    pad = " " * len(head)
    lines: list[str] = []
    current = head + args[0] + ("," if len(args) > 1 else ")") + " "
    for i, arg in enumerate(args[1:], start=1):
        piece = arg + ("," if i < len(args) - 1 else ")")
        if len(current) + len(piece) + 1 > WIDTH:
            lines.append(current.rstrip())
            current = pad
        current += piece + " "
    lines.append(current.rstrip())
    return "\n".join(lines)


@dataclass
class Entry:
    """One action: its heading, the code that repeats it, and what the code leaves out."""

    title: str
    code: str
    notes: tuple[str, ...] = ()
    imports: tuple[str, ...] = ()  # whole import lines this entry needs
    when: str = ""

    def text(self) -> str:
        """The entry as the panel shows it: heading, notes, code."""
        head = f"# {self.when}  {self.title}" if self.when else f"# {self.title}"
        return "\n".join([head, *(f"#   {n}" for n in self.notes), self.code])


@dataclass
class CommandLog:
    """The session's actions in order, as a panel listing and as a runnable script."""

    entries: list[Entry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, entry: Entry | None) -> Entry | None:
        """Record *entry* (None is ignored, so a caller may hand over what it has)."""
        if entry is None:
            return None
        if not entry.when:
            entry.when = datetime.datetime.now().strftime("%H:%M:%S")
        self.entries.append(entry)
        return entry

    def clear(self) -> None:
        self.entries.clear()

    def text(self) -> str:
        """What the panel shows."""
        if not self.entries:
            return (
                "# Nothing yet. Open a table and run a step: every action appears here as\n"
                "# the normet call that repeats it, and Save as Script… writes them out."
            )
        return "\n\n".join(e.text() for e in self.entries)

    def script(self, *, title: str = "") -> str:
        """The log as a file that runs: a header, the imports it needs, the entries."""
        imports: list[str] = []
        for e in self.entries:
            imports += [i for i in e.imports if i not in imports]
        try:
            from normet import __version__ as version
        except Exception:  # pragma: no cover
            version = "?"
        when = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        head = [
            '"""' + (title or "A normet session") + ".",
            "",
            f"Written by normet {version} on {when}, from the window's command log.",
            "Every command is the call behind one action, in the order they were taken;",
            "running this file repeats the analysis.",
            '"""',
        ]
        body = "\n\n".join(e.text() for e in self.entries)
        return "\n".join([*head, "", *sorted(imports), "", body]) + "\n"


# -- the window's actions, as entries -------------------------------------------------------
def open_table(path: str, zone: str | None = None) -> Entry:
    """A table file read as the window reads one (:func:`normet.io.read_time_series`):
    its times found, parsed and named 'date', and put on UTC from ``zone``."""
    call = f"df = read_time_series({literal(path)}"
    call += f", zone={literal(zone)})" if zone and zone.upper() != "UTC" else ")"
    return Entry("Open data", call, imports=("from normet.io import read_time_series",))


def load_example(example: Any = None) -> Entry:
    """An example as the window loads one (``._examples.EXAMPLES``); the synthetic site
    where none is named."""
    if example is None:
        from ._examples import EXAMPLES

        example = EXAMPLES["synthetic"]
    title = "Load example data" if example.key == "synthetic" else f"Load example: {example.title}"
    return Entry(title, example.code, imports=example.imports)


def exclude(lines: Sequence[str]) -> Entry:
    """The rows the Data tab's Exclude box leaves out of every step: the table as loaded
    (``df``) without them, as ``df_kept`` -- which the steps after it read. Cleared, every
    row is kept again."""
    if not lines:
        return Entry("Keep every row", "df_kept = df", notes=("The Exclude box was cleared.",))
    from ._data_tab import parse_exclusions

    spans = parse_exclusions(lines)
    conds = "\n    | ".join(
        f'df["date"].between({literal(str(a))}, {literal(str(b))})' for a, b in spans
    )
    return Entry(
        "Leave rows out",
        f"left_out = (\n    {conds}\n)\ndf_kept = df[~left_out].reset_index(drop=True)",
        notes=tuple(f"left out: {line}" for line in lines),
    )


def from_memory(label: str) -> Entry:
    """A table the window was handed in memory (the Data Studio's), which no call reads."""
    return Entry(
        f"Table from {label}",
        "df = pd.read_csv('<the table, saved as CSV>')  # replace with its path",
        notes=(
            "This table was handed to the window in memory, not read from a file.",
            "Save it as CSV where it was made (Data Studio → Save CSV) and read it here.",
        ),
        imports=("import pandas as pd",),
    )


#: The calls the window's tasks make, by function: (title, what the result is
#: assigned to, the import line, a note where the call is the GUI's own helper).
_KNOWN: dict[str, tuple[str, str, str, str]] = {
    "normet.model.train.build_model": (
        "Train model",
        "df_prep, model",
        "from normet import build_model",
        "",
    ),
    "normet.analysis.normalise.normalise": (
        "Normalise",
        "normalised",
        "from normet import normalise",
        "",
    ),
    "normet.analysis.decomposition.decompose": (
        "Decompose",
        "decomposition",
        "from normet import decompose",
        "",
    ),
    "normet.analysis.rolling.rolling": (
        "Rolling normalisation",
        "rolled",
        "from normet import rolling",
        "",
    ),
    "normet.analysis.pdp.pdp": ("Partial dependence", "pd_values", "from normet import pdp", ""),
    "normet.gui._chronos.load_estimator": (
        "Load Chronos-2",
        "df_prep, model",
        "from normet.gui._chronos import load_estimator",
        "The GUI's own helper around normet.foundation.Chronos2Estimator.",
    ),
    "normet.gui._chronos.deweather": (
        "De-weather (Chronos-2)",
        "normalised",
        "from normet.gui._chronos import deweather",
        "The GUI's own helper around Chronos2Estimator.deweather.",
    ),
    "normet.gui._chronos.finetune": (
        "Fine-tune Chronos-2",
        "model",
        "from normet.gui._chronos import finetune",
        "The GUI's own helper around Chronos2Estimator.finetune.",
    ),
    "normet.gui._multiscale.compute_multiscale": (
        "Multi-scale decomposition",
        "multiscale",
        "from normet.gui._multiscale import compute_multiscale",
        "The GUI's own helper over normet.rolling at three window widths.",
    ),
    "normet.gui._trend.trend_call": (
        "Trend test",
        "trend_result",
        "from normet.gui._trend import trend_call",
        "The GUI's own helper around normet.trend on the series the window holds.",
    ),
}


def _qualname(fn: Callable[..., Any]) -> str:
    return f"{getattr(fn, '__module__', '')}.{getattr(fn, '__name__', '')}"


def task(
    fn: Callable[..., Any],
    args: Sequence[Any],
    kwargs: dict[str, Any],
    names: Sequence[tuple[Any, str]],
) -> Entry | None:
    """The entry for a task the window ran: ``fn(*args, **kwargs)``.

    *names* pairs each object the window holds (its tables, its model, a
    series of a result) with the expression the script names it by; an
    argument found there is written as that name, by identity. A function the
    log does not know (nothing but the calls above decide a result) gives None.
    """
    known = _KNOWN.get(_qualname(fn))
    if known is None:
        return None
    title, target, imp, note = known
    notes = [note] if note else []

    def render(value: Any) -> str:
        for obj, name in names:
            if value is obj:
                return name
        if isinstance(value, (pd.DataFrame, pd.Series)) or (
            hasattr(value, "__dict__") and not isinstance(value, type)
        ):
            notes.append(f"An object the window held ({type(value).__name__}) is not named here.")
            return "None  # " + type(value).__name__
        return literal(value)

    positional = [render(a) for a in args]
    keywords = [(k, render(v)) for k, v in kwargs.items() if k != "verbose"]
    return Entry(
        title, call(target, fn.__name__, positional, keywords), notes=tuple(notes), imports=(imp,)
    )
