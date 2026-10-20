"""How far a long call has come, for whoever is listening (the GUI's status bar).

A loop reports its steps with :func:`report`; nothing is done unless a caller
has asked to hear (:func:`listening`), so a script pays nothing for it. A call
made as one part of a larger one (:func:`part`) reports within its share of
the whole, so the listener always hears the share of the whole that is done.
The state is per thread (and per asyncio task): a call on another thread
reports nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_listener: ContextVar[Callable[[float], None] | None] = ContextVar(
    "normet_progress_listener", default=None
)
#: The share of the whole the call reporting now covers, as (from, to).
_span: ContextVar[tuple[float, float]] = ContextVar("normet_progress_span", default=(0.0, 1.0))


@contextmanager
def listening(listener: Callable[[float], None]) -> Iterator[None]:
    """Have *listener* hear the share done (0 to 1) of the calls made within."""
    token = _listener.set(listener)
    span = _span.set((0.0, 1.0))
    try:
        yield
    finally:
        _span.reset(span)
        _listener.reset(token)


@contextmanager
def part(index: int, count: int) -> Iterator[None]:
    """The calls within as the *index*-th (from 0) of *count* equal parts of the
    current one."""
    lo, hi = _span.get()
    width = (hi - lo) / max(count, 1)
    token = _span.set((lo + width * index, lo + width * (index + 1)))
    try:
        yield
    finally:
        _span.reset(token)


def report(done: int, total: int) -> None:
    """*done* of *total* steps of the current call are done."""
    listener = _listener.get()
    if listener is None or total <= 0:
        return
    lo, hi = _span.get()
    listener(lo + (hi - lo) * min(max(done / total, 0.0), 1.0))
