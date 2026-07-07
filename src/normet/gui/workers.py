"""Background execution helpers for the GUI.

Long-running normet calls (training, normalisation, SCM, …) run in a
``QThread`` so the UI stays responsive.  :class:`FunctionWorker` wraps an
arbitrary callable and reports the result, any exception, and log records
back to the GUI thread via Qt signals.

normet's numerical routines cannot be interrupted mid-flight, so "cancel"
is implemented as *abandon*: the active worker is detached (its eventual
result is discarded) and the UI unlocks immediately.  The detached thread
finishes in the background and is then cleaned up.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

log = logging.getLogger(__name__)


class QtLogHandler(logging.Handler, QObject):
    """Logging handler that forwards records to a Qt signal.

    ``logging.Handler.emit`` may be called from any thread; the signal/slot
    connection marshals the message onto the GUI thread.
    """

    message = Signal(str)

    def __init__(self) -> None:
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D102
        try:
            self.message.emit(self.format(record))
        except RuntimeError:
            # The QObject may already be destroyed during shutdown.
            pass


class FunctionWorker(QObject):
    """Run ``fn(*args, **kwargs)`` in a worker thread."""

    finished = Signal(object)
    failed = Signal(str)
    done = Signal()

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception:
            log.exception("Background task failed")
            self.failed.emit(traceback.format_exc(limit=8))
        else:
            self.finished.emit(result)
        finally:
            self.done.emit()


class TaskRunner(QObject):
    """Owns the active worker thread plus any abandoned ("zombie") threads.

    The GUI creates a single ``TaskRunner``; each ``submit`` spins up a fresh
    ``QThread``/``FunctionWorker`` pair.  Only one *active* task may run at a
    time — callers should disable their run buttons while :attr:`busy` is
    True.  :meth:`abandon` detaches the active task so a new one can start.
    """

    started = Signal(str)
    finished = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._active: tuple[QThread, FunctionWorker] | None = None
        self._zombies: list[tuple[QThread, FunctionWorker]] = []
        self.task_name: str = ""

    @property
    def busy(self) -> bool:
        return self._active is not None

    def submit(
        self,
        name: str,
        fn: Callable[..., Any],
        on_result: Callable[[Any], None],
        on_error: Callable[[str], None],
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        """Start *fn* in a background thread. Returns False if already busy."""
        if self.busy:
            return False

        thread = QThread()
        worker = FunctionWorker(fn, *args, **kwargs)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(on_result)
        worker.failed.connect(on_error)
        worker.done.connect(thread.quit)
        thread.finished.connect(lambda t=thread, w=worker: self._on_thread_finished(t, w))

        self._active = (thread, worker)
        self.task_name = name
        self.started.emit(name)
        thread.start()
        return True

    def abandon(self) -> None:
        """Detach the active task; its eventual result is discarded.

        The computation itself cannot be interrupted — it keeps running in the
        background until it returns, then its thread is reaped silently.
        """
        if self._active is None:
            return
        _thread, worker = self._active
        for sig in (worker.finished, worker.failed):
            try:
                sig.disconnect()
            except RuntimeError:
                pass
        # `worker.done -> thread.quit` stays connected so the zombie reaps itself.
        self._zombies.append(self._active)
        self._active = None
        log.info(
            "Abandoned task %r — it will finish in the background and be discarded.", self.task_name
        )
        self.finished.emit()

    def _on_thread_finished(self, thread: QThread, worker: FunctionWorker) -> None:
        worker.deleteLater()
        thread.deleteLater()
        if self._active is not None and self._active[0] is thread:
            self._active = None
            self.finished.emit()
        else:
            self._zombies = [z for z in self._zombies if z[0] is not thread]

    def shutdown(self, wait_ms: int = 500) -> None:
        """Give live threads a brief chance to exit before the app closes."""
        for thread, _worker in ([self._active] if self._active else []) + self._zombies:
            if thread.isRunning():
                thread.quit()
                thread.wait(wait_ms)
