"""The time a task has taken and may still take, on the status bar beside its
progress bar.

While a task runs, the status bar reads ``Running: <task>… · 40 % · 1 min so far ·
~2 min left``. The share done is the task's own where it reports it
(:mod:`normet.utils.progress`: windows, decomposition steps, trials, receptor
times) or the time against the budget it was given (:meth:`TaskClock.expect`);
the progress bar fills with it, and stays a busy bar where nothing is known. The
time left is the time so far scaled by the share still to do, said once the share
and the time are enough to go on. When the task ends, the line its step wrote
says how long it took.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QMainWindow

from .workers import TaskRunner

#: The time left is said once this share is done, and this long has gone by: before
#: then, a guess from the first steps is wider than the time itself.
_LEFT_FROM_SHARE = 0.02
_LEFT_FROM_SECONDS = 10.0


def format_duration(seconds: float) -> str:
    """A time as it is read at a glance: 40 s, 12 min, 1 h 20 min."""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} s"
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60} min"


class TaskClock(QObject):
    """Keeps *window*'s status bar and its ``progress`` bar on the time of the task
    *runner* is running."""

    def __init__(self, window: QMainWindow, runner: TaskRunner) -> None:
        super().__init__(window)
        self._window = window
        self._name = ""
        self._t0 = 0.0
        self._share: float | None = None
        self._budget: float | None = None
        #: How long the task took, once its result is in (``settle``) until it ends.
        self._took: float | None = None
        #: The line last written, to tell whether the task's step has written its own.
        self._line = ""
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._show)
        runner.started.connect(self.start)
        runner.progress.connect(self.report)
        runner.settled.connect(self.settle)
        runner.finished.connect(self.stop)

    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    def start(self, name: str) -> None:
        self._name, self._t0 = name, time.monotonic()
        self._share, self._budget, self._took = None, None, None
        self._window.progress.setRange(0, 0)
        self._timer.start()
        self._show()

    def expect(self, seconds: float) -> None:
        """The task runs to a time budget: its share done is the time against it."""
        if not self._timer.isActive():
            return  # no task running to time (it has ended, or never started)
        self._budget = seconds if seconds > 0 else None
        self._show()

    def report(self, share: float) -> None:
        """The share the task says is done; it never goes back."""
        if not self._timer.isActive():
            return  # a report that arrives after the task's result
        self._share = max(self._share or 0.0, min(share, 1.0))
        self._show()

    def _done_share(self) -> float | None:
        if self._share is not None:
            return self._share
        if self._budget:
            return min(self.elapsed() / self._budget, 0.99)
        return None

    def _show(self) -> None:
        elapsed = self.elapsed()
        share = self._done_share()
        parts = [f"Running: {self._name}…"]
        bar = self._window.progress
        if share is not None:
            parts.append(f"{100 * share:.0f} %")
            if bar.maximum() != 1000:
                bar.setRange(0, 1000)
            bar.setValue(int(1000 * share))
        parts.append(f"{format_duration(elapsed)} so far")
        # Past its budget a task is finishing (FLAML's last fit), for a time no one can say.
        over = self._share is None and self._budget is not None and elapsed >= self._budget
        if (
            share is not None
            and not over
            and share >= _LEFT_FROM_SHARE
            and elapsed >= _LEFT_FROM_SECONDS
        ):
            parts.append(f"~{format_duration(elapsed * (1 - share) / share)} left")
        self._line = " · ".join(parts)
        self._window.statusBar().showMessage(self._line)

    def settle(self) -> None:
        """The task's result is in, its step about to write its line: the clock stops,
        so no tick writes over that line before the task has ended (``stop``)."""
        if self._timer.isActive():
            self._timer.stop()
            self._took = self.elapsed()

    def stop(self) -> None:
        """The task has ended: the line its step wrote says how long it took; where it
        wrote none (it failed, or was abandoned), that it ended."""
        if self._took is None:
            if not self._timer.isActive():
                return
            self.settle()
        took = f"took {format_duration(self._took or 0.0)}"
        self._took = None
        bar = self._window.progress
        bar.setRange(0, 1)
        bar.setValue(0)
        status = self._window.statusBar()
        line = status.currentMessage()
        if not line or line == self._line:
            status.showMessage(f"Ended: {self._name} · {took}")
        else:
            status.showMessage(f"{line} · {took}")
