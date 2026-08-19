"""Test the PyInstaller-frozen-app joblib fix (normet.gui.app._configure_frozen_runtime).

Deliberately does NOT gate on PySide6: the function under test only touches
``sys`` and ``joblib``, both always available, so this regression test runs
in every environment, not just ones with the GUI extras installed.

Background: joblib's default 'loky' backend launches worker processes by
re-executing ``sys.executable``. Inside a PyInstaller bundle that is the GUI
binary itself, so workers die immediately and any joblib.Parallel() call
without an explicit backend (pdp, scm_all, placebo_in_space/time,
uncertainty_bands) raises TerminatedWorkerError. The fix switches joblib's
default to the threading backend when ``sys.frozen`` is set.
"""

from __future__ import annotations

import sys

import joblib.parallel
import pytest

from normet.gui.app import _configure_frozen_runtime


@pytest.fixture(autouse=True)
def _restore_joblib_backend():
    original = joblib.parallel.DEFAULT_BACKEND
    had_frozen = hasattr(sys, "frozen")
    yield
    joblib.parallel.DEFAULT_BACKEND = original
    if not had_frozen:
        if hasattr(sys, "frozen"):
            del sys.frozen
    else:
        sys.frozen = True


def test_unfrozen_leaves_backend_untouched():
    if hasattr(sys, "frozen"):
        del sys.frozen
    joblib.parallel.DEFAULT_BACKEND = "loky"
    _configure_frozen_runtime()
    assert joblib.parallel.DEFAULT_BACKEND == "loky"


def test_frozen_switches_to_threading():
    sys.frozen = True
    joblib.parallel.DEFAULT_BACKEND = "loky"
    _configure_frozen_runtime()
    assert joblib.parallel.DEFAULT_BACKEND == "threading"


def test_frozen_parallel_actually_uses_threads_not_processes():
    """End-to-end: after the fix, a bare Parallel(n_jobs=...) call (as used
    by pdp/scm_all/placebo/uncertainty_bands) runs in-process."""
    import os
    import threading

    from joblib import Parallel, delayed

    sys.frozen = True
    _configure_frozen_runtime()

    main_pid = os.getpid()
    pids_and_threads = Parallel(n_jobs=2)(
        delayed(lambda i: (os.getpid(), threading.current_thread().name))(i) for i in range(4)
    )
    pids = {p for p, _ in pids_and_threads}
    assert pids == {main_pid}, "expected in-process threads, got separate worker processes"
