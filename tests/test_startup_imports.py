"""Nothing heavy before the window: the modelling stack waits for a run.

The packaged app draws a splash before ``import normet``, and the window's own import
and the layout it opens in (the Assistant beside the results) must not pull the
modelling stack back into that wait. The Assistant used to do exactly that in cume,
by importing a worker module that brought torch with it; these two tests keep the next
such reading out of normet's start-up path.

The modules are checked in a subprocess: the rest of the suite has already imported
them into this one.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("PySide6")

_HEAVY = ("torch", "sklearn", "flaml", "chronos")


def _run(code: str) -> str:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_importing_the_window_does_not_load_the_modelling_stack():
    assert (
        _run(
            "import sys, normet.gui.main_window; "
            f"print([m for m in {_HEAVY!r} if m in sys.modules])"
        )
        == "[]"
    )


def test_the_layout_the_window_opens_in_does_not_load_the_modelling_stack():
    code = (
        "import sys\n"
        "from PySide6.QtWidgets import QApplication\n"
        "app = QApplication([])\n"
        "from normet.gui.main_window import MainWindow\n"
        "win = MainWindow(); win.start_layout()\n"
        f"print([m for m in {_HEAVY!r} if m in sys.modules])\n"
    )
    assert _run(code) == "[]"
