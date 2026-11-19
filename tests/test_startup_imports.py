"""Nothing heavy before the window: the modelling stack waits for a run.

The packaged app draws a splash before ``import normet``, and the window's own import
and the layout it opens in (the Assistant beside the results) must not pull the
modelling stack back into that wait. The Assistant used to do exactly that in cume,
by importing a worker module that brought torch with it; these tests keep the next
such reading out of normet's start-up path. One of them keeps PySide6's import hook
from reading every module's source, which costs seconds rather than bytes.

The modules are checked in a subprocess: the rest of the suite has already imported
them into this one.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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


def _hook_probe(tmp_path: Path, patched: bool) -> str:
    """A subprocess that imports a top-level module of its own after Qt.

    The hook writes its entry for the module ``__import__`` returns, which for
    ``import a.b`` is ``a``: a submodule such as ``normet.gui.chatview`` never
    gets one, patched or not (PySide6 6.11.1), so a probe named after one passes
    whatever the hook did. A top-level probe does get one, which the control
    test shows.
    """
    (tmp_path / "normet_hook_probe.py").write_text("VALUE = 1\n", encoding="utf-8")
    lines = ["import sys", f"sys.path.insert(0, {str(tmp_path)!r})"]
    if patched:
        lines.append("from normet.gui.app import _skip_qt_import_scan")
    lines += ["from PySide6.QtWidgets import QApplication", "app = QApplication([])"]
    if patched:
        lines.append("_skip_qt_import_scan(); _skip_qt_import_scan()")
    lines += [
        "import normet_hook_probe",
        "from shibokensupport.feature import pyside_feature_dict",
        "print(dict.__contains__(pyside_feature_dict, 'normet_hook_probe'))",
    ]
    return _run("\n".join(lines) + "\n")


def test_the_qt_import_hook_does_not_read_our_modules(tmp_path: Path) -> None:
    """PySide6 reads the source of every module imported after it, looking for
    ``from __feature__ import``; a module already in ``pyside_feature_dict`` is
    skipped. Importing normet after PySide6 measured 6.2 s against 1.1 s in the
    other order (warm), so the window's start installs a mapping that answers
    "known" for every non-Qt module. This pins the private mapping: a PySide6
    release that stops honouring it makes start-up slow again, not broken, and
    only a test would notice.
    """
    # `in` is True by construction (the mapping answers "known" for non-Qt
    # names); the raw entry is what shows the hook did not write one.
    assert _hook_probe(tmp_path, patched=True) == "False"


def test_without_the_patch_the_hook_records_the_module(tmp_path: Path) -> None:
    """The control: without the patch the same probe gets the hook's raw entry,
    or the test above would pass whatever the hook did (as its first version,
    probing ``normet.gui.chatview``, did)."""
    assert _hook_probe(tmp_path, patched=False) == "True"


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
