"""The window in Chinese (normet.gui.uilang), offscreen."""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_every_translation_keeps_its_templates_marks():
    from normet.gui import ui_zh

    for en, tr in ui_zh.TEXTS.items():
        assert sorted(re.findall(r"\{[ns]\d+\}", en)) == sorted(re.findall(r"\{[ns]\d+\}", tr)), en


def test_the_window_in_chinese():
    """tests/_uilang_smoke.py: labels, combos, a status line, the main window."""
    here = os.path.dirname(os.path.abspath(__file__))
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "NORMET_UI_LANGUAGE": "zh"}
    proc = subprocess.run(
        [sys.executable, os.path.join(here, "_uilang_smoke.py")],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0 and "uilang smoke OK" in proc.stdout, (
        proc.returncode,
        proc.stdout[-2000:],
        proc.stderr[-4000:],
    )
