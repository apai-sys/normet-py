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


def test_a_name_does_not_take_the_facts_after_it():
    """A line of facts joined with " | ": a template's {s} ends at the bar, so each fact is
    translated -- "Backend: {s0}" once took "lightgbm | features: 9 | rows: 5,793" whole,
    and the two facts after it stayed English."""
    from normet.gui import uilang

    line = "Backend: lightgbm   |   features: 9   |   rows: 5,793"
    assert uilang.tr(line, "zh") == "后端：lightgbm | 特征：9 | 行数：5,793"


def test_a_unit_of_one_letter_is_translated():
    """A spin box's " d" after 14, and a window's "14 d": one letter each, which the
    two-letter test for text to translate once let through untranslated."""
    from normet.gui import uilang

    assert uilang.tr(" d", "zh") == " 天" and uilang.tr(" s", "zh") == " 秒"
    assert uilang.tr("365 d", "zh") == "365 天"
    assert uilang.tr("x", "zh") == "x"  # a letter the catalogue has no word for


def test_a_file_dialog_is_titled_in_the_systems_language(monkeypatch):
    """A file dialog is the system's panel, its buttons in the system's language: its
    title too, whichever language the window is in (a Chinese title on English buttons
    was asked against)."""
    from normet.gui import uilang

    monkeypatch.setattr(uilang, "system_language", lambda: "en")
    assert uilang.panel_caption("Open data") == "Open data"
    monkeypatch.setattr(uilang, "system_language", lambda: "zh")
    assert uilang.panel_caption("Open data") == "打开数据"


def test_a_label_head_and_a_text_of_several_lines():
    """As cume's: 'Name:' is the name the catalogue has with the language's colon; a
    text of several lines keeps its lines and their indent; a statistic before a formula
    is translated, one before any other words is not."""
    from normet.gui.uilang import tr

    assert tr("Resolution:", "zh") == "时间分辨率："
    assert tr("Features (predictors)\n  ws", "zh") == "特征（预测变量）\n  ws"
    assert tr("mean $\\mathregular{NO_{2}}$", "zh") == "$\\mathregular{NO_{2}}$ 均值"
    assert tr("mean absolute error", "zh") == "mean absolute error"
