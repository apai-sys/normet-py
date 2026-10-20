"""The window in Chinese (normet.gui.uilang), offscreen."""

from __future__ import annotations

import os
import re

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def zh():
    from PySide6.QtWidgets import QApplication

    from normet.gui import uilang

    app = QApplication.instance() or QApplication([])
    if not uilang._state["installed"]:
        uilang._wrap_all()
        uilang._state["installed"] = True
    uilang.set_language("zh")
    yield uilang
    uilang.set_language("en")  # the wrappers stay; in English they change nothing
    del app


def test_every_translation_keeps_its_templates_marks():
    from normet.gui import ui_zh

    for en, tr in ui_zh.TEXTS.items():
        assert sorted(re.findall(r"\{[ns]\d+\}", en)) == sorted(re.findall(r"\{[ns]\d+\}", tr)), en


def test_the_screen_shows_chinese_and_the_code_reads_english(zh):
    from PySide6.QtWidgets import QComboBox, QLabel, QPushButton

    label = QLabel()
    label.setText("Resolution")  # translated as set; a constructor's, as first shown
    assert label.text() == "Resolution"
    assert zh._orig["QLabel.text"](label) == "时间分辨率"

    box = QComboBox()
    box.addItems(["As recorded", "Daily", "PM2.5"])
    seen = []
    box.currentTextChanged.connect(seen.append)
    box.setCurrentText("Daily")
    assert box.currentText() == "Daily" and seen == ["Daily"]
    assert zh._orig["QComboBox.itemText"](box, 1) == "日均"
    assert zh._orig["QComboBox.itemText"](box, 2) == "PM2.5"  # data stays as it is

    # A status line built from parts: each part translated where it can be.
    lbl = QLabel("8,760 rows × 19 columns   |   missing values: 0")
    zh.set_language("zh")  # the walk a switch does
    assert zh._orig["QLabel.text"](lbl) == "8,760 行 × 19 列 | 缺失值：0"

    btn = QPushButton()
    btn.setText("▶ Train model")
    assert zh._orig["QAbstractButton.text"](btn) == "▶ 训练模型"
    zh.set_language("en")
    assert zh._orig["QAbstractButton.text"](btn) == "▶ Train model"


def test_the_main_window_works_in_chinese(zh):
    from normet.gui.main_window import MainWindow

    win = MainWindow()
    try:
        win.start_layout()
        win.load_example()
        assert win.target_combo.currentText() == "PM2.5"
        assert win.tabs.tabText(0) == "Data"
        assert zh._orig["QTabWidget.tabText"](win.tabs, 0) == "数据"
        assert "t2m" in win._selected_features()
    finally:
        win.close()
