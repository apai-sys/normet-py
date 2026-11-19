"""The window in Chinese, run by tests/test_gui_uilang.py in a process of its own.

Installing .uilang wraps Qt's text setters for the rest of the process; kept
out of the test run's own process.
It also exits as the app does, wrappers installed, so a crash at shutdown
shows as this process failing.
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from normet.gui import uilang  # noqa: E402


def screen_shows_chinese_and_code_reads_english(zh):
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


def main_window_works_in_chinese(zh):
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    uilang.install(app)
    uilang.set_language("zh")
    screen_shows_chinese_and_code_reads_english(uilang)
    uilang.set_language("zh")
    main_window_works_in_chinese(uilang)
    print("uilang smoke OK", flush=True)
