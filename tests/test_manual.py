"""The user manual's two books: normet_manual_zh.tex is normet_manual.tex translated."""

from __future__ import annotations

import os
import re

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "manual")


def _outline(name: str) -> dict:
    with open(os.path.join(HERE, name), encoding="utf-8") as fh:
        tex = fh.read()
    return {
        "labels": sorted(re.findall(r"\\label\{([^}]*)\}", tex)),
        "refs": sorted(set(re.findall(r"\\ref\{([^}]*)\}", tex))),
        "screenshots": re.findall(r"\\screenshot\{([^}]*)\}", tex),
        "chapters": len(re.findall(r"^\\chapter", tex, re.M)),
        "sections": len(re.findall(r"^\\section", tex, re.M)),
        "subsections": len(re.findall(r"^\\subsection", tex, re.M)),
    }


def test_the_chinese_manual_is_the_english_one_label_for_label():
    """The same chapters, sections, labels, references and screenshots, in the same
    order: a change made to one book and not the other fails here."""
    en, zh = _outline("normet_manual.tex"), _outline("normet_manual_zh.tex")
    for key in en:
        assert en[key] == zh[key], (key, en[key], zh[key])


def test_every_screenshot_is_there():
    """Each screenshot exists in English; the Chinese book falls back to it where
    figures_zh/ has none."""
    for shot in _outline("normet_manual.tex")["screenshots"]:
        assert os.path.exists(os.path.join(HERE, "figures", shot)), shot
