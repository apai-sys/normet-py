"""The Assistant's conversation, as the panel sets it out.

The analyst's messages sit in a rounded bubble on the right, as wide as their words; the
replies read as their Markdown does, in the window's own font and size -- headings bold at
the text's size, not a browser's sizes, lines a little apart, code shaded and wrapped, a
table ruled; each tool call is one quiet line: a dot, grey while it runs, green when it
answered, red when it was refused (the refusal under it), what it returned a click away;
a proposal is a card with an Apply button.

The colours are mixed from the application's palette -- the text's into the ground's --
and made again when it changes, light or dark: the dark palette's AlternateBase, which the
analyst's messages were shaded with, is a light grey under white text. The text size steps
with Cmd/Ctrl + and - (0 back to the system's; the wheel with the key held), and is kept.
The view follows the system's appearance as it changes (``follow_palette``: its own style
sheet sets its palette, so the panel holding it passes the change on).

A view at the end keeps the end as the conversation grows; scrolled up, it stays where the
analyst is reading, and a button at its corner goes back to the end. Cmd/Ctrl+F finds text
in the conversation, Enter and Shift+Enter stepping through, Esc letting it go. The menu
(a right click) copies what is selected, the message as it was written (Copy text), as the
transcript keeps it (Copy message), or the conversation entire (Copy conversation).

``toPlainText()`` is the conversation as text, as Copy conversation copies it: '› ' before
the analyst's messages, 'Assistant:' before each reply, '● ' before each call and '⎿ '
before a refusal.
"""

from __future__ import annotations

import html
import re
import time
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QEvent, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QKeySequence,
    QPainter,
    QPalette,
    QShortcut,
    QTextDocument,
    QTextDocumentFragment,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

#: Where the conversation's text size is kept (points from the system's size).
ZOOM_KEY = "assistant/zoom"


def _mix(a: QColor, b: QColor, t: float) -> QColor:
    """``b`` mixed into ``a`` by the fraction ``t``."""
    return QColor.fromRgbF(
        a.redF() + (b.redF() - a.redF()) * t,
        a.greenF() + (b.greenF() - a.greenF()) * t,
        a.blueF() + (b.blueF() - a.blueF()) * t,
    )


def contrast(a: QColor, b: QColor) -> float:
    """WCAG's contrast ratio of two colours: 1 (none) to 21 (black on white)."""

    def lum(c: QColor) -> float:
        ch = [
            v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
            for v in (c.redF(), c.greenF(), c.blueF())
        ]
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]

    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


class Look:
    """The conversation's colours and sizes, from the application's palette; the text
    ``zoom`` points from the system's size."""

    def __init__(self, zoom: float = 0.0, palette: QPalette | None = None) -> None:
        pal = palette or QApplication.palette()
        role = QPalette.ColorRole
        ground, ink = pal.color(role.Base), pal.color(role.Text)
        self.dark = dark = ground.lightnessF() < 0.5
        self.ground = ground
        self.text = _mix(ink, ground, 0.08)
        self.muted = _mix(ink, ground, 0.42)
        self.faint = _mix(ink, ground, 0.62)
        self.bubble = _mix(ground, ink, 0.12 if dark else 0.055)
        self.card = _mix(ground, ink, 0.035 if dark else 0.0)
        self.border = _mix(ground, ink, 0.20 if dark else 0.13)
        self.code = _mix(ground, ink, 0.09 if dark else 0.045)
        self.hover = _mix(ground, ink, 0.06 if dark else 0.035)
        self.accent = pal.color(role.Accent) if hasattr(role, "Accent") else pal.color(role.Link)
        self.on_accent = QColor("#ffffff" if self.accent.lightnessF() < 0.62 else "#000000")
        self.ok = QColor("#4cc38a" if dark else "#2f9e5a")
        self.err = QColor("#ff7b72" if dark else "#c62828")
        self.err_ground = _mix(ground, self.err, 0.13 if dark else 0.07)
        base = QApplication.font().pointSizeF()
        self.size = max(9.0, (base if base > 0 else 13.0) + zoom)
        self.small = max(8.0, self.size - 1.5)
        self.code_size = max(8.0, self.size - 2.0)  # a fixed-width face reads larger
        self.mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()
        #: What it was made from: a palette event with the same is no change.
        self.key = (ground.rgb(), ink.rgb(), self.size)

    def font(self, size: float | None = None, *, mono: bool = False, bold: bool = False) -> QFont:
        f = (
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
            if mono
            else QFont(QApplication.font())
        )
        f.setPointSizeF(size or self.size)
        if bold:
            f.setBold(True)
        return f


def sheet(k: Look) -> str:
    """The conversation's style sheet, from its Look."""
    c = QColor.name
    return f"""
QScrollArea#chatview {{ background: {c(k.ground)}; border: none; border-radius: 10px; }}
QScrollArea#chatview > QWidget#qt_scrollarea_viewport {{ background: transparent; }}
QWidget#chat {{ background: transparent; }}
QWidget#chat QLabel {{ color: {c(k.text)}; background: transparent; }}
QWidget#chat QLabel[tone="muted"] {{ color: {c(k.muted)}; }}
QWidget#chat QLabel[tone="faint"] {{ color: {c(k.faint)}; }}
QWidget#chat QLabel[tone="ok"] {{ color: {c(k.ok)}; }}
QWidget#chat QLabel[tone="error"] {{ color: {c(k.err)}; }}
QFrame#you {{ background: {c(k.bubble)}; border-radius: 12px; }}
QFrame#proposal {{ background: {c(k.card)}; border: 1px solid {c(k.border)};
                   border-radius: 10px; }}
QFrame#trouble {{ background: {c(k.err_ground)}; border-radius: 8px; }}
QFrame#tool {{ border-radius: 6px; }}
QFrame#tool[open="true"] {{ background: {c(k.hover)}; }}
QFrame#tool[clickable="true"]:hover {{ background: {c(k.hover)}; }}
QPushButton#apply {{ background: {c(k.accent)}; color: {c(k.on_accent)}; border: none;
                     border-radius: 6px; padding: 4px 14px; font-weight: 600; }}
QPushButton#apply:disabled {{ background: {c(k.border)}; color: {c(k.muted)}; }}
QScrollArea#wide {{ background: transparent; border: none; }}
QScrollArea#wide > QWidget > QWidget {{ background: transparent; }}
QToolButton#to_end {{ background: {c(k.card if k.dark else k.ground)}; color: {c(k.text)};
                      border: 1px solid {c(k.border)}; border-radius: 14px; }}
QToolButton#to_end:hover {{ background: {c(k.hover)}; }}
QFrame#find {{ background: {c(k.ground)}; border: 1px solid {c(k.border)};
               border-radius: 8px; }}
QFrame#find QLineEdit {{ background: transparent; border: none; color: {c(k.text)}; }}
QFrame#find QLabel {{ color: {c(k.muted)}; background: transparent; }}
QFrame#find QToolButton {{ background: transparent; border: none; color: {c(k.text)};
                           padding: 0px 4px; }}
QFrame#find QToolButton:hover {{ background: {c(k.hover)}; border-radius: 4px; }}
"""


def composer_sheet(k: Look) -> str:
    """The box the analyst writes in, set to match: rounded, the buttons inside it."""
    c = QColor.name
    return f"""
QFrame#composer {{ background: {c(k.ground)}; border: 1px solid {c(k.border)};
                   border-radius: 10px; }}
QFrame#composer[focus="true"] {{ border: 1px solid {c(k.accent)}; }}
QPlainTextEdit#composer_input {{ background: transparent; border: none; color: {c(k.text)}; }}
QPushButton#composer_send {{ background: {c(k.accent)}; color: {c(k.on_accent)}; border: none;
                             border-radius: 6px; padding: 4px 14px; font-weight: 600; }}
QPushButton#composer_send:hover {{ background: {c(k.accent.lighter(112))}; }}
QPushButton#composer_send:disabled {{ background: {c(k.border)}; color: {c(k.muted)}; }}
QPushButton#composer_stop, QPushButton#composer_new {{
    background: transparent; color: {c(k.text)}; border: 1px solid {c(k.border)};
    border-radius: 6px; padding: 4px 12px; }}
QPushButton#composer_new:hover {{ background: {c(k.hover)}; }}
QPushButton#composer_stop:disabled, QPushButton#composer_new:disabled {{ color: {c(k.faint)}; }}
"""


#: TeX a model writes inline, as the text it stands for: models write $Q_{\text{exp}}$ and
#: $\to$ in Markdown whether asked to or not, and the panel draws no TeX.
_TEX = {
    r"\to": "→",
    r"\rightarrow": "→",
    r"\leftarrow": "←",
    r"\Rightarrow": "⇒",
    r"\geq": "≥",
    r"\ge": "≥",
    r"\leq": "≤",
    r"\le": "≤",
    r"\neq": "≠",
    r"\ne": "≠",
    r"\approx": "≈",
    r"\sim": "~",
    r"\times": "×",
    r"\cdot": "·",
    r"\pm": "±",
    r"\infty": "∞",
    r"\Delta": "Δ",
    r"\delta": "δ",
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\sigma": "σ",
    r"\mu": "μ",
    r"\chi": "χ",
    r"\lambda": "λ",
    r"\rho": "ρ",
    r"\theta": "θ",
    r"\epsilon": "ε",
    r"\sum": "Σ",
    r"\%": "%",
    r"\,": " ",
    r"\;": " ",
    r"\quad": " ",
    r"\left": "",
    r"\right": "",
    r"\ldots": "…",
    r"\dots": "…",
}
_MATH_SRC = r"\$\$(.+?)\$\$|(?<![\w$])\$(?=\S)([^$\n]*?\S)\$(?![\w$])"
#: A $...$ or $$...$$ span; (the second pattern) code spans matched first so a dollar in
#: code is left alone.
_MATH = re.compile(_MATH_SRC, re.S)
_MATH_OR_CODE = re.compile(r"(```.*?```|`[^`\n]*`)|" + _MATH_SRC, re.S)


def _math(src: str) -> str:
    """Inline TeX as HTML text: commands as their symbols, _{x} and ^{x} as sub- and
    superscripts."""
    s = src.strip()
    s = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", s)
    for cmd in sorted(_TEX, key=len, reverse=True):
        s = re.sub(re.escape(cmd) + r"(?![A-Za-z])", lambda _m, c=cmd: _TEX[c], s)
    s = html.escape(s, quote=False)
    s = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"\^\{([^{}]*)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"_(\w)", r"<sub>\1</sub>", s)
    s = re.sub(r"\^(\w)", r"<sup>\1</sup>", s)
    return s.replace("{", "").replace("}", "")


def _md(md: str) -> str:
    """The model's Markdown as Qt's HTML (from ``<body``), as Qt reads GitHub's dialect,
    with inline $...$ TeX as text (``_math``). A dollar sign that opens no formula
    ("$5 and $10") stays one, and so does one in code."""
    maths: list[str] = []

    def keep(m: re.Match[str]) -> str:
        if m.group(1) is not None:  # code: untouched
            return m.group(0)
        maths.append(m.group(2) if m.group(2) is not None else m.group(3))
        return f"NORMETMATH{len(maths) - 1}X"

    doc = QTextDocument()
    doc.setMarkdown(
        _MATH_OR_CODE.sub(keep, md), QTextDocument.MarkdownFeature.MarkdownDialectGitHub
    )
    text = doc.toHtml()
    text = text[text.find("<body") :] if "<body" in text else text
    return re.sub(r"NORMETMATH(\d+)X", lambda m: _math(maths[int(m.group(1))]), text)


def markdown_html(md: str, k: Look) -> str:
    """The model's Markdown as Qt's HTML, set for the panel: a heading at the body's own
    size (bold only), paragraphs and list items closer, lines a little apart, code shaded
    (a block's lines wrapped), a table ruled in the palette's line colour, its first row
    shaded, TeX in dollars as text (``_md``)."""
    text = _md(md)
    text = re.sub(
        r"font-size:(xx-large|x-large|large|medium|small|x-small);",
        f"font-size:{k.size:.1f}pt;",
        text,
    )
    # A heading an ordinary paragraph: Qt sizes <h1>-<h6> itself, whatever the span says.
    text = re.sub(r"<h[1-6](\s[^>]*)?>", r"<p\1>", text)
    text = re.sub(r"</h[1-6]>", "</p>", text)
    text = text.replace(
        "margin-top:12px; margin-bottom:12px;", "margin-top:3px; margin-bottom:6px;"
    )
    text = re.sub(r'<(p|li) style="', r'<\1 style=" line-height:130%;', text)
    code = k.code.name()

    def table(m: re.Match[str]) -> str:
        t = re.sub(
            r'<table border="1" style="([^"]*)" cellpadding="\d+"',
            lambda s: (
                f'<table border="1" style="{s.group(1)} border-color:{k.border.name()}; '
                'border-style:solid;" cellpadding="5"'
            ),
            m.group(0),
            count=1,
        )
        first = re.search(r"<tr>.*?</tr>", t, re.S)
        if first:
            t = (
                t[: first.start()]
                + first.group(0).replace("<td", f'<td bgcolor="{code}"')
                + t[first.end() :]
            )
        return t

    text = re.sub(r'<table border="1".*?</table>', table, text, flags=re.S)

    def block(m: re.Match[str]) -> str:
        lines = re.sub(
            r'<pre style="[^"]*"',
            '<pre style=" margin-top:0px; margin-bottom:0px; white-space:pre-wrap;"',
            m.group(0).strip(),
        )
        return (
            f'<table width="100%" cellspacing="0" cellpadding="7" bgcolor="{code}" '
            f'style=" margin-top:4px; margin-bottom:6px;"><tr><td>{lines}</td></tr></table>'
        )

    text = re.sub(r"(?:<pre[^>]*>.*?</pre>\s*)+", block, text, flags=re.S)
    return text.replace(
        f"font-family:'{k.mono}';",
        f"font-family:'{k.mono}'; font-size:{k.code_size:.1f}pt; background-color:{code};",
    )


ZWSP = "\u200b"  # where a too-long word may be broken
_BREAK_AT = 24  # characters between a long run's breaks
_BREAK_MIN = 28  # runs shorter than this are left whole


def _runs(text: str) -> str:
    """A word too long for the panel (a URL, a path), given soft breaks every
    ``_BREAK_AT`` characters: a zero-width space Qt wraps a line at (a 300-character word
    is otherwise one line, its tail clipped, however narrow the label). Copying takes the
    breaks out again."""

    def soft(m: re.Match[str]) -> str:
        run = m.group(0)
        return ZWSP.join(run[i : i + _BREAK_AT] for i in range(0, len(run), _BREAK_AT))

    return re.sub(rf"\S{{{_BREAK_MIN},}}", soft, text)


def _soften(text: str, rich: bool) -> str:
    """``_runs`` over the text as written, or -- HTML -- between its tags, its entities
    whole (a break inside ``&amp;`` would print the entity as words)."""
    if not rich:
        return _runs(text)
    out = []
    for part in re.split(r"(<[^>]+>)", text):
        if part.startswith("<"):
            out.append(part)
            continue
        for piece in re.split(r"(&[#a-zA-Z0-9]+;)", part):
            out.append(piece if piece.startswith("&") else _runs(piece))
    return "".join(out)


def _tone(w: QWidget, tone: str, prop: str = "tone") -> None:
    w.setProperty(prop, tone)
    w.style().unpolish(w)
    w.style().polish(w)


def _label(
    text: str,
    k: Look,
    *,
    rich: bool,
    keep: bool,
    tone: str | None = None,
    view: ChatView | None = None,
) -> QLabel:
    """A wrapped, selectable label, as wide as it is given: a word too long for the panel
    is soft-broken (``_soften``), never widening the conversation (a table goes in
    ``_Wide``). With a *view*, its right click is the conversation's menu."""
    lbl = QLabel()
    # The analyst's and the model's own words (*keep*): left as written in any language
    # the window is in (.uilang). A note is the window's own.
    lbl.setProperty("normetKeepText", keep)
    lbl.setTextFormat(Qt.TextFormat.RichText if rich else Qt.TextFormat.PlainText)
    lbl.setWordWrap(True)
    policy = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    policy.setHeightForWidth(True)
    lbl.setSizePolicy(policy)
    lbl.setTextInteractionFlags(
        Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.LinksAccessibleByMouse
    )
    lbl.setOpenExternalLinks(True)
    if tone:
        lbl.setProperty("tone", tone)
    lbl.setFont(k.font())
    lbl.setText(_soften(text, rich))
    if view is not None:
        view._hook(lbl)
    return lbl


def _fill(lay: Any, widgets: list[QWidget]) -> None:
    """A layout's widgets replaced: the old ones out of any find at once (hidden), the
    deletion itself waiting a turn."""
    while lay.count():
        w = lay.takeAt(0).widget()
        if w is not None:
            w.hide()
            w.deleteLater()
    for w in widgets:
        lay.addWidget(w)


class _Wide(QScrollArea):
    """A table: as wide as its cells, scrolled sideways on its own where the panel is
    narrower, while the text around it wraps."""

    def __init__(self, html_text: str, k: Look, view: ChatView | None = None) -> None:
        super().__init__()
        self.setObjectName("wide")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.label = _label(html_text, k, rich=True, keep=True, view=view)
        self.label.setWordWrap(False)
        self.label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.label.adjustSize()
        self.setWidget(self.label)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self._fit()

    def set_html(self, html_text: str, k: Look) -> None:
        self.label.setFont(k.font())
        self.label.setText(_soften(html_text, True))
        self.label.adjustSize()
        self._fit()

    def _fit(self) -> None:
        over = self.label.width() > max(1, self.width())
        bar = self.horizontalScrollBar().sizeHint().height() if over else 0
        self.setFixedHeight(self.label.height() + bar)

    def resizeEvent(self, e: Any) -> None:  # noqa: N802
        super().resizeEvent(e)
        self._fit()


def _parts(md: str, k: Look) -> list[tuple[bool, str]]:
    """Markdown as the panel shows it, cut where a table stands: (is a table, its HTML)."""
    out: list[tuple[bool, str]] = []
    for part in re.split(r'(<table border="1".*?</table>)', markdown_html(md, k), flags=re.S):
        if part.strip():
            out.append((part.startswith('<table border="1"'), part))
    return out


def _blocks(md: str, k: Look, view: ChatView | None = None) -> list[QWidget]:
    """Markdown as the panel shows it: its text in labels, each table on its own."""
    return [
        _Wide(part, k, view) if wide else _label(part, k, rich=True, keep=True, view=view)
        for wide, part in _parts(md, k)
    ]


class _Reply(QWidget):
    """The model's words, as their Markdown reads; written again as they stream in."""

    def __init__(self, md: str, k: Look, view: ChatView | None = None) -> None:
        super().__init__()
        self.md, self.view = md, view
        self._kinds: list[bool] = []
        self._widgets: list[QWidget] = []
        self.col = QVBoxLayout(self)
        self.col.setContentsMargins(2, 0, 2, 0)
        self.col.setSpacing(4)
        self.restyle(k)

    @property
    def label(self) -> QLabel | None:
        """The first of its texts (a table's too, where it is first)."""
        for w in self._widgets:
            return w.label if isinstance(w, _Wide) else w  # type: ignore[return-value]
        return None

    def _set(self, k: Look) -> None:
        parts = _parts(self.md, k)
        kinds = [wide for wide, _ in parts]
        if kinds == self._kinds:  # the same blocks: their text again, nothing rebuilt
            for w, (_, part) in zip(self._widgets, parts, strict=True):
                if isinstance(w, _Wide):
                    w.set_html(part, k)
                else:
                    w.setFont(k.font())  # type: ignore[attr-defined]
                    w.setText(_soften(part, True))  # type: ignore[attr-defined]
            return
        self._kinds = kinds
        self._widgets = [
            _Wide(part, k, self.view)
            if wide
            else _label(part, k, rich=True, keep=True, view=self.view)
            for wide, part in parts
        ]
        _fill(self.col, self._widgets)

    def set_md(self, md: str, k: Look) -> None:
        self.md = md
        self._set(k)

    def restyle(self, k: Look) -> None:
        self._set(k)

    def bare(self) -> str:
        return self.md

    def plain(self) -> str:
        return "Assistant: " + self.md


class _You(QFrame):
    """The analyst's message: a bubble on the right, as wide as its words."""

    def __init__(self, text: str, k: Look, room: int) -> None:
        super().__init__()
        self.setObjectName("you")
        self.text = text
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        self.label = _label(text, k, rich=False, keep=True)
        lay.addWidget(self.label)
        self.fit(k, room)

    def fit(self, k: Look, room: int) -> None:
        self.label.setFont(k.font())
        most = max(200, int(room * 0.85))
        widest = max(
            self.label.fontMetrics().horizontalAdvance(line)
            for line in self.text.splitlines() or [""]
        )
        self.setFixedWidth(min(most, widest + 24))

    def bare(self) -> str:
        return self.text

    def plain(self) -> str:
        return "› " + self.text.replace("\n", "\n  ")


class _CallLine(QWidget):
    """A call on one line: its name in the text's colour, its arguments quieter, cut
    with an ellipsis where the panel ends (the whole call is the row's tooltip)."""

    def __init__(self, call: str, k: Look) -> None:
        super().__init__()
        cut = call.find("(")
        self.name, self.rest = (call, "") if cut < 0 else (call[:cut], call[cut:])
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.restyle(k)

    def restyle(self, k: Look) -> None:
        self.k = k
        self.f = k.font(k.code_size, mono=True)
        self.setFixedHeight(QFontMetrics(self.f).height() + 2)
        self.update()

    def paintEvent(self, e: Any) -> None:  # noqa: N802
        p = QPainter(self)
        p.setFont(self.f)
        fm = p.fontMetrics()
        y = (self.height() + fm.ascent() - fm.descent()) // 2
        name = fm.elidedText(self.name, Qt.TextElideMode.ElideRight, self.width())
        p.setPen(self.k.text)
        p.drawText(0, y, name)
        x = fm.horizontalAdvance(name)
        if self.rest and x < self.width():
            p.setPen(self.k.muted)
            p.drawText(
                x, y, fm.elidedText(self.rest, Qt.TextElideMode.ElideRight, self.width() - x)
            )
        p.end()


class _Tool(QFrame):
    """One tool call: a dot -- grey while it runs, green when it answered, red when it was
    refused -- and the call on one line; a refusal under it; what it returned shown when
    the line is clicked."""

    def __init__(self, call: str, k: Look, view: ChatView | None = None) -> None:
        super().__init__()
        self.setObjectName("tool")
        self.call, self.k, self.result, self.error, self.opened = call, k, "", False, False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(2)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(7)
        self.dot = QLabel("●")
        self.dot.setProperty("tone", "faint")
        self.line = _CallLine(call, k)
        self.chevron = QLabel("")
        self.chevron.setProperty("tone", "faint")
        row.addWidget(self.dot)
        row.addWidget(self.line, 1)
        row.addWidget(self.chevron)
        lay.addLayout(row)
        self.detail = _label("", k, rich=False, keep=True, tone="muted", view=view)
        self.detail.setContentsMargins(17, 0, 0, 2)
        self.detail.hide()
        lay.addWidget(self.detail)
        self.setToolTip(call)
        self.restyle(k)

    def restyle(self, k: Look) -> None:
        self.k = k
        self.dot.setFont(k.font(k.small - 1))
        self.line.restyle(k)
        self.chevron.setFont(k.font(k.small))
        self.detail.setFont(k.font(k.small) if self.error else k.font(k.code_size, mono=True))

    def finish(self, text: str, error: bool) -> None:
        self.result, self.error = text, error
        _tone(self.dot, "error" if error else "ok")
        if error:
            _tone(self.detail, "error")
            self.detail.setFont(self.k.font(self.k.small))
            self.detail.setText(_runs(text[:1200]))
            self.detail.show()
        elif text.strip():
            self.chevron.setText("›")
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            _tone(self, "true", "clickable")

    def mousePressEvent(self, e: Any) -> None:  # noqa: N802
        if self.error or not self.result.strip() or e.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(e)
        self.toggle()

    def keyPressEvent(self, e: Any) -> None:  # noqa: N802
        # The keyboard reaches a finished tool the way a click does.
        if (
            not self.error
            and self.result.strip()
            and e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space)
        ):
            self.toggle()
            return
        super().keyPressEvent(e)

    def plain(self) -> str:
        return "● " + self.call + ("\n  ⎿ " + self.result.strip()[:1200] if self.error else "")

    def toggle(self) -> None:
        self.opened = not self.opened
        self.detail.setText(_runs(self.result[:4000]) + ("…" if len(self.result) > 4000 else ""))
        self.detail.setVisible(self.opened)
        self.chevron.setText("⌄" if self.opened else "›")
        _tone(self, "true" if self.opened else "false", "open")


class _Card(QFrame):
    """A proposal: what kind it is, its title, the model's reasons and the button that
    carries it out (``mark_done`` once it has been)."""

    #: The small line above a card's title, by its kind (the window's own words).
    TAGS = {
        "setup": "Set-up",
        "script": "Steps in the window",
        "run": "Run in the window",
    }

    def __init__(
        self,
        title: str,
        body: str,
        apply: Callable[[], str],
        applied: str,
        action: str,
        k: Look,
        view: ChatView | None = None,
        kind: str = "",
    ) -> None:
        super().__init__()
        self.setObjectName("proposal")
        self.title, self.kind, self.done = title, kind, False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        # The tag and the title are the window's own words (translated); the body is the
        # model's.
        self.tag = _label(self.TAGS.get(kind, "Proposal"), k, rich=False, keep=False, tone="muted")
        self.tag.setVisible(bool(kind))
        self.head = _label(html.escape(title), k, rich=True, keep=False, view=view)
        self.head.setFont(k.font(bold=True))
        lay.addWidget(self.tag)
        lay.addWidget(self.head)
        self.body = _Reply(body, k, view)
        lay.addWidget(self.body)
        self.button = btn = QPushButton(action)
        btn.setObjectName("apply")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)

        def clicked() -> None:
            try:
                said = apply()
            except Exception as exc:  # noqa: BLE001 -- shown on the card
                said = f"Could not apply: {exc}"
            self.mark_done(said)

        btn.clicked.connect(clicked)
        if applied:
            self.mark_done(applied)
        lay.addWidget(btn, 0, Qt.AlignmentFlag.AlignLeft)
        self.restyle(k)

    def mark_done(self, text: str = "Applied") -> None:
        """The card applied: its button says so (*text*) and is spent."""
        self.done = True
        self.button.setText(text)
        self.button.setEnabled(False)

    def restyle(self, k: Look) -> None:
        self.tag.setFont(k.font(k.small))
        self.head.setFont(k.font(bold=True))
        self.button.setFont(k.font(k.small))
        self.body.restyle(k)

    def bare(self) -> str:
        return self.body.md

    def plain(self) -> str:
        return f"[{self.title}]\n{self.body.md}\n[{self.button.text()}]"


class _Note(QFrame):
    """A line of the panel's own: muted, or an error on a tinted ground."""

    def __init__(
        self,
        text: str,
        k: Look,
        view: ChatView | None = None,
        *,
        error: bool = False,
        rich: bool = False,
    ) -> None:
        super().__init__()
        self.text, self.rich = text, rich
        if error:
            self.setObjectName("trouble")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(*((10, 6, 10, 6) if error else (2, 0, 2, 0)))
        self.label = _label(
            text, k, rich=rich, keep=False, tone="error" if error else "muted", view=view
        )
        lay.addWidget(self.label)
        self.restyle(k)

    def restyle(self, k: Look) -> None:
        self.label.setFont(k.font(k.small))

    def plain(self) -> str:
        return _label_text(self.label).replace(ZWSP, "") if self.rich else self.text


class Working(QWidget):
    """The line that turns while the model works, above the box the analyst writes in:
    what it is doing and for how long. Laid out by the panel (``ChatView.working``)."""

    FRAMES = "·✢✳✶✻✽✻✶✳✢"

    def __init__(self, k: Look) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(8)
        self.spin, self.label = QLabel(self.FRAMES[0]), QLabel("")
        self.label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        lay.addWidget(self.spin)
        lay.addWidget(self.label, 1)
        self.what, self.t0, self.i = "", 0.0, 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.restyle(k)
        self.hide()

    def restyle(self, k: Look) -> None:
        self.spin.setFont(k.font(k.size + 1))
        self.spin.setFixedWidth(QFontMetrics(self.spin.font()).height())
        self.label.setFont(k.font(k.small))
        c = QColor.name
        self.spin.setStyleSheet(f"color: {c(k.accent)};")
        self.label.setStyleSheet(f"color: {c(k.muted)};")

    def say(self, what: str) -> None:
        """Show what is being done; the frames turn and the seconds count from the first."""
        if not self.timer.isActive():
            self.t0, self.i = time.monotonic(), 0
            self.timer.start(90)
        self.what = what
        self._tick(advance=False)
        self.show()

    def stop(self) -> None:
        self.timer.stop()
        self.hide()

    def _tick(self, advance: bool = True) -> None:
        self.i += advance
        self.spin.setText(self.FRAMES[self.i % len(self.FRAMES)])
        s = int(time.monotonic() - self.t0)
        self.label.setText(self.what + (f"  ·  {s} s" if s else ""))


def _label_text(label: QLabel) -> str:
    """A label's text as it reads: its rich text's characters, without the markup (its
    soft breaks still in)."""
    if label.textFormat() != Qt.TextFormat.RichText:
        return label.text()
    return QTextDocumentFragment.fromHtml(label.text()).toPlainText()


def _put_away(w: QWidget | None, host: QWidget) -> bool:
    """Whether *w*, or what holds it below *host*, was hidden (a call's result folded
    away) -- not merely not shown yet: a label just added is shown a moment later."""
    while w is not None and w is not host:
        if w.testAttribute(Qt.WidgetAttribute.WA_WState_ExplicitShowHide) and w.isHidden():
            return True
        w = w.parentWidget()
    return False


class _FindBar(QFrame):
    """Cmd/Ctrl+F: the text to find in the conversation, the hits stepped through."""

    def __init__(self, view: ChatView) -> None:
        super().__init__(view)
        self.setObjectName("find")
        self.view = view
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 2, 4, 2)
        lay.setSpacing(2)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Find in the conversation")
        self.edit.textChanged.connect(lambda _t: view.find_again(first=True))
        self.edit.installEventFilter(self)
        self.count = QLabel("")
        buttons = []
        for text, tip, step in (("↑", "Previous (Shift+Enter)", -1), ("↓", "Next (Enter)", 1)):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _c=False, s=step: view.find_step(s))
            buttons.append(b)
        close = QToolButton()
        close.setText("×")
        close.setToolTip("Close (Esc)")
        close.clicked.connect(view.close_find)
        lay.addWidget(self.edit, 1)
        lay.addWidget(self.count)
        for b in (*buttons, close):
            lay.addWidget(b)
        self.hide()

    def eventFilter(self, obj: Any, e: QEvent) -> bool:  # noqa: N802
        if obj is self.edit and e.type() == QEvent.Type.KeyPress:
            key = e.key()  # type: ignore[attr-defined]
            if key == Qt.Key.Key_Escape:
                self.view.close_find()
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                shift = e.modifiers() & Qt.KeyboardModifier.ShiftModifier  # type: ignore[attr-defined]
                self.view.find_step(-1 if shift else 1)
                return True
        return super().eventFilter(obj, e)


class ChatView(QScrollArea):
    """The conversation: the analyst's messages on the right, the replies as their
    Markdown reads, each tool call one quiet line, a proposal a card with Apply."""

    #: The Look, after the palette or the text size changed: the box written in follows.
    restyled = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("chatview")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QWidget()
        host.setObjectName("chat")
        self._lay = QVBoxLayout(host)
        self._lay.setContentsMargins(8, 8, 8, 8)
        self._lay.setSpacing(8)
        self._lay.addStretch(1)
        self.setWidget(host)
        self._reply: _Reply | None = None
        try:
            self.zoom = float(QSettings("apai-sys", "normet-gui").value(ZOOM_KEY, 0.0))
        except (TypeError, ValueError):
            self.zoom = 0.0
        self.look = Look(self.zoom)
        self.setStyleSheet(sheet(self.look))
        #: The line that turns while the model works: the panel lays it out (and calls
        #: ``working.say(what)`` / ``working.stop()``); restyled and stopped with the view.
        self.working = Working(self.look)
        #: Whether the view keeps the end as the conversation grows: it does until the
        #: analyst scrolls up to read, and again once they are back at the end.
        self._follow = True
        bar = self.verticalScrollBar()
        bar.rangeChanged.connect(self._on_range)
        bar.valueChanged.connect(self._on_value)
        self.to_end = QToolButton(self)
        self.to_end.setObjectName("to_end")
        self.to_end.setText("↓")
        self.to_end.setToolTip("Back to the end of the conversation")
        self.to_end.setFixedSize(28, 28)
        self.to_end.setCursor(Qt.CursorShape.PointingHandCursor)
        self.to_end.clicked.connect(self.to_bottom)
        self.to_end.hide()
        self.find_bar = _FindBar(self)
        self._hits: list[tuple[QLabel, int, int]] = []
        self._hit = -1
        self._shown: QLabel | None = None
        find = QShortcut(QKeySequence(QKeySequence.StandardKey.Find), self)
        find.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        find.activated.connect(self.open_find)
        for keys, step in (
            (QKeySequence.StandardKey.ZoomIn, 1),
            ("Ctrl+=", 1),
            (QKeySequence.StandardKey.ZoomOut, -1),
            ("Ctrl+0", 0),
        ):
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda s=step: self.zoom_by(s))

    # -- what is in it -----------------------------------------------------------------
    def _add(self, w: QWidget, align: Qt.AlignmentFlag | None = None) -> None:
        if align is None:
            self._lay.insertWidget(self._lay.count() - 1, w)
        else:
            self._lay.insertWidget(self._lay.count() - 1, w, 0, align)
        for lbl in [w] if isinstance(w, QLabel) else w.findChildren(QLabel):
            self._hook(lbl)

    def _hook(self, lbl: QLabel) -> None:
        """A label's menu is the conversation's (the copies, find), not a label's own."""
        if lbl.property("hooked"):
            return
        lbl.setProperty("hooked", True)
        lbl.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        lbl.customContextMenuRequested.connect(
            lambda pos, lbl=lbl: self.menu_for(lbl).exec(lbl.mapToGlobal(pos))
        )

    def items(self) -> list[QWidget]:
        out = []
        for i in range(self._lay.count()):
            w = self._lay.itemAt(i).widget()
            if w is not None:
                out.append(w)
        return out

    def clear(self) -> None:
        self._deselect()
        self._hits, self._hit = [], -1
        while self._lay.count() > 1:
            w = self._lay.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        self._reply = None
        self._follow = True
        self.working.stop()

    def you(self, text: str) -> None:
        self._follow = True  # the analyst's own message: the view goes to the end
        self._add(_You(text, self.look, self.viewport().width()), Qt.AlignmentFlag.AlignRight)
        self._reply = None

    @property
    def writing(self) -> bool:
        """Whether a reply is being written (words added go to it)."""
        return self._reply is not None

    def reply_live(self, md: str) -> None:
        """The reply being written, as it stands so far (a new one if none is)."""
        if self._reply is None:
            self.reply(md)
        else:
            self._reply.set_md(md, self.look)

    def reply(self, text: str, more: bool = False) -> None:
        """The model's words: a new reply, or (*more*) added to the one being written."""
        if not text.strip():
            return
        if more and self._reply is not None:
            self._reply.set_md(self._reply.md + text, self.look)
            return
        self._reply = _Reply(text, self.look, self)
        self._add(self._reply)

    def tool(self, name: str, arguments: dict[str, Any]) -> _Tool:
        args = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
        item = _Tool(f"{name}({args[:120]}{'…' if len(args) > 120 else ''})", self.look, self)
        self._add(item)
        self._reply = None
        return item

    @staticmethod
    def tool_done(item: _Tool, output: str, error: bool) -> None:
        item.finish(output, error)

    def note(self, text: str, error: bool = False, rich: bool = False) -> None:
        """A line of the panel's own: muted, or (*error*) on a tinted ground; *rich*:
        *text* is HTML (a link)."""
        self._add(_Note(text, self.look, self, error=error, rich=rich))
        self._reply = None

    def card(
        self,
        title: str,
        body: str,
        apply: Callable[[], str],
        applied: str = "",
        action: str = "Apply",
        kind: str = "",
    ) -> QPushButton:
        """A proposal with its button (``action``, what clicking it does); ``applied``,
        what applying it said where it was applied already (the Assistant's full
        automatic), on the button instead. ``kind`` (a key of ``_Card.TAGS``) names the
        sort of proposal above its title."""
        card = _Card(title, body, apply, applied, action, self.look, self, kind)
        self._add(card)
        self._reply = None
        return card.button

    def card_done(self, button: QPushButton, text: str = "Applied") -> None:
        """The card whose button *button* is (``card``'s return), marked applied."""
        entry = self._entry(button)
        if isinstance(entry, _Card):
            entry.mark_done(text)

    # -- as text ---------------------------------------------------------------------
    def _entry(self, w: QWidget | None) -> QWidget | None:
        """The entry of the conversation *w* is in (a message, a call, a card, a note)."""
        host = self.widget()
        while w is not None and w.parentWidget() is not host:
            w = w.parentWidget()
        return w

    @staticmethod
    def _plain(w: QWidget) -> str:
        if hasattr(w, "plain"):
            return str(w.plain())  # type: ignore[attr-defined]
        return w.text() if isinstance(w, QLabel) else ""

    def toPlainText(self) -> str:  # noqa: N802 -- Qt's name for a widget's text
        """The conversation as text, as Copy conversation copies it."""
        return "\n\n".join(s for s in (self._plain(w) for w in self.items()) if s.strip())

    def menu_for(self, label: QLabel | None = None) -> QMenu:
        """The menu for a label of the conversation (or none of them): copy what is
        selected, the message as it was written, the message as the transcript keeps it,
        the conversation entire; then find, and the text's size."""
        menu = QMenu(self)
        copy = QApplication.clipboard().setText
        if label is not None and label.hasSelectedText():
            selected = (
                label.selectedText()
                .replace("\u2029", "\n")
                .replace("\u2028", "\n")
                .replace(ZWSP, "")
            )
            menu.addAction("Copy", lambda: copy(selected))
        entry = self._entry(label)
        if entry is not None:
            if hasattr(entry, "bare"):
                menu.addAction("Copy text", lambda: copy(entry.bare()))  # type: ignore[attr-defined]
            menu.addAction("Copy message", lambda: copy(self._plain(entry)))
        whole = menu.addAction("Copy conversation", lambda: copy(self.toPlainText()))
        whole.setEnabled(bool(self.items()))
        menu.addSeparator()
        find = menu.addAction("Find…", self.open_find)
        find.setShortcut(QKeySequence(QKeySequence.StandardKey.Find))
        menu.addSeparator()
        menu.addAction("Larger text", lambda: self.zoom_by(1))
        menu.addAction("Smaller text", lambda: self.zoom_by(-1))
        menu.addAction("Actual size", lambda: self.zoom_by(0))
        return menu

    def contextMenuEvent(self, e: Any) -> None:  # noqa: N802
        self.menu_for(None).exec(e.globalPos())

    # -- the end -----------------------------------------------------------------------
    def _on_range(self, _lo: int, hi: int) -> None:
        if self._follow:
            self.verticalScrollBar().setValue(hi)
        self._place_floaters()

    def _on_value(self, value: int) -> None:
        bar = self.verticalScrollBar()
        self._follow = value >= bar.maximum() - 4
        self._place_floaters()

    def to_bottom(self) -> None:
        self._follow = True
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _place_floaters(self) -> None:
        """The button back to the end at the view's lower right, shown off the end; the
        find bar across its top."""
        vp = self.viewport().geometry()
        self.to_end.move(
            vp.right() - self.to_end.width() - 10, vp.bottom() - self.to_end.height() - 8
        )
        self.to_end.setVisible(not self._follow)
        self.to_end.raise_()
        width = min(vp.width() - 16, 360)
        self.find_bar.setGeometry(vp.right() - width - 8, vp.top() + 6, width, 30)
        self.find_bar.raise_()

    def resizeEvent(self, e: Any) -> None:  # noqa: N802
        super().resizeEvent(e)
        self._place_floaters()

    # -- find -------------------------------------------------------------------------
    def open_find(self) -> None:
        self._place_floaters()
        self.find_bar.show()
        self.find_bar.edit.setFocus()
        self.find_bar.edit.selectAll()
        self.find_again(first=True)

    def close_find(self) -> None:
        self.find_bar.hide()
        self._deselect()
        self._hits, self._hit = [], -1

    def _find_hits(self, needle: str) -> list[tuple[QLabel, int, int]]:
        """Where *needle* is (any case) in the labels shown: (label, start, length)."""
        hits: list[tuple[QLabel, int, int]] = []
        if not needle:
            return hits
        low = needle.lower()
        host = self.widget()
        for entry in self.items():
            labels = [entry] if isinstance(entry, QLabel) else entry.findChildren(QLabel)
            for lbl in labels:
                if _put_away(lbl, host):
                    continue
                # The shown text carries soft breaks the needle does not: a match on the
                # clean text is mapped back through where the breaks stand (it may span one).
                clean, breaks = [], []
                for ch in _label_text(lbl):
                    if ch == ZWSP:
                        breaks.append(len(clean))
                    else:
                        clean.append(ch)
                text = "".join(clean).lower()
                at = text.find(low)
                while at >= 0:
                    end = at + len(needle)
                    lo = at + bisect_right(breaks, at)
                    hi = end + bisect_left(breaks, end)
                    hits.append((lbl, lo, hi - lo))
                    at = text.find(low, end)
        return hits

    def find_again(self, *, first: bool = False) -> None:
        """The hits for the find bar's text, made again; the first one shown."""
        self._deselect()
        self._hits = self._find_hits(self.find_bar.edit.text())
        self._hit = 0 if self._hits else -1
        self._show_hit()

    def find_step(self, step: int) -> None:
        """The next (1) or the previous (-1) hit shown, round from the end to the start."""
        if not self._hits:
            return
        self._hit = (self._hit + step) % len(self._hits)
        self._show_hit()

    def _deselect(self) -> None:
        if self._shown is not None:
            try:
                self._shown.setSelection(0, 0)
            except RuntimeError:  # deleted with a cleared conversation
                pass
            self._shown = None

    def _show_hit(self) -> None:
        needle = self.find_bar.edit.text()
        if self._hit < 0:
            self.find_bar.count.setText("No matches" if needle else "")
            return
        self._deselect()
        lbl, at, n = self._hits[self._hit]
        lbl.setSelection(at, n)
        self._shown = lbl
        self._follow = False
        self.ensureWidgetVisible(lbl, 0, 40)
        self.find_bar.count.setText(f"{self._hit + 1} of {len(self._hits)}")

    # -- how it looks ------------------------------------------------------------------
    def zoom_by(self, step: int) -> None:
        """The text a point larger (1), smaller (-1) or the system's size (0); kept."""
        z = 0.0 if step == 0 else max(-4.0, min(8.0, self.zoom + step))
        if z == self.zoom:
            return
        self.zoom = z
        QSettings("apai-sys", "normet-gui").setValue(ZOOM_KEY, z)
        self.restyle()

    def restyle(self) -> None:
        """Colours and sizes made again: the palette changed (light, dark) or the zoom."""
        self.look = k = Look(self.zoom)
        self.setStyleSheet(sheet(k))
        room = self.viewport().width()
        for w in self.items():
            if isinstance(w, _You):
                w.fit(k, room)
            elif isinstance(w, (_Reply, _Tool, _Card, _Note)):
                w.restyle(k)
            elif isinstance(w, QLabel):
                w.setFont(k.font(k.small))
        self.working.restyle(k)
        self.restyled.emit(k)

    def follow_palette(self) -> None:
        """Made again in the application's palette if it has changed (light, dark). The
        view's own style sheet sets its palette, which then no longer follows the
        application's, so the panel holding it calls this too when the system's
        appearance changes. A style sheet set here posts a palette event as well: the
        same colours are no change, or it would never end."""
        if Look(self.zoom).key != self.look.key:
            self.restyle()

    def changeEvent(self, e: Any) -> None:  # noqa: N802
        super().changeEvent(e)
        if e.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self.follow_palette()

    def wheelEvent(self, e: Any) -> None:  # noqa: N802
        if e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom_by(1 if e.angleDelta().y() > 0 else -1)
            return
        super().wheelEvent(e)
