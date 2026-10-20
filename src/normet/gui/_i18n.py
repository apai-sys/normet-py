"""The catalogue engine behind the window's languages (:mod:`.uilang`).

A text is translated through a catalogue of
English templates and their translations (:mod:`.ui_zh` for Chinese). A
template is the English text with ``{n0}``, ``{n1}``, ... where a number stands
(12, 0.35, -1.2e-3) and ``{s0}``, ``{s1}``, ... where a name, a list or other
free text does (a column's name, a file's); its translation uses the same marks
in its own order. What a ``{s}`` holds is written with the catalogue's phrases
(``INNER``) where it has them and as it is otherwise. A date written with its
month's name (2023-Jul-05) is written in the language's order (``DATE_PARTS``).
A text no template matches stays English.
"""

from __future__ import annotations

import re
from functools import cache
from typing import Any

_NUM = r"[-+−]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?"
_MARK = re.compile(r"\{([ns])(\d+)\}")
_SPACE = re.compile(r"\s+")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
#: A date written with its month's name, as a figure's date axis writes one:
#: 2023-Jul, Jul-05, 2023-Jul-05 14:00.
_DATE = re.compile(
    r"(?:(\d{{4}})-)?({})(?:-(\d{{1,2}}))?( \d{{1,2}}:\d{{2}}(?::\d{{2}})?)?".format(
        "|".join(_MONTHS)
    )
)
_WORDLIKE = r"(?:\w|\{[ns]\d+\})"

#: (exact texts, templates the most literal first, phrases, a date's parts)
Catalogue = tuple[
    dict[str, str], list[tuple[Any, str]], list[tuple[Any, Any, str]], "tuple[str, ...] | None"
]


def _compile(template: str, whole: bool = True, sentences: bool = True) -> re.Pattern[str]:
    """A template's pattern: its text as it is, a number for each {n}, anything
    for each {s}. A phrase found inside a text (``whole=False``) does not start
    or end inside a word, and a {s} in it starts with a character that is not a
    space."""
    parts, pos = [], 0
    for m in _MARK.finditer(template):
        parts.append(re.escape(template[pos : m.start()]))
        name = m.group(1) + m.group(2)
        free = r".+?" if whole else r"\S.*?"
        if not sentences:
            # A {s} within its sentence and its part of a line: the window strings
            # sentences together, and facts with " | " or " · " -- a template's {s} would
            # otherwise take the sentences, or the facts, after it ("Backend: {s0}" took
            # "lightgbm | features: 9 | rows: 5,793", and those two stayed English; a
            # line a step wrote took the time the task clock adds after it).
            stop = r"(?!\. [A-Z(]| \| | · )"
            free = rf"(?:{stop}.)+?" if whole else rf"\S(?:{stop}.)*?"
        parts.append(f"(?P<{name}>{_NUM})" if m.group(1) == "n" else f"(?P<{name}>{free})")
        pos = m.end()
    parts.append(re.escape(template[pos:]))
    body = "".join(parts)
    if not whole:
        if re.match(_WORDLIKE, template):
            body = r"(?<!\w)" + body
        if re.search(_WORDLIKE + r"$", template):
            body += r"(?!\w)"
    return re.compile(body, re.S)


def build_catalogue(
    texts: dict[str, str],
    inner: dict[str, str] | None = None,
    date_parts: tuple[str, ...] | None = None,
    sentences: bool = True,
) -> Catalogue:
    """A catalogue :func:`translate` reads, from templates and their
    translations (``texts``), the phrases a name holds (``inner``) and a date's
    parts. ``sentences`` false: a template's {s} stays within its sentence."""
    exact: dict[str, str] = {}
    patterns: list[tuple[int, Any, str]] = []
    for en, tr in texts.items():
        # Matched with whitespace runs as one space; the translation keeps its own.
        en = _SPACE.sub(" ", en).strip()
        if _MARK.search(en):
            patterns.append((len(_MARK.sub("", en)), _compile(en, sentences=sentences), tr))
        else:
            exact[en] = tr
    patterns.sort(key=lambda p: -p[0])
    phrases = [
        (
            _compile(_SPACE.sub(" ", en), whole=False, sentences=sentences),
            _compile(_SPACE.sub(" ", en), sentences=sentences),
            tr,
        )
        for en, tr in (inner or {}).items()
    ]
    return exact, [(p, tr) for _, p, tr in patterns], phrases, date_parts


def _phrases(text: str, cat: Catalogue) -> str:
    """``text`` with each of the catalogue's phrases in it translated, and the
    rest (a column's name, a file's) as it is."""
    for pattern, _, tr in cat[2]:
        text = pattern.sub(lambda m, tr=tr: tr.format(**m.groupdict()), text)
    return text


def translate(text: str, lang: str, catalogue: Catalogue) -> str | None:
    """``text`` in ``lang``, or None where the catalogue has no template for it.
    Whitespace runs read as one space."""
    if lang == "en":
        return text
    exact, patterns, inner, date = catalogue
    key = _SPACE.sub(" ", text).strip()
    if key in exact:
        return exact[key]
    for pattern, tr in patterns:
        m = pattern.fullmatch(key)
        if m:
            return tr.format(
                **{
                    k: (_phrases(v, catalogue) if k[0] == "s" else v)
                    for k, v in m.groupdict().items()
                }
            )
    for _, whole, tr in inner:
        m = whole.fullmatch(key)
        if m:
            return tr.format(**m.groupdict())
    m = _DATE.fullmatch(key)
    if m and date:
        year, month, day, time = m.groups()
        return (
            (date[0].format(int(year)) if year else "")
            + date[1].format(_MONTHS.index(month) + 1)
            + (date[2].format(int(day)) if day else "")
            + (time or "")
        )
    return None


#: Where a page of rich text is cut: at each tag and comment.
TAG_SPLIT = re.compile(r"(<!--.*?-->|<[^>]*>)", re.S)
#: Elements whose text is not prose: code stays as it is written.
NOT_PROSE = ("script", "style", "pre", "code")


def tag_name(piece: str) -> str:
    """A tag piece's element, lower case ("</pre>" -> "pre"); "" where none."""
    m = re.match(r"</?\s*([A-Za-z][\w-]*)", piece)
    return m.group(1).lower() if m else ""


#: Fonts with the language's characters, the first a machine has taken:
#: macOS's, Windows', then Linux's.
_FONTS = {
    "zh": (
        "Hiragino Sans GB",
        "PingFang SC",
        "STHeiti",
        "Heiti SC",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    )
}


@cache
def figure_font(lang: str) -> str | None:
    """The font a figure's text in ``lang`` falls back to for the characters its
    own font lacks -- None where the machine has none (the figure then stays
    English rather than show boxes)."""
    if lang not in _FONTS:
        return None
    from matplotlib import font_manager as fm

    for name in _FONTS[lang]:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            return name
        except ValueError:
            continue
    return None
