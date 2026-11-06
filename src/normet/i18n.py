# src/normet/i18n.py
"""normet's reports in a language other than English.

A report is written in English and translated whole -- the page's text and its figures'
-- through a catalogue of English templates and their translations (``normet.gui.ui_zh``
for Chinese). A template is the English text with ``{n0}``, ``{n1}``, ... where a number
stands (12, 0.35, -1.2e-3) and ``{s0}``, ``{s1}``, ... where a name, a list or other free
text does (a column's name, a file's); its translation uses the same marks in its own
order. What a ``{s}`` holds is written with the catalogue's phrases (``INNER``) where it
has them and as it is otherwise; a word the catalogue names only as a ``{s}``'s whole
(``SLOTS``) is written in the language there and nowhere else. A date written with its
month's name, as a figure's date axis writes one (2023-Jul-05), is written in the
language's order (``DATE_PARTS``). A text no template matches stays English.

The window's catalogue engine (``normet.gui._i18n``) reads this module; the report's
(``normet.report``) builds its catalogue from the window's texts (``ui_zh``) laid over
its own few report words.
"""

from __future__ import annotations

import html as _html
import re
from collections.abc import Sequence
from functools import cache

#: The languages a report is written in, by code.
LANGUAGES: dict[str, str] = {"en": "English", "zh": "中文"}

_NUM = r"[-+−]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?"
_MARK = re.compile(r"\{([ns])(\d+)\}")
_SPACE = re.compile(r"\s+")
#: The page's lang attribute for each of LANGUAGES.
HTML_LANG = {"en": "en", "zh": "zh-CN"}
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
#: A date written with its month's name, as a figure's date axis writes one: 2023-Jul,
#: Jul-05, 2023-Jul-05 14:00.
_DATE = re.compile(
    r"(?:(\d{4})-)?(" + "|".join(_MONTHS) + r")(?:-(\d{1,2}))?( \d{1,2}:\d{2}(?::\d{2}))?"
)
#: The same, day first, as the window's date axes write one: 8 Jun, 8 Jun 2023, Jun 2023,
#: 14:00 8 Jun.
_DAY_FIRST = re.compile(
    r"(?:(\d{1,2}:\d{2}) )?(?:(\d{1,2}) )?(" + "|".join(_MONTHS) + r")(?: (\d{4}))?"
)


def check_language(lang: str) -> str:
    if lang not in LANGUAGES:
        raise ValueError(f"lang must be one of {', '.join(LANGUAGES)}, not {lang!r}")
    return lang


_WORDLIKE = r"(?:\w|\{[ns]\d+\})"

_COUNTED = re.compile(r"( )([a-z]{3,})(?![\w{])")


def _either_number(word: str) -> str:
    """A counted word's pattern, singular or plural: pull(s), entr(y|ies), match(es)."""
    if word.endswith("ies"):
        stem = word[:-3]
        return re.escape(stem) + "(?:y|ies)"
    if word.endswith("y") and word[-2] not in "aeiou":
        return re.escape(word[:-1]) + "(?:y|ies)"
    if word.endswith(("ches", "shes", "sses", "xes")):
        return re.escape(word[:-2]) + "(?:es)?"
    if word.endswith(("ch", "sh", "ss", "x")):
        return re.escape(word) + "(?:es)?"
    if word.endswith("s"):
        return re.escape(word[:-1]) + "s?"
    return re.escape(word) + "s?"


def _compile(template: str, whole: bool = True, sentences: bool = True):
    """A template's pattern: its text as it is, a number for each {n}, anything for each
    {s}. A phrase found inside a text (`whole=False`) does not start or end inside a word
    -- where its own first or last character is a word's, or a mark's -- and a {s} in it
    starts with a character that is not a space."""
    parts, pos = [], 0
    for m in _MARK.finditer(template):
        parts.append(re.escape(template[pos : m.start()]))
        name = m.group(1) + m.group(2)
        free = r".+?" if whole else r"\S.*?"
        if not sentences:
            # A {s} within its sentence and its part: the window strings sentences, and
            # the facts of a line (" · ", " | "), together, and a template's {s} would
            # otherwise take the ones before or after its own.
            free = r"(?:(?!\. [A-Z(]| · | \| ).)+?" if whole else r"\S(?:(?!\. [A-Z(]| · | \| ).)*?"
        parts.append(f"(?P<{name}>{_NUM})" if m.group(1) == "n" else f"(?P<{name}>{free})")
        pos = m.end()
        word = _COUNTED.match(template, pos) if m.group(1) == "n" and not sentences else None
        if word:
            # The window writes "1 pull" and "2 pulls" from one f-string: the word after a
            # number matches either, as the translation does not change with it.
            parts.append(re.escape(word.group(1)) + _either_number(word.group(2)))
            pos = word.end()
    parts.append(re.escape(template[pos:]))
    body = "".join(parts)
    if not whole:
        if re.match(_WORDLIKE, template):
            body = r"(?<!\w)" + body
        if re.search(_WORDLIKE + r"$", template):
            body += r"(?!\w)"
    return re.compile(body, re.S)


_Pat = re.Pattern[str]
Catalogue = tuple[
    dict[str, str],
    Sequence[tuple[_Pat, str]],
    Sequence[tuple[_Pat, _Pat, str]],
    tuple[str, ...] | None,
    dict[str, str],
]


def build_catalogue(
    texts: dict[str, str],
    inner: dict[str, str] | None = None,
    date_parts: tuple[str, ...] | None = None,
    sentences: bool = True,
    slots: dict[str, str] | None = None,
) -> Catalogue:
    """A catalogue `translate` reads, from templates and their translations (`texts`),
    the phrases a name holds (`inner`), a date's parts and the words translated only as a
    {s}'s whole (`slots`), as `normet.gui.ui_zh` gives them -- for another set of texts
    than a report's: the window's (`normet.gui.uilang`).
    `sentences` false: a template's {s} stays within its own sentence."""
    phrases_in = inner or {}
    exact, patterns = {}, []
    for en, tr in texts.items():
        # A text is matched with its whitespace runs as one space (a figure's line breaks
        # too); its translation keeps its own.
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
        for en, tr in phrases_in.items()
    ]
    return exact, [(p, tr) for _, p, tr in patterns], phrases, date_parts, dict(slots or {})


#: A mode's name and the unit it may carry: "size (µm)".
_SLOT_UNIT = re.compile(r"(.+?)( \([^()]*\))?")


def _slot_word(part: str, slots: dict[str, str]) -> str | None:
    m = _SLOT_UNIT.fullmatch(part)
    if m and m.group(1) in slots:
        return slots[m.group(1)] + (m.group(2) or "")
    return None


def _phrases(text: str, cat: Catalogue) -> str:
    """`text` with each of the catalogue's phrases in it written as the language writes it
    -- and the rest (a column's name, a file's) as it is. A text that is one of the
    catalogue's whole-slot words, or a list of them joined with " × ", is written in the
    language word by word."""
    parts = text.split(" × ")
    words = [_slot_word(p, cat[4]) for p in parts]
    if all(w is not None for w in words):
        return " × ".join(w for w in words if w is not None)
    for pattern, _, tr in cat[2]:
        text = pattern.sub(lambda m, tr=tr: tr.format(**m.groupdict()), text)
    return text


def translate(text: str, lang: str, catalogue: Catalogue | None = None) -> str | None:
    """`text` in `lang`, or None where the catalogue has no template for it. Whitespace
    runs read as one space. `catalogue` (`build_catalogue`) in place of the language's
    report catalogue."""
    if lang == "en":
        return text
    cat = catalogue if catalogue is not None else _catalogue(lang)
    exact, patterns, inner, date, _ = cat
    key = _SPACE.sub(" ", text).strip()
    if key in exact:
        return exact[key]
    for pattern, tr in patterns:
        m = pattern.fullmatch(key)
        if m:
            return tr.format(
                **{k: (_phrases(v, cat) if k[0] == "s" else v) for k, v in m.groupdict().items()}
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
    m = _DAY_FIRST.fullmatch(key)
    if m and date:
        time, day, month, year = m.groups()
        return (
            (date[0].format(int(year)) if year else "")
            + date[1].format(_MONTHS.index(month) + 1)
            + (date[2].format(int(day)) if day else "")
            + (" " + time if time else "")
        )
    return None


@cache
def _catalogue(lang: str) -> Catalogue:
    """The report's own catalogue: the window's texts (normet.gui.ui_zh) and the report's
    few words of its own (normet.report._REPORT_ZH) laid over one another."""
    if lang == "en":
        return build_catalogue({})
    from .gui import ui_zh
    from .report import _REPORT_ZH

    texts = {**ui_zh.TEXTS, **_REPORT_ZH}
    inner = {**ui_zh.INNER}
    return build_catalogue(texts, inner, ui_zh.DATE_PARTS, sentences=False)


@cache
def _back_translation(lang: str) -> dict[str, str]:
    """The catalogue the other way round: a translation back to the English text it is
    of. Where more than one English text is translated the same way the first wins."""
    out: dict[str, str] = {}
    for en, tr in _catalogue(lang)[0].items():
        out.setdefault(tr, en)
    return out


_CJK_CHAR = re.compile(r"[㐀-鿿豈-﫿]")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


def written_language(text: str) -> str | None:
    """The language of LANGUAGES `text` is written in: Chinese where its Chinese
    characters outnumber its English words, English where it has words and no more
    Chinese, None where it has neither (a number)."""
    cjk = len(_CJK_CHAR.findall(text or ""))
    words = len(_LATIN_WORD.findall(text or ""))
    if cjk > words:
        return "zh"
    return "en" if words else None


def localize_names(names: list[str], lang: str) -> list[str]:
    """Names as a report in `lang` writes them: a name the catalogue knows is written in
    `lang` (and back), and a name it does not know -- the analyst's own -- stays as it is
    written."""
    check_language(lang)
    if lang == "en":
        back: dict[str, str] = {}
        for code in LANGUAGES:
            if code != "en":
                back.update(_back_translation(code))
        return [back.get(n, n) for n in names]
    exact = _catalogue(lang)[0]
    return [exact.get(n, n) for n in names]


#: Where a page of rich text is cut: at each tag and comment, which split leaves
#: the runs of text between them whole (html_texts, localize_html).
TAG_SPLIT = re.compile(r"(<!--.*?-->|<[^>]*>)", re.S)
#: Elements whose text is not prose: code (the session's commands) stays as it is written.
NOT_PROSE = ("script", "style", "pre", "code")


def tag_name(piece: str) -> str:
    """A tag piece's element, lower case ("</pre>" -> "pre"); "" where it names none."""
    m = re.match(r"</?\s*([A-Za-z][\w-]*)", piece)
    return m.group(1).lower() if m else ""


def html_texts(page: str) -> list[str]:
    """The texts of a page a reader reads: between its tags, outside <script>, <style>
    and code (<pre>, <code>), entities read as the characters they stand for, whitespace
    as one space."""
    out, skip = [], 0
    for piece in TAG_SPLIT.split(page):
        if piece.startswith("<"):
            if tag_name(piece) in NOT_PROSE:
                skip += -1 if piece.startswith("</") else 1
            continue
        text = _SPACE.sub(" ", _html.unescape(piece)).strip()
        if skip <= 0 and re.search(r"[A-Za-z]{2}", text):
            out.append(text)
    return out


def localize_html(page: str, lang: str, catalogue: Catalogue | None = None) -> str:
    """The page with its texts in `lang`; its markup, scripts, styles, code and figures as
    they are (a figure's text is translated as it is drawn: `localize_figure`). A text
    with no template stays English. `catalogue` as `translate` takes it."""
    if lang == "en":
        return page
    pieces, skip = TAG_SPLIT.split(page), 0
    for i, piece in enumerate(pieces):
        if piece.startswith("<"):
            tag = tag_name(piece)
            if tag in NOT_PROSE:
                skip += -1 if piece.startswith("</") else 1
            elif tag == "html" and not piece.startswith("</"):
                code = HTML_LANG[lang]
                pieces[i] = (
                    re.sub(r"""\blang=(["'])[^"']*\1""", f'lang="{code}"', piece)
                    if re.search(r"\blang=", piece)
                    else piece[:5] + f' lang="{code}"' + piece[5:]
                )
            continue
        if skip > 0 or not piece.strip():
            continue
        tr = translate(_html.unescape(piece), lang, catalogue)
        if tr is not None:
            lead = piece[: len(piece) - len(piece.lstrip())]
            trail = piece[len(piece.rstrip()) :]
            pieces[i] = lead + _html.escape(tr, quote=False) + trail
    return "".join(pieces)


#: Fonts with the language's characters, the first a machine has taken: macOS's, Windows',
#: then Linux's.
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
    """The font a figure's text in `lang` falls back to for the characters its own font
    lacks -- None where the machine has none, and the figure's text then stays English
    rather than show boxes."""
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


def _words(s: str, lang: str) -> str:
    """A tick label in `lang` where it is words (a day, a month, a date with its month's
    name); a number, or mathtext, as it is."""
    if "$" in s or not re.search(r"[A-Za-z]{2}", s):
        return s
    if "\n" in s:  # a date axis's day over its year: each line apart
        return "\n".join(_words(line, lang) for line in s.split("\n"))
    return translate(s, lang) or s


@cache
def translating_formatter():
    """A tick formatter that writes another's labels in a language. A figure's tick labels
    are written by its axes' formatters each time it is drawn -- a label set as words
    (set_xticklabels) too -- so the formatter is what is translated, not the labels."""
    from matplotlib.ticker import Formatter

    class Translating(Formatter):
        def __init__(self, inner, lang):
            self.inner, self.lang = inner, lang

        def __call__(self, x, pos=None):
            return _words(self.inner(x, pos), self.lang)

        def format_ticks(self, values):
            return [_words(s, self.lang) for s in self.inner.format_ticks(values)]

        def get_offset(self):
            return _words(self.inner.get_offset(), self.lang)

        def set_axis(self, axis):
            super().set_axis(axis)
            self.inner.set_axis(axis)

        def set_locs(self, locs):
            self.inner.set_locs(locs)

        def format_data_short(self, value):
            return self.inner.format_data_short(value)

    return Translating


def localize_figure(fig, lang: str) -> None:
    """A figure's texts -- titles, labels, legends, notes, and its tick labels where they
    are words (days, months, a date axis's month names) -- in `lang`, drawn with the
    language's characters from `figure_font` where a text's own font lacks them. Numbers
    and names match no template and stay as they are."""
    if lang == "en":
        return
    font = figure_font(lang)
    if font is None:
        return
    from matplotlib.text import Text

    translating = translating_formatter()

    def fallback(t):
        families = list(t.get_fontfamily())
        if font not in families:
            t.set_fontfamily(families + [font])

    for t in fig.findobj(Text):
        s = t.get_text()
        if not s or not s.strip() or "$" in s:
            continue
        tr = translate(s, lang)
        if tr is not None and tr != s:
            t.set_text(tr)
            fallback(t)
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            # Axes sharing an axis share its formatters: each is wrapped once.
            for get, put in (
                (axis.get_major_formatter, axis.set_major_formatter),
                (axis.get_minor_formatter, axis.set_minor_formatter),
            ):
                if not isinstance(get(), translating):
                    put(translating(get(), lang))
            labels = axis.get_ticklabels(which="both")
            families = list(labels[0].get_fontfamily()) if labels else ["sans-serif"]
            if font not in families:
                axis.set_tick_params(which="both", labelfontfamily=families + [font])
            fallback(axis.offsetText)
