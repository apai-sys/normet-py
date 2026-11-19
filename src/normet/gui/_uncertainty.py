"""The window's helper over :func:`normet.decompose_uncertainty`.

The step trains its own ensemble members from the table as loaded, so the
window keeps the result as it came back (an :class:`UncDecomposition`) and
says from it what its own words say: where the spread's median goes, which
source that is, and which number a Methods paragraph quotes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: The window's own words for the tab: source names as the bands are drawn.
SOURCES = (
    ("weather", "met resampling (weather)"),
    ("model", "model form"),
    ("interaction", "their interaction"),
)


def dominant(
    res: Any,
) -> tuple[str, float] | None:
    """The source with the larger median share, weather or model form: its
    name as the tab says it, and the median share of the total it carries.

    None where no share stands (a run with zero total spread, or no record)
    -- the caller keeps its own words then rather than name a winner that is
    not there.
    """
    summary = res.summary().set_index("source")
    weather = float(summary.loc["weather", "pct_median"])
    model = float(summary.loc["model", "pct_median"])
    if not np.isfinite(weather) or not np.isfinite(model):
        return None
    if weather == model:  # a tie is said as a tie
        return None
    name, share = ("weather", weather) if weather > model else ("model", model)
    for key, word in SOURCES:
        if key == name:
            return word, share
    return None  # pragma: no cover


def _pct(x: float) -> str:
    """A share in the banner's words: the number where it stands, the window's
    plain word where the record carries none to read."""
    return f"{100.0 * x:.0f}%" if np.isfinite(x) else "none"


def verdict(res: Any, name: str = "the target") -> tuple[str, str]:
    """The Uncertainty tab's banner: of the band, this share is weather, that
    one is model form -- the attribution a policy evaluation must settle.

    A run whose total spread is zero on every date carries no variance to
    split: the banner says so rather than quote a share of nothing, and a
    level sharing of a real spread is said as a level stand, not a win."""
    summary = res.summary().set_index("source")
    w = float(summary.loc["weather", "pct_median"])
    m = float(summary.loc["model", "pct_median"])
    i = float(summary.loc["interaction", "pct_median"])
    if not (np.isfinite(w) and np.isfinite(m) and np.isfinite(i)):
        return (
            "warn",
            f"{name}: the total spread is zero on every date, so the record "
            "carries no variance to attribute -- check the record, the draws, "
            "and that more than one model was fitted.",
        )
    top = dominant(res)
    if top is not None:
        word, share = top
        lead = f"{word} carries the larger median share of the total spread -- {share:.0%} of it."
    else:
        lead = (
            "the two sources' median shares stand level; neither can be called the spread's source."
        )
    return (
        "ok",
        f"{name}: median spread {_pct(w)} met resampling, "
        f"{_pct(m)} model form, {_pct(i)} their interaction. " + lead,
    )
