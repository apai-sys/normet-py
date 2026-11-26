# src/normet/utils/_rust.py
"""Optional Rust kernels, imported only when a caller opts in.

``normet_rust`` is the PyO3 extension built from ``rust/normet-python``
(``normet-core`` holds the pure kernels shared with the R package). It is
opt-in: ``import normet`` never imports it, and the default
``theil_sen(..., backend="python")`` never touches it, so a missing wheel
costs nothing until a caller asks for the Rust backend.
"""

from __future__ import annotations

from typing import Any

from ._lazy import require

_BUILD_HINT = (
    "build rust/normet-python with maturin (see AGENTS.md), install the wheel, "
    "or use backend='python'"
)


def rust_core() -> Any:
    """Return the ``normet_rust`` extension module or raise a clear ImportError.

    The error names the build step because the extension is not on PyPI yet:
    it ships from ``rust/normet-python`` via ``maturin build``.
    """
    return require("normet_rust", hint=_BUILD_HINT)


def rust_available() -> bool:
    """Whether the optional extension can be imported (for tests and reports)."""
    try:
        rust_core()
    except ImportError:
        return False
    return True
