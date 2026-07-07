# src/normet/utils/_lazy.py

"""Lazy-import helper for optional dependencies.

Provides :func:`require` to safely import optional dependencies. If the
dependency is missing, raises a clear error message with an installation
hint.
"""

from __future__ import annotations

import importlib
from typing import Any


def require(module: str, hint: str | None = None) -> Any:
    """
    Import a module or object dynamically, with a helpful error if missing.

    Parameters
    ----------
    module : str
        Import path. Examples:
            - "flaml"                     (module)
            - "flaml.automl"              (submodule)
            - "flaml.automl:AutoML"       (specific object inside module)
    hint : str, optional
        Installation hint to include in the error message if import fails.

    Returns
    -------
    Any
        The imported module or object.

    Raises
    ------
    ImportError
        If the requested module or object cannot be imported.
    """
    module = module.strip()

    try:
        if ":" in module:
            # Import a specific object inside a module, e.g. "flaml.automl:AutoML"
            mod_name, attr_name = module.split(":", 1)
            mod = importlib.import_module(mod_name)
            return getattr(mod, attr_name)
        else:
            # Import the whole module or submodule
            return importlib.import_module(module)

    except ImportError as e:
        msg = f"Optional dependency '{module}' is required but not installed."
        if hint:
            msg += f" Install it via: {hint}"
        raise ImportError(msg) from e

    except AttributeError as e:
        raise ImportError(f"Module '{module}' does not provide the requested attribute.") from e
