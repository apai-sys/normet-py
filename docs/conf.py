# docs/conf.py
"""Sphinx configuration for the normet documentation site."""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Make the package importable when building docs from a clean checkout.
sys.path.insert(0, os.path.abspath("../src"))

project = "normet"
author = "Congbo Song and contributors"
copyright = f"{datetime.now().year}, {author}"

try:
    from importlib.metadata import version as _ver

    release = _ver("normet")
except Exception:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# -- General config ----------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # NumPy/Google docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]
# Markdown support is optional; activate only if myst_parser is installed.
try:
    import myst_parser  # noqa: F401

    extensions.append("myst_parser")
    _HAS_MYST = True
except Exception:
    _HAS_MYST = False

autosummary_generate = True
# Each symbol is documented once, at its canonical (defining) module. Module
# pages render the module docstring plus autosummary tables that link to a
# per-object stub; re-exported names (e.g. ``normet.NormetRun``, surfaced from
# ``normet.utils.provenance``) are not listed again on every package that
# re-exports them. Member detail is added by the per-object stub templates
# (see ``_templates/autosummary/``), so module-level ``automodule`` does not
# expand members itself.
autosummary_imported_members = False
autodoc_default_options = {
    "undoc-members": False,
    "show-inheritance": True,
}

# Re-exported names remain visible under several import paths, which makes
# cross-references like ``model`` or ``NormetRun`` ambiguous. These are benign.
suppress_warnings = ["ref.python"]
napoleon_numpy_docstring = True
napoleon_google_docstring = False
# Render docstring "Attributes" sections as inline ``:ivar:`` fields rather than
# standalone ``.. attribute::`` directives, so they don't collide with the same
# (dataclass) fields documented by autodoc ``:members:``.
napoleon_use_ivar = True

myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = (
    {".rst": "restructuredtext", ".md": "markdown"} if _HAS_MYST else {".rst": "restructuredtext"}
)

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
if not _HAS_MYST:
    # Don't try to render .md guides when myst is unavailable.
    exclude_patterns.append("guide/*.md")

# -- HTML --------------------------------------------------------------------
try:
    import sphinx_rtd_theme  # noqa: F401

    html_theme = "sphinx_rtd_theme"
except Exception:
    html_theme = "alabaster"  # built-in fallback
html_static_path = ["_static"]
html_title = f"normet {release}"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"
