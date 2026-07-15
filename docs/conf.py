from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "scCS"
author = "Emil Kriukov"
copyright = "2026, Emil Kriukov"

version = "0.8"
release = "0.8.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
]
autosummary_generate = True
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

source_suffix = {".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**/.ipynb_checkpoints"]

try:
    import furo  # noqa: F401
except ImportError:
    html_theme = "alabaster"
else:
    html_theme = "furo"
html_title = "scCS v.0.8"
html_short_title = "scCS v.0.8"

nbsphinx_execute = "never"
nbsphinx_allow_errors = False

# Project branding
html_static_path = ["_static"]
html_logo = "_static/logo.png"
