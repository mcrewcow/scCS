import os
import sys
sys.path.insert(0, os.path.abspath("../.."))

project = "scCS"
author = "Emil Kriukov"
release = "0.2.2"
copyright = "2025, Emil Kriukov"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "autoapi.extension",
    "nbsphinx",             
]
nbsphinx_execute = "never"
autoapi_dirs = ["../../scCS"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
autoapi_member_order = "bysource"
suppress_warnings = ["autoapi.python_import_resolution"]
autoapi_add_toctree_entry = True

napoleon_numpy_docstring = True
napoleon_google_docstring = False

html_theme = "furo"
html_title = "scCS"
html_static_path = ["_static"]
