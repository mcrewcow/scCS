import os
import sys
sys.path.insert(0, os.path.abspath("../.."))

project = "scCS"
author = "Emil Kriukov"
version = "0.5.0"
release = "0.5.0"
copyright = "2026, Emil Kriukov"
html_logo = "_static/logo.png"
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
html_title = f"scCS v{release}"
html_static_path = ["_static"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#0072B2",
        "color-brand-content": "#0072B2",
    },
}
# Show "vX.Y.Z" under the project name in the sidebar
html_context = {
    "version": release,
}
