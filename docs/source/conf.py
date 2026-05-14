from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.path.abspath(ROOT))

project = "SDPLab"
author = "Pavlo Pelikh"
copyright = f"{date.today().year}, {author}"
release = "0.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "numpydoc",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_mock_imports = ["cvxpy", "jax", "optax"]
autodoc_class_signature = "mixed"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "exclude-members": (
        "tree_flatten, tree_unflatten, "
        "__dict__, __weakref__, __module__"
    ),
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_validation_checks = set()

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_title = "SDPLab"
html_static_path = ["_static"]
html_extra_path = [".nojekyll"]
html_css_files = ["custom.css"]

html_theme_options = {
    "logo": {"text": "SDPLab"},
    "show_toc_level": 2,
    "navigation_depth": 3,
    "collapse_navigation": True,
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Pavlo3P/SDPLab",
            "icon": "fa-brands fa-github",
        }
    ],
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
}

html_context = {
    "github_user": "Pavlo3P",
    "github_repo": "SDPLab",
    "github_version": "main",
    "doc_path": "docs/source",
}
