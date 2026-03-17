"""Sphinx configuration for scDLKit documentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scdlkit import __version__  # noqa: E402

project = "scDLKit"
copyright = "2026, Vathanak Uddam"
author = "Vathanak Uddam"
release = __version__
version = __version__

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "_tutorials/.ipynb_checkpoints",
]

autosummary_generate = True
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

nb_execution_mode = "off"
nb_execution_timeout = 600
nb_merge_streams = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

html_theme = "pydata_sphinx_theme"
html_title = "scDLKit documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "logo": {"text": "scDLKit"},
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc"],
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/uddamvathanak/scDLKit",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/scdlkit/",
            "icon": "fa-brands fa-python",
        },
    ],
}
html_context = {
    "github_user": "uddamvathanak",
    "github_repo": "scDLKit",
    "github_version": "main",
    "doc_path": "docs",
}
