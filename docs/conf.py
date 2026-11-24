"""Sphinx configuration of the CLM-ML-JAX documentation.

Build: ``sphinx-build -W -b html docs docs/_build/html`` (dev extras; warnings are errors).
"""

import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS / "_ext"))
sys.path.insert(0, str(DOCS.parent / "src"))

project = "CLM-ML-JAX"
author = "Linnia Hawkins, Aya Lahlou"
copyright = "2026, " + author

extensions = ["myst_parser", "settings_reference"]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
source_suffix = {".md": "markdown"}

# docs/design is the development record, read on GitHub, not part of the site
exclude_patterns = ["_build", "design", "README.md"]

html_theme = "furo"
html_title = "CLM-ML-JAX"
html_theme_options = {
    "source_repository": "https://github.com/linniahawkins/clm-ml-jax/",
    "source_branch": "main",
    "source_directory": "docs/",
}

REPO = "https://github.com/linniahawkins/clm-ml-jax/blob/main"
# {repo}/path in the pages becomes a link to the file on GitHub
myst_url_schemes = {
    "http": None,
    "https": None,
    "repo": {"url": REPO + "/{{path}}", "title": "{{path}}", "classes": []},
}
