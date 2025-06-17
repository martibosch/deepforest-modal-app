"""Documentation configuration."""

import os
import sys

import pkg_resources

project = "DeepForest modal app"
author = "Martí Bosch"

release = pkg_resources.get_distribution("deepforest_modal_app").version
version = ".".join(release.split(".")[:2])


extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

autodoc_typehints = "description"
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/martibosch/deepforest-modal-app",
}
html_title = "DeepForest modal app"

# add module to path
sys.path.insert(0, os.path.abspath(".."))

# exclude patterns from sphinx-build
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# do NOT execute notebooks
nbsphinx_execute = "never"

# no prompts in rendered notebooks
# https://github.com/microsoft/torchgeo/pull/783
html_static_path = ["_static"]
html_css_files = ["notebooks.css"]
