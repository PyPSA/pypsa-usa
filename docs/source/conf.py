# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = "pypsa-usa"
copyright = "2023, Kamran Tehranchi, Trevor Barnes"
author = "Kamran Tehranchi, Trevor Barnes"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath("../../workflow/scripts"))

extensions = [
    #'sphinx.ext.autodoc',
    #'sphinx.ext.autosummary',
    "myst_parser",
    # "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.graphviz",
    # "sphinxcontrib.bibtex",
    #'sphinx.ext.pngmath',
    #'sphinxcontrib.tikz',
    #'rinoh.frontend.sphinx',
    "sphinx.ext.imgconverter",  # for SVG conversion
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_theme_options = {
    "repository_url": "https://github.com/pypsa/pypsa-usa",
    "use_repository_button": True,
    "show_navbar_depth": 1,
}

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "PyPSA-USA"

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "PyPSA-USA"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "./_static/pypsa-logo.png"
