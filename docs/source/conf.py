"""Configuration file for the Sphinx documentation builder."""

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pypsa-usa"
copyright = "2024, Kamran Tehranchi, Trevor Barnes"
author = "Kamran Tehranchi, Trevor Barnes"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.graphviz",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "sphinx.ext.imgconverter",  # raster fallback for SVG in non-HTML builds
]
myst_enable_extensions = ["html_image", "colon_fence", "amsmath", "dollarmath"]
myst_heading_anchors = 3

exclude_patterns = []

intersphinx_mapping = {
    "atlite": ("https://atlite.readthedocs.io/en/latest/", None),
}

bibtex_bibfiles = ["publications.bib"]
bibtex_default_style = "unsrt"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_theme_options = {
    "repository_url": "https://github.com/pypsa/pypsa-usa",
    "use_repository_button": True,
    "show_navbar_depth": 1,
    "show_toc_level": 2,
}

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "PyPSA-USA"

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "PyPSA-USA"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "./_static/pypsa-logo.png"
