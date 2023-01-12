# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from recommonmark.parser import CommonMarkParser
from httplib2 import FailedToDecompressContent

sys.path.insert(0, os.path.abspath('../..'))
mathjax_path = (
    "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
)

# -- Project information -----------------------------------------------------

project = 'gpetas'
copyright = '2023, chrism'
author = 'chrism'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
'sphinx.ext.autodoc',  # Core library for html generation from docstrings
'sphinx.ext.autosummary',  # Create neat summary tables
'sphinx.ext.todo',
'sphinx.ext.viewcode',
'sphinx.ext.napoleon',
'myst_parser',
'nbsphinx',
'sphinx.ext.mathjax'
]
#napoleon_google_docstring = False # disables Google style
# 'sphinx_autodoc_typehints' only with sphinx>=5.x. but
# sphinx_book_theme requires currently <5.x.


autodoc_default_options = {
    "members": "var1, var2",
    "member-order": "bysource",
    "undoc-members": True,
    "exclude-members": "__weakref__, to_tuple, from_tuple, replace"
}

autosummary_generate = True  # Turn on sphinx.ext.autosummary

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

#html_theme = 'sphinx_rtd_theme'
html_theme = 'sphinx_book_theme'

html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 2,
    "logo_only": True,
    "repository_url": "https://github.com/chris-molki/gpetas",
    "use_repository_button": True,
    "show_toc_level": 2,
}

html_logo = "_static/logo.svg"
add_module_names = False
#html_title = "v0.0.1"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

source_parsers = {
    ".md": CommonMarkParser,
}
