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
# import os
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys
sys.path.insert(0,os.path.dirname(os.getcwd()))
from py4DSTEM import __version__
from datetime import datetime

# -- Project information -----------------------------------------------------

project = 'py4dstem'
copyright = f'{datetime.today().year}, Ben Savitzky'
author = 'py4DSTEM Development Team'

# The full version, including alpha/beta/rc tags
# release = '0.14.0'
release = f"{__version__}"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.intersphinx']

# Other useful extensions
# sphinx_copybutton
# sphinx_toggleprompt
# sphinx.ext.mathjax

# Specify a standard user agent, as Sphinx default is blocked on some sites
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.54"


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Set autodoc defaults
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__'
}

# Include todo items/lists
todo_include_todos = True

#autodoc_member_order = 'bysource'


# intersphinx options 

# intersphinx_mapping = {
# 'emdfile': ('https://pypi.org/project/emdfile/0.0.4/', None)
# }

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['../_static']


# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = '../_static/py4DSTEM_logo.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = '../_static/py4DSTEM_logo_vsmall.ico'


