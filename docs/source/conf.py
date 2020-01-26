# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

import sys
import os
import sphinx_bootstrap_theme

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
try:
    # Numba is installed
    import numba
    import numba_scipy
except ImportError:
    # Numba is run from its source checkout
    sys.path.insert(0, os.path.abspath('../..'))
    import numba
    import numba_scipy


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = u'numba-scipy'
copyright = u'2019, Anaconda, Inc.'
author = u'Anaconda, Inc.'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
version = '.'.join(numba_scipy.__version__.split('.')[:2])
# The full version, including alpha/beta/rc tags.
release = numba_scipy.__version__

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['../_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# pip install sphinx_bootstrap_theme
html_theme = 'bootstrap'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
  'bootswatch_theme': "paper",
}

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['../_static']


# -- Intersphinx -------------------------------------------------------------
# Configuration for intersphinx: refer to the Python standard library
# and the Numpy documentation.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'numba': ('http://numba.pydata.org/numba-doc/latest', None),
    'llvmlite': ('http://llvmlite.pydata.org/en/latest/', None),
    }


# numpydoc options

# To silence "WARNING: toctree contains reference to nonexisting document"
numpydoc_show_class_members = False


def setup(app):
    app.add_stylesheet("numba-scipy-docs.css")
