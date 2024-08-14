# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import bokeh
import pandas
import numpy
import typing
sys.path.insert(0, os.path.abspath('../../code'))



project = 'Bokeh_task'
html_logo = 'image/saitama.png'

copyright = '2024, Yura'
author = 'Yura'
release = '2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import sphinx_rtd_theme

extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autodoc.typehints',
    'sphinx_copybutton',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.githubpages',
]


source_encoding = 'utf-8-sig'


templates_path = ['_templates']
exclude_patterns = []

language = 'ru'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': False,
    'collapse_navigation': True,
    'sticky_navigation': True,
    'includehidden': True,
    'navigation_depth': 4,
    'titles_only': False,
    'style_nav_header_background': '#343131',
}

html_static_path = ['_static']
