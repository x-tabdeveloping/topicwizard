import os
import sys

sys.path.insert(0, os.path.abspath(".."))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "topicwizard"
copyright = "2023, Márton Kardos"
author = "Márton Kardos"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
# html_collapsible_definitions = True
# html_awesome_code_headers = True
html_favicon = "_static/icon.svg"
# html_logo = "_static/icon_w_title_below.png"
html_title = "topicwizard"
html_static_path = ["_static"]
html_theme_options = {
    "light_logo": "icon.svg",
    "dark_logo": "icon.svg",
    "light_css_variables": {
        "color-api-name": "#28a4df",
        "color-api-pre-name": "#ffa671",
    },
    "dark_css_variables": {
        "color-api-name": "#28a4df",
        "color-api-pre-name": "#ffa671",
    },
}

# Napoleon settings
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings:
autodoc_type_aliases = {"ArrayLike": "ArrayLike"}
autodoc_member_order = "bysource"
