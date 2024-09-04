# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "noisy_graph_states"
copyright = "2023, Julius Wallnöfer; 2024, Julius Wallnöfer and Maria Flors Mor-Ruiz"
author = "Julius Wallnöfer"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "recommonmark",
    "autodocsumm",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


# napoleon options
napoleon_use_ivar = True
napoleon_use_rtype = False

# autodoc options
autodoc_member_order = "bysource"
autodoc_default_options = {
    "autosummary": True,
}
