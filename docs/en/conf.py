# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'EvalScope'
copyright = '2022-2024, Alibaba ModelScope'
author = 'ModelScope Authors'
version_file = '../../evalscope/version.py'

def get_version():
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']

# The full version, including alpha/beta/rc tags
version = get_version()
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx_markdown_tables',
    'sphinx_copybutton',
    'sphinx_tabs.tabs',
    'myst_parser',
]

# build the templated autosummary files
autosummary_generate = True
numpydoc_show_class_members = False

# Enable overriding of function signatures in the first line of the docstring.
autodoc_docstring_signature = True

# Disable docstring inheritance
autodoc_inherit_docstrings = False

# Show type hints in the description
autodoc_typehints = 'description'

# Add parameter types if the parameter is documented in the docstring
autodoc_typehints_description_target = 'documented_params'

autodoc_default_options = {
    'member-order': 'bysource',
}

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# The master toctree document.
root_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Auto-generated header anchors
myst_heading_anchors = 3
# Enable "colon_fence" extension of myst.
myst_enable_extensions = ['colon_fence', 'dollarmath']

# Ignore >>> when copying code
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True

napoleon_custom_sections = [
    # Custom sections for data elements.
    ('Meta fields', 'params_style'),
    ('Data fields', 'params_style'),
]