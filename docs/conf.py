import diff_stream

project = "diff-stream"
author = "Lee Tucker"
copyright = "2026, Lee Tucker"
release = diff_stream.__version__
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build"]
myst_enable_extensions = ["colon_fence", "dollarmath", "deflist", "attrs_inline"]
autodoc_typehints = "none"
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
html_theme = "furo"
html_title = f"diff-stream {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
