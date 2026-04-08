"""Sphinx configuration for nqxpack docs.

Shared defaults (theme, extensions, intersphinx, napoleon settings) come from
``neuralqxlab_sphinx_theme.conf_base``. That package lives in the sibling repo
``../neuralqxlab-sphinx-theme`` and is installed as a path dependency via the
``docs`` dependency group in ``pyproject.toml``. See ``docs/README.md`` for
build instructions.
"""

from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
import tomllib

from neuralqxlab_sphinx_theme.conf_base import *  # noqa: F401, F403
from neuralqxlab_sphinx_theme.linkcode import make_linkcode_resolve


def _project_version() -> str:
    try:
        return package_version("nqxpack")
    except PackageNotFoundError:
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text())
        return data["project"]["version"]


project = "nqxpack"
copyright = "2024, NeuralQXLab"
author = "NeuralQXLab"
release = _project_version()
version = ".".join(release.split(".")[:2])

html_context = {
    **html_context,  # noqa: F405
    "github_repo": "nqxpack",
}

html_theme_options = {
    **html_theme_options,  # noqa: F405
    "logo": {
        "image_light": "_static/logo-nav.webp",
        "image_dark": "_static/logo-nav.webp",
        "alt": "nqxpack",
    },
}

html_title = "nqxpack"
html_favicon = "_static/favicon.ico"

html_static_path = ["_static"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

linkcode_resolve = make_linkcode_resolve(
    github_repo="nqxpack",
    repo_root=Path(__file__).parent.parent,
)

# nqxpack uses Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
