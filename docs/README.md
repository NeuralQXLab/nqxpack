# Building the docs

## Dependencies

The docs use a shared theme package, `neuralqxlab-sphinx-theme`, that lives in
a sibling directory:

```
Codes/Python/
├── nqxpack/                  ← this repo
│   └── docs/
└── neuralqxlab-sphinx-theme/ ← shared theme (separate repo)
```

The shared theme provides:
- The [Sphinx Book Theme](https://sphinx-book-theme.readthedocs.io) base
- A federation org bar (NeuralQXLab brand + packages dropdown) injected above
  the per-package header on every page
- Shared Sphinx configuration defaults (`conf_base.py`)

## Local setup

Both repos must be checked out as siblings. Then install the docs dependencies:

```bash
uv sync --group docs
```

This installs `neuralqxlab-sphinx-theme` as an editable path dependency
(see `pyproject.toml` → `[dependency-groups] docs`), so local changes to the
theme are reflected immediately without reinstalling.

## Building

```bash
# One-shot build
uv run --group docs make -C docs html

# Live-reload (watches both docs/ and the package source)
uv run --group docs make -C docs livehtml
```

Output lands in `docs/_build/html/`.

## Structure

```
docs/
├── conf.py              # Sphinx config — imports shared base, sets project name
├── index.md             # Landing page
├── getting_started.md   # Installation + minimal example
├── tutorials/
│   └── index.md         # Tutorial index (placeholder)
├── api/
│   └── index.md         # API reference index (autosummary)
├── Makefile
└── requirements.txt     # Alternative pip-based install (for CI)
```

## Adding a new page

1. Create a `.md` file in the appropriate subdirectory.
2. Add it to the `toctree` in `index.md` (or the relevant section index).

## Updating the shared theme

The theme is managed in its own repo (`neuralqxlab-sphinx-theme`). To add a new
package to the federation navbar, edit `PACKAGES` in
`neuralqxlab_sphinx_theme/__init__.py` and bump the version. All package docs
pick up the change on their next build.
