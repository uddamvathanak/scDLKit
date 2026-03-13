# Contributing

## Local setup

```bash
python -m pip install -e .[dev,docs,tutorials]
```

## Development workflow

```bash
ruff check .
mypy src
pytest
python scripts/prepare_tutorial_notebooks.py --execute published
python -m sphinx -b html docs docs/_build/html -W --keep-going
```

Generated tutorial copies under `docs/_tutorials/` and built HTML under `docs/_build/` are build artifacts. Keep the notebook sources in `examples/` and do not commit generated copies.

## Pull requests

- Keep public APIs documented.
- Add or update tests with behavior changes.
- Prefer small, reviewable changes.
- Keep examples reproducible on CPU.
