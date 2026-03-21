# Contributing

Contributions should keep scDLKit focused:

- AnnData-native
- baseline-first
- reproducible
- easy to adopt for Scanpy users

## Local setup

```bash
python -m pip install -e ".[dev,docs,tutorials]"
```

## Validation

Before opening a pull request, run:

```bash
ruff check .
mypy src
pytest
python scripts/prepare_tutorial_notebooks.py --execute published
python -m sphinx -b html docs docs/_build/html -W --keep-going
```

Keep notebook sources in `examples/`. The `docs/_tutorials/` copies and `docs/_build/` output are generated during validation and should not be committed.

## Feature completeness policy

Public feature work is expected to land with documentation at the same time:

- at least one workflow tutorial must cover the feature
- an API contract page must describe parameters, input expectations, returns, and failure modes
- stable and experimental features are held to the same documentation standard

Use the docs contract checker before opening a pull request:

```bash
python scripts/check_feature_docs_contracts.py --check
```
